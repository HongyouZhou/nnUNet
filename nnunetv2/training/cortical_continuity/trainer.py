from __future__ import annotations

from collections.abc import Mapping
from os.path import isfile, join
from typing import Any

from .losses import CorticalContinuityLoss
from .network import build_cortical_continuity_network
from .packing import (
    PackContinuityTargetsTransform,
    PackedTargetLayout,
    unpack_packed_targets,
)
from .schema import CorticalContinuityHeadSchema, schema_from_plans

FORMAL_NUM_EPOCHS = 500
FORMAL_ITERATIONS_PER_EPOCH = 250
FORMAL_INITIAL_LR = 1e-3

try:
    import numpy as np
    import torch
    from batchgenerators.dataloading.nondet_multi_threaded_augmenter import (
        NonDetMultiThreadedAugmenter,
    )
    from batchgenerators.dataloading.single_threaded_augmenter import (
        SingleThreadedAugmenter,
    )
    from torch.amp import autocast
    from torch._dynamo import OptimizedModule

    from nnunetv2.training.dataloading.continuity_data_loader import ContinuityDataLoader
    from nnunetv2.training.dataloading.data_loader import nnUNetDataLoader
    from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class
    from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
    from nnunetv2.utilities.default_n_proc_DA import get_allowed_n_proc_DA
    from nnunetv2.utilities.helpers import dummy_context
except ModuleNotFoundError as exc:
    torch = None
    nnUNetTrainer = object
    _TRAINER_IMPORT_ERROR = exc
else:
    _TRAINER_IMPORT_ERROR = None


if torch is not None:

    class nnUNetTrainerCorticalContinuity(nnUNetTrainer):
        """Full-resolution flat ``C+A`` trainer skeleton.

        The plans must contain ``cortical_continuity_head_schema``. The class is
        isolated from the generic trainer tree for the MVP; import it directly
        or expose it through an external-trainer path when wiring a training
        command.
        """

        def __init__(
            self,
            plans: dict,
            configuration: str,
            fold: int,
            dataset_json: dict,
            device: torch.device = torch.device("cuda"),
        ) -> None:
            self.continuity_schema = schema_from_plans(plans)
            super().__init__(plans, configuration, fold, dataset_json, device)
            self.enable_deep_supervision = False
            self.inference_allowed_mirroring_axes = None
            self.num_epochs = FORMAL_NUM_EPOCHS
            self.num_iterations_per_epoch = FORMAL_ITERATIONS_PER_EPOCH
            self.initial_lr = FORMAL_INITIAL_LR
            if len(self.configuration_manager.patch_size) != 3:
                raise RuntimeError(
                    "Cortical continuity v1 is frozen to 3-D full-resolution training"
                )
            if self.configuration_manager.preprocessor_name not in {
                "ChariteCorticalPreprocessor",
                "CorticalContinuityPreprocessor",
            }:
                raise RuntimeError(
                    "Plans must use ChariteCorticalPreprocessor for the frozen five-sidecar contract"
                )
            if not np.allclose(
                np.asarray(self.continuity_schema.spacing_mm_zyx),
                np.asarray(self.configuration_manager.spacing),
                rtol=0.0,
                atol=1e-6,
            ):
                raise RuntimeError(
                    "Cortical head schema spacing does not match configuration target spacing"
                )

        @staticmethod
        def build_network_architecture(
            plans_manager: Any,
            configuration_manager: Any,
            num_input_channels: int,
            num_output_channels: int,
            enable_deep_supervision: bool = True,
        ) -> Any:
            del num_output_channels
            schema = schema_from_plans(plans_manager.plans)
            return build_cortical_continuity_network(
                configuration_manager,
                num_input_channels,
                schema,
                enable_deep_supervision=False,
            )

        def _build_loss(self) -> CorticalContinuityLoss:
            return CorticalContinuityLoss(self.continuity_schema)

        def get_training_transforms(
            self,
            patch_size,
            rotation_for_DA,
            deep_supervision_scales,
            mirror_axes,
            do_dummy_2d_data_aug,
            use_mask_for_norm=None,
            is_cascaded=False,
            foreground_labels=None,
            regions=None,
            ignore_label=None,
        ):
            del deep_supervision_scales, regions, ignore_label
            if is_cascaded:
                raise RuntimeError("Cortical continuity v1 does not support cascaded training")
            # Native through-plane resolution is encoded in fixed z/y/x axes.
            # Arbitrary rotations would invalidate that physical supervision
            # contract, so v1 keeps scaling/mirroring but disables rotation.
            base = nnUNetTrainer.get_training_transforms(
                patch_size,
                (0.0, 0.0),
                None,
                mirror_axes,
                do_dummy_2d_data_aug,
                use_mask_for_norm=use_mask_for_norm,
                is_cascaded=False,
                foreground_labels=foreground_labels,
                regions=None,
                ignore_label=None,
            )
            return PackContinuityTargetsTransform(base, self.continuity_schema)

        def get_validation_transforms(
            self,
            deep_supervision_scales,
            is_cascaded=False,
            foreground_labels=None,
            regions=None,
            ignore_label=None,
        ):
            del deep_supervision_scales, regions, ignore_label
            if is_cascaded:
                raise RuntimeError("Cortical continuity v1 does not support cascaded validation")
            base = nnUNetTrainer.get_validation_transforms(
                None,
                is_cascaded=False,
                foreground_labels=foreground_labels,
                regions=None,
                ignore_label=None,
            )
            return PackContinuityTargetsTransform(base, self.continuity_schema)

        def get_dataloaders(self):
            if self.dataset_class is None:
                self.dataset_class = infer_dataset_class(self.preprocessed_dataset_folder)
            patch_size = self.configuration_manager.patch_size
            (
                rotation_for_DA,
                do_dummy_2d_data_aug,
                initial_patch_size,
                mirror_axes,
            ) = self.configure_rotation_dummyDA_mirroring_and_inital_patch_size()
            training_transforms = self.get_training_transforms(
                patch_size,
                rotation_for_DA,
                None,
                mirror_axes,
                do_dummy_2d_data_aug,
                use_mask_for_norm=self.configuration_manager.use_mask_for_norm,
                is_cascaded=False,
                foreground_labels=self.label_manager.foreground_labels,
                regions=None,
                ignore_label=None,
            )
            validation_transforms = self.get_validation_transforms(
                None,
                is_cascaded=False,
                foreground_labels=self.label_manager.foreground_labels,
                regions=None,
                ignore_label=None,
            )
            dataset_tr, dataset_val = self.get_tr_and_val_datasets()
            dl_tr = ContinuityDataLoader(
                dataset_tr,
                self.batch_size,
                initial_patch_size,
                patch_size,
                self.label_manager,
                oversample_foreground_percent=0.0,
                sampling_probabilities=None,
                pad_sides=None,
                transforms=training_transforms,
                probabilistic_oversampling=False,
            )
            # Validation deliberately retains the ordinary nnU-Net
            # random/annotated-centre loader instead of the 40/30/20/10 policy.
            dl_val = nnUNetDataLoader(
                dataset_val,
                self.batch_size,
                patch_size,
                patch_size,
                self.label_manager,
                oversample_foreground_percent=0.0,
                sampling_probabilities=None,
                pad_sides=None,
                transforms=validation_transforms,
                probabilistic_oversampling=False,
            )
            allowed_num_processes = get_allowed_n_proc_DA()
            if allowed_num_processes == 0:
                train_generator = SingleThreadedAugmenter(dl_tr, None)
                validation_generator = SingleThreadedAugmenter(dl_val, None)
            else:
                train_generator = NonDetMultiThreadedAugmenter(
                    data_loader=dl_tr,
                    transform=None,
                    num_processes=allowed_num_processes,
                    num_cached=max(6, allowed_num_processes // 2),
                    seeds=None,
                    pin_memory=self.device.type == "cuda",
                    wait_time=0.002,
                )
                validation_generator = NonDetMultiThreadedAugmenter(
                    data_loader=dl_val,
                    transform=None,
                    num_processes=max(1, allowed_num_processes // 2),
                    num_cached=max(3, allowed_num_processes // 4),
                    seeds=None,
                    pin_memory=self.device.type == "cuda",
                    wait_time=0.002,
                )
            _ = next(train_generator)
            _ = next(validation_generator)
            return train_generator, validation_generator

        def configure_rotation_dummyDA_mirroring_and_inital_patch_size(self):
            result = super().configure_rotation_dummyDA_mirroring_and_inital_patch_size()
            # Training mirroring remains useful because A is regenerated from
            # the augmented ID map. Inference mirror TTA is directionally wrong
            # unless edge channels and source coordinates are remapped.
            self.inference_allowed_mirroring_axes = None
            return result

        def train_step(self, batch: dict) -> dict:
            data = batch["data"].to(self.device, non_blocking=True)
            target = _move_to_device(batch["target"], self.device)
            _validate_packed_target(target, self.continuity_schema)
            self.optimizer.zero_grad(set_to_none=True)
            context = autocast(self.device.type, enabled=True) if self.device.type == "cuda" else dummy_context()
            with context:
                output = self.network(data)
                loss = self.loss(output, target)

            if self.grad_scaler is not None:
                self.grad_scaler.scale(loss).backward()
                self.grad_scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
                self.grad_scaler.step(self.optimizer)
                self.grad_scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
                self.optimizer.step()
            return {"loss": loss.detach().cpu().numpy()}

        def validation_step(self, batch: dict) -> dict:
            data = batch["data"].to(self.device, non_blocking=True)
            target = _move_to_device(batch["target"], self.device)
            _validate_packed_target(target, self.continuity_schema)
            context = autocast(self.device.type, enabled=True) if self.device.type == "cuda" else dummy_context()
            with context:
                output = self.network(data)
                loss = self.loss(output, target)

            cortex_logits = self.continuity_schema.split(output, channel_axis=1)["cortex"]
            unpacked = unpack_packed_targets(target, self.continuity_schema, channel_axis=1)
            cortex_target = unpacked["cortex_target"].bool()
            cortex_valid = unpacked["cortex_valid"].bool()
            cortex_prediction = torch.sigmoid(cortex_logits) >= 0.5
            tp = (cortex_prediction & cortex_target & cortex_valid).sum(dtype=torch.float32)
            fp = (cortex_prediction & ~cortex_target & cortex_valid).sum(dtype=torch.float32)
            fn = (~cortex_prediction & cortex_target & cortex_valid).sum(dtype=torch.float32)
            return {
                "loss": loss.detach().cpu().numpy(),
                "tp_hard": np.asarray([tp.detach().cpu().item()]),
                "fp_hard": np.asarray([fp.detach().cpu().item()]),
                "fn_hard": np.asarray([fn.detach().cpu().item()]),
            }

        def on_train_start(self):
            super().on_train_start()
            if self.local_rank == 0:
                self.continuity_schema.save(
                    join(self.output_folder_base, "cortical_continuity_head_schema.json")
                )

        def save_checkpoint(self, filename: str) -> None:
            if self.local_rank != 0:
                return
            if self.disable_checkpointing:
                self.print_to_log_file("No checkpoint written, checkpointing is disabled")
                return
            module = self.network.module if self.is_ddp else self.network
            if isinstance(module, OptimizedModule):
                module = module._orig_mod
            checkpoint = {
                "network_weights": module.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "grad_scaler_state": (
                    self.grad_scaler.state_dict()
                    if self.grad_scaler is not None
                    else None
                ),
                "logging": self.logger.get_checkpoint(),
                "_best_ema": self._best_ema,
                "current_epoch": self.current_epoch + 1,
                "init_args": self.my_init_kwargs,
                "trainer_name": self.__class__.__name__,
                "inference_allowed_mirroring_axes": None,
                "cortical_continuity_head_schema": self.continuity_schema.to_dict(),
            }
            torch.save(checkpoint, filename)

        def perform_actual_validation(self, save_probabilities: bool = False):
            del save_probabilities
            raise RuntimeError(
                "The generic nnU-Net validator/exporter cannot interpret directional affinity heads. "
                "Use CorticalContinuityPredictor and the cortical continuity evaluation pipeline."
            )


    def _move_to_device(value: Any, device: torch.device) -> Any:
        if isinstance(value, torch.Tensor):
            return value.to(device, non_blocking=True)
        if isinstance(value, Mapping):
            return {key: _move_to_device(item, device) for key, item in value.items()}
        if hasattr(value, "__dataclass_fields__"):
            return {
                key: _move_to_device(getattr(value, key), device)
                for key in value.__dataclass_fields__
            }
        if isinstance(value, np.ndarray):
            return torch.as_tensor(value, device=device)
        raise TypeError(f"Unsupported target container {type(value).__name__}")


    def _target(targets: Mapping[str, torch.Tensor], name: str) -> torch.Tensor:
        return targets[name]


    def _validate_packed_target(
        target: Any,
        schema: CorticalContinuityHeadSchema,
    ) -> None:
        if not isinstance(target, torch.Tensor):
            raise RuntimeError(
                "Cortical continuity dataloader must return one packed target tensor; "
                f"got {type(target).__name__}. Re-run preprocessing with "
                "ChariteCorticalPreprocessor and use ContinuityDataLoader."
            )
        expected = PackedTargetLayout.from_schema(schema).total_channels
        if target.ndim != 5 or int(target.shape[1]) != expected:
            raise RuntimeError(
                "Packed target contract is [B,C_t,C_valid,A(E),A_valid(E),Z,Y,X] "
                f"with {expected} channels on axis 1; got {tuple(target.shape)}"
            )

else:

    class nnUNetTrainerCorticalContinuity:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError(
                "nnUNetTrainerCorticalContinuity requires a complete PyTorch nnU-Net environment"
            ) from _TRAINER_IMPORT_ERROR
