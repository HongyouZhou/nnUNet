"""Matched baseline and voxel-continuity separator trainers."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Union

import numpy as np
import torch
import torch.distributed as dist
from batchgenerators.dataloading.nondet_multi_threaded_augmenter import NonDetMultiThreadedAugmenter
from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter
from batchgeneratorsv2.transforms.spatial.spatial import SpatialTransform
from torch.amp import autocast
from torch._dynamo import OptimizedModule

from nnunetv2.paths import nnUNet_raw
from nnunetv2.training.dataloading.data_loader import nnUNetDataLoader
from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.collate_outputs import collate_outputs
from nnunetv2.utilities.default_n_proc_DA import get_allowed_n_proc_DA
from nnunetv2.utilities.helpers import dummy_context

from .contract import (
    CONTINUITY_PLANS_KEY,
    CONTINUITY_WEIGHT,
    DENSITY_WEIGHT,
    FORMAL_PREPROCESSOR,
    IGNORE_LABEL,
    SEMANTIC_CHANNEL,
    DensityCalibration,
    sha256_file,
    validate_continuity_plans_contract,
)
from .losses import CorticalSeparatorRegularizedLoss
from .sampling import SeparatorContinuityDataLoader


_COMPONENT_KEYS = (
    "base_loss",
    "continuity_loss",
    "density_loss",
    "valid_same_pairs",
    "valid_different_pairs",
    "valid_density_pairs",
)


class _nnUNetTrainerCorticalSeparatorContinuityBase(nnUNetTrainer):
    continuity_weight = 0.0
    density_weight = 0.0

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ) -> None:
        try:
            data_identifier = plans["configurations"][configuration]["data_identifier"]
        except KeyError as error:
            raise RuntimeError("Incomplete separator continuity plans") from error
        self.continuity_contract = validate_continuity_plans_contract(
            plans, str(data_identifier)
        )
        super().__init__(plans, configuration, fold, dataset_json, device)
        if self.configuration_manager.preprocessor_name != FORMAL_PREPROCESSOR:
            raise RuntimeError(
                f"Separator continuity plans must use {FORMAL_PREPROCESSOR}"
            )
        if len(self.configuration_manager.patch_size) != 3:
            raise RuntimeError("Separator continuity supports 3-D full resolution only")
        if self.label_manager.has_regions or self.label_manager.ignore_label != IGNORE_LABEL:
            raise RuntimeError(
                "Separator continuity requires softmax labels 0/1/2 and ignore label 3"
            )
        if self.label_manager.num_segmentation_heads != 3:
            raise RuntimeError("Separator continuity requires exactly three output classes")
        self.num_epochs = 1000
        self.save_every = 50
        self.density_calibration_path: Path | None = None
        self.density_calibration: DensityCalibration | None = None
        if self.density_weight:
            (
                self.density_calibration_path,
                self.density_calibration,
            ) = self._load_and_validate_calibration()
        for key in _COMPONENT_KEYS:
            self.logger.local_logger.my_fantastic_logging.setdefault(key, [])

    def _load_and_validate_calibration(self) -> tuple[Path, DensityCalibration]:
        configured = os.environ.get("CORTICAL_CONTINUITY_CALIBRATION_DIR")
        root = (
            Path(configured).expanduser().resolve()
            if configured
            else Path(self.preprocessed_dataset_folder_base)
            / "separator_continuity_calibration"
        )
        path = root / f"fold_{self.fold}.json"
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing fold-specific density calibration {path}; run preparation first"
            )
        calibration = DensityCalibration.load(path)
        if calibration.fold != int(self.fold):
            raise RuntimeError("Density calibration fold does not match trainer fold")
        base = Path(self.preprocessed_dataset_folder_base)
        splits_path = base / "splits_final.json"
        plans_path = base / f"{self.plans_manager.plans_name}.json"
        if sha256_file(splits_path) != calibration.splits_sha256:
            raise RuntimeError("Density calibration splits hash mismatch")
        if sha256_file(plans_path) != calibration.plans_sha256:
            raise RuntimeError("Density calibration plans hash mismatch")
        splits = json.loads(splits_path.read_text(encoding="utf-8"))
        expected_train = tuple(sorted(str(value) for value in splits[int(self.fold)]["train"]))
        if expected_train != tuple(sorted(calibration.train_cases)):
            raise RuntimeError("Density calibration train cases do not match the fold")
        if nnUNet_raw.is_set():
            manifest = (
                Path(nnUNet_raw.require())
                / self.plans_manager.dataset_name
                / "continuity_manifest.json"
            )
            if sha256_file(manifest) != calibration.source_manifest_sha256:
                raise RuntimeError("Density calibration source manifest hash mismatch")
        return path, calibration

    def _build_loss(self) -> CorticalSeparatorRegularizedLoss:
        weights = None
        if self.enable_deep_supervision:
            scales = self._get_deep_supervision_scales()
            values = np.asarray(
                [1 / (2**index) for index in range(len(scales))], dtype=np.float64
            )
            values[-1] = 1e-6 if self.is_ddp and not self._do_i_compile() else 0.0
            values /= values.sum()
            weights = values.tolist()
        return CorticalSeparatorRegularizedLoss(
            batch_dice=self.configuration_manager.batch_dice,
            ddp=self.is_ddp,
            deep_supervision_weights=weights,
            continuity_weight=self.continuity_weight,
            density_weight=self.density_weight,
            density_calibration=self.density_calibration,
        )

    def get_dataloaders(self):
        if self.dataset_class is None:
            self.dataset_class = infer_dataset_class(self.preprocessed_dataset_folder)
        patch_size = self.configuration_manager.patch_size
        deep_supervision_scales = self._get_deep_supervision_scales()
        rotation, dummy_2d, initial_patch_size, mirror_axes = (
            self.configure_rotation_dummyDA_mirroring_and_inital_patch_size()
        )
        training_transforms = self.get_training_transforms(
            patch_size,
            rotation,
            deep_supervision_scales,
            mirror_axes,
            dummy_2d,
            use_mask_for_norm=self.configuration_manager.use_mask_for_norm,
            is_cascaded=False,
            foreground_labels=self.label_manager.foreground_labels,
            regions=None,
            ignore_label=self.label_manager.ignore_label,
            segmentation_interpolation_mode="nearest",
        )
        _assert_joint_nearest_spatial_transform(training_transforms)
        validation_transforms = self.get_validation_transforms(
            deep_supervision_scales,
            is_cascaded=False,
            foreground_labels=self.label_manager.foreground_labels,
            regions=None,
            ignore_label=self.label_manager.ignore_label,
        )
        dataset_tr, dataset_val = self.get_tr_and_val_datasets()
        dl_tr = SeparatorContinuityDataLoader(
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
        allowed = get_allowed_n_proc_DA()
        if allowed == 0:
            train_generator = SingleThreadedAugmenter(dl_tr, None)
            validation_generator = SingleThreadedAugmenter(dl_val, None)
        else:
            train_generator = NonDetMultiThreadedAugmenter(
                data_loader=dl_tr,
                transform=None,
                num_processes=allowed,
                num_cached=max(6, allowed // 2),
                seeds=None,
                pin_memory=self.device.type == "cuda",
                wait_time=0.002,
            )
            validation_generator = NonDetMultiThreadedAugmenter(
                data_loader=dl_val,
                transform=None,
                num_processes=max(1, allowed // 2),
                num_cached=max(3, allowed // 4),
                seeds=None,
                pin_memory=self.device.type == "cuda",
                wait_time=0.002,
            )
        _ = next(train_generator)
        _ = next(validation_generator)
        return train_generator, validation_generator

    def validation_step(self, batch: dict) -> dict:
        data = batch["data"].to(self.device, non_blocking=True)
        target = batch["target"]
        if isinstance(target, list):
            target = [value.to(self.device, non_blocking=True) for value in target]
        else:
            target = target.to(self.device, non_blocking=True)
        context = autocast(self.device.type, enabled=True) if self.device.type == "cuda" else dummy_context()
        with context:
            output = self.network(data)
            loss = self.loss(output, target)
        result = {
            "loss": loss.detach().cpu().numpy(),
            **{
                key: self.loss.last_components[key].detach().cpu().numpy()
                for key in _COMPONENT_KEYS
            },
        }
        if self.enable_deep_supervision:
            output = output[0]
            target = target[0]
        semantic = target[:, SEMANTIC_CHANNEL : SEMANTIC_CHANNEL + 1].clone()
        axes = [0] + list(range(2, output.ndim))
        output_seg = output.argmax(1)[:, None]
        prediction_onehot = torch.zeros(
            output.shape, device=output.device, dtype=torch.float16
        )
        prediction_onehot.scatter_(1, output_seg, 1)
        mask = (semantic != self.label_manager.ignore_label).float()
        semantic[semantic == self.label_manager.ignore_label] = 0
        tp, fp, fn, _ = get_tp_fp_fn_tn(
            prediction_onehot, semantic, axes=axes, mask=mask
        )
        result.update(
            {
                "tp_hard": tp.detach().cpu().numpy()[1:],
                "fp_hard": fp.detach().cpu().numpy()[1:],
                "fn_hard": fn.detach().cpu().numpy()[1:],
            }
        )
        return result

    def on_validation_epoch_end(self, val_outputs: list[dict]):
        super().on_validation_epoch_end(val_outputs)
        collated = collate_outputs(val_outputs)
        for key in _COMPONENT_KEYS:
            if key.startswith("valid_"):
                value = float(np.sum(collated[key]))
            else:
                value = float(np.mean(collated[key]))
            if self.is_ddp:
                gathered = [None for _ in range(dist.get_world_size())]
                dist.all_gather_object(gathered, value)
                value = float(sum(gathered)) if key.startswith("valid_") else float(np.mean(gathered))
            self.logger.log(key, value, self.current_epoch)
        self.print_to_log_file(
            "separator_regularizers",
            {key: self.logger.get_value(key, step=-1) for key in _COMPONENT_KEYS},
        )

    def on_train_start(self):
        super().on_train_start()
        if self.local_rank == 0:
            contract_path = Path(self.output_folder_base) / "continuity_contract.json"
            contract_path.write_text(
                json.dumps(self.continuity_contract, indent=2) + "\n", encoding="utf-8"
            )
            if self.density_calibration_path is not None:
                shutil.copy2(
                    self.density_calibration_path,
                    Path(self.output_folder_base) / f"density_calibration_fold_{self.fold}.json",
                )

    def save_checkpoint(self, filename: str) -> None:
        if self.local_rank != 0 or self.disable_checkpointing:
            if self.local_rank == 0:
                self.print_to_log_file("No checkpoint written, checkpointing is disabled")
            return
        module = self.network.module if self.is_ddp else self.network
        if isinstance(module, OptimizedModule):
            module = module._orig_mod
        checkpoint = {
            "network_weights": module.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "grad_scaler_state": self.grad_scaler.state_dict() if self.grad_scaler is not None else None,
            "logging": self.logger.get_checkpoint(),
            "_best_ema": self._best_ema,
            "current_epoch": self.current_epoch + 1,
            "init_args": self.my_init_kwargs,
            "trainer_name": self.__class__.__name__,
            "inference_allowed_mirroring_axes": self.inference_allowed_mirroring_axes,
            CONTINUITY_PLANS_KEY: self.continuity_contract,
        }
        torch.save(checkpoint, filename)

    def load_checkpoint(self, filename_or_checkpoint: Union[dict, str]) -> None:
        if isinstance(filename_or_checkpoint, str):
            payload = torch.load(
                filename_or_checkpoint, map_location=self.device, weights_only=False
            )
        else:
            payload = filename_or_checkpoint
        if payload.get(CONTINUITY_PLANS_KEY) != self.continuity_contract:
            raise RuntimeError("Checkpoint continuity contract differs from current plans")
        if not self.was_initialized:
            self.initialize()
        new_state_dict = {}
        for key, value in payload["network_weights"].items():
            mapped = key
            if mapped not in self.network.state_dict() and mapped.startswith("module."):
                mapped = mapped[7:]
            new_state_dict[mapped] = value
        self.my_init_kwargs = payload["init_args"]
        self.current_epoch = payload["current_epoch"]
        self.logger.load_checkpoint(payload["logging"])
        for key in _COMPONENT_KEYS:
            self.logger.local_logger.my_fantastic_logging.setdefault(key, [])
        self._best_ema = payload["_best_ema"]
        self.inference_allowed_mirroring_axes = payload.get(
            "inference_allowed_mirroring_axes", self.inference_allowed_mirroring_axes
        )
        module = self.network.module if self.is_ddp else self.network
        if isinstance(module, OptimizedModule):
            module = module._orig_mod
        module.load_state_dict(new_state_dict)
        self.optimizer.load_state_dict(payload["optimizer_state"])
        if self.grad_scaler is not None and payload["grad_scaler_state"] is not None:
            self.grad_scaler.load_state_dict(payload["grad_scaler_state"])


class nnUNetTrainerCorticalSeparatorMatchedBase(
    _nnUNetTrainerCorticalSeparatorContinuityBase
):
    """Matched Dice+CE baseline with the shared 40/40/20 sampler."""


class nnUNetTrainerCorticalSeparatorContinuity(
    _nnUNetTrainerCorticalSeparatorContinuityBase
):
    """Dice+CE plus voxel-space material continuity."""

    continuity_weight = CONTINUITY_WEIGHT


class nnUNetTrainerCorticalSeparatorContinuityDensityPrior(
    _nnUNetTrainerCorticalSeparatorContinuityBase
):
    """Dice+CE plus continuity and conditional endpoint density."""

    continuity_weight = CONTINUITY_WEIGHT
    density_weight = DENSITY_WEIGHT


def _assert_joint_nearest_spatial_transform(training_transforms) -> None:
    """Fail closed if the six discrete target channels are not transformed jointly."""

    spatial_transforms = [
        transform
        for transform in getattr(training_transforms, "transforms", ())
        if isinstance(transform, SpatialTransform)
    ]
    if len(spatial_transforms) != 1:
        raise RuntimeError(
            "Separator continuity training requires exactly one SpatialTransform"
        )
    if spatial_transforms[0].mode_seg != "nearest":
        raise RuntimeError(
            "Separator continuity target augmentation must use joint nearest-neighbour interpolation"
        )


class _CorticalSeparatorSmokeMixin:
    """One complete nnU-Net epoch in a separate output folder for throughput gating."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.num_epochs = 1
        self.save_every = 1


class nnUNetTrainerCorticalSeparatorMatchedBaseSmoke(
    _CorticalSeparatorSmokeMixin, nnUNetTrainerCorticalSeparatorMatchedBase
):
    """Full-epoch performance smoke for the matched baseline."""


class nnUNetTrainerCorticalSeparatorContinuitySmoke(
    _CorticalSeparatorSmokeMixin, nnUNetTrainerCorticalSeparatorContinuity
):
    """Full-epoch performance smoke for continuity."""


class nnUNetTrainerCorticalSeparatorContinuityDensityPriorSmoke(
    _CorticalSeparatorSmokeMixin,
    nnUNetTrainerCorticalSeparatorContinuityDensityPrior,
):
    """Full-epoch performance smoke for conditional density."""
