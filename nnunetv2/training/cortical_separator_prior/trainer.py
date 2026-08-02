"""Paired control and density-prior trainers for Dataset778 separators."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch
from batchgenerators.dataloading.nondet_multi_threaded_augmenter import (
    NonDetMultiThreadedAugmenter,
)
from batchgenerators.dataloading.single_threaded_augmenter import (
    SingleThreadedAugmenter,
)
from torch.amp import autocast

from nnunetv2.paths import nnUNet_raw
from nnunetv2.training.cortical_separator_prior.contract import (
    PRIOR_CONTRACT_VERSION,
    PRIOR_PLANS_KEY,
    SEMANTIC_CHANNEL,
    DensityCalibration,
    sha256_file,
)
from nnunetv2.training.cortical_separator_prior.losses import PairedSeparatorLoss
from nnunetv2.training.cortical_separator_prior.sampling import (
    SeparatorPriorDataLoader,
)
from nnunetv2.training.dataloading.data_loader import nnUNetDataLoader
from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn
from nnunetv2.utilities.default_n_proc_DA import get_allowed_n_proc_DA
from nnunetv2.utilities.helpers import dummy_context


class _nnUNetTrainerCorticalSeparatorPaired(nnUNetTrainer):
    use_density_prior = False

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ) -> None:
        contract = plans.get(PRIOR_PLANS_KEY)
        if not isinstance(contract, dict) or int(contract.get("schema_version", -1)) != PRIOR_CONTRACT_VERSION:
            raise RuntimeError(
                f"Paired separator trainers require {PRIOR_PLANS_KEY} schema "
                f"{PRIOR_CONTRACT_VERSION}"
            )
        super().__init__(plans, configuration, fold, dataset_json, device)
        if self.configuration_manager.preprocessor_name != "ChariteDensityPriorSeparatorPreprocessor":
            raise RuntimeError(
                "Paired separator plans must use ChariteDensityPriorSeparatorPreprocessor"
            )
        if len(self.configuration_manager.patch_size) != 3:
            raise RuntimeError("Paired separator training supports 3-D full-resolution only")
        self.density_calibration_path, self.density_calibration = (
            self._load_and_validate_calibration()
        )

    def _load_and_validate_calibration(self) -> tuple[Path, DensityCalibration]:
        configured = os.environ.get("CORTICAL_DENSITY_CALIBRATION_DIR")
        calibration_root = (
            Path(configured).expanduser().resolve()
            if configured
            else Path(self.preprocessed_dataset_folder_base)
            / "density_prior_calibration"
        )
        path = calibration_root / f"fold_{self.fold}.json"
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing fold-specific density calibration {path}. Run the density-prior "
                "calibrate and audit CPU gate before training."
            )
        calibration = DensityCalibration.load(path)
        if calibration.fold != int(self.fold):
            raise RuntimeError(
                f"Calibration fold {calibration.fold} does not match trainer fold {self.fold}"
            )
        base = Path(self.preprocessed_dataset_folder_base)
        splits_path = base / "splits_final.json"
        plans_path = base / f"{self.plans_manager.plans_name}.json"
        for required in (splits_path, plans_path):
            if not required.is_file():
                raise FileNotFoundError(required)
        if sha256_file(splits_path) != calibration.splits_sha256:
            raise RuntimeError("Density calibration splits_final.json hash mismatch")
        if sha256_file(plans_path) != calibration.plans_sha256:
            raise RuntimeError("Density calibration plans JSON hash mismatch")
        splits = json.loads(splits_path.read_text(encoding="utf-8"))
        expected_train = tuple(sorted(str(value) for value in splits[int(self.fold)]["train"]))
        if expected_train != tuple(sorted(calibration.train_cases)):
            raise RuntimeError("Density calibration train cases do not match trainer fold")
        if nnUNet_raw.is_set():
            manifest_path = (
                Path(nnUNet_raw.require())
                / self.plans_manager.dataset_name
                / "continuity_manifest.json"
            )
            if not manifest_path.is_file():
                raise FileNotFoundError(manifest_path)
            if sha256_file(manifest_path) != calibration.source_manifest_sha256:
                raise RuntimeError("Density calibration source manifest hash mismatch")
        return path, calibration

    def _build_loss(self) -> PairedSeparatorLoss:
        if self.label_manager.has_regions or self.label_manager.ignore_label != 3:
            raise RuntimeError(
                "Density-prior separator requires softmax labels 0/1/2 with ignore label 3"
            )
        weights = None
        if self.enable_deep_supervision:
            scales = self._get_deep_supervision_scales()
            values = np.asarray([1 / (2**index) for index in range(len(scales))], dtype=np.float64)
            values[-1] = 1e-6 if self.is_ddp and not self._do_i_compile() else 0.0
            values /= values.sum()
            weights = values.tolist()
        return PairedSeparatorLoss(
            self.density_calibration,
            use_density_prior=self.use_density_prior,
            batch_dice=self.configuration_manager.batch_dice,
            ddp=self.is_ddp,
            deep_supervision_weights=weights,
            density_auxiliary_weight=0.5,
        )

    def get_dataloaders(self):
        if self.dataset_class is None:
            self.dataset_class = infer_dataset_class(self.preprocessed_dataset_folder)
        patch_size = self.configuration_manager.patch_size
        deep_supervision_scales = self._get_deep_supervision_scales()
        (
            rotation_for_DA,
            do_dummy_2d_data_aug,
            initial_patch_size,
            mirror_axes,
        ) = self.configure_rotation_dummyDA_mirroring_and_inital_patch_size()
        training_transforms = self.get_training_transforms(
            patch_size,
            rotation_for_DA,
            deep_supervision_scales,
            mirror_axes,
            do_dummy_2d_data_aug,
            use_mask_for_norm=self.configuration_manager.use_mask_for_norm,
            is_cascaded=False,
            foreground_labels=self.label_manager.foreground_labels,
            regions=None,
            ignore_label=self.label_manager.ignore_label,
        )
        validation_transforms = self.get_validation_transforms(
            deep_supervision_scales,
            is_cascaded=False,
            foreground_labels=self.label_manager.foreground_labels,
            regions=None,
            ignore_label=self.label_manager.ignore_label,
        )
        dataset_tr, dataset_val = self.get_tr_and_val_datasets()
        dl_tr = SeparatorPriorDataLoader(
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
            density_median_hu=self.density_calibration.q50_hu,
            use_density_prior=self.use_density_prior,
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
        context = (
            autocast(self.device.type, enabled=True)
            if self.device.type == "cuda"
            else dummy_context()
        )
        with context:
            output = self.network(data)
            loss = self.loss(output, target)
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
        return {
            "loss": loss.detach().cpu().numpy(),
            "tp_hard": tp.detach().cpu().numpy()[1:],
            "fp_hard": fp.detach().cpu().numpy()[1:],
            "fn_hard": fn.detach().cpu().numpy()[1:],
        }

    def on_train_start(self):
        super().on_train_start()
        if self.local_rank == 0:
            destination = Path(self.output_folder_base) / f"density_calibration_fold_{self.fold}.json"
            shutil.copy2(self.density_calibration_path, destination)


class nnUNetTrainerCorticalSeparatorControl(_nnUNetTrainerCorticalSeparatorPaired):
    """Density-neutral paired control with matched surface exposure."""

    use_density_prior = False


class nnUNetTrainerCorticalSeparatorDensityPrior(
    _nnUNetTrainerCorticalSeparatorPaired
):
    """Soft density-weighted separator trainer."""

    use_density_prior = True
