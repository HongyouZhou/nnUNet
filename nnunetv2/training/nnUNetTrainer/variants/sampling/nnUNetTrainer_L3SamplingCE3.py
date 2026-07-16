import os

import numpy as np
import torch
from batchgenerators.dataloading.nondet_multi_threaded_augmenter import NonDetMultiThreadedAugmenter
from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter
from torch import distributed as dist
from torch import nn
from torch._dynamo import OptimizedModule

from nnunetv2.training.dataloading.data_loader import nnUNetDataLoader
from nnunetv2.training.dataloading.label_oversampling_data_loader import nnUNetLabelTargetedDataLoader
from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class
from nnunetv2.training.loss.compound_losses import DC_and_CE_loss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.default_n_proc_DA import get_allowed_n_proc_DA
from nnunetv2.utilities.ddp_allgather import AllGatherGrad


class nnUNetTrainer_L3SamplingCE3(nnUNetTrainer):
    """
    Dataset777 ABBC trainer variant for emphasizing fracture/sticky regions:
    - probabilistic foreground oversampling
    - most foreground-oversampled patches are centered on label 3 when present
    - cross entropy assigns 3x weight to label 3
    """

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.label3_label = 3
        self.label3_sampling_probability = 0.75
        self.label3_ce_weight = 3.0
        self.probabilistic_oversampling = True
        self.oversample_foreground_percent = 0.7
        self.hard_patch_cases = ()
        self.hard_patch_force_fg_percent = 0.0
        self.hard_patch_oversample_label_percent = None
        self.print_to_log_file(
            "Label 3 emphasis enabled:",
            f"oversample_foreground_percent={self.oversample_foreground_percent}",
            f"label3_sampling_probability={self.label3_sampling_probability}",
            f"label3_ce_weight={self.label3_ce_weight}",
        )

    def _set_batch_size_and_oversample(self):
        if not self.is_ddp:
            self.batch_size = self.configuration_manager.batch_size
        else:
            world_size = dist.get_world_size()
            my_rank = dist.get_rank()

            global_batch_size = self.configuration_manager.batch_size
            assert global_batch_size >= world_size, 'Cannot run DDP if the batch size is smaller than the number of GPUs.'

            batch_size_per_gpu = [global_batch_size // world_size] * world_size
            batch_size_per_gpu = [
                batch_size_per_gpu[i] + 1
                if (batch_size_per_gpu[i] * world_size + i) < global_batch_size
                else batch_size_per_gpu[i]
                for i in range(len(batch_size_per_gpu))
            ]
            assert sum(batch_size_per_gpu) == global_batch_size
            print("worker", my_rank, "batch_size", batch_size_per_gpu[my_rank])
            print("worker", my_rank, "oversample", self.oversample_foreground_percent)

            self.batch_size = batch_size_per_gpu[my_rank]

    def _build_loss(self):
        if self.label_manager.has_regions:
            raise RuntimeError(f"{self.__class__.__name__} expects regular label training, not region-based labels.")
        if self.label3_label >= self.label_manager.num_segmentation_heads:
            raise RuntimeError(
                f"{self.__class__.__name__} expects label {self.label3_label}, "
                f"but this dataset has {self.label_manager.num_segmentation_heads} segmentation heads."
            )

        ce_class_weights = torch.ones(
            self.label_manager.num_segmentation_heads,
            dtype=torch.float32,
            device=self.device,
        )
        ce_class_weights[self.label3_label] = self.label3_ce_weight

        loss = DC_and_CE_loss(
            {
                'batch_dice': self.configuration_manager.batch_dice,
                'smooth': 1e-5,
                'do_bg': False,
                'ddp': self.is_ddp,
            },
            {'weight': ce_class_weights},
            weight_ce=1,
            weight_dice=1,
            ignore_label=self.label_manager.ignore_label,
            dice_class=MemoryEfficientSoftDiceLoss,
        )

        if self._do_i_compile():
            loss.dc = torch.compile(loss.dc)

        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            if self.is_ddp and not self._do_i_compile():
                weights[-1] = 1e-6
            else:
                weights[-1] = 0

            weights = weights / weights.sum()
            loss = DeepSupervisionWrapper(loss, weights)

        return loss

    def _get_training_sampling_probabilities(self, dataset_tr):
        return None

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

        tr_transforms = self.get_training_transforms(
            patch_size, rotation_for_DA, deep_supervision_scales, mirror_axes, do_dummy_2d_data_aug,
            use_mask_for_norm=self.configuration_manager.use_mask_for_norm,
            is_cascaded=self.is_cascaded, foreground_labels=self.label_manager.foreground_labels,
            regions=self.label_manager.foreground_regions if self.label_manager.has_regions else None,
            ignore_label=self.label_manager.ignore_label)

        val_transforms = self.get_validation_transforms(deep_supervision_scales,
                                                        is_cascaded=self.is_cascaded,
                                                        foreground_labels=self.label_manager.foreground_labels,
                                                        regions=self.label_manager.foreground_regions if
                                                        self.label_manager.has_regions else None,
                                                        ignore_label=self.label_manager.ignore_label)

        dataset_tr, dataset_val = self.get_tr_and_val_datasets()
        dl_tr = nnUNetLabelTargetedDataLoader(dataset_tr, self.batch_size,
                                              initial_patch_size,
                                              self.configuration_manager.patch_size,
                                              self.label_manager,
                                              oversample_foreground_percent=self.oversample_foreground_percent,
                                              sampling_probabilities=self._get_training_sampling_probabilities(dataset_tr),
                                              pad_sides=None, transforms=tr_transforms,
                                              probabilistic_oversampling=self.probabilistic_oversampling,
                                              oversample_label=self.label3_label,
                                              oversample_label_percent=self.label3_sampling_probability,
                                              hard_patch_cases=getattr(self, 'hard_patch_cases', ()),
                                              hard_patch_force_fg_percent=getattr(
                                                  self, 'hard_patch_force_fg_percent', 0.0),
                                              hard_patch_oversample_label_percent=getattr(
                                                  self, 'hard_patch_oversample_label_percent', None))
        dl_val = nnUNetDataLoader(dataset_val, self.batch_size,
                                  self.configuration_manager.patch_size,
                                  self.configuration_manager.patch_size,
                                  self.label_manager,
                                  oversample_foreground_percent=self.oversample_foreground_percent,
                                  sampling_probabilities=None, pad_sides=None, transforms=val_transforms,
                                  probabilistic_oversampling=self.probabilistic_oversampling)

        allowed_num_processes = get_allowed_n_proc_DA()
        if allowed_num_processes == 0:
            mt_gen_train = SingleThreadedAugmenter(dl_tr, None)
            mt_gen_val = SingleThreadedAugmenter(dl_val, None)
        else:
            mt_gen_train = NonDetMultiThreadedAugmenter(data_loader=dl_tr, transform=None,
                                                        num_processes=allowed_num_processes,
                                                        num_cached=max(6, allowed_num_processes // 2), seeds=None,
                                                        pin_memory=self.device.type == 'cuda', wait_time=0.002)
            mt_gen_val = NonDetMultiThreadedAugmenter(data_loader=dl_val,
                                                      transform=None, num_processes=max(1, allowed_num_processes // 2),
                                                      num_cached=max(3, allowed_num_processes // 4), seeds=None,
                                                      pin_memory=self.device.type == 'cuda',
                                                      wait_time=0.002)
        _ = next(mt_gen_train)
        _ = next(mt_gen_val)
        return mt_gen_train, mt_gen_val


class Label3BinaryDiceAuxLoss(nn.Module):
    def __init__(self, base_loss: nn.Module, label: int = 3, weight: float = 0.5, smooth: float = 1e-5,
                 batch_dice: bool = True, ddp: bool = False, ignore_label: int = None):
        super().__init__()
        self.base_loss = base_loss
        self.label = int(label)
        self.weight = float(weight)
        self.smooth = smooth
        self.batch_dice = batch_dice
        self.ddp = ddp
        self.ignore_label = ignore_label

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        main_loss = self.base_loss(net_output, target)

        if net_output.shape[1] <= self.label:
            raise RuntimeError(f'net_output has {net_output.shape[1]} channels, cannot compute label {self.label} Dice')

        target_labels = target[:, :1] if target.ndim == net_output.ndim else target[:, None]
        prob_label = torch.softmax(net_output, dim=1)[:, self.label:self.label + 1]
        target_label = (target_labels == self.label).to(dtype=prob_label.dtype)

        if self.ignore_label is not None:
            mask = (target_labels != self.ignore_label).to(dtype=prob_label.dtype)
            prob_label = prob_label * mask
            target_label = target_label * mask

        axes = tuple(range(2, prob_label.ndim))
        if self.batch_dice:
            axes = (0, *axes)

        intersect = (prob_label * target_label).sum(axes, dtype=torch.float32)
        sum_pred = prob_label.sum(axes, dtype=torch.float32)
        sum_gt = target_label.sum(axes, dtype=torch.float32)

        if self.ddp and self.batch_dice:
            intersect = AllGatherGrad.apply(intersect).sum(0, dtype=torch.float32)
            sum_pred = AllGatherGrad.apply(sum_pred).sum(0, dtype=torch.float32)
            sum_gt = AllGatherGrad.apply(sum_gt).sum(0, dtype=torch.float32)

        dice = (2 * intersect + self.smooth) / (sum_pred + sum_gt + self.smooth).clamp_min(1e-8)
        return main_loss - self.weight * dice.mean()


class nnUNetTrainer_L3SamplingCE3_HardSourceBalanced(nnUNetTrainer_L3SamplingCE3):
    charite_sampling_probability = 0.4
    pengwin_sampling_probability = 0.6
    hard_case_sampling_weight = 2.0

    hard_charite_cases = (
        'charite_3058',
        'charite_2394',
        'charite_3161',
        'charite_41',
        'charite_95',
        'charite_224',
        'charite_1602',
        'charite_16',
    )
    hard_pengwin_cases = (
        'pengwin_016',
        'pengwin_044',
        'pengwin_025',
        'pengwin_035',
        'pengwin_086',
        'pengwin_088',
        'pengwin_009',
        'pengwin_034',
    )

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.charite_sampling_probability = float(self.__class__.charite_sampling_probability)
        self.pengwin_sampling_probability = float(self.__class__.pengwin_sampling_probability)
        self.hard_case_sampling_weight = float(self.__class__.hard_case_sampling_weight)
        self.print_to_log_file(
            "Hard/source-balanced case sampling enabled:",
            f"charite_sampling_probability={self.charite_sampling_probability}",
            f"pengwin_sampling_probability={self.pengwin_sampling_probability}",
            f"hard_case_sampling_weight={self.hard_case_sampling_weight}",
        )

    @staticmethod
    def _source_of_case(case_identifier: str) -> str:
        if case_identifier.startswith('charite_'):
            return 'charite'
        if case_identifier.startswith('pengwin_'):
            return 'pengwin'
        return 'other'

    def _get_training_sampling_probabilities(self, dataset_tr):
        identifiers = list(dataset_tr.identifiers)
        if not identifiers:
            return None

        hard_cases = set(self.hard_charite_cases) | set(self.hard_pengwin_cases)
        source_targets = {
            'charite': self.charite_sampling_probability,
            'pengwin': self.pengwin_sampling_probability,
        }
        groups = {}
        for idx, case_identifier in enumerate(identifiers):
            groups.setdefault(self._source_of_case(case_identifier), []).append(idx)

        probabilities = np.zeros(len(identifiers), dtype=np.float64)
        for source, indices in groups.items():
            if source == 'other':
                continue
            weights = np.array([
                self.hard_case_sampling_weight if identifiers[idx] in hard_cases else 1.0
                for idx in indices
            ], dtype=np.float64)
            probabilities[indices] = source_targets[source] * weights / weights.sum()

        other_indices = groups.get('other', [])
        if other_indices:
            remaining_probability = max(0.0, 1.0 - probabilities.sum())
            probabilities[other_indices] = remaining_probability / len(other_indices)

        if probabilities.sum() <= 0:
            probabilities[:] = 1.0 / len(probabilities)
        else:
            probabilities /= probabilities.sum()

        source_sums = {
            source: float(probabilities[indices].sum())
            for source, indices in groups.items()
        }
        hard_cases_present = sorted([case_identifier for case_identifier in identifiers if case_identifier in hard_cases])
        self.print_to_log_file(
            "Training case sampling probabilities:",
            f"num_cases={len(identifiers)}",
            f"source_sums={source_sums}",
            f"hard_cases_present={hard_cases_present}",
        )
        return probabilities


class nnUNetTrainer_L3SamplingCE3_ChariteV2FineTune250(nnUNetTrainer_L3SamplingCE3_HardSourceBalanced):
    charite_sampling_probability = 0.5
    pengwin_sampling_probability = 0.5
    hard_case_sampling_weight = 3.0

    hard_charite_cases = (
        'charite_2394',
        'charite_3161',
        'charite_118',
        'charite_38',
        'charite_676',
        'charite_41',
        'charite_1753',
        'charite_5069',
        'charite_411',
        'charite_95',
        'charite_224',
        'charite_16',
        'charite_3058',
        'charite_1602',
    )
    hard_pengwin_cases = (
        'pengwin_016',
        'pengwin_044',
        'pengwin_025',
        'pengwin_035',
        'pengwin_086',
        'pengwin_088',
    )

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.initial_lr = 1e-3
        self.num_epochs = 250
        self.print_to_log_file(
            "Charite v2 fine-tune enabled:",
            f"initial_lr={self.initial_lr}",
            f"num_epochs={self.num_epochs}",
        )


class nnUNetTrainer_L3SamplingCE3_ChariteV3FineTune150(nnUNetTrainer_L3SamplingCE3_HardSourceBalanced):
    charite_sampling_probability = 0.55
    pengwin_sampling_probability = 0.45
    hard_case_sampling_weight = 4.0
    full_init_checkpoint_env_var = 'NNUNET_FULL_INIT_CHECKPOINT'

    hard_charite_cases = (
        'charite_2394',
        'charite_3161',
        'charite_38',
        'charite_118',
        'charite_41',
        'charite_676',
        'charite_95',
        'charite_1753',
        'charite_16',
        'charite_5069',
        'charite_224',
        'charite_411',
    )
    hard_pengwin_cases = (
        'pengwin_016',
        'pengwin_044',
        'pengwin_025',
        'pengwin_035',
        'pengwin_086',
        'pengwin_088',
    )

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.initial_lr = 3e-4
        self.num_epochs = 150
        self.print_to_log_file(
            "Charite v3 low-LR fine-tune enabled:",
            f"initial_lr={self.initial_lr}",
            f"num_epochs={self.num_epochs}",
            f"full_init_checkpoint_env_var={self.full_init_checkpoint_env_var}",
        )

    def initialize(self):
        super().initialize()
        full_init_checkpoint = os.environ.get(self.full_init_checkpoint_env_var)
        if not full_init_checkpoint:
            self.print_to_log_file(
                "No full network initialization checkpoint provided; "
                "assuming checkpoint loading or an intentional from-scratch run.",
                f"env_var={self.full_init_checkpoint_env_var}",
            )
            return
        if not os.path.isfile(full_init_checkpoint):
            raise FileNotFoundError(full_init_checkpoint)

        checkpoint = torch.load(full_init_checkpoint, map_location=self.device, weights_only=False)
        if self.is_ddp:
            module = self.network.module
        else:
            module = self.network
        if isinstance(module, OptimizedModule):
            module = module._orig_mod

        new_state_dict = {}
        current_state_dict = module.state_dict()
        for key, value in checkpoint['network_weights'].items():
            target_key = key
            if target_key not in current_state_dict and target_key.startswith('module.'):
                target_key = target_key[7:]
            new_state_dict[target_key] = value

        module.load_state_dict(new_state_dict)

        self.print_to_log_file(
            "Loaded full network initialization checkpoint:",
            full_init_checkpoint,
        )


class nnUNetTrainer_L3SamplingCE3_ChariteV4FineTune100(nnUNetTrainer_L3SamplingCE3_ChariteV3FineTune150):
    charite_sampling_probability = 0.55
    pengwin_sampling_probability = 0.45
    hard_case_sampling_weight = 3.5

    hard_charite_cases = (
        'charite_2394',
        'charite_3161',
        'charite_38',
        'charite_118',
        'charite_41',
        'charite_676',
        'charite_95',
        'charite_1753',
        'charite_16',
        'charite_5069',
        'charite_224',
        'charite_411',
        'charite_3058',
        'charite_1602',
    )
    hard_pengwin_cases = (
        'pengwin_016',
        'pengwin_044',
        'pengwin_025',
        'pengwin_035',
        'pengwin_086',
        'pengwin_088',
    )

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        nnUNetTrainer_L3SamplingCE3_HardSourceBalanced.__init__(self, plans, configuration, fold, dataset_json, device)
        self.initial_lr = 1e-4
        self.num_epochs = 100
        self.print_to_log_file(
            "Charite v4 low-LR fine-tune enabled:",
            f"initial_lr={self.initial_lr}",
            f"num_epochs={self.num_epochs}",
            f"full_init_checkpoint_env_var={self.full_init_checkpoint_env_var}",
        )


class nnUNetTrainer_L3SamplingCE3_ChariteV5HardPatchFineTune80(nnUNetTrainer_L3SamplingCE3_ChariteV4FineTune100):
    charite_sampling_probability = 0.58
    pengwin_sampling_probability = 0.42
    hard_case_sampling_weight = 4.0

    hard_charite_cases = (
        'charite_2394',
        'charite_3161',
        'charite_38',
        'charite_118',
        'charite_95',
        'charite_41',
        'charite_676',
        'charite_1753',
        'charite_16',
        'charite_3058',
        'charite_5069',
        'charite_411',
        'charite_224',
        'charite_154',
        'charite_1385',
        'charite_103',
        'charite_5097',
    )
    hard_pengwin_cases = (
        'pengwin_016',
        'pengwin_044',
        'pengwin_035',
        'pengwin_019',
        'pengwin_022',
        'pengwin_094',
        'pengwin_018',
    )

    hard_patch_cases = (
        'charite_2394',
        'charite_3161',
        'charite_38',
        'charite_118',
        'charite_95',
        'charite_41',
        'charite_676',
        'charite_1753',
        'charite_16',
        'charite_3058',
        'charite_5069',
        'charite_411',
        'charite_224',
        'charite_154',
        'pengwin_016',
        'pengwin_044',
        'pengwin_035',
    )

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        nnUNetTrainer_L3SamplingCE3_HardSourceBalanced.__init__(self, plans, configuration, fold, dataset_json, device)
        self.initial_lr = 5e-5
        self.num_epochs = 80
        self.hard_patch_cases = tuple(self.__class__.hard_patch_cases)
        self.hard_patch_force_fg_percent = 0.95
        self.hard_patch_oversample_label_percent = 0.95
        self.print_to_log_file(
            "Charite v5 hard-patch fine-tune enabled:",
            f"initial_lr={self.initial_lr}",
            f"num_epochs={self.num_epochs}",
            f"full_init_checkpoint_env_var={self.full_init_checkpoint_env_var}",
            f"hard_patch_force_fg_percent={self.hard_patch_force_fg_percent}",
            f"hard_patch_oversample_label_percent={self.hard_patch_oversample_label_percent}",
            f"hard_patch_cases={self.hard_patch_cases}",
        )


class nnUNetTrainer_L3SamplingCE3_BinaryDice05(nnUNetTrainer_L3SamplingCE3):
    """
    Adds a label-3 binary Dice auxiliary loss on top of the successful L3 sampling + CE3 setup.
    """

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.label3_binary_dice_weight = 0.5
        self.print_to_log_file(
            "Label 3 binary Dice auxiliary loss enabled:",
            f"label3_binary_dice_weight={self.label3_binary_dice_weight}",
        )

    def _build_loss(self):
        if self.label_manager.has_regions:
            raise RuntimeError(f"{self.__class__.__name__} expects regular label training, not region-based labels.")
        if self.label3_label >= self.label_manager.num_segmentation_heads:
            raise RuntimeError(
                f"{self.__class__.__name__} expects label {self.label3_label}, "
                f"but this dataset has {self.label_manager.num_segmentation_heads} segmentation heads."
            )

        ce_class_weights = torch.ones(
            self.label_manager.num_segmentation_heads,
            dtype=torch.float32,
            device=self.device,
        )
        ce_class_weights[self.label3_label] = self.label3_ce_weight

        base_loss = DC_and_CE_loss(
            {
                'batch_dice': self.configuration_manager.batch_dice,
                'smooth': 1e-5,
                'do_bg': False,
                'ddp': self.is_ddp,
            },
            {'weight': ce_class_weights},
            weight_ce=1,
            weight_dice=1,
            ignore_label=self.label_manager.ignore_label,
            dice_class=MemoryEfficientSoftDiceLoss,
        )

        if self._do_i_compile():
            base_loss.dc = torch.compile(base_loss.dc)

        loss = Label3BinaryDiceAuxLoss(
            base_loss,
            label=self.label3_label,
            weight=self.label3_binary_dice_weight,
            smooth=1e-5,
            batch_dice=self.configuration_manager.batch_dice,
            ddp=self.is_ddp,
            ignore_label=self.label_manager.ignore_label,
        )

        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            if self.is_ddp and not self._do_i_compile():
                weights[-1] = 1e-6
            else:
                weights[-1] = 0

            weights = weights / weights.sum()
            loss = DeepSupervisionWrapper(loss, weights)

        return loss
