from typing import Iterable, Optional

import numpy as np
import torch
from acvl_utils.cropping_and_padding.bounding_boxes import crop_and_pad_nd
from threadpoolctl import threadpool_limits

from nnunetv2.training.dataloading.data_loader import nnUNetDataLoader


class nnUNetLabelTargetedDataLoader(nnUNetDataLoader):
    def __init__(
        self,
        *args,
        oversample_label: int = 3,
        oversample_label_percent: float = 0.75,
        hard_patch_cases: Optional[Iterable[str]] = None,
        hard_patch_force_fg_percent: float = 0.0,
        hard_patch_oversample_label_percent: Optional[float] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.oversample_label = int(oversample_label)
        self.oversample_label_percent = float(oversample_label_percent)
        self.hard_patch_cases = set(hard_patch_cases or ())
        self.hard_patch_force_fg_percent = float(hard_patch_force_fg_percent)
        self.hard_patch_oversample_label_percent = (
            None
            if hard_patch_oversample_label_percent is None
            else float(hard_patch_oversample_label_percent)
        )

    def _target_class_key(self, class_locations: dict) -> Optional[int]:
        if class_locations is None:
            return None

        for class_key, locations in class_locations.items():
            try:
                key_matches = not isinstance(class_key, tuple) and int(class_key) == self.oversample_label
            except (TypeError, ValueError):
                key_matches = False

            if key_matches and len(locations) > 0:
                return class_key

        return None

    def _is_hard_patch_case(self, case_identifier: str) -> bool:
        return str(case_identifier) in self.hard_patch_cases

    def _get_label_oversample_percent(self, case_identifier: str) -> float:
        if self._is_hard_patch_case(case_identifier) and self.hard_patch_oversample_label_percent is not None:
            return self.hard_patch_oversample_label_percent
        return self.oversample_label_percent

    def _get_overwrite_class(self, force_fg: bool, class_locations: dict,
                             case_identifier: str) -> Optional[int]:
        oversample_label_percent = self._get_label_oversample_percent(case_identifier)
        if not force_fg or oversample_label_percent <= 0:
            return None
        if np.random.uniform() >= oversample_label_percent:
            return None
        return self._target_class_key(class_locations)

    def generate_train_batch(self):
        selected_keys = self.get_indices()
        data_all = None
        seg_all = None

        with torch.no_grad():
            with threadpool_limits(limits=1, user_api=None):
                for j, i in enumerate(selected_keys):
                    force_fg = self.get_do_oversample(j)
                    if self._is_hard_patch_case(i) and np.random.uniform() < self.hard_patch_force_fg_percent:
                        force_fg = True

                    data, seg, seg_prev, properties = self._data.load_case(i)
                    shape = data.shape[1:]
                    class_locations = properties['class_locations']
                    overwrite_class = self._get_overwrite_class(force_fg, class_locations, i)

                    bbox_lbs, bbox_ubs = self.get_bbox(
                        shape,
                        force_fg,
                        class_locations,
                        overwrite_class=overwrite_class,
                    )
                    bbox = [[i, j] for i, j in zip(bbox_lbs, bbox_ubs)]

                    data_cropped = torch.from_numpy(crop_and_pad_nd(data, bbox, 0)).float()
                    seg_cropped = torch.from_numpy(
                        crop_and_pad_nd(seg, bbox, -1, cast_cropped_to=np.int16)
                    ).to(torch.int16)
                    if seg_prev is not None:
                        seg_prev_cropped = torch.from_numpy(
                            crop_and_pad_nd(seg_prev, bbox, -1, cast_cropped_to=np.int16)
                        ).to(torch.int16)
                        seg_cropped = torch.cat((seg_cropped, seg_prev_cropped[None]), dim=0)

                    if self.patch_size_was_2d:
                        data_cropped = data_cropped[:, 0]
                        seg_cropped = seg_cropped[:, 0]

                    if self.transforms is not None:
                        transformed = self.transforms(**{'image': data_cropped, 'segmentation': seg_cropped})
                        data_sample = transformed['image']
                        seg_sample = transformed['segmentation']
                    else:
                        data_sample = data_cropped
                        seg_sample = seg_cropped

                    if data_all is None:
                        data_all = torch.empty((self.batch_size, *data_sample.shape), dtype=torch.float32)
                    data_all[j] = data_sample

                    if isinstance(seg_sample, list):
                        if seg_all is None:
                            seg_all = [torch.empty((self.batch_size, *s.shape), dtype=s.dtype) for s in seg_sample]
                        for s_idx, s in enumerate(seg_sample):
                            seg_all[s_idx][j] = s
                    else:
                        if seg_all is None:
                            seg_all = torch.empty((self.batch_size, *seg_sample.shape), dtype=seg_sample.dtype)
                        seg_all[j] = seg_sample
        return {'data': data_all, 'target': seg_all, 'keys': selected_keys}
