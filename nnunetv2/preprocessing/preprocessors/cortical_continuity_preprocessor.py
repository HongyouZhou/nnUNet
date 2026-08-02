"""Preprocessor for the frozen Charité cortical-continuity sidecar contract."""

from __future__ import annotations

from pathlib import Path
from typing import List, Union

import numpy as np
from batchgenerators.utilities.file_and_folder_operations import load_json

from nnunetv2.preprocessing.preprocessors.charite_supervision_roi import (
    crop_to_supervision_roi,
    restore_full_grid_crop_properties,
)
from nnunetv2.preprocessing.preprocessors.default_preprocessor import DefaultPreprocessor
from nnunetv2.training.cortical_continuity.packing import (
    PREPROCESSED_SOURCE_CHANNELS,
    SOURCE_CHANNELS,
    build_source_segmentation,
    semantic_contact_mask,
)
from nnunetv2.training.cortical_continuity.sampling import (
    CONTINUITY_CONTACT_KEY,
    CONTINUITY_INSTANCE_KEY_PREFIX,
    CONTINUITY_INSTANCES_PROPERTY,
    CONTINUITY_SUPPORT_KEY,
)
from nnunetv2.training.cortical_continuity.sidecar_contract import (
    assert_exact_grid as _assert_exact_grid,
    assert_instance_retention,
)
from nnunetv2.training.cortical_separator_prior.contract import (
    HU_CODE_OFFSET,
    HU_CODE_CHANNEL,
    PRIOR_CONTRACT_VERSION,
    PRIOR_PLANS_KEY,
    PRIOR_SEPARATOR_KEY,
    PRIOR_SURFACE_HU_KEY,
    PRIOR_SURFACE_KEY,
    SEMANTIC_VALID_BIT,
    TARGET_CHANNELS,
    encode_hu,
)
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager


_SIDECAR_DIRECTORIES = {
    "instances": "corticalInstancesTr",
    "overlap": "corticalOverlapTr",
    "validity": "validMasksTr",
    "support": "supportTr",
}
_TEMPORARY_CONTACT_CHANNEL = len(SOURCE_CHANNELS)
_SOURCE_CONTRACT_VERSION = 1


class ChariteCorticalPreprocessor(DefaultPreprocessor):
    """Load five synchronized source channels and retain target-aware centres.

    Raw training labels are resolved from the semantic ``labelsTr`` path:

    ``[C, I, O, U, S]`` = binary cortex, cortical instance ID, overlap,
    validity bitmask, and support. A temporary label-2 channel survives
    transpose/crop/resampling only long enough to compute contact coordinates.
    Three persisted case-constant channels encode the native minimum resolved
    z/y/x step, allowing dynamic affinities below native through-plane
    resolution to be ignored after augmentation.
    """

    def run_case(
        self,
        image_files: List[str],
        seg_file: Union[str, None],
        plans_manager: PlansManager,
        configuration_manager: ConfigurationManager,
        dataset_json: Union[dict, str],
    ):
        if seg_file is None:
            return super().run_case(
                image_files,
                seg_file,
                plans_manager,
                configuration_manager,
                dataset_json,
            )
        if isinstance(dataset_json, str):
            dataset_json = load_json(dataset_json)

        semantic_path, sidecar_paths = _resolve_sidecar_paths(seg_file, dataset_json)
        rw = plans_manager.image_reader_writer_class()
        data, data_properties = rw.read_images(image_files)
        semantic, semantic_properties = rw.read_seg(str(semantic_path))
        loaded: dict[str, np.ndarray] = {}
        for role, path in sidecar_paths.items():
            value, value_properties = rw.read_seg(str(path))
            _assert_exact_grid(
                data,
                data_properties,
                value,
                value_properties,
                role=role,
                path=path,
            )
            loaded[role] = value
        _assert_exact_grid(
            data,
            data_properties,
            semantic,
            semantic_properties,
            role="semantic label",
            path=semantic_path,
        )

        source = build_source_segmentation(
            semantic,
            loaded["instances"],
            loaded["overlap"],
            loaded["validity"],
            loaded["support"],
        )
        contact = semantic_contact_mask(semantic).astype(np.int16, copy=False)
        source_with_contact = np.concatenate((source, contact[None]), axis=0)
        data, source_with_contact, roi_crop = crop_to_supervision_roi(
            data,
            source_with_contact,
            loaded["support"] == 1,
            data_properties,
        )
        # Release the full-grid sidecars before allocating resampled targets.
        loaded.clear()
        del loaded, semantic, source, contact
        if self.verbose:
            print(seg_file)
        processed_data, processed_seg, processed_properties = self.run_case_npy(
            data,
            source_with_contact,
            data_properties,
            plans_manager,
            configuration_manager,
            dataset_json,
        )
        restore_full_grid_crop_properties(
            processed_properties,
            roi_crop,
            plans_manager.transpose_forward,
        )
        return processed_data, processed_seg, processed_properties

    def run_case_npy(
        self,
        data: np.ndarray,
        seg: Union[np.ndarray, None],
        properties: dict,
        plans_manager: PlansManager,
        configuration_manager: ConfigurationManager,
        dataset_json: Union[dict, str],
    ):
        if seg is None:
            return super().run_case_npy(
                data,
                seg,
                properties,
                plans_manager,
                configuration_manager,
                dataset_json,
            )
        source = np.asarray(seg)
        if source.ndim != 4 or source.shape[0] not in {
            len(SOURCE_CHANNELS),
            len(SOURCE_CHANNELS) + 1,
        }:
            raise ValueError(
                "CorticalContinuityPreprocessor requires [C,I,O,U,S] "
                f"(plus an internal contact channel); got {source.shape}"
            )
        if source.shape[0] == len(SOURCE_CHANNELS):
            source = np.concatenate(
                (source, np.zeros((1, *source.shape[1:]), dtype=source.dtype)),
                axis=0,
            )
        source_instances_before_resampling = np.asarray(source[1]).copy()
        native_spacing_zyx = np.asarray(properties["spacing"], dtype=np.float64)
        target_spacing_zyx = np.asarray(
            configuration_manager.spacing,
            dtype=np.float64,
        )
        native_min_steps_zyx = np.maximum(
            1,
            np.ceil(native_spacing_zyx / target_spacing_zyx - 1e-8),
        ).astype(np.int16)

        data, processed, properties = super().run_case_npy(
            data,
            source,
            properties,
            plans_manager,
            configuration_manager,
            dataset_json,
        )
        _validate_processed_source(processed)
        retention = assert_instance_retention(
            source_instances_before_resampling,
            processed[1],
        )
        contact = processed[_TEMPORARY_CONTACT_CHANNEL] == 1
        processed = processed[: len(SOURCE_CHANNELS)]
        minimum_step_channels = np.stack(
            [
                np.full(
                    processed.shape[1:],
                    int(minimum_steps),
                    dtype=processed.dtype,
                )
                for minimum_steps in native_min_steps_zyx
            ],
            axis=0,
        )
        processed = np.concatenate(
            (processed, minimum_step_channels),
            axis=0,
        )

        standard_locations = _standard_class_locations(
            self,
            processed,
            plans_manager,
            dataset_json,
        )
        contact_locations = _sample_mask_coordinates(contact, seed=1201)
        support_locations = _sample_mask_coordinates(processed[4] == 1, seed=1202)
        instance_locations = {
            int(instance_id): _sample_mask_coordinates(
                processed[1] == instance_id,
                seed=1300 + int(instance_id),
            )
            for instance_id in np.unique(processed[1])
            if int(instance_id) > 0
        }

        # The custom training loader receives class_locations through the
        # standard nnU-Net loader API. Keep normal label-manager keys as well so
        # the validation loader retains standard random/foreground-centred
        # behaviour.
        standard_locations[CONTINUITY_CONTACT_KEY] = contact_locations
        standard_locations[CONTINUITY_SUPPORT_KEY] = support_locations
        for instance_id, coordinates in instance_locations.items():
            standard_locations[(CONTINUITY_INSTANCE_KEY_PREFIX, instance_id)] = coordinates

        properties["class_locations"] = standard_locations
        properties[CONTINUITY_CONTACT_KEY] = contact_locations
        properties[CONTINUITY_INSTANCES_PROPERTY] = instance_locations
        properties[CONTINUITY_SUPPORT_KEY] = support_locations
        properties["cortical_continuity_source_contract_version"] = _SOURCE_CONTRACT_VERSION
        properties["cortical_continuity_source_channels"] = list(
            PREPROCESSED_SOURCE_CHANNELS
        )
        properties["cortical_continuity_native_spacing_mm_zyx"] = [
            float(value) for value in native_spacing_zyx
        ]
        properties["cortical_continuity_native_min_steps_zyx"] = [
            int(value) for value in native_min_steps_zyx
        ]
        properties["cortical_continuity_thin_cortex_retention"] = retention
        properties["cortical_continuity_spacing_mm_zyx"] = [
            float(i) for i in configuration_manager.spacing
        ]
        return data, processed, properties


# Backward-compatible descriptive name. The frozen plans use
# ``ChariteCorticalPreprocessor``.
CorticalContinuityPreprocessor = ChariteCorticalPreprocessor


class ChariteSeparatorPreprocessor(DefaultPreprocessor):
    """Crop separator training cases around supervised cortical support."""

    def run_case(
        self,
        image_files: List[str],
        seg_file: Union[str, None],
        plans_manager: PlansManager,
        configuration_manager: ConfigurationManager,
        dataset_json: Union[dict, str],
    ):
        if seg_file is None:
            return super().run_case(
                image_files,
                seg_file,
                plans_manager,
                configuration_manager,
                dataset_json,
            )
        if isinstance(dataset_json, str):
            dataset_json = load_json(dataset_json)
        rw = plans_manager.image_reader_writer_class()
        data, data_properties = rw.read_images(image_files)
        semantic, semantic_properties = rw.read_seg(seg_file)
        _assert_exact_grid(
            data,
            data_properties,
            semantic,
            semantic_properties,
            role="separator semantic label",
            path=seg_file,
        )
        data, semantic, roi_crop = crop_to_supervision_roi(
            data,
            semantic,
            semantic != 0,
            data_properties,
        )
        processed_data, processed_seg, processed_properties = super().run_case_npy(
            data,
            semantic,
            data_properties,
            plans_manager,
            configuration_manager,
            dataset_json,
        )
        restore_full_grid_crop_properties(
            processed_properties,
            roi_crop,
            plans_manager.transpose_forward,
        )
        return processed_data, processed_seg, processed_properties


class ChariteDensityPriorSeparatorPreprocessor(DefaultPreprocessor):
    """Persist semantic/support/validity/surface/HU targets for paired training.

    The HU channel is reconstructed immediately after CT normalization and
    encoded as a discrete target. It therefore follows spatial augmentation
    but is never modified by intensity augmentation. The network still sees
    the original single normalized CT input channel only.
    """

    def run_case(
        self,
        image_files: List[str],
        seg_file: Union[str, None],
        plans_manager: PlansManager,
        configuration_manager: ConfigurationManager,
        dataset_json: Union[dict, str],
    ):
        if seg_file is None:
            return super().run_case(
                image_files,
                seg_file,
                plans_manager,
                configuration_manager,
                dataset_json,
            )
        if isinstance(dataset_json, str):
            dataset_json = load_json(dataset_json)
        contract = plans_manager.plans.get(PRIOR_PLANS_KEY)
        if not isinstance(contract, dict) or int(contract.get("schema_version", -1)) != PRIOR_CONTRACT_VERSION:
            raise RuntimeError(
                f"Plans must contain {PRIOR_PLANS_KEY} schema {PRIOR_CONTRACT_VERSION}"
            )

        semantic_path, sidecar_paths = _resolve_sidecar_paths(seg_file, dataset_json)
        rw = plans_manager.image_reader_writer_class()
        data, data_properties = rw.read_images(image_files)
        semantic, semantic_properties = rw.read_seg(str(semantic_path))
        support, support_properties = rw.read_seg(str(sidecar_paths["support"]))
        validity, validity_properties = rw.read_seg(str(sidecar_paths["validity"]))
        for role, value, properties, path in (
            ("semantic", semantic, semantic_properties, semantic_path),
            ("support", support, support_properties, sidecar_paths["support"]),
            ("validity", validity, validity_properties, sidecar_paths["validity"]),
        ):
            _assert_exact_grid(
                data,
                data_properties,
                value,
                properties,
                role=role,
                path=path,
            )

        source = np.concatenate((semantic, support, validity), axis=0)
        data, source, roi_crop = crop_to_supervision_roi(
            data,
            source,
            support == 1,
            data_properties,
        )
        processed_data, processed, processed_properties = super().run_case_npy(
            data,
            source,
            data_properties,
            plans_manager,
            configuration_manager,
            dataset_json,
        )
        restore_full_grid_crop_properties(
            processed_properties,
            roi_crop,
            plans_manager.transpose_forward,
        )
        if processed.shape[0] != 3:
            raise RuntimeError(
                "Density-prior separator target must contain semantic/support/validity "
                f"before HU packing; got {processed.shape}"
            )
        _validate_values_with_crop_sentinel(processed[0], {-1, 0, 1, 2, 3}, "separator semantic")
        _validate_values_with_crop_sentinel(processed[1], {-1, 0, 1}, "fragment support")
        _validate_values_with_crop_sentinel(
            processed[2], {-1, 0, 1, 2, 3, 4, 5, 6, 7}, "validity bitmask"
        )

        schemes = tuple(str(value) for value in configuration_manager.normalization_schemes)
        if schemes != ("CTNormalization",):
            raise RuntimeError(
                "Density-prior separator requires one CTNormalization input channel; "
                f"got {schemes}"
            )
        intensity = plans_manager.foreground_intensity_properties_per_channel["0"]
        mean = float(intensity["mean"])
        std = float(intensity["std"])
        hu = processed_data[0].astype(np.float32, copy=False) * std + mean
        hu_code = encode_hu(hu)

        semantic_processed = np.asarray(processed[0])
        support_processed = np.asarray(processed[1]) == 1
        validity_processed = np.asarray(processed[2], dtype=np.int16)
        valid = (validity_processed & SEMANTIC_VALID_BIT) != 0
        separator = semantic_processed == 2
        normal_surface = _normal_fragment_surface(
            support_processed,
            separator,
            valid,
            spacing_mm_zyx=configuration_manager.spacing,
            separator_exclusion_mm=2.0,
        )

        standard_locations = _standard_class_locations(
            self,
            processed[:1],
            plans_manager,
            dataset_json,
        )
        separator_locations = _sample_mask_coordinates(separator, seed=2201)
        surface_locations = _sample_mask_coordinates(normal_surface, seed=2202)
        if len(surface_locations):
            indices = tuple(surface_locations[:, axis] for axis in range(1, 4))
            surface_hu = hu_code[indices].astype(np.int32) - HU_CODE_OFFSET
            surface_records = np.concatenate(
                (surface_locations.astype(np.int32), surface_hu[:, None]),
                axis=1,
            )
        else:
            surface_records = np.empty((0, 5), dtype=np.int32)
        standard_locations[PRIOR_SEPARATOR_KEY] = separator_locations
        standard_locations[PRIOR_SURFACE_KEY] = surface_locations
        standard_locations[PRIOR_SURFACE_HU_KEY] = surface_records
        processed_properties["class_locations"] = standard_locations
        processed_properties[PRIOR_SEPARATOR_KEY] = separator_locations
        processed_properties[PRIOR_SURFACE_KEY] = surface_locations
        processed_properties[PRIOR_SURFACE_HU_KEY] = surface_records
        processed_properties[PRIOR_PLANS_KEY] = {
            "schema_version": PRIOR_CONTRACT_VERSION,
            "target_channels": TARGET_CHANNELS,
            "hu_code_channel": HU_CODE_CHANNEL,
            "normal_surface_voxels": int(np.count_nonzero(normal_surface)),
            "separator_voxels": int(np.count_nonzero(separator)),
        }

        packed = np.concatenate(
            (
                processed.astype(np.int16, copy=False),
                normal_surface.astype(np.int16, copy=False)[None],
                hu_code[None],
            ),
            axis=0,
        )
        if packed.shape[0] != TARGET_CHANNELS:
            raise AssertionError(packed.shape)
        return processed_data, packed, processed_properties


def _normal_fragment_surface(
    support: np.ndarray,
    separator: np.ndarray,
    semantic_valid: np.ndarray,
    *,
    spacing_mm_zyx: Union[List[float], tuple[float, ...], np.ndarray],
    separator_exclusion_mm: float,
) -> np.ndarray:
    try:
        from scipy.ndimage import binary_erosion, distance_transform_edt
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "SciPy is required to construct density-prior normal surfaces"
        ) from error
    support_mask = np.asarray(support, dtype=bool)
    separator_mask = np.asarray(separator, dtype=bool)
    valid_mask = np.asarray(semantic_valid, dtype=bool)
    if not (support_mask.shape == separator_mask.shape == valid_mask.shape):
        raise ValueError("support, separator, and semantic_valid shapes differ")
    inner_surface = support_mask & ~binary_erosion(
        support_mask,
        structure=np.ones((3, 3, 3), dtype=bool),
        border_value=0,
    )
    if np.any(separator_mask):
        distance = distance_transform_edt(
            ~separator_mask,
            sampling=tuple(float(value) for value in spacing_mm_zyx),
        )
        away_from_separator = distance > float(separator_exclusion_mm)
    else:
        away_from_separator = np.ones(separator_mask.shape, dtype=bool)
    return inner_surface & valid_mask & away_from_separator


def _resolve_sidecar_paths(
    seg_file: str,
    dataset_json: dict,
) -> tuple[Path, dict[str, Path]]:
    semantic_path = Path(seg_file).expanduser().resolve(strict=True)
    if semantic_path.parent.name != "labelsTr":
        raise ValueError(
            "Cortical continuity semantic labels must be under labelsTr; "
            f"got {semantic_path}"
        )
    try:
        file_ending = str(dataset_json["file_ending"])
    except KeyError as exc:
        raise KeyError("dataset.json is missing required file_ending") from exc
    if not semantic_path.name.endswith(file_ending):
        raise ValueError(
            f"Semantic label {semantic_path.name!r} does not end with dataset file ending "
            f"{file_ending!r}"
        )
    case_identifier = semantic_path.name[: -len(file_ending)]
    dataset_root = semantic_path.parent.parent
    paths = {
        role: dataset_root / directory / f"{case_identifier}{file_ending}"
        for role, directory in _SIDECAR_DIRECTORIES.items()
    }
    missing = [f"{role}={path}" for role, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Cortical continuity sidecar contract is incomplete for "
            f"{case_identifier}: {', '.join(missing)}"
        )
    return semantic_path, paths


def _validate_processed_source(processed: np.ndarray) -> None:
    if processed.ndim != 4 or processed.shape[0] != len(SOURCE_CHANNELS) + 1:
        raise ValueError(
            "Resampling changed the cortical source channel contract; "
            f"got {processed.shape}"
        )
    if not np.isfinite(processed).all() or not np.allclose(processed, np.rint(processed), atol=1e-4):
        raise ValueError("Resampled cortical source channels are not discrete")
    value = np.rint(processed).astype(np.int64, copy=False)
    _validate_values_with_crop_sentinel(value[0], {-1, 0, 1}, "cortex")
    if np.any(value[1] < -1):
        raise ValueError("Resampled cortical instance IDs contain values below the crop sentinel -1")
    _validate_values_with_crop_sentinel(value[2], {-1, 0, 1}, "overlap")
    allowed_valid = {-1, 0, 1, 2, 3, 4, 5, 6, 7}
    _validate_values_with_crop_sentinel(value[3], allowed_valid, "validity bitmask")
    _validate_values_with_crop_sentinel(value[4], {-1, 0, 1}, "support")
    _validate_values_with_crop_sentinel(value[5], {-1, 0, 1}, "contact")


def _validate_values_with_crop_sentinel(
    value: np.ndarray,
    allowed: set[int],
    name: str,
) -> None:
    invalid = set(int(i) for i in np.unique(value)).difference(allowed)
    if invalid:
        raise ValueError(f"Resampled {name} contains invalid values {sorted(invalid)}")


def _standard_class_locations(
    preprocessor: DefaultPreprocessor,
    processed: np.ndarray,
    plans_manager: PlansManager,
    dataset_json: Union[dict, str],
) -> dict:
    label_manager = plans_manager.get_label_manager(dataset_json)
    collect = list(
        label_manager.foreground_regions
        if label_manager.has_regions
        else label_manager.foreground_labels
    )
    if label_manager.has_ignore_label:
        collect.append(tuple([-1] + list(label_manager.all_labels)))
    return preprocessor._sample_foreground_locations(
        processed[:1],
        collect,
        verbose=preprocessor.verbose,
    )


def _sample_mask_coordinates(
    mask: np.ndarray,
    *,
    seed: int,
    min_num_samples: int = 10_000,
    min_percent_coverage: float = 0.01,
) -> np.ndarray:
    coordinates = np.argwhere(np.asarray(mask, dtype=bool))
    if coordinates.size == 0:
        return np.empty((0, mask.ndim + 1), dtype=np.int64)
    target = min(min_num_samples, len(coordinates))
    target = max(target, int(np.ceil(len(coordinates) * min_percent_coverage)))
    if target < len(coordinates):
        random_state = np.random.RandomState(seed)
        coordinates = coordinates[
            random_state.choice(len(coordinates), target, replace=False)
        ]
    channel = np.zeros((len(coordinates), 1), dtype=np.int64)
    return np.concatenate((channel, coordinates.astype(np.int64, copy=False)), axis=1)
