"""Independent six-channel preprocessor for separator continuity priors."""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Union

import numpy as np
from batchgenerators.utilities.file_and_folder_operations import load_json

from nnunetv2.preprocessing.preprocessors.charite_supervision_roi import (
    crop_to_supervision_roi,
    restore_full_grid_crop_properties,
)
from nnunetv2.preprocessing.preprocessors.default_preprocessor import DefaultPreprocessor
from nnunetv2.training.cortical_separator_continuity.contract import (
    CONTINUITY_CONTRACT_VERSION,
    CONTINUITY_PLANS_KEY,
    HU_CODE_CHANNEL,
    INSTANCE_CHANNEL,
    RELATION_VALID_BIT,
    SEMANTIC_VALID_BIT,
    SEPARATOR_LABEL,
    SEPARATOR_SAMPLING_KEY,
    SURFACE_SAMPLING_KEY,
    TARGET_CHANNELS,
    encode_hu,
    validate_continuity_plans_contract,
)
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager


_SIDECAR_DIRECTORIES = {
    "support": "supportTr",
    "validity": "validMasksTr",
    "instances": "corticalInstancesTr",
}


class ChariteCorticalSeparatorContinuityPreprocessor(DefaultPreprocessor):
    """Build ``semantic/support/validity/surface/HU/instance`` targets.

    Instance IDs travel as a segmentation channel and are nearest-neighbour
    resampled. HU code is reconstructed from the normalized/resampled CT and
    then stored in the target, so later intensity augmentation cannot alter
    it. Neither channel is included in the network input.
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
        validate_continuity_plans_contract(
            plans_manager.plans, configuration_manager.data_identifier
        )

        semantic_path, sidecar_paths = _resolve_sidecar_paths(seg_file, dataset_json)
        reader_writer = plans_manager.image_reader_writer_class()
        data, data_properties = reader_writer.read_images(image_files)
        if data.shape[0] != 1:
            raise ValueError(
                "Separator continuity preprocessing requires exactly one CT channel"
            )
        semantic, semantic_properties = reader_writer.read_seg(str(semantic_path))
        loaded: dict[str, np.ndarray] = {}
        for role, path in sidecar_paths.items():
            value, value_properties = reader_writer.read_seg(str(path))
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
            role="semantic",
            path=semantic_path,
        )

        semantic_3d = _discrete_channel(semantic, "semantic")
        support_3d = _discrete_channel(loaded["support"], "support")
        validity_3d = _discrete_channel(loaded["validity"], "validity")
        instances_3d = _discrete_channel(loaded["instances"], "cortical instances")
        _validate_raw_channels(semantic_3d, support_3d, validity_3d, instances_3d)
        source = np.stack(
            (semantic_3d, support_3d, validity_3d, instances_3d),
            axis=0,
        ).astype(np.int16, copy=False)
        source_ids = {int(value) for value in np.unique(instances_3d) if value > 0}

        data, source, roi_crop = crop_to_supervision_roi(
            data,
            source,
            support_3d == 1,
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
        _validate_processed_channels(processed)

        semantic_processed = processed[0]
        support_processed = processed[1]
        validity_processed = processed[2]
        instances_processed = processed[3]
        retention = _assert_instance_retention(source_ids, instances_processed)
        schemes = tuple(
            str(value) for value in configuration_manager.normalization_schemes
        )
        if schemes != ("CTNormalization",):
            raise RuntimeError(
                "Separator continuity requires one CTNormalization input channel; "
                f"got {schemes}"
            )
        intensity = plans_manager.foreground_intensity_properties_per_channel["0"]
        hu = (
            processed_data[0].astype(np.float32, copy=False) * float(intensity["std"])
            + float(intensity["mean"])
        )
        hu_processed = encode_hu(hu)
        valid = (validity_processed.astype(np.int64) & SEMANTIC_VALID_BIT) != 0
        separator = semantic_processed == SEPARATOR_LABEL
        normal_surface = normal_surface_per_instance(
            instances_processed,
            separator,
            valid,
            spacing_mm_zyx=configuration_manager.spacing,
            separator_exclusion_mm=2.0,
        )

        packed = np.stack(
            (
                semantic_processed,
                support_processed,
                validity_processed,
                normal_surface.astype(np.int16, copy=False),
                hu_processed,
                instances_processed,
            ),
            axis=0,
        ).astype(np.int16, copy=False)
        if packed.shape[0] != TARGET_CHANNELS:
            raise AssertionError(packed.shape)

        standard_locations = _standard_class_locations(
            self, packed[:1], plans_manager, dataset_json
        )
        separator_locations = _sample_mask_coordinates(separator, seed=3201)
        surface_locations = _sample_mask_coordinates(normal_surface, seed=3202)
        standard_locations[SEPARATOR_SAMPLING_KEY] = separator_locations
        standard_locations[SURFACE_SAMPLING_KEY] = surface_locations
        processed_properties["class_locations"] = standard_locations
        processed_properties[SEPARATOR_SAMPLING_KEY] = separator_locations
        processed_properties[SURFACE_SAMPLING_KEY] = surface_locations
        processed_properties[CONTINUITY_PLANS_KEY] = {
            "schema_version": CONTINUITY_CONTRACT_VERSION,
            "target_channels": TARGET_CHANNELS,
            "hu_code_channel": HU_CODE_CHANNEL,
            "instance_channel": INSTANCE_CHANNEL,
            "normal_surface_voxels": int(np.count_nonzero(normal_surface)),
            "separator_voxels": int(np.count_nonzero(separator)),
            "instance_retention": retention,
        }
        return processed_data, packed, processed_properties


def normal_surface_per_instance(
    instance_ids: np.ndarray,
    separator: np.ndarray,
    semantic_valid: np.ndarray,
    *,
    spacing_mm_zyx: Union[List[float], tuple[float, ...], np.ndarray],
    separator_exclusion_mm: float,
) -> np.ndarray:
    """Union boundaries extracted independently for every exclusive instance."""

    try:
        from scipy.ndimage import binary_erosion, distance_transform_edt
    except ModuleNotFoundError as error:
        raise RuntimeError("SciPy is required to construct normal surfaces") from error
    instances = np.asarray(instance_ids)
    separator_mask = np.asarray(separator, dtype=bool)
    valid_mask = np.asarray(semantic_valid, dtype=bool)
    if not (instances.shape == separator_mask.shape == valid_mask.shape):
        raise ValueError("instance, separator, and semantic-valid shapes differ")
    if not np.allclose(instances, np.rint(instances), atol=1e-4):
        raise ValueError("instance IDs must remain discrete")
    instances = np.rint(instances).astype(np.int64, copy=False)
    if np.any(instances < -1):
        raise ValueError("instance IDs may only use -1 as crop padding")

    surface = np.zeros(instances.shape, dtype=bool)
    structure = np.ones((3, 3, 3), dtype=bool)
    for instance_id in (int(value) for value in np.unique(instances) if value > 0):
        mask = instances == instance_id
        surface |= mask & ~binary_erosion(
            mask, structure=structure, border_value=0
        )
    if np.any(separator_mask):
        distance = distance_transform_edt(
            ~separator_mask,
            sampling=tuple(float(value) for value in spacing_mm_zyx),
        )
        away_from_separator = distance > float(separator_exclusion_mm)
    else:
        away_from_separator = np.ones(separator_mask.shape, dtype=bool)
    return surface & valid_mask & away_from_separator


def _resolve_sidecar_paths(
    seg_file: str, dataset_json: dict
) -> tuple[Path, dict[str, Path]]:
    semantic_path = Path(seg_file).expanduser().resolve(strict=True)
    if semantic_path.parent.name != "labelsTr":
        raise ValueError(f"Semantic labels must be under labelsTr: {semantic_path}")
    ending = str(dataset_json["file_ending"])
    if not semantic_path.name.endswith(ending):
        raise ValueError(f"Unexpected label filename {semantic_path.name!r}")
    identifier = semantic_path.name[: -len(ending)]
    dataset_root = semantic_path.parent.parent
    paths = {
        role: dataset_root / directory / f"{identifier}{ending}"
        for role, directory in _SIDECAR_DIRECTORIES.items()
    }
    missing = [f"{role}={path}" for role, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Incomplete separator continuity sidecars for {identifier}: "
            + ", ".join(missing)
        )
    return semantic_path, paths


def _validate_raw_channels(
    semantic: np.ndarray,
    support: np.ndarray,
    validity: np.ndarray,
    instances: np.ndarray,
) -> None:
    if len({semantic.shape, support.shape, validity.shape, instances.shape}) != 1:
        raise ValueError("Raw target channels do not share a grid")
    _validate_values(semantic, {0, 1, 2, 3}, "semantic")
    _validate_values(support, {0, 1}, "support")
    _validate_values(validity, set(range(8)), "validity")
    if np.any(instances < 0) or np.any(instances > np.iinfo(np.int16).max):
        raise ValueError("Raw cortical instance IDs must be non-negative int16 values")
    relation_valid = (validity.astype(np.int64) & RELATION_VALID_BIT) != 0
    if np.any((instances > 0) & ~relation_valid):
        raise ValueError(
            "Owned cortical voxels must be relation-valid; overlap/unassigned voxels use ID 0"
        )
    owned = instances > 0
    if np.any(owned & ~np.isin(semantic, (1, 2))):
        raise ValueError("Cortical instance ownership lies outside semantic cortex")
    if np.any(owned & (support != 1)):
        raise ValueError("Cortical instance ownership lies outside support")


def _validate_processed_channels(processed: np.ndarray) -> None:
    if processed.ndim != 4 or processed.shape[0] != 4:
        raise ValueError(f"Expected four resampled source channels, got {processed.shape}")
    if not np.isfinite(processed).all() or not np.allclose(
        processed, np.rint(processed), atol=1e-4
    ):
        raise ValueError("Nearest-neighbour target resampling produced non-discrete values")
    value = np.rint(processed).astype(np.int64, copy=False)
    _validate_values(value[0], {-1, 0, 1, 2, 3}, "semantic")
    _validate_values(value[1], {-1, 0, 1}, "support")
    _validate_values(value[2], {-1, *range(8)}, "validity")
    if np.any(value[3] < -1) or np.any(value[3] > np.iinfo(np.int16).max):
        raise ValueError("Instance ID lies outside the discrete int16 contract")
    relation_valid = (value[2] & RELATION_VALID_BIT) != 0
    if np.any((value[3] > 0) & ~relation_valid):
        raise ValueError("Resampled ownership is not relation-valid")


def _assert_instance_retention(
    source_ids: set[int], resampled_instances: np.ndarray
) -> dict[str, Any]:
    retained = {int(value) for value in np.unique(resampled_instances) if value > 0}
    missing = sorted(source_ids - retained)
    unexpected = sorted(retained - source_ids)
    if missing:
        raise ValueError(f"0.5-mm resampling removed cortical instance IDs {missing}")
    if unexpected:
        raise ValueError(f"Resampling introduced cortical instance IDs {unexpected}")
    return {
        "source_instance_count": len(source_ids),
        "resampled_instance_count": len(retained),
        "missing_instance_ids": missing,
        "status": "PASS",
    }


def _discrete_channel(value: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim == 4 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 3 or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite 3-D discrete image; got {array.shape}")
    rounded = np.rint(array)
    if not np.allclose(array, rounded, atol=1e-4):
        raise ValueError(f"{name} contains non-integer values")
    return rounded.astype(np.int64, copy=False)


def _validate_values(value: np.ndarray, allowed: set[int], name: str) -> None:
    invalid = set(int(item) for item in np.unique(value)).difference(allowed)
    if invalid:
        raise ValueError(f"{name} contains unsupported values {sorted(invalid)}")


def _standard_class_locations(
    preprocessor: DefaultPreprocessor,
    semantic: np.ndarray,
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
        semantic, collect, verbose=preprocessor.verbose
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


def _assert_exact_grid(
    reference: np.ndarray,
    reference_properties: dict,
    candidate: np.ndarray,
    candidate_properties: dict,
    *,
    role: str,
    path: Path,
) -> None:
    if tuple(reference.shape[1:]) != tuple(candidate.shape[1:]):
        raise ValueError(f"{role} shape differs from CT: {path}")
    _assert_geometry(reference_properties.get("spacing"), candidate_properties.get("spacing"), role, "spacing", path)
    for metadata_key, fields in (
        ("nibabel_stuff", ("original_affine", "reoriented_affine")),
        ("sitk_stuff", ("spacing", "origin", "direction")),
    ):
        reference_metadata = reference_properties.get(metadata_key)
        candidate_metadata = candidate_properties.get(metadata_key)
        if isinstance(reference_metadata, dict) and isinstance(candidate_metadata, dict):
            for field in fields:
                _assert_geometry(
                    reference_metadata.get(field),
                    candidate_metadata.get(field),
                    role,
                    field,
                    path,
                )
            return
    raise ValueError(f"Cannot prove exact CT/{role} grid equality: {path}")


def _assert_geometry(reference: Any, candidate: Any, role: str, field: str, path: Path) -> None:
    if reference is None or candidate is None:
        raise ValueError(f"Missing {field} while checking {role}: {path}")
    first = np.asarray(reference, dtype=np.float64)
    second = np.asarray(candidate, dtype=np.float64)
    if first.shape != second.shape or not np.allclose(first, second, rtol=0.0, atol=1e-6):
        raise ValueError(f"{role} {field} differs from CT grid: {path}")
