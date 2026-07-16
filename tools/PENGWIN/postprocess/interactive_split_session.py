"""UI-independent state for interactive fragment-instance splitting.

The instance map is the authoritative editable segmentation. ABBC is retained
only as hidden guidance for partitioning a selected instance.
"""

from __future__ import annotations

import dataclasses
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Sequence

import numpy as np
import SimpleITK as sitk

from tools.PENGWIN.postprocess.cortical_anchored_split import (
    SplitConfig,
    split_instance_core_first,
)


PromptKind = Literal["source", "sink"]


@dataclasses.dataclass(frozen=True)
class ImageGrid:
    """SimpleITK grid metadata without retaining another image-sized buffer."""

    size_xyz: tuple[int, int, int]
    spacing_xyz: tuple[float, float, float]
    origin_xyz: tuple[float, float, float]
    direction_xyz: tuple[float, ...]

    @property
    def shape_zyx(self) -> tuple[int, int, int]:
        return tuple(reversed(self.size_xyz))

    @property
    def spacing_zyx(self) -> tuple[float, float, float]:
        return tuple(reversed(self.spacing_xyz))

    @classmethod
    def from_image(cls, image: sitk.Image) -> "ImageGrid":
        if image.GetDimension() != 3:
            raise ValueError(f"expected a 3D image, got {image.GetDimension()}D")
        return cls(
            size_xyz=tuple(int(v) for v in image.GetSize()),
            spacing_xyz=tuple(float(v) for v in image.GetSpacing()),
            origin_xyz=tuple(float(v) for v in image.GetOrigin()),
            direction_xyz=tuple(float(v) for v in image.GetDirection()),
        )

    def assert_matches(self, other: "ImageGrid", name: str) -> None:
        if self.size_xyz != other.size_xyz:
            raise ValueError(
                f"{name} size {other.size_xyz} does not match CT size {self.size_xyz}"
            )
        for field in ("spacing_xyz", "origin_xyz", "direction_xyz"):
            expected = np.asarray(getattr(self, field), dtype=float)
            actual = np.asarray(getattr(other, field), dtype=float)
            if not np.allclose(expected, actual, rtol=0.0, atol=1e-5):
                raise ValueError(f"{name} {field} does not match the CT grid")

    def physical_point_lps(self, index_zyx: Sequence[int | float]) -> tuple[float, ...]:
        index = np.asarray(index_zyx, dtype=float)
        if index.shape != (3,):
            raise ValueError("index_zyx must contain exactly three coordinates")
        index_xyz = index[::-1]
        direction = np.asarray(self.direction_xyz, dtype=float).reshape(3, 3)
        point_xyz = (
            np.asarray(self.origin_xyz, dtype=float)
            + direction @ (index_xyz * np.asarray(self.spacing_xyz, dtype=float))
        )
        return tuple(float(v) for v in point_xyz)

    def affine_zyx(self) -> np.ndarray:
        """Return a napari affine mapping array ZYX indices to physical ZYX."""
        reverse = np.eye(3, dtype=float)[::-1]
        direction = np.asarray(self.direction_xyz, dtype=float).reshape(3, 3)
        spacing = np.diag(np.asarray(self.spacing_xyz, dtype=float))
        affine = np.eye(4, dtype=float)
        affine[:3, :3] = reverse @ direction @ spacing @ reverse
        affine[:3, 3] = reverse @ np.asarray(self.origin_xyz, dtype=float)
        return affine

    def image_from_array(self, array: np.ndarray) -> sitk.Image:
        if tuple(array.shape) != self.shape_zyx:
            raise ValueError(
                f"array shape {array.shape} does not match image grid {self.shape_zyx}"
            )
        image = sitk.GetImageFromArray(array)
        image.SetSpacing(self.spacing_xyz)
        image.SetOrigin(self.origin_xyz)
        image.SetDirection(self.direction_xyz)
        return image


@dataclasses.dataclass(frozen=True)
class SplitPreview:
    base_revision: int
    selected_instance: int
    new_instance: int
    instances_after: np.ndarray
    cut_mask: np.ndarray
    diagnostics: dict
    source_point_zyx: tuple[int, int, int]
    sink_point_zyx: tuple[int, int, int]
    prompt_radius_mm: float


@dataclasses.dataclass(frozen=True)
class UndoRecord:
    selected_instance: int
    new_instance: int


@dataclasses.dataclass(frozen=True)
class RedoRecord:
    selected_instance: int
    new_instance: int
    lower_zyx: tuple[int, int, int]
    shape_zyx: tuple[int, int, int]
    packed_new_mask: np.ndarray


def _read_image(path: Path, name: str) -> tuple[np.ndarray, ImageGrid]:
    if not path.is_file():
        raise FileNotFoundError(f"{name} image does not exist: {path}")
    image = sitk.ReadImage(str(path))
    grid = ImageGrid.from_image(image)
    array = sitk.GetArrayFromImage(image)
    return array, grid


def _as_label_array(array: np.ndarray, name: str, maximum: int | None = None) -> np.ndarray:
    if not np.issubdtype(array.dtype, np.integer):
        if not np.all(np.isfinite(array)) or not np.allclose(array, np.rint(array)):
            raise ValueError(f"{name} must contain integer labels")
        array = np.rint(array)
    if array.size and int(array.min()) < 0:
        raise ValueError(f"{name} must not contain negative labels")
    if maximum is not None and array.size and int(array.max()) > maximum:
        raise ValueError(f"{name} contains labels above {maximum}")
    largest = int(array.max()) if array.size else 0
    dtype = np.uint16 if largest <= np.iinfo(np.uint16).max else np.uint32
    return np.asarray(array, dtype=dtype)


def spacing_aware_prompt_mask(
    shape: Sequence[int],
    point_zyx: Sequence[int | float],
    spacing_zyx: Sequence[float],
    radius_mm: float,
    allowed_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Create a physical sphere around one prompt without a full-volume EDT."""
    shape_array = np.asarray(shape, dtype=int)
    point = np.rint(np.asarray(point_zyx, dtype=float)).astype(int)
    spacing = np.asarray(spacing_zyx, dtype=float)
    if shape_array.shape != (3,) or point.shape != (3,) or spacing.shape != (3,):
        raise ValueError("shape, point_zyx and spacing_zyx must be three-dimensional")
    if np.any(spacing <= 0):
        raise ValueError("spacing values must be positive")
    if radius_mm < 0:
        raise ValueError("prompt radius must be non-negative")
    if np.any(point < 0) or np.any(point >= shape_array):
        raise ValueError(f"prompt point {tuple(point)} lies outside the image")
    if allowed_mask is not None and tuple(allowed_mask.shape) != tuple(shape_array):
        raise ValueError("allowed_mask must have the requested output shape")

    radii = np.ceil(radius_mm / spacing).astype(int)
    lower = np.maximum(point - radii, 0)
    upper = np.minimum(point + radii + 1, shape_array)
    local_shape = upper - lower
    coordinates = np.ogrid[tuple(slice(0, int(v)) for v in local_shape)]
    squared_distance = np.zeros(tuple(local_shape), dtype=np.float32)
    for axis, coordinate in enumerate(coordinates):
        offset = (coordinate + lower[axis] - point[axis]) * spacing[axis]
        squared_distance += np.asarray(offset * offset, dtype=np.float32)
    local_sphere = squared_distance <= float(radius_mm) ** 2 + 1e-6

    result = np.zeros(tuple(shape_array), dtype=bool)
    crop = tuple(slice(int(lo), int(hi)) for lo, hi in zip(lower, upper))
    if allowed_mask is not None:
        local_sphere &= allowed_mask[crop]
    result[crop] = local_sphere
    return result


class InteractiveSplitSession:
    """Mutable interactive state with revision-checked split commits."""

    def __init__(
        self,
        ct: np.ndarray,
        abbc: np.ndarray,
        instances: np.ndarray,
        grid: ImageGrid,
        *,
        gt: np.ndarray | None = None,
        prob_label_3: np.ndarray | None = None,
        cfg: SplitConfig | None = None,
        source_paths: dict[str, str] | None = None,
    ) -> None:
        expected_shape = grid.shape_zyx
        for name, array in (("CT", ct), ("ABBC", abbc), ("instances", instances)):
            if tuple(array.shape) != expected_shape:
                raise ValueError(
                    f"{name} shape {array.shape} does not match image grid {expected_shape}"
                )
        if gt is not None and tuple(gt.shape) != expected_shape:
            raise ValueError("GT shape does not match the image grid")
        if prob_label_3 is not None and tuple(prob_label_3.shape) != expected_shape:
            raise ValueError("Label-3 probability shape does not match the image grid")

        self.ct = np.asarray(ct)
        self.abbc = np.asarray(abbc, dtype=np.uint8)
        self.instances = _as_label_array(instances, "instances")
        self.gt = None if gt is None else _as_label_array(gt, "GT")
        self.prob_label_3 = (
            None if prob_label_3 is None else np.asarray(prob_label_3, dtype=np.float32)
        )
        self.grid = grid
        self.cfg = cfg if cfg is not None else SplitConfig()
        self.source_paths = dict(source_paths or {})
        self.selected_instance: int | None = None
        self.source_point_zyx: tuple[int, int, int] | None = None
        self.sink_point_zyx: tuple[int, int, int] | None = None
        self.revision = 0
        self.history: list[UndoRecord] = []
        self.redo_stack: list[RedoRecord] = []
        self.interactions: list[dict] = []
        self.last_cut_mask: np.ndarray | None = None

    @classmethod
    def from_files(
        cls,
        ct_path: str | Path,
        abbc_path: str | Path,
        *,
        instances_path: str | Path,
        gt_path: str | Path | None = None,
        prob_label_3_path: str | Path | None = None,
        cfg: SplitConfig | None = None,
    ) -> "InteractiveSplitSession":
        ct_path = Path(ct_path).expanduser().resolve()
        abbc_path = Path(abbc_path).expanduser().resolve()
        ct, grid = _read_image(ct_path, "CT")
        abbc_raw, abbc_grid = _read_image(abbc_path, "ABBC")
        grid.assert_matches(abbc_grid, "ABBC")
        abbc = _as_label_array(abbc_raw, "ABBC", maximum=3).astype(np.uint8)

        resolved = Path(instances_path).expanduser().resolve()
        raw_instances, instances_grid = _read_image(resolved, "instances")
        grid.assert_matches(instances_grid, "instances")
        instances = _as_label_array(raw_instances, "instances")
        paths = {
            "ct": str(ct_path),
            "abbc": str(abbc_path),
            "instances": str(resolved),
        }

        gt = None
        if gt_path is not None:
            resolved = Path(gt_path).expanduser().resolve()
            raw_gt, gt_grid = _read_image(resolved, "GT")
            grid.assert_matches(gt_grid, "GT")
            gt = _as_label_array(raw_gt, "GT")
            paths["gt"] = str(resolved)

        prob3 = None
        if prob_label_3_path is not None:
            resolved = Path(prob_label_3_path).expanduser().resolve()
            raw_prob3, prob3_grid = _read_image(resolved, "Label-3 probability")
            grid.assert_matches(prob3_grid, "Label-3 probability")
            prob3 = np.asarray(raw_prob3, dtype=np.float32)
            if not np.all(np.isfinite(prob3)):
                raise ValueError("Label-3 probability contains non-finite values")
            paths["prob_label_3"] = str(resolved)

        return cls(
            ct,
            abbc,
            instances,
            grid,
            gt=gt,
            prob_label_3=prob3,
            cfg=cfg,
            source_paths=paths,
        )

    def _voxel_index(self, point_zyx: Sequence[int | float]) -> tuple[int, int, int]:
        point = np.rint(np.asarray(point_zyx, dtype=float)).astype(int)
        if point.shape != (3,):
            raise ValueError("a point must contain Z, Y and X coordinates")
        shape = np.asarray(self.instances.shape)
        if np.any(point < 0) or np.any(point >= shape):
            raise ValueError(f"point {tuple(point)} lies outside the image")
        return tuple(int(v) for v in point)

    def select_instance_at(self, point_zyx: Sequence[int | float]) -> int:
        point = self._voxel_index(point_zyx)
        return self.select_instance_id(int(self.instances[point]))

    def select_instance_id(self, instance_id: int) -> int:
        instance_id = int(instance_id)
        if instance_id <= 0 or not np.any(self.instances == instance_id):
            raise ValueError(f"instance {instance_id} is not present")
        if self.selected_instance != instance_id:
            self.source_point_zyx = None
            self.sink_point_zyx = None
        self.selected_instance = instance_id
        return instance_id

    def set_prompt(
        self, kind: PromptKind, point_zyx: Sequence[int | float]
    ) -> tuple[int, int, int]:
        if kind not in ("source", "sink"):
            raise ValueError("prompt kind must be 'source' or 'sink'")
        if self.selected_instance is None:
            raise ValueError("select an instance before placing prompts")
        point = self._voxel_index(point_zyx)
        point_instance = int(self.instances[point])
        if point_instance != self.selected_instance:
            raise ValueError(
                f"{kind} point is in instance {point_instance}, not selected instance "
                f"{self.selected_instance}"
            )
        if kind == "source":
            self.source_point_zyx = point
        else:
            self.sink_point_zyx = point
        return point

    def clear_prompts(self) -> None:
        self.source_point_zyx = None
        self.sink_point_zyx = None

    def clear_prompt(self, kind: PromptKind) -> None:
        if kind == "source":
            self.source_point_zyx = None
        elif kind == "sink":
            self.sink_point_zyx = None
        else:
            raise ValueError("prompt kind must be 'source' or 'sink'")

    def clear_selection(self) -> None:
        self.selected_instance = None
        self.clear_prompts()

    def prompt_mask(self, kind: PromptKind, radius_mm: float) -> np.ndarray:
        if self.selected_instance is None:
            raise ValueError("select an instance before creating prompt masks")
        point = self.source_point_zyx if kind == "source" else self.sink_point_zyx
        if point is None:
            raise ValueError(f"{kind} prompt has not been placed")
        return spacing_aware_prompt_mask(
            self.instances.shape,
            point,
            self.grid.spacing_zyx,
            radius_mm,
            allowed_mask=self.instances == self.selected_instance,
        )

    def compute_split(self, prompt_radius_mm: float = 2.0) -> SplitPreview:
        if self.selected_instance is None:
            raise ValueError("select an instance before running the split")
        if self.source_point_zyx is None or self.sink_point_zyx is None:
            raise ValueError("place both source and sink prompts before running the split")
        source_mask = self.prompt_mask("source", prompt_radius_mm)
        sink_mask = self.prompt_mask("sink", prompt_radius_mm)
        if np.any(source_mask & sink_mask):
            raise ValueError("source and sink prompt spheres overlap")

        selected_instance = self.selected_instance
        refinement_mode = any(
            selected_instance in (record.selected_instance, record.new_instance)
            for record in self.history
        )
        split_cfg = dataclasses.replace(
            self.cfg,
            bottleneck_use_core_growth_seeds=not refinement_mode,
        )
        instances_after, cut_mask, diagnostics = split_instance_core_first(
            abbc_pred=self.abbc,
            instances=self.instances,
            instance_id=selected_instance,
            source_interaction_mask=source_mask,
            sink_interaction_mask=sink_mask,
            prob_label_3=self.prob_label_3,
            image=self.ct,
            spacing_zyx=self.grid.spacing_zyx,
            cfg=split_cfg,
        )
        diagnostics["interactive_refinement_mode"] = refinement_mode
        selected_mask = self.instances == selected_instance
        if not np.array_equal(instances_after != 0, self.instances != 0):
            raise RuntimeError(
                "split changed the input instance support; ABBC must not add or remove voxels"
            )
        if not np.array_equal(
            instances_after[~selected_mask], self.instances[~selected_mask]
        ):
            raise RuntimeError("split changed an instance that was not selected")
        return SplitPreview(
            base_revision=self.revision,
            selected_instance=selected_instance,
            new_instance=int(diagnostics["new_instance"]),
            instances_after=instances_after,
            cut_mask=cut_mask,
            diagnostics=diagnostics,
            source_point_zyx=self.source_point_zyx,
            sink_point_zyx=self.sink_point_zyx,
            prompt_radius_mm=float(prompt_radius_mm),
        )

    def commit_split(self, preview: SplitPreview) -> dict:
        if preview.base_revision != self.revision:
            raise RuntimeError("the segmentation changed while the split was running")
        self.instances = preview.instances_after
        self.last_cut_mask = preview.cut_mask
        self.history.append(
            UndoRecord(preview.selected_instance, preview.new_instance)
        )
        self.redo_stack.clear()
        self.revision += 1
        self.selected_instance = preview.selected_instance
        self.clear_prompts()
        event = {
            "action": "split",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "revision": self.revision,
            "selected_instance": preview.selected_instance,
            "new_instance": preview.new_instance,
            "source_point_zyx": list(preview.source_point_zyx),
            "sink_point_zyx": list(preview.sink_point_zyx),
            "source_point_lps_xyz_mm": list(
                self.grid.physical_point_lps(preview.source_point_zyx)
            ),
            "sink_point_lps_xyz_mm": list(
                self.grid.physical_point_lps(preview.sink_point_zyx)
            ),
            "prompt_radius_mm": preview.prompt_radius_mm,
            "diagnostics": preview.diagnostics,
        }
        self.interactions.append(event)
        return event

    def record_failed_split(self, error: Exception | str, prompt_radius_mm: float) -> dict:
        event = {
            "action": "split_failed",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "revision": self.revision,
            "selected_instance": self.selected_instance,
            "source_point_zyx": (
                None if self.source_point_zyx is None else list(self.source_point_zyx)
            ),
            "sink_point_zyx": (
                None if self.sink_point_zyx is None else list(self.sink_point_zyx)
            ),
            "prompt_radius_mm": float(prompt_radius_mm),
            "error": str(error),
        }
        self.interactions.append(event)
        return event

    def undo_last_split(self) -> UndoRecord:
        if not self.history:
            raise ValueError("there is no committed split to undo")
        record = self.history[-1]
        new_mask = self.instances == record.new_instance
        if not np.any(new_mask):
            raise RuntimeError(
                f"cannot undo: instance {record.new_instance} is no longer present"
            )
        coordinates = np.argwhere(new_mask)
        lower = coordinates.min(axis=0)
        upper = coordinates.max(axis=0) + 1
        crop = tuple(slice(int(lo), int(hi)) for lo, hi in zip(lower, upper))
        local_mask = np.ascontiguousarray(new_mask[crop])
        self.redo_stack.append(
            RedoRecord(
                selected_instance=record.selected_instance,
                new_instance=record.new_instance,
                lower_zyx=tuple(int(value) for value in lower),
                shape_zyx=tuple(int(value) for value in local_mask.shape),
                packed_new_mask=np.packbits(local_mask, axis=None),
            )
        )
        self.history.pop()
        self.instances[new_mask] = record.selected_instance
        self.last_cut_mask = None
        self.revision += 1
        self.selected_instance = record.selected_instance
        self.clear_prompts()
        self.interactions.append(
            {
                "action": "undo",
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "revision": self.revision,
                "selected_instance": record.selected_instance,
                "removed_instance": record.new_instance,
            }
        )
        return record

    def redo_last_split(self) -> RedoRecord:
        if not self.redo_stack:
            raise ValueError("there is no undone split to redo")
        record = self.redo_stack[-1]
        if np.any(self.instances == record.new_instance):
            raise RuntimeError(
                f"cannot redo: instance {record.new_instance} is already present"
            )
        lower = np.asarray(record.lower_zyx, dtype=int)
        shape = np.asarray(record.shape_zyx, dtype=int)
        upper = lower + shape
        crop = tuple(slice(int(lo), int(hi)) for lo, hi in zip(lower, upper))
        voxel_count = int(np.prod(shape))
        local_mask = np.unpackbits(
            record.packed_new_mask,
            count=voxel_count,
        ).reshape(record.shape_zyx).astype(bool, copy=False)
        local_instances = self.instances[crop]
        if not np.all(local_instances[local_mask] == record.selected_instance):
            raise RuntimeError(
                "cannot redo: voxels from the undone split were modified"
            )

        local_instances[local_mask] = record.new_instance
        self.redo_stack.pop()
        self.history.append(
            UndoRecord(record.selected_instance, record.new_instance)
        )
        self.last_cut_mask = None
        self.revision += 1
        self.selected_instance = record.selected_instance
        self.clear_prompts()
        self.interactions.append(
            {
                "action": "redo",
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "revision": self.revision,
                "selected_instance": record.selected_instance,
                "restored_instance": record.new_instance,
            }
        )
        return record

    def save_instances(self, path: str | Path) -> Path:
        path = Path(path).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        sitk.WriteImage(self.grid.image_from_array(self.instances), str(path), True)
        return path

    def save_interactions(self, path: str | Path) -> Path:
        path = Path(path).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "source_paths": self.source_paths,
            "grid": dataclasses.asdict(self.grid),
            "split_config": dataclasses.asdict(self.cfg),
            "revision": self.revision,
            "interactions": self.interactions,
        }
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
        return path
