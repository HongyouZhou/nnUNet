"""Specialized flat-head predictor for cortical continuity.

This module deliberately does not modify the generic nnU-Net predictor.
Directional affinities are accumulated as logits on the preprocessed grid and
activated only after sliding-window and fold averaging.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from nnunetv2.training.cortical_continuity.schema import (
    CorticalContinuityHeadSchema,
    schema_from_plans,
)

try:
    import torch

    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
except ModuleNotFoundError as exc:
    torch = None
    nnUNetPredictor = object
    _PREDICTOR_IMPORT_ERROR = exc
else:
    _PREDICTOR_IMPORT_ERROR = None


if torch is not None:

    class _OutputCountLabelManagerProxy:
        def __init__(self, wrapped: Any, output_channels: int) -> None:
            self._wrapped = wrapped
            self.num_segmentation_heads = int(output_channels)

        def __getattr__(self, name: str) -> Any:
            return getattr(self._wrapped, name)


    class CorticalContinuityPredictor(nnUNetPredictor):
        def __init__(
            self,
            schema: CorticalContinuityHeadSchema,
            *,
            use_mirroring: bool = False,
            **kwargs: Any,
        ) -> None:
            if use_mirroring:
                raise ValueError(
                    "Mirror TTA is disabled for directional cortical affinity channels; "
                    "channel and source-coordinate remapping is not implemented"
                )
            self.continuity_schema = schema
            super().__init__(use_mirroring=False, **kwargs)

        def initialize_from_trained_model_folder(
            self,
            model_training_output_dir: str,
            use_folds,
            checkpoint_name: str = "checkpoint_final.pth",
        ):
            from batchgenerators.utilities.file_and_folder_operations import join, load_json

            model_schema = schema_from_plans(load_json(join(model_training_output_dir, "plans.json")))
            if model_schema != self.continuity_schema:
                raise ValueError("Predictor schema does not match the schema frozen in the model plans")
            result = super().initialize_from_trained_model_folder(
                model_training_output_dir,
                use_folds,
                checkpoint_name,
            )
            if self.allowed_mirroring_axes is not None:
                raise ValueError(
                    "Directional-affinity checkpoint declares inference mirroring axes; "
                    "refusing unsafe mirror TTA"
                )
            self.use_mirroring = False
            return result

        def manual_initialization(
            self,
            network,
            plans_manager,
            configuration_manager,
            parameters,
            dataset_json,
            trainer_name,
            inference_allowed_mirroring_axes,
        ):
            if inference_allowed_mirroring_axes is not None:
                raise ValueError(
                    "Directional-affinity inference requires inference_allowed_mirroring_axes=None"
                )
            model_schema = schema_from_plans(plans_manager.plans)
            if model_schema != self.continuity_schema:
                raise ValueError("Predictor schema does not match the schema frozen in the supplied plans")
            result = super().manual_initialization(
                network,
                plans_manager,
                configuration_manager,
                parameters,
                dataset_json,
                trainer_name,
                None,
            )
            self.use_mirroring = False
            return result

        def _internal_maybe_mirror_and_predict(self, x: torch.Tensor) -> torch.Tensor:
            if self.use_mirroring or self.allowed_mirroring_axes is not None:
                raise RuntimeError(
                    "Mirror TTA cannot be used with directional affinity logits in schema v1"
                )
            prediction = self.network(x)
            if isinstance(prediction, (tuple, list)):
                if len(prediction) != 1:
                    raise ValueError(
                        "Cortical continuity inference expects one full-resolution flat output"
                    )
                prediction = prediction[0]
            if not isinstance(prediction, torch.Tensor):
                raise TypeError(
                    "Cortical continuity network must return a flat tensor, "
                    f"got {type(prediction).__name__}"
                )
            if prediction.shape[1] != self.continuity_schema.total_channels:
                raise ValueError(
                    f"Network returned {prediction.shape[1]} channels; "
                    f"schema requires {self.continuity_schema.total_channels}"
                )
            return prediction

        def _internal_predict_sliding_window_return_logits(
            self,
            data: torch.Tensor,
            slicers,
            do_on_device: bool = True,
        ):
            original_label_manager = self.label_manager
            self.label_manager = _OutputCountLabelManagerProxy(
                original_label_manager,
                self.continuity_schema.total_channels,
            )
            try:
                return super()._internal_predict_sliding_window_return_logits(
                    data,
                    slicers,
                    do_on_device,
                )
            finally:
                self.label_manager = original_label_manager

        def activate_flat_logits(
            self,
            flat_logits: torch.Tensor,
            *,
            channel_axis: int = 0,
        ) -> dict[str, torch.Tensor]:
            return self.continuity_schema.activate(flat_logits, channel_axis)

        @torch.inference_mode()
        def predict_evidence_from_preprocessed_data(
            self,
            data: torch.Tensor,
        ) -> dict[str, torch.Tensor]:
            flat_logits = self.predict_logits_from_preprocessed_data(data)
            return self.activate_flat_logits(flat_logits, channel_axis=0)

else:

    class CorticalContinuityPredictor:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError(
                "CorticalContinuityPredictor requires a complete PyTorch nnU-Net environment"
            ) from _PREDICTOR_IMPORT_ERROR


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_array(path: Path, key: str) -> np.ndarray:
    source = path.expanduser().resolve(strict=True)
    if source.suffix == ".npy":
        return np.load(source, allow_pickle=False)
    with np.load(source, allow_pickle=False) as payload:
        if key not in payload.files:
            raise ValueError(
                f"{source} does not contain {key!r}; keys={sorted(payload.files)}"
            )
        return np.asarray(payload[key])


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Export fold-averaged C+A evidence on an nnU-Net preprocessed grid"
        )
    )
    parser.add_argument("--model-folder", type=Path, required=True)
    parser.add_argument("--folds", nargs="+", default=["0", "1", "2", "3", "4"])
    parser.add_argument("--checkpoint", default="checkpoint_final.pth")
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="4-D [C,Z,Y,X] preprocessed .npy/.npz array",
    )
    parser.add_argument("--input-key", default="data")
    parser.add_argument(
        "--provisional",
        type=Path,
        help="optional same-grid provisional instance .npy/.npz",
    )
    parser.add_argument("--provisional-key", default="provisional_instances")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--tile-step-size", type=float, default=0.5)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--no-tqdm", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    if torch is None:
        raise ImportError(
            "cortical continuity prediction requires the HPC PyTorch nnU-Net environment"
        ) from _PREDICTOR_IMPORT_ERROR
    args = _parser().parse_args(argv)
    model_folder = args.model_folder.expanduser().resolve(strict=True)
    output = args.output.expanduser().resolve()
    if output.suffix != ".npz":
        raise ValueError("--output must have an explicit .npz suffix")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    diagnostics = output.with_name(output.stem + ".provenance.json")
    if diagnostics.exists():
        raise FileExistsError(f"refusing to overwrite {diagnostics}")

    plans_path = model_folder / "plans.json"
    plans = json.loads(plans_path.read_text(encoding="utf-8"))
    schema = schema_from_plans(plans)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    predictor = CorticalContinuityPredictor(
        schema,
        tile_step_size=args.tile_step_size,
        use_gaussian=True,
        use_mirroring=False,
        perform_everything_on_device=device.type == "cuda",
        device=device,
        verbose=args.verbose,
        verbose_preprocessing=False,
        allow_tqdm=not args.no_tqdm,
    )
    predictor.initialize_from_trained_model_folder(
        str(model_folder),
        args.folds,
        args.checkpoint,
    )
    image = np.asarray(
        _load_array(args.input, args.input_key),
        dtype=np.float32,
    )
    if image.ndim != 4:
        raise ValueError(
            f"preprocessed input must be [C,Z,Y,X], got {image.shape}"
        )
    with torch.inference_mode():
        flat_logits = predictor.predict_logits_from_preprocessed_data(
            torch.from_numpy(image)
        )
        evidence = predictor.activate_flat_logits(
            flat_logits,
            channel_axis=0,
        )
    cortex_probability = evidence["cortex"][0].detach().cpu().numpy()
    affinity_probability = evidence["affinity"].detach().cpu().numpy()
    payload: dict[str, np.ndarray] = {
        "cortex_probability": cortex_probability.astype(np.float32, copy=False),
        "affinity": affinity_probability.astype(np.float32, copy=False),
        "affinity_offsets_zyx": np.asarray(
            [
                offset.voxel_offset_zyx
                for offset in schema.affinity_offsets
            ],
            dtype=np.int16,
        ),
        "spacing_zyx": np.asarray(schema.spacing_mm_zyx, dtype=np.float64),
        "affinity_kind": np.asarray("probabilities"),
    }
    if args.provisional is not None:
        provisional = _load_array(
            args.provisional,
            args.provisional_key,
        )
        if provisional.shape != cortex_probability.shape:
            raise ValueError(
                "provisional instance grid does not match model evidence"
            )
        if not np.issubdtype(provisional.dtype, np.integer):
            raise ValueError("provisional instances must use an integer dtype")
        payload["provisional_instances"] = provisional
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **payload)

    checkpoint_records = []
    for fold in args.folds:
        checkpoint_path = (
            model_folder / f"fold_{fold}" / args.checkpoint
        ).resolve(strict=True)
        checkpoint_records.append(
            {
                "fold": str(fold),
                "path": str(checkpoint_path),
                "sha256": _sha256(checkpoint_path),
            }
        )
    diagnostics.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "input": {
                    "path": str(args.input.expanduser().resolve(strict=True)),
                    "sha256": _sha256(
                        args.input.expanduser().resolve(strict=True)
                    ),
                },
                "model_folder": str(model_folder),
                "plans": {
                    "path": str(plans_path),
                    "sha256": _sha256(plans_path),
                },
                "checkpoints": checkpoint_records,
                "head_schema": schema.to_dict(),
                "mirror_tta": False,
                "fold_and_tile_aggregation": "mean_logits_then_head_activation",
                "output": str(output),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "output": str(output),
                "provenance": str(diagnostics),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
