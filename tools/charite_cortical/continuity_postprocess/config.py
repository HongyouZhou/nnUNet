from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any, Mapping, Optional


@dataclasses.dataclass(frozen=True)
class ContinuityConfig:
    """Frozen thresholds for cortical grouping and conservative split acceptance."""

    cortex_low_threshold: float = 0.30
    cortex_high_threshold: float = 0.60
    attractive_probability_threshold: float = 0.70
    repulsive_probability_threshold: float = 0.30
    separator_probability_threshold: float = 0.60
    min_high_confidence_volume_mm3: float = 0.0
    min_repulsive_edges_per_cluster: int = 1
    min_normalized_energy_gain: float = 0.0
    min_piece_volume_mm3: float = 0.0
    max_clusters_per_instance: int = 64
    # Python edge objects are intentionally capped conservatively in the MVP.
    # A future compact/region graph may raise these defaults.
    max_graph_nodes: Optional[int] = 500_000
    max_graph_edges: Optional[int] = 2_000_000
    distance_tie_tolerance: float = 1e-9

    def __post_init__(self) -> None:
        probability_fields = (
            "cortex_low_threshold",
            "cortex_high_threshold",
            "attractive_probability_threshold",
            "repulsive_probability_threshold",
            "separator_probability_threshold",
        )
        for name in probability_fields:
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {value}")
        if self.cortex_low_threshold > self.cortex_high_threshold:
            raise ValueError(
                "cortex_low_threshold must not exceed cortex_high_threshold"
            )
        if (
            self.repulsive_probability_threshold
            >= self.attractive_probability_threshold
        ):
            raise ValueError(
                "repulsive_probability_threshold must be below "
                "attractive_probability_threshold"
            )
        if self.min_high_confidence_volume_mm3 < 0:
            raise ValueError("min_high_confidence_volume_mm3 must be non-negative")
        if self.min_repulsive_edges_per_cluster < 0:
            raise ValueError("min_repulsive_edges_per_cluster must be non-negative")
        if self.min_normalized_energy_gain < 0:
            raise ValueError("min_normalized_energy_gain must be non-negative")
        if self.min_piece_volume_mm3 < 0:
            raise ValueError("min_piece_volume_mm3 must be non-negative")
        if self.max_clusters_per_instance < 2:
            raise ValueError("max_clusters_per_instance must be at least two")
        for name in ("max_graph_nodes", "max_graph_edges"):
            value = getattr(self, name)
            if value is not None and value < 1:
                raise ValueError(f"{name} must be positive or None")
        if self.distance_tie_tolerance < 0:
            raise ValueError("distance_tie_tolerance must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "ContinuityConfig":
        unknown = set(values).difference(field.name for field in dataclasses.fields(cls))
        if unknown:
            raise ValueError(f"unknown continuity config fields: {sorted(unknown)}")
        return cls(**dict(values))

    def save_json(self, path: str | Path) -> None:
        destination = Path(path)
        destination.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load_json(cls, path: str | Path) -> "ContinuityConfig":
        values = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(values, dict):
            raise ValueError("continuity config JSON must contain an object")
        return cls.from_dict(values)
