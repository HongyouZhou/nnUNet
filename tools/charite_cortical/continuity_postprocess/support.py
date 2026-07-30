from __future__ import annotations

from collections import deque

import numpy as np


def _fallback_binary_propagation(seed: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Six-connected binary propagation used when SciPy is unavailable."""
    reached = np.asarray(seed, dtype=bool).copy()
    allowed = np.asarray(mask, dtype=bool)
    queue = deque(tuple(int(v) for v in p) for p in np.argwhere(reached))
    shape = reached.shape
    while queue:
        point = queue.popleft()
        for axis in range(3):
            for delta in (-1, 1):
                neighbour = list(point)
                neighbour[axis] += delta
                if neighbour[axis] < 0 or neighbour[axis] >= shape[axis]:
                    continue
                neighbour_tuple = tuple(neighbour)
                if allowed[neighbour_tuple] and not reached[neighbour_tuple]:
                    reached[neighbour_tuple] = True
                    queue.append(neighbour_tuple)
    return reached


def hysteresis_cortex_support(
    cortex_probability: np.ndarray,
    instance_mask: np.ndarray,
    low_threshold: float,
    high_threshold: float,
) -> np.ndarray:
    """Keep low-threshold cortex connected to a high-confidence cortical seed."""
    probability = np.asarray(cortex_probability)
    mask = np.asarray(instance_mask, dtype=bool)
    if probability.ndim != 3 or probability.shape != mask.shape:
        raise ValueError(
            "cortex_probability and instance_mask must have the same 3D shape"
        )
    if not np.all(np.isfinite(probability)):
        raise ValueError("cortex_probability contains NaN or infinite values")
    if not 0 <= low_threshold <= high_threshold <= 1:
        raise ValueError("expected 0 <= low_threshold <= high_threshold <= 1")

    low = mask & (probability >= low_threshold)
    high = mask & (probability >= high_threshold)
    if not np.any(high):
        return np.zeros(mask.shape, dtype=bool)
    try:
        from scipy.ndimage import binary_propagation
    except ImportError:
        return _fallback_binary_propagation(high, low)

    structure = np.zeros((3, 3, 3), dtype=bool)
    structure[1, 1, :] = True
    structure[1, :, 1] = True
    structure[:, 1, 1] = True
    return binary_propagation(high, structure=structure, mask=low)
