import numpy as np
import pytest
from scipy.ndimage import binary_erosion

from nnunetv2.preprocessing.preprocessors.cortical_separator_continuity_preprocessor import (
    _assert_instance_retention,
    _validate_processed_channels,
    _validate_raw_channels,
    normal_surface_per_instance,
)


def test_per_instance_surface_keeps_touching_interface_lost_by_union_first():
    instances = np.zeros((9, 9, 9), dtype=np.int16)
    instances[2:7, 2:7, 1:4] = 1
    instances[2:7, 2:7, 4:7] = 2
    valid = np.ones_like(instances, dtype=bool)
    separator = np.zeros_like(instances, dtype=bool)

    surface = normal_surface_per_instance(
        instances,
        separator,
        valid,
        spacing_mm_zyx=(0.5, 0.5, 0.5),
        separator_exclusion_mm=2.0,
    )
    union = instances > 0
    union_first = union & ~binary_erosion(
        union, structure=np.ones((3, 3, 3), dtype=bool), border_value=0
    )

    assert surface[4, 4, 3]
    assert surface[4, 4, 4]
    assert not union_first[4, 4, 3]
    assert not union_first[4, 4, 4]


def test_surface_respects_validity_and_separator_distance():
    instances = np.zeros((15, 15, 15), dtype=np.int16)
    instances[2:13, 2:13, 2:13] = 1
    valid = np.ones_like(instances, dtype=bool)
    valid[2, 2, 2] = False
    separator = np.zeros_like(instances, dtype=bool)
    separator[7, 7, 4] = True
    surface = normal_surface_per_instance(
        instances,
        separator,
        valid,
        spacing_mm_zyx=(0.5, 0.5, 0.5),
        separator_exclusion_mm=2.0,
    )
    assert not surface[2, 2, 2]
    assert not surface[7, 7, 2]  # one mm from the separator
    assert surface[7, 7, 12]


def test_instance_validation_overlap_padding_and_retention():
    semantic = np.ones((3, 3, 3), dtype=np.int16)
    support = np.ones_like(semantic)
    validity = np.full_like(semantic, 3)
    instances = np.ones_like(semantic)
    instances[1, 1, 1] = 0  # overlap/unassigned ownership
    validity[1, 1, 1] = 1  # semantic-valid, relation-invalid
    _validate_raw_channels(semantic, support, validity, instances)

    bad = instances.copy()
    bad[1, 1, 1] = 2
    with pytest.raises(ValueError, match="relation-valid"):
        _validate_raw_channels(semantic, support, validity, bad)

    processed = np.zeros((4, 3, 3, 3), dtype=np.int16)
    processed[:, 0] = -1
    processed[3, 1, 1, 1] = 1
    _validate_processed_channels(processed)
    assert _assert_instance_retention({1}, processed[3])["status"] == "PASS"
    with pytest.raises(ValueError, match="removed"):
        _assert_instance_retention({1, 2}, processed[3])
