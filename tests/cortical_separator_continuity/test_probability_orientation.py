import itertools
import pickle

import nibabel as nib
import numpy as np
import pytest

from nnunetv2.imageio.nibabel_reader_writer import NibabelIO, NibabelIOWithReorient
from tools.charite_cortical.continuity_workflow import _load_probability_channel


ORIENTATIONS = list(itertools.product(itertools.permutations(range(3)),
                                      itertools.product((-1, 1), repeat=3)))


@pytest.mark.parametrize("shape", [(3, 4, 5), (4, 4, 4)])
@pytest.mark.parametrize("axes,signs", ORIENTATIONS)
def test_probability_restoration_preserves_each_native_voxel(tmp_path, shape, axes, signs):
    affine = np.eye(4)
    affine[:3, :3] = 0
    for native_axis, world_axis in enumerate(axes):
        affine[world_axis, native_axis] = signs[native_axis] * (native_axis + 1)
    affine[:3, 3] = (11, -7, 3)
    expected = np.arange(np.prod(shape), dtype=np.float32).reshape(shape) / np.float32(np.prod(shape))
    path = tmp_path / "prediction.nii.gz"
    nib.save(nib.Nifti1Image(expected, affine), path)
    data, properties = NibabelIOWithReorient().read_images([str(path)])
    npz = tmp_path / "prediction.npz"
    np.savez_compressed(npz, probabilities=np.concatenate((1 - data, data * 0, data)))
    npz.with_suffix(".pkl").write_bytes(pickle.dumps(properties))

    actual = _load_probability_channel(npz, nib.load(path), 2)
    np.testing.assert_array_equal(actual, expected)


def test_probability_restoration_handles_non_reoriented_nibabel_exports(tmp_path):
    expected = np.arange(24, dtype=np.float32).reshape(2, 3, 4) / 24
    path = tmp_path / "prediction.nii.gz"
    nib.save(nib.Nifti1Image(expected, np.diag([-1., -2., 3., 1.])), path)
    data, properties = NibabelIO().read_images([str(path)])
    npz = tmp_path / "prediction.npz"
    np.savez_compressed(npz, probabilities=data)
    npz.with_suffix(".pkl").write_bytes(pickle.dumps(properties))
    np.testing.assert_array_equal(_load_probability_channel(npz, nib.load(path), 0), expected)


def test_probability_restoration_rejects_mismatched_geometry(tmp_path):
    npz = tmp_path / "prediction.npz"
    np.savez_compressed(npz, probabilities=np.zeros((1, 4, 3, 2), dtype=np.float32))
    npz.with_suffix(".pkl").write_bytes(pickle.dumps({
        "nibabel_stuff": {"original_affine": np.eye(4)}
    }))
    image = nib.Nifti1Image(np.zeros((2, 3, 4)), np.diag([2., 1., 1., 1.]))
    with pytest.raises(ValueError, match="original affines differ"):
        _load_probability_channel(npz, image, 0)
