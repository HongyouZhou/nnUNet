from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from tools.PENGWIN.inference.inference_abbc2instance import (
    convert_abbc_to_instance,
)


class ABBCInferenceImportTests(unittest.TestCase):
    def test_watershed_does_not_load_original_converter(self) -> None:
        abbc = np.zeros((3, 3, 3), dtype=np.uint8)
        abbc[1, 1, 1] = 2
        ct = np.zeros_like(abbc, dtype=np.float32)
        expected = np.zeros_like(abbc, dtype=np.uint16)
        expected[1, 1, 1] = 1

        with (
            patch(
                "tools.PENGWIN.inference.inference_abbc2instance.process_volume",
                return_value=expected,
            ) as process_volume,
            patch(
                "tools.PENGWIN.inference.inference_abbc2instance._load_original_converter"
            ) as load_original,
        ):
            instances, count = convert_abbc_to_instance(
                abbc,
                ct_array=ct,
                method="watershed",
            )

        np.testing.assert_array_equal(instances, expected)
        self.assertEqual(count, 1)
        process_volume.assert_called_once()
        load_original.assert_not_called()

    def test_original_method_loads_converter_on_demand(self) -> None:
        abbc = np.zeros((2, 2, 2), dtype=np.uint8)
        expected = np.ones_like(abbc, dtype=np.uint16)

        with patch(
            "tools.PENGWIN.inference.inference_abbc2instance._load_original_converter"
        ) as load_original:
            load_original.return_value.return_value = (expected, 1)
            instances, count = convert_abbc_to_instance(
                abbc,
                method="original",
                progressbar=False,
            )

        np.testing.assert_array_equal(instances, expected)
        self.assertEqual(count, 1)
        load_original.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
