from __future__ import annotations

import unittest

from tools.PENGWIN.utils.parallel import tqdmp


def _scale_and_shift(value: int, *, scale: int, shift: int) -> int:
    return value * scale + shift


def _add(left: int, right: int, *, shift: int = 0) -> int:
    return left + right + shift


def _pair(value: int) -> tuple[int, int]:
    return value, value * value


class ProgressMapTests(unittest.TestCase):
    def test_synchronous_map_preserves_order_and_passes_kwargs(self) -> None:
        result = tqdmp(
            _scale_and_shift,
            [3, 1, 2],
            None,
            disable=True,
            scale=2,
            shift=5,
        )
        self.assertEqual(result, [11, 7, 9])

    def test_zero_processes_is_synchronous(self) -> None:
        self.assertEqual(tqdmp(abs, [-2, -1], 0, disable=True), [2, 1])

    def test_parallel_map_preserves_order(self) -> None:
        result = tqdmp(
            _scale_and_shift,
            [4, 1, 3, 2],
            2,
            disable=True,
            scale=3,
            shift=-1,
        )
        self.assertEqual(result, [11, 2, 8, 5])

    def test_multiple_iterables_are_unpacked(self) -> None:
        result = tqdmp(
            _add,
            ([1, 2, 3], [10, 20, 30]),
            None,
            mult_iter=True,
            disable=True,
            shift=4,
        )
        self.assertEqual(result, [15, 26, 37])

    def test_multiple_outputs_are_unzipped(self) -> None:
        result = tqdmp(_pair, [2, 3], None, mult_out=True, disable=True)
        self.assertEqual(result, ([2, 3], [4, 9]))

    def test_invalid_parallel_options_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "num_processes"):
            tqdmp(abs, [-1], -1, disable=True)
        with self.assertRaisesRegex(ValueError, "chunksize"):
            tqdmp(abs, [-1], None, chunksize=0, disable=True)


if __name__ == "__main__":
    unittest.main()
