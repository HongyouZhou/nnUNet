"""Small multiprocessing helpers used by the legacy PENGWIN tools.

This module intentionally keeps the call shape of the external ``tqdmp``
package so that the PENGWIN utilities do not depend on an otherwise optional
package merely to combine ``multiprocessing.Pool`` with a progress bar.
"""

from __future__ import annotations

from functools import partial
from multiprocessing import Pool
from typing import Any, Callable, Iterable, Optional, Sequence, Union

from tqdm import tqdm


def _run_indexed(
    indexed_value: tuple[int, Any],
    *,
    function: Callable[..., Any],
    mult_iter: bool,
    kwargs: dict[str, Any],
) -> tuple[int, Any]:
    index, value = indexed_value
    if mult_iter:
        result = function(*value, **kwargs)
    else:
        result = function(value, **kwargs)
    return index, result


def tqdmp(
    function: Callable[..., Any],
    iterable: Union[Iterable[Any], tuple[Iterable[Any], ...]],
    num_processes: Optional[int],
    mult_iter: bool = False,
    mult_out: bool = False,
    chunksize: int = 1,
    desc: Optional[str] = None,
    disable: bool = False,
    **kwargs: Any,
) -> list[Any] | tuple[list[Any], ...]:
    """Map ``function`` over input values, with optional worker processes.

    ``None`` and ``0`` execute in the current process. When ``mult_iter`` is
    true, the supplied iterables are zipped and their values are passed as
    positional arguments. Results retain input order even though parallel
    workers may finish out of order.
    """
    if num_processes is not None and num_processes < 0:
        raise ValueError("num_processes must be None or a non-negative integer")
    if chunksize < 1:
        raise ValueError("chunksize must be at least 1")

    if mult_iter:
        if not isinstance(iterable, tuple) or not iterable:
            raise ValueError("mult_iter=True requires a non-empty tuple of iterables")
        inputs: Sequence[Any] = list(zip(*iterable))
    else:
        inputs = list(iterable)

    worker = partial(
        _run_indexed,
        function=function,
        mult_iter=mult_iter,
        kwargs=kwargs,
    )
    results: list[Any] = [None] * len(inputs)

    if num_processes in (None, 0):
        iterator = map(worker, enumerate(inputs))
        for index, result in tqdm(
            iterator,
            total=len(inputs),
            desc=desc,
            disable=disable,
        ):
            results[index] = result
    else:
        with Pool(processes=num_processes) as pool:
            iterator = pool.imap_unordered(
                worker,
                enumerate(inputs),
                chunksize=chunksize,
            )
            for index, result in tqdm(
                iterator,
                total=len(inputs),
                desc=desc,
                disable=disable,
            ):
                results[index] = result

    if mult_out:
        if not results:
            return tuple()
        return tuple(list(values) for values in zip(*results))
    return results
