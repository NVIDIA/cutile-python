# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterable, Callable, Sequence

from cuda.tile._execution import TileDispatcher
from cuda.tile._exception import TileCompilerTimeoutError, TileCompilerExecutionError
from cuda.tile._cext import default_tile_context
import random
import torch
import logging
import functools
import threading
import warnings

logger = logging.getLogger(__name__)

_MAX_SEARCH_ITEMS = 10_000  # safety cap for very large / infinite streams

_autotune_lock = threading.RLock()


def _with_autotune_lock(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with _autotune_lock:
            return func(*args, **kwargs)
    return wrapper


def _shape_dtype_stride(arg: Any) -> tuple[tuple[int, ...], str, tuple[int, ...] | None]:
    shape = tuple(arg.shape)
    dtype = arg.dtype
    stride = None
    if hasattr(arg, "stride"):                     # PyTorch, etc.
        s = arg.stride() if callable(arg.stride) else arg.stride
        stride = tuple(int(x) for x in s)
    elif hasattr(arg, "strides"):                  # NumPy, etc. (bytes)
        itemsize = getattr(arg, "itemsize", 1)
        stride = tuple(int(b // itemsize) for b in arg.strides)

    return shape, dtype, stride


def _default_key(args: tuple[Any, ...]):
    """Default cache key for autotune.
    The key(for now) is:
    - a tuple of (shape, dtype, stride) for each argument in the runtime argument (tensor),
    - or its type name for each argument in the runtime argument (other types).
    """
    tinfo = []
    for arg in args:
        if hasattr(arg, "shape") and hasattr(arg, "dtype"):
            shape, dtype, stride = _shape_dtype_stride(arg)
            tinfo.append((shape, dtype, stride))
        else:
            tinfo.append(type(arg).__name__)
    return tuple(tinfo)


def _time_ms(run_once, *, get_args, stream, warmup=2, rep=10):
    stream.synchronize()
    for _ in range(warmup):
        run_once(get_args())

    args_per_run = [get_args() for _ in range(rep)]
    stream.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record(stream)
    for i in range(rep):
        run_once(args_per_run[i])
    end.record(stream)
    end.synchronize()

    ms = start.elapsed_time(end)
    return ms / max(1, rep)


@dataclass
class TunedResult:
    # The tuned config
    tuned_config: Any
    # The grid to be used for launching the kernel
    grid: tuple[int, ...]
    # The updated tile dispatcher to be used for launching the kernel
    kernel: TileDispatcher
    # The tuning record: per-config timings from this tuning run
    tuning_record: Sequence[tuple[Any, float]]
    # Whether this result came from cache(True) or from a new tuning run(False)
    cache_hit: bool


@dataclass
class _CacheEntry:
    best_cfg: Any
    best_grid: tuple[int, ...]
    best_kernel: TileDispatcher
    tuning_record: Sequence[tuple[Any, float]]


@_with_autotune_lock
def clear_autotune_cache(*, kernel: TileDispatcher | None = None, key: Any | None = None):
    """Clear entries from the autotuner cache.

    .. deprecated::
        Use :func:`cuda.tile.tune.exhaustive_search` instead.

    The cache is organized as a two-level mapping:
    {kernel_key -> {arg_key -> _CacheEntry}}

    Args:
        kernel:
            If provided, restricts clearing to this kernel's cache only.
            If ``None``, the operation applies to all kernels.
        key:
            If provided, restricts clearing to this argument key.
            If ``None``, all keys at the selected level are cleared.
    """
    warnings.warn(
        "clear_autotune_cache is deprecated, use cuda.tile.tune.exhaustive_search and "
        "manage your own cache instead",
        DeprecationWarning,
        stacklevel=2,
    )
    autotune_cache = default_tile_context.autotune_cache
    if autotune_cache is None:
        return

    if kernel is None:
        if key is not None:
            raise ValueError("key must be None when kernel is None")
        autotune_cache.clear()
    else:
        kernel_key = kernel._pyfunc
        per_kernel = autotune_cache.get(kernel_key)
        if per_kernel is None:
            return
        if key is None:
            per_kernel.clear()
        else:
            per_kernel.pop(key, None)


@contextmanager
def compiler_timeout(timeout_sec: int):
    old_timeout = default_tile_context.config.compiler_timeout_sec
    default_tile_context.config.compiler_timeout_sec = timeout_sec
    try:
        yield
    finally:
        default_tile_context.config.compiler_timeout_sec = old_timeout


def _reservoir_sample(
    iterable: Iterable[Any],
    k: int,
    *,
    rng: random.Random,
    max_items: int,
) -> list[Any]:
    """Uniformly sample up to k items from an iterable using reservoir sampling.

    The sample is limited to the first max_items items.
    """
    reservoir: list[Any] = []
    n_seen = 0

    for item in iterable:
        n_seen += 1
        if n_seen > max_items:
            break
        if len(reservoir) < k:
            reservoir.append(item)
        else:
            j = rng.randint(0, n_seen - 1)
            if j < k:
                reservoir[j] = item
    return reservoir


def autotune_launch(stream, grid_fn, kernel,
                    args_fn: Callable[[Any], tuple[Any, ...]],
                    launch_args_fn: Callable[[Any], tuple[Any, ...]] | None = None,
                    hints_fn: Callable[[Any], dict[str, Any]] | None = None,
                    *,
                    search_space: Iterable[Any] | Callable[[], Iterable[Any]],
                    key: Any | None = None,
                    max_iter: int = 60,
                    compiler_time_limit_sec: int = 10,
                    seed: int | None = None,
                    force_retune: bool = False) -> TunedResult:
    """
    Run the autotuned kernel and return its result.

    .. deprecated::
        Use :func:`cuda.tile.tune.exhaustive_search` instead.

    It performs the following steps:
    1) picks a configuration from the search space or reuses the cached
        best configuration for the given (kernel, key) pair (unless ``force_retune=True``),
    2) launches the kernel with the best configuration,
    3) returns the tuned result.

    The autotuner uses a two-level cache:
        - outer: kernel key (``kernel._pyfunc``)
        - inner: arg key (either the value of ``key`` if provided,
           or the result of ``_default_key`` based on the runtime arguments)

    If both keys hit an entry in the cache, the cached config is reused. To force a new
    tuning run in this case, you can:
      - set ``force_retune=True``, or
      - clear the cache entry with ``clear_cache(kernel=kernel, key=key)``.

    In particular, if you change the seach space but reuses the same key, by default the
    autotuner will continue to reuse the cached config. In that case, you may want to explicitly
    rerun the tuning as mentioned above.

    Args:
        stream:
            CUDA stream to use for all kernel launches during tuning and
            for the final run.
        grid_fn:
            Callable that takes the named arguments and a single
            positional config object and returns a tuple of grid
            dimensions.
        kernel:
            The kernel to autotune.
        args_fn:
            Callable that takes a single positional config object and
            returns a tuple of runtime arguments for ``kernel`` to be passed to
            tuning.
        launch_args_fn:
            Callable that takes a single positional config object and
            returns a tuple of runtime arguments for ``kernel`` to be passed to
            ``ct.launch``.
        hints_fn:
            Callable that takes a single positional config object and
            returns a dictionary of hints to be used for tuning.
        search_space:
            Iterable of config objects to sample from or a callable that
            returns an iterable of config objects to sample from when called.
        key:
            Optional hashable key to use for caching the best config.
            If ``None``, the default key is used.
            The default key is a tuple of (shape, dtype, stride) for each argument
            in the runtime argument (tensor), or its type name for each argument
            in the runtime argument (other types).
        max_iter:
            Maximum number of (valid) configurations to sample from the
            search space.
        compiler_time_limit_sec:
            The compilation time limit for each kernel.
        seed:
            Optional seed for the random number generator used when
            sampling configurations. If ``None``, the global random number
            generator state is used.
        force_retune:
            If ``True``, ignore any cached best config for this key and
            re-run the search. The new best config is then written back
            to the cache.
    """
    warnings.warn(
        "autotune_launch is deprecated, use cuda.tile.tune.exhaustive_search instead",
        DeprecationWarning,
        stacklevel=2,
    )
    import cuda.tile as ct

    if callable(search_space):
        search_space = search_space()

    rng = random.Random(seed)
    search_space = _reservoir_sample(
        search_space,
        k=max_iter,
        rng=rng,
        max_items=_MAX_SEARCH_ITEMS,
    )
    if len(search_space) == 0:
        raise ValueError("Search space must contain at least 1 configuration")

    # --------- Tuning + cache updates under the autotune lock --------- #
    with _autotune_lock:
        autotune_cache = default_tile_context.autotune_cache
        if autotune_cache is None:
            autotune_cache = {}
            default_tile_context.autotune_cache = autotune_cache

        kernel_key = kernel._pyfunc
        per_kernel = autotune_cache.get(kernel_key)
        if per_kernel is None:
            per_kernel = {}
            autotune_cache[kernel_key] = per_kernel
        # _default_key is in the critical path for launch, it can add some overhead.
        if key is None:
            arg_key = _default_key(args_fn(search_space[0]))
        else:
            arg_key = key

        tuning_entries: list[tuple[Any, float]] = []
        cache_hit = False

        if not force_retune and arg_key in per_kernel:
            logger.debug(f"Using cached config for key {key}")
            cache_hit = True
        else:
            indices = list(range(len(search_space)))
            rng.shuffle(indices)

            best_time_ms, best_cfg, best_kernel = float("inf"), None, None
            for i, cfg_idx in enumerate(indices):
                cfg = search_space[cfg_idx]

                grid = grid_fn(cfg)
                hints = hints_fn(cfg) if hints_fn else {}
                updated_kernel = ct.kernel(
                    kernel._pyfunc,
                    **hints
                )

                def run_once(args):
                    ct.launch(stream, grid, updated_kernel, args)

                try:
                    with compiler_timeout(compiler_time_limit_sec):
                        time_ms = _time_ms(
                            run_once,
                            get_args=lambda: args_fn(cfg), # noqa
                            stream=stream,
                        )
                except TileCompilerTimeoutError as e:
                    logger.debug(f"{cfg} compilation timeout: {e}")
                    continue
                except TileCompilerExecutionError as e:
                    logger.debug(f"{cfg} compilation error: {e}")
                    continue

                if time_ms < best_time_ms:
                    best_time_ms = time_ms
                    best_cfg, best_grid, best_kernel = cfg, grid, updated_kernel
                    logger.debug(
                        f"Iteration {i} updated best config to {cfg}: {best_time_ms} ms"
                    )
                # Record the tuning result
                tuning_entries.append((cfg, time_ms))

            # Save the best config and kernel.
            if best_cfg is None:
                raise ValueError("No valid config found")
            per_kernel[arg_key] = _CacheEntry(best_cfg, best_grid, best_kernel, tuning_entries)

        # Lanunch the kernel with the best config
        cache_entry = per_kernel[arg_key]

    # --------- Launch outside the lock --------- #

    # Use the original runtime arguments to run the kernel with the best config
    best_args = (
        launch_args_fn(cache_entry.best_cfg)
        if launch_args_fn
        else args_fn(cache_entry.best_cfg)
    )
    ct.launch(stream, cache_entry.best_grid, cache_entry.best_kernel, best_args)

    # Return the tuned result
    return TunedResult(
        tuned_config=cache_entry.best_cfg,
        grid=cache_entry.best_grid,
        kernel=cache_entry.best_kernel,
        tuning_record=cache_entry.tuning_record,
        cache_hit=cache_hit
    )
