# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Generic, Sequence, TypeVar

from cuda.tile._cext import (
    _benchmark,
    _synchronize_context,
)
import logging
import sys

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass(frozen=True, kw_only=True)
class Measurement(Generic[T]):
    """Holds a configuration and its timing result."""

    config: T
    """The configuration"""

    mean_us: float
    """Mean time in microseconds"""

    num_samples: int
    """Number of samples taken for the measurement"""

    error_margin_us: float
    """Half of the 95% confidence interval of the measurement"""


@dataclass(frozen=True, kw_only=True)
class TuningResult(Generic[T]):
    """Holds the measurement result for each config."""

    best: Measurement
    """The best measurement"""

    successes: Sequence[Measurement]
    """Measurement of each succeeded config"""

    failures: Sequence[tuple[T, str, str]]
    """`(config, exc_type, message)` for each failed config"""

    def summary(self, *, top_k=10, bottom_k=2) -> str:
        """Return a summary of the result.

        Args:
            top_k (int): Max number of configs to be included, sorted by timing.
        """

        n_ok = len(self.successes)
        n_fail = len(self.failures)
        header = f"{n_ok} succeeded, {n_fail} failed"
        lines = [header]
        ranked = sorted(self.successes, key=lambda t: t.mean_us)

        start_skip, end_skip = top_k, n_ok - bottom_k
        num_skipped = (end_skip - start_skip)
        # if there is only one line to skip, might as well show everything
        if (num_skipped == 1):
            start_skip += 1

        # get max width to align each field
        cw = max(len(str(x.config)) for x in self.successes)
        mw = max(len(f'{x.mean_us:.1f}') for x in self.successes)
        ew = max(len(f'{x.error_margin_us:.1f}') for x in self.successes)
        nw = max(len(str(x.num_samples)) for x in self.successes)

        for i, measure in enumerate(ranked):
            if (i >= start_skip and i < end_skip):
                if (i == start_skip):
                    lines.append(f"    ... {num_skipped} more not shown")
                continue
            marker = "*" if measure == self.best else " "
            lines.append(f"{marker} {str(measure.config):<{cw}}: "
                         f"{measure.mean_us:>{mw}.1f}±{measure.error_margin_us:<{ew}.1f} us "
                         f"({measure.num_samples:{nw}} samples)")

        if self.failures:
            lines.append(f"  {n_fail} failed:")
            for cfg, err_type, msg in self.failures[:top_k]:
                first_line = msg.split("\n", 1)[0]
                if len(first_line) > 60:
                    first_line = first_line[:57] + "..."
                lines.append(f"    {cfg}: {err_type}: {first_line}")
            if n_fail > top_k:
                lines.append(f"    ... {n_fail - top_k} more not shown")
        if n_ok > top_k or n_fail > top_k:
            lines.append("Use .successes and .failures for full results.")
        return "\n".join(lines)

    def __str__(self):
        return self.summary()


_spinner = ['|', '/', '-', '\\']


def progress(n: int, total: int, errors: int):
    try:
        isatty = sys.stdout.isatty()
    except AttributeError:
        return

    if isatty:
        if n == 0:
            print()
        marker = _spinner[n % len(_spinner)]
        width = len(str(total))
        end = "\r" if n < total - 1 else "\r\033[K"
        print(f"{marker}  Progress: {n:{width}}/{total} | Errors: {errors:{width}}",
              end=end, flush=True)


def exhaustive_search(
    search_space: Sequence[T],
    stream,
    grid_fn: Callable[[T], tuple[int, ...]],
    kernel,
    args_fn: Callable[[T], tuple[Any, ...]],
    hints_fn: Callable[[T], dict[str, Any]] | None = None,
    *,
    quiet: bool = False
) -> TuningResult[T]:
    """Searches the entire search space and return the best configuration.

    Args:
        search_space: Sequence of configs to evaluate.
        stream: The CUDA stream to execute kernel on.
        grid_fn: Maps a config to grid dimensions.
        kernel: The kernel to tune.
        args_fn: Maps a config to kernel arguments for timing.
        hints_fn: Maps a config to compiler hints. Default: no hints.
        quiet: If true, avoid printing any progress or result.


    Returns:
        TuningResult with the best config and its time in microseconds.

    Examples:

    .. testcode::
        :template: setup_only.py

        # Define the kernel

        @ct.kernel
        def matmul(X, Y, Out,
                   tm: ct.Constant[int],
                   tn: ct.Constant[int],
                   tk: ct.Constant[int]):

            i, j =  ct.bid(0), ct.bid(1)

            x_view = X.tiled_view((tm, tk), padding_mode=ct.PaddingMode.ZERO)
            y_view = Y.tiled_view((tk, tn), padding_mode=ct.PaddingMode.ZERO)
            acc = ct.zeros((tm, tn), ct.float32)
            for k in range(x_view.num_tiles(1)):
                tx = x_view.load((i, k))
                ty = y_view.load((k, j))
                acc = ct.mma(tx, ty, acc)
            ct.store(Out, (i, j), acc.astype(Out.dtype))

        # Tune the kernel

        from itertools import product

        def tune(x, y, out) -> ct.tune.TuningResult:
            keys = ("tm", "tn", "tk", "num_ctas")
            search_space = [dict(zip(keys, vals))
                            for vals in product(
                            (64, 128),
                            (64, 128),
                            (32, 64),
                            (1, 2))]
            grid = lambda cfg: (ct.cdiv(M, cfg['tm']), ct.cdiv(N, cfg['tn']))
            args = lambda cfg: (x, y, out.clone(), cfg['tm'], cfg['tn'], cfg['tk'])
            hints = lambda cfg: {'num_ctas': cfg['num_ctas']}
            stream = torch.cuda.current_stream()
            tuning_result = ct.tune.exhaustive_search(search_space,
                                                      stream,
                                                      grid,
                                                      matmul,
                                                      args,
                                                      hints)
            return tuning_result

        M, N, K = 1024, 256, 512
        x = torch.rand((M, K), dtype=torch.float16, device='cuda')
        y = torch.rand((K, N), dtype=torch.float16, device='cuda')
        out = torch.zeros((M, N), dtype=torch.float16, device='cuda')

        result = tune(x, y, out)
        print(f"Best config: {result.best.config} ({result.best.mean_us:.1f}us)")

        # Launch the kernel with tuned result

        tm, tn, tk, num_ctas = result.best.config.values()
        kernel = matmul.replace_hints(num_ctas=num_ctas)
        ct.launch(torch.cuda.current_stream(),
                  (ct.cdiv(M, tm), ct.cdiv(N, tn)),
                  kernel,
                  (x, y, out, tm, tn, tk))

        torch.testing.assert_close(out, x @ y)

    .. testoutput::

       16 succeeded, 0 failed
       ...
       Best config: {'tm': ..., 'tn': ..., 'tk': ..., 'num_ctas': ...} (...us)
    """

    successes = []
    errors = []

    best_time_us = float("inf")
    best_cfg_id = None
    total = len(search_space)

    for i, cfg in enumerate(search_space):
        grid = grid_fn(cfg)
        hints = hints_fn(cfg) if hints_fn is not None else {}
        updated_kernel = kernel.replace_hints(**hints)

        try:
            avg_us, error_bar, repeats = _time_us(
                stream, grid, updated_kernel,
                lambda _cfg=cfg: args_fn(_cfg),
            )
        except Exception as e:
            err_type = type(e).__name__
            msg = str(e)
            errors.append((cfg, err_type, msg))
            continue
        else:
            measure = Measurement(config=cfg,
                                  mean_us=avg_us,
                                  error_margin_us=error_bar,
                                  num_samples=repeats)
            successes.append(measure)

            if avg_us < best_time_us:
                best_time_us = avg_us
                best_cfg_id = len(successes) - 1
            if not quiet:
                progress(i, total, len(errors))

    if len(search_space) == 0:
        raise ValueError("Search space is empty.")
    elif best_cfg_id is None:
        cfg, exc_type, msg = errors[0]
        raise ValueError(f"No valid config found in search space."
                         f"\nConfig: {cfg}\n{exc_type}: {msg}")

    result = TuningResult(best=successes[best_cfg_id],
                          successes=tuple(successes),
                          failures=tuple(errors))

    if not quiet:
        print(result)
    return result


_MAX_MEASURE_TIME_US = 5_000_000  # 5s
_MIN_REPEATS = 20
_MAX_REPEATS = 1000
_WARM_UP_STEPS = 10


def _time_us(stream, grid, kernel, get_args) -> tuple[float, float, int]:
    _synchronize_context()

    # Warmup
    for _ in range(_WARM_UP_STEPS):
        _benchmark(stream, grid, kernel, get_args())

    _synchronize_context()

    repeats = 0
    running_mean = 0
    m2 = 0
    while True:
        repeats += 1
        t = _benchmark(stream, grid, kernel, get_args())
        # Welford algorithm for running mean and variance
        old_mean = running_mean
        running_mean += (t - old_mean) / repeats
        m2 += (t - old_mean) * (t - running_mean)
        if repeats >= _MIN_REPEATS:
            sample_var = m2 / (repeats - 1)
            var = sample_var / repeats
            estimated_error = math.sqrt(var) * 1.96  # 95% confidence interval
            # Stop if...
            if (estimated_error <= 0.01 * running_mean  # estimated relative error is <1%,
                    or repeats >= _MAX_REPEATS  # ... or we ran too many times,
                    or running_mean * repeats > _MAX_MEASURE_TIME_US):  # ... or taking too long.
                return running_mean, estimated_error, repeats
