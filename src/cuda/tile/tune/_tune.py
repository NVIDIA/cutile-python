# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import Any, Callable, Generic, Sequence, TypeVar

from cuda.tile._cext import _benchmark
from cuda.tile.tune._tune_utils import benchmark_with_timeout
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


def progress(phase: int, n: int, total: int, errors: int):
    if n == 0:
        print()
    marker = _spinner[n % len(_spinner)]
    width = len(str(total))
    end = "\r\033[K" if n == total - 1 else ""
    if phase == 0:
        message = "Warmup & initial run"
    elif phase == 1:
        message = f"Converging Top-{total} configs"
    elif phase == 2:
        message = "Converging all configs"
    else:
        raise ValueError(f"Invalid phase: {phase}")
    print(f"\r{marker} [Phase {phase}/2] {message}: {n:{width}}/{total} | Errors: {errors:{width}}",
          end=end, flush=True)


def _in_terminal() -> bool:
    try:
        return sys.stdout.isatty()
    except AttributeError:
        return False


def exhaustive_search(
    search_space: Sequence[T],
    stream,
    grid_fn: Callable[[T], tuple[int, ...]],
    kernel,
    args_fn: Callable[[T], tuple[Any, ...]],
    hints_fn: Callable[[T], dict[str, Any]] | None = None,
    *,
    quiet: bool = False,
    single_run_timeout_sec: float | None = None,
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
        single_run_timeout_sec: Wall-time timeout (in seconds) per kernel launch,
            enforced by running benchmarks in a subprocess. When None (the
            default), timeouts are disabled and kernels run directly without a
            subprocess.


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
        from cuda.tile import ByTarget

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
            hints = lambda cfg: {'num_ctas': ByTarget(sm_100=cfg['num_ctas'])}
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
        kernel = matmul.replace_hints(num_ctas=ByTarget(sm_100=num_ctas))
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

    total = len(search_space)
    isatty = _in_terminal()

    if total == 0:
        raise ValueError("Search space is empty.")

    # min-heap of running candidates, sorted by (mean_us, error_margin_us).
    # The integer tie-breaker keeps heap operations stable when timings match.
    running: list[tuple[float, float, int, _TimingCandidate[T]]] = []
    converged: list[_TimingCandidate[T]] = []
    errors = []

    # Phase 0: Warmup and initial run for each config.
    # Also Filter out timeout candidates.
    for i, cfg in enumerate(search_space):
        if not quiet and isatty:
            progress(0, i, total, len(errors))
        grid = grid_fn(cfg)
        hints = hints_fn(cfg) if hints_fn is not None else {}
        updated_kernel = kernel.replace_hints(**hints)
        candidate = _TimingCandidate(
            config=cfg,
            grid=grid,
            kernel=updated_kernel,
            get_args=lambda _cfg=cfg: args_fn(_cfg),
        )

        try:
            candidate.warmup(stream, _WARM_UP_REPEATS, single_run_timeout_sec)
            candidate.run_benchmark(stream, _BATCH_REPEATS)
        except Exception as e:
            errors.append((cfg, type(e).__name__, str(e)))
            continue

        if candidate.converged():
            converged.append(candidate)
        else:
            heapq.heappush(
                running, (candidate.mean_us, candidate.error_margin_us, i, candidate))

    # Phase 1: Run benchmarks until we have at least _TOP_K converged configs.
    while len(converged) < _TOP_K and running:
        _, _, order, candidate = heapq.heappop(running)
        try:
            candidate.run_benchmark(stream, _BATCH_REPEATS)
        except Exception as e:
            errors.append((candidate.config, type(e).__name__, str(e)))
            continue

        if candidate.converged():
            if not quiet and isatty:
                progress(1, len(converged), min(_TOP_K, total), len(errors))
            converged.append(candidate)
        else:
            heapq.heappush(
                running, (candidate.mean_us, candidate.error_margin_us, order, candidate))

    # Phase 2: Run remaining candidates until they cannot beat the Top-K.
    if converged and running:
        if len(converged) <= _TOP_K:
            cutoff_mean_us = max(candidate.mean_us for candidate in converged)
        else:
            top_k_converged = heapq.nsmallest(
                _TOP_K,
                converged,
                key=lambda candidate: candidate.mean_us,
            )
            cutoff_mean_us = max(candidate.mean_us for candidate in top_k_converged)
        for _, _, _, candidate in running:
            try:
                while (not candidate.converged() and
                       candidate.mean_us - candidate.error_margin_us < cutoff_mean_us):
                    candidate.run_benchmark(stream, _BATCH_REPEATS)
            except Exception as e:
                errors.append((candidate.config, type(e).__name__, str(e)))
                continue

            if not quiet and isatty:
                progress(2, len(converged), total, len(errors))
            converged.append(candidate)
        running = []

    successes = [candidate.to_measurement() for candidate in converged]

    if not successes:
        cfg, exc_type, msg = errors[0]
        raise ValueError(f"No valid config found in search space."
                         f"\nConfig: {cfg}\n{exc_type}: {msg}")

    best = min(successes, key=lambda measure: measure.mean_us)
    result = TuningResult(best=best,
                          successes=tuple(successes),
                          failures=tuple(errors))

    if not quiet:
        print(result)
    return result


_MAX_MEASURE_TIME_US = 5_000_000  # 5s
_MAX_REPEATS = 1000
_MIN_REPEATS = 5
_BATCH_REPEATS = 5
_WARM_UP_REPEATS = 3
_TOP_K = 5


@dataclass
class _TimingCandidate(Generic[T]):
    config: T
    grid: tuple[int, ...]
    kernel: Any
    get_args: Callable[[], tuple[Any, ...]]
    num_samples: int = 0
    mean_us: float = 0.0
    m2: float = 0.0
    error_margin_us: float = 0.0

    def warmup(self, stream, num_times, launch_timeout_sec):
        assert num_times > 0

        for i in range(num_times):
            if i == 0 and launch_timeout_sec is not None:
                # First warmup is timed to ensure it doesn't deadlock.
                benchmark_with_timeout(
                    stream, self.grid, self.kernel, self.get_args(), launch_timeout_sec)
            else:
                _benchmark(stream, self.grid, self.kernel, self.get_args())

    def run_benchmark(self, stream, num_times):
        for _ in range(min(num_times, _MAX_REPEATS - self.num_samples)):
            self._add_sample(_benchmark(stream, self.grid, self.kernel, self.get_args()))

    def converged(self) -> bool:
        # Stop if ...
        return self.num_samples >= _MIN_REPEATS and (
            self.error_margin_us <= 0.01 * self.mean_us  # estimated relative error is <1%,
            or self.error_margin_us <= 0.5  # ... or estimated absolute error is <=0.5us,
            or self.num_samples >= _MAX_REPEATS   # ... or we ran too many times,
            or self.mean_us * self.num_samples > _MAX_MEASURE_TIME_US)  # ... or taking too long

    def to_measurement(self) -> Measurement[T]:
        return Measurement(config=self.config,
                           mean_us=self.mean_us,
                           error_margin_us=self.error_margin_us,
                           num_samples=self.num_samples)

    def _add_sample(self, elapsed_us: float):
        # Welford algorithm for running mean and variance
        self.num_samples += 1
        old_mean = self.mean_us
        self.mean_us += (elapsed_us - old_mean) / self.num_samples
        self.m2 += (elapsed_us - old_mean) * (elapsed_us - self.mean_us)
        if self.num_samples > 1:
            sample_var = self.m2 / (self.num_samples - 1)
            var = sample_var / self.num_samples
            self.error_margin_us = math.sqrt(var) * 1.96  # 95% confidence interval
