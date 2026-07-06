# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import cuda.tile as ct
import math
from functools import partial
from util import assert_equal

from cuda.tile._exception import TileCompilerTimeoutError, TileCompilerExecutionError
from cuda.tile.tune import _tune as tune_mod
from cuda.tile.tune import _tune_utils as tune_utils
from cuda.tile.tune import exhaustive_search, TuningResult
from operator import attrgetter


@ct.kernel
def dummy_kernel(x, TILE_SIZE: ct.Constant[int]):
    pass


@ct.kernel
def copy_kernel(x, out, TILE_SIZE: ct.Constant[int]):
    bid = ct.bid(0)
    t = ct.load(x, index=(bid,), shape=(TILE_SIZE,))
    ct.store(out, index=(bid,), tile=t)


def grid_fn_on_x(x, cfg):
    return (math.ceil(x.shape[0] / cfg), 1, 1)


# ========== Test basic exhaustive search ==========
def test_exhaustive_search_returns_best(monkeypatch):
    x = torch.empty((256,), device="cuda")
    search_space = [64, 128, 256]

    times = {64: 5.0, 128: 1.0, 256: 3.0}

    def fake_benchmark(stream, grid, kernel, pyargs):
        cfg = pyargs[1]
        return times[cfg]

    monkeypatch.setattr(tune_mod, "_benchmark", fake_benchmark, raising=True)

    result = exhaustive_search(
            search_space,
            torch.cuda.current_stream(),
            grid_fn=partial(grid_fn_on_x, x),
            kernel=dummy_kernel,
            args_fn=lambda cfg: (x, cfg),
    )

    assert isinstance(result, TuningResult)
    assert result.best.config == 128
    assert result.best.mean_us == 1.0
    assert result.failures == ()
    assert len(result.successes) == 3
    assert "3 succeeded, 0 failed" in str(result)

    sorted_successes = sorted(result.successes, key=lambda m: m.mean_us)
    assert list(map(attrgetter("config"), sorted_successes)) == [128, 256, 64]
    assert list(map(attrgetter("mean_us"), sorted_successes)) == [1.0, 3.0, 5.0]


def test_exhaustive_search_skips_slow_configs(monkeypatch):
    x = torch.empty((256,), device="cuda")
    search_space = range(1, 8)
    times = {
        1: [1.0],
        2: [2.0],
        3: [3.0],
        4: [4.0],
        5: [5.0],
        6: [1.1, 1.1, 9.0, 1.1, 9.0, 1.1, 20.0, 20.0, 20.0, 20.0, 20.0],
        7: [20, 20.0, 20.5, 20.0, 20.5, 20.0],
    }
    sample_counts = {cfg: 0 for cfg in search_space}

    def fake_benchmark(stream, grid, kernel, pyargs):
        cfg = pyargs[1]
        sample_index = min(sample_counts[cfg], len(times[cfg]) - 1)
        sample_counts[cfg] += 1
        return times[cfg][sample_index]

    monkeypatch.setattr(tune_mod, "_TOP_K", 5, raising=True)
    monkeypatch.setattr(tune_mod, "_WARM_UP_REPEATS", 1, raising=True)
    monkeypatch.setattr(tune_mod, "_MIN_REPEATS", 2, raising=True)
    monkeypatch.setattr(tune_mod, "_BATCH_REPEATS", 1, raising=True)
    monkeypatch.setattr(tune_mod, "_benchmark", fake_benchmark, raising=True)

    result = exhaustive_search(
        search_space,
        torch.cuda.current_stream(),
        grid_fn=partial(grid_fn_on_x, x),
        kernel=dummy_kernel,
        args_fn=lambda cfg: (x, cfg),
        quiet=True,
    )

    assert result.best.config == 1
    assert len(result.successes) == 7
    assert len(result.failures) == 0
    sorted_successes = sorted(result.successes, key=lambda m: m.mean_us)
    assert [m.config for m in sorted_successes] == [1, 2, 3, 4, 5, 6, 7]
    assert [m.num_samples for m in sorted_successes[:5]] == [2, 2, 2, 2, 2]
    assert sorted_successes[5].num_samples > 2  # config 6 keep running until it cannot beat Top-K
    assert sorted_successes[6].num_samples == 1  # config 7 stopped after 1 sample for too slow


# ========== Test empty search space ==========
def test_empty_search_space_raises():
    x = torch.empty((256,), device="cuda")
    with pytest.raises(ValueError, match=r"Search space is empty"):
        exhaustive_search(
            [],
            torch.cuda.current_stream(),
            grid_fn=partial(grid_fn_on_x, x),
            kernel=dummy_kernel,
            args_fn=lambda cfg: (x, cfg),
        )


# ========== Test error skips bad configs ==========
def test_skips_failed_configs(monkeypatch):
    x = torch.empty((256,), device="cuda")

    failures = {
        64: TileCompilerTimeoutError("simulated timeout", "", None),
        256: TileCompilerExecutionError(1, "simulated error", "", None),
    }

    def fake_benchmark(stream, grid, kernel, pyargs):
        cfg = pyargs[1]
        if cfg in failures:
            raise failures[cfg]
        return 2.0

    monkeypatch.setattr(tune_mod, "_benchmark", fake_benchmark, raising=True)

    result = exhaustive_search(
        [64, 128, 256],
        torch.cuda.current_stream(),
        grid_fn=partial(grid_fn_on_x, x),
        kernel=dummy_kernel,
        args_fn=lambda cfg: (x, cfg),
    )

    assert result.best.config == 128
    assert result.best.mean_us == 2.0
    assert len(result.failures) == 2

    err_cfg, err_type, err_msg = result.failures[0]
    assert (err_cfg, err_type) == (64, "TileCompilerTimeoutError")
    assert "simulated timeout" in err_msg

    err_cfg, err_type, _ = result.failures[1]
    assert (err_cfg, err_type) == (256, "TileCompilerExecutionError")
    assert "1 succeeded, 2 failed" in str(result)

    assert len(result.successes) == 1
    m = result.successes[0]
    assert m.config == 128
    assert m.mean_us == 2.0


# ========== Test all configs fail ==========
def test_all_configs_fail_raises(monkeypatch):
    x = torch.empty((256,), device="cuda")

    def fake_benchmark(*args, **kwargs):
        raise TileCompilerTimeoutError("always fails", "", None)

    monkeypatch.setattr(tune_mod, "benchmark_with_timeout",
                        fake_benchmark, raising=True)
    monkeypatch.setattr(tune_mod, "_benchmark", fake_benchmark, raising=True)

    with pytest.raises(ValueError, match=r"No valid config") as exc_info:
        exhaustive_search(
            [64, 128],
            torch.cuda.current_stream(),
            grid_fn=partial(grid_fn_on_x, x),
            kernel=dummy_kernel,
            args_fn=lambda cfg: (x, cfg),
        )
    assert "No valid config found" in str(exc_info.value)


# ========== Test kernel that mutates input ==========
@ct.kernel
def inplace_kernel(x, TILE_SIZE: ct.Constant[int]):
    bid = ct.bid(0)
    tx = ct.load(x, index=(bid,), shape=(TILE_SIZE,))
    tx_updated = tx + 1
    ct.store(x, index=(bid,), tile=tx_updated)


def test_inplace_plus_one():
    x = torch.ones((1024,), device="cuda")
    original_x = x.clone()

    result = exhaustive_search(
        [64, 128, 256],
        torch.cuda.current_stream(),
        grid_fn=lambda cfg: (math.ceil(1024 / cfg), 1, 1),
        kernel=inplace_kernel,
        args_fn=lambda cfg: (x.clone(), cfg),
    )

    ct.launch(
        torch.cuda.current_stream(),
        (math.ceil(1024 / result.best.config), 1, 1),
        inplace_kernel,
        (x, result.best.config),
    )
    assert_equal(x, original_x + 1)


# ========== Test tune with list-of-arrays argument ==========
@ct.kernel
def add_arrays(arrays, out):
    res = ct.zeros((16, 16), dtype=out.dtype)
    for i in range(len(arrays)):
        t = ct.load(arrays[i], (0, 0), (16, 16))
        res += t
    ct.store(out, (0, 0), res)


def test_tune_list_of_arrays():
    arrays = [torch.ones(16, 16, dtype=torch.int32, device="cuda") for _ in range(3)]
    out = torch.zeros(16, 16, dtype=torch.int32, device="cuda")

    result = exhaustive_search(
        [1],
        torch.cuda.current_stream(),
        grid_fn=lambda cfg: (1,),
        kernel=add_arrays,
        args_fn=lambda cfg: (arrays, out.clone()),
    )

    assert len(result.failures) == 0


def test_tune_list_of_arrays_ipc(monkeypatch):
    arrays = [torch.ones(16, 16, dtype=torch.int32, device="cuda") for _ in range(3)]
    out = torch.zeros(16, 16, dtype=torch.int32, device="cuda")

    monkeypatch.setattr(
        tune_utils, "_benchmark",
        lambda *args: pytest.fail("Should use IPC payload call"),
        raising=True)

    elapsed_us, wall_time_sec = tune_utils.benchmark_with_timeout(
        torch.cuda.current_stream(), (1,), add_arrays, (arrays, out), 5.0)

    assert elapsed_us >= 0
    assert wall_time_sec is not None
    assert_equal(out, torch.full_like(out, 3))


# ========== [IPC] Test tune handles launch timeout ==========
@ct.kernel
def conditional_dead_loop_kernel(x, out, TILE_SIZE: ct.Constant[int]):
    t = ct.load(x, index=(0,), shape=(TILE_SIZE,))
    t2 = ct.add(t, t)
    ct.store(out, index=(0,), tile=t2)

    dead_loop_flag = ct.extract(t, 0, ()).item()
    while dead_loop_flag != 0:
        ct.atomic_add(out, 0, 1)


def test_ipc_tune_handles_launch_timeout(monkeypatch):
    single_run_timeout_sec = 4.0
    tile_size = 16
    result = exhaustive_search(
        [0, 1, 0],
        torch.cuda.current_stream(),
        grid_fn=lambda cfg: (1,),
        kernel=conditional_dead_loop_kernel,
        args_fn=lambda cfg: (
            torch.full((tile_size,), cfg, dtype=torch.int32, device="cuda"),
            torch.zeros((tile_size,), dtype=torch.int32, device="cuda"),
            tile_size),
        single_run_timeout_sec=single_run_timeout_sec,
    )

    assert len(result.failures) == 1
    assert len(result.successes) == 2

    err_cfg, err_type, err_msg = result.failures[0]
    assert (err_cfg, err_type) == (1, "TileLaunchTimeoutError")
    assert "CUDA kernel launch exceeded timeout" in err_msg


def test_timeout_disabled_runs_without_subprocess(monkeypatch):
    x = torch.empty((1,), device="cuda")
    search_space = [1, 2, 3]

    monkeypatch.setattr(
        tune_mod, "benchmark_with_timeout",
        lambda *args, **kwargs: pytest.fail(
            "benchmark_with_timeout should be skipped when timeout is disabled"),
        raising=True)
    monkeypatch.setattr(tune_mod, "_benchmark", lambda *args: 1.0, raising=True)

    result = exhaustive_search(
        search_space,
        torch.cuda.current_stream(),
        grid_fn=partial(grid_fn_on_x, x),
        kernel=dummy_kernel,
        args_fn=lambda cfg: (x, cfg),
        quiet=True,
    )

    assert len(result.successes) == len(search_space)
    assert len(result.failures) == 0


# ========== Test edge cases ==========
def test_ipc_export_none_falls_back(monkeypatch):
    monkeypatch.setattr(
        tune_utils, "_export_ipc_benchmark_payload", lambda *args: None, raising=True)
    monkeypatch.setattr(tune_utils, "_benchmark", lambda *args: 123, raising=True)

    assert tune_utils.benchmark_with_timeout(None, (1,), None, (), 5.0) == (123, None)


def test_ipc_skip_cache_and_recompile_kernel(monkeypatch):
    x = torch.arange(16, dtype=torch.float32, device="cuda")
    out = torch.empty_like(x)
    stream = torch.cuda.current_stream()
    ct.launch(stream, (1,), copy_kernel, (x, out, 16))
    assert_equal(out, x)
    out.zero_()
    assert_equal(out, torch.zeros_like(out))

    monkeypatch.setattr(
        tune_utils, "_benchmark",
        lambda *args: pytest.fail("Should use IPC payload call"),
        raising=True)
    tune_utils.benchmark_with_timeout(
        stream, (1,), copy_kernel, (x, out, 16), 5.0)
    assert_equal(out, x)

    out.zero_()
    assert_equal(out, torch.zeros_like(out))
    ct.launch(stream, (1,), copy_kernel, (x, out, 16))
    assert_equal(out, x)


def test_tune_scalar_only_ipc(monkeypatch):
    @ct.kernel
    def scalar_only_kernel(n: int):
        if n != 0:
            n = n + 1

    monkeypatch.setattr(
        tune_utils, "_benchmark",
        lambda *args: pytest.fail("Should use IPC payload call"),
        raising=True)
    elapsed_us, _ = tune_utils.benchmark_with_timeout(
        torch.cuda.current_stream(), (1,), scalar_only_kernel, (7,), 5.0)
    assert elapsed_us >= 0
