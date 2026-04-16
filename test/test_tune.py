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
from cuda.tile.tune import exhaustive_search, TuningResult
from operator import attrgetter


@ct.kernel
def dummy_kernel(x, TILE_SIZE: ct.Constant[int]):
    pass


def grid_fn_on_x(x, cfg):
    return (math.ceil(x.shape[0] / cfg), 1, 1)


# ========== Test basic exhaustive search ==========
def test_exhaustive_search_returns_best(monkeypatch):
    x = torch.empty((256,), device="cuda")
    search_space = [64, 128, 256]

    times = {64: 5.0, 128: 1.0, 256: 3.0}

    def fake_time_us(stream, grid, kernel, get_args):
        args = get_args()
        cfg = args[1]
        return times[cfg], 1, 20

    monkeypatch.setattr(tune_mod, "_time_us", fake_time_us, raising=True)

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

    assert list(map(attrgetter("config"), result.successes)) == [64, 128, 256]
    assert list(map(attrgetter("mean_us"), result.successes)) == [5.0, 1.0, 3.0]


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

    def fake_time_us(stream, grid, kernel, get_args):
        args = get_args()
        cfg = args[1]
        if cfg in failures:
            raise failures[cfg]
        return 2.0, 1, 20

    monkeypatch.setattr(tune_mod, "_time_us", fake_time_us, raising=True)

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

    def fake_time_us(stream, grid, kernel, get_args):
        raise TileCompilerTimeoutError("always fails", "", None)

    monkeypatch.setattr(tune_mod, "_time_us", fake_time_us, raising=True)

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
