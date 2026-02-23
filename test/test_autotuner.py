# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import cuda.tile as ct
import math
from functools import partial
from util import assert_equal

from cuda.tile._cext import default_tile_context
from cuda.tile._exception import TileCompilerTimeoutError, TileCompilerExecutionError

ct_experimental = pytest.importorskip("cuda.tile_experimental")
autotuner_mod = ct_experimental._autotuner
autotune_launch = ct_experimental.autotune_launch
clear_autotune_cache = ct_experimental.clear_autotune_cache


@ct.kernel
def dummy_kernel(x, TILE_SIZE: ct.Constant[int]):
    pass


@ct.kernel
def other_dummy_kernel(x, TILE_SIZE: ct.Constant[int]):
    pass


def grid_fn_on_x(x, cfg):
    return (math.ceil(x.shape[0] / cfg), 1, 1)


@pytest.fixture
def _patch_timer_and_launch(monkeypatch):
    calls = {"count": 0}

    def fake_time_ms(run_once, *, get_args, stream, warmup=2, rep=10):
        calls["count"] += 1
        return 1

    monkeypatch.setattr(autotuner_mod, "_time_ms", fake_time_ms, raising=True)
    monkeypatch.setattr(ct, "launch", lambda *a, **k: None, raising=True)
    return calls


# ========== Test Clear Cache ==========#
def test_clear_autotune_cache(_patch_timer_and_launch):
    x = torch.empty((256,), device="cuda")

    grid_fn = partial(grid_fn_on_x, x)

    def args_fn(cfg):
        return (x, cfg)

    search_space = [64, 128]

    # 1) Clear entire cache → next tune for dummy_kernel should re-benchmark
    clear_autotune_cache()
    autotune_launch(
        torch.cuda.current_stream(), grid_fn, dummy_kernel, args_fn,
        search_space=search_space
    )
    first_count = _patch_timer_and_launch["count"]
    assert first_count > 0, "Expected timing to run after clear_autotune_cache()"

    # 2) Clear by kernel + key
    clear_autotune_cache()
    default_key = autotuner_mod._default_key(args_fn(search_space[0]))
    custom_key = (0.0, )

    # populate cache for default key
    autotune_launch(
        torch.cuda.current_stream(), grid_fn, dummy_kernel, args_fn,
        search_space=search_space
    )
    # populate cache for custom key
    autotune_launch(
        torch.cuda.current_stream(), grid_fn, dummy_kernel, args_fn, key=custom_key,
        search_space=search_space
    )

    before_key_clear = _patch_timer_and_launch["count"]

    # now clear only the entry for (dummy_kernel, default_key)
    clear_autotune_cache(kernel=dummy_kernel, key=default_key)

    # default key should re-tune
    autotune_launch(
        torch.cuda.current_stream(), grid_fn, dummy_kernel, args_fn,
        search_space=search_space
    )
    after_key_clear = _patch_timer_and_launch["count"]
    assert after_key_clear > before_key_clear, (
        "Expected re-tune after clear_autotune_cache(kernel=dummy_kernel, key=default_key)"
    )

    # custom key should not re-tune
    autotune_launch(
        torch.cuda.current_stream(), grid_fn, dummy_kernel, args_fn, key=custom_key,
        search_space=search_space
    )
    after_custom_key_clear = _patch_timer_and_launch["count"]
    assert after_custom_key_clear == after_key_clear, (
        "Expected no additional timing calls after "
        "clear_autotune_cache(kernel=dummy_kernel, key=default_key)"
    )

    # 3) Clear by kernel only
    clear_autotune_cache()
    autotune_launch(
        torch.cuda.current_stream(), grid_fn, dummy_kernel, args_fn,
        search_space=search_space
    )
    autotune_launch(
        torch.cuda.current_stream(), grid_fn, other_dummy_kernel, args_fn,
        search_space=search_space
    )
    before_kernel_clear = _patch_timer_and_launch["count"]

    clear_autotune_cache(kernel=dummy_kernel)

    # dummy_kernel should re-tune
    autotune_launch(
        torch.cuda.current_stream(), grid_fn, dummy_kernel, args_fn,
        search_space=search_space
    )
    after_kernel_clear = _patch_timer_and_launch["count"]
    assert after_kernel_clear > before_kernel_clear, (
        "Expected timing to run after clear_autotune_cache(kernel=dummy_kernel)"
    )

    # other_dummy_kernel should not re-tune
    autotune_launch(
        torch.cuda.current_stream(), grid_fn, other_dummy_kernel, args_fn,
        search_space=search_space
    )
    after_other_kernel_clear = _patch_timer_and_launch["count"]
    assert after_other_kernel_clear == after_kernel_clear, (
        "Expected no additional timing calls after clear_autotune_cache(kernel=dummy_kernel)"
    )


# ========== Test tuning with different keys but same kernel ==========#
def test_different_keys_same_kernel(_patch_timer_and_launch):
    x = torch.empty((256,), device="cuda")

    grid_fn = partial(grid_fn_on_x, x)
    custom_key = (0.0, )

    clear_autotune_cache()
    # 1) First tune
    res1 = autotune_launch(
        torch.cuda.current_stream(),
        grid_fn,
        dummy_kernel,
        args_fn=lambda cfg: (x, 0.0, cfg),
        key=custom_key,
        search_space=[64, 128]
    )
    first_count = _patch_timer_and_launch["count"]
    assert first_count > 0, "Expected timing to run on first tune (cache miss)"
    assert not res1.cache_hit
    assert len(res1.tuning_record) == first_count

    # 2) Second tune with same args → cache hit (no new timings)
    res2 = autotune_launch(
        torch.cuda.current_stream(),
        grid_fn,
        dummy_kernel,
        args_fn=lambda cfg: (x, 0.0, cfg),
        key=custom_key,
        search_space=[64, 128]
    )
    second_count = _patch_timer_and_launch["count"]
    assert second_count == first_count, "Expected cache hit: no additional timing calls"
    assert res2.cache_hit
    assert res2.tuning_record == res1.tuning_record

    # 3) Different scalar value -> cache miss (re-tune)
    res3 = autotune_launch(
        torch.cuda.current_stream(),
        grid_fn,
        dummy_kernel,
        args_fn=lambda cfg: (x, 1.0, cfg),
        key=(1.0, ),
        search_space=[64, 128]
    )
    third_count = _patch_timer_and_launch["count"]
    assert third_count > second_count, "Expected timing to run after scalar value change"
    assert not res3.cache_hit
    assert len(res3.tuning_record) == third_count - second_count


# ========== Test tuning with different kernels but same key ==========#
def test_different_kernels_same_key(_patch_timer_and_launch):
    x = torch.empty((256,), device="cuda")

    grid_fn = partial(grid_fn_on_x, x)
    custom_key = (0.0, )
    clear_autotune_cache()
    # 1) First tune
    res1 = autotune_launch(
        torch.cuda.current_stream(),
        grid_fn,
        dummy_kernel,
        args_fn=lambda cfg: (x, 0.0, cfg),
        key=custom_key,
        search_space=[64, 128]
    )
    first_count = _patch_timer_and_launch["count"]
    assert first_count > 0, "Expected timing to run on first tune (cache miss)"
    assert not res1.cache_hit
    assert len(res1.tuning_record) == first_count

    # 2) Second tune with same args → cache hit (no new timings)
    res2 = autotune_launch(
        torch.cuda.current_stream(),
        grid_fn,
        dummy_kernel,
        args_fn=lambda cfg: (x, 0.0, cfg),
        key=custom_key,
        search_space=[64, 128]
    )
    second_count = _patch_timer_and_launch["count"]
    assert second_count == first_count, "Expected cache hit: no additional timing calls"
    assert res2.cache_hit
    assert res2.tuning_record == res1.tuning_record

    # 3) Different kernel -> cache miss (re-tune)
    res3 = autotune_launch(
        torch.cuda.current_stream(),
        grid_fn,
        other_dummy_kernel,
        args_fn=lambda cfg: (x, 0.0, cfg),
        key=custom_key,
        search_space=[64, 128]
    )
    third_count = _patch_timer_and_launch["count"]
    assert third_count > second_count, "Expected timing to run after different kernel"
    assert not res3.cache_hit
    assert len(res3.tuning_record) == third_count - second_count


# ========== Test Arg Policy: custom transforms ==========#
def test_custom_tuning_args(monkeypatch):
    # Record the packed args passed to ct.launch
    launches = []
    monkeypatch.setattr(ct, "launch", lambda *a: launches.append(a), raising=True)

    x = torch.empty((256,), device="cuda")
    # Custom value: a recognizable tensor
    custom_x = torch.full_like(x, 7)

    clear_autotune_cache()
    tuned_result = autotune_launch(
        stream=torch.cuda.current_stream(),
        grid_fn=partial(grid_fn_on_x, x),
        kernel=dummy_kernel,
        args_fn=lambda cfg: (custom_x, cfg),
        launch_args_fn=lambda cfg: (x, cfg),
        search_space=[64, 128]
    )

    # At least two launches should have occurred
    assert len(launches) >= 2, "ct.launch was not called during tuning"

    # Check that the y argument passed to launch is our scratch (not the real y)
    # packed order for dummy_kernel is (x, TILE_SIZE)
    # Notice the last launch is the one with the best config so it does not run our custom transform
    _, _, _, packed_tune = launches[-2]  # (stream, grid, kernel, packed_args)
    assert packed_tune[0] is custom_x
    assert packed_tune[0] is not x

    _, _, _, packed_best = launches[-1]  # (stream, grid, kernel, packed_args)
    assert packed_best[0] is x

    # Then test the tuned result - we can still use reguar ct.launch with the tuned result
    num_launches = len(launches)
    ct.launch(
        torch.cuda.current_stream(),
        tuned_result.grid,
        tuned_result.kernel,
        (x, tuned_result.tuned_config)
    )
    assert len(launches) == num_launches + 1


# ========== Test timeout / failed configs handling ==========#
def test_autotune_handles_timeout_and_raises_when_all_configs_fail(monkeypatch, caplog):
    old_timeout = default_tile_context.config.compiler_timeout_sec

    x = torch.empty((256,), device="cuda")

    def fake_time_ms(run_once, *, get_args, stream, warmup=2, rep=10):
        if default_tile_context.config.compiler_timeout_sec <= 1:
            raise TileCompilerTimeoutError("simulated compiler timeout", "", None)
        return 1

    monkeypatch.setattr(autotuner_mod, "_time_ms", fake_time_ms, raising=True)

    clear_autotune_cache()
    # No timeout
    with caplog.at_level("DEBUG"):
        autotune_launch(
            stream=torch.cuda.current_stream(),
            grid_fn=partial(grid_fn_on_x, x),
            kernel=dummy_kernel,
            args_fn=lambda cfg: (x, cfg),
            search_space=[64, 128]
        )
    assert "compilation timeout" not in caplog.text

    # Timeout
    caplog.clear()
    with caplog.at_level("DEBUG", logger=autotuner_mod.logger.name):
        with pytest.raises(ValueError, match=r"No valid config"):
            autotune_launch(
                stream=torch.cuda.current_stream(),
                grid_fn=partial(grid_fn_on_x, x),
                kernel=dummy_kernel,
                args_fn=lambda cfg: (x, cfg),
                search_space=[64, 128],
                compiler_time_limit_sec=1,
                force_retune=True,
            )
    assert "compilation timeout" in caplog.text
    # Make sure the timeout is restored
    assert default_tile_context.config.compiler_timeout_sec == old_timeout


# ========== Test search_space as callable ==========#
def test_search_space_callable(_patch_timer_and_launch):
    x = torch.empty((256,), device="cuda")

    clear_autotune_cache()
    autotune_launch(
        stream=torch.cuda.current_stream(),
        grid_fn=partial(grid_fn_on_x, x),
        kernel=dummy_kernel,
        args_fn=lambda cfg: (x, cfg),
        search_space=lambda: [64, 128],
    )

    assert _patch_timer_and_launch["count"] > 0


# ========== Test search_space accepts a plain iterator ==========#
def test_search_space_iterator_support(_patch_timer_and_launch):
    x = torch.empty((256,), device="cuda")

    def args_fn(cfg):
        return (x, cfg)

    def make_search_space_iter():
        for cfg in [64, 128]:
            yield cfg

    clear_autotune_cache()
    stream = torch.cuda.current_stream()
    grid_fn = partial(grid_fn_on_x, x)

    autotune_launch(
        stream=stream,
        grid_fn=grid_fn,
        kernel=dummy_kernel,
        args_fn=args_fn,
        search_space=make_search_space_iter(),
    )
    count = _patch_timer_and_launch["count"]
    assert count > 0, "Expected timing to run for iterator search_space (cache miss)"


# ========== Test max_iter limiting number of timed configs ==========#
def test_max_iter_limits_number_of_timed_configs(_patch_timer_and_launch):
    x = torch.empty((256,), device="cuda")

    search_space = list(range(32, 32 + 20))  # 20 configs
    max_iter = 5

    clear_autotune_cache()
    autotune_launch(
        stream=torch.cuda.current_stream(),
        grid_fn=partial(grid_fn_on_x, x),
        kernel=dummy_kernel,
        args_fn=lambda cfg: (x, cfg),
        search_space=search_space,
        max_iter=max_iter,
    )

    # Each successful config causes one call to _time_ms, so this should
    # never exceed max_iter.
    assert _patch_timer_and_launch["count"] == max_iter


# ========== Test force_retune ignores cache and re-benchmarks ==========#
def test_force_retune_ignores_cache_and_rebenchmarks(_patch_timer_and_launch):
    x = torch.empty((256,), device="cuda")
    search_space = [64, 128]

    def args(cfg):
        return (x, cfg)

    clear_autotune_cache()
    stream = torch.cuda.current_stream()
    grid_fn = partial(grid_fn_on_x, x)

    # First autotune: fills cache
    res1 = autotune_launch(
        stream=stream,
        grid_fn=grid_fn,
        kernel=dummy_kernel,
        args_fn=args,
        search_space=search_space,
    )
    first_count = _patch_timer_and_launch["count"]
    assert first_count > 0
    assert not res1.cache_hit

    # Second autotune with force_retune=True: should run timing again
    res2 = autotune_launch(
        stream=stream,
        grid_fn=grid_fn,
        kernel=dummy_kernel,
        args_fn=args,
        search_space=search_space,
        force_retune=True,
    )
    second_count = _patch_timer_and_launch["count"]
    assert second_count > first_count, "Expected additional timing calls when force_retune=True"
    assert not res2.cache_hit
    assert len(res2.tuning_record) == second_count - first_count


# ========== Test execution error handling (skip bad configs) ==========#
def test_autotune_skips_execution_error_and_uses_other_configs(monkeypatch):
    x = torch.empty((256,), device="cuda")
    search_space = [64, 128]

    # Track how many configs we attempted to time
    calls = {"count": 0}

    def fake_time_ms(run_once, *, get_args, stream, warmup=2, rep=10):
        calls["count"] += 1
        # First timed config: simulate a compiler execution error
        if calls["count"] == 1:
            raise TileCompilerExecutionError(1, "simulated compiler error", "", None)
        # Second timed config: succeed
        return 1.0

    monkeypatch.setattr(autotuner_mod, "_time_ms", fake_time_ms, raising=True)
    monkeypatch.setattr(ct, "launch", lambda *a, **k: None, raising=True)

    clear_autotune_cache()
    # Should not raise, because at least one config (the second) succeeds.
    autotune_launch(
        stream=torch.cuda.current_stream(),
        grid_fn=partial(grid_fn_on_x, x),
        kernel=dummy_kernel,
        args_fn=lambda cfg: (x, cfg),
        search_space=search_space,
        force_retune=True,
    )

    assert calls["count"] >= 2


# ========== Real use case: test Inplace Plus One with clone policy ==========#
@ct.kernel
def inplace_kernel(
    x,
    TILE_SIZE: ct.Constant[int]
):
    bid = ct.bid(0)
    tx = ct.load(x, index=(bid,), shape=(TILE_SIZE,))
    tx_updated = tx + 1
    ct.store(x, index=(bid,), tile=tx_updated)


def inplace_plus_one_base(stream, x):
    autotune_launch(
        stream,
        grid_fn=partial(grid_fn_on_x, x),
        kernel=inplace_kernel,
        args_fn=lambda cfg: (x.clone(), cfg),
        launch_args_fn=lambda cfg: (x, cfg),
        search_space=[64, 128]
    )
    return x


def test_inplace_plus_one():
    x = torch.empty((1024,), device="cuda")
    original_x = x.clone()
    inplace_plus_one_base(torch.cuda.current_stream(), x)
    assert_equal(x, original_x + 1)
