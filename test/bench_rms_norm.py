# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from conftest import dtype_id, shape_id, get_tileiras_version

import pytest
import torch
import cuda.tile as ct
from cuda.tile.tune import exhaustive_search
from cuda.tile._bytecode import BytecodeVersion
import itertools
from math import ceil
from util import estimate_bench_iter
import platform
from kernels.rms_norm import (
    rms_norm_kernel, rms_norm_kernel_gather, rms_norm_kernel_static_persistent
)
from functools import partial
from types import SimpleNamespace


timeout = 1  # sec


def get_shape_params():
    return [(65536, 1024),
            (65536, 2048),
            (65536, 4096)]


@pytest.fixture(params=get_shape_params(), ids=shape_id)
def shape(request):
    return request.param


@pytest.fixture(params=[torch.float16, torch.float32, torch.bfloat16], ids=dtype_id)
def dtype(request):
    return request.param


@pytest.mark.benchmark(group='rms_norm')
@pytest.mark.parametrize('algo', ['persistent', 'gather', 'regular'])
def bench_rms_norm(shape, dtype, algo, backend, benchmark):
    x_shape = shape
    w_shape = (shape[1], )
    x = torch.rand(x_shape, dtype=dtype, device="cuda")
    weight = torch.randn(w_shape, dtype=dtype, device="cuda")

    eps = 1e-5

    if algo == 'persistent':
        if platform.machine().lower().startswith(('arm', 'aarch64')):
            pytest.xfail("autotune static persistent hangs on arm64")
        if get_tileiras_version() == BytecodeVersion.V_13_2:
            pytest.xfail("autotune static persistent hangs on tileiras 13.2")

    if algo == 'persistent':
        static_persistent, gather = True, False
    elif algo == 'gather':
        static_persistent, gather = False, True
    else:
        static_persistent, gather = False, False

    if algo == 'persistent' and shape[1] > 1024:
        pytest.skip("No valid configuration")

    o = backend(x, weight, eps, static_persistent, gather)
    ref = ref_rms_norm(x, weight, eps)
    torch.testing.assert_close(o, ref, atol=1e-2, rtol=5e-2)
    torch.cuda.synchronize()

    warmup_rounds, iterations, rounds = estimate_bench_iter(
        backend, (x, weight, eps, static_persistent, gather),
        cudagraph=True
    )

    benchmark.pedantic(
        backend, (x, weight, eps, static_persistent, gather),
        rounds=rounds, warmup_rounds=warmup_rounds, iterations=iterations,
        cudagraph=True
    )

    M, N = x.shape
    flop_count = M * (4 * N + 2)
    bytes_rw = sum([t.numel() * t.dtype.itemsize for t in (x, weight, o)])
    benchmark.extra_info['flop_count'] = flop_count
    benchmark.extra_info['bytes_rw'] = bytes_rw


def _static_persistent_autotune_grid(x, cfg):
    """Grid function for static persistent RMS Norm autotuning"""
    NUM_SMS = torch.cuda.get_device_properties(
        "cuda"
    ).multi_processor_count
    M, N = x.shape[0], x.shape[1]
    grid_size = min(
        NUM_SMS,
        ceil(M / cfg.TILE_SIZE_M) * ceil(N / cfg.TILE_SIZE_N),
    )
    return (grid_size,)


def _static_persistent_autotune_configs():
    """Iterator of autotune configurations for RMS Norm kernel."""
    ts_m_vals = [2, 4, 8, 16]
    ts_n_vals = [2**9, 2**10, 2**11, 2**12, 2**13, 2**14]
    num_ctas_vals = [1, 2]
    occupancy_vals = [1, 2, 4, 8]

    for ts_m, ts_n, s, w in itertools.product(ts_m_vals, ts_n_vals, num_ctas_vals, occupancy_vals):
        yield SimpleNamespace(
            TILE_SIZE_M=ts_m,
            TILE_SIZE_N=ts_n,
            num_ctas=s,
            occupancy=w,
        )


def _static_persistent_autotune_predicate(x, cfg):
    """Predicate function for static persistent RMS Norm autotuning"""
    return x.shape[1] * 2 > cfg.TILE_SIZE_N >= x.shape[1]


def _standard_autotune_configs():
    """Get autotune configurations for RMS Norm kernel"""
    ts_vals = [2**7, 2**8, 2**9, 2**10, 2**11, 2**12]
    num_ctas_vals = [1, 2]
    occupancy_vals = [1, 2, 4, 8, 16, 32]
    for ts, s, w in itertools.product(ts_vals, num_ctas_vals, occupancy_vals):
        yield SimpleNamespace(
            TILE_SIZE=ts,
            num_ctas=s,
            occupancy=w,
        )


# Autotuning cache
_tuning_cache = {}


def _tuning_cache_key(kind, x: torch.Tensor):
    return (kind, x.dtype, x.shape[0], x.shape[1])


def _rms_norm_static_persistent_base(stream, x, y, weight, eps):
    key = _tuning_cache_key("static", x)
    if (key not in _tuning_cache):
        search_space = [
            cfg for cfg in _static_persistent_autotune_configs()
            if _static_persistent_autotune_predicate(x, cfg)
        ]
        result = exhaustive_search(
            search_space,
            torch.cuda.current_stream(),
            grid_fn=partial(_static_persistent_autotune_grid, x),
            kernel=rms_norm_kernel_static_persistent,
            args_fn=lambda cfg: (x, y.clone(), weight, cfg.TILE_SIZE_M, cfg.TILE_SIZE_N, eps),
            hints_fn=lambda cfg: {
                "num_ctas": cfg.num_ctas,
                "occupancy": cfg.occupancy,
            },
        )
        cfg = result.best.config
        kernel = rms_norm_kernel_static_persistent.replace_hints(num_ctas=cfg.num_ctas,
                                                                 occupancy=cfg.occupancy)
        _tuning_cache[key] = kernel, cfg

    kernel, cfg = _tuning_cache[key]
    grid = _static_persistent_autotune_grid(x, cfg)
    ct.launch(
        stream, grid,
        kernel,
        (x, y, weight, cfg.TILE_SIZE_M, cfg.TILE_SIZE_N, eps),
    )
    return y


def _rms_norm_standard_gather_base(stream, x, weight, y, rstd, N, eps):
    key = _tuning_cache_key("gather", x)
    if (key not in _tuning_cache):
        result = exhaustive_search(
            list(_standard_autotune_configs()),
            torch.cuda.current_stream(),
            grid_fn=lambda cfg: (x.shape[0], ),
            kernel=rms_norm_kernel_gather,
            args_fn=lambda cfg: (x, weight, y.clone(), rstd.clone(), N, eps, cfg.TILE_SIZE),
            hints_fn=lambda cfg: {
                "num_ctas": cfg.num_ctas,
                "occupancy": cfg.occupancy,
            },
        )
        cfg = result.best.config
        kernel = rms_norm_kernel_gather.replace_hints(num_ctas=cfg.num_ctas,
                                                      occupancy=cfg.occupancy)
        _tuning_cache[key] = (kernel, cfg)

    kernel, cfg = _tuning_cache[key]
    ct.launch(
        stream, (x.shape[0],),
        kernel,
        (x, weight, y, rstd, N, eps, cfg.TILE_SIZE),
    )
    return y


def _rms_norm_standard_tiled_base(stream, x, weight, y, rstd, N, eps):
    key = _tuning_cache_key("standard", x)
    if (key not in _tuning_cache):
        result = exhaustive_search(
            list(_standard_autotune_configs()),
            torch.cuda.current_stream(),
            grid_fn=lambda cfg: (x.shape[0], ),
            kernel=rms_norm_kernel,
            args_fn=lambda cfg: (x, weight, y.clone(), rstd.clone(), N, eps, cfg.TILE_SIZE),
            hints_fn=lambda cfg: {
                "num_ctas": cfg.num_ctas,
                "occupancy": cfg.occupancy,
            },
        )
        cfg = result.best.config
        kernel = rms_norm_kernel.replace_hints(num_ctas=cfg.num_ctas, occupancy=cfg.occupancy)
        _tuning_cache[key] = (kernel, cfg)

    kernel, cfg = _tuning_cache[key]
    ct.launch(
        stream, (x.shape[0],), kernel,
        (x, weight, y, rstd, N, eps, cfg.TILE_SIZE),
    )
    return y


def cutile_rms_norm(x, weight, eps, static_persistent, gather):
    x = x.contiguous()
    weight = weight.contiguous()

    # Allocate output tensor
    y = torch.empty_like(x)
    M, N = x.shape

    with ct.compiler_timeout(timeout):
        if static_persistent:
            _rms_norm_static_persistent_base(torch.cuda.current_stream(), x, y, weight, eps)
        else:
            rstd = torch.empty((M,), dtype=torch.float32, device='cuda')
            if gather:
                _rms_norm_standard_gather_base(torch.cuda.current_stream(),
                                               x, weight, y, rstd, N, eps)
            else:
                _rms_norm_standard_tiled_base(
                    torch.cuda.current_stream(), x, weight, y, rstd, N, eps
                )
        return y.view(*x.shape)


def torch_rms_norm(input, weight, eps, static_persistent=False, gather=False):
    # layer norm should always be calculated in float32
    normalized_shape = weight.shape
    dims = tuple(i for i in range(-1, -len(normalized_shape) - 1, -1))
    variance = input.to(torch.float32).pow(2).mean(dims, keepdim=True)
    input = input * torch.rsqrt(variance + eps)
    # convert into half-precision if necessary
    if weight.dtype in [torch.float16, torch.bfloat16]:
        input = input.to(weight.dtype)

    return weight * input


def ref_rms_norm(input, weight, eps):
    return torch_rms_norm(input, weight, eps)
