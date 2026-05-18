# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from functools import partial
from math import ceil
import torch
import cuda.tile as ct
import pytest

from kernels.layer_norm import layer_norm_bwd_dwdb, layer_norm_fwd, layer_norm_bwd_dx_partial_dwdb
from conftest import dtype_id, shape_id
from util import estimate_bench_iter


@pytest.fixture(params=[
    (8, 16, 512),
    (100, 250),
    (1000, 4000),
    (2048, 8192),
    (65536, 16384),
], ids=shape_id)
def shape(request):
    return request.param


@pytest.fixture(params=[
    torch.bfloat16, torch.float32,
    # FIXME: f16 raises TileCompilerError: uses too much shared memory even on (64, 64)
    # torch.float16
], ids=dtype_id)
def dtype(request):
    return request.param


@pytest.mark.benchmark(group='layer_norm')
@pytest.mark.parametrize("mode", ["forward", "backward"])
def bench_layer_norm(shape, dtype, mode, backend, benchmark):
    weight = torch.randn(shape[-1], dtype=dtype, device='cuda', requires_grad=True)
    bias = torch.randn(shape[-1], dtype=dtype, device='cuda', requires_grad=True)
    x = -2.3 + 0.5 * torch.randn(shape, dtype=dtype, device='cuda')
    dy = 0.1 * torch.randn_like(x)
    x.requires_grad_(True)

    eps = 1e-5

    atol, rtol = {
        torch.float32: (1e-4, 1e-3),
        torch.bfloat16: (1e-2, 1e-2),
    }[dtype]

    # Run in non default stream so backward graph can be captured without
    # sync with default stream
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        y = backend(x, weight, bias, eps)
        y_ref = torch_layer_norm(x, weight, bias, eps)
        if mode == "forward":
            torch.testing.assert_close(y, y_ref, atol=atol, rtol=rtol)
            bench_f, bench_args = backend, (x, weight, bias, eps)
        else:
            y.backward(dy, retain_graph=True)
            dx, dw, db = [_.grad.clone() for _ in [x, weight, bias]]
            x.grad, weight.grad, bias.grad = None, None, None

            y_ref.backward(dy, retain_graph=True)
            dx_ref, dw_ref, db_ref = [_.grad.clone() for _ in [x, weight, bias]]

            torch.testing.assert_close(dx, dx_ref, atol=atol, rtol=rtol)
            torch.testing.assert_close(dw, dw_ref, atol=atol, rtol=rtol)
            torch.testing.assert_close(db, db_ref, atol=atol, rtol=rtol)

            bench_f, bench_args = partial(y.backward, retain_graph=True), (dy,)

        warmup_rounds, iterations, rounds = estimate_bench_iter(bench_f, bench_args, cudagraph=True)

        benchmark.pedantic(
            bench_f, bench_args,
            rounds=rounds, warmup_rounds=warmup_rounds, iterations=iterations,
            cudagraph=True
        )


class CuTileLayerNorm(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, weight, bias, eps):
        x = input.reshape(-1, input.shape[-1])
        y = torch.empty_like(x)
        M, _ = x.shape
        mean = torch.empty(M, dtype=torch.float32, device=x.device)
        rstd = torch.empty(M, dtype=torch.float32, device=x.device)

        TILE_N = 1024
        ct.launch(torch.cuda.current_stream(), (M,), layer_norm_fwd,
                  (x, weight, bias, y, mean, rstd, eps, TILE_N))

        ctx.save_for_backward(x, weight, bias, mean, rstd)
        ctx.TILE_N = TILE_N

        return y.reshape(*input.shape)

    @staticmethod
    def backward(ctx, grad_output):
        x, weight, bias, mean, rstd = ctx.saved_tensors
        TILE_N = ctx.TILE_N
        M, N = x.shape
        GROUP_SIZE_M = 64
        dy = grad_output.reshape(-1, grad_output.shape[-1])
        dx = torch.empty_like(dy)
        dw = torch.zeros((GROUP_SIZE_M, N), dtype=torch.float32, device=weight.device)
        db = torch.zeros((GROUP_SIZE_M, N), dtype=torch.float32, device=bias.device)
        locks = torch.zeros(GROUP_SIZE_M, dtype=torch.int32, device=weight.device)
        ct.launch(torch.cuda.current_stream(), (M,), layer_norm_bwd_dx_partial_dwdb,
                  (dx, dy, dw, db, x, weight, mean, rstd, locks, TILE_N))

        final_dw = torch.empty((N,), dtype=weight.dtype, device=weight.device)
        final_db = torch.empty((N,), dtype=bias.dtype, device=bias.device)
        TILE_M = 32
        ct.launch(torch.cuda.current_stream(), (ceil(N / TILE_N),), layer_norm_bwd_dwdb,
                  (dw, db, final_dw, final_db, TILE_M, TILE_N))

        return dx.reshape(*grad_output.shape), final_dw, final_db, None


def cutile_layer_norm(x, weight, bias, eps):
    return CuTileLayerNorm.apply(x, weight, bias, eps)


def torch_layer_norm(x, weight, bias, eps):
    return torch.nn.functional.layer_norm(x, weight.shape, weight, bias, eps)
