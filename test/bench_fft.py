# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import math
import os

import cuda.tile as ct
from conftest import dtype_id, shape_id
from util import estimate_bench_iter
from kernels.fft import fft_kernel
from kernels.fft_ref import fft_kernel as fft_kernel_ref


@pytest.fixture(params=[(16 * 1024, 16, 16, 16)], ids=shape_id)
def shape(request):
    return request.param


@pytest.fixture(params=[torch.complex32,
                        torch.complex64],
                ids=dtype_id)
def dtype(request):
    return request.param


@pytest.fixture(params=["cutile", "torch"])
def fft_backend(request):
    match(request.param):
        case "torch": return torch_fft
        case "cutile":
            impl = os.environ.get('FFT_IMPL', None)
            if impl is None:
                return cutile_fft
            elif impl == 'ref':
                return cutile_fft_ref
            else:
                raise f"Unexpected env FFT_IMPL: {impl}"
        case _: raise RuntimeError(f'Unknown backend: {request.param}')


tolerance_map = {
    torch.complex32: 1e-2,
    torch.complex64: 1e-5,
}

complex_to_real_dtype = {
    torch.complex32: torch.float16,
    torch.complex64: torch.float32,
}


@pytest.mark.benchmark(group='fft')
def bench_fft(shape, dtype, fft_backend, benchmark):
    batch_size, decomp = shape[0], shape[1:]
    N = math.prod(decomp)
    x = torch.rand((batch_size, N), dtype=dtype, device='cuda')
    weights = make_twiddles(decomp, complex_to_real_dtype[dtype])
    args = (x, decomp, *weights)
    y_test = fft_backend(*args)
    y_ref = torch_fft(*args)
    l2error = (y_ref - y_test).norm() / (y_ref).norm()
    assert l2error < tolerance_map[dtype]
    warmup_rounds, iterations, rounds = estimate_bench_iter(fft_backend, args, cudagraph=True)
    benchmark.pedantic(
        fft_backend, args,
        rounds=rounds, warmup_rounds=warmup_rounds, iterations=iterations,
        cudagraph=True
    )

    flop_count = 0  # TODO
    bytes_rw = sum([
        t.numel() * t.dtype.itemsize if isinstance(t, torch.Tensor) else 0
        for t in [*args, y_ref]
    ])
    benchmark.extra_info['flop_count'] = flop_count
    benchmark.extra_info['bytes_rw'] = bytes_rw


def cutile_fft(x, decomp, W0, W1, W2, T0, T1):
    x_ri = torch.view_as_real(x)
    F0, F1, F2 = decomp
    y_ri = torch.zeros_like(x_ri)
    batch, N, _ = x_ri.shape
    assert x_ri.shape[2] == 2
    BS = 2  # Tunable row tile size
    D = 64  # Last dim size, workaround for tile.load performance
    grid = (math.ceil(batch / BS), 1, 1)
    ct.launch(torch.cuda.current_stream(), grid, fft_kernel, (
        x_ri.view(batch, N * 2 // D, D),
        y_ri.view(batch, N * 2 // D, D),
        W0.view(F0, F0, 2),
        W1.view(F1, F1, 2),
        W2.view(F2, F2, 2),
        T0.view(F0, F1 * F2, 2),
        T1.view(F1, F2, 2),
        N, F0, F1, F2, BS, D))
    return torch.view_as_complex(y_ri)


def cutile_fft_ref(x, decomp, W0, W1, W2, T0, T1):
    x_ri = torch.view_as_real(x)
    F0, F1, F2 = decomp
    y_ri = torch.zeros_like(x_ri)
    batch, N, _ = x_ri.shape
    grid = (batch, 1, 1)
    ct.launch(torch.cuda.current_stream(), grid, fft_kernel_ref, (
        x_ri, y_ri, W0, W1, W2, T0, T1,
        N, F0, F1, F2
    ))
    return torch.view_as_complex(y_ri)


def torch_fft(x, *args):
    return torch.fft.fft(x, axis=-1)


def twiddles(rows: int, cols: int, factor: int):
    (I, J) = torch.meshgrid(torch.arange(rows, device='cuda'),
                            torch.arange(cols, device='cuda'),
                            indexing='ij')
    W = torch.exp(-2*math.pi*1j*(I*J) / factor)
    return torch.view_as_real(W)


def make_twiddles(decomp, precision):
    F0, F1, F2 = decomp
    N = F0*F1*F2
    F1F2 = F1*F2

    # Generate twiddle factors for each dimension
    W0_ri = twiddles(F0, F0, F0).to(precision)
    W1_ri = twiddles(F1, F1, F1).to(precision)
    W2_ri = twiddles(F2, F2, F2).to(precision)
    # Generate twiddle factors for dimension transitions
    T0_ri = twiddles(F0, F1F2, N).to(precision)
    # Because of the data repetition, we can use a smaller twiddle factor
    T1_ri = twiddles(F1, F2, F1F2).to(precision)
    return (W0_ri, W1_ri, W2_ri, T0_ri, T1_ri)
