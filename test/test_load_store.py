# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import torch
import pytest

from math import ceil
from conftest import float_dtypes, bool_dtypes, get_tileiras_version, int_dtypes, dtype_id
from cuda.tile._bytecode.version import BytecodeVersion
from cuda.tile._ir.cast_ops import _is_implicit_cast_ok
from cuda.tile._ir.typing_support import to_dtype
from util import assert_close, assert_equal, filecheck, get_bytecode
from torch.testing import make_tensor
# example-begin
import cuda.tile as ct

TILE_SIZE = 16


@ct.kernel
def load_store_with_hints_kernel(x, y):
    bid = ct.bid(0)
    tx = ct.load(
        x,
        index=(bid,),
        shape=(TILE_SIZE,),
        latency=8,        # high-latency DRAM load
    )
    ct.store(
        y,
        index=(bid,),
        tile=tx,
        latency=2,        # cheaper write
        allow_tma=False,  # disallow TMA
    )
# example-end


def test_load_store_with_hints():
    x = make_tensor((32,), dtype=torch.float16, device='cuda')
    y = torch.zeros_like(x)
    grid = (ceil(x.shape[0] / TILE_SIZE), 1, 1)
    ct.launch(torch.cuda.current_stream(), grid, load_store_with_hints_kernel, (x, y))
    assert_equal(y, x)


def make_ct_matmul_kernel(latency: int | None, allow_tma: bool | None):
    def kernel(A, B, C,
               tm: ct.Constant[int],
               tn: ct.Constant[int],
               tk: ct.Constant[int]):
        bidx = ct.bid(0)
        bidy = ct.bid(1)
        num_tiles = ct.num_tiles(A, axis=1, shape=(tm, tk))
        sum = ct.full((tm, tn), 0, dtype=np.float32)

        for k in range(num_tiles):
            a = ct.load(A, index=(bidx, k), shape=(tm, tk), latency=latency, allow_tma=allow_tma)
            b = ct.load(B, index=(k, bidy), shape=(tk, tn), latency=latency, allow_tma=allow_tma)
            sum = ct.mma(a, b, sum)
            sum = ct.astype(sum, C.dtype)
            ct.store(C, index=(bidx, bidy), tile=sum, latency=latency, allow_tma=allow_tma)
    return kernel


def make_array_copy_2d_kernel(latency: int | None):
    def kernel(x, y,
               TILE_X: ct.Constant[int],
               TILE_Y: ct.Constant[int]):
        bidx = ct.bid(0)
        bidy = ct.bid(1)

        i = bidx * TILE_X + ct.arange(TILE_X, dtype=np.int32)
        j = bidy * TILE_Y + ct.arange(TILE_Y, dtype=np.int32)
        indices = (i[:, None], j)
        tx = ct.gather(x, indices, latency=latency)
        ct.scatter(y, indices, tx, latency=latency)
    return kernel


def _create_check_directive(allow_tma: bool | None,
                            latency: int | None,
                            load_op: str,
                            store_op: str) -> str:
    version = get_tileiras_version()
    wildcard = "{{.*}}"
    check_directives = ["// CHECK: div_by<16>"]
    if allow_tma is None and latency is None:
        # no optimization hints
        check_directives.append(f"// CHECK-NOT: {load_op} {wildcard} optimization_hints")
        check_directives.append(f"// CHECK-NOT: {store_op} {wildcard} optimization_hints")
    else:
        allow_tma_hint = [f"allow_tma = {str(allow_tma).lower()}"] if allow_tma is not None else []
        latency_hint = [f"latency = {latency}"] if latency is not None else []
        hints = ", ".join(allow_tma_hint + latency_hint)
        target = "default = " if version >= BytecodeVersion.V_13_3 else wildcard
        check_directives.append(
            f"// CHECK: {load_op} {wildcard} optimization_hints{wildcard}{target}{{{hints}}}"
        )
        check_directives.append(
            f"// CHECK: {store_op} {wildcard} optimization_hints{wildcard}{target}{{{hints}}}"
        )
    return "\n".join(check_directives)


class TestArrayAssumption:

    @pytest.mark.use_mlir
    @pytest.mark.parametrize("latency", [5, None])
    @pytest.mark.parametrize("allow_tma", [False, None])
    def test_load_store(self, latency, allow_tma):
        m, n, k = 32, 32, 128
        A = torch.randn((m, k), dtype=torch.float32, device="cuda")
        B = torch.randn((k, n), dtype=torch.float32, device="cuda")
        C = torch.zeros((m, n), dtype=torch.float32, device="cuda")
        tm, tn, tk = 32, 16, 64
        grid = (ceil(m / tm), ceil(n / tn), 1)
        kernel = ct.kernel(make_ct_matmul_kernel(latency, allow_tma))
        bytecode = get_bytecode(kernel, (A, B, C, tm, tn, tk))
        check_directive = _create_check_directive(
            allow_tma, latency, "load_view_tko", "store_view_tko"
        )
        filecheck(bytecode, check_directive)
        ct.launch(torch.cuda.current_stream(), grid, kernel, (A, B, C, tm, tn, tk))
        assert_close(C, A @ B, atol=1e-3, rtol=1e-3)

    @pytest.mark.use_mlir
    @pytest.mark.parametrize("latency", [5, None])
    def test_gather_scatter(self, latency):
        shape = (1024, 1024)
        tile = (128, 128)
        x = make_tensor(shape, dtype=torch.float32, device="cuda")
        y = torch.zeros_like(x)
        grid = (*(ceil(i / j) for i, j in zip(shape, tile)), 1)
        kernel = ct.kernel(make_array_copy_2d_kernel(latency))
        bytecode = get_bytecode(kernel, (x, y, tile[0], tile[1]))
        check_directive = _create_check_directive(
            None, latency, "load_ptr_tko", "store_ptr_tko"
        )

        filecheck(bytecode, check_directive)
        ct.launch(torch.cuda.current_stream(), grid, kernel, (x, y, tile[0], tile[1]))
        assert_equal(y, x)


@ct.kernel
def array_copy_1d(x, y, TILE: ct.Constant[int]):
    bid = ct.bid(0)
    tx = ct.load(x, index=(bid,), shape=TILE)
    ct.store(y, index=(bid,), tile=tx)


@pytest.mark.parametrize("shape", [(128,), (225,)])
@pytest.mark.parametrize("tile", [64, 128])
@pytest.mark.parametrize("x_dtype", float_dtypes+int_dtypes+bool_dtypes, ids=dtype_id)
@pytest.mark.parametrize("y_dtype", float_dtypes+int_dtypes+bool_dtypes, ids=dtype_id)
def test_array_copy_dtype_implicit_cast(shape, tile, x_dtype, y_dtype):
    x = make_tensor(shape, dtype=x_dtype, device='cuda')
    y = torch.zeros_like(x, dtype=y_dtype, device='cuda')
    grid = (ceil(shape[0] / tile), 1, 1)

    def launch():
        ct.launch(torch.cuda.current_stream(), grid, array_copy_1d, (x, y, tile))

    if _is_implicit_cast_ok(to_dtype(x_dtype), to_dtype(y_dtype)):
        launch()
        assert_equal(y, x.to(y.dtype))


@ct.kernel
def load_store_0d_shape(x, y):
    for i in range(x.shape[0]):
        tx = ct.load(x, (i,), shape=())
        ct.store(y, index=(i,), tile=tx)


@ct.kernel
def load_store_scalar(x, y):
    for i in range(x.shape[0]):
        tx = ct.load(x, i, shape=1)
        s = tx.item()
        ct.store(y, index=i, tile=s)


@ct.kernel
def load_store_0d_tile_index(x, y):
    for i in range(x.shape[0]):
        idx = ct.full((), i, dtype=ct.int32)
        tx = ct.load(x, idx, shape=(1,))
        ct.store(y, index=(idx,), tile=tx)


@pytest.mark.parametrize("kernel", [load_store_0d_shape,
                                    load_store_scalar,
                                    load_store_0d_tile_index])
def test_load_store_scalar_or_0d(kernel):
    x = make_tensor((5,), dtype=torch.float16, device='cuda')
    y = torch.zeros_like(x)
    ct.launch(torch.cuda.current_stream(), (1,), kernel, (x, y))
    assert_equal(y, x)
