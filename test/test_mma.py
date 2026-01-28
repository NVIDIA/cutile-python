# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from unittest.mock import patch
import pytest
import torch

import cuda.tile as ct
from util import assert_close, assert_equal, torch_to_tf32, is_ampere_or_ada
from conftest import dtype_id
from cuda.tile._exception import TileTypeError, TileUnsupportedFeatureError


# ================ ct.mma =================

@ct.kernel
def mma_kernel(A, B, C,
               tm: ct.Constant[int],
               tn: ct.Constant[int],
               tk: ct.Constant[int]):
    tx = ct.load(A, index=(0, 0), shape=(tm, tk))
    ty = ct.load(B, index=(0, 0), shape=(tk, tn))
    acc = ct.load(C, index=(0, 0), shape=(tm, tn))
    acc = ct.mma(tx, ty, acc)
    ct.store(C, index=(0, 0), tile=acc)


@ct.kernel
def mma_tf32_kernel(A, B, C,
                    tm: ct.Constant[int],
                    tn: ct.Constant[int],
                    tk: ct.Constant[int]):
    tx = ct.load(A, index=(0, 0), shape=(tm, tk)).astype(ct.tfloat32)
    ty = ct.load(B, index=(0, 0), shape=(tk, tn)).astype(ct.tfloat32)
    acc = ct.load(C, index=(0, 0), shape=(tm, tn))
    acc = ct.mma(tx, ty, acc)
    ct.store(C, index=(0, 0), tile=acc)


def get_tolerance(dtype) -> tuple[float, float]:
    if dtype == torch.float8_e5m2:
        return 1e-1, 1e-1
    elif dtype == torch.float8_e4m3fn:
        return 1e-2, 1e-2
    if dtype == torch.float16:
        return 1e-3, 1e-3
    elif dtype == torch.bfloat16:
        return 1e-2, 1e-2
    elif dtype == torch.float32:
        return 1e-5, 1e-5
    elif dtype == torch.float64:
        return 1e-6, 1e-6
    return 0, 0


@dataclass(frozen=True)
class _TestCase:
    dtype: torch.dtype
    acc_dtype: torch.dtype

    def __str__(self):
        return f'{dtype_id(self.dtype)}-{dtype_id(self.acc_dtype)}'


bf16 = torch.bfloat16
f16 = torch.float16
f32 = torch.float32
f64 = torch.float64
f8e4m3fn = torch.float8_e4m3fn
f8e5m2 = torch.float8_e5m2
u8 = torch.uint8
u16 = torch.uint16
u32 = torch.uint32
i8 = torch.int8
i16 = torch.int16
i32 = torch.int32


TC = _TestCase
regular_float_cases = [
    TC(bf16, f32),
    TC(f16, f16),
    TC(f16, f32),
    TC(f32, f32),
    TC(f64, f64),
]
fp8_cases = [
    TC(f8e4m3fn, f16),
    TC(f8e4m3fn, f32),
    TC(f8e5m2, f16),
    TC(f8e5m2, f32),
]
int_cases = [
    TC(i8, i32),
    TC(u8, i32),
]


@pytest.mark.parametrize("tile_size", [(2, 8, 16)])
@pytest.mark.parametrize("case", regular_float_cases, ids=str)
def test_mma_regular_float(tile_size, case):
    m, n, k = tile_size
    A = torch.randn((m, k), dtype=case.dtype, device="cuda")
    B = torch.randn((k, n), dtype=case.dtype, device="cuda")
    C = torch.ones((m, n), dtype=case.acc_dtype, device="cuda")
    ref = torch.mm(A, B, out_dtype=C.dtype) + C
    ct.launch(torch.cuda.current_stream(), (1,), mma_kernel,
              (A, B, C, m, n, k))
    atol, rtol = get_tolerance(A.dtype)
    assert_close(C, ref, atol=atol, rtol=rtol)


@pytest.mark.skipif(is_ampere_or_ada(), reason="float8 not supported on Ampere or Ada")
@pytest.mark.parametrize("tile_size", [(16, 16, 16)])
@pytest.mark.parametrize("case", fp8_cases, ids=str)
def test_mma_fp8(tile_size, case):
    m, n, k = tile_size
    A = torch.randn((m, k), dtype=torch.float32, device="cuda").to(case.dtype)
    B = torch.randn((n, k), dtype=torch.float32, device="cuda").to(case.dtype)
    C = torch.ones((m, n), dtype=case.acc_dtype, device="cuda")
    scale = torch.tensor([1.0], dtype=torch.float32, device="cuda")
    try:
        ref = torch._scaled_mm(A, B.T, scale, scale, out_dtype=C.dtype) + C
    except (RuntimeError, ValueError) as e:
        assert 'Multiplication of two Float8_e5m2 matrices is not supported' in str(e)
        ref = None
    ct.launch(torch.cuda.current_stream(), (1,), mma_kernel,
              (A, B.T, C, m, n, k))
    if ref is not None:
        atol, rtol = get_tolerance(A.dtype)
        assert_close(C, ref, atol=atol, rtol=rtol)


@pytest.mark.parametrize("tile_size", [(8, 2, 4)])
def test_mma_tf32(tile_size):
    m, n, k = tile_size
    A = torch.randn((m, k), dtype=torch.float32, device="cuda")
    B = torch.randn((k, n), dtype=torch.float32, device="cuda")
    C = torch.ones((m, n), dtype=torch.float32, device="cuda")
    ref = C + torch_to_tf32(A) @ torch_to_tf32(B)
    ct.launch(torch.cuda.current_stream(), (1,), mma_tf32_kernel,
              (A, B, C, m, n, k))
    if is_ampere_or_ada():
        # ampere has loose tfloat32 numerics
        atol, rtol = 5e-3, 5e-3
    else:
        # use float16 for tolerance because tf32 has the same precision
        atol, rtol = get_tolerance(torch.float16)
    assert_close(C, ref, atol=atol, rtol=rtol)


@pytest.mark.parametrize("tile_size", [(2, 2, 1)])
@pytest.mark.parametrize("case", int_cases, ids=str)
def test_mma_int(tile_size, case):
    m, n, k = tile_size
    A = torch.randint(32, (m, k), dtype=case.dtype, device="cuda")
    B = torch.randint(32, (k, n), dtype=case.dtype, device="cuda")
    C = torch.ones((m, n), dtype=case.acc_dtype, device="cuda")
    ref = C + (A.to(torch.float32) @ B.to(torch.float32)).to(C.dtype)
    ct.launch(torch.cuda.current_stream(), (1,), mma_kernel,
              (A, B, C, m, n, k))
    assert_equal(C, ref)


@pytest.mark.parametrize("tile_size", [(2, 2, 1)])
def test_mma_mixed_int_uint(tile_size):
    m, n, k = tile_size
    A = torch.randint(32, (m, k), dtype=torch.int8, device="cuda")
    B = torch.randint(32, (k, n), dtype=torch.uint8, device="cuda")
    C = torch.ones((m, n), dtype=torch.int32, device="cuda")
    ref = C + (A.to(torch.float32) @ B.to(torch.float32)).to(C.dtype)
    ct.launch(torch.cuda.current_stream(), (1,), mma_kernel,
              (A, B, C, m, n, k))
    assert_equal(C, ref)


@ct.kernel
def mma_batch_kernel(A, B, C,
                     tb: ct.Constant[int],
                     tm: ct.Constant[int],
                     tn: ct.Constant[int],
                     tk: ct.Constant[int]):
    tx = ct.load(A, index=(0, 0), shape=(tm, tk))
    ty = ct.load(B, index=(0, 0, 0), shape=(tb, tk, tn))
    acc = ct.load(C, index=(0, 0, 0), shape=(tb, tm, tn))
    acc = ct.mma(tx, ty, acc)
    ct.store(C, index=(0, 0, 0), tile=acc)


def test_batch_mma():
    b, m, n, k = 2, 4, 8, 16
    dtype = torch.float32
    A = torch.randn((m, k), device="cuda").to(dtype)
    B = torch.randn((b, k, n), device="cuda").to(dtype)
    C = torch.ones((b, m, n), device="cuda").to(dtype)
    ref = A @ B + C
    ct.launch(torch.cuda.current_stream(), (1,), mma_batch_kernel,
              (A, B, C, b, m, n, k))
    atol, rtol = get_tolerance(A.dtype)
    assert_close(C, ref, atol=atol, rtol=rtol)


@dataclass
class DtypeErrorTestCase:
    x_dtype: torch.dtype
    y_dtype: torch.dtype
    acc_dtype: torch.dtype
    message: str


DTC = DtypeErrorTestCase
dtype_error_cases = [
    DTC(f16, bf16, f32, "x and y must have the same dtype"),
    DTC(i16, i16, f32, "Unsupported input dtype"),
    DTC(bf16, bf16, f16, "Unsupported acc dtype"),
]


@pytest.mark.parametrize("case", dtype_error_cases, ids=str)
def test_mma_dtype_error(case):
    A = torch.randn((2, 2), device='cuda').to(case.x_dtype)
    B = torch.randn((2, 2), device='cuda').to(case.y_dtype)
    C = torch.randn((2, 2), device='cuda').to(case.acc_dtype)
    with pytest.raises(TileTypeError, match=case.message):
        ct.launch(torch.cuda.current_stream(),
                  (1,), mma_kernel,
                  (A, B, C, 2, 2, 2))

# ================ ct.matmul =================


@ct.kernel
def matmul_kernel(A, B, C,
                  tm: ct.Constant[int],
                  tn: ct.Constant[int],
                  tk: ct.Constant[int]):
    tx = ct.load(A, index=(0, 0), shape=(tm, tk))
    ty = ct.load(B, index=(0, 0), shape=(tk, tn))
    acc = ct.matmul(tx, ty)
    ct.store(C, index=(0, 0), tile=acc)


unsupported_promotion = [(f16, bf16), (bf16, f16)]


@pytest.mark.parametrize("tile_size", [(2, 8, 16)])
@pytest.mark.parametrize("x_dtype", [bf16, f16, f32, f64], ids=dtype_id)
@pytest.mark.parametrize("y_dtype", [bf16, f16, f32, f64], ids=dtype_id)
def test_matmul(tile_size, x_dtype, y_dtype):
    m, n, k = tile_size
    acc_dtype = torch.promote_types(x_dtype, y_dtype)
    A = torch.randn((m, k), dtype=x_dtype, device="cuda")
    B = torch.randn((k, n), dtype=y_dtype, device="cuda")
    C = torch.zeros((m, n), dtype=acc_dtype, device="cuda")
    if (x_dtype, y_dtype) in unsupported_promotion:
        with pytest.raises(TileTypeError, match="Implicit promotion of .* and .* is not supported"):
            ct.launch(torch.cuda.current_stream(), (1,), matmul_kernel,
                      (A, B, C, m, n, k))
    else:
        ref = A.to(acc_dtype) @ B.to(acc_dtype)
        ct.launch(torch.cuda.current_stream(), (1,), matmul_kernel,
                  (A, B, C, m, n, k))
        atol, rtol = get_tolerance(A.dtype)
        assert_close(C, ref, atol=atol, rtol=rtol)


@pytest.mark.skipif(is_ampere_or_ada(), reason="float8 not supported on Ampere or Ada")
@pytest.mark.parametrize("tile_size", [(16, 16, 16)])
@pytest.mark.parametrize("dtype", [f8e4m3fn, f8e5m2], ids=dtype_id)
def test_matmul_fp8(tile_size, dtype):
    m, n, k = tile_size
    A = torch.randn((m, k), device="cuda").to(dtype)
    B = torch.randn((n, k), device="cuda").to(dtype)
    C = torch.zeros((m, n), dtype=dtype, device="cuda")
    scale = torch.tensor([1.0], dtype=torch.float32, device="cuda")
    try:
        ref = torch._scaled_mm(A, B.T, scale, scale, out_dtype=torch.float16).to(dtype)
    except (RuntimeError, ValueError) as e:
        assert 'Multiplication of two Float8_e5m2 matrices is not supported' in str(e)
        ref = None
    ct.launch(torch.cuda.current_stream(), (1,), matmul_kernel,
              (A, B.T, C, m, n, k))
    if ref is not None:
        atol, rtol = get_tolerance(A.dtype)
        assert_close(C.to(torch.float16),
                     ref.to(torch.float16),
                     atol=atol, rtol=rtol)


@pytest.mark.parametrize("tile_size", [(16, 16, 16)])
@pytest.mark.parametrize("dtype", [u8, i8], ids=dtype_id)
def test_matmul_int(tile_size, dtype):
    m, n, k = tile_size
    A = torch.randint(32, (m, k), dtype=dtype, device="cuda")
    B = torch.randint(32, (k, n), dtype=dtype, device="cuda")
    C = torch.zeros((m, n), dtype=dtype, device="cuda")
    ref = (A.cpu() @ B.cpu()).cuda()
    ct.launch(torch.cuda.current_stream(), (1,), matmul_kernel,
              (A, B, C, m, n, k))
    assert_equal(C, ref)


dtype_error_cases = [
    DTC(f16, bf16, f32, "Implicit promotion of float16 and bfloat16 is not supported"),
    DTC(f8e4m3fn, f16, f16, "Implicit promotion of float8_e4m3fn and float16 is not supported"),
    DTC(u8, i8, i32, "Implicit promotion of uint8 and int8 is not supported"),
    DTC(i32, i32, i32, "Unsupported input dtype"),
    DTC(i16, i16, i16, "Unsupported input dtype"),
]


@pytest.mark.parametrize("case", dtype_error_cases, ids=str)
def test_matmul_dtype_error(case):
    A = torch.randn((2, 2), device='cuda').to(case.x_dtype)
    B = torch.randn((2, 2), device='cuda').to(case.y_dtype)
    C = torch.randn((2, 2), device='cuda').to(case.acc_dtype)
    with pytest.raises(TileTypeError, match=case.message):
        ct.launch(torch.cuda.current_stream(),
                  (1,), matmul_kernel,
                  (A, B, C, 2, 2, 2))


@ct.kernel
def matmul_nd_kernel(A, B, C,
                     tb: ct.Constant[int],
                     tm: ct.Constant[int],
                     tn: ct.Constant[int],
                     tk: ct.Constant[int]):

    if A.ndim == 1 and B.ndim == 1:
        tx = ct.load(A, index=(0,), shape=(tk,))
        ty = ct.load(B, index=(0,), shape=(tk,))
        acc = ct.matmul(tx, ty)
        ct.store(C, index=(0,), tile=acc)
    if A.ndim == 1 and B.ndim == 2:
        tx = ct.load(A, index=(0,), shape=(tk,))
        ty = ct.load(B, index=(0, 0), shape=(tk, tn))
        acc = ct.matmul(tx, ty)
        ct.store(C, index=(0,), tile=acc)
    if A.ndim == 1 and B.ndim == 3:
        tx = ct.load(A, index=(0,), shape=(tk,))
        ty = ct.load(B, index=(0, 0, 0), shape=(tb, tk, tn))
        acc = ct.matmul(tx, ty)
        ct.store(C, index=(0, 0), tile=acc)
    if A.ndim == 2 and B.ndim == 2:
        tx = ct.load(A, index=(0, 0), shape=(tm, tk))
        ty = ct.load(B, index=(0, 0), shape=(tk, tn))
        acc = ct.matmul(tx, ty)
        ct.store(C, index=(0, 0), tile=acc)
    if A.ndim == 2 and B.ndim == 3:
        tx = ct.load(A, index=(0, 0), shape=(tm, tk))
        ty = ct.load(B, index=(0, 0, 0), shape=(tb, tk, tn))
        acc = ct.matmul(tx, ty)
        ct.store(C, index=(0, 0, 0), tile=acc)
    if A.ndim == 3 and B.ndim == 3:
        tx = ct.load(A, index=(0, 0, 0), shape=(tb, tm, tk))
        ty = ct.load(B, index=(0, 0, 0), shape=(tb, tk, tn))
        acc = ct.matmul(tx, ty)
        ct.store(C, index=(0, 0, 0), tile=acc)


def _get_shape(rank, batch, m_or_n, k, transpose: bool):
    if rank == 1:
        return (k,)
    elif rank == 2:
        return (m_or_n, k) if not transpose else (k, m_or_n)
    elif rank == 3:
        return (batch, m_or_n, k) if not transpose else (batch, k, m_or_n)
    raise NotImplementedError()


@pytest.mark.parametrize("ranks", [(1, 1),
                                   (1, 2),
                                   (1, 3),
                                   (2, 2),
                                   (2, 3)])
def test_matmul_nd(ranks):
    b, m, n, k = 2, 4, 8, 16
    dtype = torch.float32
    a_shape = _get_shape(ranks[0], b, m, k, transpose=False)
    b_shape = _get_shape(ranks[1], b, n, k, transpose=True)
    A = torch.randn(a_shape, device="cuda").to(dtype)
    B = torch.randn(b_shape, device="cuda").to(dtype)
    ref = A @ B
    if len(ref.shape) == 0:
        # WAR: tileir doesn't support store in to 0d array
        ref.unsqueeze_(0)
    C = torch.zeros(ref.shape, device="cuda").to(dtype)
    ct.launch(torch.cuda.current_stream(), (1,), matmul_nd_kernel,
              (A, B, C, b, m, n, k))
    atol, rtol = get_tolerance(A.dtype)
    assert_close(C, ref, atol=atol, rtol=rtol)


@pytest.mark.parametrize("dtype", [f8e4m3fn, f8e5m2], ids=dtype_id)
def test_ampere_fp8_error(dtype):
    A = torch.randn((16, 16), device="cuda").to(dtype)
    B = torch.randn((16, 16), device="cuda").to(dtype)
    C = torch.zeros((16, 16), dtype=torch.float16, device="cuda")
    with patch("cuda.tile._compile.get_sm_arch", return_value="sm_80"):
        with pytest.raises(TileUnsupportedFeatureError,
                           match="float8 dtype is not supported on Ampere"):
            ct.launch(torch.cuda.current_stream(), (1,), mma_kernel,
                      (A, B, C, 16, 16, 16))
