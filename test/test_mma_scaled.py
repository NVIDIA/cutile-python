# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import Tuple
import pytest
import torch

import cuda.tile as ct
from cuda.tile._datatype import float4_e2m1fn, float8_e4m3fn, float8_e5m2, float8_e8m0fnu
from cuda.tile._exception import TileTypeError
from cuda.tile._bytecode.version import BytecodeVersion
from conftest import requires_tileiras
from util import assert_close, require_blackwell_or_newer, require_rubin_cc107
from cuda.tile._bytecode import SimpleType
from cuda.tile._bytecode.float import float_from_bits

pytestmark = [require_blackwell_or_newer(), requires_tileiras(BytecodeVersion.V_13_3)]

f8e4m3fn = torch.float8_e4m3fn
f8e5m2 = torch.float8_e5m2
f8e8m0fnu = torch.float8_e8m0fnu
f32 = torch.float32
f16 = torch.float16
bf16 = torch.bfloat16


@ct.kernel
def mma_scaled_f8_kernel(X, X_scale, Y, Y_scale, Z,
                         tm: ct.Constant[int], tn: ct.Constant[int],
                         tk: ct.Constant[int], tks: ct.Constant[int]):
    x = ct.load(X, index=(0, 0), shape=(tm, tk))
    x_scale = ct.load(X_scale, index=(0, 0), shape=(tm, tks))
    y = ct.load(Y, index=(0, 0), shape=(tk, tn))
    y_scale = ct.load(Y_scale, index=(0, 0), shape=(tks, tn))
    acc = ct.load(Z, index=(0, 0), shape=(tm, tn))
    acc = ct.mma_scaled(x, x_scale, y, y_scale, acc)
    ct.store(Z, index=(0, 0), tile=acc)


@ct.kernel
def batch_mma_scaled_fp8_kernel(X, X_scale, Y, Y_scale, Z,
                                tb: ct.Constant[int],
                                tm: ct.Constant[int], tn: ct.Constant[int],
                                tk: ct.Constant[int], tks: ct.Constant[int]):
    x = ct.load(X, index=(0, 0), shape=(tm, tk))
    x_scale = ct.load(X_scale, index=(0, 0), shape=(tm, tks))
    y = ct.load(Y, index=(0, 0, 0), shape=(tb, tk, tn))
    y_scale = ct.load(Y_scale, index=(0, 0, 0), shape=(tb, tks, tn))
    acc = ct.load(Z, index=(0, 0, 0), shape=(tb, tm, tn))
    acc = ct.mma_scaled(x, x_scale, y, y_scale, acc)
    ct.store(Z, index=(0, 0, 0), tile=acc)


@ct.kernel
def mma_scaled_general_kernel(X, X_scale, Y, Y_scale, Z,
                              tm: ct.Constant[int], tn: ct.Constant[int],
                              tk: ct.Constant[int], tks: ct.Constant[int],
                              dtype_id: ct.Constant[int],
                              elem_per_byte: ct.Constant[int],
                              packed_scale: ct.Constant[bool]):
    dtype = (float4_e2m1fn, float8_e4m3fn, float8_e5m2)[dtype_id]

    x_bytes = ct.load(X, index=(0, 0), shape=(tm, tk // elem_per_byte)).reshape((-1,))
    x = ct.unpack_from_bytes(x_bytes, dtype).reshape((tm, tk))
    x_scale = ct.load(X_scale, index=(0, 0), shape=(tm, tks))
    if packed_scale:
        x_scale = ct.unpack_from_bytes(x_scale.reshape((-1,)), ct.float8_e5m3fnu).reshape((tm, tks))

    y_bytes = ct.load(Y, index=(0, 0), shape=(tk, tn // elem_per_byte)).reshape((-1,))
    y = ct.unpack_from_bytes(y_bytes, dtype).reshape((tk, tn))
    y_scale = ct.load(Y_scale, index=(0, 0), shape=(tks, tn))
    if packed_scale:
        y_scale = ct.unpack_from_bytes(y_scale.reshape((-1,)), ct.float8_e5m3fnu).reshape((tks, tn))

    acc = ct.load(Z, index=(0, 0), shape=(tm, tn))
    acc = ct.mma_scaled(x, x_scale, y, y_scale, acc)
    ct.store(Z, index=(0, 0), tile=acc)


@pytest.mark.parametrize("input_dtype", [f8e4m3fn, f8e5m2], ids=str)
def test_mma_scaled_fp8(input_dtype):
    m, n, k = 16, 16, 64
    scaling_block_size = 32
    ks = k // scaling_block_size

    X = torch.randn((m, k), device='cuda').to(input_dtype)
    X_scale = torch.randn((m, ks), device='cuda').to(f8e8m0fnu)
    Y = torch.randn((k, n), device='cuda').to(input_dtype)
    Y_scale = torch.randn((ks, n), device='cuda').to(f8e8m0fnu)
    Z = torch.randn((m, n), dtype=f32, device='cuda')

    ref_X_scale = torch.repeat_interleave(X_scale, scaling_block_size, dim=1).to(f32)
    ref_Y_scale = torch.repeat_interleave(Y_scale, scaling_block_size, dim=0).to(f32)
    ref = (X.to(f32) * ref_X_scale) @ (Y.to(f32) * ref_Y_scale) + Z

    ct.launch(torch.cuda.current_stream(), (1,), mma_scaled_f8_kernel,
              (X, X_scale, Y, Y_scale, Z, m, n, k, ks))
    assert_close(Z, ref, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("input_dtype,dtype_id", [(f8e4m3fn, 1), (f8e5m2, 2),], ids=str)
def test_mma_scaled_general_fp8(input_dtype, dtype_id):
    """Test fp8 through the general unpack_from_bytes kernel path."""
    m, n, k = 16, 16, 64
    scaling_block_size = 32
    ks = k // scaling_block_size

    X_raw = torch.randint(0, 256, (m, k), dtype=torch.uint8, device='cuda')
    Y_raw = torch.randint(0, 256, (k, n), dtype=torch.uint8, device='cuda')
    X_scale = torch.randn((m, ks), device='cuda').to(f8e8m0fnu)
    Y_scale = torch.randn((ks, n), device='cuda').to(f8e8m0fnu)
    Z = torch.randn((m, n), dtype=f32, device='cuda')

    ref_X = X_raw.view(input_dtype).to(f32)
    ref_Y = Y_raw.view(input_dtype).to(f32)
    ref_X_scale = torch.repeat_interleave(X_scale, scaling_block_size, dim=1).to(f32)
    ref_Y_scale = torch.repeat_interleave(Y_scale, scaling_block_size, dim=0).to(f32)
    ref = (ref_X * ref_X_scale) @ (ref_Y * ref_Y_scale) + Z

    ct.launch(torch.cuda.current_stream(), (1,), mma_scaled_general_kernel,
              (X_raw, X_scale, Y_raw, Y_scale, Z,
               m, n, k, ks, dtype_id, 1, False))
    assert_close(Z, ref, atol=1e-2, rtol=1e-2)


def test_batch_mma_scaled_fp8():
    b, m, n, k = 2, 16, 16, 128
    scaling_block_size = 32
    ks = k // scaling_block_size

    X = torch.randn((m, k), device='cuda').to(f8e4m3fn)
    X_scale = torch.randn((m, ks), device='cuda').to(f8e8m0fnu)
    Y = torch.randn((b, k, n), device='cuda').to(f8e4m3fn)
    Y_scale = torch.randn((b, ks, n), device='cuda').to(f8e8m0fnu)
    Z = torch.randn((b, m, n), dtype=f32, device='cuda')

    ref_X_scale = torch.repeat_interleave(X_scale, scaling_block_size, dim=-1).to(f32)
    ref_Y_scale = torch.repeat_interleave(Y_scale, scaling_block_size, dim=-2).to(f32)
    ref = (X.to(f32) * ref_X_scale) @ (Y.to(f32) * ref_Y_scale) + Z

    ct.launch(torch.cuda.current_stream(), (1,), batch_mma_scaled_fp8_kernel,
              (X, X_scale, Y, Y_scale, Z, b, m, n, k, ks))
    assert_close(Z, ref, atol=1e-2, rtol=1e-2)


# float4_e2m1fn lookup table: nibble -> float value
_F4_TABLE = torch.tensor([
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0
], dtype=torch.float32)


def _unpack_f4e2m1fn_to_f32(packed: torch.Tensor, shape: Tuple) -> torch.Tensor:
    table = _F4_TABLE.to(packed.device)
    low = (packed & 0x0F).long()
    high = ((packed >> 4) & 0x0F).long()
    vals = torch.stack([table[low], table[high]], dim=-1).reshape(-1)
    return vals.reshape(shape)


@pytest.mark.parametrize("scale_dtype,scaling_block_size", [
    (f8e8m0fnu, 32), (f8e8m0fnu, 16), (f8e4m3fn, 16),
], ids=str)
def test_mma_scaled_general_f4(scale_dtype, scaling_block_size):
    m, n, k = 16, 16, 64
    ks = k // scaling_block_size

    X_packed = torch.randint(0, 256, (m, k // 2), dtype=torch.uint8, device='cuda')
    Y_packed = torch.randint(0, 256, (k, n // 2), dtype=torch.uint8, device='cuda')
    X_scale = torch.randn((m, ks), device='cuda').to(scale_dtype)
    Y_scale = torch.randn((ks, n), device='cuda').to(scale_dtype)
    Z = torch.randn((m, n), dtype=f32, device='cuda')

    ref_X = _unpack_f4e2m1fn_to_f32(X_packed, (m, k))
    ref_Y = _unpack_f4e2m1fn_to_f32(Y_packed, (k, n))
    ref_X_scale = torch.repeat_interleave(X_scale, scaling_block_size, dim=1).to(f32)
    ref_Y_scale = torch.repeat_interleave(Y_scale, scaling_block_size, dim=0).to(f32)
    ref = (ref_X * ref_X_scale) @ (ref_Y * ref_Y_scale) + Z
    ct.launch(torch.cuda.current_stream(), (1,), mma_scaled_general_kernel,
              (X_packed, X_scale, Y_packed, Y_scale, Z, m, n, k, ks, 0, 2, False))
    assert_close(Z, ref, atol=1e-2, rtol=1e-2)


@require_rubin_cc107()
@requires_tileiras(BytecodeVersion.V_13_4)
@pytest.mark.parametrize("scaling_block_size", [
    32, 16,
], ids=str)
def test_mma_e5m3fnu_scaled_f4(scaling_block_size):
    m, n, k = 16, 16, 64
    ks = k // scaling_block_size

    X_packed = torch.randint(0, 256, (m, k // 2), dtype=torch.uint8, device='cuda')
    Y_packed = torch.randint(0, 256, (k, n // 2), dtype=torch.uint8, device='cuda')
    X_scale = torch.randint(0, 255, (m, ks), dtype=torch.uint8, device="cuda")
    Y_scale = torch.randint(0, 255, (ks, n), dtype=torch.uint8, device="cuda")

    Z = torch.randn((m, n), dtype=f32, device='cuda')

    ref_X = _unpack_f4e2m1fn_to_f32(X_packed, (m, k))
    ref_Y = _unpack_f4e2m1fn_to_f32(Y_packed, (k, n))

    ref_X_scale = torch.repeat_interleave(X_scale, scaling_block_size, dim=1)
    ref_X_scale = [float_from_bits(int(i), SimpleType.F8E5M3FNU) for i in ref_X_scale.view(-1)]
    ref_X_scale = torch.tensor(ref_X_scale, dtype=f32, device='cuda').reshape((m, k))

    ref_Y_scale = torch.repeat_interleave(Y_scale, scaling_block_size, dim=0)
    ref_Y_scale = [float_from_bits(int(i), SimpleType.F8E5M3FNU) for i in ref_Y_scale.view(-1)]
    ref_Y_scale = torch.tensor(ref_Y_scale, dtype=f32, device='cuda').reshape((k, n))

    ref = (ref_X * ref_X_scale) @ (ref_Y * ref_Y_scale) + Z
    ct.launch(torch.cuda.current_stream(), (1,), mma_scaled_general_kernel,
              (X_packed, X_scale, Y_packed, Y_scale, Z, m, n, k, ks, 0, 2, True))
    assert_close(Z, ref, atol=1e-2, rtol=1e-2)


@ct.kernel
def mma_scaled_error_kernel(X, X_scale, Y, Y_scale, Z,
                            tm: ct.Constant[int], tn: ct.Constant[int],
                            tk: ct.Constant[int],
                            tms: ct.Constant[int], tks: ct.Constant[int],
                            tns: ct.Constant[int]):
    x = ct.load(X, index=(0, 0), shape=(tm, tk))
    x_scale = ct.load(X_scale, index=(0, 0), shape=(tms, tks))
    y = ct.load(Y, index=(0, 0), shape=(tk, tn))
    y_scale = ct.load(Y_scale, index=(0, 0), shape=(tks, tns))
    acc = ct.load(Z, index=(0, 0), shape=(tm, tn))
    acc = ct.mma_scaled(x, x_scale, y, y_scale, acc)
    ct.store(Z, index=(0, 0), tile=acc)


@dataclass
class DtypeErrorCase:
    x_dtype: torch.dtype
    y_dtype: torch.dtype
    x_scale_dtype: torch.dtype
    y_scale_dtype: torch.dtype
    acc_dtype: torch.dtype
    message: str

    def __str__(self):
        dtypes = (self.x_dtype, self.y_dtype, self.x_scale_dtype,
                  self.y_scale_dtype, self.acc_dtype)
        return "-".join(str(d).removeprefix("torch.") for d in dtypes)


DTC = DtypeErrorCase
dtype_error_cases = [
    DTC(f8e4m3fn, f8e5m2, f8e8m0fnu, f8e8m0fnu, f32,
        "x and y must have the same dtype"),
    DTC(f8e4m3fn, f8e4m3fn, f8e8m0fnu, f8e4m3fn, f32,
        "x_scale and y_scale must have the same dtype"),
    DTC(f16, f16, f8e8m0fnu, f8e8m0fnu, f32,
        "Unsupported input dtype"),
    DTC(f8e4m3fn, f8e4m3fn, bf16, bf16, f32,
        "Unsupported scale dtype"),
    DTC(f8e4m3fn, f8e4m3fn, f8e8m0fnu, f8e8m0fnu, f16,
        "Unsupported acc dtype"),
]


@pytest.mark.parametrize("case", dtype_error_cases, ids=str)
def test_mma_scaled_dtype_error(case):
    m, n, k, ks = 16, 16, 64, 2
    X = torch.randn((m, k), device='cuda').to(case.x_dtype)
    X_scale = torch.randn((m, ks), device='cuda').to(case.x_scale_dtype)
    Y = torch.randn((k, n), device='cuda').to(case.y_dtype)
    Y_scale = torch.randn((ks, n), device='cuda').to(case.y_scale_dtype)
    Z = torch.zeros((m, n), device='cuda').to(case.acc_dtype)
    with pytest.raises(TileTypeError, match=case.message):
        ct.launch(torch.cuda.current_stream(), (1,), mma_scaled_error_kernel,
                  (X, X_scale, Y, Y_scale, Z, m, n, k, m, ks, n))


@dataclass
class ShapeErrorCase:
    tm: int
    tn: int
    tk: int
    tms: int
    tns: int
    tks: int
    message: str

    def __str__(self):
        return f"{self.tm}-{self.tn}-{self.tk}-{self.tms}-{self.tns}-{self.tks}"


STC = ShapeErrorCase
shape_error_cases = [
    # x_scale M != x M
    STC(16, 16, 64, 8, 16, 2,
        "x_scale shape .* is not compatible with x shape"),
    # y_scale N != y N
    STC(16, 16, 64, 16, 8, 2,
        "y_scale shape .* is not compatible with y shape"),
    # K not divisible by K_scale
    STC(16, 16, 16, 16, 16, 32,
        r"x\.shape\[1\] must be an exact multiple of x_scale\.shape\[1\] with scaling block size B = K // K_s in \{32\}, got x\.shape\[1\] = 16 and x_scale\.shape\[1\] = 32"),  # noqa
    # Invalid scaling block size (f8 requires 32, but 64/8=8 is not allowed)
    STC(16, 16, 64, 16, 16, 8,
        r"x\.shape\[1\] must be an exact multiple of x_scale\.shape\[1\] with scaling block size B = K // K_s in \{32\}, got x\.shape\[1\] = 64 and x_scale\.shape\[1\] = 8"),  # noqa
]


@pytest.mark.parametrize("case", shape_error_cases, ids=str)
def test_mma_scaled_shape_error(case):
    X = torch.randn((case.tm, case.tk), device='cuda').to(f8e4m3fn)
    X_scale = torch.randn((case.tms, case.tks), device='cuda').to(f8e8m0fnu)
    Y = torch.randn((case.tk, case.tn), device='cuda').to(f8e4m3fn)
    Y_scale = torch.randn((case.tks, case.tns), device='cuda').to(f8e8m0fnu)
    Z = torch.zeros((case.tm, case.tn), dtype=f32, device='cuda')
    with pytest.raises(TileTypeError, match=case.message):
        ct.launch(torch.cuda.current_stream(), (1,), mma_scaled_error_kernel,
                  (X, X_scale, Y, Y_scale, Z,
                   case.tm, case.tn, case.tk, case.tms, case.tks, case.tns))


def test_mma_scaled_1d_shape_error():
    @ct.kernel
    def mma_scaled_1d_error_kernel():
        x = ct.ones((16 * 64,), float8_e4m3fn)
        x_scale = ct.ones((16, 2), float8_e8m0fnu)
        y = ct.ones((64, 16), float8_e4m3fn)
        y_scale = ct.ones((2, 16), float8_e8m0fnu)
        acc = ct.zeros((16, 16), ct.float32)
        acc = ct.mma_scaled(x, x_scale, y, y_scale, acc)
        ct.printf("%f", acc)

    with pytest.raises(TileTypeError, match=r"Expect shape of `x` to be 2D or 3D"):
        ct.launch(torch.cuda.current_stream(), (1,), mma_scaled_1d_error_kernel, ())
