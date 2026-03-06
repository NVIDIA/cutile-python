# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from torch.testing import make_tensor
from unittest.mock import patch

import cuda.tile as ct
from cuda.tile._bytecode.version import BytecodeVersion
from cuda.tile._compile import compile_tile
from cuda.tile._compiler_options import CompilerOptions
from cuda.tile._exception import TileUnsupportedFeatureError, TileValueError
from conftest import dtype_id

# TODO: remove when feature is out of development only
from cuda.tile._datatype import float8_e8m0fnu, float4_e2m1fn
ct.float8_e8m0fnu = float8_e8m0fnu
ct.float4_e2m1fn = float4_e2m1fn


def compile(pyfunc, args):
    return compile_tile(pyfunc, args, CompilerOptions())


def compile_with(pyfunc, args, arch: str, version: BytecodeVersion):
    with patch('cuda.tile._compile.get_sm_arch', return_value=arch):
        with patch('cuda.tile._compile._get_max_supported_bytecode_version',
                   return_value=version):
            return compile_tile(pyfunc, args, CompilerOptions())


@pytest.mark.parametrize("dtype", [
    torch.float8_e4m3fn,
    torch.float8_e5m2,
    torch.float8_e8m0fnu
], ids=dtype_id)
def test_fp8_not_supported_on_sm80(dtype):
    x = make_tensor((64,), dtype=torch.float32, device='cuda').to(dtype)

    def kernel(x):
        tx = ct.gather(x, 0)
        ct.scatter(x, 0, tx)

    with pytest.raises(TileUnsupportedFeatureError, match="is not supported on sm_80"):
        compile_with(kernel, (x,), "sm_80", BytecodeVersion.V_13_2)


@pytest.mark.parametrize("arch", ["sm_80", "sm_90"])
def test_fp4_not_supported_on_arch(arch):
    def kernel():
        t = ct.full((2,), 1.5, dtype=ct.float4_e2m1fn)
        ct.printf("%f", t)

    with pytest.raises(TileUnsupportedFeatureError, match=f"is not supported on {arch}"):
        compile_with(kernel, (), arch, BytecodeVersion.V_13_3)


def test_f8e8m0fnu_requires_13_2():
    def kernel(x, y):
        tx = ct.gather(x, 0)
        ct.scatter(y, 0, tx)

    with pytest.raises(TileUnsupportedFeatureError,
                       match=r"float8_e8m0fnu requires tileiras 13\.2"):
        x = make_tensor((1,), dtype=torch.uint8, device='cuda').view(torch.float8_e8m0fnu)
        y = torch.zeros_like(x)
        compile_with(kernel, (x, y), "sm_100", BytecodeVersion.V_13_1)


def test_f4e2m1fn_requires_13_3():
    def kernel(x):
        t = ct.full((2,), 1.5, dtype=ct.float4_e2m1fn)
        ct.store(x, 0, tile=t.astype(ct.uint8))

    x = make_tensor((1,), dtype=torch.uint8, device='cuda')
    with pytest.raises(TileUnsupportedFeatureError,
                       match=r"float4_e2m1fn requires tileiras 13\.3"):
        compile_with(kernel, (x,), "sm_100",  BytecodeVersion.V_13_2)


@pytest.mark.parametrize("val", [-1.0, -0.0, float("-inf"), float("-nan")])
def test_f8e8m0fnu_rejects_negative(val):
    def kernel():
        t = ct.full((2,), val, dtype=ct.float8_e8m0fnu)
        ct.printf("%f", t)

    with pytest.raises(TileValueError,
                       match="negative values cannot be represented in float8_e8m0fnu"):
        compile_with(kernel, (), "sm_100", BytecodeVersion.V_13_2)


@pytest.mark.parametrize("val", [float("nan"), float("-nan")])
def test_f4e2m1fn_rejects_nan(val):
    def kernel():
        t = ct.full((2,), val, dtype=ct.float4_e2m1fn)
        ct.printf("%f", t)

    with pytest.raises(TileValueError,
                       match="NaN cannot be represented in float4_e2m1fn"):
        compile_with(kernel, (), "sm_100", BytecodeVersion.V_13_3)
