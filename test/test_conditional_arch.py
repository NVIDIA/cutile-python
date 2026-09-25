# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from io import BytesIO

import pytest

import cuda.tile as ct
from cuda.tile._cext import CallingConvention
from cuda.tile._exception import TileUnsupportedFeatureError


def compile_bytecode(pyfunc, arch, version):
    kernel = ct.kernel(pyfunc)
    sig = ct.compilation.KernelSignature(
        [], CallingConvention.cutile_python_v1(), symbol="kernel")
    output = BytesIO()
    ct.compilation.export_kernel(kernel, [sig], output_file=output, gpu_code=arch,
                                 output_format="tileir_bytecode", bytecode_version=version)
    assert output.getvalue()


@pytest.mark.parametrize("arch", ["sm_90", "sm_90a", "sm_100a", "sm_100f", "sm_120a", "sm_121a"])
def test_conditional_arch_dtype_check(arch):
    def kernel():
        t = ct.full((2,), 1.5, dtype=ct.float32)
        ct.printf("%f", t)

    compile_bytecode(kernel, arch, "13.2")


@pytest.mark.parametrize("arch", ["sm_90a", "sm_100f"])
def test_conditional_arch_preserves_dtype_limits(arch):
    def kernel():
        t = ct.full((2,), 1.5, dtype=ct.float8_e5m3fnu)
        ct.printf("%f", t)

    with pytest.raises(TileUnsupportedFeatureError, match=f"is not supported on {arch}"):
        compile_bytecode(kernel, arch, "13.4")


@pytest.mark.parametrize("arch", ["sm_100a", "sm_100f"])
def test_conditional_arch_preserves_bytecode_limits(arch):
    def kernel():
        t = ct.full((2,), 1.5, dtype=ct.float4_e2m1fn)
        ct.printf("%f", t)

    with pytest.raises(TileUnsupportedFeatureError,
                       match=r"float4_e2m1fn requires tileiras 13\.3"):
        compile_bytecode(kernel, arch, "13.2")


@pytest.mark.parametrize("arch", ["sm_100aa", "sm_100af", "sm_100ff", "sm_100x"])
def test_conditional_arch_rejects_invalid_suffix(arch):
    def kernel():
        ct.printf("%d", ct.bid(0))

    with pytest.raises(ValueError, match="invalid literal for int"):
        compile_bytecode(kernel, arch, "13.2")
