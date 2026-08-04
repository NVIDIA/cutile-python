# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from io import BytesIO

import pytest

import cuda.tile as ct
from cuda.tile._cext import cconv_v3_enabled
from cuda.tile.compilation import (
    CallingConvention, KernelSignature, demangle_kernel_name, export_kernel)


@pytest.mark.skipif(cconv_v3_enabled(), reason="Requires cconv v3 disabled")
def test_development_calling_convention_is_rejected():
    with pytest.raises(AttributeError, match="has no attribute 'cutile_python_v3'"):
        CallingConvention.cutile_python_v3()
    with pytest.raises(ValueError, match="Unknown calling convention code 't3'"):
        CallingConvention.from_code("t3")
    with pytest.raises(ValueError, match="Unknown calling convention code 't3'"):
        demangle_kernel_name("kernel_Kt3")


@pytest.mark.skipif(not cconv_v3_enabled(), reason="Requires cconv v3 enabled")
def test_development_calling_convention():
    cconv = CallingConvention.cutile_python_v3()
    assert cconv.name == "cutile_python_v3"
    assert cconv.code == "t3"
    assert cconv.version == 3
    assert CallingConvention.from_code("t3") == cconv


@pytest.mark.skipif(not cconv_v3_enabled(), reason="Requires cconv v3 enabled")
def test_aot_export_with_development_calling_convention():
    @ct.kernel
    def kernel():
        pass

    output = BytesIO()
    export_kernel(kernel, [KernelSignature([], CallingConvention.cutile_python_v3())], output,
                  gpu_code="sm_100", output_format="tileir_bytecode")
    assert output.getvalue()
