# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
import enum
import re

import pytest
import torch
from cuda.tile._cext import cconv_v3_enabled


# example-begin imports
import cuda.tile as ct
# example-end imports


# example-begin constant
def needs_constant(x: ct.Constant):
    pass


def needs_constant_int(x: ct.Constant[int]):
    pass
# example-end constant


# TODO: Run with `mypy --check-untyped-defs` or another static type checker.
def test_constant_type_hints() -> None:
    int_constant: ct.Constant[int] = 42
    float_constant: ct.Constant[float] = 3.14

    needs_constant(int_constant)
    needs_constant(float_constant)
    needs_constant_int(int_constant)
    needs_constant_int(float_constant)  # Should fail type checking


class MyEnum(enum.Enum):
    FOO = 100
    BAR = 200


@pytest.mark.skipif(not cconv_v3_enabled(), reason="Requires cconv3 enabled")
def test_enum_kernel_argument():
    @ct.kernel
    def kern(c: ct.Constant, out):
        ct.scatter(out, 0, c == MyEnum.FOO)
        ct.scatter(out, 1, c == MyEnum.BAR)

    out = torch.full((2,), -1, dtype=torch.int32, device="cuda")
    ct.launch(torch.cuda.current_stream(), (1,), kern, (MyEnum.FOO, out))
    assert out.tolist() == [1, 0]

    out = torch.full((2,), -1, dtype=torch.int32, device="cuda")
    ct.launch(torch.cuda.current_stream(), (1,), kern, (MyEnum.BAR, out))
    assert out.tolist() == [0, 1]


@pytest.mark.skipif(not cconv_v3_enabled(), reason="Requires cconv3 enabled")
def test_dtype_kernel_argument():
    @ct.kernel
    def kern(c: ct.Constant, out):
        ct.scatter(out, 0, c == ct.float32)
        ct.scatter(out, 1, c == ct.int32)

    out = torch.full((2,), -1, dtype=torch.int32, device="cuda")
    ct.launch(torch.cuda.current_stream(), (1,), kern, (ct.float32, out))
    assert out.tolist() == [1, 0]

    out = torch.full((2,), -1, dtype=torch.int32, device="cuda")
    ct.launch(torch.cuda.current_stream(), (1,), kern, (torch.float32, out))
    assert out.tolist() == [1, 0]

    out = torch.full((2,), -1, dtype=torch.int32, device="cuda")
    ct.launch(torch.cuda.current_stream(), (1,), kern, (ct.int32, out))
    assert out.tolist() == [0, 1]

    out = torch.full((2,), -1, dtype=torch.int32, device="cuda")
    ct.launch(torch.cuda.current_stream(), (1,), kern, (torch.int32, out))
    assert out.tolist() == [0, 1]


@pytest.mark.skipif(not cconv_v3_enabled(), reason="Requires cconv3 enabled")
def test_none_kernel_argument():
    @ct.kernel
    def kern(c: ct.Constant, out):
        ct.scatter(out, (), c is None)

    out = torch.full((), -1, dtype=torch.int32, device="cuda")
    ct.launch(torch.cuda.current_stream(), (1,), kern, (None, out))
    assert out.item() == 1

    out = torch.full((), -1, dtype=torch.int32, device="cuda")
    ct.launch(torch.cuda.current_stream(), (1,), kern, (123, out))
    assert out.tolist() == 0


@pytest.mark.skipif(not cconv_v3_enabled(), reason="Requires cconv3 enabled")
def test_str_kernel_argument():
    @ct.kernel
    def kern(c: ct.Constant, out):
        ct.scatter(out, 0, c == "hello")
        ct.scatter(out, 1, c == "test string for test_str_kernel_argument!")

    # Repeat a few times to make sure we exercise different branches that depend on string interning
    for i in range(3):
        out = torch.full((2,), -1, dtype=torch.int32, device="cuda")
        ct.launch(torch.cuda.current_stream(), (1,), kern, ("hello", out))
        assert out.tolist() == [1, 0]

        out = torch.full((2,), -1, dtype=torch.int32, device="cuda")
        ct.launch(torch.cuda.current_stream(), (1,), kern,
                  ("test string for test_str_kernel_argument!", out))
        assert out.tolist() == [0, 1]


@pytest.mark.skipif(not cconv_v3_enabled(), reason="Requires cconv3 enabled")
def test_str_passed_for_nonconstant_param():
    @ct.kernel
    def kern(x):
        print(x)

    with pytest.raises(TypeError,
                       match=re.escape("Invalid kernel argument #0: Objects of type 'str' are only"
                                       " accepted for parameters annotated as Constant.")):
        ct.launch(torch.cuda.current_stream(), (1,), kern, ("hello",))
