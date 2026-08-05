# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
import enum

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
