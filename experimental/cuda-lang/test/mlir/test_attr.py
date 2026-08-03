# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import cuda.lang._mlir as mlir
import pytest


def test_integer_attr_printing():
    assert str(mlir.IntegerAttr.make(mlir.IntegerType.signless(32), 123)) == "123 : i32"
    assert str(mlir.IntegerAttr.make(mlir.IntegerType.signless(32), -5)) == "-5 : i32"


def test_float_attr_printing():
    assert str(mlir.FloatAttr(type=mlir.Float32Type(), value=mlir.APFloat(1.25))) == "1.25 : f32"


@pytest.mark.parametrize(
    "float_type,value,expected",
    [
        (mlir.Float16Type(), float("inf"), "0x7C00 : f16"),
        (mlir.Float16Type(), -float("inf"), "0xFC00 : f16"),
        (mlir.Float16Type(), float("nan"), "0x7E00 : f16"),
        (mlir.BFloat16Type(), float("inf"), "0x7F80 : bf16"),
        (mlir.BFloat16Type(), -float("inf"), "0xFF80 : bf16"),
        (mlir.BFloat16Type(), float("nan"), "0x7FC0 : bf16"),
        (mlir.Float32Type(), float("inf"), "0x7F800000 : f32"),
        (mlir.Float32Type(), -float("inf"), "0xFF800000 : f32"),
        (mlir.Float32Type(), float("nan"), "0x7FC00000 : f32"),
        (mlir.Float64Type(), float("inf"), "0x7FF0000000000000 : f64"),
        (mlir.Float64Type(), -float("inf"), "0xFFF0000000000000 : f64"),
        (mlir.Float64Type(), float("nan"), "0x7FF8000000000000 : f64"),
    ],
)
def test_special_float_attr_printing(float_type, value, expected):
    assert str(mlir.FloatAttr(type=float_type, value=mlir.APFloat(value))) == expected


@pytest.mark.parametrize(
    "float_type,value,expected",
    [
        (mlir.Float16Type(), 2**-24, "0x0001 : f16"),
        (mlir.BFloat16Type(), 2**-133, "0x0001 : bf16"),
        (mlir.Float32Type(), 2**-149, "0x00000001 : f32"),
        (mlir.Float64Type(), 2**-1074, "0x0000000000000001 : f64"),
        (mlir.Float32Type(), 0.0, "0x00000000 : f32"),
        (mlir.Float32Type(), -0.0, "0x80000000 : f32"),
    ],
)
def test_non_normal_float_attr_printing(float_type, value, expected):
    assert str(mlir.FloatAttr(type=float_type, value=mlir.APFloat(value))) == expected


@pytest.mark.parametrize(
    "float_type,value,expected",
    [
        (mlir.Float16Type(), 2**-14, "6.103515625e-05 : f16"),
        (mlir.BFloat16Type(), 2**-126, "1.1754943508222875e-38 : bf16"),
        (mlir.Float32Type(), 2**-126, "1.1754943508222875e-38 : f32"),
        (mlir.Float64Type(), 2**-1022, "2.2250738585072014e-308 : f64"),
    ],
)
def test_min_normal_float_attr_printing(float_type, value, expected):
    assert str(mlir.FloatAttr(type=float_type, value=mlir.APFloat(value))) == expected


def test_string_attr_printing():
    assert str(mlir.StringAttr(value="")) == '""'
    assert str(mlir.StringAttr(value="Hello, world")) == '"Hello, world"'
    assert str(mlir.StringAttr(value='x"y\nz\\w')) == '"x\\22y\\0Az\\\\w"'


def test_dense_array_printing():
    assert str(mlir.DenseI32ArrayAttr([10, -20, 0, 30])) == "array<i32: 10, -20, 0, 30>"
    assert str(mlir.DenseI32ArrayAttr([])) == "array<i32>"


def test_type_attr_printing():
    assert str(mlir.TypeAttr(value=mlir.IntegerType.signed(16))) == "si16"


@pytest.mark.parametrize(
    "location,expected",
    (
        (
            mlir.FileLineColRange(
                filename=mlir.StringAttr(value="kernel.py"),
                start_line=2,
                start_column=3,
                end_line=2,
                end_column=3,
            ),
            '"kernel.py":2:3',
        ),
        (
            mlir.FileLineColRange(
                filename=mlir.StringAttr(value="kernel.py"),
                start_line=2,
                start_column=3,
                end_line=2,
                end_column=7,
            ),
            '"kernel.py":2:3 to :7',
        ),
        (
            mlir.FileLineColRange(
                filename=mlir.StringAttr(value="kernel.py"),
                start_line=2,
                start_column=3,
                end_line=4,
                end_column=7,
            ),
            '"kernel.py":2:3 to 4:7',
        ),
        (mlir.UnknownLoc(), "?"),
    ),
)
def test_location_printing(location, expected):
    assert str(location) == expected


def test_call_site_location_printing():
    callee = mlir.FileLineColRange(
        filename=mlir.StringAttr(value="helper.py"),
        start_line=4,
        start_column=5,
        end_line=4,
        end_column=5,
    )
    caller = mlir.FileLineColRange(
        filename=mlir.StringAttr(value="kernel.py"),
        start_line=12,
        start_column=9,
        end_line=12,
        end_column=9,
    )
    location = mlir.CallSiteLoc(callee=callee, caller=caller)

    assert str(location) == 'callsite("helper.py":4:5 at "kernel.py":12:9)'
