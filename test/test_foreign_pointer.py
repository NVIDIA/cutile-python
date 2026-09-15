# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import cuda.tile as ct
from cuda.tile._datatype import (
    DType,
    foreign_pointer_dtype,
    foreign_pointer_pointee_dtype,
    is_foreign_pointer_dtype,
    is_numeric,
    is_pointer_dtype,
)
from cuda.tile._exception import TileTypeError
from util import assert_equal


def test_foreign_pointer_dtype():
    dtype = foreign_pointer_dtype(ct.float32)

    assert isinstance(dtype, DType)
    assert dtype is foreign_pointer_dtype(ct.float32)
    assert dtype.name == "foreign_pointer[float32]"
    assert dtype.bitwidth == 64
    assert foreign_pointer_pointee_dtype(dtype) is ct.float32
    assert is_foreign_pointer_dtype(dtype)
    assert not is_pointer_dtype(dtype)
    assert not is_numeric(dtype)

    with pytest.raises(TypeError, match="nested foreign pointer dtypes are not supported"):
        foreign_pointer_dtype(dtype)


def test_foreign_pointer_array_round_trip_static():
    @ct.kernel
    def kernel(x, y):
        view = ct.Array.from_foreign(x.foreign_pointer, (16,), (1,))
        ct.store(y, (0,), ct.load(view, (0,), shape=(16,)))

    x = torch.arange(16, dtype=torch.float32, device="cuda:0")
    y = torch.zeros_like(x)
    ct.launch(torch.cuda.current_stream(), (1,), kernel, (x, y))
    assert_equal(y, x)


def test_foreign_pointer_array_round_trip_dynamic():
    @ct.kernel
    def kernel(x, y):
        half = x.shape[0] // 2
        second_half = x.slice(axis=0, start=half, stop=x.shape[0])
        view = ct.Array.from_foreign(
            second_half.foreign_pointer, (half, 16), second_half.strides
        )
        ct.store(y, (0, 0), ct.load(view, (0, 0), shape=(16, 16)))

    x = torch.arange(32 * 16, dtype=torch.float32, device="cuda:0").reshape(32, 16)
    y = torch.zeros((16, 16), dtype=torch.float32, device="cuda:0")
    ct.launch(torch.cuda.current_stream(), (1,), kernel, (x, y))
    assert_equal(y, x[16:, :])


def test_foreign_pointer_can_be_broadcast_and_reshaped():
    @ct.kernel
    def kernel(x, y):
        pointers = ct.broadcast_to(x.foreign_pointer, (4,))
        pointers = ct.reshape(pointers, (2, 2))
        pointer = ct.extract(pointers, (1, 1), shape=())
        x_view = ct.Array.from_foreign(pointer, (16,), (1,))
        y_view = ct.Array.from_foreign(y.foreign_pointer, (16,), (1,))
        ct.store(y_view, (0,), ct.load(x_view, (0,), shape=(16,)))

    x = torch.arange(16, dtype=torch.float32, device="cuda:0")
    y = torch.zeros_like(x)
    ct.launch(torch.cuda.current_stream(), (1,), kernel, (x, y))
    assert_equal(y, x)


def _add(pointer): return pointer + 1
def _astype(pointer): return pointer.astype(ct.uint64)
def _compare(pointer): return pointer == pointer
def _compare_function(pointer): return ct.equal(pointer, pointer)
def _dtype_constructor_source(pointer): return ct.uint64(pointer)
def _logical_not(pointer): return not pointer
def _positive(pointer): return +pointer
def _bitcast_source(pointer): return ct.bitcast(pointer, ct.uint64)
def _bitcast_target(pointer): return ct.bitcast(ct.uint64(0), pointer.dtype)


@pytest.mark.parametrize(
    "operation,match",
    [
        (_add, "non-arithmetic dtype"),
        (_astype, "is_numeric"),
        (_compare, "is_numeric"),
        (_compare_function, "is_numeric"),
        (_dtype_constructor_source, "is_numeric"),
        (_logical_not, "is_numeric"),
        (_positive, "is_numeric"),
        (_bitcast_source, "foreign pointer"),
        (_bitcast_target, "foreign pointer"),
    ],
)
def test_foreign_pointer_rejects_numeric_operations(operation, match):
    @ct.kernel
    def kernel(x):
        operation(x.foreign_pointer)

    x = torch.zeros(16, dtype=torch.float32, device="cuda:0")
    with pytest.raises(TileTypeError, match=match):
        ct.launch(torch.cuda.current_stream(), (1,), kernel, (x,))


@pytest.mark.parametrize(
    "condition,expected_output",
    [
        (True, tuple(range(16))),
        (False, tuple(range(16, 32))),
    ],
)
def test_foreign_pointer_supports_where(condition, expected_output):
    @ct.kernel
    def kernel(x, y):
        pointer = ct.where(
            ct.astile(condition, dtype=ct.bool_),
            x.foreign_pointer,
            x.slice(axis=0, start=16, stop=32).foreign_pointer,
        )
        view = ct.Array.from_foreign(pointer, (16,), (1,))
        ct.store(y, (0,), ct.load(view, (0,), shape=(16,)))

    x = torch.arange(32, dtype=torch.float32, device="cuda:0")
    y = torch.zeros(16, dtype=torch.float32, device="cuda:0")
    expected_output = torch.tensor(
        expected_output, dtype=torch.float32, device="cuda:0"
    )
    ct.launch(torch.cuda.current_stream(), (1,), kernel, (x, y))
    assert_equal(y, expected_output)


@pytest.mark.parametrize(
    "shape,strides,match",
    [
        ((16, 1), (1,), "same rank"),
        ((-1,), (1,), "non-negative"),
        ((1,), (-1,), "non-negative"),
        ((1.0,), (1,), "signed integer"),
    ],
)
def test_array_from_foreign_validation(shape, strides, match):
    @ct.kernel
    def kernel(x):
        ct.Array.from_foreign(x.foreign_pointer, shape, strides)

    x = torch.zeros(16, dtype=torch.float32, device="cuda:0")
    with pytest.raises(TileTypeError, match=match):
        ct.launch(torch.cuda.current_stream(), (1,), kernel, (x,))
