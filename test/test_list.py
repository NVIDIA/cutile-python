# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import math
from unittest.mock import patch

import cuda.tile
import cuda.tile as ct
from cuda.tile._bytecode import BytecodeVersion
from typing import Annotated
from util import assert_equal
from conftest import requires_tileiras


@ct.kernel
def add_arrays(arrays, out):
    res = ct.zeros((16, 16), dtype=out.dtype)
    for i in range(len(arrays)):
        t = ct.load(arrays[i], (0, 0), (16, 16))
        res += t
    ct.store(out, (0, 0), res)


@ct.kernel
def add_arrays_with_const_index(arrays, out):
    tx = ct.load(arrays[0], (0, 0), (16, 16))
    ty = ct.load(arrays[1], (0, 0), (16, 16))
    tz = ct.load(arrays[2], (0, 0), (16, 16))
    res = tx + ty + tz
    ct.store(out, (0, 0), res)


@ct.kernel
def add_arrays_with_0d_tile_index(arrays, out):
    bid = ct.full((), 0, dtype=ct.int32)
    tx = ct.load(arrays[bid], (0, 0), (16, 16))
    ty = ct.load(arrays[bid + 1], (0, 0), (16, 16))
    tz = ct.load(arrays[bid + 2], (0, 0), (16, 16))
    res = tx + ty + tz
    ct.store(out, (0, 0), res)


@pytest.mark.parametrize("kernel", [
    add_arrays,
    add_arrays_with_const_index,
    add_arrays_with_0d_tile_index
    ])
def test_add_list_of_arrays(kernel):
    arrays = [torch.randint(0, 100, (16, 16), dtype=torch.int32, device="cuda") for _ in range(3)]
    out = torch.zeros(16, 16, dtype=torch.int32, device="cuda")
    ref = sum(arrays)

    ct.launch(torch.cuda.current_stream(), (1,), kernel, (arrays, out))
    assert_equal(out, ref)


ListOfArrayIndexedWithInt64 = Annotated[
    list, ct.ListAnnotation(element=ct.IndexedWithInt64)
]

ListWithStaticShape = Annotated[
    list, ct.ListAnnotation(element=ct.ArrayAnnotation(static_shape_dims=(0, 1)))
]

ListWithStaticStride = Annotated[
    list, ct.ListAnnotation(element=ct.ArrayAnnotation(static_stride_dims=(0,)))
]


@ct.kernel
def add_int64_index_arrays(
    arrays: ListOfArrayIndexedWithInt64,
    out: ct.IndexedWithInt64,
    TILE: ct.Constant[int]
):
    bid = ct.bid(0)
    res = ct.zeros((TILE, 1), dtype=out.dtype)
    for i in range(len(arrays)):
        t = ct.load(arrays[i], (bid, 0), (TILE, 1))
        res += t
    ct.store(out, (bid, 0), res)


@ct.kernel
def add_static_shape_arrays(arrays: ListWithStaticShape, out):
    res = ct.zeros((16, 16), dtype=out.dtype)
    for i in range(len(arrays)):
        t = ct.load(arrays[i], (0, 0), (16, 16))
        res += t
    ct.store(out, (0, 0), res)


def test_add_list_static_shape():
    k = cuda.tile.kernel(add_static_shape_arrays._pyfunc)

    # (16,16) → 1st compile; (32,32) → 2nd compile; (16,16) again → cache hit.
    shapes = [(16, 16), (32, 32), (16, 16)]
    with patch('cuda.tile._compile.compile_tile',
               side_effect=cuda.tile._compile.compile_tile) as mock_compile:
        for shape in shapes:
            arrays = [torch.randint(0, 100, shape, dtype=torch.int32, device="cuda")
                      for _ in range(3)]
            out = torch.zeros((16, 16), dtype=torch.int32, device="cuda")
            ct.launch(torch.cuda.current_stream(), (1,), k, (arrays, out))
            assert_equal(out, sum(a[:16, :16] for a in arrays))

    assert mock_compile.call_count == 2


def test_add_list_static_shape_mismatch():
    k = cuda.tile.kernel(add_static_shape_arrays._pyfunc)

    arrays = [
        torch.zeros((16, 16), dtype=torch.int32, device="cuda"),
        torch.zeros((32, 16), dtype=torch.int32, device="cuda"),
    ]
    out = torch.zeros((16, 16), dtype=torch.int32, device="cuda")
    with pytest.raises(ValueError, match="vary in static shape at axis 0"):
        ct.launch(torch.cuda.current_stream(), (1,), k, (arrays, out))


@ct.kernel
def add_static_stride_arrays(arrays: ListWithStaticStride, out):
    a = arrays[0]
    ct.static_assert(a.strides[0] == 10)
    res = ct.zeros((4, 8), dtype=out.dtype)
    for i in range(len(arrays)):
        t = ct.load(arrays[i], (0, 0), (4, 8))
        res += t
    ct.store(out, (0, 0), res)


def _padded(pitch, fill):
    return torch.full((4, pitch), fill, dtype=torch.int32, device="cuda")[:, :8]


def test_add_list_static_stride():
    k = cuda.tile.kernel(add_static_stride_arrays._pyfunc)
    arrays = [_padded(10, v) for v in (1, 2, 3)]
    out = torch.zeros((4, 8), dtype=torch.int32, device="cuda")
    ct.launch(torch.cuda.current_stream(), (1,), k, (arrays, out))
    assert_equal(out, sum(arrays))


def test_add_list_static_stride_mismatch():
    k = cuda.tile.kernel(add_static_stride_arrays._pyfunc)

    # Items disagree on the annotated row stride.
    arrays = [_padded(10, 1), _padded(12, 2)]
    out = torch.zeros((4, 8), dtype=torch.int32, device="cuda")
    with pytest.raises(ValueError, match="vary in static stride at axis 0"):
        ct.launch(torch.cuda.current_stream(), (1,), k, (arrays, out))


@requires_tileiras(BytecodeVersion.V_13_3)
def test_add_list_of_int64_index_arrays():
    """
    Sum a list of large 2D arrays whose stride[0] exceeds INT32_MAX.

    This test may be excluded from selected CI jobs with
    ``-k "not int64_index"`` because it requires a very large allocation.
    Keep ``int64_index`` in the test name unless those CI filters are updated.
    """
    TILE = 2048
    n = (1 << 32) + TILE  # shape[0] > UINT32_MAX
    arrays = [torch.full((n, 1), i + 1, device='cuda', dtype=torch.int8) for i in range(3)]
    out = torch.zeros(n, 1, device='cuda', dtype=torch.int8)

    grid = (math.ceil(n / TILE), 1, 1)
    ct.launch(torch.cuda.current_stream(), grid, add_int64_index_arrays, (arrays, out, TILE))
    assert (out == 6).all().item()
