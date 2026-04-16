# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import math

import cuda.tile as ct
from cuda.tile._bytecode import BytecodeVersion
from cuda.tile._datatype import int64
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
    list, ct.ListAnnotation(element=ct.ArrayAnnotation(index_dtype=int64))
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
