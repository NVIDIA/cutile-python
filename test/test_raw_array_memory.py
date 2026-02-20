# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from math import ceil
import cuda.tile as ct
from util import assert_equal
from conftest import float_dtypes, int_dtypes, dtype_id
from torch.testing import make_tensor


@ct.kernel
def copy_via_raw_array_memory_1d(x, y, TILE: ct.Constant[int]):
    """Copy 1D array using get_raw_memory().load_offset/store_offset."""
    bid = ct.bid(0)
    mem_in = x.get_raw_memory()
    mem_out = y.get_raw_memory()
    indices = ct.arange(TILE, dtype=ct.int64) + bid * TILE
    t = mem_in.load_offset(indices)
    mem_out.store_offset(indices, t)


@pytest.mark.parametrize("shape", [(128,), (256,)])
@pytest.mark.parametrize("tile", [64, 128])
@pytest.mark.parametrize("dtype", float_dtypes + int_dtypes, ids=dtype_id)
def test_copy_via_raw_array_memory_1d(shape, tile, dtype):
    x = make_tensor(shape, dtype=dtype, device="cuda")
    y = torch.zeros_like(x)
    grid = (ceil(shape[0] / tile), 1, 1)
    ct.launch(torch.cuda.current_stream(), grid, copy_via_raw_array_memory_1d, (x, y, tile))
    assert_equal(x, y)


@ct.kernel
def scalar_load_store_via_raw_array_memory(x, y):
    """Load and store a single element via get_raw_memory."""
    mem_in = x.get_raw_memory()
    mem_out = y.get_raw_memory()
    val = mem_in.load_offset(0)
    mem_out.store_offset(0, val)


def test_scalar_load_store_via_get_raw_memory():
    x = torch.full((1,), 42.0, dtype=torch.float32, device="cuda")
    y = torch.zeros_like(x)
    ct.launch(torch.cuda.current_stream(), (1,), scalar_load_store_via_raw_array_memory, (x, y))
    assert y.cpu().item() == 42.0


@ct.kernel
def load_offset_with_padding_value(x, y, pad_val: ct.Constant[float]):
    """Use load_offset with mask; masked-out positions get padding_value."""
    mem_in = x.get_raw_memory()
    mem_out = y.get_raw_memory()
    indices = ct.arange(8, dtype=ct.int64)
    mask = indices < 5
    t = mem_in.load_offset(indices, mask=mask, padding_value=pad_val)
    mem_out.store_offset(indices, t)


def test_load_offset_with_padding_value():
    x = torch.arange(100, 108, dtype=torch.float32, device="cuda")
    y = torch.zeros(8, dtype=torch.float32, device="cuda")
    pad_val = -1.0
    ct.launch(torch.cuda.current_stream(), (1,), load_offset_with_padding_value, (x, y, pad_val))
    expected = torch.cat([x[:5], torch.full((3,), pad_val, dtype=torch.float32, device="cuda")])
    assert_equal(y, expected)


@ct.kernel
def load_store_offset_with_latency(x, y, TILE: ct.Constant[int]):
    """Exercise latency hint on load_offset and store_offset."""
    bid = ct.bid(0)
    mem_in = x.get_raw_memory()
    mem_out = y.get_raw_memory()
    indices = ct.arange(TILE, dtype=ct.int64) + bid * TILE
    t = mem_in.load_offset(indices, latency=5)
    mem_out.store_offset(indices, t, latency=3)


def test_load_store_offset_with_latency():
    x = make_tensor((64,), dtype=torch.float32, device="cuda")
    y = torch.zeros_like(x)
    ct.launch(torch.cuda.current_stream(), (1,), load_store_offset_with_latency, (x, y, 64))
    assert_equal(x, y)


@ct.kernel
def sparse_2d_copy_gather_scatter(x, y, row_indices, col_indices, N_INDICES: ct.Constant[int]):
    """Copy a few 2D elements using gather/scatter with logical indices (loaded from arrays)."""
    row_tile = ct.load(row_indices, (0,), shape=(N_INDICES,))
    col_tile = ct.load(col_indices, (0,), shape=(N_INDICES,))
    t = ct.gather(x, (row_tile, col_tile))
    ct.scatter(y, (row_tile, col_tile), t)


@ct.kernel
def sparse_2d_copy_load_offset(x, y, offsets, N_INDICES: ct.Constant[int]):
    """Copy a few 2D elements using load_offset/store_offset with pre-computed offsets."""
    offsets_tile = ct.load(offsets, (0,), shape=(N_INDICES,))
    mem_in = x.get_raw_memory()
    mem_out = y.get_raw_memory()
    t = mem_in.load_offset(offsets_tile)
    mem_out.store_offset(offsets_tile, t)


def test_2d_sparse_load_offset_vs_gather_scatter():
    """Compare load_offset/store_offset (memory offsets) with gather/scatter (logical indices)."""
    M, N = 8, 8
    rows = [0, 1, 2, 5, 3, 6, 7, 4]
    cols = [0, 3, 1, 4, 7, 2, 5, 6]
    n = len(rows)

    x = make_tensor((M, N), dtype=torch.float32, device="cuda")
    y_gather = torch.zeros_like(x)
    y_offsets = torch.zeros_like(x)

    # Logical indices for gather/scatter
    row_idx = torch.tensor(rows, dtype=torch.int64, device="cuda")
    col_idx = torch.tensor(cols, dtype=torch.int64, device="cuda")

    ct.launch(
        torch.cuda.current_stream(), (1,),
        sparse_2d_copy_gather_scatter,
        (x, y_gather, row_idx, col_idx, n),
    )

    # Element memory offsets (row-major: offset = row * N + col)
    offsets = [r * N + c for r, c in zip(rows, cols)]
    off_tensor = torch.tensor(offsets, dtype=torch.int64, device="cuda")

    ct.launch(
        torch.cuda.current_stream(), (1,),
        sparse_2d_copy_load_offset,
        (x, y_offsets, off_tensor, n),
    )

    assert_equal(y_gather, y_offsets)
    assert_equal(y_gather[rows, cols], x[rows, cols])


def test_load_offset_broadcast_padding_value_scalar():
    """padding_value is scalar; broadcast to offset shape (8,). Result shape is (8,) not scalar."""
    @ct.kernel
    def load_offset_scalar_padding(x, y):
        """Offset shape (8,), padding_value scalar -> result shape (8,)."""
        mem_in = x.get_raw_memory()
        mem_out = y.get_raw_memory()
        offsets = ct.arange(8, dtype=ct.int64)
        pad = 0.0
        t = mem_in.load_offset(offsets, padding_value=pad)
        mem_out.store_offset(offsets, t)

    x = torch.arange(100, 108, dtype=torch.float32, device="cuda")
    y = torch.zeros(8, dtype=torch.float32, device="cuda")
    ct.launch(torch.cuda.current_stream(), (1,), load_offset_scalar_padding, (x, y))
    assert_equal(x, y)


def test_store_offset_broadcast_value_scalar():
    """value is scalar; broadcast to offset shape (8,). Offset is not broadcast to value."""
    @ct.kernel
    def store_offset_scalar_value(y):
        """Offset shape (8,), value scalar -> store same value to all 8 positions."""
        mem_out = y.get_raw_memory()
        offsets = ct.arange(8, dtype=ct.int64)
        mem_out.store_offset(offsets, 7.0)

    y = torch.zeros(8, dtype=torch.float32, device="cuda")
    ct.launch(torch.cuda.current_stream(), (1,), store_offset_scalar_value, (y,))
    expected = torch.full((8,), 7.0, dtype=torch.float32, device="cuda")
    assert_equal(y, expected)


def test_store_offset_value_broadcasts_to_offset_shape():
    """Result shape is offset shape (4,2). Value (4,1) broadcasts to (4,2), not the other way."""
    @ct.kernel
    def store_offset_value_shape_4x1_offset_4x2(y, fill_val: ct.Constant[float]):
        """Offset shape (4, 2), value shape (4, 1) -> value broadcasts to (4, 2)."""
        mem_out = y.get_raw_memory()
        offsets = ct.reshape(ct.arange(8, dtype=ct.int64), (4, 2))
        value_4x1 = ct.full((4, 1), fill_val, dtype=ct.float32)
        mem_out.store_offset(offsets, value_4x1)

    y = torch.zeros(8, dtype=torch.float32, device="cuda")
    ct.launch(
        torch.cuda.current_stream(), (1,),
        store_offset_value_shape_4x1_offset_4x2,
        (y, 11.0),
    )
    assert y.cpu().tolist() == [11.0] * 8


def test_load_offset_broadcast_mask():
    """mask (1,) broadcasts to offset shape (8,)."""
    @ct.kernel
    def load_offset_mask_shape_1(x, y):
        """Offset shape (8,), mask shape (1,) -> mask broadcasts to (8,)."""
        mem_in = x.get_raw_memory()
        mem_out = y.get_raw_memory()
        offsets = ct.arange(8, dtype=ct.int64)
        mask = ct.full((1,), True, dtype=ct.bool_)
        t = mem_in.load_offset(offsets, mask=mask, padding_value=0.0)
        mem_out.store_offset(offsets, t)

    x = torch.arange(100, 108, dtype=torch.float32, device="cuda")
    y = torch.zeros(8, dtype=torch.float32, device="cuda")
    ct.launch(torch.cuda.current_stream(), (1,), load_offset_mask_shape_1, (x, y))
    assert_equal(x, y)


def test_no_offset_to_value_broadcasting():
    """Offset (1,) and value (8,): no broadcast of offset to value shape; Expect TileTypeError."""

    @ct.kernel
    def store_offset_shape_1_value_shape_8(y):
        """Offset shape (1,), value shape (8,) -> value must broadcast to (1,); (8,) cannot."""
        mem_out = y.get_raw_memory()
        offsets_1 = ct.arange(1, dtype=ct.int64)
        value_8 = ct.arange(8, dtype=ct.float32)
        mem_out.store_offset(offsets_1, value_8)

    y = torch.zeros(8, dtype=torch.float32, device="cuda")
    with pytest.raises(ct.TileTypeError, match="broadcastable"):
        ct.launch(
            torch.cuda.current_stream(), (1,),
            store_offset_shape_1_value_shape_8,
            (y,),
        )


@ct.kernel
def interleave_load_store_gather_scatter_offset(a):
    """Interleave ct.store/ct.load, ct.gather/ct.scatter, load_offset/store_offset on same array.
    Ordering must be respected so each read sees the latest write.
    """
    mem = a.get_raw_memory()
    # 1) ct.store at index 0
    ct.store(a, (0,), ct.full((1,), 100.0, dtype=ct.float32))
    # 2) load_offset(0), store_offset(1) -> a[1] = 100
    t = mem.load_offset(0)
    mem.store_offset(1, t)
    # 3) ct.load at (1,), add 1, ct.store at (2,) -> a[2] = 101
    t = ct.load(a, (1,), shape=(1,))
    t = t + 1.0
    ct.store(a, (2,), t)
    # 4) gather from (0,1,2,3), add 1, scatter to (4,5,6,7)
    # at the same time load_offset from (1) -> s, store_offset to (3)
    gather_idx = ct.arange(4, dtype=ct.int64)
    s = mem.load_offset(1)
    g = ct.gather(a, (gather_idx,))
    g = g + 1.0
    scatter_idx = ct.arange(4, dtype=ct.int64) + 4
    mem.store_offset(3, s)
    ct.scatter(a, (scatter_idx,), g)
    # 5) load_offset(4..7), store_offset(8..11) -> copy a[4:8] to a[8:12]
    off_in = ct.arange(4, dtype=ct.int64) + 4
    off_out = ct.arange(4, dtype=ct.int64) + 8
    t = mem.load_offset(off_in)
    mem.store_offset(off_out, t)


def test_interleave_load_store_gather_scatter_offset_ordering():
    """Interleaving ct.load/ct.store, gather/scatter, load_offset/store_offset on same array.
    Verifies token/ordering is correct: each read sees the preceding writes.
    """
    a = torch.zeros(16, dtype=torch.float32, device="cuda")
    ct.launch(
        torch.cuda.current_stream(), (1,),
        interleave_load_store_gather_scatter_offset,
        (a,),
    )
    # After step 1: a[0]=100
    # After step 2: a[1]=100
    # After step 3: a[2]=101
    # After step 4: gather (0,1,2,3) -> (100,100,101,0), +1 -> scatter to (4..7): 101,101,102,1
    #               load_offset from (1) -> 100, store_offset to (3) -> 100
    # After step 5: a[8:12] = a[4:8] -> 101, 101, 102, 1
    assert a[0].item() == 100.0
    assert a[1].item() == 100.0
    assert a[2].item() == 101.0
    assert a[3].item() == 100.0
    assert a[4].item() == 101.0
    assert a[5].item() == 101.0
    assert a[6].item() == 102.0
    assert a[7].item() == 1.0
    assert a[8].item() == 101.0
    assert a[9].item() == 101.0
    assert a[10].item() == 102.0
    assert a[11].item() == 1.0


@pytest.mark.parametrize("dtype", float_dtypes + int_dtypes, ids=dtype_id)
def test_raw_array_memory_dtype(dtype):
    """RawArrayMemory.dtype returns the element dtype of the backing array."""

    @ct.kernel
    def check_raw_dtype(x):
        mem = x.get_raw_memory()
        mem_dtype = mem.dtype
        ct.static_assert(mem_dtype == x.dtype)

    x = make_tensor((4,), dtype=dtype, device="cuda")
    ct.launch(torch.cuda.current_stream(), (1,), check_raw_dtype, (x,))
