# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from math import ceil
import cuda.tile as ct
from cuda.tile import TileTypeError
from util import assert_equal, assert_close
from conftest import int_dtypes, float_dtypes, dtype_id


@ct.kernel
def arange_dynamic_start_step(x, step, TILE: ct.Constant[int]):
    bid = ct.bid(0)
    tx = ct.arange(TILE, start=bid * TILE, step=step, dtype=x.dtype)
    ct.store(x, index=(bid,), tile=tx)


@pytest.mark.parametrize("shape", [(128,)])
@pytest.mark.parametrize("tile", [64])
@pytest.mark.parametrize("dtype", int_dtypes + float_dtypes, ids=dtype_id)
def test_arange_dynamic_start_step(shape, dtype, tile):
    x = torch.zeros(shape, dtype=dtype, device='cuda')
    grid = (ceil(shape[0] / tile), 1, 1)
    ct.launch(torch.cuda.current_stream(), grid, arange_dynamic_start_step, (x, 1, tile))
    ref = torch.arange(len(x), dtype=dtype, device=x.device)
    assert_equal(x, ref)


@pytest.mark.parametrize("size,start,step", [
    (128, None, None),  # arange(size)
    (64, 8, None), (64, -16, None),  # arange(size, start)
    (64, 0, 2), (64, 64, -1), (16, -8, -3), (4, 8.5, -1.1),  # arange(size, start, step)
    (8, 10, 0)  # step=0
])
@pytest.mark.parametrize("dtype", int_dtypes + float_dtypes, ids=dtype_id)
def test_arange(size, start, step, dtype):
    @ct.kernel
    def arange_kernel(x):
        if start is None:
            tx = ct.arange(size, dtype=x.dtype)
        elif step is None:
            tx = ct.arange(size, start=start, dtype=x.dtype)
        else:
            tx = ct.arange(size, start=start, step=step, dtype=x.dtype)
        ct.store(x, index=(0,), tile=tx)

    x = torch.zeros(size, dtype=dtype, device='cuda')
    ct.launch(torch.cuda.current_stream(), (1, 1, 1), arange_kernel, (x,))
    if step == 0:
        ref = torch.full((size,), start, dtype=dtype, device=x.device)
    else:
        start = 0 if start is None else start
        step = 1 if step is None else step
        ref = torch.arange(start, start + size * step, step, dtype=dtype, device=x.device)
    assert_close(x, ref)


@pytest.mark.parametrize("size,start,step,error_message", [
    (3, None, None, "Result tile shape must be power of 2"),
    (5, 0, 2, "Result tile shape must be power of 2"),
    (0.1, None, None, 'Expected an integer constant')
])
def test_arange_invalid_size(size, start, step, error_message):
    @ct.kernel
    def arange_kernel(x):
        if start is None:
            tx = ct.arange(size, dtype=x.dtype)
        elif step is None:
            tx = ct.arange(size, start=start, dtype=x.dtype)
        else:
            tx = ct.arange(size, start=start, step=step, dtype=x.dtype)
        ct.store(x, index=(0,), tile=tx)

    with pytest.raises(TileTypeError, match=error_message):
        x = torch.zeros(1, dtype=torch.int32, device='cuda')
        ct.launch(torch.cuda.current_stream(), (1, 1, 1), arange_kernel, (x,))


def test_arange_reject_dynamic_size():
    @ct.kernel
    def arange_dynamic_size(x):
        tx = ct.arange(ct.bid(0), dtype=x.dtype)
        ct.store(x, index=(0,), tile=tx)

    with pytest.raises(TileTypeError, match="Expected an integer constant"):
        x = torch.zeros(1, dtype=torch.int32, device='cuda')
        ct.launch(torch.cuda.current_stream(), (1, 1, 1), arange_dynamic_size, (x,))
