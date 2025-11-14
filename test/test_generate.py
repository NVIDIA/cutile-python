# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from math import ceil
import cuda.tile as ct
from util import assert_equal
from conftest import int_dtypes, dtype_id


@ct.kernel
def arange(x, TILE: ct.Constant[int]):
    bid = ct.astype(ct.bid(0), x.dtype)
    start = bid * TILE
    tx = start + ct.arange(TILE, dtype=x.dtype)
    ct.store(x, index=(bid,), tile=tx)


@pytest.mark.parametrize("shape", [(128,)])
@pytest.mark.parametrize("tile", [64])
@pytest.mark.parametrize("dtype", int_dtypes, ids=dtype_id)
def test_arange(shape, dtype, tile):
    x = torch.zeros(shape, dtype=dtype, device='cuda')
    grid = (ceil(shape[0] / tile), 1, 1)
    ct.launch(torch.cuda.current_stream(), grid, arange, (x, tile))
    ref = torch.arange(len(x), dtype=dtype, device=x.device)
    assert_equal(x, ref)
