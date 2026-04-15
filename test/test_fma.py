# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from math import ceil
import cuda.tile as ct
from util import assert_close, filecheck, get_bytecode
from torch.testing import make_tensor


def mul_add_kernel(x, y, z, output,
                   TILE: ct.Constant[int],
                   DIM: ct.Constant[int]):
    bidx = ct.bid(0)
    tx = ct.load(x, index=(bidx, 0), shape=(TILE, DIM))
    ty = ct.load(y, index=(bidx, 0), shape=(TILE, DIM))
    tz = ct.load(z, index=(bidx, 0), shape=(TILE, DIM))
    output_tile = tx * ty + tz
    ct.store(output, index=(bidx, 0), tile=output_tile)


def mul_add_kernel_local_var(x, y, z, output,
                             TILE: ct.Constant[int],
                             DIM: ct.Constant[int]):
    bidx = ct.bid(0)
    tx = ct.load(x, index=(bidx, 0), shape=(TILE, DIM))
    ty = ct.load(y, index=(bidx, 0), shape=(TILE, DIM))
    tz = ct.load(z, index=(bidx, 0), shape=(TILE, DIM))
    tmp = tx * ty
    output_tile = tmp + tz
    ct.store(output, index=(bidx, 0), tile=output_tile)


def mul_sub_kernel(x, y, z, output,
                   TILE: ct.Constant[int],
                   DIM: ct.Constant[int]):
    bidx = ct.bid(0)
    tx = ct.load(x, index=(bidx, 0), shape=(TILE, DIM))
    ty = ct.load(y, index=(bidx, 0), shape=(TILE, DIM))
    tz = ct.load(z, index=(bidx, 0), shape=(TILE, DIM))
    output_tile = tx * ty - tz
    ct.store(output, index=(bidx, 0), tile=output_tile)


def add_mul_kernel(x, y, z, output,
                   TILE: ct.Constant[int],
                   DIM: ct.Constant[int]):
    bidx = ct.bid(0)
    tx = ct.load(x, index=(bidx, 0), shape=(TILE, DIM))
    ty = ct.load(y, index=(bidx, 0), shape=(TILE, DIM))
    tz = ct.load(z, index=(bidx, 0), shape=(TILE, DIM))
    output_tile = tz + tx * ty
    ct.store(output, index=(bidx, 0), tile=output_tile)


def sub_mul_kernel(x, y, z, output,
                   TILE: ct.Constant[int],
                   DIM: ct.Constant[int]):
    bidx = ct.bid(0)
    tx = ct.load(x, index=(bidx, 0), shape=(TILE, DIM))
    ty = ct.load(y, index=(bidx, 0), shape=(TILE, DIM))
    tz = ct.load(z, index=(bidx, 0), shape=(TILE, DIM))
    output_tile = tz - tx * ty      # c - a*b
    ct.store(output, index=(bidx, 0), tile=output_tile)


@ct.kernel
def mul_add_same_operand_kernel(x, output,
                                TILE: ct.Constant[int],
                                DIM: ct.Constant[int]):
    bidx = ct.bid(0)
    tx = ct.load(x, index=(bidx, 0), shape=(TILE, DIM))
    tmp = tx * tx
    output_tile = tmp + tmp
    ct.store(output, index=(bidx, 0), tile=output_tile)


def test_fma_skip_when_new_op_uses_deleted_var():
    shape = (128, 32)
    x = make_tensor(shape, dtype=torch.float32, device='cuda')
    output = make_tensor(shape, dtype=torch.float32, device='cuda')
    TILE = 32
    grid = (ceil(shape[0] / TILE), 1, 1)
    ct.launch(torch.cuda.current_stream(), grid, mul_add_same_operand_kernel,
              (x, output, TILE, shape[1]))
    assert_close(output, 2 * x * x, atol=1e-3, rtol=1e-3)


@pytest.mark.use_mlir
@pytest.mark.parametrize(
    "kernel, kernel_ref",
    [
        pytest.param(mul_add_kernel_local_var, lambda x, y, z: x * y + z),
        pytest.param(mul_add_kernel, lambda x, y, z: x * y + z),
        pytest.param(mul_sub_kernel, lambda x, y, z: x * y - z),
        pytest.param(add_mul_kernel, lambda x, y, z: z + x * y),
        pytest.param(sub_mul_kernel, lambda x, y, z: z - x * y),
    ]
)
def test_fma(kernel, kernel_ref):
    shape = (128, 32)
    x = make_tensor(shape, dtype=torch.float32, device='cuda')
    y = make_tensor(shape, dtype=torch.float32, device='cuda')
    z = make_tensor(shape, dtype=torch.float32, device='cuda')
    output = make_tensor(shape, dtype=torch.float32, device='cuda')
    TILE = 32
    grid = (ceil(shape[0] / TILE), 1, 1)
    kernel = ct.kernel(kernel)
    bytecode = get_bytecode(kernel, (x, y, z, output, TILE, shape[1]))
    check_directive = """\
    // CHECK: %[[VAL:.*]] = fma
    // CHECK-NOT: mulf
    // CHECK-NOT: addf
    // CHECK-NOT: subf
    """
    filecheck(bytecode, check_directive)
    ct.launch(torch.cuda.current_stream(), grid, kernel, (x, y, z, output, TILE, shape[1]))
    assert_close(output, kernel_ref(x, y, z), atol=1e-3, rtol=1e-3)
