# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import pytest

import cuda.tile as ct
from cuda.tile._bytecode.version import BytecodeVersion
from cuda.tile._cext import default_tile_context
from cuda.tile._ir.ops import Loop, MakePartitionView
from cuda.tile._compile import _get_final_ir
from cuda.tile._ir.ir import KernelArgument
from cuda.tile._ir.type import ArrayTy


# MakePartitionView must not be hoisted away from its consumer (version < V_13_3)
@pytest.mark.parametrize("version", [BytecodeVersion.V_13_1, BytecodeVersion.V_13_2])
def test_partition_view_grouped_with_consumer(version):
    def kernel(x):
        for i in range(10):
            n = ct.num_tiles(x, 0, shape=(1,))
            v = ct.load(x, i, shape=(1,))
            ct.store(x, i, v + n)

    x_arg = KernelArgument(type=ArrayTy(ct.float32,
                                        shape=(None,),
                                        strides=(1,),
                                        elements_disjoint=True,
                                        base_ptr_div_by=None,
                                        stride_div_by=(None,),
                                        shape_div_by=(None,)))
    root_block = _get_final_ir(kernel, (x_arg,), default_tile_context.config,
                               version).body

    root_pvs = [op for op in root_block if isinstance(op, MakePartitionView)]
    assert len(root_pvs) == 1, "Expected 1 MakePartitionView hoisted to root (for num_tiles)"

    loop = next(op for op in root_block if isinstance(op, Loop))
    body_pvs = [op for op in loop.body if isinstance(op, MakePartitionView)]
    assert len(body_pvs) == 2, "Expected 2 MakePartitionViews inside loop (for load and store)"
