# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import Mapping

from cuda.lang._enums import (SwizzleMode, TensorMapInterleave,
                              TensorMapFloatOOBFill, TensorMapL2Promotion)
from cuda.lang._ir import ir
from cuda.lang._ir._host_program import HostProgram
from cuda.lang._ir.ops import CreateTensorMap
from cuda.lang._exception import TypeCheckingError
from cuda.tile._ir.ir import Var
from cuda.tile._ir.core_ops import assign


@dataclass
class HoistedTensorMap:
    rank: int
    data_type: int
    base_ptr_param: int
    shape_stride_program: HostProgram
    tile_shape: tuple[int, ...]
    interleave: TensorMapInterleave
    swizzle: SwizzleMode
    l2_promotion: TensorMapL2Promotion
    oob_fill: TensorMapFloatOOBFill


def hoist_tensor_maps(kernel_body: ir.Block,
                      host_program_by_var: Mapping[str, HostProgram]) -> list[HoistedTensorMap]:
    ops = [op for op in kernel_body.traverse() if isinstance(op, CreateTensorMap)]
    if len(ops) == 0:
        return []

    def param_idx(x: Var):
        for i, param in enumerate(kernel_body.params):
            if param.name == x.name:
                return i
        raise TypeCheckingError(
            "Array used for tensor map creation must be a kernel parameter",
            loc=x.loc
        )

    def var_to_host_program(x: Var, prog: HostProgram):
        var_prog = host_program_by_var.get(x.name)
        if var_prog is None:
            raise TypeCheckingError(
                "Array used for tensor map creation must be a kernel parameter",
                loc=x.loc
            )
        prog.extend(var_prog)

    hoisted_maps = []
    new_params = []
    with ir.TileBuilder(kernel_body.ctx, kernel_body.loc) as builder:
        for op in ops:
            new_param = builder.ir_ctx.make_temp(op.loc)
            new_param.set_type(op.result_var.get_type())
            assign(new_param, op.result_var)
            new_params.append(new_param)

            shape_stride_program = HostProgram()
            for x in op.array_shape:
                var_to_host_program(x, shape_stride_program)
            for x in op.array_strides:
                var_to_host_program(x, shape_stride_program)

            hoisted_maps.append(HoistedTensorMap(
                    rank=len(op.array_shape),
                    data_type=op.data_type.value,
                    base_ptr_param=param_idx(op.base_ptr),
                    shape_stride_program=shape_stride_program,
                    tile_shape=tuple(dim.get_constant() for dim in op.tile_shape),
                    interleave=op.interleave,
                    swizzle=op.swizzle,
                    l2_promotion=op.l2_promotion,
                    oob_fill=op.oob_fill))

    removed_count = kernel_body.remove_if(lambda op: isinstance(op, CreateTensorMap))
    assert removed_count == len(ops)

    kernel_body[:0] = builder.ops
    kernel_body.params += tuple(new_params)
    return hoisted_maps
