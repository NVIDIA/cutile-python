# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import Mapping

from cuda.lang._enums import SwizzleMode, TensorMapL2Promotion
from cuda.lang._ir import ir
from cuda.lang._ir._host_program import HostProgram
from cuda.lang._ir.ops import CreateTensorMap
from cuda.lang._ir.type import TensorMapTy
from cuda.lang._exception import TypeCheckingError
from cuda.tile._ir.ir import Var
from cuda.tile._ir.core_ops import assign
from cuda.tile import _cext


@dataclass
class HoistedTensorMap:
    rank: int
    data_type: int
    base_ptr_param: int
    shape_stride_program: HostProgram
    tile_shape: tuple[int, ...]
    swizzle: SwizzleMode
    l2_promotion: TensorMapL2Promotion


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
            map_ty: TensorMapTy = op.result_var.get_type()
            new_param.set_type(map_ty)
            assign(new_param, op.result_var)
            new_params.append(new_param)

            shape_stride_program = HostProgram()
            for x in op.array_shape:
                var_to_host_program(x, shape_stride_program)
            for x in op.array_strides:
                var_to_host_program(x, shape_stride_program)

            hoisted_maps.append(HoistedTensorMap(
                    rank=len(op.array_shape),
                    data_type=getattr(_cext, map_ty.data_type),
                    base_ptr_param=param_idx(op.base_ptr),
                    shape_stride_program=shape_stride_program,
                    tile_shape=map_ty.tile_shape,
                    swizzle=map_ty.swizzle,
                    l2_promotion=map_ty.l2_promotion))

    removed_count = kernel_body.remove_if(lambda op: isinstance(op, CreateTensorMap))
    assert removed_count == len(ops)

    kernel_body[:0] = builder.ops
    kernel_body.params += tuple(new_params)
    return hoisted_maps
