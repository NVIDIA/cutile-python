# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
from typing import Mapping

from cuda.lang._ir import ir
from cuda.lang._ir._host_program import HostProgram, host_program_to_ir
from cuda.lang._ir.ops import AllocDynSharedMemory, GetDynSharedMemoryBasePtr, \
    get_dyn_shared_memory_base_ptr, _pointer_with_offset, reinterpret_pointer
from cuda.lang._datatype import int32
from cuda.lang._exception import TileTypeError
from cuda.tile._datatype import PointerInfo
from cuda.tile._ir.ops import assign, _is_power_of_2
from cuda.tile._ir.type import TileTy


def handle_dynamic_shared_memory(kernel_body: ir.Block,
                                 host_program_by_var: Mapping[str, HostProgram]
                                 ) -> HostProgram | None:
    alloc_ops = [op for op in kernel_body.traverse() if isinstance(op, AllocDynSharedMemory)]
    if len(alloc_ops) == 0:
        return None

    # Sort allocations by decreasing alignment to minimize padding.
    alloc_ops.sort(key=_get_alignment, reverse=True)
    max_alignment = _get_alignment(alloc_ops[0])
    if max_alignment > GetDynSharedMemoryBasePtr.initial_alignment:
        raise TileTypeError(
            "Dynamic shared memory alignment cannot exceed "
            f"{GetDynSharedMemoryBasePtr.initial_alignment} bytes",
            loc=alloc_ops[0].loc,
        )

    # Per each array, build a HostProgram that computes its padded size in bytes.
    array_programs = tuple(_build_host_program(op, host_program_by_var) for op in alloc_ops)

    # Bump-allocate array pointers
    array_pointers = []
    with ir.TileBuilder(kernel_body.ctx, kernel_body.loc) as builder:
        ptr = get_dyn_shared_memory_base_ptr()
        array_pointers.append(ptr)
        for prev_prog in array_programs[:-1]:
            prev_arr_size = host_program_to_ir(prev_prog, kernel_body.params)
            ptr = _pointer_with_offset(ptr, prev_arr_size)
            array_pointers.append(ptr)

        for op, ptr in zip(alloc_ops, array_pointers, strict=True):
            ptr = reinterpret_pointer(ptr, op.result_var.get_type().dtype)
            assign(ptr, op.result_var)

    # Remove AllocDynSharedMemory operations
    removed_count = kernel_body.remove_if(lambda op: isinstance(op, AllocDynSharedMemory))
    assert removed_count == len(alloc_ops)

    # Prepend the newly generated code
    kernel_body[:0] = builder.ops

    # Make a final program that computes the total shared memory size
    total_program = HostProgram()
    for i, array_prog in enumerate(array_programs):
        total_program.extend(array_prog)
        if i > 0:
            total_program.opcodes.append("Add")

    return total_program


def _get_alignment(alloc_op: AllocDynSharedMemory) -> int:
    if alloc_op.alignment is not None:
        return alloc_op.alignment
    return _get_item_size(alloc_op)


def _get_item_size(alloc_op: AllocDynSharedMemory) -> int:
    pointer_tile_ty = alloc_op.result_var.get_type()
    assert isinstance(pointer_tile_ty, TileTy)
    info = PointerInfo(pointer_tile_ty.dtype)
    pointee_dtype = info.pointee_dtype
    assert pointee_dtype.bitwidth % 8 == 0
    return pointee_dtype.bitwidth // 8


def _round_up(value: int, alignment: int) -> int:
    assert _is_power_of_2(alignment)
    mask = alignment - 1
    return (value + mask) & ~mask


def _build_host_program(alloc_op: AllocDynSharedMemory,
                        host_program_by_var: Mapping[str, HostProgram]) -> HostProgram:
    pad_to_alignment = _get_alignment(alloc_op)
    program = HostProgram()
    constant_factor = _get_item_size(alloc_op)
    individual_size_programs = []
    for size_var in alloc_op.shape:
        var_prog = host_program_by_var.get(size_var.name)
        if var_prog is None:
            raise TileTypeError("Size of shared array must be either a constant"
                                " or a kernel parameter", loc=size_var.loc)
        const_val = var_prog.as_const()
        if const_val is None:
            if size_var.get_type() != TileTy(int32, ()):
                raise TileTypeError(f"Kernel parameter used as shared array size must be int32,"
                                    f" got {size_var.get_type()}",
                                    loc=size_var.loc)
            individual_size_programs.append(var_prog)
        else:
            constant_factor *= const_val

    if pad_to_alignment is not None:
        if len(individual_size_programs) == 0:
            constant_factor = _round_up(constant_factor, pad_to_alignment)
            needs_round_up = False
        elif constant_factor % pad_to_alignment != 0:
            needs_round_up = True
        else:
            needs_round_up = False
    else:
        needs_round_up = False

    first_factor = True
    if constant_factor != 1 or len(individual_size_programs) == 0:
        program.opcodes.append("Const")
        program.op_attrs.append(constant_factor)
        first_factor = False

    for prog in individual_size_programs:
        program.extend(prog)
        if first_factor:
            first_factor = False
        else:
            program.opcodes.append("Mul")

    if needs_round_up:
        program.opcodes.append("RoundUpToPow2")
        program.op_attrs.append(pad_to_alignment)

    return program
