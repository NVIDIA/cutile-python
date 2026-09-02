# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from cuda.tile._exception import Loc, TileInternalError, TileSyntaxError
from cuda.tile._ir.core_ops import Assign, TypedConst
from cuda.tile._ir.ir import Block, Mapper, Operation, Var
from cuda.tile._ir.ops import TileReduce, TileScan


def _definitions(root_block: Block) -> dict[str, Operation]:
    return {
        result.name: op
        for op in root_block.traverse()
        for result in op.result_vars
    }


def _captures(body: Block) -> list[tuple[Var, Loc]]:
    local_names = {var.name for var in body.params}
    local_names.update(
        result.name
        for op in body.operations
        for result in op.result_vars
    )

    captures = []
    captured_names = set()
    for op in body.operations:
        for operand in op.all_inputs():
            if operand.name not in local_names and operand.name not in captured_names:
                captures.append((operand, op.loc))
                captured_names.add(operand.name)
    return captures


def _kind(op: TileReduce | TileScan) -> str:
    return "reduction" if isinstance(op, TileReduce) else "scan"


def _remember_reduce_scan_capture_names(root_block: Block) -> None:
    """Record source-level capture names before Assign operations are removed."""
    definitions = _definitions(root_block)
    names = {}
    for region_op in root_block.traverse():
        if not isinstance(region_op, TileReduce | TileScan):
            continue
        for value, _ in _captures(region_op.body):
            canonical_value = value
            defining_op = definitions.get(canonical_value.name)
            while isinstance(defining_op, Assign):
                canonical_value = defining_op.value
                defining_op = definitions.get(canonical_value.name)
            key = (region_op.op, region_op.loc, canonical_value.name)
            names[key] = value.get_original_name()
    root_block.ctx._reduce_scan_capture_names = names


def _original_capture_name(region_op: TileReduce | TileScan, value: Var) -> str:
    names = getattr(value.ctx, "_reduce_scan_capture_names", {})
    key = (region_op.op, region_op.loc, value.name)
    return names.get(key, value.get_original_name())


def legalize_reduce_scan_captures(root_block: Block) -> None:
    """Rematerialize constant captures and reject runtime captures."""
    definitions = _definitions(root_block)

    for region_op in root_block.traverse():
        if not isinstance(region_op, TileReduce | TileScan):
            continue

        mapper = Mapper(root_block.ctx)
        constants = []
        for value, consuming_loc in _captures(region_op.body):
            defining_op = definitions.get(value.name)
            if value.is_constant():
                constant_value = value.get_constant()
            elif isinstance(defining_op, TypedConst):
                constant_value = defining_op.value
            else:
                original_name = _original_capture_name(region_op, value)
                raise TileSyntaxError(
                    f"{_kind(region_op)} body captures runtime value '{original_name}'. "
                    "Only function arguments and compile-time constants are supported.",
                    consuming_loc,
                )

            local_value = mapper.clone_var(value)
            constants.append(TypedConst(
                value=constant_value,
                result_vars=(local_value,),
                loc=value.loc,
            ))

        if constants:
            for op in region_op.body.operations:
                op.remap_operands(mapper)
            region_op.body[:0] = constants


def verify_reduce_scan_isolation(root_block: Block) -> None:
    """Verify that reduce and scan bodies only use available region-local values."""
    for region_op in root_block.traverse():
        if not isinstance(region_op, TileReduce | TileScan):
            continue

        local_names = {var.name for var in region_op.body.params}
        local_names.update(
            result.name
            for op in region_op.body.operations
            for result in op.result_vars
        )
        available = {var.name for var in region_op.body.params}
        for op in region_op.body.operations:
            for operand in op.all_inputs():
                if operand.name not in available:
                    original_name = operand.get_original_name()
                    problem = (
                        "is used before its definition"
                        if operand.name in local_names
                        else "is defined outside the region"
                    )
                    raise TileInternalError(
                        f"{_kind(region_op)} body value '{original_name}' {problem}",
                        op.loc,
                    )
            available.update(result.name for result in op.result_vars)
