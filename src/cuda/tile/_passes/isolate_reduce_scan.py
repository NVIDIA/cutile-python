# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Keep custom reduce and scan callback bodies self-contained.

A reduce or scan callback is a pure combine function of its block parameters, so the body we
emit for it should not depend on anything from the enclosing scope. Compile-time constants the
callback uses are materialized inside the body, while captures of runtime values are rejected
with a source-level diagnostic.
"""

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
    """Return the values used in `body` but defined outside of it, with the location of their
    first use. Reduce and scan bodies contain no nested blocks, so one level suffices."""
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


def collect_reduce_scan_capture_names(root_block: Block) -> dict[str, str]:
    """Map the values captured by reduce/scan bodies to their source-level names.

    This must run before `eliminate_assign_ops`: a variable such as `m = ct.gather(...)` is an
    Assign of a temporary, and once the Assign is gone only the temporary's name is left for
    diagnostics. The keys are the names of the values that remain after Assign elimination.
    """
    definitions = _definitions(root_block)
    names: dict[str, str] = {}
    for region_op in root_block.traverse():
        if not isinstance(region_op, TileReduce | TileScan):
            continue
        for value, _ in _captures(region_op.body):
            canonical = value
            defining_op = definitions.get(canonical.name)
            while isinstance(defining_op, Assign):
                canonical = defining_op.value
                defining_op = definitions.get(canonical.name)
            names.setdefault(canonical.name, value.get_original_name())
    return names


def legalize_reduce_scan_captures(root_block: Block, capture_names: dict[str, str]) -> None:
    """Rematerialize constant captures inside each body and reject runtime captures.

    This must run after `materialize_constants_pass`: that pass emits every dataflow-proven
    constant at the start of the root block, which turns uses inside a callback body into
    captures. Cloning such constants back into the body keeps the body self-contained.
    `capture_names` comes from `collect_reduce_scan_capture_names`.
    """
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
                name = capture_names.get(value.name, value.get_original_name())
                raise TileSyntaxError(
                    f"{_kind(region_op)} body captures runtime value '{name}'. Only the "
                    "callback's own parameters and scalar compile-time constants are supported.",
                    consuming_loc,
                )

            # Shapes were validated when the body was built (see `_require_scalar_body` in
            # ops.py), so the capture is a scalar and can simply be cloned into the body.
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
    """Verify that reduce and scan bodies only use their parameters and body-local values."""
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
