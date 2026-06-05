# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import pytest

import cuda.lang as cl  # noqa: F401
from cuda.lang._ir.ir import IRContext, Region, Loc, Builder
from cuda.lang._ir.ops import branch, cond_branch

# Importing this module registers cuda.lang's lowerings
import cuda.lang._ir.ops as cl_ops  # noqa: F401

from cuda.tile._ir.control_flow_ops import Return
from cuda.tile._ir.ir import add_operation_variadic
from .util import filecheck, get_source


@pytest.fixture
def builder():
    ctx = IRContext(log_ir_on_error=False)
    loc = Loc.unknown()
    region = Region(ctx)
    with Builder(region, loc) as builder:
        yield builder


def test_branch_ir_construction(builder):
    ctx = builder.ctx
    loc = builder.loc

    x, y, phi, cond = (
        ctx.make_var(name=name, loc=loc) for name in ["x", "y", "phi", "cond"]
    )

    entry = ctx.make_block("entry", loc, params=(cond, x, y))
    then = ctx.make_block("then", loc)
    else_ = ctx.make_block("else", loc)
    merge = ctx.make_block("merge", loc, params=(phi,))

    # CHECK: ^entry(cond, x, y):
    # CHECK:     cond_br cond ^then() ^else()
    with builder.block_builder(entry):
        cond_branch(
            cond=cond,
            true_args=(),
            false_args=(),
            true_target=then,
            false_target=else_,
        )

    # CHECK: ^then():
    # CHECK:     br ^merge(x)
    with builder.block_builder(then):
        branch(merge, (x,))

    # CHECK: ^else():
    # CHECK:     br ^merge(y)
    with builder.block_builder(else_):
        branch(merge, (y,))

    # CHECK: ^merge(phi):
    # CHECK:     return
    with builder.block_builder(merge):
        add_operation_variadic(Return, ())

    filecheck(str(builder.region), get_source())


def test_branch_loc_propagation():
    ctx = IRContext(log_ir_on_error=False)
    loc = Loc(filename="some_file.py", line=75, col=4)
    region = Region(ctx)

    with Builder(region, loc) as builder:
        x, y, phi, cond = (
            ctx.make_var(name=name, loc=loc) for name in ["x", "y", "phi", "cond"]
        )

        # CHECK: ^entry(cond, x, y):
        with builder.block_builder(ctx.make_block("entry", loc, params=(cond, x, y))):
            # CHECK-NEXT: cond_br cond ^then(x, y) ^else(y, x)
            # CHECK-SAME: some_file.py:75:4
            cond_branch(
                cond=cond,
                true_args=(x, y),
                false_args=(y, x),
                true_target=ctx.make_block("then", loc),
                false_target=ctx.make_block("else", loc),
            )

    filecheck(builder.region.to_string(include_loc=True), get_source())
