# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from contextlib import contextmanager
from functools import singledispatchmethod, partial
from typing import Sequence

from cuda.lang._ir.ops import IfElse, EndBranch, Loop, Continue, Break
from cuda.lang._ir.ir import IRContext, Region, Block, TileBlock, Operation, Var
from cuda.lang._ir.type import ScalarTy
from cuda.lang._datatype import bool_
from cuda.lang._ir.ops import (
    Branch,
    CondBranch,
    RawComparisonOperation,
    RawBinaryArithmeticOperation,
)
from cuda.lang._exception import InternalError


def _checked_getattr(self, attr: str) -> Var | Block:
    if (attr_value := getattr(self, attr)) is not None:
        return attr_value
    raise InternalError(f"self.{attr} is unexpectedly None")


class CFGFlattener:
    """
    Flattens the control flow graph of a TileBlock into a list of blocks.
    """

    def __init__(self, ctx: IRContext):
        self.ctx = ctx
        self.region = Region(ctx)
        self._merge: Block | None = None
        self._loop_break: Block | None = None
        self._loop_continue: Block | None = None
        self._loop_iv: Var | None = None
        self._loop_step: Var | None = None

        for attribute in (
            "merge",
            "loop_break",
            "loop_continue",
            "loop_iv",
            "loop_step",
        ):
            setattr(
                self.__class__,
                attribute,
                property(
                    partial(_checked_getattr, attr="_" + attribute),
                ),
            )

    def __call__(self, body: TileBlock) -> Region:
        current = self.ctx.make_block("entry", body.loc, body.params)
        self.region.blocks.append(current)
        self.flatten_ops(body.operations, current)
        return self.region

    def flatten_ops(
        self, ops: TileBlock | Sequence[Operation], current: Block
    ) -> Block:
        for op in ops:
            current = self.flatten(op, current)
        return current

    @singledispatchmethod
    def flatten(self, op: Operation, current: Block) -> Block:
        current.append(op)
        return current

    def is_in_for_loop(self):
        # When a while-loop is entered, we push the loop state and the step
        # is None, so checking the step tells us if the most recent loop
        # was a for or not.
        return self._loop_step is not None

    @flatten.register
    def flatten_continue(self, op: Continue, current: Block) -> Block:
        args = op.values
        if self.is_in_for_loop():
            next_iv = self.ctx.make_temp()
            self.ctx.copy_type_information(self.loop_step, next_iv)
            current.append(
                RawBinaryArithmeticOperation(
                    (next_iv,),
                    op.loc,
                    fn="add",
                    lhs=self.loop_iv,
                    rhs=self.loop_step,
                )
            )
            args = (next_iv,) + args

        current.append(Branch((), op.loc, target=self.loop_continue, args=args))
        return current

    @flatten.register
    def flatten_break(self, op: Break, current: Block) -> Block:
        current.append(Branch((), op.loc, target=self.loop_break, args=op.values))
        return current

    @flatten.register
    def flatten_loop(self, op: Loop, current: Block) -> Block:
        if op.is_for_loop:
            return self.flatten_for_loop(op, current)
        else:
            return self.flatten_while_loop(op, current)

    def flatten_for_loop(self, op: Loop, current: Block) -> Block:
        loop_header_bb = self.ctx.make_block("header", op.loc, op.body.params)
        loop_bb = self.ctx.make_block("loop", op.loc, ())
        loop_exit_bb = self.ctx.make_block("exit", op.loc, op.result_vars)
        iv = op.induction_var
        carried_vars = op.body.params[1:]

        current.append(
            Branch(
                (),
                op.loc,
                target=loop_header_bb,
                args=(op.start,) + op.initial_values,
            )
        )

        cv = self.ctx.make_temp()
        cv.set_type(ScalarTy(bool_))

        # Tile's range object requires the step to be positive so
        # we can always use "lt" here.
        loop_header_bb.append(
            RawComparisonOperation(
                (cv,),
                op.loc,
                fn="lt",
                lhs=iv,
                rhs=op.stop,
            )
        )
        loop_header_bb.append(
            CondBranch(
                (),
                op.loc,
                cond=cv,
                true_args=(),
                false_args=carried_vars,
                true_target=loop_bb,
                false_target=loop_exit_bb,
            )
        )

        with self.push_loop_state(
            loop_break=loop_exit_bb,
            loop_continue=loop_header_bb,
            loop_iv=iv,
            loop_step=op.step,
        ):
            self.region.blocks.append(loop_header_bb)
            self.region.blocks.append(loop_bb)
            self.flatten_ops(op.body, loop_bb)
            self.region.blocks.append(loop_exit_bb)

        return loop_exit_bb

    def flatten_while_loop(self, op: Loop, current: Block) -> Block:
        loop_bb = self.ctx.make_block("loop", op.loc, op.body.params)
        loop_exit_bb = self.ctx.make_block("exit", op.loc, op.result_vars)

        current.append(Branch((), op.loc, target=loop_bb, args=op.initial_values))

        with self.push_loop_state(loop_break=loop_exit_bb, loop_continue=loop_bb):
            self.region.blocks.append(loop_bb)
            self.flatten_ops(op.body, loop_bb)
            self.region.blocks.append(loop_exit_bb)

        return loop_exit_bb

    @flatten.register
    def flatten_ifelse(self, op: IfElse, current: Block) -> Block:
        then_bb = self.ctx.make_block("then", op.loc)
        else_bb = self.ctx.make_block("else", op.loc)
        phi_bb = self.ctx.make_block("phi", op.loc, op.result_vars)

        current.append(
            CondBranch(
                (),
                op.loc,
                cond=op.cond,
                true_args=(),
                false_args=(),
                true_target=then_bb,
                false_target=else_bb,
            )
        )

        with self.push_ifelse_state(merge=phi_bb):
            self.region.blocks.append(then_bb)
            self.flatten_ops(op.then_block, then_bb)

            self.region.blocks.append(else_bb)
            self.flatten_ops(op.else_block, else_bb)

            self.region.blocks.append(phi_bb)

        return phi_bb

    @flatten.register
    def flatten_endbranch(self, op: EndBranch, current: Block) -> Block:
        current.append(Branch((), op.loc, target=self.merge, args=op.outputs))
        return current

    @contextmanager
    def push_ifelse_state(
        self,
        merge: Block | None = None,
    ):
        prev_state = self._merge
        self._merge = merge
        yield
        self._merge = prev_state

    @contextmanager
    def push_loop_state(
        self,
        loop_break: Block | None = None,
        loop_continue: Block | None = None,
        loop_iv: Var | None = None,
        loop_step: Var | None = None,
    ):
        prev_state = (
            self._loop_break,
            self._loop_continue,
            self._loop_iv,
            self._loop_step,
        )
        self._loop_break = loop_break
        self._loop_continue = loop_continue
        self._loop_iv = loop_iv
        self._loop_step = loop_step
        yield
        self._loop_break, self._loop_continue, self._loop_iv, self._loop_step = (
            prev_state
        )


def flatten_cfg(body: TileBlock, ctx: IRContext) -> Region:
    flattener = CFGFlattener(ctx)
    region = flattener(body)
    return region


__all__ = ("flatten_cfg",)
