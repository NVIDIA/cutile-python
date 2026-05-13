# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from contextlib import contextmanager
from dataclasses import dataclass, field
import itertools
from collections import defaultdict

from cuda.tile._ir.ir import (
    Block as TileBlock,
    Builder as TileBuilder,
    IRContext as TileIRContext,
    Operation,
    Var,
    Loc,
    operand,
    attribute,
    add_operation,
    format_var,
    AggregateValue,
)


class Builder:
    def __init__(self, region: "Region", loc: Loc):
        self._region = region
        self._loc = loc
        self._builder = TileBuilder(region.ctx, loc)
        self._current_block = None

    @property
    def ctx(self) -> "IRContext":
        return self._region.ctx

    @property
    def loc(self) -> Loc:
        return self._loc

    @property
    def region(self) -> "Region":
        return self._region

    def __enter__(self):
        self._builder.__enter__()
        return self

    def __exit__(self, *args):
        self._flush()
        self._builder.__exit__(*args)

    def _flush(self):
        if self._current_block is not None and self._builder.ops:
            self._current_block.extend(self._builder.ops)
            self._builder._ops.clear()

    @contextmanager
    def block_builder(self, block: "Block"):
        self._flush()
        self._current_block = block
        self.region.blocks.append(block)
        self._builder.is_terminated = False
        try:
            yield
        finally:
            self._flush()
            self._current_block = None


class Block(TileBlock):
    @property
    def _name(self) -> str:
        return self.ctx.block_name(self)

    def to_string(
        self,
        indent: int = 0,
        highlight_loc: Loc | None = None,
        include_loc: bool = False,
    ) -> str:
        params = ", ".join(format_var(p) for p in self.params)
        ops = "\n".join(
            op.to_string(indent + 4, highlight_loc, include_loc)
            for op in self.operations
        )
        return f"{' ' * indent}^{self._name}({params}):\n{ops}"


class IRContext(TileIRContext):
    def __init__(self, log_ir_on_error: bool = True):
        self._block_names: dict[int, str] = {}
        self._block_counter: dict[str, itertools.count] = defaultdict(itertools.count)
        super().__init__(log_ir_on_error, tileiras_version=None)

    def make_block(self, name: str, loc: Loc, params: tuple[Var, ...] = ()) -> Block:
        block = Block(self, loc)
        block.params = params
        if (counter := next(self._block_counter[name])) > 0:
            name = f"{name}.{counter}"
        self._block_names[block] = name
        return block

    def block_name(self, block: Block) -> str:
        assert block in self._block_names, f"Block {block} not found in context"
        return self._block_names[block]


@dataclass(eq=False)
class Region:
    ctx: IRContext
    blocks: list[Block] = field(default_factory=list)

    def to_string(
        self,
        indent: int = 0,
        highlight_loc: Loc | None = None,
        include_loc: bool = False,
    ) -> str:
        lines = []

        for block in self.blocks:
            lines.append(block.to_string(indent, highlight_loc, include_loc))

        return "\n".join(lines)

    def __str__(self) -> str:
        return self.to_string()


class LocalArrayContextManagerValue(AggregateValue):
    pass


__all__ = (
    "Block",
    "Builder",
    "TileBuilder",
    "IRContext",
    "Loc",
    "Operation",
    "Region",
    "Var",
    "add_operation",
    "attribute",
    "format_var",
    "operand",
)
