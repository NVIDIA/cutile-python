# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import enum
import itertools
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from io import StringIO
from typing import Sequence, Optional, Literal, Union


class APInt:
    def __init__(self, value: int, width: int):
        assert -1 <= (value >> (width - 1)) <= 1
        self.value = value
        self.width = width

    def __int__(self):
        return self.value

    def __index__(self):
        return self.value

    def __bool__(self):
        return bool(self.value)

    def __repr__(self):
        return f"APInt(value={self.value}, width={self.width})"

    def __str__(self):
        return str(self.value)


class APFloat:
    def __init__(self, value: float):
        self.value = value

    def __float__(self):
        return self.value

    def __repr__(self):
        return f"APFloat({self.value})"

    def __str__(self):
        return str(self.value)


@dataclass
class AffineExpr:
    # Empty string represents a constant expression
    kind: Literal["s", "d", "", " + ", " * ", " floordiv ", " ceildiv ", " mod "]
    lhs: Union["AffineExpr", int]
    rhs: Optional["AffineExpr"] = None

    @staticmethod
    def symbol(position: int) -> "AffineExpr":
        return AffineExpr("s", position)

    @staticmethod
    def dim(position: int) -> "AffineExpr":
        return AffineExpr("d", position)

    @staticmethod
    def constant(value: int) -> "AffineExpr":
        return AffineExpr("", value)

    @staticmethod
    def _get(x: Union["AffineExpr", int]) -> "AffineExpr":
        if isinstance(x, int):
            return AffineExpr.constant(x)
        else:
            assert isinstance(x, AffineExpr)
            return x

    def __add__(self, other: Union["AffineExpr", int]) -> "AffineExpr":
        return AffineExpr(" + ", self, AffineExpr._get(other))

    def __radd__(self, other):
        return AffineExpr(" + ", AffineExpr._get(other), self)

    def print_mlir(self, p: "MlirPrinter"):
        if self.kind in ("s" | "m" | ""):
            p(self.kind)
            p(self.lhs)
        else:
            p("(")
            self.lhs.print_mlir(p)
            p(self.kind)
            self.rhs.print_mlir(p)
            p(")")


@dataclass
class AffineMap:
    num_dims: int
    num_symbols: int
    results: Sequence[AffineExpr]

    def is_identity(self) -> bool:
        return (self.num_dims == len(self.results)
                and all(e.kind == "d" and e.lhs == i for i, e in enumerate(self.results)))

    def print_mlir(self, p: "MlirPrinter"):
        p("(")
        prefix = "d"
        for i in range(self.num_dims):
            p(prefix)
            p(i)
            prefix = ", d"
        p(")")

        if self.num_symbols > 0:
            p("[")
            prefix = "s"
            for i in range(self.num_symbols):
                p(prefix)
                p(i)
                prefix = ", s"
            p("]")

        p(" -> (")
        comma = ""
        for expr in self.results:
            p(comma)
            expr.print_mlir(p)
            comma = ", "
        p(")")


class DenseResourceElementsHandle:
    pass


class IntegerSet:
    pass


@dataclass
class NamedAttribute:
    name: "Attribute"  # StringAttr
    value: "Attribute"

    @staticmethod
    def make(name: str, value: "Attribute"):
        from . import StringAttr
        name = StringAttr(value=name)
        return NamedAttribute(name, value)


class SignednessSemantics(enum.IntEnum):
    SIGNLESS = 0
    SIGNED = 1
    UNSIGNED = 2


class Attribute:
    def __init_subclass__(cls,
                          dialect: str | None = None,
                          mnemonic: str | None = None):
        cls._dialect = dialect
        cls._mnemonic = mnemonic

    def _print_mlir_unqualified(self, p: "MlirPrinter"):
        raise NotImplementedError()

    def print_mlir(self, p: "MlirPrinter", qualified: bool = True, elide_type: bool = False):
        body_io = StringIO()
        self._print_mlir_unqualified(MlirPrinter(body_io))
        body = body_io.getvalue()

        if qualified and self._dialect is not None:
            if body:
                p(f"#{self._dialect}<{self._mnemonic} ")
                p(body)
                p(">")
            else:
                p(f"#{self._dialect}.{self._mnemonic}")
        else:
            p(body)
        from . import NoneType, TypedAttr
        if not elide_type and isinstance(self, TypedAttr):
            ty = self.get_type()
            if not isinstance(ty, NoneType):
                p(" : ")
                ty.print_mlir(p)

    def __str__(self):
        io = StringIO()
        self.print_mlir(MlirPrinter(io))
        return io.getvalue()


class Type:
    def __init_subclass__(cls,
                          dialect: str | None = None,
                          mnemonic: str | None = None):
        cls._dialect = dialect
        cls._mnemonic = mnemonic

    def _print_mlir_unqualified(self, p: "MlirPrinter"):
        pass

    def print_mlir(self, p: "MlirPrinter"):
        if self._dialect is not None:
            p(f"!{self._dialect}.{self._mnemonic}")
        self._print_mlir_unqualified(p)

    def __str__(self):
        io = StringIO()
        self.print_mlir(MlirPrinter(io))
        return io.getvalue()


@dataclass(eq=False)
class Value:
    type: Type
    value_id: str | None = None


@dataclass(eq=False)
class BlockLabel:
    block_id: str | None = None


@dataclass
class Operation:
    name: str
    results: Sequence[Value]
    operands: Sequence[Value]
    properties: Sequence[tuple[str, Attribute]]
    attributes: Sequence[tuple[str, Attribute]]
    regions: Sequence["Region"]
    successors: Sequence[BlockLabel]
    location: Attribute | None = None

    def __str__(self):
        io = StringIO()
        MlirPrinter(io).print_operation(self)
        return io.getvalue()


@dataclass(eq=False)
class Block:
    args: Sequence[Value] = dataclasses.field(default_factory=list)
    operations: list[Operation] = dataclasses.field(default_factory=list)
    label: BlockLabel = dataclasses.field(default_factory=BlockLabel)

    @contextmanager
    def append_here(self):
        with _Cursor(self).as_current():
            yield self

    @contextmanager
    def prepend_here(self):
        with _Cursor(self, insert_pos=0).as_current():
            yield self

    def append(self, op: Operation):
        self.operations.append(op)

    def __getitem__(self, item):
        return self.operations[item]


@dataclass
class Region:
    blocks: list[Block] = dataclasses.field(default_factory=list)

    def new_block(self, args: Sequence[Value] = (), block_id: str | None = None) -> Block:
        block = Block(args=args, label=BlockLabel(block_id))
        self.blocks.append(block)
        return block


class _Cursor:
    def __init__(self, block: Block, insert_pos: int | None = None):
        self._block = block
        self._insert_pos = insert_pos

    @staticmethod
    def current() -> Optional["_Cursor"]:
        return _current_cursor.cursor

    @contextmanager
    def as_current(self):
        old = _current_cursor.cursor
        _current_cursor.cursor = self
        try:
            yield self
        finally:
            _current_cursor.cursor = old

    def add_operation(self,
                      name: str,
                      result_type: Type | None | Sequence[Type | None | Sequence[Type]],
                      operands: Sequence[Value] = (),
                      properties: Sequence[tuple[str, Attribute]] = (),
                      attributes: Sequence[tuple[str, Attribute]] = (),
                      regions: Sequence[Region] = (),
                      successors: Sequence[BlockLabel] = ()):
        if result_type is None:
            ret = None
            results = []
        elif isinstance(result_type, Type):
            ret = Value(result_type)
            results = [ret]
        else:
            ret = []
            results = []
            for r in result_type:
                if r is None:
                    ret.append(None)
                elif isinstance(r, Type):
                    v = Value(r)
                    ret.append(v)
                    results.append(v)
                else:
                    values = []
                    for t in r:
                        assert isinstance(t, Type)
                        values.append(Value(t))
                    ret.append(tuple(values))
                    results.extend(values)
            ret = tuple(ret)

        op = Operation(name=name, results=results, operands=operands, properties=properties,
                       attributes=attributes, regions=regions, successors=successors,
                       location=_current_location.location)
        if self._insert_pos is None:
            self._block.append(op)
        else:
            self._block.operations.insert(self._insert_pos, op)
            self._insert_pos += 1
        return ret


def add_operation(name: str,
                  result_type: Type | None | Sequence[Type | None | Sequence[Type]],
                  operands: Sequence[Value],
                  properties: Sequence[tuple[str, Attribute]],
                  attributes: Sequence[tuple[str, Attribute]] = (),
                  regions: Sequence[Region] = (),
                  successors: Sequence[BlockLabel] = ()):
    return _Cursor.current().add_operation(name, result_type, operands, properties, attributes,
                                           regions, successors)


class _CurrentCursor(threading.local):
    cursor: _Cursor | None = None


_current_cursor = _CurrentCursor()


class _CurrentLocation(threading.local):
    location: Attribute | None = None


_current_location = _CurrentLocation()


@contextmanager
def use_location(location: Attribute | None):
    old = _current_location.location
    _current_location.location = location
    try:
        yield
    finally:
        _current_location.location = old


class MlirPrinter:
    def __init__(self, file):
        self._f = file
        self._block_id_map = dict()
        self._value_id_map = dict()
        self._indent = ""
        self._temp_id_seq = itertools.count()

    def print_operation(self, op: Operation):
        self(self._indent)

        if len(op.results) > 0:
            comma = ""
            for v in op.results:
                self(comma)
                self.print_value_id(v)
                comma = ", "
            self(" = ")

        self(f'"{op.name}"(')

        # Operands
        comma = ""
        for v in op.operands:
            self(comma)
            self.print_value_id(v)
            comma = ", "
        self(")")

        # Successors
        if len(op.successors) > 0:
            self(" [")
            comma = ""
            for s in op.successors:
                self(comma)
                self.print_block_label(s)
                comma = ", "
            self("]")

        # Properties
        if len(op.properties) > 0:
            self(" <{")
            _print_named_attributes(op.properties, self)
            self("}>")

        # Regions
        if len(op.regions) > 0:
            self(" (")
            comma = ""
            for region in op.regions:
                self(comma)
                self.print_region(region)
                comma = ", "
            self(")")

        # Attributes
        if len(op.attributes) > 0:
            self(" {")
            _print_named_attributes(op.attributes, self)
            self("}")

        # Type signature: operands

        self(" : (")
        comma = ""
        for v in op.operands:
            self(comma)
            v.type.print_mlir(self)
            comma = ", "
        self(") -> ")

        # Type signature: results

        if len(op.results) == 1:
            from . import FunctionType
            parens = isinstance(op.results[0], FunctionType)
        else:
            parens = True

        if parens:
            self("(")

        comma = ""
        for v in op.results:
            self(comma)
            v.type.print_mlir(self)
            comma = ", "

        if parens:
            self(")")

        if op.location is not None:
            self(" loc(")
            op.location.print_mlir(self, qualified=False)
            self(")")

    def print_region(self, region: Region):
        if len(region.blocks) == 0:
            self("{}")
            return

        self("{\n")

        self.print_block(region.blocks[0], print_args=len(region.blocks[0].args) > 0)

        for block in region.blocks[1:]:
            self.print_block(block)

        self(f"{self._indent}}}")

    def print_block(self, block: Block, print_args: bool = True):
        if print_args:
            self(self._indent)
            self.print_block_label(block.label)
            if len(block.args) > 0:
                self(" (")
                comma = ""
                for arg in block.args:
                    self(comma)
                    self.print_value_id(arg)
                    self(" : ")
                    arg.type.print_mlir(self)
                    comma = ", "
                self(")")
            self(":\n")

        with self.indent():
            for op in block.operations:
                self.print_operation(op)
                self("\n")

    @contextmanager
    def indent(self):
        old_indent = self._indent
        self._indent += "  "
        try:
            yield
        finally:
            self._indent = old_indent

    def print_value_id(self, value: Value):
        if value.value_id is None:
            value_id = self._value_id_map.get(value)
            if value_id is None:
                value_id = next(self._temp_id_seq)
                self._value_id_map[value] = value_id
        else:
            value_id = value.value_id
        self(f"%{value_id}")

    def print_block_label(self, label: BlockLabel):
        if label.block_id is None:
            block_id = self._block_id_map.get(label)
            if block_id is None:
                block_id = next(self._temp_id_seq)
                self._block_id_map[label] = block_id
        else:
            block_id = label.block_id
        self(f"^{block_id}")

    def print_escaped_string(self, value: str):
        self('"')
        for c in value.encode():
            if c == 0x5c:  # backslash
                self("\\\\")
            elif 0x20 <= c <= 0x7e and c != 0x22:  # printable chars other than "
                self(chr(c))
            else:
                self(f"\\{c:02X}")
        self('"')

    def print_bit_enum(self,
                       value: int,
                       bit_groups: tuple[tuple[int, str], ...],
                       individual_bits: tuple[tuple[int, str], ...]):
        comma = ""
        for mask, s in bit_groups:
            if value & mask == mask:
                self(comma)
                self(s)
                comma = ","
                value &= ~mask
        for mask, s in individual_bits:
            if value & mask == mask:
                self(comma)
                self(s)
                comma = ","

    def __call__(self, s):
        print(s, end="", file=self._f)

    def if_present(self, val, call):
        if val is not None:
            call()

    def print_custom_PrettyLLVMType(self, t):
        t.print_mlir(self)

    def print_custom_FunctionTypes(self, params, var_arg):
        comma = ""
        for param in params:
            self(comma)
            param.print_mlir(self)
            comma = ", "
        if var_arg:
            self(comma)
            self("...")
        self(")")


def _print_named_attributes(named_attrs: Sequence[tuple[str, Attribute]], p):
    from . import UnitAttr
    comma = ""
    for name, attr in named_attrs:
        p(comma)
        p(name)
        if not isinstance(attr, UnitAttr):
            p(" = ")
            attr.print_mlir(p, True, False)
        comma = ", "
