# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import itertools
import threading
from collections import OrderedDict
from collections import defaultdict
from collections.abc import Mapping
from contextlib import contextmanager
from copy import copy
from dataclasses import dataclass
from types import MappingProxyType
from typing import (
    List, Optional, Dict, Tuple, Set, Any, TYPE_CHECKING, Sequence, Iterator
)

from typing_extensions import override

from .type import Type, UNDEFINED
from .typing_support import typeof_pyval, get_constant_value, loose_type_of_pyval
from cuda.tile._exception import (
    TileTypeError,
    TileValueError,
    Loc, TileInternalError
)

if TYPE_CHECKING:
    from cuda.tile._ir2bytecode import BytecodeContext
    from cuda.tile._passes.typeinfer import TypingContext


@dataclass
class RangeInfo:
    known_step: int


class IRContext:
    def __init__(self):
        self._all_vars: Dict[str, str] = {}
        self._counter_by_name: Dict[str, Iterator[int]] = defaultdict(itertools.count)
        self._temp_counter = itertools.count()
        self.typemap: Dict[str, Type] = dict()
        self.constants: Dict[str, Any] = dict()
        self._loose_typemap: Dict[str, Type] = dict()
        self.range_infos: Dict[str, RangeInfo] = dict()

    #  Make a Var with a unique name based on `name`.
    def make_var(self, name: str, loc: Loc, undefined: bool = False) -> Var:
        var_name = name
        while var_name in self._all_vars:
            var_name = f"{name}.{next(self._counter_by_name[name])}"
        self._all_vars[var_name] = name
        return Var(var_name, loc, self, undefined)

    def make_var_like(self, var: Var) -> Var:
        return self.make_var(self.get_original_name(var.name), var.loc, var.is_undefined())

    def make_temp(self, loc: Loc) -> Var:
        return self.make_var(f"${next(self._temp_counter)}", loc)

    def get_original_name(self, var_name: str) -> str:
        return self._all_vars[var_name]

    def copy_type_information(self, src: Var, dst: Var):
        if src.name in self.typemap:
            self.typemap[dst.name] = self.typemap[src.name]
        if src.name in self._loose_typemap:
            self._loose_typemap[dst.name] = self._loose_typemap[src.name]
        if src.name in self.constants:
            self.constants[dst.name] = self.constants[src.name]
        if src.name in self.range_infos:
            self.range_infos[dst.name] = self.range_infos[src.name]


class Var:
    def __init__(self, name: str, loc: Loc, ctx: IRContext, undefined: bool = False):
        self.name = name
        self.loc = loc
        self.ctx = ctx
        self._undefined = undefined

    def try_get_type(self) -> Optional[Type]:
        return self.ctx.typemap.get(self.name)

    def get_type(self) -> Type:
        try:
            return self.ctx.typemap[self.name]
        except KeyError:
            if self._undefined:
                return UNDEFINED
            raise TileInternalError(f"Type of variable {self.name} not found")

    def set_type(self, ty: Type, force: bool = False):
        if not force:
            assert self.name not in self.ctx.typemap
        self.ctx.typemap[self.name] = ty

    def is_constant(self) -> bool:
        return self.name in self.ctx.constants

    def get_constant(self):
        return self.ctx.constants[self.name]

    def set_constant(self, value):
        assert self.name not in self.ctx.constants
        self.ctx.constants[self.name] = value

    def get_loose_type(self) -> Type:
        ty = self.ctx._loose_typemap.get(self.name, None)
        return self.get_type() if ty is None else ty

    def set_loose_type(self, ty: Type, force: bool = False):
        if not force:
            assert self.name not in self.ctx._loose_typemap
        self.ctx._loose_typemap[self.name] = ty

    def has_range_info(self) -> bool:
        return self.name in self.ctx.range_infos

    def get_range_info(self) -> RangeInfo:
        return self.ctx.range_infos[self.name]

    def set_range_info(self, range_info: RangeInfo):
        assert self.name not in self.ctx.range_infos
        self.ctx.range_infos[self.name] = range_info

    def is_undefined(self) -> bool:
        return self._undefined

    def set_undefined(self):
        self._undefined = True

    def __str__(self) -> str:
        return self.name


TypeResult = list[Type] | Type


def terminator(cls):
    cls._is_terminator = True
    return cls


def has_side_effects(cls):
    cls._has_side_effects = True
    return cls


class Mapper:
    def __init__(self, ctx: IRContext, preserve_vars: bool = False):
        self._ctx = ctx
        self._var_map: Dict[str, Var] = dict()
        self._object_map = dict()
        self._preserve_vars = preserve_vars

    def clone_var(self, var: Var) -> Var:
        if self._preserve_vars:
            return self.get_var(var)
        else:
            new_var = self._ctx.make_var_like(var)
            self._var_map[var.name] = new_var
            self._ctx.copy_type_information(var, new_var)
            return new_var

    def clone_vars(self, vars: Sequence[Var]) -> Tuple[Var, ...]:
        return tuple(self.clone_var(v) for v in vars)

    def get_var(self, old_var: Var) -> Var:
        return self._var_map.get(old_var.name, old_var)

    def set_var(self, old_var: Var, new_var: Var):
        assert old_var.name not in self._var_map
        self._var_map[old_var.name] = new_var

    def get_object(self, old):
        return self._object_map.get(old, old)

    def set_object(self, old, new):
        assert old not in self._object_map
        self._object_map[old] = new


def add_operation(op_class,
                  result_ty: Type | None | Tuple[Type | None, ...],
                  **attrs_and_operands) -> Var | Tuple[Var, ...]:
    builder = Builder.get_current()
    if isinstance(result_ty, tuple):
        ret = tuple(builder._ctx.make_temp(builder._loc) for _ in result_ty)
        for var, ty in zip(ret, result_ty, strict=True):
            if ty is not None:
                var.set_type(ty)
        if len(ret) > 0:
            attrs_and_operands["result_vars"] = ret
    else:
        ret = builder._ctx.make_temp(builder._loc)
        if result_ty is not None:
            ret.set_type(result_ty)
        attrs_and_operands["result_var"] = ret

    builder._ops.append(op_class(**attrs_and_operands, loc=builder._loc))
    return ret


class Builder:
    def __init__(self, ctx: IRContext, loc: Loc):
        self._ctx = ctx
        self._loc = loc
        self._ops = []
        self._entered = False
        self._prev_builder = None

    @property
    def ops(self) -> Sequence[Operation]:
        return self._ops

    def append_verbatim(self, op: Operation):
        self._ops.append(op)

    def extend_verbatim(self, ops: Sequence[Operation]):
        self._ops.extend(ops)

    @staticmethod
    def get_current() -> "Builder":
        ret = _current_builder.builder
        assert ret is not None, "No IR builder is currently active"
        return ret

    @contextmanager
    def change_loc(self, loc: Loc):
        old_loc = self._loc
        self._loc = loc
        try:
            yield
        finally:
            self._loc = old_loc

    def __enter__(self):
        assert not self._entered
        self._prev_builder = _current_builder.builder
        _current_builder.builder = self
        self._entered = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        assert self._entered
        _current_builder.builder = self._prev_builder
        self._prev_builder = None
        self._entered = False


class _CurrentBuilder(threading.local):
    builder = None


_current_builder = _CurrentBuilder()


class Operation:
    _has_side_effects = False

    def __init__(
        self,
        op: str,
        operands: dict[str, Optional[Var | Tuple[Var, ...]]],
        result_vars: List[Var],
        attributes: Optional[Dict[str, Any]] = None,
        nested_blocks: Optional[List[Block]] = None,
        loc: Loc = Loc.unknown(),
    ):
        self.op = op
        self.result_vars = result_vars or []
        self.attributes = attributes or {}
        self.nested_blocks = nested_blocks or []
        self.loc = loc

        self._operands = OrderedDict()
        for k, v in operands.items():
            self._add_operand(k, v)
        self._is_terminator = getattr(self.__class__, "_is_terminator", False)
        self._parent_block = None

    def clone(self, mapper: Mapper) -> Operation:
        assert len(self.nested_blocks) == 0
        result_vars = mapper.clone_vars(self.result_vars)
        return self._clone_impl(mapper, result_vars)

    def _clone_impl(self, mapper: Mapper, result_vars: Sequence[Var]) -> Operation:
        new_nested_blocks = []
        for old_block in self.nested_blocks:
            new_block = Block(old_block.ctx)
            for old_op in old_block:
                new_block.append(old_op.clone(mapper))
            new_nested_blocks.append(new_block)

        ret = copy(self)
        ret._operands = OrderedDict()
        for name, var in self._operands.items():
            if isinstance(var, Var):
                ret._operands[name] = mapper.get_var(var)
            elif var is None:
                ret._operands[name] = None
            else:
                ret._operands[name] = tuple(mapper.get_var(v) for v in var)

        ret.attributes = dict(ret.attributes)
        ret.result_vars = result_vars
        ret.parent_block = None
        return ret

    @property
    def operands(self) -> Mapping[str, Var | Tuple[Var, ...]]:
        return MappingProxyType(self._operands)

    def all_inputs(self) -> Iterator[Var]:
        for x in self._operands.values():
            if isinstance(x, tuple):
                yield from iter(x)
            elif x is not None:
                yield x

    @property
    def is_terminator(self) -> bool:
        return self._is_terminator

    @property
    def has_side_effects(self) -> bool:
        return self._has_side_effects

    def _add_operand(self, name: str, var: Var | Tuple[Var, ...]):
        self._operands[name] = var

    def update_operand(self, name: str, var: Var | Tuple[Var, ...]):
        self._add_operand(name, var)

    def __getattr__(self, name: str) -> Any:
        if name == "__setstate__":
            raise AttributeError(name)

        if name in self.operands:
            return self.operands[name]
        if name in self.attributes:
            return self.attributes[name]
        raise AttributeError(f"{self.__class__.__name__} has no operand or attribute {name}")

    @property
    def result_var(self) -> Var:
        if len(self.result_vars) != 1:
            raise ValueError(f"Operation {self.op} has {len(self.result_vars)} results")
        return self.result_vars[0]

    @property
    def parent_block(self) -> Block:
        return self._parent_block

    @parent_block.setter
    def parent_block(self, block: Block):
        self._parent_block = block

    def infer_type(self, typing_context: "TypingContext") -> TypeResult:
        raise NotImplementedError(f"Operation {self.op} must implement infer_type")

    def generate_bytecode(self, ctx: "BytecodeContext"):
        raise NotImplementedError(f"Operation {self.op} must implement generate_bytecode")

    def _to_string_block_prefixes(self) -> List[str]:
        return []

    def _to_string_rhs(self) -> str:
        operands_str_list = []
        for name, val in self.operands.items():
            if isinstance(val, Var):
                operands_str_list.append(f"{name}={str(val)}")
            elif isinstance(val, tuple) and all(isinstance(v, Var) for v in val):
                operands_str_list.append(f"{name}=({', '.join(str(v) for v in val)})")
            elif val is None:
                operands_str_list.append(f"{name}=None")
            else:
                raise ValueError(f"Unexpected operand type: {type(val)}")
        operands_str = ", ".join(operands_str_list)
        if self.attributes:
            attr_parts = []
            for attr_name, attribute in self.attributes.items():
                if isinstance(attribute, str):
                    attr_parts.append(f'{attr_name}="{attribute}"')
                else:
                    attr_parts.append(f'{attr_name}={attribute}')
            attr_str = ", ".join(attr_parts)
        else:
            attr_str = ""
        delimiter_str = ", " if self.operands and self.attributes else ""
        return f"{self.op}({operands_str}{delimiter_str}{attr_str})"

    def to_string(self,
                  indent: int = 0,
                  highlight_loc: Optional[Loc] = None,
                  include_loc: bool = False) -> str:
        def format_var(var: Var):
            ty = var.try_get_type()
            if ty is None:
                return var.name
            else:
                const_prefix = "const " if var.is_constant() else ""
                return f"{var.name}: {const_prefix}{ty}"

        indent_str = " " * indent
        lhs = (
            ", ".join(format_var(var) for var in self.result_vars)
            if self.result_vars
            else ""
        )
        rhs = self._to_string_rhs()
        loc_str = f"  // {self.loc}" if include_loc and self.loc else ""

        result = f"{indent_str}{lhs + ' = ' if lhs else ''}{rhs}{loc_str}"

        block_prefixs = self._to_string_block_prefixes()
        if len(block_prefixs) != len(self.nested_blocks):
            raise ValueError(
                f"Operation {self.op} has {len(block_prefixs)} block prefixes, "
                f"but {len(self.nested_blocks)} nested blocks"
            )
        for block_prefix, nested_block in zip(block_prefixs, self.nested_blocks):
            result += f"\n{indent_str}{block_prefix}\n" if block_prefix else "\n"
            block_str = nested_block.to_string(
                indent + 4,
                include_loc=include_loc
            )
            result += f"{block_str}"

        if highlight_loc is not None and self.loc == highlight_loc:
            return f"\033[91m{result}\033[0m"
        return result

    def __str__(self) -> str:
        return self.to_string()


class AstOperation(Operation):
    def __init__(
        self,
        op: str,
        operands: OrderedDict[str, Var | Tuple[Var, ...]],
        result_vars: List[Var],
        attributes: Optional[Dict[str, Any]] = None,
        nested_blocks: Optional[List[Block]] = None,
        loc: Optional[Loc] = None,
    ):
        super().__init__(op, operands, result_vars, attributes, nested_blocks, loc)

    @override
    def infer_type(self, typing_context: TypingContext) -> List[Type] | Type:
        raise NotImplementedError(f"AstOperation {self.op} should not be used in type inference")

    @override
    def generate_bytecode(self, ctx: BytecodeContext):
        raise NotImplementedError(f"AstOperation {self.op}"
                                  f" should not be used in bytecode generation")


class TypedOperation(Operation):
    """
    Marker base class for operations that are created with type information already set.
    Type inference pass will skip these.
    """


class Block:
    def __init__(self, ctx: IRContext):
        self.ctx = ctx
        self._operations: List[Operation] = []

    def append(self, op: Operation):
        self._operations.append(op)
        op.parent_block = self

    def extend(self, ops: Sequence[Operation]):
        self._operations.extend(ops)
        for op in ops:
            op.parent_block = self

    def __len__(self):
        return len(self._operations)

    def __getitem__(self, i):
        return self._operations[i]

    def __setitem__(self, i, value):
        if isinstance(i, slice):
            self._replace(i, value)
        else:
            self._replace(slice(i, i + 1), (value,))

    def __delitem__(self, i):
        self._replace(i if isinstance(i, slice) else slice(i, i + 1), ())

    def _replace(self, s: slice, new_ops: Sequence[Operation]):
        for op in new_ops:
            op.parent_block = self
        self._operations[s] = new_ops

    def detach_all(self):
        ret, self._operations = self._operations, []
        for op in ret:
            op.parent_block = None
        return ret

    @property
    def operations(self) -> Sequence[Operation]:
        return tuple(self._operations)

    @operations.setter
    def operations(self, ops: Sequence[Operation]):
        # Clear parent links on the old ops
        for old in self._operations:
            old.parent_block = None

        # Replace list and re-link parents
        self._operations = list(ops)
        for op in self._operations:
            op.parent_block = self

    def make_temp_var(self, loc: Loc) -> Var:
        return self.ctx.make_temp(loc)

    def make_temp_vars(self, loc: Loc, count: int) -> Tuple[Var, ...]:
        return tuple(self.ctx.make_temp(loc) for _ in range(count))

    def to_string(self,
                  indent: int = 0,
                  highlight_loc: Optional[Loc] = None,
                  include_loc: bool = False) -> str:
        op_strings = (
            op.to_string(
                indent,
                highlight_loc,
                include_loc
            ) for op in self.operations
        )
        return "\n".join(op_strings)

    def traverse(self) -> Iterator[Operation]:
        for op in self.operations:
            for b in op.nested_blocks:
                yield from b.traverse()
            yield op

    def __str__(self) -> str:
        return self.to_string()


@dataclass(frozen=True, slots=True, repr=False)
class Function:
    qualname: str
    root_block: Block
    parameters: Tuple[Var, ...]
    return_value: Optional[Var]  # Only set for helper functions
    loc: Loc

    def bind_arguments(self, args: Tuple[Any, ...], constant_args: Set[str])\
            -> Tuple[Argument, ...]:
        # TODO: unify this logic with dispatcher from c extension
        # Refactor "extract_cuda_args" to return type descriptor
        # that can be wrapped as IR Type for type inference.
        if len(args) != len(self.parameters):
            msg = f"Expected {len(self.parameters)} arguments, got {len(args)}"
            raise TileValueError(msg, self.loc)

        ir_args = []
        for param, arg_value in zip(self.parameters, args):
            const_val = None
            is_const = param.name in constant_args
            ty = typeof_pyval(arg_value)
            loose_type = ty
            if is_const:
                try:
                    const_val = get_constant_value(arg_value)
                except TileTypeError:
                    raise TileTypeError(
                        f"Argument {param.name} is a constexpr, "
                        f"but the value is not a supported constant.", self.loc)
                loose_type = loose_type_of_pyval(arg_value)
            ir_args.append(Argument(type=ty,
                                    loose_type=loose_type,
                                    is_const=is_const,
                                    const_value=const_val))
        return tuple(ir_args)

    def to_string(self,
                  indent: int = 0,
                  highlight_loc: Optional[Loc] = None,
                  include_loc: bool = False) -> str:
        def format_param(p: Var):
            ty = p.try_get_type()
            if ty is not None:
                const_prefix = "const " if p.is_constant() else ""
                return f"{p.name}: {const_prefix}{ty}"
            else:
                return p.name

        args_str = ", ".join(format_param(p) for p in self.parameters)
        loc_str = f"  // {self.loc}" if include_loc and self.loc else ""
        header = f"func @{self.qualname}({args_str}):{loc_str}"
        body = self.root_block.to_string(
            indent=indent+4,
            highlight_loc=highlight_loc,
            include_loc=include_loc
        )
        return f"{header}\n{body}"

    def __str__(self) -> str:
        return self.to_string()


class Argument:
    def __init__(self,
                 type: Type,
                 loose_type: Type,
                 is_const: bool = False,
                 const_value: Any = None):
        self._type = type
        self._loose_type = loose_type
        self._is_const = is_const
        self._const_value = const_value

    @property
    def is_const(self) -> bool:
        return self._is_const

    @property
    def const_value(self) -> Any:
        return self._const_value

    @property
    def type(self) -> Type:
        return self._type

    @property
    def loose_type(self) -> Type:
        return self._loose_type

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, Argument):
            return False
        return (
            self.type == value.type and
            self.is_const == value.is_const and
            self.const_value == value.const_value
        )
