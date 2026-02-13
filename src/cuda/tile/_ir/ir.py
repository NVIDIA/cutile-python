# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import dataclasses
import enum
import itertools
import threading
from collections import defaultdict
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from types import MappingProxyType
from typing import (
    List, Optional, Dict, Tuple, Any, TYPE_CHECKING, Sequence, Iterator
)
from .type import Type, InvalidType
from cuda.tile._exception import (
    TileTypeError, Loc, TileInternalError
)
from .. import TileSyntaxError
from .._context import TileContextConfig
from cuda.tile._bytecode.version import BytecodeVersion

if TYPE_CHECKING:
    from cuda.tile._ir2bytecode import BytecodeContext


class IRContext:
    def __init__(self, config: TileContextConfig, tileiras_version: BytecodeVersion):
        self._all_vars: Dict[str, str] = {}
        self._counter_by_name: Dict[str, Iterator[int]] = defaultdict(itertools.count)
        self._temp_counter = itertools.count()
        self.typemap: Dict[str, Type] = dict()
        self.constants: Dict[str, Any] = dict()
        self._loose_typemap: Dict[str, Type] = dict()
        self.config: TileContextConfig = config
        self._aggregate_values: Dict[str, Any] = dict()
        self.tileiras_version: BytecodeVersion = tileiras_version

    #  Make a Var with a unique name based on `name`.
    def make_var(self, name: str, loc: Loc, undefined: bool = False) -> Var:
        var_name = name
        while var_name in self._all_vars:
            var_name = f"{name}.{next(self._counter_by_name[name])}"
        self._all_vars[var_name] = name
        return Var(var_name, loc, self, undefined)

    def make_var_like(self, var: Var) -> Var:
        return self.make_var(self.get_original_name(var.name), var.loc, var.is_undefined())

    def make_temp(self, loc: Loc, undefined: bool = False) -> Var:
        return self.make_var(f"${next(self._temp_counter)}", loc, undefined=undefined)

    def get_original_name(self, var_name: str) -> str:
        return self._all_vars[var_name]

    def copy_type_information(self, src: Var, dst: Var):
        if src.name in self.typemap:
            self.typemap[dst.name] = self.typemap[src.name]
        if src.name in self._loose_typemap:
            self._loose_typemap[dst.name] = self._loose_typemap[src.name]
        if src.name in self.constants:
            self.constants[dst.name] = self.constants[src.name]
        if src.name in self._aggregate_values:
            self._aggregate_values[dst.name] = self._aggregate_values[src.name]


class ConstantState(enum.Enum):
    UNSET = 0
    MAY_BE_CONSTANT = 1
    NONCONSTANT = 2


@dataclass
class PhiState:
    ty: Type | None = None
    loose_ty: Type | None = None
    last_loc: Loc = Loc.unknown()
    initial_constant_state: ConstantState = ConstantState.UNSET

    # Constant propagation state, per aggregate item.
    # We initialize it to None because we don't know yet how many items we have.
    constant_state: list[ConstantState] | None = None
    constant_value: list[Any] | None = None

    def propagate(self, src: Var, fail_eagerly: bool = False, allow_loose_typing: bool = True):
        # Type & loose type propagation
        src_ty = src.get_type_allow_invalid()
        src_loose_ty = src.get_loose_type_allow_invalid() if allow_loose_typing else src_ty
        if self.ty is None:
            self.ty = src_ty
            self.loose_ty = src_loose_ty
            self.last_loc = src.loc
        elif isinstance(src_ty, InvalidType):
            if fail_eagerly and not isinstance(self.ty, InvalidType):
                raise TileTypeError(src_ty.error_message, src_ty.loc)
            self.ty = self.loose_ty = src_ty
        elif not isinstance(self.ty, InvalidType):
            var_name = src.get_original_name()
            if src_ty != self.ty:
                msg = (f"Type of `{var_name}` depends on path taken:"
                       f" {src_ty} (line {src.loc.line}) vs. {self.ty} (line {self.last_loc.line})")
                if fail_eagerly:
                    raise TileTypeError(msg, src.loc)
                else:
                    self.ty = self.loose_ty = InvalidType(msg, loc=src.loc)

            # If the loose types don't match exactly, "unify" them to the concrete type
            self.last_loc = src.loc
            if self.loose_ty != src_loose_ty:
                self.loose_ty = self.ty

        # Constant propagation
        if isinstance(src_ty, InvalidType):
            self.constant_state = None
            self.initial_constant_state = ConstantState.NONCONSTANT
        else:
            agg_items = tuple(src.flatten_aggregate())
            if self.constant_state is None:
                self.constant_state = [self.initial_constant_state for _ in range(len(agg_items))]
                self.constant_value = [None for _ in range(len(agg_items))]
            else:
                # This should be true because we already checked the type.
                # If the type matches, it must have the same aggregate length.
                assert len(self.constant_state) == len(agg_items)

            for i, item in enumerate(agg_items):
                if item.is_constant():
                    new_const = item.get_constant()
                    if self.constant_state[i] == ConstantState.UNSET:
                        self.constant_state[i] = ConstantState.MAY_BE_CONSTANT
                        self.constant_value[i] = new_const
                    elif (self.constant_state[i] == ConstantState.MAY_BE_CONSTANT
                          and new_const != self.constant_value[i]):
                        self.constant_state[i] = ConstantState.NONCONSTANT
                else:
                    self.constant_state[i] = ConstantState.NONCONSTANT

    def finalize_constant_and_loose_type(self, dst: Var):
        assert self.constant_state is not None
        for item, state, val in zip(dst.flatten_aggregate(),
                                    self.constant_state, self.constant_value, strict=True):
            if state == ConstantState.MAY_BE_CONSTANT:
                item.set_constant(val)
        dst.set_loose_type(self.loose_ty)


class AggregateValue:
    def as_tuple(self) -> tuple["Var", ...]:
        raise NotImplementedError()


class Var:
    def __init__(self, name: str, loc: Loc, ctx: IRContext, undefined: bool = False):
        self.name = name
        self.loc = loc
        self.ctx = ctx
        self._undefined = undefined

    def try_get_type(self) -> Optional[Type]:
        return self.ctx.typemap.get(self.name)

    def get_type(self) -> Type:
        ty = self.get_type_allow_invalid()
        if isinstance(ty, InvalidType):
            raise TileTypeError(ty.error_message, ty.loc)
        return ty

    def get_type_allow_invalid(self) -> Type:
        try:
            return self.ctx.typemap[self.name]
        except KeyError:
            if self._undefined:
                return InvalidType(f"Use of potentially undefined variable"
                                   f" `{self.get_original_name()}`", loc=Loc.unknown())
            raise TileInternalError(f"Type of variable {self.name} not found")

    def set_type(self, ty: Type, force: bool = False):
        assert isinstance(ty, Type)
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
        if ty is not None:
            return ty
        ty = self.get_type()
        if isinstance(ty, InvalidType):
            raise TileTypeError(ty.error_message, ty.loc)
        return ty

    def get_loose_type_allow_invalid(self) -> Type:
        ty = self.ctx._loose_typemap.get(self.name, None)
        return self.get_type_allow_invalid() if ty is None else ty

    def set_loose_type(self, ty: Type, force: bool = False):
        assert isinstance(ty, Type)
        if not force:
            assert self.name not in self.ctx._loose_typemap
        self.ctx._loose_typemap[self.name] = ty

    def is_undefined(self) -> bool:
        return self._undefined

    def set_undefined(self):
        self._undefined = True

    def get_original_name(self) -> str:
        return self.ctx.get_original_name(self.name)

    def is_aggregate(self) -> bool:
        return self.name in self.ctx._aggregate_values

    def get_aggregate(self) -> AggregateValue:
        return self.ctx._aggregate_values[self.name]

    def set_aggregate(self, agg_value: AggregateValue):
        self.ctx._aggregate_values[self.name] = agg_value

    def flatten_aggregate(self) -> Iterator["Var"]:
        if self.is_aggregate():
            for x in self.get_aggregate().as_tuple():
                yield from x.flatten_aggregate()
        else:
            yield self

    def __repr__(self):
        return f"Var<{self.name} @{self.loc}>"

    def __str__(self) -> str:
        return self.name


@dataclass
class TupleValue(AggregateValue):
    items: tuple[Var, ...]

    def as_tuple(self) -> tuple["Var", ...]:
        return self.items


@dataclass
class RangeValue(AggregateValue):
    start: Var
    stop: Var
    step: Var

    def as_tuple(self) -> tuple[Var, ...]:
        return self.start, self.stop, self.step


@dataclass
class BoundMethodValue(AggregateValue):
    bound_self: Var

    def as_tuple(self) -> tuple[Var, ...]:
        return (self.bound_self,)


@dataclass
class ArrayValue(AggregateValue):
    base_ptr: Var
    shape: tuple[Var, ...]
    strides: tuple[Var, ...]

    def as_tuple(self) -> tuple[Var, ...]:
        return self.base_ptr, *self.shape, *self.strides


@dataclass
class ListValue(AggregateValue):
    base_ptr: Var
    length: Var

    def as_tuple(self) -> tuple[Var, ...]:
        return self.base_ptr, self.length


@dataclass
class ClosureValue(AggregateValue):
    # Default values of parameters. These need to be carried by the closure's value
    # because default expressions are evaluated at definition time, not when the closure is called.
    # Should have the same length as the corresponding `ClosureTy.default_value_types`.
    default_values: tuple[Var, ...]

    # Tuple of the same length as `ty.func_hir.enclosing_functions`
    # and `ty.frozen_capture_types_by_depth`, where `ty` is the `ClosureTy` of this closure.
    #
    # For each depth `i`, `frozen_captures_by_depth[i]` is either:
    #   - None: means the enclosing function's LocalScope is still live;
    #   - tuple[Var, ...]: means the enclosing function's LocalScope is no longer live.
    #       The tuple contains the final values of the variables captured from the enclosing
    #       function's scope. Its length should be the same as `ty.func_hir.captures_by_depth`.
    frozen_captures_by_depth: tuple[tuple[Var, ...] | None, ...]

    def as_tuple(self) -> tuple["Var", ...]:
        return (
            *self.default_values,
            *(v for values in self.frozen_captures_by_depth
              if values is not None for v in values)
        )


class MemoryEffect(enum.IntEnum):
    # Int value assigned here is meaningful.
    # It implies the relative strength of memory effects.
    # For example, NONE < LOAD < STORE.
    NONE = 0
    LOAD = 1
    STORE = 2


class Mapper:
    def __init__(self, ctx: IRContext, preserve_vars: bool = False):
        self._ctx = ctx
        self._var_map: Dict[str, Var] = dict()
        self._preserve_vars = preserve_vars

    def is_empty(self):
        return len(self._var_map) == 0

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


def add_operation(op_class,
                  result_ty: Type | None | Tuple[Type | None, ...],
                  **attrs_and_operands) -> Var | Tuple[Var, ...]:
    return Builder.get_current().add_operation(op_class, result_ty, attrs_and_operands)


def make_aggregate(value: AggregateValue,
                   ty: Type | None,
                   loose_ty: Type | None = None):
    return Builder.get_current().make_aggregate(value, ty, loose_ty)


@dataclass
class LoopVarState:
    body_phi: PhiState
    result_phi: PhiState

    def finalize_loopvar_type(self, body_var: Var):
        # Body type should always be populated from the initial value's type.
        assert self.body_phi.ty is not None

        # "While" loop without a "break" statement. Should perhaps be rejected as an infinite loop,
        # but this is not this function's responsibility. Just use the body variable's type.
        if self.result_phi.ty is None:
            return

        # Ignore Invalid types (unless both are Invalid, in which case we preserve one of them).
        if isinstance(self.result_phi.ty, InvalidType):
            pass
        elif isinstance(self.body_phi.ty, InvalidType):
            # It is OK to overwrite the body variable's type because the loop body has
            # already been type-inferred by this point.
            body_var.set_type(self.result_phi.ty, force=True)
            body_var.set_loose_type(self.result_phi.loose_ty, force=True)
        elif self.body_phi.ty != self.result_phi.ty:
            # TODO: split the variable into two to actually support this case?
            var_name = body_var.get_original_name()
            raise TileTypeError(f"Variable {var_name} has changed its type inside a loop"
                                f" from {self.body_phi.ty} (line {self.body_phi.last_loc.line})"
                                f" to {self.result_phi.ty} (line {self.result_phi.last_loc.line}")


class Builder:
    def __init__(self, ctx: IRContext, loc: Loc, reduction_body: bool = False,
                 scan_body: bool = False):
        self.ir_ctx = ctx
        self.is_terminated = False
        self._loc = loc
        self._ops = []
        self._entered = False
        self._prev_builder = None
        self._var_map: Dict[str, Var] = dict()
        self.reduction_body = reduction_body
        self.scan_body = scan_body

    def add_operation(self, op_class,
                      result_ty: Type | None | Tuple[Type | None, ...],
                      attrs_and_operands,
                      result: Var | Sequence[Var] | None = None) -> Var | Tuple[Var, ...]:
        if (self.reduction_body or self.scan_body) and op_class.memory_effect != MemoryEffect.NONE:
            if self.reduction_body:
                msg = "Operations with memory effects are not supported inside reduction body"
            else:
                msg = "Operations with memory effects are not supported inside scan body"
            raise TileSyntaxError(msg)

        assert not self.is_terminated
        force_type = False
        if isinstance(result_ty, tuple):
            if result is None:
                result = tuple(self.ir_ctx.make_temp(self._loc) for _ in result_ty)
            else:
                result = tuple(result)
                assert all(isinstance(v, Var) for v in result)
                force_type = True

            for var, ty in zip(result, result_ty, strict=True):
                if ty is not None:
                    var.set_type(ty, force=force_type)

            result_vars = result
        else:
            if result is None:
                result = self.ir_ctx.make_temp(self._loc)
            else:
                assert isinstance(result, Var)
                force_type = True
            if result_ty is not None:
                result.set_type(result_ty, force=force_type)

            result_vars = (result,)

        new_op = op_class(**attrs_and_operands, loc=self._loc, result_vars=result_vars)
        self._ops.append(new_op)
        if new_op.is_terminator:
            self.is_terminated = True
        return result

    def make_aggregate(self,
                       value: AggregateValue,
                       ty: Type | None,
                       loose_ty: Type | None = None,
                       result_var: Var | None = None) -> Var:
        force_type = True
        if result_var is None:
            result_var = self.ir_ctx.make_temp(self._loc)
            force_type = False

        if ty is not None:
            result_var.set_type(ty, force=force_type)
        if loose_ty is not None:
            result_var.set_loose_type(ty, force=force_type)
        result_var.set_aggregate(value)
        return result_var

    @property
    def ops(self) -> list[Operation]:
        return self._ops

    @property
    def loc(self) -> Loc:
        return self._loc

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


@contextmanager
def enter_nested_block(loc: Loc, reduction_body: bool = False, scan_body: bool = False):
    prev_builder = Builder.get_current()
    block = Block(prev_builder.ir_ctx, loc=loc)
    with Builder(prev_builder.ir_ctx, loc,
                 reduction_body=reduction_body or prev_builder.reduction_body,
                 scan_body=scan_body or prev_builder.scan_body) as builder:
        yield block
    block.extend(builder.ops)


class _CurrentBuilder(threading.local):
    builder = None


_current_builder = _CurrentBuilder()


class _FieldKind(enum.IntEnum):
    OPERAND = 0
    ATTRIBUTE = 1
    NESTED_BLOCK = 2


_FIELD_METADATA_KEY = "operation_field_kind"


def attribute(*, default=dataclasses.MISSING) -> dataclasses.Field:
    return dataclasses.field(default=default, metadata={_FIELD_METADATA_KEY: _FieldKind.ATTRIBUTE},
                             kw_only=True)


def operand(*, default=dataclasses.MISSING) -> dataclasses.Field:
    return dataclasses.field(default=default, metadata={_FIELD_METADATA_KEY: _FieldKind.OPERAND},
                             kw_only=True)


def nested_block() -> dataclasses.Field:
    return dataclasses.field(metadata={_FIELD_METADATA_KEY: _FieldKind.NESTED_BLOCK},
                             kw_only=True)


def _get_result_vars_tuple_for_single_result_op(self):
    return (self.result_var,)


@dataclass(eq=False)
class Operation:
    result_vars: tuple[Var, ...]
    loc: Loc

    def __init_subclass__(cls,
                          opcode: str,
                          terminator: bool = False,
                          memory_effect: MemoryEffect = MemoryEffect.NONE):
        cls._opcode = opcode
        cls._is_terminator = terminator
        cls.memory_effect = memory_effect

        operand_names = []
        attribute_names = []
        nested_block_names = []
        for field_name in cls.__annotations__.keys():
            f = getattr(cls, field_name, None)
            kind = f.metadata.get(_FIELD_METADATA_KEY) if isinstance(f, dataclasses.Field) else None
            if kind == _FieldKind.OPERAND:
                operand_names.append(field_name)
            elif kind == _FieldKind.ATTRIBUTE:
                attribute_names.append(field_name)
            elif kind == _FieldKind.NESTED_BLOCK:
                nested_block_names.append(field_name)
            else:
                raise TypeError(f"Field {field_name} of {cls} must be annotated with either"
                                f" operand(), attribute() or nested_block()")

        cls._operand_names = tuple(operand_names)
        cls._attribute_names = tuple(attribute_names)
        cls._nested_block_names = tuple(nested_block_names)

    def __post_init__(self):
        for var in self.all_inputs():
            assert isinstance(var, Var | tuple) or var is None
            if isinstance(var, tuple):
                assert all(isinstance(x, Var) for x in var)

            if isinstance(var, Var) and var.is_aggregate() and self.op != "assign":
                # Don't allow aggregate values as operands, except for arrays and lists.
                # All other aggregates should only exist in the HIR level.
                # Also make an exception for the Assign op, until we find a better way to handle it.
                agg_val = var.get_aggregate()
                assert isinstance(agg_val, ArrayValue | ListValue)

        for nb in self.nested_blocks:
            assert isinstance(nb, Block)

    def clone(self, mapper: Mapper) -> Operation:
        result_vars = mapper.clone_vars(self.result_vars)
        return self._clone_impl(mapper, result_vars)

    def _clone_impl(self, mapper: Mapper, result_vars: Sequence[Var]) -> Operation:
        new_fields = {}

        for name in self._attribute_names:
            new_fields[name] = getattr(self, name)

        for name in self._operand_names:
            var = getattr(self, name)
            if isinstance(var, Var):
                new_var = mapper.get_var(var)
            elif var is None:
                new_var = None
            else:
                new_var = tuple(mapper.get_var(v) for v in var)
            new_fields[name] = new_var

        for name in self._nested_block_names:
            old_block = getattr(self, name)
            new_block = Block(old_block.ctx, old_block.loc)
            new_block.params = mapper.clone_vars(old_block.params)
            for old_op in old_block:
                new_block.append(old_op.clone(mapper))
            new_fields[name] = new_block

        return type(self)(result_vars=tuple(result_vars), loc=self.loc, **new_fields)

    @property
    def op(self) -> str:
        return self._opcode

    @property
    def operands(self) -> Mapping[str, Var | Tuple[Var, ...]]:
        return MappingProxyType({name: getattr(self, name) for name in self._operand_names})

    @property
    def attributes(self):
        return MappingProxyType({name: getattr(self, name) for name in self._attribute_names})

    @property
    def nested_blocks(self):
        return tuple(getattr(self, name) for name in self._nested_block_names)

    def all_inputs(self) -> Iterator[Var]:
        for name in self._operand_names:
            x = getattr(self, name)
            if isinstance(x, tuple):
                yield from iter(x)
            elif x is not None:
                yield x

    @property
    def is_terminator(self) -> bool:
        return self._is_terminator

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

    def generate_bytecode(self, ctx: "BytecodeContext"):
        raise NotImplementedError(f"Operation {self.op} must implement generate_bytecode")

    def _to_string_block_prefixes(self) -> List[str]:
        return []

    def _to_string_rhs(self) -> str:
        operands_str_list = []
        for name, val in self.operands.items():
            if isinstance(val, Var):
                operands_str_list.append(f"{name}={var_aggregate_name(val)}")
            elif isinstance(val, tuple) and all(isinstance(v, Var) for v in val):
                tup_str = ', '.join(var_aggregate_name(v) for v in val)
                operands_str_list.append(f"{name}=({tup_str})")
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
            params_str = (" (" + ", ".join(format_var(v) for v in nested_block.params) + ")"
                          if len(nested_block.params) > 0 else "")
            result += f"\n{indent_str}{block_prefix}{params_str}\n" if block_prefix else "\n"
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


def var_aggregate_name(var: Var) -> str:
    ret = var.name
    if var.is_aggregate():
        ret += "{" + ", ".join(x.name for x in var.flatten_aggregate()) + "}"
    return ret


def format_var(var: Var) -> str:
    ret = var_aggregate_name(var)

    ty = var.try_get_type()
    if ty is not None:
        const_prefix = "const " if var.is_constant() else ""
        ret += f": {const_prefix}{ty}"

    return ret


class Block:
    def __init__(self, ctx: IRContext, loc: Loc):
        self.ctx = ctx
        self._operations: List[Operation] = []
        self.params = ()
        self.loc = loc

    def empty_like_self(self: "Block") -> "Block":
        return Block(self.ctx, self.loc)

    def append(self, op: Operation):
        self._operations.append(op)
        op.parent_block = self

    def extend(self, ops: Sequence[Operation]):
        self._operations.extend(ops)
        for op in ops:
            op.parent_block = self

    def __len__(self):
        return len(self._operations)

    def __iter__(self):
        return iter(self._operations)

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
        params = ", ".join(format_var(p) for p in self.params)
        ops = "\n".join(
            op.to_string(
                indent,
                highlight_loc,
                include_loc
            ) for op in self.operations
        )
        return f"{' ' * indent}({params}):\n{ops}"

    def traverse(self) -> Iterator[Operation]:
        for op in self.operations:
            for b in op.nested_blocks:
                yield from b.traverse()
            yield op

    def __str__(self) -> str:
        return self.to_string()


@dataclass
class Function:
    body: Block
    name: str
    loc: Loc


@dataclass
class KernelArgument:
    type: Type
    is_const: bool = False
    const_value: Any = None
