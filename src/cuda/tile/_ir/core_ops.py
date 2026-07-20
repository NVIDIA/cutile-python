# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
import dataclasses
import functools
import operator
from dataclasses import dataclass
from types import MethodType, FunctionType, BuiltinFunctionType, MappingProxyType
from typing import Any, Optional, Sequence

from typing_extensions import override

import cuda.tile._bytecode as bc
from cuda.tile import TileTypeError
import cuda.tile._datatype as datatype
from cuda.tile._bytecode import float_to_bits
from cuda.tile._bytecode.float import float_from_bits
from cuda.tile._datatype import numeric_dtype_category, is_integral
from cuda.tile._exception import Loc, TileSyntaxError, TileValueError, TypeCheckingError
from cuda.tile._ir import hir_stubs, hir
from cuda.tile._ir.hir import ResolvedName
from cuda.tile._ir.ir import Operation, attribute, Var, Builder, make_aggregate, operand, \
    MemoryEffect, add_operation_variadic
from cuda.tile._ir.op_impl import ImplRegistry, require_dataclass_type, require_constant_str, \
    OverloadNotFoundError, WILDCARD, require_signed_integer_scalar_type, require_constant_slice, \
    require_constant_int, PrintfValidator, ensure_string_or_formatted_string
from cuda.tile._ir.scope import Scope
from cuda.tile._ir.type import Type, DTypeSpec, TensorLikeTy, TupleTy, TupleValue, Symbol, \
    DataclassInfo, DataclassTy, DataclassValue, BoundMethodValue, BoundMethodTy, InvalidType, \
    ContextManagerTy, ContextManagerLifecycle, LiveCapturedScope, ClosureTy, ClosureValue, \
    RangeIterType, RangeValue, TypeTy, ModuleTy, NONE, SliceType, StringTy, FormattedStringTy, \
    StringFormat, FormattedStringValue, FormattedPiece, DictTy, DictValue, EnumTy, TokenTy
from cuda.tile._ir.typing_support import type_of_constant_python_value, \
    loose_type_of_constant_python_value, get_dataclass_info, as_third_party_dtype_spec, \
    create_dataclass_instance, find_method, dataclass_has_default_repr
from cuda.tile._ir2bytecode import BytecodeContext
from cuda.tile._mutex import tile_mutex

_registry = ImplRegistry()
impl = _registry.impl
overload_dispatcher = _registry.overload_dispatcher


def core_impl_registry() -> ImplRegistry:
    return _registry


# ===========================================================================================
# Overload dispatchers
# ===========================================================================================

@overload_dispatcher(operator.add, fixed_args=["+"])
@overload_dispatcher(operator.sub, fixed_args=["-"])
@overload_dispatcher(operator.mul, fixed_args=["*"])
@overload_dispatcher(operator.floordiv, fixed_args=["//"])
@overload_dispatcher(operator.truediv, fixed_args=["/"])
@overload_dispatcher(operator.pow, fixed_args=["**"])
@overload_dispatcher(operator.mod, fixed_args=["%"])
@overload_dispatcher(operator.eq, fixed_args=["=="])
@overload_dispatcher(operator.ne, fixed_args=["!="])
@overload_dispatcher(operator.lt, fixed_args=["<"])
@overload_dispatcher(operator.le, fixed_args=["<="])
@overload_dispatcher(operator.gt, fixed_args=[">"])
@overload_dispatcher(operator.ge, fixed_args=[">="])
@overload_dispatcher(operator.and_, fixed_args=["&"])
@overload_dispatcher(operator.or_, fixed_args=["|"])
@overload_dispatcher(operator.xor, fixed_args=["^"])
@overload_dispatcher(operator.lshift, fixed_args=["<<"])
@overload_dispatcher(operator.rshift, fixed_args=[">>"])
@overload_dispatcher(operator.matmul, fixed_args=["@"])
@overload_dispatcher(hir_stubs.is_contained_in, fixed_args=["'in'"])
@overload_dispatcher(min, fixed_args=["min"])
@overload_dispatcher(max, fixed_args=["max"])
@overload_dispatcher(divmod, fixed_args=["divmod"])
def binop_overload_dispatcher(name: str, x: Var, y: Var):
    x_ty = x.get_type()
    y_ty = y.get_type()
    try:
        yield type(x_ty), type(y_ty)
    except OverloadNotFoundError:
        raise TileTypeError(f"Unsupported operand types for {name}: {x_ty} and {y_ty}")


@impl(hir_stubs.is_not_contained_in)
async def is_not_contained_in_impl(x: Var, y: Var):
    from .._passes.hir2ir import call_function
    contained = await call_function(hir_stubs.is_contained_in, x, y)
    return await call_function(operator.not_, contained)


def comparison_operator_impl(registry: ImplRegistry, lhs_ty: type[Type], rhs_ty: type[Type]):
    def decorate(func):
        for name in ("eq", "ne", "lt", "le", "gt", "ge"):
            registry.impl(getattr(operator, name), fixed_args=[name],
                          overload=(lhs_ty, rhs_ty))(func)
        return func

    return decorate


@overload_dispatcher(operator.not_, fixed_args=["not"])
@overload_dispatcher(operator.pos, fixed_args=["+"])
@overload_dispatcher(operator.invert, fixed_args=["~"])
@overload_dispatcher(operator.neg, fixed_args=["-"])
def unary_overload_dispatcher(name: str, x: Var):
    x_ty = x.get_type()
    try:
        yield (type(x_ty),)
    except OverloadNotFoundError:
        raise TileTypeError(f"Unsupported operand type for {name}: {x_ty}")


@overload_dispatcher(getattr)
def getattr_overload_dispatcher(object: Var, name: Var):
    ty = object.get_type()
    attr_name = require_constant_str(name)
    try:
        yield type(ty), attr_name
    except OverloadNotFoundError:
        raise TileTypeError(f"No such attribute '{attr_name}' for object of type {ty}")


@overload_dispatcher(operator.getitem)
def getitem_overload_dispatcher(object: Var, key: Var):
    object_ty = object.get_type()
    key_ty = key.get_type()
    try:
        yield type(object_ty), type(key_ty)
    except OverloadNotFoundError as e:
        if e.found_overload_matching_first_param:
            raise TileTypeError(f"Object of type {object_ty} is not subscriptable with {key_ty}")
        else:
            raise TileTypeError(f"Object of type {object_ty} is not subscriptable")


@overload_dispatcher(operator.setitem)
def setitem_overload_dispatcher(object: Var, key: Var, value: Var):
    object_ty = object.get_type()
    key_ty = key.get_type()
    value_ty = value.get_type()
    try:
        yield type(object_ty), type(key_ty), type(value_ty)
    except OverloadNotFoundError as e:
        if e.found_overload_matching_first_param:
            raise TileTypeError(f"Object of type {object_ty} does not support"
                                f" item assignment with key type {key_ty}"
                                f" and value type {value_ty}")
        else:
            raise TileTypeError(f"Object of type {object_ty} does not support item assignment")


@overload_dispatcher(len)
def len_overload_dispatcher(x: Var):
    ty = x.get_type()
    try:
        yield (type(ty),)
    except OverloadNotFoundError:
        raise TileTypeError(f"Object of type {ty} has no len()")


@overload_dispatcher(str)
def str_overload_dispatcher(x: Var):
    ty = x.get_type()
    try:
        yield (type(ty),)
    except OverloadNotFoundError:
        raise TileTypeError(f"Object of type {ty} has no str()")


@impl(str, overload=(WILDCARD,))
async def str_fallback_impl(x: Var):
    if x.is_constant():
        return loosely_typed_const(str(x.get_constant()))

    from cuda.tile._passes.hir2ir import call_function
    return await call_function(repr, x)


@impl(str, overload=(StringTy,))
@impl(str, overload=(FormattedStringTy,))
def str_str_impl(x: Var[StringTy]):
    return x


@impl(str, overload=(DataclassTy,))
async def str_dataclass_impl(x: Var[DataclassTy]):
    from cuda.tile._passes.hir2ir import call_function

    cls = x.get_type().cls
    method = find_method(cls, "__str__")
    if method is object.__str__ or method is NotImplemented:
        # Fall back to repr()
        return await call_function(repr, x)

    res = await call_function(method, x)
    if not isinstance(res.get_type(), StringTy | FormattedStringTy):
        raise TypeCheckingError(f"Expected {cls.__name__}.__str__()"
                                f" to return a string, got {res.get_type()}")
    return res


@overload_dispatcher(repr)
def repr_overload_dispatcher(x: Var):
    ty = x.get_type()
    try:
        yield (type(ty),)
    except OverloadNotFoundError:
        raise TileTypeError(f"Object of type {ty} has no repr()")


@impl(repr, overload=(WILDCARD,))
def repr_fallback_impl(x: Var):
    if x.is_constant():
        return loosely_typed_const(repr(x.get_constant()))
    return loosely_typed_const(f"<object of type {x.get_type()}>")


@impl(repr, overload=(FormattedStringTy,))
def repr_formatted_str_impl(x: Var[FormattedStringTy]):
    val = x.get_aggregate()
    assert isinstance(val, FormattedStringValue)
    have_single_quote = have_double_quote = False
    for piece in val.format.pieces:
        if isinstance(piece, str):
            if "'" in piece:
                have_single_quote = True
            if '"' in piece:
                have_double_quote = True
    quote = '"' if have_single_quote and not have_double_quote else "'"

    builder = FormattedStringBuilder()
    builder.append_literal_piece(quote)

    for piece in val.format.pieces:
        if isinstance(piece, FormattedPiece):
            builder.append_formatted_piece(val.values[piece.value_idx], piece.format_spec)
        else:
            assert isinstance(piece, str)
            builder.append_literal_piece(_str_escape_for_repr(piece, quote))

    builder.append_literal_piece(quote)
    return builder.build()


def _str_escape_for_repr(s: str, quote: str) -> str:
    # HACK: Add the other kind of quote to force Python to pick the quote we want
    other_quote = "'" if quote == '"' else '"'
    r = repr(other_quote + s)
    assert r.startswith(quote)
    assert r.endswith(quote)
    r = r[1:-1]
    # The extra quote we added could have been escaped
    if r.startswith("\\"):
        r = r[1:]
    assert r.startswith(other_quote)
    return r[1:]


@impl(repr, overload=(TensorLikeTy,))
def repr_tensorlike_impl(x: Var[TensorLikeTy]):
    if x.is_constant():
        return loosely_typed_const(str(x.get_constant()))

    builder = FormattedStringBuilder()
    builder.append_formatted_piece(x, None)
    return builder.build()


@impl(repr, overload=(TupleTy,))
async def repr_tuple_impl(x: Var[TupleTy]):
    from cuda.tile._passes.hir2ir import call_function
    agg = x.get_aggregate()
    assert isinstance(agg, TupleValue)
    builder = FormattedStringBuilder()
    builder.append_literal_piece("(")
    comma = ""
    for item in agg.items:
        r = await call_function(repr, item)
        builder.append_literal_piece(comma)
        builder.append_string_var(r)
        comma = ", "
    builder.append_literal_piece(",)" if len(agg.items) == 1 else ")")
    return builder.build()


@impl(repr, overload=(DataclassTy,))
async def repr_dataclass_impl(x: Var[DataclassTy]):
    from cuda.tile._passes.hir2ir import call_function
    cls = x.get_type().cls

    # If __repr__ is the default generated implementation, emulate it
    if dataclass_has_default_repr(cls):
        agg = x.get_aggregate()
        assert isinstance(agg, DataclassValue)
        builder = FormattedStringBuilder()
        builder.append_literal_piece(f"{cls.__qualname__}(")
        comma = ""
        for f, item in zip(dataclasses.fields(cls), agg.items, strict=True):
            if not f.repr:
                continue
            r = await call_function(repr, item)
            builder.append_literal_piece(f"{comma}{f.name}=")
            builder.append_string_var(r)
            comma = ", "
        builder.append_literal_piece(")")
        return builder.build()

    method = find_method(cls, "__repr__")
    if method is object.__repr__ or method is NotImplemented:
        # Disabled repr: resort to the Python object-like default, but without an object ID.
        return loosely_typed_const(f"<{cls.__qualname__} object>")

    res = await call_function(method, x)
    if not isinstance(res.get_type(), StringTy | FormattedStringTy):
        raise TypeCheckingError(f"Expected {cls.__name__}.__repr__()"
                                f" to return a string, got {res.get_type()}")
    return res


@overload_dispatcher(tuple)
def tuple_overload_dispatcher(iterable: Var):
    ty = iterable.get_type()
    try:
        yield (type(ty),)
    except OverloadNotFoundError:
        raise TileTypeError(f"Object of type {ty} cannot be converted to a tuple")


@overload_dispatcher(hir_stubs.enter_context)
def enter_context_overload_dispatcher(manager: Var):
    ty = manager.get_type()
    if not isinstance(ty, ContextManagerTy):
        raise TileTypeError(f"Object of type {ty} cannot be used as a context manager")

    state = ty.get_context_manager_state()
    if state.lifecycle != ContextManagerLifecycle.FRESH:
        raise TileTypeError("Context manager cannot be reused")
    state.lifecycle = ContextManagerLifecycle.ENTERED
    Scope.get_current().context_stack.append(state)

    try:
        yield (type(ty),)
    except OverloadNotFoundError:
        raise TileTypeError(f"Object of type {ty} cannot be used as a context manager")

# ===========================================================================================


@impl(operator.is_)
def operator_is_impl(x: Var, y: Var):
    return _is_none_compare(x, y, negate=False, op_name="is")


@impl(operator.is_not)
def operator_is_not_impl(x: Var, y: Var):
    return _is_none_compare(x, y, negate=True, op_name="is not")


def _is_none_compare(x: Var, y: Var, *, negate: bool, op_name: str) -> Var:
    x_is_none = x.get_type() is NONE
    y_is_none = y.get_type() is NONE
    if not (x_is_none or y_is_none):
        raise TileTypeError(f"Operator '{op_name}' expects one of the operands to be None")
    return loosely_typed_const((x_is_none == y_is_none) ^ negate)


@dataclass(eq=False)
class TypedConst(Operation, opcode="typed_const"):
    value: Any = attribute()

    @override
    def generate_bytecode(self, ctx: BytecodeContext) -> bc.Value:
        return ctx.constant(self.value, ctx.typeof(self.result_var))


def loosely_typed_const(value: Any,
                        ty: Optional[Type] = None,
                        loose_ty: Optional[Type] = None,
                        result_var: Var | None = None) -> Var:
    builder = Builder.get_current()
    if ty is None:
        ty = type_of_constant_python_value(value, builder.ir_ctx.typing_hooks)
    assert not ty.is_aggregate(), "Use sym2var(value, constant_only=True) instead"

    # Normalize third party dtype spec objects (e.g. torch.float32 -> ct.float32)
    if isinstance(ty, DTypeSpec):
        value = ty.dtype

    ret = _strictly_typed_const_inner(builder, value, ty, result_var=result_var)
    if loose_ty is None:
        loose_ty = loose_type_of_constant_python_value(value, builder.ir_ctx.typing_hooks)
    ret.set_loose_type(loose_ty)
    return ret


def strictly_typed_const(value: Any, ty: Type) -> Var:
    return _strictly_typed_const_inner(Builder.get_current(), value, ty)


def _map_nested_tuple(func, value):
    return (tuple(_map_nested_tuple(func, x) for x in value)
            if isinstance(value, tuple) else func(value))


def _strictly_typed_const_inner(builder: Builder,
                                value: Any, ty: Type, result_var: Var | None = None) -> Var:
    if isinstance(ty, TensorLikeTy):
        dtype = ty.tensor_dtype()
        if is_integral(dtype):
            mask = -1 << dtype.bitwidth

            def truncate(x):
                x = int(x) & ~mask
                assert x >= 0
                if datatype.is_signed(dtype) and (x >> (dtype.bitwidth - 1)):
                    # High bit set? Need to sign-extend it.
                    x |= mask
                    assert x < 0
                return x

            value = _map_nested_tuple(truncate, value)
        elif datatype.is_float(dtype):
            bc_type = datatype.dtype_simple_bytecode_type(dtype)

            def round_float(x):
                x = float(x)
                try:
                    bits = float_to_bits(x, bc_type)
                except ValueError as e:
                    raise TileValueError(str(e))
                return float_from_bits(bits, bc_type)

            value = _map_nested_tuple(round_float, value)

    ret = builder.add_operation(TypedConst, ty, dict(value=value), result=result_var)
    if not isinstance(ty, TensorLikeTy) or ty.tensor_shape() == ():
        # We currently don't have a way to represent an N-dimensional tile constant
        ret.set_constant(value)
    return ret


@impl(float, fixed_args=[float])
@impl(int, fixed_args=[int])
@impl(bool, fixed_args=[bool])
def builtin_numeric_ctor_impl(ctor_obj: Any, x: Var) -> Var:
    if not x.is_constant():
        raise TileTypeError(f"{ctor_obj.__name__}() expects a constant argument")
    const = x.get_constant()
    try:
        value = ctor_obj(const)
    except (ValueError, TypeError, OverflowError):
        raise TileTypeError(f"Invalid argument for {ctor_obj.__name__}({const})")
    return loosely_typed_const(value)


# ===========================================================================================
# Tuple
# ===========================================================================================

def build_tuple(items: Sequence[Var], result_var: Var | None = None) -> Var:
    items = tuple(items)
    ty = TupleTy(tuple(x.get_type() for x in items))
    loose_ty = TupleTy(tuple(x.get_loose_type() for x in items))
    res = Builder.get_current().make_aggregate(TupleValue(items), ty, loose_ty,
                                               result_var=result_var)
    if all(x.is_constant() for x in items):
        res.set_constant(tuple(x.get_constant() for x in items))
    return res


@impl(hir_stubs.build_tuple)
def build_tuple_impl(items: tuple[Var, ...]) -> Var:
    return build_tuple(items)


def tuple_item(tup: TupleValue, index: int) -> Var:
    assert isinstance(tup, TupleValue)
    try:
        return tup.items[index]
    except IndexError:
        raise TileTypeError(
            f"Index {index} is out of range for a tuple of length {len(tup.items)}")


@impl(operator.getitem, overload=(TupleTy, TensorLikeTy))
def getitem_tuple_item_impl(object: Var[TupleTy], key: Var[TensorLikeTy]) -> Var:
    tuple_val = object.get_aggregate()
    assert isinstance(tuple_val, TupleValue)

    key_ty = key.get_type()
    if key_ty.tensor_shape() != () or not is_integral(key_ty.tensor_dtype()):
        raise TileTypeError(f"Tuple indices must be integers or slices, not {key_ty}")

    if not key.is_constant():
        raise TileTypeError("Tuple indices must be constant")

    idx = key.get_constant()
    return tuple_item(tuple_val, idx)


@impl(operator.getitem, overload=(TupleTy, SliceType))
def getitem_tuple_slice_impl(object: Var, key: Var) -> Var:
    tuple_val = object.get_aggregate()
    assert isinstance(tuple_val, TupleValue)

    slc = require_constant_slice(key)
    items = tuple_val.items[slc]
    return build_tuple(items)


@impl(operator.getitem, overload=(TupleTy, WILDCARD))
def getitem_tuple_fallback_impl(object: Var, key: Var) -> Var:
    key_ty = key.get_type()
    raise TileTypeError(f"Tuple indices must be integers or slices, not {key_ty}")


@impl(operator.setitem, overload=(TupleTy, WILDCARD, WILDCARD))
def setitem_tuple_impl(object: Var, key: Var, value: Var):
    raise TileTypeError("Tuples are immutable: item assignment is not supported.")


@impl(operator.getitem, overload=(DataclassTy, WILDCARD))
async def getitem_dataclass_impl(object: Var[DataclassTy], key: Var) -> Var:
    ty = object.get_type()
    dunder = find_method(ty.cls, "__getitem__")
    if dunder is NotImplemented:
        raise TileTypeError(f"'{ty.cls.__qualname__}' object is not subscriptable")
    from cuda.tile._passes.hir2ir import call_function
    return await call_function(dunder, object, key)


@impl(operator.setitem, overload=(DataclassTy, WILDCARD, WILDCARD))
async def setitem_dataclass_impl(object: Var[DataclassTy], key: Var, value: Var) -> Var:
    ty = object.get_type()
    dunder = find_method(ty.cls, "__setitem__")
    if dunder is NotImplemented:
        raise TileTypeError(f"'{ty.cls.__qualname__}' object does not support item assignment")
    from cuda.tile._passes.hir2ir import call_function
    return await call_function(dunder, object, key, value)


@comparison_operator_impl(_registry, TupleTy, TupleTy)
async def comparison_operator_tuple_impl(fn: str, x: Var[TupleTy], y: Var[TupleTy]) -> Var:
    if fn not in ("eq", "ne"):
        raise TileTypeError(f"Operator '{fn}' is not supported for tuples")

    x_ty = x.get_type()
    y_ty = y.get_type()

    if x.is_constant() and y.is_constant():
        res = x.get_constant() == y.get_constant()
        return loosely_typed_const(res if fn == "eq" else not res)

    if len(x_ty) != len(y_ty):
        return loosely_typed_const(fn == "ne")

    x_items = x.get_aggregate().items
    y_items = y.get_aggregate().items

    for item in (*x_items, *y_items):
        item_ty = item.get_type()
        if isinstance(item_ty, TensorLikeTy) and len(item_ty.tensor_shape()) > 0:
            raise TileTypeError("Tuple comparison is not supported for non-scalar elements")
        if not isinstance(item_ty, (TensorLikeTy, TupleTy, DTypeSpec, StringTy)):
            raise TileTypeError(
                f"Tuple comparison is not supported for elements of type {item_ty}"
            )

    from cuda.tile._passes.hir2ir import call_function
    from cuda.tile._ir.arithmetic_ops import binary_bitwise_tensorlike
    elem_cmps = [await call_function(operator.eq, xi, yi) for xi, yi in zip(x_items, y_items)]
    result = functools.reduce(lambda a, b: binary_bitwise_tensorlike("and_", a, b), elem_cmps,
                              loosely_typed_const(True))

    if fn == "ne":
        from cuda.tile._ir.arithmetic_ops import logical_not_impl
        result = logical_not_impl(result)

    return result


@impl(len, overload=(TupleTy,))
def len_tuple_impl(x: Var[TupleTy]) -> Var:
    return loosely_typed_const(len(x.get_type()))


@impl(hir_stubs.is_contained_in, overload=(WILDCARD, TupleTy))
async def is_contained_in_tuple_impl(x: Var, y: Var[TupleTy]) -> Var:
    from cuda.tile._passes.hir2ir import call_function
    from cuda.tile._ir.arithmetic_ops import binary_bitwise_tensorlike

    tuple_val = y.get_aggregate()
    assert isinstance(tuple_val, TupleValue)
    items = tuple_val.items

    result = None
    for item in items:
        cmp = await call_function(operator.eq, x, item)
        cmp_ty = cmp.get_type()
        if isinstance(cmp_ty, TensorLikeTy) and cmp_ty.tensor_shape() != ():
            raise TileTypeError(
                f"'in' requires scalar operands, but got shape {cmp_ty.tensor_shape()}")
        if cmp.is_constant() and cmp.get_constant():
            return cmp
        result = cmp if result is None else binary_bitwise_tensorlike("or_", result, cmp)
    return loosely_typed_const(False) if result is None else result


@impl(operator.add, overload=(TupleTy, TupleTy))
def add_tuple_impl(x: Var[TupleTy], y: Var[TupleTy]):
    x_items = x.get_aggregate().items
    y_items = y.get_aggregate().items
    return build_tuple(x_items + y_items)

# ===========================================================================================
# Dictionary
# ===========================================================================================


def build_dict(keys: tuple[str, ...], values: tuple[Var, ...]) -> Var:
    keys = tuple(keys)
    values = tuple(values)
    assert len(keys) == len(values)

    ty = DictTy(keys, tuple(x.get_type() for x in values))
    loose_ty = DictTy(keys, tuple(x.get_loose_type() for x in values))
    res = make_aggregate(DictValue(values), ty, loose_ty)
    if all(x.is_constant() for x in values):
        items = [(k, v.get_constant()) for k, v in zip(keys, values, strict=True)]
        res.set_constant(MappingProxyType(dict(items)))
    return res


def _find_dict_key_index(key: Var, dict_ty: DictTy) -> int | None:
    key_ty = key.get_type()
    if not isinstance(key_ty, StringTy):
        # Python would happily report that the key is not found when a "wrong" key type is passed,
        # but we can add a stronger check here.
        raise TileTypeError(f"Dictionary keys must be strings, not {key_ty}")

    return dict_ty.keys.index(key_ty.value) if key_ty.value in dict_ty.keys else None


@impl(hir_stubs.is_contained_in, overload=(WILDCARD, DictTy))
async def is_contained_in_dict_impl(x: Var, y: Var[DictTy]):
    return loosely_typed_const(_find_dict_key_index(x, y.get_type()) is not None)


@impl(getattr, overload=(DictTy, "get"))
def getattr_dict_method(object: Var, name: Var):
    name = require_constant_str(name)
    unbound_func = getattr(dict, name)
    return bind_method(object, unbound_func)


@impl(operator.getitem, overload=(DictTy, WILDCARD))
def getitem_dict_impl(object: Var[DictTy], key: Var):
    idx = _find_dict_key_index(key, object.get_type())
    if idx is None:
        raise TileTypeError(f"Key '{key.get_type().value}' not found in dictionary")
    dict_value = object.get_aggregate()
    assert isinstance(dict_value, DictValue)
    return dict_value.values[idx]


@impl(dict.get)
def dict_get_impl(self: Var, key: Var, default: Var):
    dict_ty = self.get_type()
    if not isinstance(dict_ty, DictTy):
        raise TileTypeError(f"dict.get() expects a dictionary as `self`, got {dict_ty}")

    idx = _find_dict_key_index(key, dict_ty)
    if idx is None:
        return default

    dict_value = self.get_aggregate()
    assert isinstance(dict_value, DictValue)
    return dict_value.values[idx]


# ===========================================================================================
# Dataclass
# ===========================================================================================

def build_dataclass_instance(items: tuple[Var, ...], info: DataclassInfo) -> Var:
    cls = info.cls
    ty = DataclassTy(cls, tuple(x.get_type() for x in items))
    loose_ty = DataclassTy(cls, tuple(x.get_loose_type() for x in items))
    res = make_aggregate(DataclassValue(items, info), ty, loose_ty)
    if all(x.is_constant() for x in items):
        items_values = [x.get_constant() for x in items]
        const_val = create_dataclass_instance(cls, items_values)
        res.set_constant(const_val)
    return res


@impl(dataclasses.replace)
def dataclasses_replace_impl(obj: Var, changes: dict[str, Var]):
    dataclass_ty = require_dataclass_type(obj)
    if get_dataclass_info(dataclass_ty.cls).init_signature is None:
        raise TypeCheckingError("dataclasses.replace() is only allowed for dataclasses with"
                                " a default generated __init__() method")
    dataclass_val = obj.get_aggregate()
    assert isinstance(dataclass_val, DataclassValue)
    name2idx = dataclass_val.info.field_name_to_idx
    new_items = list(dataclass_val.items)
    for name, val in changes.items():
        try:
            idx = name2idx[name]
        except KeyError:
            raise TileTypeError(f"Dataclass '{dataclass_ty.cls.__name__}'"
                                f" has no such field '{name}'")
        new_items[idx] = val
    return build_dataclass_instance(tuple(new_items), dataclass_val.info)

# ===========================================================================================


def bind_method(object: Var, func) -> Var:
    agg_value = BoundMethodValue(object)
    res_ty = BoundMethodTy(object.get_type(), func)
    return make_aggregate(agg_value, res_ty)


def sym2var(x: Any, constant_only: bool = False) -> Var:
    # TODO: verify we don't have a stale closure

    if isinstance(x, Symbol):
        if constant_only:
            raise TileTypeError("Cannot create a constant from a symbolic value")
        return x._var

    if isinstance(x, tuple):
        return build_tuple(tuple(sym2var(item, constant_only=constant_only) for item in x))

    cls = type(x)
    if dataclasses.is_dataclass(cls):
        info = get_dataclass_info(cls)
        field_vars = tuple(sym2var(getattr(x, f.name), constant_only=constant_only)
                           for f in dataclasses.fields(cls))
        return build_dataclass_instance(field_vars, info)

    if isinstance(x, MethodType):
        self_var = sym2var(x.__self__, constant_only=constant_only)
        if not isinstance(x.__func__, FunctionType | BuiltinFunctionType):
            raise TileTypeError(f"Object of type {type(x).__name__}"
                                f" cannot be used as a function for binding a method")
        return bind_method(self_var, x.__func__)

    # Transform a third party typed scalar (e.g., np.int16(5)) into a strictly typed constant
    dtype_spec = as_third_party_dtype_spec(type(x))
    if dtype_spec is not None:
        pyval = numeric_dtype_category(dtype_spec.dtype).pytype(x)
        ty = Builder.get_current().ir_ctx.typing_hooks.get_tensor_like_type(dtype_spec.dtype, ())
        return strictly_typed_const(pyval, ty)

    return loosely_typed_const(x)


@dataclass(eq=False)
class Assign(Operation, opcode="assign"):
    value: Var = operand()

    @override
    def generate_bytecode(self, ctx: BytecodeContext) -> bc.Value:
        # FIXME: Ideally, all Assign ops should be eliminated before the bytecode generation stage.
        #        But keep this for now just in case.
        return ctx.get_value(self.value)

    @override
    def _to_string_rhs(self) -> str:
        return f"{self.value.name}"


@impl(hir_stubs.identity)
def identity_impl(x: Var) -> Var:
    if x.is_constant():
        ty = x.get_type()
        if ty.is_aggregate():
            return make_aggregate(x.get_aggregate(), ty, x.get_loose_type())
        else:
            return loosely_typed_const(x.get_constant(), x.get_type(), x.get_loose_type())
    else:
        return x


def assign(value: Var, res: Var) -> None:
    Builder.get_current().append_verbatim(Assign(value=value, result_vars=(res,), loc=res.loc))
    res.ctx.copy_type_information(value, res)


def store_var(local_idx: int, value: Var, loc: Loc | None = None):
    scope = Scope.get_current()
    new_var = scope.local.redefine(local_idx, loc or Builder.get_current().loc)
    assign(value, new_var)


def store_invalid(local_idx: int, ty: Type, loc: Loc | None = None):
    assert isinstance(ty, InvalidType)
    scope = Scope.get_current()
    new_var = scope.local.redefine(local_idx, loc or Builder.get_current().loc)
    new_var.set_type(ty)


@impl(hir_stubs.store_var)
def store_var_impl(rn: ResolvedName, value: Var):
    store_var(rn.index, value)


@impl(hir_stubs.load_var)
def load_var_impl(rn, name: Var):
    scope = Scope.get_current()
    if rn.depth >= 0:
        ret = scope.local_scopes[rn.depth][rn.index]
        ret.get_type()  # Trigger an InvalidType check
        return ret
    elif rn.index >= 0:
        val = scope.func_hir.frozen_global_values[rn.index]
        return sym2var(val, constant_only=True)
    else:
        raise TileSyntaxError(f"Undefined variable {name.get_constant()} used")


@impl(hir_stubs.pop_context)
def pop_context_impl():
    ctx_state = Scope.get_current().context_stack.pop()
    ctx_state.lifecycle = ContextManagerLifecycle.EXITED
    ctx_state.exit_callback()


@impl(hir_stubs.make_closure)
def make_closure_impl(func_hir: hir.Function, default_values: tuple[Var, ...]):
    default_value_types = tuple(v.get_type() for v in default_values)

    frozen_captures_by_depth = []
    frozen_capture_types_by_depth = []
    captured_scopes = []

    builder = Builder.get_current()
    scope = Scope.get_current()
    for depth, (local_scope, captured_indices) in (enumerate(
            zip(scope.local_scopes, func_hir.captures_by_depth, strict=True))):
        if local_scope.frozen:
            frozen_vars = tuple(local_scope.get(idx, builder.loc) for idx in captured_indices)
            frozen_captures_by_depth.append(frozen_vars)
            frozen_types = tuple(v.get_type_allow_invalid() for v in frozen_vars)
            frozen_capture_types_by_depth.append(frozen_types)
        else:
            captured_scopes.append(LiveCapturedScope(depth, local_scope))
            frozen_captures_by_depth.append(None)
            frozen_capture_types_by_depth.append(None)

    closure_ty = ClosureTy(func_hir, default_value_types, tuple(captured_scopes),
                           tuple(frozen_capture_types_by_depth))
    closure_val = ClosureValue(default_values, tuple(frozen_captures_by_depth))
    return make_aggregate(closure_val, closure_ty)


# ===========================================================================================
# Getattr
# ===========================================================================================

@impl(getattr, overload=(ModuleTy, WILDCARD))
def getattr_module_impl(object: Var, name: Var):
    ty = object.get_type()
    attr_name = require_constant_str(name)
    try:
        return sym2var(getattr(ty.py_mod, attr_name), constant_only=True)
    except AttributeError:
        raise TileTypeError(f"Module '{ty.py_mod.__name__}' has no attribute '{attr_name}'")


@impl(getattr, overload=(TypeTy, WILDCARD))
def getattr_type_impl(object: Var, name: Var):
    ty = object.get_type()
    attr_name = require_constant_str(name)
    try:
        return sym2var(getattr(ty.ty, attr_name), constant_only=True)
    except AttributeError:
        raise TileTypeError(f"'{ty.ty.__name__}' object has no attribute '{attr_name}'")


@impl(getattr, overload=(DataclassTy, WILDCARD))
async def getattr_dataclass_impl(object: Var, name: Var):
    ty = object.get_type()
    val = object.get_aggregate()
    assert isinstance(val, DataclassValue)
    attr_name = require_constant_str(name)
    field_idx = val.info.field_name_to_idx.get(attr_name)
    if field_idx is not None:
        return val.items[field_idx]

    cls = ty.cls
    try:
        cls_attr = getattr(cls, attr_name)
    except AttributeError:
        raise TileTypeError(f"'{cls.__name__}' object has no attribute '{attr_name}'")

    if isinstance(cls_attr, FunctionType | BuiltinFunctionType):
        return bind_method(object, cls_attr)
    elif isinstance(cls_attr, property):
        from .._passes.hir2ir import call
        getter = loosely_typed_const(cls_attr.fget)
        return await call(getter, (object,), {})
    else:
        return sym2var(cls_attr, constant_only=True)


@impl(getattr, overload=(EnumTy, "name"))
def getattr_enum_name_impl(object: Var, name: Var):
    return sym2var(object.get_constant().name)


@impl(getattr, overload=(EnumTy, "value"))
def getattr_enum_value_impl(object: Var, name: Var):
    return sym2var(object.get_constant().value)

# ===========================================================================================


@impl(range)
def range_(args: tuple[Var, ...]) -> Var:
    if not 1 <= len(args) <= 3:
        raise TileTypeError(f"Invalid number of arguments: {len(args)}")
    for arg in args:
        require_signed_integer_scalar_type(arg)

    get_tensor_ty = args[0].ctx.typing_hooks.get_tensor_like_type

    if len(args) == 1:
        start = strictly_typed_const(0, get_tensor_ty(datatype.default_int_type, ()))
        stop = args[0]
        step = strictly_typed_const(1, get_tensor_ty(datatype.default_int_type, ()))
    elif len(args) == 2:
        start, stop = args[0], args[1]
        step = strictly_typed_const(1, get_tensor_ty(datatype.default_int_type, ()))
    else:
        start, stop, step = args[0], args[1], args[2]
        # FIXME(Issue 314): Support negative step.
        # Error out if step is constant and not positive.
        if step.is_constant() and step.get_constant() <= 0:
            raise TileTypeError(f"Step must be positive, got {step.get_constant()}")

    agg_value = RangeValue(start, stop, step)
    ty = RangeIterType(datatype.default_int_type)
    return make_aggregate(agg_value, ty)


@impl(hir_stubs.unpack)
def unpack_impl(iterable: Var, expected_len: Var) -> Var:
    ty = iterable.get_type()
    # Don't use the require_tuple_type() helper because we'd like to customize the error message
    if not isinstance(ty, TupleTy):
        raise TileTypeError("Expected a tuple", iterable.loc)
    expected_len = require_constant_int(expected_len)
    if len(ty.value_types) != expected_len:
        few_many = "few" if len(ty.value_types) < expected_len else "many"
        raise TileValueError(f"Too {few_many} values to unpack"
                             f" (expected {expected_len}, got {len(ty.value_types)})")
    # Return the input tuple. If we add support for additional iterables,
    # the idea is to cast them to a tuple here.
    return iterable


@comparison_operator_impl(_registry, DTypeSpec, DTypeSpec)
def comparison_dtype_spec_impl(fn: str, x: Var, y: Var):
    from cuda.tile._ir.arithmetic_ops import binop_propagate_constant
    return binop_propagate_constant(fn, x.get_type().dtype, y.get_type().dtype, None)


@comparison_operator_impl(_registry, StringTy, StringTy)
def comparison_string_impl(fn: str, x: Var, y: Var):
    from cuda.tile._ir.arithmetic_ops import binop_propagate_constant
    return binop_propagate_constant(fn, x.get_type().value, y.get_type().value, None)


@comparison_operator_impl(_registry, EnumTy, EnumTy)
def comparison_enum_impl(fn: str, x: Var, y: Var):
    from cuda.tile._ir.arithmetic_ops import binop_propagate_constant
    return binop_propagate_constant(fn, x.get_constant(), y.get_constant(), None)


# ===========================================================================================
# Print
# ===========================================================================================

@dataclass(eq=False)
class TilePrintf(Operation, opcode="tile_printf", memory_effect=MemoryEffect.STORE):
    format: str = attribute()
    args: tuple[Var, ...] = operand()
    token: Optional[Var] = operand(default=None)

    @override
    def generate_bytecode(self, ctx: BytecodeContext):
        arg_vars = [ctx.get_value(arg) for arg in self.args]
        token = None if self.token is None else ctx.get_value(self.token)
        if ctx.builder.version >= bc.BytecodeVersion.V_13_2:
            result_typeid = ctx.type_table.Token
            return bc.encode_PrintTkoOp(ctx.builder, result_typeid, arg_vars, token, self.format)
        else:
            with tile_mutex("print_mutex", ctx):
                result_typeid = None
                bc.encode_PrintTkoOp(ctx.builder, result_typeid, arg_vars, None, self.format)

                # Bytecode < 13.2 does not produce or consume print ordering tokens.
                # Return a dummy only to satisfy the IR result_var mapping.
                return bc.encode_MakeTokenOp(ctx.builder, ctx.type_table.Token)


@impl(print)
async def print_impl(args: tuple[Var, ...], sep: Var, end: Var) -> None:
    from cuda.tile._passes.hir2ir import call_function

    sep = ensure_string_or_formatted_string(sep)
    end = ensure_string_or_formatted_string(end)

    builder = FormattedStringBuilder()
    for i, arg_var in enumerate(args):
        if i > 0:
            builder.append_string_var(sep)
        arg_str = await call_function(str, arg_var)
        builder.append_string_var(arg_str)
    builder.append_string_var(end)

    format_parts = []
    for piece in builder.pieces:
        if isinstance(piece, str):
            format_parts.append(PrintfValidator.escape_str(piece))
        else:
            assert isinstance(piece, FormattedPiece)
            val = builder.values[piece.value_idx]
            ty = val.get_type()
            assert isinstance(ty, TensorLikeTy)
            if piece.format_spec is not None:
                format_parts.append(PrintfValidator.apply_python_spec(
                    piece.format_spec, ty.tensor_dtype()))
            else:
                format_parts.append(PrintfValidator.infer_format(ty.tensor_dtype()))

    add_operation_variadic(TilePrintf, (TokenTy(),), format="".join(format_parts),
                           args=tuple(builder.values))


class FormattedStringBuilder:
    def __init__(self):
        self.pieces = []
        self.values = []

    def append_formatted_piece(self, value: Var, format_spec: str | None):
        self.pieces.append(FormattedPiece(len(self.values), format_spec))
        self.values.append(value)

    def append_literal_piece(self, s: str):
        assert isinstance(s, str)
        if s == "":
            return

        if len(self.pieces) > 0 and isinstance(self.pieces[-1], str):
            self.pieces[-1] += s
        else:
            self.pieces.append(s)

    def append_string_var(self, var: Var[StringTy | FormattedStringTy]):
        ty = var.get_type()
        if isinstance(ty, StringTy):
            self.append_literal_piece(ty.value)
        else:
            assert isinstance(ty, FormattedStringTy), str(ty)
            value = var.get_aggregate()
            assert isinstance(value, FormattedStringValue)
            for piece in value.format.pieces:
                if isinstance(piece, FormattedPiece):
                    self.append_formatted_piece(value.values[piece.value_idx], piece.format_spec)
                else:
                    self.append_literal_piece(piece)

    def build(self) -> Var:
        if len(self.pieces) == 0:
            return loosely_typed_const("")
        elif len(self.pieces) == 1 and isinstance(self.pieces[0], str):
            return loosely_typed_const(self.pieces[0])
        else:
            format = StringFormat(tuple(self.pieces))
            ty = FormattedStringTy(format, tuple(v.get_type() for v in self.values))
            return make_aggregate(FormattedStringValue(format, tuple(self.values)), ty)


def string_concat(*chunks: Var[StringTy | FormattedStringTy]) -> Var[FormattedStringTy]:
    builder = FormattedStringBuilder()
    for chunk in chunks:
        builder.append_string_var(chunk)
    return builder.build()


@impl(operator.add, overload=(StringTy, StringTy))
@impl(operator.add, overload=(StringTy, FormattedStringTy))
@impl(operator.add, overload=(FormattedStringTy, FormattedStringTy))
@impl(operator.add, overload=(FormattedStringTy, StringTy))
def string_concat_impl(x: Var[StringTy | FormattedStringTy], y: Var[StringTy | FormattedStringTy]):
    return string_concat(x, y)


@impl(hir_stubs.build_formatted_string)
async def build_formatted_string(format: StringFormat, values: tuple[Var, ...]) -> Var:
    builder = FormattedStringBuilder()

    for piece in format.pieces:
        if isinstance(piece, str):
            builder.append_literal_piece(piece)
            continue

        value = values[piece.value_idx]
        val_ty = value.get_type()
        if isinstance(val_ty, TensorLikeTy):
            builder.append_formatted_piece(value, piece.format_spec)
            continue

        if isinstance(val_ty, DataclassTy):
            custom_format_dunder = find_method(val_ty.cls, "__format__")
            if (custom_format_dunder is not NotImplemented
                    and custom_format_dunder is not object.__format__):
                raise TypeCheckingError("Formatting dataclass values with custom __format__()"
                                        " is not supported")

        if piece.format_spec is not None:
            raise TileTypeError(
                f"f-string: cannot apply format spec to a value of type {val_ty}",
                value.loc)

        from cuda.tile._passes.hir2ir import call_function
        value = await call_function(str, value)
        builder.append_string_var(value)

    return builder.build()

# ===========================================================================================
