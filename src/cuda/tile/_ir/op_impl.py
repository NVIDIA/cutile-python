# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import functools
import inspect
import threading
import re
from enum import EnumMeta
from typing import Optional, NamedTuple, Tuple, Sequence, Any, Union

from cuda.tile._datatype import (
        is_integral, is_float, is_restricted_float,
        is_boolean, is_signed, DType)
from cuda.tile._exception import TileTypeError
from cuda.tile._ir.ops_utils import get_dtype

from .typing_support import datatype, get_signature
from .ir import Var, TupleValue
from .type import TupleTy, TileTy, DTypeSpec, EnumTy, StringTy, ArrayTy, SliceType, \
    ListTy, PointerTy, LooselyTypedScalar, RangeIterType, FunctionTy, ClosureTy, BoundMethodTy, \
    DTypeConstructor
from .. import _datatype


def _verify_params_match(stub_sig: inspect.Signature, func_sig: inspect.Signature):
    assert len(stub_sig.parameters) == len(func_sig.parameters), (
        f"Stub and implementation must have same number of parameters."
        f" Signatures: {stub_sig}, {func_sig}.")
    for i, (stub_param, func_param) in enumerate(zip(stub_sig.parameters.values(),
                                                     func_sig.parameters.values(), strict=True)):
        assert stub_param.name == func_param.name, (
            f"Stub and implementation have different parameter names at position {i}"
            f" Signatures: {stub_sig}, {func_sig}.")


op_implementations = dict()


def impl(stub, *, fixed_args: Sequence[Any] = ()):
    stub_sig = get_signature(stub)

    def decorate(func):
        orig_func = func
        if len(fixed_args) > 0:
            func = functools.partial(orig_func, *fixed_args)

        func_sig = get_signature(func)
        _verify_params_match(stub_sig, func_sig)
        is_coroutine = inspect.iscoroutinefunction(func)
        if is_coroutine:
            @functools.wraps(func)
            async def wrapper(*args, **kwargs):
                # Memorize the stub and the args so that we can automatically
                # provide context for error messages.
                old = _current_stub.stub_and_args
                _current_stub.stub_and_args = (stub, stub_sig, func_sig, args, kwargs)
                try:
                    return await func(*args, **kwargs)
                finally:
                    _current_stub.stub_and_args = old
        else:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                # Memorize the stub and the args so that we can automatically
                # provide context for error messages.
                old = _current_stub.stub_and_args
                _current_stub.stub_and_args = (stub, stub_sig, func_sig, args, kwargs)
                try:
                    return func(*args, **kwargs)
                finally:
                    _current_stub.stub_and_args = old

        wrapper._is_coroutine = is_coroutine
        op_implementations[stub] = wrapper
        return orig_func

    return decorate


class _CurrentStub(threading.local):
    stub_and_args = None


_current_stub = _CurrentStub()


def require_constant_int(var: Var) -> int:
    if not var.is_constant():
        raise _make_type_error("Expected an integer constant, but given value is not constant", var)
    ty = var.get_type()
    if not is_integral(ty):
        raise _make_type_error(f"Expected an integer constant, but given value has type {ty}",
                               var)
    return var.get_constant()


def require_optional_constant_int(var: Var) -> Optional[int]:
    if var.is_constant() and var.get_constant() is None:
        return None
    return require_constant_int(var)


def require_constant_bool(var: Var) -> bool:
    if not var.is_constant():
        raise _make_type_error("Expected a boolean constant, but given value is not constant", var)
    ty = var.get_type()
    if not is_boolean(ty):
        raise _make_type_error(f"Expected a boolean constant, but given value has type {ty}", var)
    return var.get_constant()


def require_constant_scalar(var: Var) -> bool | int | float:
    ty = var.get_type()
    if not isinstance(ty, DType):
        raise _make_type_error(f"Expected a scalar constant, but given value has type {ty}", var)
    if not var.is_constant():
        raise _make_type_error(f"Expected a constant, but given value has non-constant type {ty}",
                               var)
    ret = var.get_constant()
    assert isinstance(ret, bool | int | float)
    return ret


def require_constant_scalar_tuple(var: Var) -> tuple[bool | int | float, ...]:
    ty = require_tuple_type(var)
    ret = []
    tuple_val = var.get_aggregate()
    assert isinstance(tuple_val, TupleValue)
    for i, (item_ty, item) in enumerate(zip(ty.value_types, tuple_val.items, strict=True)):
        if not isinstance(item_ty, DType):
            raise _make_type_error(f"Expected a tuple of scalar constants,"
                                   f" but item at position #{i} has type {item_ty}", var)
        if not item.is_constant():
            raise _make_type_error(f"Expected a tuple of scalar constants,"
                                   f" but item at position #{i} has non-constant type {ty}", var)
        value = item.get_constant()
        assert isinstance(value, bool | int | float)
        ret.append(value)
    return tuple(ret)


def require_optional_constant_bool(var: Var) -> Optional[bool]:
    if var.is_constant() and var.get_constant() is None:
        return None
    return require_constant_bool(var)


def require_constant_str(var: Var) -> str:
    if not var.is_constant():
        raise _make_type_error("Expected a string constant, but given value is not constant", var)
    ty = var.get_type()
    if not isinstance(ty, StringTy):
        raise _make_type_error(f"Expected a string constant, but given value has type {ty}", var)
    return ty.value


def require_optional_constant_str(var: Var) -> Optional[str]:
    if var.is_constant() and var.get_constant() is None:
        return None
    return require_constant_str(var)


def require_constant_slice(var: Var) -> slice:
    if not var.is_constant():
        raise _make_type_error("Expected a slice constant, but given value is not constant", var)
    ty = var.get_type()
    if not isinstance(ty, SliceType):
        raise _make_type_error(f"Expected a slice constant, but given value has type {ty}", var)
    return var.get_constant()


def require_dtype_spec(var: Var) -> DType:
    ty = var.get_type()
    if not isinstance(ty, DTypeSpec):
        raise _make_type_error(f"Expected a dtype constant, but given value has type {ty}", var)
    return ty.dtype


def require_optional_constant_enum(var: Var, enum: EnumMeta):
    if var.is_constant() and var.get_constant() is None:
        return None
    return require_constant_enum(var, enum)


def require_optional_range_type(var: Var) -> RangeIterType | None:
    if var.is_constant() and var.get_constant() is None:
        return None
    ty = var.get_type()
    if not isinstance(ty, RangeIterType):
        raise _make_type_error(f"Expected a range object, but given value has type {ty}", var)
    return ty


def require_constant_enum(var: Var, enum: EnumMeta):
    if not var.is_constant():
        raise _make_type_error(f"Expected {enum.__name__} constant,"
                               f" but given value is not constant", var)
    ty = var.get_type()
    if not isinstance(ty, EnumTy) or ty.enum_ty is not enum:
        raise _make_type_error(f"Expected {enum.__name__}, but given value has type {ty}", var)
    return var.get_constant()


def normalize_axis(axis: int, ndim: int, var: Optional[Var] = None) -> int:
    orig_axis = axis
    if axis < 0:
        axis += ndim
    if axis < 0 or axis >= ndim:
        raise _make_type_error(f"Axis {orig_axis} is out of range for rank {ndim}'", var)
    return axis


def require_constant_int_tuple(var: Var, allow_single_int: bool = False) -> Tuple[int, ...]:
    if not var.is_constant():
        raise _make_type_error("Expected a constant integer tuple,"
                               " but given value is not constant", var)

    ty = var.get_type()
    if allow_single_int and isinstance(ty, DType):
        return require_constant_int(var),

    if not isinstance(ty, TupleTy):
        raise _make_type_error(f"Expected a tuple, but given value has type {ty}", var)

    for i, item_ty in enumerate(ty.value_types):
        if not is_integral(item_ty):
            raise _make_type_error(f"Expected a tuple of integers,"
                                   f" but element #{i} has type {item_ty}", var)

    val = var.get_constant()
    assert isinstance(val, tuple)
    assert all(isinstance(x, int) for x in val)
    return val


def require_constant_shape(var: Var,
                           allow_single_int: bool = False,
                           expected_rank: Optional[int] = None,
                           allow_0d_shape: bool = False) -> Tuple[int, ...]:
    shape = require_constant_int_tuple(var, allow_single_int=allow_single_int)

    if (expected_rank is not None and len(shape) != expected_rank
            and not (allow_0d_shape and len(shape) == 0)):
        raise _make_type_error(f"Expected shape length to be {expected_rank}, got {len(shape)}",
                               var)

    for i, x in enumerate(shape):
        if x <= 0:
            raise _make_type_error(f"Dimension #{i} of shape {tuple(shape)} is not positive", var)
        if x & (x - 1) != 0:
            raise _make_type_error(f"Dimension #{i} of shape {tuple(shape)} is not a power of two",
                                   var)

    return shape


def require_constant_axis_order(var: Var, rank: int) -> Tuple[int, ...]:
    """
    Helper for matching the 'order' argument of functions like cuda.tile.load() etc.
    The order can either be a string literal "C" or "F" (which represents a NumPy-style
    contiguous axis order), or a literal sequence of integers, e.g. (1, -2, 0).

    Returns a tuple that contains the "normalized" (i.e. non-negative) axis indices.
    """
    if not var.is_constant():
        raise _make_type_error("Expected a constant string or integer tuple", var)

    value = var.get_constant()
    if value == "C":
        return tuple(range(rank))
    elif value == "F":
        return tuple(range(rank - 1, -1, -1))
    elif isinstance(value, str):
        raise _make_type_error(f"Expected 'C' or 'F', got '{value}'", var)

    value = require_constant_int_tuple(var)
    if len(value) != rank:
        raise _make_type_error(f"Expected tuple of length {rank}, got {len(value)}", var)

    return tuple(normalize_axis(x, rank, var) for x in value)


def require_tile_type(var: Var) -> TileTy:
    ty = var.get_type()
    if not isinstance(ty, TileTy):
        raise _make_type_error(f"Expected a tile, but given value has type {ty}", var)
    return ty


def require_tile_or_tile_tuple_type(var: Var) -> TileTy | TupleTy:
    ty = var.get_type()
    if isinstance(ty, TileTy):
        return ty
    if isinstance(ty, TupleTy) and all(isinstance(x, TileTy) for x in ty.value_types):
        return ty
    raise _make_type_error(f"Expected a tile or a tuple of tiles, but given value has type {ty}",
                           var)


def require_tile_or_scalar_type(var: Var) -> TileTy | DType | PointerTy:
    ty = var.get_type()
    if not isinstance(ty, TileTy | DType | PointerTy):
        raise _make_type_error(f"Expected a tile or a scalar, but given value has type {ty}", var)
    return ty


def require_tile_or_scalar_maybe_loose_type(var: Var) \
        -> TileTy | DType | PointerTy | LooselyTypedScalar:
    ty = var.get_loose_type()
    if isinstance(ty, LooselyTypedScalar):
        return ty
    return require_tile_or_scalar_type(var)


def require_scalar_or_0d_tile_type(var: Var) -> TileTy | DType | PointerTy:
    ty = var.get_type()
    if not (isinstance(ty, DType | PointerTy) or (isinstance(ty, TileTy) and ty.ndim == 0)):
        raise _make_type_error(f"Expected a scalar or a 0D tile, but given value has type {ty}",
                               var)
    return ty


def require_signed_integer_scalar_or_0d_tile_type(var: Var) -> TileTy | DType:
    ty = require_scalar_or_0d_tile_type(var)
    if isinstance(ty, TileTy):
        dtype = ty.dtype
    elif isinstance(ty, DType):
        dtype = ty
    else:
        dtype = None

    if dtype is None or not datatype.is_integral(dtype) or not datatype.is_signed(dtype):
        raise _make_type_error(f"Expected a signed integer scalar or a 0D signed integer tile,"
                               f" but got {ty}", var)
    return ty


def require_bool(var: Var) -> TileTy | DType:
    ty = var.get_type()
    if not (ty == _datatype.bool_
            or (isinstance(ty, TileTy) and ty.ndim == 0 and ty.dtype == _datatype.bool_)):
        raise _make_type_error(f"Expected a bool, but given value has type {ty}", var)
    return ty


def require_scalar_or_0d_tile_maybe_loose_type(var: Var) \
        -> TileTy | DType | PointerTy | LooselyTypedScalar:
    ty = var.get_loose_type()
    if isinstance(ty, LooselyTypedScalar):
        return ty
    return require_scalar_or_0d_tile_type(var)


def require_array_type(var: Var) -> ArrayTy:
    ty = var.get_type()
    if not isinstance(ty, ArrayTy):
        raise _make_type_error(f"Expected an array, but given value has type {ty}", var)
    return ty


def require_list_type(var: Var) -> ListTy:
    ty = var.get_type()
    if not isinstance(ty, ListTy):
        raise _make_type_error(f"Expected a list, but given value has type {ty}", var)
    return ty


def require_integer_dtype(var: Var) -> DType:
    ty = var.get_type()
    if not isinstance(ty, DType) or not is_integral(ty):
        raise TileTypeError(f"Expected an integer scalar, but given value has type {ty}")
    return ty


def require_tuple_type(var: Var) -> TupleTy:
    ty = var.get_type()
    if not isinstance(ty, TupleTy):
        raise _make_type_error(f"Expected a tuple, but given value has type {ty}", var)
    return ty


def require_index_or_index_tuple_type(var: Var,
                                      allow_nd_tiles: bool = False,
                                      allow_unsigned: bool = False) \
        -> TupleTy | TileTy | DType:
    ty = var.get_type()
    if isinstance(ty, TupleTy):
        item_types = ty.value_types
    else:
        item_types = ty,

    for i, item_ty in enumerate(item_types):
        if isinstance(item_ty, TileTy) and (allow_nd_tiles or item_ty.ndim == 0):
            dtype = item_ty.dtype
        elif isinstance(item_ty, DType):
            dtype = item_ty
        else:
            dtype = None

        if dtype is None or not is_integral(dtype) or not (allow_unsigned or is_signed(dtype)):
            what = f"item #{i}" if isinstance(ty, TupleTy) else "given value"
            signed = "" if allow_unsigned else "signed "
            if allow_nd_tiles:
                raise _make_type_error(f"Expected a tuple of {signed}integer scalars/tiles"
                                       f" or a single {signed}integer scalar/tile,"
                                       f" but {what} has type {item_ty}", var)
            else:
                raise _make_type_error(f"Expected a tuple of {signed}integers or a single"
                                       f" {signed}integer scalar, but {what} has type {item_ty}",
                                       var)

    return ty


def require_callable_type(var: Var) -> FunctionTy | BoundMethodTy | ClosureTy | DTypeConstructor:
    ty = var.get_type()
    if not isinstance(ty, FunctionTy | BoundMethodTy | ClosureTy | DTypeConstructor):
        raise _make_type_error(f"Expected a callable object, but given value has type {ty}", var)
    return ty


class PrintfValidator:
    # c-format string has the following: %[flags][width][.precision][length]specifier
    # we only support a subset which makes sense in the tile context
    float_specifiers = {'e', 'E', 'f', 'F', 'g', 'G', 'a', 'A'}
    int_specifiers = {'d', 'i', 'u', 'o', 'x', 'X'}
    flags = r"([0 #+-])?"
    width = r"([0-9]+)?"
    precision = r"(\.[0-9]+)?"
    length = r"(hh|h|ll|l)?"
    specifiers = r"([diuoxXeEfFgGaAcspn])"
    pattern = re.compile("%" + flags + width + precision + length + specifiers)

    @classmethod
    def validate_dtype(cls, dtype: DType, specifier: str) -> bool:
        if is_boolean(dtype) or is_integral(dtype):
            return specifier in cls.int_specifiers
        elif is_float(dtype) or is_restricted_float(dtype):
            return specifier in cls.float_specifiers
        else:
            return False

    @classmethod
    def parse_format(cls, format: str, arg_types: Tuple[Union[TileTy, DType], ...]) -> str:
        last_pos = pos = 0
        arg_idx = 0
        tokens = []
        while pos < len(format):
            if format[pos] == "%":
                tokens.append(format[last_pos:pos])
                last_pos = pos
                # escape "%%"
                if (pos + 1 < len(format) and format[pos + 1] == "%"):
                    pos += 2
                    continue
                elif (m := cls.pattern.match(format, pos)):
                    # get a format match
                    _, _, _, _, sp = m.groups()
                    fmt = m.group(0)
                    if not (sp in cls.int_specifiers or sp in cls.float_specifiers):
                        raise TileTypeError(f"Specifier {sp} in {fmt} is not supported")
                    # pop argument
                    if arg_idx >= len(arg_types):
                        raise TileTypeError("Not enough arguments for format string")
                    ty = arg_types[arg_idx]
                    # validate arg type against fmt
                    if not cls.validate_dtype(get_dtype(ty), sp):
                        raise TileTypeError(f"Format {fmt} for arg #{arg_idx} got unexpected type of {ty}")  # noqa: E501
                    arg_idx += 1
                    pos = m.end()
                    tokens.append(format[last_pos:pos])
                    last_pos = pos
                    continue
                else:
                    raise TileTypeError("Invalid format string")
            pos += 1
        tokens.append(format[last_pos:pos])
        if arg_idx < len(arg_types):
            raise TileTypeError("Too many arguments for format string")
        return "".join(tokens)


class _ErrorContext(NamedTuple):
    function_name: str
    param_name_or_idx: str | int


def _recover_error_context(var: Optional[Var]) -> Optional[_ErrorContext]:
    if var is None:
        return None
    cur_stub_and_args = _current_stub.stub_and_args
    if cur_stub_and_args is None:
        return None
    stub, stub_sig, func_sig, args, kwargs = cur_stub_and_args
    bound_args: inspect.BoundArguments = func_sig.bind(*args, **kwargs)
    for param_name, arg in bound_args.arguments.items():
        if arg is var:
            stub_param = stub_sig.parameters[param_name]
            if stub_param.kind == inspect.Parameter.POSITIONAL_ONLY:
                param_name_or_idx = next(i for i, pname in enumerate(stub_sig.parameters.keys(), 1)
                                         if pname == param_name)
            else:
                param_name_or_idx = param_name
            return _ErrorContext(stub.__name__, param_name_or_idx)
    return None


def _make_type_error(what: str, var: Optional[Var]) -> TileTypeError:
    context = _recover_error_context(var)
    if context is None:
        context_str = ""
    else:
        if isinstance(context.param_name_or_idx, int):
            arg_name = f"#{context.param_name_or_idx}"
        else:
            arg_name = f'"{context.param_name_or_idx}"'
        context_str = f"Invalid argument {arg_name} of {context.function_name}(): "
    return TileTypeError(context_str + what)
