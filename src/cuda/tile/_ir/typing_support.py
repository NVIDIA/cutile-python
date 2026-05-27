# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
import inspect
import operator
import dataclasses
from enum import Enum, IntEnum
from functools import lru_cache
from types import ModuleType, FunctionType
from typing import Any, Callable, Mapping, Union

from cuda.tile import _datatype as datatype, DType
from cuda.tile._exception import TileTypeError, TileValueError
from .type import DataclassInfo, PointerInfoTy

from .type import Type, TupleTy, DTypeConstructor, DTypeSpec, NONE, StringTy, \
    ELLIPSIS, SLICE, ModuleTy, FunctionTy, EnumTy, TypeTy, LooselyTypedScalar, \
    TileTy


BOOL_TY = TileTy(datatype.bool_)
I32_TY = TileTy(datatype.int32)
I64_TY = TileTy(datatype.int64)
U64_TY = TileTy(datatype.uint64)


# Store mapping from 3rd party dtype objects
# e.g. np.float32 -> float32, torch.bfloat16 -> bfloat16
_dtype_registry: dict[Any, DTypeSpec] = {}


def register_dtypes(dtypes: Mapping[Any, datatype.DType], usable_as_constructor=False):
    cls = DTypeConstructor if usable_as_constructor else DTypeSpec
    for t1, t2 in dtypes.items():
        _dtype_registry[t1] = cls(t2)


def to_dtype(x: Any):
    if isinstance(x, DType):
        return x
    return _dtype_registry[x].dtype


def _safe_get(dict, key, default=None):
    try:
        return dict.get(key, default)
    except TypeError:  # if not hashable
        return default


def is_dtype(x: Any):
    return isinstance(x, DType) or _safe_get(_dtype_registry, x) is not None


def _is_dtype_allowed_as_constructor(dtype: DType) -> bool:
    # Only allow byte aligned numeric dtypes as constructors
    return datatype.is_numeric(dtype) and (dtype.bitwidth % 8 == 0)


def is_dtype_constructor(x: Any) -> bool:
    if isinstance(x, DType):
        return _is_dtype_allowed_as_constructor(x)
    else:
        return isinstance(_safe_get(_dtype_registry, x), DTypeConstructor)


# Store mapping from a type to a handler that convert value of that type to IR Type
# e.g. torch.Tensor -> Array
# The key can also be a str object interface such as "__cuda_array_interface__" or "__dlpack__"
TypeHandler = Callable[[Type], Any]
TypeKey = Union[type, str]


class TypeHandlerTable(dict[TypeKey, TypeHandler]):
    _types_with_subtypes = []

    def __missing__(self, key: TypeKey) -> TypeHandler:
        if isinstance(key, type):
            for parent_ty in self._types_with_subtypes:
                if issubclass(key, parent_ty):
                    self[key] = self[parent_ty]
                    return self[key]
        raise KeyError


BUILTIN_FUNCS = {
    abs: lambda x: None,
    len: lambda x, /: None,
    max: lambda x, y, /: None,
    min: lambda x, y, /: None,
    range: lambda *args: None,
    slice: lambda start, stop, step: None,
    operator.add: lambda x, y, /: None,
    operator.sub: lambda x, y, /: None,
    operator.mul: lambda x, y, /: None,
    operator.floordiv: lambda x, y, /: None,
    operator.truediv: lambda x, y, /: None,
    operator.mod: lambda x, y, /: None,
    operator.pow: lambda x, y, /: None,
    operator.or_: lambda x, y, /: None,
    operator.xor: lambda x, y, /: None,
    operator.and_: lambda x, y, /: None,
    operator.lshift: lambda x, y, /: None,
    operator.rshift: lambda x, y, /: None,
    operator.matmul: lambda x, y, /: None,
    operator.eq: lambda x, y, /: None,
    operator.ne: lambda x, y, /: None,
    operator.lt: lambda x, y, /: None,
    operator.le: lambda x, y, /: None,
    operator.gt: lambda x, y, /: None,
    operator.ge: lambda x, y, /: None,
    operator.is_: lambda x, y, /: None,
    operator.is_not: lambda x, y, /: None,
    operator.invert: lambda x, /: None,
    operator.not_: lambda x, /: None,
    operator.pos: lambda x, /: None,
    operator.neg: lambda x, /: None,
    getattr: lambda object, name, /: None,
    operator.getitem: lambda object, key, /: None,
    operator.setitem: lambda object, key, value, /: None,
    float: lambda x=0, /: None,
    int: lambda x=0, /: None,
    bool: lambda x=False, /: None,
    print: lambda *args, sep=' ', end='\n': None,
    dataclasses.replace: dataclasses.replace,
}


def get_signature(f) -> inspect.Signature:
    if stub := BUILTIN_FUNCS.get(f):
        f = stub
    elif is_dtype_constructor(f):
        # Data type constructors
        f = lambda x=0, /: None  # noqa: E731
    return inspect.signature(f)


def is_supported_builtin_func(x: Any) -> bool:
    return _safe_get(BUILTIN_FUNCS, x) is not None or getattr(x, '_cutile_is_builtin', False)


def dtype_of_constant_scalar(val: bool | int | float) -> DType:
    if isinstance(val, bool):
        return datatype.bool_
    elif isinstance(val, int):
        if -2**31 <= val < 2**31:
            return datatype.int32
        elif -2**63 <= val < 2**63:
            return datatype.int64
        elif 0 <= val < 2**64:
            return datatype.uint64
        else:
            # FIXME: delay the error and allow arbitrary-precision intermediate constant values
            raise TileValueError(f"Constant {val} is out of range of any supported integer type")
    elif isinstance(val, float):
        return datatype.default_float_type
    else:
        raise TypeError(f'Python value {val} of type {type(val)} is not supported.')


def typeof_pyval(val) -> Type:
    if val is None:
        return NONE
    if (t := _safe_get(_dtype_registry, type(val))):
        return TileTy(t.dtype)
    if isinstance(val, bool | int | float):
        return TileTy(dtype_of_constant_scalar(val))
    if isinstance(val, Enum):
        return EnumTy(type(val))
    if isinstance(val, str):
        return StringTy(val)
    if isinstance(val, tuple):
        return TupleTy(tuple(typeof_pyval(v) for v in val))
    if val is Ellipsis:
        return ELLIPSIS
    if isinstance(val, slice):
        return SLICE
    if isinstance(val, ModuleType):
        return ModuleTy(val)
    if isinstance(val, FunctionType):
        return FunctionTy(val)
    if is_supported_builtin_func(val):
        return FunctionTy(val)
    if isinstance(val, datatype.DType):
        if _is_dtype_allowed_as_constructor(val):
            return DTypeConstructor(val)
        else:
            return DTypeSpec(val)
    if (t := _safe_get(_dtype_registry, val)) is not None:
        return t
    if isinstance(val, datatype.PointerInfo):
        return PointerInfoTy(val)

    if isinstance(val, type):
        return TypeTy(val)

    # TODO: should we add dlpack?
    raise TypeError(f'Python value {val} of type {type(val)} is not supported.')


def loose_type_of_pyval(value: Any) -> Type:
    if isinstance(value, bool | int | float):
        return LooselyTypedScalar(value)
    elif isinstance(value, tuple):
        return TupleTy(tuple(loose_type_of_pyval(x) for x in value))
    else:
        return typeof_pyval(value)


_SUPPORTED_CONST_TYPES = (int, float, bool, str, ModuleType, FunctionType, type, Enum, IntEnum)


def get_constant_value(val: Any) -> Any:
    if val is None or isinstance(val, _SUPPORTED_CONST_TYPES) or is_supported_builtin_func(val):
        return val
    if is_dtype(val):
        return to_dtype(val)
    if isinstance(val, tuple):
        return tuple(get_constant_value(x) for x in val)
    typ = type(val)
    prefix = "" if typ.__module__ == "builtins" else f"{typ.__module__}."
    raise TileTypeError(f"Cannot create constant from value of type {prefix}{typ.__qualname__}.")


@lru_cache
def get_dataclass_info(cls) -> DataclassInfo:
    params = cls.__dataclass_params__
    if not params.frozen:
        raise TileTypeError("Only frozen dataclasses are supported")

    if not params.init:
        raise TileTypeError("Dataclasses with init=False are not supported")

    # HACK: There seems to be no clean way to detect whether a dataclass has a user-defined
    #       __init__() method. This is the best I could come up with.
    #       Explanation: for a frozen dataclass (which we check above), the generated __init__()
    #       method needs to call `object.__setattr__()` to set the initial values of frozen fields.
    #       Since the builtin `object` name may be shadowed, the dataclass implementation stores
    #       the `object` class in a captured variable named "__dataclass_builtins_object__".
    if "__dataclass_builtins_object__" not in cls.__init__.__code__.co_freevars:
        raise TileTypeError("Dataclasses with custom __init__ are not supported")

    if hasattr(cls, "__post_init__"):
        raise TileTypeError("Dataclasses with __post_init__ are not supported")

    if "__new__" in cls.__dict__:
        raise TileTypeError("Dataclasses with custom __new__ are not supported")

    if len(cls.__bases__) != 1 or cls.__bases__[0] is not object:
        # TODO: This is something we could partially relax,
        #       e.g. dataclass inheriting from another dataclass.
        raise TileTypeError("Only dataclasses without a base class are supported")

    field_name_to_idx = {}
    field_names = []
    for i, f in enumerate(dataclasses.fields(cls)):
        if f.default_factory is not dataclasses.MISSING:
            # TODO: This is something we could relax
            raise TileTypeError("Dataclasses with default_factory fields are not supported")

        if not f.init:
            # It probably doesn't make sense to relax this constraint for a frozen dataclass.
            raise TileTypeError("Dataclasses with init=False fields are not supported")

        field_names.append(f.name)
        field_name_to_idx[f.name] = i

    init_signature = inspect.signature(cls.__init__)
    return DataclassInfo(cls, field_names, field_name_to_idx, init_signature)


# ========= Numpy support ===========

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    np = None

if HAS_NUMPY:

    # register numpy dtype types
    register_dtypes({
        np.float64: datatype.float64,
        np.float32: datatype.float32,
        np.float16: datatype.float16,
        np.int64: datatype.int64,
        np.int32: datatype.int32,
        np.int16: datatype.int16,
        np.int8: datatype.int8,
        np.uint64: datatype.uint64,
        np.uint32: datatype.uint32,
        np.uint16: datatype.uint16,
        np.uint8: datatype.uint8,
        np.bool_: datatype.bool_
    }, usable_as_constructor=True)
    # register numpy dtype objects
    register_dtypes({
        np.dtype('float64'): datatype.float64,
        np.dtype('float32'): datatype.float32,
        np.dtype('float16'): datatype.float16,
        np.dtype('int64'): datatype.int64,
        np.dtype('int32'): datatype.int32,
        np.dtype('int16'): datatype.int16,
        np.dtype('int8'): datatype.int8,
        np.dtype('uint64'): datatype.uint64,
        np.dtype('uint32'): datatype.uint32,
        np.dtype('uint16'): datatype.uint16,
        np.dtype('uint8'): datatype.uint8,
        np.dtype('bool'): datatype.bool_
    })

# ========= JAX MLDtype support ===========
try:
    import ml_dtypes
    HAS_ML_DTYPES = True
except ImportError:
    HAS_ML_DTYPES = False
    ml_dtypes = None


if HAS_NUMPY and HAS_ML_DTYPES:
    register_dtypes({
        np.dtype(ml_dtypes.bfloat16): datatype.bfloat16,
        np.dtype(ml_dtypes.float8_e4m3fn): datatype.float8_e4m3fn,
        np.dtype(ml_dtypes.float8_e5m2): datatype.float8_e5m2,
        np.dtype(ml_dtypes.float8_e8m0fnu): datatype.float8_e8m0fnu,
    })


# ===== PyTorch ===========

try:
    import torch as torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None


if HAS_TORCH:
    # register torch dtypes
    register_dtypes({
        torch.float64: datatype.float64,
        torch.float32: datatype.float32,
        torch.float16: datatype.float16,
        torch.int64: datatype.int64,
        torch.int32: datatype.int32,
        torch.int16: datatype.int16,
        torch.int8: datatype.int8,
        torch.uint64: datatype.uint64,
        torch.uint32: datatype.uint32,
        torch.uint16: datatype.uint16,
        torch.uint8: datatype.uint8,
        torch.bool: datatype.bool_,
        torch.bfloat16: datatype.bfloat16,
        torch.float8_e4m3fn: datatype.float8_e4m3fn,
        torch.float8_e5m2: datatype.float8_e5m2,
        torch.float8_e8m0fnu: datatype.float8_e8m0fnu,
    })


# ===== Cuda Array Interface ===========
BYTE_BITWIDTH = 8


def _compute_elem_strides(shape, dtype_bytewidth, byte_strides):
    if byte_strides is not None:
        return tuple(bs // dtype_bytewidth for bs in byte_strides)

    if len(shape) == 0:
        return tuple()

    reverse_elem_strides = [1]
    for i in shape[-1:0:-1]:
        reverse_elem_strides.append(reverse_elem_strides[-1] * i)

    return tuple(reverse_elem_strides[::-1])
