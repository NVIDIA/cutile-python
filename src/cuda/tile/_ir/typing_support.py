# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
import inspect
import operator
from enum import Enum
from types import ModuleType, FunctionType
from typing import Any, Callable, Mapping, Union
from cuda.tile import _datatype as datatype
from cuda.tile._exception import TileTypeError
from cuda.tile._cext import ArraySpecialization

from .type import Type, SizeTy, TupleTy, DTypeConstructor, DTypeSpec, ListTy, NONE, StringTy, \
    ELLIPSIS, SLICE, ModuleTy, FunctionTy, ArrayTy, EnumTy, TypeTy, LooselyTypedScalar

# Store mapping from 3rd party dtype objects
# e.g. np.float32 -> float32, torch.bfloat16 -> bfloat16
dtype_registry = {}


def register_dtypes(dtypes: Mapping[Any, datatype.DType], usable_as_constructor=False):
    cls = DTypeConstructor if usable_as_constructor else DTypeSpec
    for t1, t2 in dtypes.items():
        dtype_registry[t1] = cls(t2)


def to_dtype(x: Any):
    return dtype_registry[x].dtype


def _safe_get(dict, key, default=None):
    try:
        return dict.get(key, default)
    except TypeError:  # if not hashable
        return default


def is_dtype(x: Any):
    return _safe_get(dtype_registry, x) is not None


def is_dtype_constructor(x: Any):
    return isinstance(_safe_get(dtype_registry, x), DTypeConstructor)


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


_type_handler = TypeHandlerTable()


def register_type_handler(t: Union[type, str], allow_subtypes=False):

    def wrapped(handler: TypeHandler):
        _type_handler[t] = handler
        if allow_subtypes and isinstance(t, type):
            _type_handler._types_with_subtypes.append(t)
        return handler

    return wrapped


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
    operator.invert: lambda x, /: None,
    operator.not_: lambda x, /: None,
    operator.pos: lambda x, /: None,
    operator.neg: lambda x, /: None,
    getattr: lambda object, name, /: None,
    operator.getitem: lambda object, key, /: None,
}


def get_signature(f):
    if stub := BUILTIN_FUNCS.get(f):
        f = stub
    elif f in dtype_registry:
        # Data type constructors
        f = lambda x=0, /: None  # noqa: E731
    return inspect.signature(f)


def is_supported_builtin_func(x: Any) -> bool:
    return _safe_get(BUILTIN_FUNCS, x) is not None


def typeof_pyval(val) -> Type:
    if val is None:
        return NONE
    if (t := _safe_get(dtype_registry, type(val))):
        return t.dtype
    if isinstance(val, bool):
        return datatype.bool_
    if isinstance(val, int):
        return datatype.default_int_type
    if isinstance(val, float):
        return datatype.default_float_type
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
    if (t := _safe_get(dtype_registry, val)) is not None:
        return t
    if isinstance(val, list):
        if len(val) == 0:
            raise TypeError('Empty lists are not yet supported.')
        first, *tail = val
        item_ty = typeof_pyval(first)
        if isinstance(item_ty, ArrayTy):
            for x in tail:
                x_ty = typeof_pyval(x)
                if not isinstance(x_ty, ArrayTy):
                    raise TypeError("Expected all list items to be arrays")
                new_item_ty = item_ty.unify(x_ty)
                if new_item_ty is None:
                    raise TypeError(f"Array types {item_ty} and {x_ty}"
                                    f" found in list are not compatible")
                item_ty = new_item_ty
            return ListTy(item_ty)
        else:
            raise TypeError(f'Items of type {type(val)} are not supported as list elements')

    try:
        return _type_handler[type(val)](val)
    except KeyError:
        pass

    if isinstance(val, type):
        return TypeTy(val)
    if isinstance(val, Enum):
        return EnumTy(type(val))

    if hasattr(val, '__cuda_array_interface__'):
        try:
            return _type_handler['__cuda_array_interface__'](val)
        except KeyError:
            pass

    # TODO: should we add dlpack?
    raise TypeError(f'Python value {val} of type {type(val)} is not supported.')


def loose_type_of_pyval(value: Any) -> Type:
    if isinstance(value, bool | int | float):
        return LooselyTypedScalar(value)
    elif isinstance(value, tuple):
        return TupleTy(tuple(loose_type_of_pyval(x) for x in value))
    else:
        return typeof_pyval(value)


def get_constant_value(val: Any) -> Any:
    if val is None or isinstance(val, (int, float, bool, ModuleType, FunctionType, type, Enum))\
            or is_supported_builtin_func(val):
        return val
    if is_dtype(val):
        return to_dtype(val)
    if isinstance(val, tuple) and not any(isinstance(x, tuple) for x in val):
        return tuple(get_constant_value(x) for x in val)
    typ = type(val)
    prefix = "" if typ.__module__ == "builtins" else f"{typ.__module__}."
    raise TileTypeError(f"Cannot create constant from value of type {prefix}{typ.__qualname__}.")


# =====CuTile native support ===========
# register cuTile native dtype types
for dtype in datatype.dtype_to_enum:
    register_dtypes({dtype: dtype}, usable_as_constructor=True)


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
        torch.float8_e5m2: datatype.float8_e5m2
    })

    @register_type_handler(torch.Tensor, allow_subtypes=True)
    def torch_tensor_to_array(x: torch.Tensor) -> ArrayTy:
        if x.device.type != 'cuda':
            raise ValueError('Tensor must be on cuda device')
        # Assume dynamic shape
        shape = TupleTy(SizeTy(None) for i in x.shape)
        dtype = to_dtype(x.dtype)

        arr_special = ArraySpecialization(
            x.data_ptr(), dtype.bitwidth, x.shape, x.stride())

        # Assume dynamic strides except for the ones marked static
        strides = TupleTy(SizeTy(i) if static else SizeTy(None)
                          for i, static in
                          zip(x.stride(), arr_special.stride_is_static))

        return ArrayTy(dtype, shape=shape, strides=strides,
                       elements_disjoint=arr_special.elements_disjoint,
                       base_ptr_div_by=arr_special.base_ptr_div_by,
                       stride_div_by=arr_special.stride_div_by,
                       shape_div_by=arr_special.shape_div_by)


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


@register_type_handler('__cuda_array_interface__')
def from_cuda_array_interface(x: Any) -> ArrayTy:
    desc = x.__cuda_array_interface__
    # Assume dynamic shape
    shape = TupleTy(SizeTy(None) for i in desc['shape'])
    dtype = to_dtype(np.dtype(desc['typestr']))
    if dtype.bitwidth % BYTE_BITWIDTH != 0:
        raise ValueError("Only byte-aligned types should be supported by __cuda_array_interface__")

    dtype_bytewidth = dtype.bitwidth // BYTE_BITWIDTH
    byte_strides = desc.get('strides', None)
    elem_strides = _compute_elem_strides(desc['shape'], dtype_bytewidth, byte_strides)

    assert len(elem_strides) == len(shape)

    arr_special = ArraySpecialization(
            desc['data'][0], dtype.bitwidth, tuple(desc['shape']), elem_strides)

    # Assume dynamic strides except for the ones marked static
    strides = TupleTy(SizeTy(i) if static else SizeTy(None)
                      for i, static in
                      zip(elem_strides, arr_special.stride_is_static))

    return ArrayTy(dtype, shape=shape, strides=strides,
                   elements_disjoint=arr_special.elements_disjoint,
                   base_ptr_div_by=arr_special.base_ptr_div_by,
                   stride_div_by=arr_special.stride_div_by,
                   shape_div_by=arr_special.shape_div_by)
