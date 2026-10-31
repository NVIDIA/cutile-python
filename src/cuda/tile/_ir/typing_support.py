# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
import ast
import importlib
import inspect
import operator
import dataclasses
from contextlib import _GeneratorContextManager
from functools import lru_cache, cache
from types import ModuleType, FunctionType, BuiltinFunctionType, MethodDescriptorType
from typing import Any, Sequence

from cuda.tile import _datatype as datatype, DType
from cuda.tile._exception import TileTypeError, TileValueError
from .ir import TypingHooks
from .type import DataclassInfo, PointerInfoTy, NotImplementedTy

from .type import Type, DTypeConstructor, DTypeSpec, NONE, StringTy, \
    ELLIPSIS, SLICE, ModuleTy, FunctionTy, EnumTy, TypeTy, LooselyTypedScalar
from .._cext import classify_constant, ConstantKind, foreign_dtype_object_to_native
from .._cext import Tile, Scalar, Vector, Pointer, OpaqueValue
from .._execution import is_function_wrapper


def to_dtype(x: Any):
    if isinstance(x, DType):
        return x
    ret = foreign_dtype_object_to_native(x)
    if ret is None:
        raise TypeError(f"{x} is not a dtype")
    return ret


@cache
def _get_dtype_spec(dtype: DType) -> DTypeSpec:
    assert isinstance(dtype, DType)
    return DTypeConstructor(dtype) if _is_dtype_allowed_as_constructor(dtype) else DTypeSpec(dtype)


def _is_dtype_allowed_as_constructor(dtype: DType) -> bool:
    # Only allow byte aligned numeric dtypes as constructors
    return datatype.is_numeric(dtype) and (dtype.bitwidth % 8 == 0)


def is_dtype_constructor(x: Any) -> bool:
    if not isinstance(x, DType):
        x = foreign_dtype_object_to_native(x)
        if x is None:
            return False
    return _is_dtype_allowed_as_constructor(x)


BUILTIN_FUNC_SIGNATURES = {
    abs: lambda x: None,
    len: lambda x, /: None,
    max: lambda x, y, /: None,
    min: lambda x, y, /: None,
    range: lambda *args: None,
    repr: lambda x, /: None,
    slice: lambda start, stop, step: None,
    str: lambda x, /: None,
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
    dict.get: dict.get,
    _GeneratorContextManager: lambda func, args, kwargs: None,
    str.join: lambda self, iterable, /: None
}


# Types defined in the C++ extension. Their docstrings are generated from _cext.pyi and carry
# the signatures: either as a text signature (__text_signature__), or, when a default can't be
# written as a literal (e.g. an enum member), as the first line of __doc__.
# inspect.signature() only supports the former, and only with literal defaults.
_CEXT_CLASSES = (Tile, Scalar, Vector, Pointer, OpaqueValue)


# Parses parameters like "self, /, shape, *, alignment=None". Default values are evaluated
# in `namespace`, so they can refer to its names, e.g. "MemoryOrder.ACQUIRE".
def _parse_params(params: str, namespace: ModuleType) -> inspect.Signature:
    definitions = {}
    exec(f"def f({params}): pass", vars(namespace), definitions)
    return inspect.signature(definitions["f"])


def _cext_params(f) -> tuple[str, ModuleType] | None:
    """Returns the parameters of a class or a method defined in the C++ extension, e.g.
    "self, /, shape", and the namespace for evaluating their defaults, or None if `f` isn't one"""
    if isinstance(f, type) and f in _CEXT_CLASSES:
        cls, is_method = f, False
    elif isinstance(f, MethodDescriptorType) and f.__objclass__ in _CEXT_CLASSES:
        cls, is_method = f.__objclass__, True
    else:
        return None
    namespace = importlib.import_module(cls.__module__)

    if f.__text_signature__ is not None:
        # E.g. "($self, /, shape)"; `$self` marks the bound parameter
        return f.__text_signature__[1:-1].replace("$self", "self"), namespace

    # E.g. "atomic_load(*, memory_order=MemoryOrder.ACQUIRE)" on the first line of __doc__,
    # which doesn't include `self`. Like in a text signature, `self` is positional-only.
    first_line = (f.__doc__ or "").split("\n", 1)[0]
    prefix = f"{f.__name__}("
    if first_line.startswith(prefix) and first_line.endswith(")"):
        params = first_line[len(prefix):-1]
        if is_method:
            if "/" in params:
                params = "self, " + params
            else:
                params = "self, /, " + params if params else "self, /"
        return params, namespace

    if is_method:
        raise TypeError(f"{cls.__name__}.{f.__name__} has no signature in its docstring")
    # A class without a constructor
    return "", namespace


@cache
def _cext_signature(f) -> inspect.Signature | None:
    """Returns the signature of a class or a method defined in the C++ extension,
    or None if `f` isn't one"""
    params = _cext_params(f)
    if params is None:
        return None
    return _parse_params(*params)


def get_parameter_names(f) -> tuple[str, ...]:
    """Like get_signature(f).parameters, but doesn't evaluate default values. This allows checking
    the parameters while the modules that define the defaults may not be fully imported yet."""
    params = _cext_params(f)
    if params is None:
        return tuple(get_signature(f).parameters)
    args = ast.parse(f"def f({params[0]}): pass").body[0].args
    names = [arg.arg for arg in args.posonlyargs + args.args]
    if args.vararg is not None:
        names.append(args.vararg.arg)
    names += [arg.arg for arg in args.kwonlyargs]
    if args.kwarg is not None:
        names.append(args.kwarg.arg)
    return tuple(names)


def get_signature(f) -> inspect.Signature:
    if stub := BUILTIN_FUNC_SIGNATURES.get(f):
        f = stub
    elif is_dtype_constructor(f):
        # Data type constructors
        f = lambda x=0, /: None  # noqa: E731
    elif (sig := _cext_signature(f)) is not None:
        return sig

    if isinstance(f, type):
        return inspect.signature(f)

    while is_function_wrapper(f):
        f = f.__wrapped__
    return inspect.signature(f, follow_wrapped=False)


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


def type_of_constant_python_value(val, typing_hooks: TypingHooks) -> Type:
    kind = classify_constant(val, False)
    match kind:
        case None: pass
        case ConstantKind.Bool | ConstantKind.Int | ConstantKind.Float:
            return typing_hooks.get_tensor_like_type(dtype_of_constant_scalar(val), ())
        case ConstantKind.None_:
            return NONE
        case ConstantKind.String:
            return StringTy(val)
        case ConstantKind.Enum:
            return EnumTy(val)
        case ConstantKind.NativeDType:
            return _get_dtype_spec(val)
        case ConstantKind.ForeignDType:
            return _get_dtype_spec(foreign_dtype_object_to_native(val))
        case _: assert False

    if val is Ellipsis:
        return ELLIPSIS
    if val is NotImplemented:
        return NotImplementedTy.instance()
    if isinstance(val, slice):
        return SLICE
    if isinstance(val, ModuleType):
        return ModuleTy(val)
    if isinstance(val, FunctionType | BuiltinFunctionType):
        return FunctionTy(val)
    if isinstance(val, datatype.PointerInfo):
        return PointerInfoTy(val)
    if isinstance(val, type):
        return TypeTy(val)
    if (ty := typing_hooks.get_python_constant_type(val)) is not None:
        return ty

    ty = type(val)
    prefix = "" if ty.__module__ == "builtins" else f"{ty.__module__}."
    raise TileTypeError(f"Cannot create constant from value of type {prefix}{ty.__qualname__}.")


def loose_type_of_constant_python_value(value: Any, typing_hooks: TypingHooks) -> Type:
    if isinstance(value, bool | int | float):
        return LooselyTypedScalar(value)
    else:
        return type_of_constant_python_value(value, typing_hooks)


@lru_cache
def get_dataclass_info(cls) -> DataclassInfo:
    params = cls.__dataclass_params__
    if not params.frozen:
        raise TileTypeError("Only frozen dataclasses are supported")

    if "__dataclass_params__" not in cls.__dict__:
        raise TileTypeError("Non-dataclass subclasses of a dataclass are not supported")

    if _dataclass_has_default_init(cls):
        init_signature = inspect.signature(cls.__init__)
    else:
        init_signature = None

    if find_method(cls, "__new__") is not object.__new__:
        raise TileTypeError("Dataclasses with custom __new__ are not supported")

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

    post_init = find_method(cls, "__post_init__")
    return DataclassInfo(cls, field_names, field_name_to_idx, init_signature, post_init)


def create_dataclass_instance(cls, field_values: Sequence[Any]):
    info = get_dataclass_info(cls)
    ret = cls.__new__(cls)
    for name, val in zip(info.field_names, field_values, strict=True):
        object.__setattr__(ret, name, val)
    return ret


def _dataclass_has_default_init(cls) -> bool:
    if not cls.__dataclass_params__.init:
        return False

    code = cls.__init__.__code__

    if "__dataclass_builtins_object__" in code.co_freevars:
        return True

    # If the dataclass is not empty, return False
    if dataclasses.fields(cls):
        return False

    return _is_code_seemingly_generated_by_dataclasses_library(code)


def dataclass_has_default_cmp(cls, method: str):
    if method == "__ne__":
        return cls.__ne__ is object.__ne__ and dataclass_has_default_cmp(cls, "__eq__")

    if method == "__eq__":
        flag = cls.__dataclass_params__.eq
    else:
        assert method in ("__lt__", "__gt__", "__le__", "__ge__")
        flag = cls.__dataclass_params__.order
    if not flag:
        return False
    return _is_code_seemingly_generated_by_dataclasses_library(getattr(cls, method).__code__)


def _is_code_seemingly_generated_by_dataclasses_library(code):
    co_qualname = getattr(code, "co_qualname", None)
    if co_qualname is not None:
        return co_qualname == "__create_fn__.<locals>." + code.co_name

    # HACK: This is a fallback for python 3.10 as co_qualname was introduced in python 3.11+
    return code.co_filename == "<string>"


def dataclass_has_default_repr(cls) -> bool:
    return (
            cls.__dataclass_params__.repr
            and (
                    # HACK HACK HACK!
                    (wrapped := getattr(cls.__repr__, "__wrapped__", None)) is not None
                    and getattr(wrapped, "__qualname__", "") == "__create_fn__.<locals>.__repr__"
            )
    )


def find_method(cls, name: str):
    for b in cls.__mro__:
        if name in b.__dict__:
            return b.__dict__[name]
    return NotImplemented


def _compute_elem_strides(shape, dtype_bytewidth, byte_strides):
    if byte_strides is not None:
        return tuple(bs // dtype_bytewidth for bs in byte_strides)

    if len(shape) == 0:
        return tuple()

    reverse_elem_strides = [1]
    for i in shape[-1:0:-1]:
        reverse_elem_strides.append(reverse_elem_strides[-1] * i)

    return tuple(reverse_elem_strides[::-1])
