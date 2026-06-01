# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
import inspect
import typing
from dataclasses import dataclass
from typing import Callable, Any, Annotated, NamedTuple

from cuda.lang._execution import stub
from cuda.lang._ir.type import TileTy
from cuda.tile import DType, TileValueError, TileTypeError
from cuda.tile._ir.op_impl import require_integer_0d_tile_type, require_scalar_pointer_type, \
    require_scalar_type, require_vector_type, require_any_vector_type, \
    require_any_scalar_or_vector_type
from cuda.tile._ir.ir import Var, add_operation_variadic
from cuda.tile._ir.ops import build_tuple
from cuda.tile._ir.cast_ops import implicit_cast
from cuda.tile import _datatype as datatype
from cuda.tile._ir.type import Type
from cuda.tile._memory_model import MemorySpace


def _raw_nvvm_intrinsic_impl(stub, *args: Var):
    from cuda.lang._ir.ops import RawNVVMIntrinsic
    name = stub._nvvm_intrinsic_name
    if name is None:
        name = stub.__name__.replace("_", ".")

    prepared_operands, result_types, make_retval = _prepare_common(stub, args)
    return make_retval(add_operation_variadic(
        RawNVVMIntrinsic,
        tuple(result_types),
        intrinsic="llvm.nvvm." + name,
        operands_=tuple(prepared_operands),
    ))


def _libdevice_func_impl(stub, *args: Var):
    from cuda.lang._ir.ops import ForeignFunction
    name = stub.__name__
    if not name.startswith("__nv_"):
        name = "__nv_" + name

    prepared_operands, result_types, make_retval = _prepare_common(stub, args)
    return make_retval(add_operation_variadic(
        ForeignFunction,
        tuple(result_types),
        function_name=name,
        operands_=tuple(prepared_operands),
    ))


class PreparedCall(NamedTuple):
    prepared_operands: tuple[Var, ...]
    result_types: tuple[Type, ...]
    make_retval: Callable


def _prepare_common(stub, args: tuple[Var, ...]):
    stub_sig = inspect.signature(stub)

    prepared_operands = []
    for param_idx, (arg, param) in enumerate(zip(args, stub_sig.parameters.values(), strict=True)):
        ann = _get_annotation(param.annotation)
        if isinstance(ann, _IntrinsicDTypeAnnotation):
            if ann.vector_length is None:
                require_scalar_type(arg)
            else:
                require_vector_type(arg, ann.vector_length)
            arg = _implicit_cast_with_fallback(arg, ann.dtype, f"Invalid argument #{param_idx}")
        elif isinstance(ann, _IntrinsicPredicateAnnotation):
            ann.predicate(arg)
        else:
            assert False

        prepared_operands.append(arg)

    if stub_sig.return_annotation is None:
        ret_type_hints = []
        def make_retval(_): return None
    elif typing.get_origin(stub_sig.return_annotation) is tuple:
        ret_type_hints = typing.get_args(stub_sig.return_annotation)
        def make_retval(result_vars): return build_tuple(result_vars)
    else:
        ret_type_hints = [stub_sig.return_annotation]
        def make_retval(result_vars): return result_vars[0]

    result_types = []
    for h in ret_type_hints:
        ann = _get_annotation(h)
        assert isinstance(ann, _IntrinsicDTypeAnnotation)
        shape = () if ann.vector_length is None else (ann.vector_length,)
        result_types.append(TileTy(ann.dtype, shape))

    return PreparedCall(tuple(prepared_operands), tuple(result_types), make_retval)


def _implicit_cast_with_fallback(src: Var, target_dtype: DType, error_context: str) -> Var:
    try:
        return implicit_cast(src, target_dtype, error_context)
    except (TileTypeError, TileValueError):
        if not (datatype.is_integral(src.get_type().dtype) and datatype.is_integral(target_dtype)):
            raise

    # LLVM integers are signless, so we need to try both possibilities (signed, unsigned)
    # for the implicit cast target
    fallback_dtype = datatype.integer_dtype(target_dtype.bitwidth, signed=False)
    return implicit_cast(src, fallback_dtype, error_context)


_raw_nvvm_intrinsic_impl._is_coroutine = False
_libdevice_func_impl._is_coroutine = False


def nvvm_intrinsic_stub(func, *, name: str | None):
    func = stub(func)
    func._cutile_custom_implementation_handler = _raw_nvvm_intrinsic_impl
    func._nvvm_intrinsic_name = name
    return func


def libdevice_function_stub(func):
    func = stub(func)
    func._cutile_custom_implementation_handler = _libdevice_func_impl
    return func


@dataclass
class _IntrinsicDTypeAnnotation:
    dtype: DType
    vector_length: int | None = None


@dataclass
class _IntrinsicPredicateAnnotation:
    predicate: Callable[[Var], Any]


def _get_annotation(type_hint) -> _IntrinsicDTypeAnnotation | _IntrinsicPredicateAnnotation:
    assert typing.get_origin(type_hint) is Annotated, f"{type_hint} {typing.get_origin(type_hint)}"
    _, ann = typing.get_args(type_hint)
    assert isinstance(ann, _IntrinsicDTypeAnnotation | _IntrinsicPredicateAnnotation)
    return ann


B = Annotated[Any, _IntrinsicDTypeAnnotation(datatype.bool_)]
BF16 = Annotated[Any, _IntrinsicDTypeAnnotation(datatype.bfloat16)]
F16 = Annotated[Any, _IntrinsicDTypeAnnotation(datatype.float16)]
F32 = Annotated[Any, _IntrinsicDTypeAnnotation(datatype.float32)]
F64 = Annotated[Any, _IntrinsicDTypeAnnotation(datatype.float64)]
I8 = Annotated[Any, _IntrinsicDTypeAnnotation(datatype.int8)]
I16 = Annotated[Any, _IntrinsicDTypeAnnotation(datatype.int16)]
I32 = Annotated[Any, _IntrinsicDTypeAnnotation(datatype.int32)]
I64 = Annotated[Any, _IntrinsicDTypeAnnotation(datatype.int64)]
U32 = Annotated[Any, _IntrinsicDTypeAnnotation(datatype.uint32)]
U64 = Annotated[Any, _IntrinsicDTypeAnnotation(datatype.uint64)]
IX = Annotated[Any, _IntrinsicPredicateAnnotation(require_integer_0d_tile_type)]
P0 = Annotated[Any, _IntrinsicDTypeAnnotation(datatype.opaque_pointer_dtype())]
P1 = Annotated[Any, _IntrinsicDTypeAnnotation(datatype.opaque_pointer_dtype(MemorySpace.GLOBAL))]
P3 = Annotated[Any, _IntrinsicDTypeAnnotation(datatype.opaque_pointer_dtype(MemorySpace.SHARED))]
P4 = Annotated[Any, _IntrinsicDTypeAnnotation(datatype.opaque_pointer_dtype(MemorySpace.CONSTANT))]
P5 = Annotated[Any, _IntrinsicDTypeAnnotation(datatype.opaque_pointer_dtype(MemorySpace.LOCAL))]
P6 = Annotated[Any, _IntrinsicDTypeAnnotation(datatype.opaque_pointer_dtype(MemorySpace.TENSOR))]
P7 = Annotated[Any, _IntrinsicDTypeAnnotation(
    datatype.opaque_pointer_dtype(MemorySpace.SHARED_CLUSTER))]
PX = Annotated[Any, _IntrinsicPredicateAnnotation(require_scalar_pointer_type)]
VX = Annotated[Any, _IntrinsicPredicateAnnotation(require_any_vector_type)]
X = Annotated[Any, _IntrinsicPredicateAnnotation(require_any_scalar_or_vector_type)]
