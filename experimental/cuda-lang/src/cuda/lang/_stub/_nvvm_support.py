# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
import enum
import importlib
import inspect
import typing
from dataclasses import dataclass
from functools import cache, lru_cache
from typing import Callable, Any, Annotated, NamedTuple, Sequence

from cuda.lang._execution import stub
from cuda.lang._ir.op_defs import call_intrinsic
from cuda.lang._ir.type import PointerTy, ScalarTy, VectorTy, make_rank0_ty
from cuda.lang._ir.type_checking_helpers import require_vector_type, require_scalar_or_vector_type
from cuda.lang._exception import TypeCheckingError, InvalidValueError
from cuda.lang._passes.ir2llvm import DIRECTLY_SUPPORTED_FLOATS
from cuda.tile import DType
from cuda.tile._datatype import is_pointer_dtype, PointerInfo, is_integral
from cuda.tile._ir.op_impl import require_scalar_type, make_type_checking_error
from cuda.tile._ir.ir import Var
from cuda.tile._ir.ops import build_tuple
from cuda.tile._ir.cast_ops import implicit_cast
from cuda.tile import _datatype as datatype
from cuda.tile._ir.type import Type, StringTy
from cuda.tile._memory_model import MemorySpace


class RawIntrinsicImpl:
    _is_coroutine = False

    def __init__(self, prefix: str):
        self.prefix = prefix

    def __call__(self, stub, *args: Var):
        return call_intrinsic(stub, *args)


_nvvm_intrinsic_impl = RawIntrinsicImpl("llvm.nvvm.")
_llvm_intrinsic_impl = RawIntrinsicImpl("llvm.")


def _libdevice_func_impl(stub, *args: Var):
    from cuda.lang._ir.op_defs import call_libdevice_function
    return call_libdevice_function(stub, *args)


@dataclass
class TypeArgument:
    first_param_idx: int
    ty: Type


class MatchedSignature(NamedTuple):
    prepared_operands: tuple[Var | None, ...]
    result_types: tuple[Type, ...]
    make_retval: Callable
    metadata_args: tuple[Any, ...]


def match_intrinsic_signature(stub, args: tuple[Var, ...]) -> MatchedSignature:
    from cuda.lang._ir.type import VectorTy
    stub_sig = inspect.signature(stub)

    prepared_operands = []
    type_arguments = []
    metadata_args = []
    for param_idx, (arg, param) in enumerate(zip(args, stub_sig.parameters.values(), strict=True)):
        ann = _get_annotation(param.annotation)
        if isinstance(ann, _IntrinsicDTypeAnnotation):
            if ann.vector_length is None:
                require_scalar_type(arg)
            else:
                require_vector_type(arg, ann.vector_length)
            arg = _implicit_cast_with_fallback(arg, ann.dtype, f"Invalid argument #{param_idx}")
        elif isinstance(ann, _IntrinsicSameShapeAnnotation):
            assert ann.index < len(type_arguments) and type_arguments[ann.index] is not None
            actual_ty = require_scalar_or_vector_type(arg)
            arg = _implicit_cast_with_fallback(arg, ann.dtype, f"Invalid argument #{param_idx}")

            ref_ty = type_arguments[ann.index]
            if ref_ty.tensor_shape() != actual_ty.tensor_shape():
                raise make_type_checking_error(
                        f"Argument shape mismatch:"
                        f" {ref_ty.tensor_shape()} vs {actual_ty.tensor_shape()}")
        elif isinstance(ann, _IntrinsicGenericAnnotation):
            if ann.index < len(type_arguments) and type_arguments[ann.index] is not None:
                if type_arguments[ann.index].ty != arg.get_type():
                    raise TypeCheckingError(
                        f"Types arguments #{param_idx}"
                        f" and #{type_arguments[ann.index].first_param_idx} don't match"
                        f" ({type_arguments[ann.index].ty} vs {arg.get_type()})")
                else:
                    match ann.vector_kind:
                        case _IntrinsicVectorKind.Any:
                            dtype = require_scalar_or_vector_type(arg).dtype
                        case _IntrinsicVectorKind.Scalar:
                            dtype = require_scalar_type(arg).tensor_dtype()
                        case _IntrinsicVectorKind.Vector:
                            dtype = require_vector_type(arg).element_dtype
                        case x:
                            assert False, x

                    match ann.element_kind:
                        case _IntrinsicGenericKind.Any:
                            pass
                        case _IntrinsicGenericKind.Integer:
                            if not datatype.is_integral(dtype):
                                raise make_type_checking_error(
                                    f"Expected an integer scalar, got {dtype}", arg)
                        case _IntrinsicGenericKind.Float:
                            if dtype not in DIRECTLY_SUPPORTED_FLOATS:
                                raise make_type_checking_error(
                                    f"Expected a scalar of a LLVM-supported float type,"
                                    f" got {dtype}", arg)
                        case _IntrinsicGenericKind.Pointer:
                            if not is_pointer_dtype(dtype):
                                raise make_type_checking_error(
                                    f"Expected a pointer scalar, got {dtype}", arg)
                        case x: assert False, x
            else:
                while len(type_arguments) <= ann.index:
                    type_arguments.append(None)
                type_arguments[ann.index] = arg.get_type()
        elif isinstance(ann, _IntrinsicMetadataAnnotation):
            meta = require_llvm_metadata(arg)
            metadata_args.append(meta)
            arg = None
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
        if isinstance(ann, _IntrinsicDTypeAnnotation):
            if ann.vector_length is None:
                ty = make_rank0_ty(ann.dtype)
            else:
                ty = VectorTy(ann.dtype, ann.vector_length)
        elif isinstance(ann, _IntrinsicSameShapeAnnotation):
            ty = type_arguments[ann.index]
            if ty is None:
                raise TypeCheckingError("Failed to infer return type of intrinsic")

            if isinstance(ty, ScalarTy):
                ty = ScalarTy(ann.dtype)
            else:
                assert isinstance(ty, VectorTy)
                ty = VectorTy(ann.dtype, ty.length)
        elif isinstance(ann, _IntrinsicGenericAnnotation):
            ty = type_arguments[ann.index]
            if ty is None:
                raise TypeCheckingError("Failed to infer return type of intrinsic")
        else:
            assert False
        result_types.append(ty)

    return MatchedSignature(tuple(prepared_operands), tuple(result_types), make_retval,
                            tuple(metadata_args))


def require_llvm_metadata(var: Var):
    ty = var.get_type()
    if isinstance(ty, StringTy):
        return ty.value

    raise make_type_checking_error(f"Expected an LLVM metadata, got {ty}", var)


def mangle_intrinsic_name(intrinsic_name: str, operand_types: Sequence[Type | None]):
    stub = find_intrinsic_stub(intrinsic_name)
    stub_sig = inspect.signature(stub)

    parts = [intrinsic_name]
    for ty, param in zip(operand_types, stub_sig.parameters.values(), strict=True):
        ann = _get_annotation(param.annotation)
        if isinstance(ann, _IntrinsicDTypeAnnotation | _IntrinsicSameShapeAnnotation):
            assert ty is not None
        elif isinstance(ann, _IntrinsicMetadataAnnotation):
            assert ty is None
        elif isinstance(ann, _IntrinsicGenericAnnotation):
            parts.append(_mangle_type_name(ty))
        else:
            assert False, type(param)
    return ".".join(parts)


def _mangle_type_name(ty: Type) -> str:
    if isinstance(ty, PointerTy | ScalarTy):
        return _mangle_dtype_name(ty.tensor_dtype())
    elif isinstance(ty, VectorTy):
        return f"v{ty.length}" + _mangle_dtype_name(ty.element_dtype)
    else:
        raise NotImplementedError()


def _mangle_dtype_name(dtype: DType) -> str:
    if is_pointer_dtype(dtype):
        space = PointerInfo(dtype).memory_space._value_
        return f"p{space}"
    elif is_integral(dtype):
        return f"i{dtype.bitwidth}"
    elif (kind := DIRECTLY_SUPPORTED_FLOATS.get(dtype)) is not None:
        return kind._name_
    else:
        raise NotImplementedError(f"Dtype {dtype} not supported as LLVM intrinsic generic operand")


@lru_cache
def find_intrinsic_stub(intrinsic_name: str):
    assert intrinsic_name.startswith("llvm.")
    intrinsic_name = intrinsic_name.removeprefix("llvm.")
    if intrinsic_name.startswith("nvvm."):
        modules = nvvm_stub_modules
        intrinsic_name = intrinsic_name.removeprefix("nvvm.")
    else:
        modules = llvm_stub_modules

    intrinsic_name = intrinsic_name.replace(".", "_")

    for module_name in reversed(modules):
        stubs = _maybe_import_stubs(module_name)
        s = stubs.get(intrinsic_name)
        if s is not None:
            return s
    raise ValueError(f"Could not find the signature of intrinsic {intrinsic_name}")


nvvm_stub_modules = ["cuda.lang._stub.nvvm"]
llvm_stub_modules = ["cuda.lang._stub.llvm"]


@cache
def _maybe_import_stubs(module_name: str):
    try:
        module = importlib.import_module(module_name)
    except ImportError:
        return {}

    try:
        return module.__dict__
    except AttributeError:
        return {}


def _implicit_cast_with_fallback(src: Var, target_dtype: DType, error_context: str) -> Var:
    try:
        return implicit_cast(src, target_dtype, error_context)
    except (TypeCheckingError, InvalidValueError):
        if not (datatype.is_integral(src.get_type().dtype) and datatype.is_integral(target_dtype)):
            raise

    # LLVM integers are signless, so we need to try both possibilities (signed, unsigned)
    # for the implicit cast target
    fallback_dtype = datatype.integer_dtype(target_dtype.bitwidth, signed=False)
    return implicit_cast(src, fallback_dtype, error_context)


_libdevice_func_impl._is_coroutine = False


def nvvm_intrinsic_stub(func, *, name: str | None):
    func = stub(func)
    func._cutile_custom_implementation_handler = _nvvm_intrinsic_impl
    func._llvm_intrinsic_name = name
    return func


def llvm_intrinsic_stub(func, *, name: str | None):
    func = stub(func)
    func._cutile_custom_implementation_handler = _llvm_intrinsic_impl
    func._llvm_intrinsic_name = name
    return func


def libdevice_function_stub(func):
    func = stub(func)
    func._cutile_custom_implementation_handler = _libdevice_func_impl
    return func


@dataclass
class _IntrinsicDTypeAnnotation:
    dtype: DType
    vector_length: int | None = None


class _IntrinsicVectorKind(enum.Enum):
    Any = 0
    Vector = 1
    Scalar = 2


class _IntrinsicGenericKind(enum.Enum):
    Any = 0
    Integer = 1
    Float = 2
    Pointer = 3


@dataclass
class _IntrinsicGenericAnnotation:
    vector_kind: _IntrinsicVectorKind
    element_kind: _IntrinsicGenericKind
    index: int


@dataclass
class _IntrinsicMetadataAnnotation:
    pass


@dataclass
class _IntrinsicSameShapeAnnotation:
    dtype: DType
    index: int


_IntrinsicAnnotation = (_IntrinsicDTypeAnnotation | _IntrinsicGenericAnnotation
                        | _IntrinsicMetadataAnnotation | _IntrinsicSameShapeAnnotation)


def _get_annotation(type_hint) -> _IntrinsicAnnotation:
    assert typing.get_origin(type_hint) is Annotated, f"{type_hint} {typing.get_origin(type_hint)}"
    _, ann = typing.get_args(type_hint)
    assert isinstance(ann, _IntrinsicAnnotation)
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
P0 = Annotated[Any, _IntrinsicDTypeAnnotation(datatype.opaque_pointer_dtype())]
P1 = Annotated[Any, _IntrinsicDTypeAnnotation(datatype.opaque_pointer_dtype(MemorySpace.GLOBAL))]
P3 = Annotated[Any, _IntrinsicDTypeAnnotation(datatype.opaque_pointer_dtype(MemorySpace.SHARED))]
P4 = Annotated[Any, _IntrinsicDTypeAnnotation(datatype.opaque_pointer_dtype(MemorySpace.CONSTANT))]
P5 = Annotated[Any, _IntrinsicDTypeAnnotation(datatype.opaque_pointer_dtype(MemorySpace.LOCAL))]
P6 = Annotated[Any, _IntrinsicDTypeAnnotation(datatype.opaque_pointer_dtype(MemorySpace.TENSOR))]
P7 = Annotated[Any, _IntrinsicDTypeAnnotation(
    datatype.opaque_pointer_dtype(MemorySpace.SHARED_CLUSTER))]
Meta = Annotated[Any, _IntrinsicMetadataAnnotation()]
