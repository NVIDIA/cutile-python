# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Literal, TypeVar

from cuda.tile._datatype import is_pointer_dtype
from cuda.tile._memory_model import MemoryScope, MemorySpace
import cuda.lang._datatype as datatype
from cuda.lang._enums import FenceProxyKind, MemoryOrder
from cuda.lang._execution import stub
from cuda.lang._ir.enum_to_mlir import cl_enum_to_mlir_attribute
import cuda.lang._mlir as mlir
from cuda.lang._mlir.nvvm import MemScopeKind, MemOrderKind, ProxyKind, SharedSpace
from cuda.lang._stub._nvvm_support import (
    _IntrinsicDTypeAnnotation,
    _IntrinsicPredicateAnnotation,
)
from cuda.lang._exception import TypeCheckingError, InvalidValueError
from cuda.tile._ir.op_impl import (
    require_constant_bool,
    require_constant_enum,
    require_constant_int,
    require_tuple_type,
)
from cuda.tile._ir.ir import Var, add_operation_variadic
from cuda.tile._ir.ops import (
    implicit_cast,
    build_tuple,
)


FuncTy = TypeVar("FuncTy", bound=Callable[..., Any])


@dataclass(frozen=True)
class ArgSpec:
    type: object
    kind: Literal["operand", "attribute"] = "operand"
    optional: bool = False
    variadic: bool = False
    unit: bool = False
    name: str = ""


@dataclass(frozen=True)
class ResultSpec:
    name: str
    type: object
    optional: bool = False
    variadic: bool = False


@dataclass(frozen=True)
class AliasedEnumAttr:
    cl_enum: type[Enum]
    mlir_enum: type[Enum]
    value_map: tuple[tuple[Enum, Enum], ...]

    def cl2mlir(self, enum_val) -> Any:
        for cl_val, mlir_val in self.value_map:
            if enum_val == cl_val:
                return mlir_val
        valid = ", ".join(str(value) for value, _ in self.value_map)
        raise InvalidValueError(f"Expected one of {valid}, got {enum_val}")

    def mlir2cl(self, enum_val) -> Any:
        for cl_val, mlir_val in self.value_map:
            if enum_val == mlir_val:
                return cl_val
        valid = ", ".join(str(value) for _, value in self.value_map)
        raise InvalidValueError(f"Expected one of {valid}, got {enum_val}")


SharedSpaceAttr = AliasedEnumAttr(
    cl_enum=MemorySpace,
    mlir_enum=SharedSpace,
    value_map=(
        (MemorySpace.SHARED, SharedSpace.shared_cta),
        (MemorySpace.SHARED_CLUSTER, SharedSpace.shared_cluster),
    ),
)


MemoryOrderAttr = AliasedEnumAttr(
    cl_enum=MemoryOrder,
    mlir_enum=MemOrderKind,
    value_map=(
        (MemoryOrder.WEAK, MemOrderKind.WEAK),
        (MemoryOrder.RELAXED, MemOrderKind.RELAXED),
        (MemoryOrder.ACQUIRE, MemOrderKind.ACQUIRE),
        (MemoryOrder.RELEASE, MemOrderKind.RELEASE),
        (MemoryOrder.ACQ_REL, MemOrderKind.ACQ_REL),
    ),
)


MemoryScopeAttr = AliasedEnumAttr(
    cl_enum=MemoryScope,
    mlir_enum=MemScopeKind,
    value_map=(
        (MemoryScope.BLOCK, MemScopeKind.CTA),
        (MemoryScope.CLUSTER, MemScopeKind.CLUSTER),
        (MemoryScope.DEVICE, MemScopeKind.GPU),
        (MemoryScope.SYS, MemScopeKind.SYS),
    ),
)


FenceProxyKindAttr = AliasedEnumAttr(
    cl_enum=FenceProxyKind,
    mlir_enum=ProxyKind,
    value_map=(
        (FenceProxyKind.ALIAS, ProxyKind.alias),
        (FenceProxyKind.ASYNC, ProxyKind.async_),
        (FenceProxyKind.ASYNC_GLOBAL, ProxyKind.async_global),
        (FenceProxyKind.ASYNC_SHARED, ProxyKind.async_shared),
        (FenceProxyKind.TENSORMAP, ProxyKind.TENSORMAP),
        (FenceProxyKind.GENERIC, ProxyKind.GENERIC),
    ),
)


def is_none_constant(value: Var) -> bool:
    return value.is_constant() and value.get_constant() is None


def is_enum_type(ty) -> bool:
    return isinstance(ty, type) and issubclass(ty, Enum)


def make_enum_attr(enum_type: type[Enum], value: Enum) -> mlir.Attribute:
    attr_cls = getattr(mlir.nvvm, enum_type.__name__ + "Attr")
    return attr_cls(value=value)


def cast_operand(spec: ArgSpec, arg: Var) -> Var:
    target_type = spec.type
    src_type = arg.get_type()
    ctx = f"Attempting to cast argument to {target_type=}"
    match target_type:
        case _IntrinsicPredicateAnnotation():
            target_type.predicate(arg)
            return arg
        case _IntrinsicDTypeAnnotation():
            return implicit_cast(arg, target_type.dtype, ctx)
        case tuple():
            for target in target_type:
                try:
                    return implicit_cast(arg, target.dtype, ctx)
                except (TypeCheckingError, InvalidValueError):
                    pass
            options = ", ".join([str(t) for t in target_type])
            raise TypeCheckingError(
                f"Could not cast arg of type {src_type} to any of {options}"
            )
        case _:
            raise TypeCheckingError("Expected a predicate, a dtype, or a tuple of dtypes")


def make_mlir_attribute(spec: ArgSpec, arg: Var) -> tuple[str, mlir.Attribute] | None:
    if spec.optional and is_none_constant(arg):
        return None

    if isinstance(spec.type, AliasedEnumAttr):
        arg = require_constant_enum(arg, spec.type.cl_enum)
        return spec.name, cl_enum_to_mlir_attribute(arg)

    if is_enum_type(spec.type):
        arg = require_constant_enum(arg, spec.type)
        return spec.name, make_enum_attr(spec.type, arg)

    if spec.unit:
        arg = require_constant_bool(arg)
        return (spec.name, mlir.UnitAttr()) if arg else None

    dtype = spec.type.dtype
    if dtype is datatype.bool_:
        arg = require_constant_bool(arg)
        return spec.name, mlir.BoolAttr(value=arg)

    if datatype.is_integral(dtype):
        arg = require_constant_int(arg)
        ty = mlir.IntegerType.signless(dtype.bitwidth)
        attr = mlir.IntegerAttr.make(ty, int(arg))
        return spec.name, attr

    raise TypeCheckingError(f"Cannot convert argument into attribute: {spec}")


def get_raw_mlir_parts(
    arg_specs, has_operand_segment_sizes, args: tuple[Var, ...]
) -> tuple[tuple[Var, ...], tuple[tuple[str, mlir.Attribute], ...]]:
    operands = []
    attributes = []
    operand_segment_sizes = []
    for arg, spec in zip(args, arg_specs, strict=True):
        if spec.kind == "attribute":
            attr = make_mlir_attribute(spec, arg)
            if attr is not None:
                attributes.append(attr)

        elif spec.optional and is_none_constant(arg):
            operand_segment_sizes.append(0)

        elif spec.variadic:
            require_tuple_type(arg)
            aggregate = tuple(arg.flatten_aggregate())
            operands.extend(cast_operand(spec, item) for item in aggregate)
            operand_segment_sizes.append(len(aggregate))
        else:
            operands.append(cast_operand(spec, arg))
            operand_segment_sizes.append(1)

    if has_operand_segment_sizes:
        attributes.append(
            ("operandSegmentSizes", mlir.DenseI32ArrayAttr(operand_segment_sizes))
        )

    return tuple(operands), tuple(attributes)


def _raw_nvvm_mlir_operation_impl(stub_func, *args: Var):
    from cuda.lang._ir.type import ScalarTy, PointerTy
    from cuda.lang._ir.ops import RawMLIROperation

    result_types = tuple(PointerTy(ty.type.dtype) if is_pointer_dtype(ty.type.dtype)
                         else ScalarTy(ty.type.dtype)
                         for ty in stub_func._results)
    operands, attrs = get_raw_mlir_parts(
        stub_func._args, stub_func._attr_sized_operand_segments, args
    )
    results = add_operation_variadic(
        RawMLIROperation,
        result_types,
        op_name=stub_func._op_name,
        operands_=operands,
        mlir_attributes=attrs,
    )
    match len(stub_func._results):
        case 0:
            return None
        case 1:
            return results[0]
        case _:
            return build_tuple(results)


_raw_nvvm_mlir_operation_impl._is_coroutine = False


def nvvm_mlir_interface_stub(
    *,
    op_name: str,
    attr_sized_operand_segments: bool = False,
    results: tuple[ResultSpec, ...] = (),
    args: tuple[ArgSpec, ...] = (),
) -> Callable[[FuncTy], FuncTy]:
    def decorate(func: FuncTy) -> FuncTy:
        func = stub(func)
        func._cutile_custom_implementation_handler = _raw_nvvm_mlir_operation_impl
        func._op_name = op_name
        func._attr_sized_operand_segments = attr_sized_operand_segments
        func._results = results
        func._args = args
        return func

    return decorate


__all__ = (
    "AliasedEnumAttr",
    "ArgSpec",
    "FenceProxyKind",
    "FenceProxyKindAttr",
    "MemoryOrder",
    "MemoryOrderAttr",
    "MemoryScope",
    "MemoryScopeAttr",
    "ResultSpec",
    "nvvm_mlir_interface_stub",
)
