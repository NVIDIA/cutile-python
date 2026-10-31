# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import cuda.lang._datatype as datatype
from cuda.lang._enums import MemoryScope
from cuda.lang._exception import TypeCheckingError
from cuda.lang._ir.ir import Var
from cuda.lang._ir.type import MemorySpace, ScalarTy, VectorTy
from cuda.lang._ir.type_checking_helpers import (
    require_concrete_pointer_type,
    require_mbarrier_ptr,
    require_pointer_in_memory_space,
    require_scalar_or_vector_type,
    require_scalar_type,
)
from cuda.lang._stub import store_async
from cuda.tile._ir.arithmetic_ops import astype
from cuda.tile._ir.core_ops import strictly_typed_const
from cuda.tile._ir.op_impl import (
    ImplRegistry,
    require_constant_bool,
    require_constant_enum,
)
from cuda.tile._ir.ops import implicit_cast

from .core_api_impl import bitcast
from .inline_ptx_impl import inline_ptx
from .vector_impl import vector_getitem_raw


_registry = ImplRegistry()
impl = _registry.impl


def store_async_impl_registry() -> ImplRegistry:
    return _registry


def as_integer(value: Var, bitwidth: int) -> Var:
    int_type = getattr(datatype, f"int{bitwidth}")
    return bitcast(value, int_type)


def _require_typed_pointer(pointer: Var, memory_space: MemorySpace):
    pointer_type = require_concrete_pointer_type(pointer)
    require_pointer_in_memory_space(pointer, (memory_space,))
    return pointer_type


@impl(store_async.store_async_global)
def store_async_global_impl(
    destination_address: Var,
    value: Var,
    scope: Var,
    is_multimem: Var,
) -> None:
    destination_type = _require_typed_pointer(destination_address, MemorySpace.GENERIC)
    value = implicit_cast(
        value,
        destination_type.pointee_dtype,
        "Stored value type is incompatible with pointer type",
    )
    value_type = require_scalar_type(value)
    scope = require_constant_enum(scope, MemoryScope)
    is_multimem = require_constant_bool(is_multimem)
    bitwidth = value_type.dtype.bitwidth

    valid = (8, 16, 32, 64)
    if bitwidth not in valid:
        raise TypeCheckingError(f"Valid value bitwidths are {valid}")

    if scope is MemoryScope.DEVICE:
        scope_suffix = "gpu"
    elif scope is MemoryScope.SYS:
        scope_suffix = "sys"
    else:
        raise TypeCheckingError(
            "store_async_global scope must be MemoryScope.DEVICE or MemoryScope.SYS"
        )

    if value_type.dtype is datatype.bool_:
        value = astype(value, datatype.int8)
    else:
        value = as_integer(value, bitwidth)

    mnemonic = ("multimem." if is_multimem else "") + "st.async"
    text = f"{mnemonic}.release.{scope_suffix}.global.b{bitwidth} [%0], %1;"
    inline_ptx(text, destination_address, value)


@impl(store_async.store_async_cluster)
def store_async_cluster_impl(
    destination_address: Var,
    value: Var,
    mbarrier: Var,
) -> None:
    destination_type = _require_typed_pointer(
        destination_address, MemorySpace.SHARED_CLUSTER
    )
    require_mbarrier_ptr(mbarrier, (MemorySpace.SHARED_CLUSTER,))
    value = implicit_cast(
        value,
        destination_type.pointee_dtype,
        "Stored value type is incompatible with pointer type",
    )
    value_type = require_scalar_or_vector_type(value)

    if isinstance(value_type, ScalarTy):
        bitwidth = value_type.dtype.bitwidth
        if bitwidth not in (32, 64):
            raise TypeCheckingError(
                "store_async_cluster requires a 32- or 64-bit scalar value"
            )
        registers = (as_integer(value, bitwidth),)
        value_suffix = f".b{bitwidth}"
    else:
        assert isinstance(value_type, VectorTy)
        bitwidth = value_type.element_dtype.bitwidth
        valid = ((2, 32), (2, 64), (4, 32))
        if (value_type.length, bitwidth) not in valid:
            message = f"Valid combinations of vector length and bitwidth are {valid}"
            raise TypeCheckingError(message)
        index_type = ScalarTy(datatype.int32)
        elements = tuple(
            vector_getitem_raw(value, strictly_typed_const(index, index_type))
            for index in range(value_type.length)
        )
        registers = tuple(as_integer(element, bitwidth) for element in elements)
        value_suffix = f".v{value_type.length}.b{bitwidth}"

    if len(registers) == 1:
        value_operand = "%1"
    else:
        value_operand = "{" + ", ".join(f"%{i}" for i, _ in enumerate(registers, 1)) + "}"

    inline_ptx(
        f"st.async.shared::cluster.mbarrier::complete_tx::bytes{value_suffix}"
        f" [%0], {value_operand}, [%{len(registers) + 1}];",
        destination_address, *registers, mbarrier
    )
