# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import cuda.lang._datatype as datatype
from cuda.lang._enums import MbarrierScope
from cuda.lang._exception import InternalError, TypeCheckingError
from cuda.lang._ir.ir import Var, add_operation
from cuda.lang._ir.op_defs import RawNVVMIntrinsic
from cuda.lang._ir.type import MemorySpace, ScalarTy
from cuda.lang._ir.type_checking_helpers import is_none, require_mbarrier_ptr
from cuda.lang._stub import mbarrier
from cuda.tile._ir.arithmetic_ops import astype
from cuda.tile._ir.ir import add_operation_variadic
from cuda.tile._ir.op_impl import (
    ImplRegistry,
    require_constant_bool,
    require_constant_enum,
)
from cuda.lang._enums import MemoryOrder

_registry = ImplRegistry()
impl = _registry.impl


def mbarrier_impl_registry() -> ImplRegistry:
    return _registry


@impl(mbarrier.mbarrier_initialize)
def mbarrier_initialize_impl(mbar: Var, participants: Var) -> Var:
    require_mbarrier_ptr(mbar)
    participants = astype(participants, datatype.int32)
    add_operation_variadic(
        RawNVVMIntrinsic,
        tuple(),
        intrinsic="llvm.nvvm.mbarrier.init.shared",
        operands_=(mbar, participants),
    )


@impl(mbarrier.mbarrier_invalidate)
def mbarrier_invalidate_impl(mbar: Var) -> Var:
    require_mbarrier_ptr(mbar)
    add_operation_variadic(
        RawNVVMIntrinsic,
        tuple(),
        intrinsic="llvm.nvvm.mbarrier.inval.shared",
        operands_=(mbar,),
    )


def _mbar_space_scope_suffix(scope: MbarrierScope, space: MemorySpace) -> str:
    match space:
        case MemorySpace.SHARED:
            space_str = "cta"
        case MemorySpace.SHARED_CLUSTER:
            space_str = "cluster"
        case _:
            raise InternalError(f"Unexpected {space=}")
    return ".scope." + scope.value + ".space." + space_str


def require_mbarrier_memory_order(
    memory_order_var: Var,
    valid_memory_orders: tuple[MemoryOrder, ...],
) -> MemoryOrder:
    memory_order = require_constant_enum(memory_order_var, MemoryOrder)
    if memory_order not in valid_memory_orders:
        formatted = ", ".join(str(o) for o in valid_memory_orders)
        raise TypeCheckingError(
            f"Invalid mbarrier memory order {memory_order}, expected one of {formatted}"
        )
    return memory_order


ARRIVE_MEMORY_ORDERS = (MemoryOrder.RELEASE, MemoryOrder.RELAXED)
WAIT_MEMORY_ORDERS = (MemoryOrder.ACQUIRE, MemoryOrder.RELAXED)


@impl(mbarrier.mbarrier_arrive)
def mbarrier_arrive_impl(
    mbar: Var,
    count: Var,
    drop: Var,
    scope: Var,
    memory_order: Var,
) -> Var | None:
    count = astype(count, datatype.int32)
    drop = require_constant_bool(drop)
    scope = require_constant_enum(scope, MbarrierScope)
    memory_order = require_mbarrier_memory_order(memory_order, ARRIVE_MEMORY_ORDERS)
    space = require_mbarrier_ptr(mbar).memory_space
    intrinsic = "llvm.nvvm.mbarrier.arrive"
    if drop:
        intrinsic += ".drop"
    if memory_order is MemoryOrder.RELAXED:
        intrinsic += ".relaxed"
    intrinsic += _mbar_space_scope_suffix(scope, space)

    return_type = (ScalarTy(datatype.uint64),) if space is MemorySpace.SHARED else ()
    results = add_operation_variadic(
        RawNVVMIntrinsic,
        return_type,
        intrinsic=intrinsic,
        operands_=(mbar, count),
    )
    return results[0] if return_type else None


@impl(mbarrier.mbarrier_arrive_expect_transaction)
def mbarrier_arrive_expect_transaction_impl(
    mbar: Var,
    bytes: Var,
    drop: Var,
    scope: Var,
    memory_order: Var,
) -> Var | None:
    bytes = astype(bytes, datatype.int32)
    drop = require_constant_bool(drop)
    scope = require_constant_enum(scope, MbarrierScope)
    memory_order = require_mbarrier_memory_order(memory_order, ARRIVE_MEMORY_ORDERS)
    space = require_mbarrier_ptr(mbar).memory_space
    intrinsic = "llvm.nvvm.mbarrier.arrive"
    if drop:
        intrinsic += ".drop"
    intrinsic += ".expect.tx"
    if memory_order is MemoryOrder.RELAXED:
        intrinsic += ".relaxed"
    intrinsic += _mbar_space_scope_suffix(scope, space)

    return_type = (ScalarTy(datatype.uint64),) if space is MemorySpace.SHARED else ()
    results = add_operation_variadic(
        RawNVVMIntrinsic,
        return_type,
        intrinsic=intrinsic,
        operands_=(mbar, bytes),
    )
    return results[0] if return_type else None


@impl(mbarrier.mbarrier_expect_transaction)
def mbarrier_expect_transaction_impl(mbar: Var, bytes: Var, scope: Var):
    space = require_mbarrier_ptr(mbar).memory_space
    bytes = astype(bytes, datatype.int32)
    scope = require_constant_enum(scope, MbarrierScope)
    intrinsic = "llvm.nvvm.mbarrier.expect.tx"
    intrinsic += _mbar_space_scope_suffix(scope, space)
    add_operation_variadic(
        RawNVVMIntrinsic,
        (),
        intrinsic=intrinsic,
        operands_=(mbar, bytes),
    )


@impl(mbarrier.mbarrier_complete_transaction)
def mbarrier_complete_transaction_impl(mbar: Var, bytes: Var, scope: Var) -> Var:
    space = require_mbarrier_ptr(mbar).memory_space
    bytes = astype(bytes, datatype.int32)
    scope = require_constant_enum(scope, MbarrierScope)
    intrinsic = "llvm.nvvm.mbarrier.complete.tx"
    intrinsic += _mbar_space_scope_suffix(scope, space)
    add_operation_variadic(
        RawNVVMIntrinsic,
        (),
        intrinsic=intrinsic,
        operands_=(mbar, bytes),
    )


@impl(mbarrier.mbarrier_test_wait)
def mbarrier_test_wait_impl(
    mbar: Var, state: Var, scope: Var, memory_order: Var
) -> Var:
    scope = require_constant_enum(scope, MbarrierScope)
    state = astype(state, datatype.int64)
    require_mbarrier_ptr(mbar, (MemorySpace.SHARED,))
    memory_order = require_mbarrier_memory_order(memory_order, WAIT_MEMORY_ORDERS)
    intrinsic = "llvm.nvvm.mbarrier.test.wait"
    if memory_order is MemoryOrder.RELAXED:
        intrinsic += ".relaxed"
    intrinsic += _mbar_space_scope_suffix(scope, MemorySpace.SHARED)
    return add_operation(
        RawNVVMIntrinsic,
        ScalarTy(datatype.bool_),
        intrinsic=intrinsic,
        operands_=(mbar, state),
    )


@impl(mbarrier.mbarrier_test_wait_parity)
def mbarrier_test_wait_parity_impl(
    mbar: Var, parity: Var, scope: Var, memory_order: Var
) -> Var:
    require_mbarrier_ptr(mbar, (MemorySpace.SHARED,))
    parity = astype(parity, datatype.int32)
    scope = require_constant_enum(scope, MbarrierScope)
    memory_order = require_mbarrier_memory_order(memory_order, WAIT_MEMORY_ORDERS)
    intrinsic = "llvm.nvvm.mbarrier.test.wait.parity"
    if memory_order is MemoryOrder.RELAXED:
        intrinsic += ".relaxed"
    intrinsic += _mbar_space_scope_suffix(scope, MemorySpace.SHARED)
    return add_operation(
        RawNVVMIntrinsic,
        ScalarTy(datatype.bool_),
        intrinsic=intrinsic,
        operands_=(mbar, parity),
    )


@impl(mbarrier.mbarrier_try_wait)
def mbarrier_try_wait_impl(
    mbar: Var,
    state: Var,
    time_hint: Var,
    scope: Var,
    memory_order: Var,
) -> Var:
    require_mbarrier_ptr(mbar, (MemorySpace.SHARED,))
    state = astype(state, datatype.int64)
    scope = require_constant_enum(scope, MbarrierScope)
    memory_order = require_mbarrier_memory_order(memory_order, WAIT_MEMORY_ORDERS)
    intrinsic = "llvm.nvvm.mbarrier.try.wait"
    args = (mbar, state)
    if not is_none(time_hint):
        intrinsic += ".tl"
        time_hint = astype(time_hint, datatype.int32)
        args = (*args, time_hint)
    if memory_order is MemoryOrder.RELAXED:
        intrinsic += ".relaxed"
    intrinsic += _mbar_space_scope_suffix(scope, MemorySpace.SHARED)
    return add_operation(
        RawNVVMIntrinsic,
        ScalarTy(datatype.bool_),
        intrinsic=intrinsic,
        operands_=args,
    )


@impl(mbarrier.mbarrier_try_wait_parity)
def mbarrier_try_wait_parity_impl(
    mbar: Var,
    parity: Var,
    time_hint: Var,
    scope: Var,
    memory_order: Var,
) -> Var:
    require_mbarrier_ptr(mbar, (MemorySpace.SHARED,))
    parity = astype(parity, datatype.int32)
    scope = require_constant_enum(scope, MbarrierScope)
    memory_order = require_mbarrier_memory_order(memory_order, WAIT_MEMORY_ORDERS)
    intrinsic = "llvm.nvvm.mbarrier.try.wait.parity"
    args = (mbar, parity)
    if not is_none(time_hint):
        time_hint = astype(time_hint, datatype.int32)
        args = (*args, time_hint)
        intrinsic += ".tl"
    if memory_order is MemoryOrder.RELAXED:
        intrinsic += ".relaxed"
    intrinsic += _mbar_space_scope_suffix(scope, MemorySpace.SHARED)
    return add_operation(
        RawNVVMIntrinsic,
        ScalarTy(datatype.bool_),
        intrinsic=intrinsic,
        operands_=args,
    )
