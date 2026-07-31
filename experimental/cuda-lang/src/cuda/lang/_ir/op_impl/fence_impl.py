# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from cuda.lang._enums import MemoryOrder, MemoryScope
from cuda.lang._exception import TypeCheckingError
from cuda.lang._ir.op_defs import InlinePTX
from cuda.lang._stub import fence as fence_stub
from cuda.tile._ir.ir import add_operation_variadic
from cuda.tile._ir.op_impl import ImplRegistry, require_constant_enum


_registry = ImplRegistry()
impl = _registry.impl


def fence_impl_registry() -> ImplRegistry:
    return _registry


@impl(fence_stub.fence)
def fence_impl(order, scope) -> None:
    order = require_constant_enum(order, MemoryOrder)
    scope = require_constant_enum(scope, MemoryScope)

    valid_orders = (
        MemoryOrder.ACQUIRE,
        MemoryOrder.RELEASE,
        MemoryOrder.ACQ_REL,
    )
    if order not in valid_orders:
        formatted = ", ".join(str(value) for value in valid_orders)
        raise TypeCheckingError(
            f"Invalid fence memory order {order}, expected one of {formatted}"
        )

    scope_suffixes = {
        MemoryScope.BLOCK: "cta",
        MemoryScope.CLUSTER: "cluster",
        MemoryScope.DEVICE: "gpu",
        MemoryScope.SYS: "sys",
    }
    if scope not in scope_suffixes:
        formatted = ", ".join(str(value) for value in scope_suffixes)
        raise TypeCheckingError(
            f"Invalid fence memory scope {scope}, expected one of {formatted}"
        )

    add_operation_variadic(
        InlinePTX,
        (),
        ptx_code=f"fence.{order.value}.{scope_suffixes[scope]};",
        read_only_operands=(),
        write_only_operands=(),
        read_write_operands=(),
    )


__all__ = ("fence_impl_registry",)
