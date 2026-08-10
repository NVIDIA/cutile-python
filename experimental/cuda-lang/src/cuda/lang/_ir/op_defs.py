# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import Optional
from enum import Enum, auto

import cuda.lang._mlir as mlir
from cuda.lang._enums import MemoryOrder
from cuda.tile._memory_model import MemoryScope
from cuda.tile._ir.ir import MemoryEffect
import cuda.lang._datatype as datatype
from cuda.lang._enums import VectorReduction
from .ir import Operation, Var, attribute, operand
from .type import VectorTy, ScalarTy


@dataclass(eq=False)
class RawNVVMIntrinsic(
    Operation, opcode="nvvm.call_intrinsic", memory_effect=MemoryEffect.STORE
):
    intrinsic: str = attribute()
    operands_: tuple[Var, ...] = operand()


@dataclass(eq=False)
class RawMLIROperation(
    Operation, opcode="mlir.operation", memory_effect=MemoryEffect.STORE
):
    op_name: str = attribute()
    operands_: tuple[Var, ...] = operand()
    mlir_attributes: tuple[tuple[str, mlir.Attribute], ...] = attribute(default=())


@dataclass(eq=False)
class InlinePTX(Operation, opcode="inline_ptx", memory_effect=MemoryEffect.STORE):
    ptx_code: str = attribute()
    read_only_operands: tuple[Var, ...] = operand()
    write_only_operands: tuple[datatype.DType, ...] = attribute()
    read_write_operands: tuple[Var, ...] = operand()

    class RMWMode(Enum):
        READ_ONLY = auto()
        WRITE_ONLY = auto()
        READ_WRITE = auto()


@dataclass(eq=False)
class Fence(Operation, opcode="fence", memory_effect=MemoryEffect.STORE):
    memory_order: MemoryOrder = attribute()
    memory_scope: MemoryScope = attribute()


@dataclass(eq=False)
class ForeignFunction(
    Operation, opcode="foreign_function", memory_effect=MemoryEffect.STORE
):
    function_name: str = attribute()
    operands_: tuple[Var, ...] = operand()


@dataclass(eq=False)
class VectorGetItem(
    Operation, opcode="vector_getitem", memory_effect=MemoryEffect.LOAD
):
    x: Var[VectorTy] = operand()
    index: Var[ScalarTy] = operand()


@dataclass(eq=False)
class VectorReduce(Operation, opcode="vector_reduce"):
    x: Var[VectorTy] = operand()
    kind: VectorReduction = attribute()
    propagate_nan: bool = attribute(default=False)
    reassociate: bool = attribute(default=False)


@dataclass(eq=False)
class BitCast(Operation, opcode="bitcast"):
    x: Var = operand()


@dataclass(eq=False)
class StorePointer(Operation, opcode="store_pointer", memory_effect=MemoryEffect.STORE):
    pointer: Var = operand()
    value: Var = operand()
    alignment: Optional[int] = attribute()


@dataclass(eq=False)
class LoadPointer(Operation, opcode="load_pointer", memory_effect=MemoryEffect.LOAD):
    pointer: Var = operand()
    alignment: Optional[int] = attribute()


@dataclass(eq=False)
class AtomicStore(Operation, opcode="atomic_store", memory_effect=MemoryEffect.STORE):
    pointer: Var = operand()
    value: Var = operand()
    alignment: int = attribute()
    memory_order: MemoryOrder = attribute()
    memory_scope: MemoryScope = attribute()
    mmio: bool = attribute()

    VALID_MEMORY_ORDERS = (
        MemoryOrder.RELAXED,
        MemoryOrder.RELEASE,
    )
    VALID_MMIO_MEMORY_ORDERS = (
        MemoryOrder.RELAXED,
        MemoryOrder.RELEASE,
    )


@dataclass(eq=False)
class AtomicLoad(Operation, opcode="atomic_load", memory_effect=MemoryEffect.LOAD):
    pointer: Var = operand()
    alignment: int = attribute()
    memory_order: MemoryOrder = attribute()
    memory_scope: MemoryScope = attribute()
    mmio: bool = attribute()

    VALID_MEMORY_ORDERS = (
        MemoryOrder.RELAXED,
        MemoryOrder.ACQUIRE,
    )
    VALID_MMIO_MEMORY_ORDERS = (
        MemoryOrder.RELAXED,
        MemoryOrder.ACQUIRE,
    )

    @property
    def has_observable_effect(self) -> bool:
        return self.mmio or self.memory_order is MemoryOrder.ACQUIRE


@dataclass(eq=False)
class ReinterpretPointerAsArray(Operation, opcode="reinterpret_ptr_as_array"):
    pointer: Var = operand()


@dataclass
class TensorMapAsOpaquePtr(Operation, opcode="tensor_map_as_opaque_ptr"):
    tensor_map: Var = operand()
