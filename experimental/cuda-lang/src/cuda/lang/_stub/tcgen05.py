# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Literal

from .._datatype import uint32, uint64
from cuda.lang._execution import stub, function
from .._enums import CTAGroup
from .bits import set_bit32, set_bit64, set_bits32, set_bits64
from .nvvm import P3, P6
from . import nvvm as _nvvm
from .._enums import (
    Tcgen05MMAKind,
    Tcgen05MMACollectorOp,
    Tcgen05LdStShape,
    Tcgen05WaitKind,
)
from cuda.tile import static_assert


@function
def tcgen05_wait(for_operation_kind: Tcgen05WaitKind) -> None:
    static_assert(for_operation_kind in (Tcgen05WaitKind.LOAD, Tcgen05WaitKind.STORE))
    if for_operation_kind == Tcgen05WaitKind.LOAD:
        _nvvm.tcgen05_wait_ld()
    else:
        _nvvm.tcgen05_wait_st()


@function
def tcgen05_relinquish_allocation_permit(group: CTAGroup = CTAGroup.CTA_1) -> None:
    static_assert(group in (CTAGroup.CTA_1, CTAGroup.CTA_2))
    if group == CTAGroup.CTA_1:
        _nvvm.tcgen05_relinq_alloc_permit_cg1()
    else:
        _nvvm.tcgen05_relinq_alloc_permit_cg2()


@function
def tcgen05_shift_down(address, group: CTAGroup = CTAGroup.CTA_1) -> None:
    """
    Asynchronously shift down the rows of the matrix in the Tensor Memory for a warp.

    Args:
        address: pointer in address space 6
        group: cta group 1 or 2
    """
    static_assert(group in (CTAGroup.CTA_1, CTAGroup.CTA_2))
    if group == CTAGroup.CTA_1:
        _nvvm.tcgen05_shift_down_cg1(address)
    else:
        _nvvm.tcgen05_shift_down_cg2(address)


@stub
def tcgen05_alloc(
    addr: P3,
    ncols: int,
    *,
    cta_group: CTAGroup = CTAGroup.CTA_1,
) -> None:
    """Allocate tensor memory columns and write the tensor-memory address to ``addr``."""
    ...


@stub
def tcgen05_dealloc(
    addr: P6,
    ncols: int,
    *,
    cta_group: CTAGroup = CTAGroup.CTA_1,
) -> None:
    """Deallocate tensor memory columns starting at ``addr``."""
    ...


@stub
def tcgen05_commit(
    mbar: P3,
    *,
    multicast_mask: int | None = None,
    cta_group: CTAGroup = CTAGroup.CTA_1,
) -> None:
    """Commit tcgen05 tensor memory operations and arrive at ``mbar``."""
    ...


@stub
def tcgen05_ld(
    shape: Tcgen05LdStShape,
    tmem_addr: P6,
    *,
    count: int = 1,
    pack: bool | None = None,
    offset: int | None = None,
) -> Any:
    """Load registers from tensor memory using a tcgen05 load shape."""
    ...


class _Tcgen05Tf32Type(IntEnum):
    TF32 = 2


class _Tcgen05F16Type(IntEnum):
    F16 = 0
    BF16 = 1


class _Tcgen05F8F6F4Type(IntEnum):
    E4M3 = 0
    E5M2 = 1
    E2M3 = 3
    E3M2 = 4
    E2M1 = 5


class _Tcgen05I8Type(IntEnum):
    U8 = 0
    S8 = 1


class _Tcgen05Mxf4Type(IntEnum):
    E2M1 = 1


class _DType(IntEnum):
    F16 = 0
    F32 = 1
    S32 = 2


class _MaxShift(IntEnum):
    NoShift = 0
    MaxShift8 = 1
    MaxShift16 = 2
    MaxShift32 = 3


class _Mxf8f6f4ScaleFormat(IntEnum):
    UE8M0 = 1


class _Mxf4ScaleFormat(IntEnum):
    UE4M3 = 0
    UE8M0 = 1


class _Mxf4KDimension(IntEnum):
    DenseK64OrSparseK128 = 0
    DenseK96 = 1


@dataclass(frozen=True)
class Tcgen05InstructionDescriptor:
    """
    Instruction descriptor format for .kind::tf32, .kind::f16, .kind::f8f6f4 and .kind::i8
    """

    Tf32Type = _Tcgen05Tf32Type
    F16Type = _Tcgen05F16Type
    F8F6F4Type = _Tcgen05F8F6F4Type
    I8Type = _Tcgen05I8Type
    DType = _DType
    MaxShift = _MaxShift

    sparsity_selector: int = 0
    sparse: bool = False
    saturate: bool = False
    d_type: DType = DType.F16
    a_type: Tf32Type | F16Type | F8F6F4Type | I8Type = F16Type.F16
    b_type: Tf32Type | F16Type | F8F6F4Type | I8Type = F16Type.F16
    negate_a: bool = False
    negate_b: bool = False
    transpose_a: bool = False
    transpose_b: bool = False
    n: int = 0
    m: int = 0
    max_shift: MaxShift = MaxShift.NoShift

    def encode(self) -> int:
        desc = uint32(0x0000_0000)
        desc = set_bits32(desc, self.sparsity_selector, 0, 2)
        desc = set_bit32(desc, 2, self.sparse)
        desc = set_bit32(desc, 3, self.saturate)
        desc = set_bits32(desc, self.d_type, 4, 2)
        desc = set_bits32(desc, self.a_type, 7, 3)
        desc = set_bits32(desc, self.b_type, 10, 3)
        desc = set_bit32(desc, 13, self.negate_a)
        desc = set_bit32(desc, 14, self.negate_b)
        desc = set_bit32(desc, 15, self.transpose_a)
        desc = set_bit32(desc, 16, self.transpose_b)
        desc = set_bits32(desc, self.n >> 3, 17, 6)
        desc = set_bits32(desc, self.m >> 4, 24, 5)
        desc = set_bits32(desc, self.max_shift, 30, 2)
        return desc


@dataclass(frozen=True)
class Tcgen05Mxf8f6f4InstructionDescriptor:
    """Instruction descriptor format for .kind::mxf8f6f4"""

    Type = _Tcgen05F8F6F4Type
    ScaleFormat = _Mxf8f6f4ScaleFormat

    sparse: bool = False
    b_scale_id: Literal[0, 1, 2, 3] = 0
    a_type: Type = Type.E4M3
    b_type: Type = Type.E4M3
    negate_a: bool = False
    negate_b: bool = False
    transpose_a: bool = False
    transpose_b: bool = False
    n: int = 0
    scale_format: ScaleFormat = ScaleFormat.UE8M0
    m: int = 0
    a_scale_id: Literal[0, 1, 2, 3] = 0

    def encode(self) -> int:
        desc = uint32(0x0000_0000)
        desc = set_bit32(desc, 2, self.sparse)
        desc = set_bits32(desc, self.b_scale_id, 4, 2)
        desc = set_bits32(desc, self.a_type, 7, 3)
        desc = set_bits32(desc, self.b_type, 10, 3)
        desc = set_bit32(desc, 13, self.negate_a)
        desc = set_bit32(desc, 14, self.negate_b)
        desc = set_bit32(desc, 15, self.transpose_a)
        desc = set_bit32(desc, 16, self.transpose_b)
        desc = set_bits32(desc, self.n >> 3, 17, 6)
        desc = set_bit32(desc, 23, self.scale_format)
        desc = set_bits32(desc, self.m >> 7, 27, 2)
        desc = set_bits32(desc, self.a_scale_id, 29, 2)
        return desc


@dataclass(frozen=True)
class Tcgen05Mxf4InstructionDescriptor:
    """Instruction descriptor format for .kind::mxf4 and .kind::mxf4nvf4"""

    Type = _Tcgen05Mxf4Type
    ScaleFormat = _Mxf4ScaleFormat
    KDimension = _Mxf4KDimension

    sparse: bool = False
    b_scale_id: Literal[0, 2] = 0
    a_type: Type = Type.E2M1
    b_type: Type = Type.E2M1
    negate_a: bool = False
    negate_b: bool = False
    transpose_a: bool = False
    transpose_b: bool = False
    n: int = 0
    scale_format: ScaleFormat = ScaleFormat.UE8M0
    m: int = 0
    a_scale_id: Literal[0, 2] = 0
    k_dimension: KDimension = KDimension.DenseK64OrSparseK128

    def encode(self) -> int:
        desc = uint32(0x0000_0000)
        desc = set_bit32(desc, 2, self.sparse)
        desc = set_bits32(desc, self.b_scale_id, 4, 2)
        desc = set_bits32(desc, self.a_type, 7, 3)
        desc = set_bits32(desc, self.b_type, 10, 2)
        desc = set_bit32(desc, 13, self.negate_a)
        desc = set_bit32(desc, 14, self.negate_b)
        desc = set_bit32(desc, 15, self.transpose_a)
        desc = set_bit32(desc, 16, self.transpose_b)
        desc = set_bits32(desc, self.n >> 3, 17, 6)
        desc = set_bit32(desc, 23, self.scale_format)
        desc = set_bits32(desc, self.m >> 7, 27, 2)
        desc = set_bits32(desc, self.a_scale_id, 29, 2)
        desc = set_bit32(desc, 31, self.k_dimension)
        return desc


@dataclass(frozen=True)
class Tcgen05SharedMemoryDescriptor:
    class LeadingDimMode(IntEnum):
        ByteOffsetRelative = 0
        ByteAddressAbsolute = 1

    class SwizzleMode(IntEnum):
        NoSwizzling = 0
        Swizzle128B32BAtomic = 1
        Swizzle128B = 2
        Swizzle64B = 4
        Swizzle32B = 6

    matrix_start_address: int
    leading_dim_offset: int
    stride_dim_offset: int
    base_offset: int = 0
    leading_dim_mode: LeadingDimMode = LeadingDimMode.ByteOffsetRelative
    swizzle_mode: SwizzleMode = SwizzleMode.NoSwizzling

    def encode(self) -> int:
        desc = uint64(0x0000_0000_0000_0000)
        desc = set_bits64(desc, (self.matrix_start_address & 0x3FFFF) >> 4, 0, 14)
        desc = set_bits64(desc, (self.leading_dim_offset & 0x3FFFF) >> 4, 16, 14)
        desc = set_bits64(desc, (self.stride_dim_offset & 0x3FFFF) >> 4, 32, 14)
        desc = set_bits64(desc, 0b001, 46, 3)
        desc = set_bits64(desc, self.base_offset, 49, 3)
        desc = set_bit64(desc, 52, self.leading_dim_mode)
        desc = set_bits64(desc, self.swizzle_mode, 61, 3)
        return desc


@stub
def tcgen05_mma(
    kind,
    cta_group,
    matrix_d,
    matrix_a,
    matrix_b,
    idesc,
    enable_input_d,
    scale_input_d=None,
    disable_output_lane=None,
    collector_op=Tcgen05MMACollectorOp.DISCARD,
    a_shift=None,
) -> None:
    """
    Args:
        kind (Tcgen05MMAKind):
        cta_group (CTAGroup):
        matrix_d (P6):
        matrix_a (P6 | int64):
        matrix_b (int64):
        idesc (int32):
        enable_input_d (bool):
        scale_input_d (int32 | None):
        disable_output_lane (vector | None):
        collector_op (Tcgen05MMACollectorOp):
        a_shift (bool | None):
    """


__all__ = (
    "CTAGroup",
    "Tcgen05MMAKind",
    "Tcgen05WaitKind",
    "Tcgen05MMACollectorOp",
    "Tcgen05LdStShape",
    "Tcgen05InstructionDescriptor",
    "Tcgen05Mxf8f6f4InstructionDescriptor",
    "Tcgen05Mxf4InstructionDescriptor",
    "Tcgen05SharedMemoryDescriptor",
    "tcgen05_alloc",
    "tcgen05_dealloc",
    "tcgen05_commit",
    "tcgen05_ld",
    "tcgen05_mma",
    "tcgen05_wait",
    "tcgen05_shift_down",
    "tcgen05_relinquish_allocation_permit",
)
