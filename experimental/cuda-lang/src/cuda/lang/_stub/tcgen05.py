# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Literal

from .._datatype import uint32
from cuda.lang._execution import stub, function
from .bits import set_bit32, set_bits32
from .nvvm import P3, P6
from . import nvvm as _nvvm
from .._enums import (
    CTAGroup,
    SwizzleMode,
    Tcgen05MMAKind,
    Tcgen05MMABlockScaleKind,
    Tcgen05MMAScaleVectorSize,
    Tcgen05MMACollectorBBuffer,
    Tcgen05MMACollectorOp,
    Tcgen05LoadStoreShape,
    Tcgen05CopyMulticast,
    Tcgen05CopyShape,
    Tcgen05CopySourceFormat,
)
import cuda.lang as cl


@function()
def tcgen05_wait_load() -> None:
    _nvvm.tcgen05_wait_ld()


@function()
def tcgen05_wait_store() -> None:
    _nvvm.tcgen05_wait_st()


@function()
def tcgen05_fence_before_thread_sync() -> None:
    """
    Orders all prior async tcgen05 operations with respect to the subsequent
    tcgen05 and execution ordering operations
    """
    _nvvm.tcgen05_fence_before_thread_sync()


@function()
def tcgen05_fence_after_thread_sync() -> None:
    """
    Orders all subsequent async tcgen05 operations with respect to the prior
    tcgen05 and execution ordering operations
    """
    _nvvm.tcgen05_fence_after_thread_sync()


@function()
def tcgen05_relinquish_allocation_permit(cta_group: CTAGroup = CTAGroup.CTA_1) -> None:
    cl.static_assert(cta_group in (CTAGroup.CTA_1, CTAGroup.CTA_2))
    if cta_group == CTAGroup.CTA_1:
        _nvvm.tcgen05_relinq_alloc_permit_cg1()
    else:
        _nvvm.tcgen05_relinq_alloc_permit_cg2()


@function()
def tcgen05_shift_down(address, cta_group: CTAGroup = CTAGroup.CTA_1) -> None:
    """
    Asynchronously shift down the rows of the matrix in the Tensor Memory for a warp.

    Args:
        address: pointer in tensor memory
        cta_group: cta group 1 or 2
    """
    cl.static_assert(cta_group in (CTAGroup.CTA_1, CTAGroup.CTA_2))
    if cta_group == CTAGroup.CTA_1:
        _nvvm.tcgen05_shift_down_cg1(address)
    else:
        _nvvm.tcgen05_shift_down_cg2(address)


@stub
def tcgen05_allocate(
    address: P3,
    number_of_columns: int,
    *,
    cta_group: CTAGroup = CTAGroup.CTA_1,
) -> None:
    """Allocate tensor memory columns and write their address to ``address``."""
    ...


@stub
def tcgen05_deallocate(
    address: P6,
    number_of_columns: int,
    *,
    cta_group: CTAGroup = CTAGroup.CTA_1,
) -> None:
    """Deallocate tensor memory columns starting at ``address``."""
    ...


@stub
def tcgen05_tmem_offset(
    pointer: P6,
    *,
    lane_offset: int = 0,
    column_offset: int = 0,
) -> P6:
    """Offset a tensor memory pointer by lane and column coordinates.

    Args:
        pointer: Pointer in tensor memory.
        lane_offset (int): Number of tensor memory lanes to add.
        column_offset (int): Number of tensor memory columns to add.

    Returns:
        A tensor memory pointer with the same pointee type as ``pointer``.
    """
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
def tcgen05_load(
    shape: Tcgen05LoadStoreShape,
    tensor_memory_address: P6,
    *,
    count: int = 1,
    pack: bool | None = None,
    offset: int | None = None,
) -> Any:
    """Load registers from tensor memory using a tcgen05 load shape."""
    ...


@stub
def tcgen05_copy(
    address,
    shared_memory_descriptor,
    *,
    shape: Tcgen05CopyShape,
    cta_group: CTAGroup = CTAGroup.CTA_1,
    multicast: Tcgen05CopyMulticast | None = None,
    source_format: Tcgen05CopySourceFormat | None = None,
):
    """
    Initiates an asynchronous copy operation from shared memory to the
    location specified by ``address``.

    Args:
        address: Pointer in tensor memory allocated by tcgen05_allocate.
        shared_memory_descriptor: Shared memory descriptor encoded
            as a 64-bit integer.
        cta_group:
        shape:
        multicast:
        source_format:
    """


@stub
def tcgen05_store(
    shape: Tcgen05LoadStoreShape,
    tensor_memory_address,
    value,
    *,
    unpack: bool = False,
    offset: int | None = None,
):
    """
    Store registers to tensor memory using a tcgen05 store shape.

    Args:
        shape:
        tensor_memory_address: Pointer in tensor memory (address space 6).
        value: 32-bit signless integer or vector of 32-bit signless integer
            values of length 2/4/8/16/32/64/128
        unpack: unpack a 32-bit element in the register into two 16-bit
            elements and store them in adjacent columns.
        offset: When shape 16x32bx2 is used, base address of the first access is
            specified by ``tensor_memory_address`` and the base address of the second
            access is specified by ``tensor_memory_address + offset``, where offset is
            an immediate argument.
    """


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
    """Describe the shared-memory layout of a matrix operand for tcgen05.

    The encoded descriptor is passed as ``matrix_a`` or ``matrix_b`` to
    :func:`tcgen05_mma`. Its address, strides, and swizzle mode must describe the
    same physical layout that was used to populate shared memory.

    How tcgen05 organizes a swizzled matrix
    ---------------------------------------

    The PTX specification describes tcgen05 layouts using 16-byte *cells*. A
    *swizzle layout atom* is the smallest rectangular group of cells that repeats
    to form a matrix in shared memory. The atom shape is part of the tcgen05
    matrix layout; it is not determined by the TMA tile shape.

    The atom has a *leading dimension* and a *stride dimension*:

    * For a K-major matrix, K is the leading dimension. A row of the atom runs
      along K, and the stride dimension advances through M for matrix A or N for
      matrix B.
    * For an MN-major matrix, M or N is the leading dimension and K is the stride
      dimension.

    The instruction descriptor's transpose bits select K-major or MN-major. The
    following atom shapes come from Table 58 of the PTX ISA. Every shape is
    written as ``M-or-N extent x K extent``; both extents count 16-byte cells,
    before converting them to the matrix element type. Major-ness specifies
    which of these two dimensions is the leading dimension, not the order in
    which the shape is printed.

    .. list-table:: tcgen05 swizzle atom shapes
       :header-rows: 1

       * - Swizzle mode
         - K-major atom (M/N x K)
         - MN-major atom (M/N x K)
         - Atom bytes
       * - 128B, 16B atomicity
         - ``8 x 8``
         - ``8 x 8``
         - 1024
       * - 128B, 32B atomicity
         - unsupported
         - ``8 x 4``
         - 512
       * - 64B, 16B atomicity
         - ``8 x 4``
         - ``4 x 8``
         - 512
       * - 32B, 16B atomicity
         - ``8 x 2``
         - ``2 x 8``
         - 256
       * - no swizzle
         - ``8 x 1``
         - ``1 x 8``
         - 128

    In the table, the K-major Swizzle 128B atom shape ``8 x 8`` means eight matrix
    rows with eight 16-byte cells in each row. Converting that shape to bytes
    gives::

        bytes per row = 8 cells * 16 bytes = 128 bytes
        bytes per atom = 8 rows * 128 bytes = 1024 bytes

    The swizzle permutes the cells within this 1024-byte region; it does not
    change the region's total size.

    The same calculation applies to the other K-major modes. A 64B atom is eight
    rows by four cells, or ``8 * 64 = 512`` bytes. A 32B atom is eight rows by two
    cells, or ``8 * 32 = 256`` bytes. Without swizzling, the atom is eight rows by
    one cell, or ``8 * 16 = 128`` bytes.

    For element types smaller than 16 bytes, expand the atom along its leading
    dimension. One 16-byte cell holds eight BF16 values, for example. The K-major
    BF16 atom shapes for 128B, 64B, and 32B swizzling are therefore respectively
    ``8 x 64``, ``8 x 32``, and ``8 x 16`` elements, where the first number is
    the number of M or N rows and the second number is the K extent.

    In a K-major descriptor, ``stride_dimension_byte_offset`` is the byte
    distance from the first group of eight rows to the next group of eight rows.
    When atoms are stored consecutively, this is the atom size: 1024 bytes for
    128B swizzling, 512 bytes for 64B swizzling, and 256 bytes for 32B
    swizzling.

    TMA swizzling and tcgen05 layout are separate requirements. A TMA tensor map
    controls how a copy permutes addresses in shared memory. A tcgen05 shared-
    memory descriptor controls how MMA interprets those addresses as a matrix.
    Using the same :class:`SwizzleMode` in both descriptors is necessary, but it
    does not by itself make their layouts compatible.

    For example, in a K-major tensor map, descriptor axis zero is K. TMA permits
    a 128B-swizzled BF16 tile with a K extent of 32, meaning
    ``tile_shape[0] == 32``. One tile row is then only ``32 * 2 = 64`` bytes,
    which is within TMA's 128-byte limit. That tile is not a complete K-major
    tcgen05 atom row. The table above requires 64 BF16 values, or 128 bytes, per
    row. To feed tcgen05 directly, the TMA tile axes must therefore arrange eight
    such rows into each 1024-byte atom.

    For the authoritative definitions and diagrams, see:

    * `PTX ISA: tcgen05 shared-memory layout and swizzling
      <https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-shared-memory-layout-and-swizzling>`_
      for atom shapes, major-ness, element-size conversion, and examples.
    * `PTX ISA: tensor swizzling modes
      <https://docs.nvidia.com/cuda/parallel-thread-execution/#swizzling-modes>`_
      for the address permutations produced by TMA, supported atomicities, and
      repeating-pattern alignment.

    Constructing a 128B K-major BF16 operand with TMA
    --------------------------------------------------

    Suppose a BF16 matrix is K-major: values adjacent along K are adjacent in
    global memory. The table above requires each shared-memory atom to contain
    eight rows and 64 K values per row.

    To express this layout, split K into a position within a 64-element segment
    and a segment number::

        k = 64 * k_outer + k_inner

    Call these coordinates ``K_inner`` and ``K_outer``, respectively. Reshape the
    global matrix into a three-dimensional view whose axes retain the natural
    ``(M, K_outer, K_inner)`` order::

        shape   = (M, K // 64, 64)
        strides = (K, 64, 1)

    This view does not move or copy data. Its strides preserve the original
    address calculation: element ``(m, k_outer, k_inner)`` is at offset
    ``m * K + 64 * k_outer + k_inner``. Set ``order=(2, 0, 1)`` to select view
    axes 2, 0, and 1, producing the permuted descriptor order
    ``(K_inner, M, K_outer)``::

        a_map = cl.tensor_map_tiled(
            a_view,
            (64, BLOCK_M, BLOCK_K // 64),
            order=(2, 0, 1),
            swizzle=cl.SwizzleMode.SWIZZLE_128B,
        )

    Before applying the swizzle permutation, TMA packs the tile in descriptor-
    axis order. The logical shared-memory byte offset is therefore::

        2 * (k_inner + 64 * (m + BLOCK_M * k_outer))

    The corresponding nesting is ``[K_outer][M][K_inner]``: all M rows for one
    64-element K segment are contiguous. Because a 128B atom covers eight rows,
    the shared-memory descriptor uses 1024 bytes as the distance between
    eight-row groups::

        a_desc = cl.Tcgen05SharedMemoryDescriptor(
            matrix_start_address=a_smem_addr,
            leading_dimension_byte_offset=16,
            stride_dimension_byte_offset=8 * 128,
            swizzle_mode=cl.SwizzleMode.SWIZZLE_128B,
        ).encode()

    To see why the split and permutation are needed, first consider using the
    original matrix, whose shape and major-to-minor nesting are ``(M, K)`` and
    ``[M][K]``. A logical ``(BLOCK_M, BLOCK_K)`` matrix tile would be described
    as follows::

        a_map = cl.tensor_map_tiled(
            a,
            (BLOCK_K, BLOCK_M),  # Descriptor-axis order is (K, M).
            order=(1, 0),
            swizzle=cl.SwizzleMode.SWIZZLE_128B,
        )

    For BF16, ``BLOCK_K <= 64`` satisfies TMA's Swizzle 128B width constraint;
    other tensor-map restrictions still apply. Tcgen05 imposes the stricter
    layout requirement that an atom row contain exactly 64 BF16 values:

    * ``BLOCK_K < 64`` may be valid for TMA but does not fill a tcgen05 atom row.
    * ``BLOCK_K == 64`` fills one 128-byte tcgen05 atom row.
    * ``BLOCK_K > 64`` is not valid as one TMA tile row with Swizzle 128B.

    If a wider K tile is copied as multiple 64-element segments while retaining
    the natural ``[M][K]`` nesting, the segments remain grouped by matrix row.
    For a tile with two K segments, shared memory would contain::

        Byte range       Matrix data
        0    .. 127      row 0, K segment 0    [atom row 0, 128B]
        128  .. 255      row 0, K segment 1    [atom row 1, 128B]
        256  .. 383      row 1, K segment 0    [atom row 2, 128B]
        384  .. 511      row 1, K segment 1    [atom row 3, 128B]

    The K values from matrix row 0 are therefore split across atom rows 0 and 1.
    A K-major Swizzle 128B atom has a different interpretation: its eight atom
    rows represent eight consecutive M rows at the same K segment.

    Splitting the view into ``(M, K_outer, K_inner)`` and applying
    ``order=(2, 0, 1)`` changes the major-to-minor nesting to
    ``[K_outer][M][K_inner]``. Shared memory then contains::

        Byte range       Matrix data
        0    .. 127      row 0, K segment 0    [atom row 0, 128B]
        128  .. 255      row 1, K segment 0    [atom row 1, 128B]
        ...
        896  .. 1023     row 7, K segment 0    [atom row 7, 128B]
        1024 .. 1151     row 8, K segment 0    [next atom, row 0, 128B]

    Each matrix row's 64 ``K_inner`` values remain contiguous within one atom
    row, while eight consecutive M rows at the same ``K_outer`` fill one complete
    atom. This is the K-major Swizzle 128B layout consumed by tcgen05.

    Compatibility invariant
    -----------------------

    A TMA-produced shared-memory layout can be consumed by tcgen05 when every
    region that tcgen05 interprets as one swizzle atom contains exactly the
    logical matrix coordinates specified by the PTX atom shape. To preserve this
    invariant for any supported swizzle mode, major-ness, and element type:

    * Use the same :class:`SwizzleMode` for the TMA tensor map and the
      :class:`Tcgen05SharedMemoryDescriptor`.
    * Convert the PTX atom shape from 16-byte cells to the operand's element
      extents. The resulting layout must contain complete atom rows.
    * Factor the logical tile into an outer grid of complete atoms. For a K-major
      ``BLOCK_M x BLOCK_K`` tile, its conceptual major-to-minor shape is::

          [BLOCK_K // atom_row_size][BLOCK_M // 8][8][atom_row_size]

      Here, ``atom_row_size`` is the number of operand elements in one complete
      atom row (for example, 64 BF16 elements for Swizzle 128B). The innermost
      ``[8][atom_row_size]`` region is one atom. ``BLOCK_M // 8`` enumerates
      eight-row groups within a K segment, and ``BLOCK_K // atom_row_size``
      enumerates the K segments. The tensor-map descriptor axes, listed
      minor-to-major, must produce this nesting. For MN-major, use the
      corresponding MN-major atom extents from the table above, with M or N as
      the fastest-moving dimension.
    * Set the shared-memory descriptor's start address, base offset, and strides
      to the same atom boundaries populated by TMA.

    When these conditions hold, TMA's physical address permutation and tcgen05's
    interpretation of that permutation describe the same shared-memory layout.
    """

    class LeadingDimensionMode(IntEnum):
        ByteOffsetRelative = 0
        ByteAddressAbsolute = 1

    matrix_start_address: int
    leading_dimension_byte_offset: int
    stride_dimension_byte_offset: int
    base_offset: int = 0
    leading_dimension_mode: LeadingDimensionMode = (
        LeadingDimensionMode.ByteOffsetRelative
    )
    swizzle_mode: SwizzleMode = SwizzleMode.SWIZZLE_NONE

    @stub
    def encode(self) -> int: ...


@stub
def tcgen05_mma(
    kind,
    matrix_d,
    matrix_a,
    matrix_b,
    instruction_descriptor,
    *,
    accumulate,
    cta_group=CTAGroup.CTA_1,
    sparse_metadata=None,
    scale_input_d=None,
    disable_output_lane=None,
    collector_op=Tcgen05MMACollectorOp.DISCARD,
    a_shift=False,
) -> None:
    """
    Perform the 5th generation of matrix multiply and accumulate operation.

    Args:
        kind (Tcgen05MMAKind): Data type the operation should be performed in.
        matrix_d (P6): Pointer in tensor memory to the destination and optional
            accumulator matrix D.
        matrix_a (P6 | int64): Matrix A encoded as either a 64-bit shared-memory
            descriptor or a pointer in tensor memory.
        matrix_b (int64): Matrix B encoded as a 64-bit shared-memory descriptor.
        instruction_descriptor (int32 | uint32): Encoded instruction descriptor.
        accumulate (bool): Whether input matrix D is included in the result.
        cta_group (CTAGroup): Controlls whether the operation takes place in
            one block or a pair of blocks.
        sparse_metadata (P6 | None): Optional pointer in tensor memory containing
            sparsity metadata for packed sparse matrix A. ``None`` selects dense
            MMA; presence selects sparse MMA and must be compile-time known.
        scale_input_d (int | None): Optional compile-time exponent in
            ``[0, 15]`` that scales input D by ``2**-scale_input_d``.
            Supported only for ``F16`` and ``TF32`` kinds.
        disable_output_lane (vector | None): Optional vector mask selecting
            tensor-memory lanes that must not be updated.
        collector_op (Tcgen05MMACollectorOp): Collector-buffer operation for
            matrix A.
        a_shift (bool): Shifts the rows of the A matrix down by one row and
            can only be applied if A is in tensor memory
    """


@stub
def tcgen05_mma_block_scale(
    kind,
    matrix_d,
    matrix_a,
    matrix_b,
    instruction_descriptor,
    scale_a,
    scale_b,
    *,
    accumulate,
    sparse_metadata=None,
    cta_group=CTAGroup.CTA_1,
    scale_vector_size=Tcgen05MMAScaleVectorSize.DEFAULT,
    collector_op=Tcgen05MMACollectorOp.DISCARD,
) -> None:
    """
    Performs block scaled MMA operation on 5th-generation tensor cores.

    Args:
        kind (Tcgen05MMABlockScaleKind): Data type the operation should be
            performed in.
        matrix_d (P6): Pointer in tensor memory to the destination and optional
            accumulator matrix D.
        matrix_a (P6 | int64): Matrix A encoded as either a 64-bit shared-memory
            descriptor or a pointer in tensor memory.
        matrix_b (int64): Matrix B encoded as a 64-bit shared-memory descriptor.
        instruction_descriptor (int32 | uint32): Encoded instruction descriptor.
        scale_a (P6): Pointer in tensor memory to matrix A scale factors.
        scale_b (P6): Pointer in tensor memory to matrix B scale factors.
        accumulate (bool): Whether input matrix D is included in the result.
        sparse_metadata (P6 | None): Optional pointer in tensor memory containing
            sparsity metadata for packed sparse matrix A. ``None`` selects dense
            MMA; presence selects sparse MMA and must be compile-time known.
        cta_group (CTAGroup): Controlls whether the operation takes place in
            one block or a pair of blocks.
        scale_vector_size (Tcgen05MMAScaleVectorSize): Scale-vector layout.
        collector_op (Tcgen05MMACollectorOp): Collector-buffer operation for
            matrix A.
    """


@stub
def tcgen05_mma_weight_stationary(
    kind,
    matrix_d,
    matrix_a,
    matrix_b,
    instruction_descriptor,
    *,
    accumulate,
    sparse_metadata=None,
    zero_column_mask=None,
    collector_op=Tcgen05MMACollectorOp.DISCARD,
    collector_b_buffer=Tcgen05MMACollectorBBuffer.BUFFER_0,
) -> None:
    """
    Perform the 5th generation of weight stationary convolution matrix
    multiply and accumulate operation.

    Args:
        kind (Tcgen05MMAKind): Data type the operation should be performed in.
        matrix_d (P6): Pointer in tensor memory to the destination and optional
            accumulator matrix D.
        matrix_a (P6 | int64): Matrix A encoded as either a 64-bit shared-memory
            descriptor or a pointer in tensor memory.
        matrix_b (int64): Matrix B encoded as a 64-bit shared-memory descriptor.
        instruction_descriptor (int32 | uint32): Encoded instruction descriptor.
        accumulate (bool): Whether input matrix D is included in the result.
        sparse_metadata (P6 | None): Optional pointer in tensor memory containing
            sparsity metadata for packed sparse matrix A. ``None`` selects dense
            MMA; presence selects sparse MMA and must be compile-time known.
        zero_column_mask (int64 | None): Optional integral scalar containing a 64-bit
            zero-column mask descriptor for matrix B.
        collector_op (Tcgen05MMACollectorOp): Collector-buffer operation for
            matrix A.
        collector_b_buffer (Tcgen05MMACollectorBBuffer):
    """


__all__ = (
    "CTAGroup",
    "Tcgen05MMAKind",
    "Tcgen05MMABlockScaleKind",
    "Tcgen05MMAScaleVectorSize",
    "Tcgen05MMACollectorBBuffer",
    "Tcgen05MMACollectorOp",
    "Tcgen05LoadStoreShape",
    "Tcgen05CopyMulticast",
    "Tcgen05CopyShape",
    "Tcgen05CopySourceFormat",
    "Tcgen05InstructionDescriptor",
    "Tcgen05Mxf8f6f4InstructionDescriptor",
    "Tcgen05Mxf4InstructionDescriptor",
    "Tcgen05SharedMemoryDescriptor",
    "tcgen05_allocate",
    "tcgen05_deallocate",
    "tcgen05_tmem_offset",
    "tcgen05_commit",
    "tcgen05_load",
    "tcgen05_copy",
    "tcgen05_store",
    "tcgen05_mma",
    "tcgen05_mma_block_scale",
    "tcgen05_mma_weight_stationary",
    "tcgen05_wait_load",
    "tcgen05_wait_store",
    "tcgen05_fence_before_thread_sync",
    "tcgen05_fence_after_thread_sync",
    "tcgen05_shift_down",
    "tcgen05_relinquish_allocation_permit",
)
