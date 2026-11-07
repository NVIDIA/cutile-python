# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from enum import Enum, auto
from cuda.tile import MemoryScope, RoundingMode
from cuda.tile._memory_model import MemorySpace
from cuda.tile._cext import (
    SwizzleMode,
    TensorMapInterleave,
    TensorMapFloatOOBFill,
    TensorMapL2Promotion,
    TMALoadMode,
    TMAStoreMode,
)


class AxisOrder(Enum):
    """Ordering of array axes in a tensor-map descriptor."""

    KEEP = "keep"
    """Keep the array axes in their original order."""

    REVERSE = "reverse"
    """Reverse the order of the array axes."""


class MemoryOrder(Enum):
    """Memory ordering semantics of a memory operation."""

    WEAK = "weak"
    RELAXED = "relaxed"
    ACQUIRE = "acquire"
    RELEASE = "release"
    ACQ_REL = "acq_rel"
    SEQ_CST = "seq_cst"


class AtomicOp(Enum):
    """Operation for :func:`cuda.lang.atomic_rmw`."""

    ADD = "add"
    SUB = "sub"
    AND = "and"
    OR = "or"
    XOR = "xor"
    MIN = "min"
    MAX = "max"
    INC = "inc"
    DEC = "dec"
    EXCH = "exch"
    CAS = "cas"


class SaturationMode(Enum):
    """Saturation mode for floating-point and integer operations."""

    NONE = "none"
    """Do not saturate the result."""

    SATFINITE = "satfinite"
    """Limit the result to the largest finite value of its type."""

    SAT = "sat"
    """Limit a floating-point result to ``[0.0, 1.0]``."""


class MbarrierLayout(Enum):
    """Layout of an mbarrier object.

    Different layouts are capable of holding different ranges of values.
    Refer to the PTX documentation:
    https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#mbarrier-counts
    """
    V0 = 0
    V1 = 1


class MbarrierScope(Enum):
    """Scope of the threads that observe an mbarrier operation."""

    BLOCK = "cta"
    CLUSTER = "cluster"


class Tcgen05MMAKind(Enum):
    """MMA operation families for :func:`~cuda.lang.tcgen05_mma`.

    One MMA instruction computes an MxN output tile. Each output element sums
    K products from A and B. For a fixed MxN tile, a reduction of length
    ``K_tile`` requires ``K_tile / K`` instructions when ``K_tile`` is a
    multiple of K. Each instruction covers the next K-wide slice of the
    reduction dimension.

    The `PTX MMA shape table
    <https://docs.nvidia.com/cuda/archive/13.4.2/parallel-thread-execution/index.html#tcgen05-kind-shapes>`_
    (ISA 9.4, CUDA 13.4.2, Table 48) lists the supported MxNxK shapes. For
    :func:`~cuda.lang.tcgen05_mma`, use rows marked "No .ws" in the
    "Has .ws" column. The ``.ws`` rows describe
    :func:`~cuda.lang.tcgen05_mma_weight_stationary`.

    The instruction descriptor specifies M, N, and the exact types of A, B,
    and D. The applicable table entry matches those types, ``kind``,
    ``cta_group``, and the dense/sparse variant selected by ``sparse_metadata``.
    For E2M1 operands, the descriptor dtype is :data:`~cuda.lang.float4_e2m1fn_x2`.

    When an entry offers multiple K values, the descriptor's `K Dimension field
    <https://docs.nvidia.com/cuda/archive/13.4.2/parallel-thread-execution/index.html#tcgen05-instruction-descriptor>`_
    selects the value. :class:`~cuda.lang.Tcgen05InstructionDescriptor`
    currently sets this field to zero. The table's `target notes
    <https://docs.nvidia.com/cuda/archive/13.4.2/parallel-thread-execution/index.html#tcgen05-matrix-shape-target-isa-note>`_
    identify which architectures support each K choice.
    """

    F16 = 0
    TF32 = 1
    F8F6F4 = 2
    I8 = 3


class Tcgen05MMABlockScaleKind(Enum):
    """Arithmetic families for :func:`~cuda.lang.tcgen05_mma_block_scale`.

    The members select the PTX ``.kind::mxf8f6f4``, ``.kind::mxf4``, and
    ``.kind::mxf4nvf4`` qualifiers respectively:

    * ``MXF8F6F4`` supports E4M3, E5M2, E2M3, E3M2, and E2M1 operands
      with UE8M0 scales. Its descriptor is
      :class:`~cuda.lang.Tcgen05Mxf8f6f4InstructionDescriptor`.
    * ``MXF4`` uses E2M1 operands with UE8M0 scales.
    * ``MXF4NVF4`` uses E2M1 operands with the scale format selected by the
      descriptor. :class:`~cuda.lang.Tcgen05Mxf4InstructionDescriptor`, used
      by both F4 kinds, exposes UE8M0 and UE4M3 scale formats.

    E2M1 operands use :data:`~cuda.lang.float4_e2m1fn_x2` in the instruction descriptor.

    :class:`~cuda.lang.Tcgen05MMAScaleVectorSize` documents the accepted
    scaling-block choices. The `PTX block-scaling tables
    <https://docs.nvidia.com/cuda/archive/13.4.2/parallel-thread-execution/index.html#tcgen05-mma-scale-valid-vec-size>`_
    give the complete type combinations and target requirements; the
    `MMA shape table
    <https://docs.nvidia.com/cuda/archive/13.4.2/parallel-thread-execution/index.html#tcgen05-kind-shapes>`_
    gives their MxNxK instruction shapes.
    """

    MXF8F6F4 = 0
    MXF4 = 1
    MXF4NVF4 = 2


class Tcgen05MMAScaleVectorSize(Enum):
    """Scaling granularity for :func:`~cuda.lang.tcgen05_mma_block_scale`.

    ``BLOCK_16`` and ``BLOCK_32`` select the PTX ``.block16`` and
    ``.block32`` layouts. Scale factors apply to chunks of each A row and
    each B column. The `PTX scaling-layout tables
    <https://docs.nvidia.com/cuda/archive/13.4.2/parallel-thread-execution/index.html#tcgen05-mma-scale-valid-vec-size>`_
    specify the scale-matrix dimensions for each kind, K, and sparse/dense
    variant, along with type and target restrictions.

    The API accepts these combinations:

    .. list-table::
       :header-rows: 1

       * - Block-scale kind
         - Accepted values
       * - ``MXF8F6F4``
         - ``DEFAULT``, ``BLOCK_32``
       * - ``MXF4``
         - ``DEFAULT``, ``BLOCK_32``
       * - ``MXF4NVF4``
         - ``BLOCK_16``, ``BLOCK_32``

    ``DEFAULT`` selects the same layout as ``BLOCK_32`` for ``MXF8F6F4``
    and ``MXF4``. ``MXF4NVF4`` requires an explicit block size.
    Scale-factor placement in TMEM follows the `PTX block-scaling layouts
    <https://docs.nvidia.com/cuda/archive/13.4.2/parallel-thread-execution/index.html#tcgen05-block-scaling>`_.
    """

    DEFAULT = 0
    BLOCK_16 = 1
    BLOCK_32 = 2


class Tcgen05MMACollectorBBuffer(Enum):
    """Matrix B collector selection for :func:`~cuda.lang.tcgen05_mma_weight_stationary`.

    ``BUFFER_0`` through ``BUFFER_3`` select PTX collectors ``b0`` through
    ``b3``. :class:`~cuda.lang.Tcgen05MMACollectorOp` controls how the selected
    collector is filled or reused. A fill and its subsequent uses refer to
    the same buffer. See the `PTX weight-stationary collector rules
    <https://docs.nvidia.com/cuda/archive/13.4.2/parallel-thread-execution/index.html#tcgen05-mma-instructions-mma-ws>`_.
    """

    BUFFER_0 = 0
    BUFFER_1 = 1
    BUFFER_2 = 2
    BUFFER_3 = 3


class Tcgen05MMACollectorOp(Enum):
    """Operand reuse through a Tensor Core collector buffer.

    ``collector_op`` applies to matrix A in :func:`~cuda.lang.tcgen05_mma`
    and :func:`~cuda.lang.tcgen05_mma_block_scale`, and to matrix B in
    :func:`~cuda.lang.tcgen05_mma_weight_stationary`.

    * ``FILL`` places the operand read from memory into the collector for reuse.
    * ``USE`` allows the operand to be read from an existing collector fill.
    * ``LASTUSE`` allows the operand to be read from the collector and its
      contents to be discarded after that read.
    * ``DISCARD`` allows the collector contents to be discarded.

    ``USE`` and ``LASTUSE`` require an earlier fill whose contents remain
    valid for the operation. The `PTX MMA collector rules
    <https://docs.nvidia.com/cuda/archive/13.4.2/parallel-thread-execution/index.html#tcgen05-mma-instructions-mma>`_
    describe operand reuse and its restrictions. Weight-stationary MMA uses
    the buffer selected by :class:`~cuda.lang.Tcgen05MMACollectorBBuffer`.
    """

    DISCARD = 0
    LASTUSE = 1
    FILL = 2
    USE = 3


class Tcgen05LoadStoreShape(Enum):
    """Load/store shapes supported by tcgen05 tensor memory operations.

    Each enum value spells the corresponding PTX shape: for example,
    ``SHAPE_16x128b`` selects ``.16x128b``. The first dimension counts TMEM
    lanes (rows); the width is measured in bits (``b``).
    ``SHAPE_16x32bx2`` selects two ``.16x32b`` accesses separated by ``offset``.

    .. list-table:: PTX ISA 9.4 (CUDA 13.4.2) lane/register layouts
       :header-rows: 1

       * - Enum member
         - PTX shape and diagram
         - 32-bit registers per thread at ``.x1``
       * - ``SHAPE_32x32b``
         - `.32x32b (Figure 186)
           <https://docs.nvidia.com/cuda/archive/13.4.2/parallel-thread-execution/index.html#tcgen05-mma-fragment-3232b>`_
         - 1
       * - ``SHAPE_16x64b``
         - `.16x64b (Figure 187)
           <https://docs.nvidia.com/cuda/archive/13.4.2/parallel-thread-execution/index.html#tcgen05-mma-fragment-1664b>`_
         - 1
       * - ``SHAPE_16x128b``
         - `.16x128b (Figure 188)
           <https://docs.nvidia.com/cuda/archive/13.4.2/parallel-thread-execution/index.html#tcgen05-mma-fragment-16128b>`_
         - 2
       * - ``SHAPE_16x256b``
         - `.16x256b (Figure 189)
           <https://docs.nvidia.com/cuda/archive/13.4.2/parallel-thread-execution/index.html#tcgen05-mma-fragment-16256b>`_
         - 4
       * - ``SHAPE_16x32bx2``
         - `.16x32bx2 (Figure 190)
           <https://docs.nvidia.com/cuda/archive/13.4.2/parallel-thread-execution/index.html#tcgen05-mma-fragment-1632b2>`_
         - 1

    The diagrams specify the TMEM coordinates accessed by each thread,
    relative to the supplied TMEM base. :func:`~cuda.lang.tcgen05_tmem_offset`
    selects that base's lane and column offsets. In the diagrams, ``Tn``
    identifies thread ``n`` within the warp, and ``rj`` identifies its j-th
    32-bit register.

    For :func:`~cuda.lang.tcgen05_load` with a 32-bit ``dtype``, ``rj`` becomes
    returned vector element ``j`` (or the scalar for ``r0``). Other dtypes
    reinterpret the registers in order, with low bits first: a 16-bit dtype
    maps ``rj`` to elements ``2*j`` and ``2*j+1``; a 64-bit dtype combines
    ``r(2*j)`` and ``r(2*j+1)`` into element ``j``, with the first register
    supplying the low bits.

    The PTX ``.xN`` repetition count multiplies the register counts above.
    A load result contains ``element_count * dtype.bitwidth / 32`` registers
    per thread. Dividing by the shape's register count at ``.x1`` gives the
    PTX repetition count. With ``pack=True``, the `PTX packing layout
    <https://docs.nvidia.com/cuda/archive/13.4.2/parallel-thread-execution/index.html#tcgen05-ld-st-pack-unpack>`_
    (Figure 196) first packs two 16-bit elements from adjacent TMEM columns
    into each register; the dtype reinterpretation then applies to that result.

    For example, loading with ``SHAPE_16x128b``, ``dtype=cl.float32``,
    ``element_count=2``, and ``pack=False`` selects ``.16x128b.x1``. Thread
    ``t`` receives element 0 from TMEM row ``t // 4``, column ``t % 4``,
    and element 1 from row ``t // 4 + 8`` at the same column, relative to
    the supplied base.
    """

    SHAPE_16x64b = "16x64b"
    SHAPE_16x128b = "16x128b"
    SHAPE_16x256b = "16x256b"
    SHAPE_32x32b = "32x32b"
    SHAPE_16x32bx2 = "16x32bx2"


class Tcgen05CopyMulticast(Enum):
    """TMEM warp distribution for :func:`~cuda.lang.tcgen05_copy`.

    Warp indices are relative to a warpgroup:

    * ``WARPX2_02_13`` forms pairs (0, 2) and (1, 3).
    * ``WARPX2_01_23`` forms pairs (0, 1) and (2, 3).
    * ``WARPX4`` multicasts the data to all four warps.

    With either ``WARPX2`` pattern, each pair receives the copied data and
    each warp in the pair receives half of it. The allowed patterns depend
    on :class:`~cuda.lang.Tcgen05CopyShape`. See the `PTX copy multicast rules
    <https://docs.nvidia.com/cuda/archive/13.4.2/parallel-thread-execution/index.html#tcgen05-instructions-tcgen05-cp>`_.
    """

    WARPX2_02_13 = 1
    WARPX2_01_23 = 2
    WARPX4 = 3


class Tcgen05CopyShape(Enum):
    """Data movement shapes for :func:`~cuda.lang.tcgen05_copy`.

    Each member selects the corresponding PTX shape: ``SHAPE_128x256b``
    selects ``.128x256b``, for example. The first dimension counts TMEM
    lanes (rows); the second gives the width across columns in bits.

    ``SHAPE_64x128b`` requires one of the ``WARPX2`` patterns in
    :class:`~cuda.lang.Tcgen05CopyMulticast`. ``SHAPE_32x128b`` requires
    ``WARPX4``. The other shapes use ``multicast=None``.
    The `PTX copy instruction reference
    <https://docs.nvidia.com/cuda/archive/13.4.2/parallel-thread-execution/index.html#tcgen05-instructions-tcgen05-cp>`_
    defines these shapes and multicast combinations.
    """
    SHAPE_128x256b = 0
    SHAPE_4x256b = 1
    SHAPE_128x128b = 2
    SHAPE_64x128b = 3
    SHAPE_32x128b = 4


class Tcgen05CopySourceFormat(Enum):
    """Packed source formats expanded by :func:`~cuda.lang.tcgen05_copy`.

    Each format describes a 16-byte group in shared memory:

    * ``B6x16_P32``: sixteen 6-bit values followed by 32 padding bits
      (PTX ``.b6x16_p32``).
    * ``B4x16_P64``: sixteen 4-bit values followed by 64 padding bits
      (PTX ``.b4x16_p64``).

    The copy expands each group into sixteen 8-bit elements in TMEM
    (``.b8x16``). The `PTX 6-bit decompression diagram
    <https://docs.nvidia.com/cuda/archive/13.4.2/parallel-thread-execution/index.html#tcgen05-decompression-6b8b>`_
    and `4-bit decompression diagram
    <https://docs.nvidia.com/cuda/archive/13.4.2/parallel-thread-execution/index.html#tcgen05-decompression-4b8b>`_
    show the source padding and destination bit layout. ``source_format=None``
    selects a copy without this decompression.
    """

    B6x16_P32 = 0
    B4x16_P64 = 1


class CTAGroup(Enum):
    """CTA scope of a tcgen05 operation.

    ``CTA_1`` selects the current CTA (PTX ``.cta_group::1``).
    ``CTA_2`` selects a pair of CTAs in the same cluster (``.cta_group::2``).
    The pair's cluster ranks differ only in their lowest bit, as defined by
    the `PTX CTA-pair rules
    <https://docs.nvidia.com/cuda/archive/13.4.2/parallel-thread-execution/index.html#tcgen05-cta-pair>`_.

    All tcgen05 instructions in a kernel use the same CTA group. Each API's
    participation contract specifies which threads issue the operation.
    See :ref:`lang-tcgen05` for allocation and participation examples, and
    the `PTX issue-granularity reference
    <https://docs.nvidia.com/cuda/archive/13.4.2/parallel-thread-execution/index.html#tcgen05-issue-granularity>`_
    for the two-CTA protocols.
    """

    CTA_1 = "cg1"
    CTA_2 = "cg2"


class Tcgen05WaitKind(Enum):
    LOAD = 0
    STORE = 1


class FenceProxyKind(Enum):
    ALIAS = "alias"
    ASYNC = "async"
    ASYNC_GLOBAL = "async.global"
    ASYNC_SHARED = "async.shared"
    TENSORMAP = "tensormap"
    GENERIC = "generic"


class FenceProxy(Enum):
    """Memory access proxy used by a fence."""

    ALIAS = "alias"
    ASYNC = "async"
    TENSORMAP = "tensormap"
    GENERIC = "generic"
    FABRIC = "fabric"


class BarrierReductionKind(Enum):
    POP_COUNT = auto()
    AND = auto()
    OR = auto()


class ShuffleKind(Enum):
    """Source-lane selection for :func:`cuda.lang.shuffle_sync`."""

    INDEX = "idx"
    """Read from an indexed lane within each sub-warp."""
    UP = "up"
    """Read from a lane at a lower index."""
    DOWN = "down"
    """Read from a lane at a higher index."""
    XOR = "bfly"
    """Select the source lane by XORing the lane index with the offset."""


class VectorReduction(Enum):
    """Operations that reduce a vector to one scalar."""

    add = "add"
    mul = "mul"
    bitwise_and = "bitwise_and"
    bitwise_or = "bitwise_or"
    bitwise_xor = "bitwise_xor"
    max = "max"
    min = "min"


class CachePolicy(Enum):
    L2_EVICT_LAST = "L2::evict_last"
    L2_EVICT_NORMAL = "L2::evict_normal"
    L2_EVICT_FIRST = "L2::evict_first"
    L2_EVICT_UNCHANGED = "L2::evict_unchanged"


class PrefetchLevel(Enum):
    L1 = auto()
    L2 = auto()


class MatrixLoadShape(Enum):
    M8N8 = "m8n8"
    M8N16 = "m8n16"
    M16N16 = "m16n16"


class MatrixStoreShape(Enum):
    M8N8 = "m8n8"
    M16N8 = "m16n8"


class MatrixLoadSourceFormat(Enum):
    B6X16_P32 = "b6x16_p32"
    B4X16_P64 = "b4x16_p64"


class AsyncCopyReportMechanism(Enum):
    DISABLED = auto()
    VALIDITY_PER_16_BYTES_80000000 = auto()
    VALIDITY_PER_16_BYTES_8000 = auto()
    VALIDITY_PER_16_BYTES_80 = auto()
    VALIDITY_PER_16_BYTES_8 = auto()
    VALIDITY_PER_ELEMENT_FF = auto()


__all__ = (
    "MemorySpace",
    "MemoryScope",
    "MemoryOrder",
    "AtomicOp",
    "RoundingMode",
    "SaturationMode",
    "SwizzleMode",
    "TensorMapInterleave",
    "TensorMapFloatOOBFill",
    "TensorMapL2Promotion",
    "MbarrierScope",
    "MbarrierLayout",
    "TMALoadMode",
    "TMAStoreMode",
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
    "Tcgen05WaitKind",
    "FenceProxyKind",
    "FenceProxy",
    "BarrierReductionKind",
    "VectorReduction",
    "CachePolicy",
    "PrefetchLevel",
    "MatrixStoreShape",
    "MatrixLoadShape",
    "MatrixLoadSourceFormat",
)
