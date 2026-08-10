# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from enum import Enum, auto
from cuda.tile import _cext
from cuda.tile._memory_model import MemorySpace, MemoryScope
from cuda.tile._numeric_semantics import RoundingMode


class MemoryOrder(Enum):
    """Memory ordering semantics of a memory operation."""

    WEAK = "weak"
    RELAXED = "relaxed"
    ACQUIRE = "acquire"
    RELEASE = "release"
    ACQ_REL = "acq_rel"
    SEQ_CST = "seq_cst"


class SaturationMode(Enum):
    """Saturation mode for floating-point and integer operations."""

    NONE = "none"
    """Do not saturate the result."""

    SATFINITE = "satfinite"
    """Limit the result to the largest finite value of its type."""

    SAT = "sat"
    """Limit a floating-point result to ``[0.0, 1.0]``."""


class SwizzleMode(Enum):
    """Shared-memory swizzle modes for tensor operations."""

    SWIZZLE_NONE = _cext.CU_TENSOR_MAP_SWIZZLE_NONE
    SWIZZLE_32B = _cext.CU_TENSOR_MAP_SWIZZLE_32B
    SWIZZLE_64B = _cext.CU_TENSOR_MAP_SWIZZLE_64B
    SWIZZLE_128B = _cext.CU_TENSOR_MAP_SWIZZLE_128B
    SWIZZLE_128B_ATOM_32B = _cext.CU_TENSOR_MAP_SWIZZLE_128B_ATOM_32B
    SWIZZLE_128B_ATOM_32B_FLIP_8B = _cext.CU_TENSOR_MAP_SWIZZLE_128B_ATOM_32B_FLIP_8B
    SWIZZLE_128B_ATOM_64B = _cext.CU_TENSOR_MAP_SWIZZLE_128B_ATOM_64B


class TensorMapL2Promotion(Enum):
    """L2-promotion size encoded in a tensor-map descriptor."""

    NONE = _cext.CU_TENSOR_MAP_L2_PROMOTION_NONE
    L2_64B = _cext.CU_TENSOR_MAP_L2_PROMOTION_L2_64B
    L2_128B = _cext.CU_TENSOR_MAP_L2_PROMOTION_L2_128B
    L2_256B = _cext.CU_TENSOR_MAP_L2_PROMOTION_L2_256B


class MbarrierScope(Enum):
    """Scope of the threads that observe an mbarrier operation."""

    BLOCK = "cta"
    CLUSTER = "cluster"


class TMALoadMode(Enum):
    TILE = 0
    IM2COL = 1
    IM2COL_W = 2
    IM2COL_W_128 = 3
    TILE_GATHER4 = 4


class TMAStoreMode(Enum):
    TILE = 0
    IM2COL = 1
    TILE_SCATTER4 = 2


class Tcgen05MMAKind(Enum):
    F16 = 0
    TF32 = 1
    F8F6F4 = 2
    I8 = 3


class Tcgen05MMABlockScaleKind(Enum):
    MXF8F6F4 = 0
    MXF4 = 1
    MXF4NVF4 = 2


class Tcgen05MMAScaleVectorSize(Enum):
    DEFAULT = 0
    BLOCK_16 = 1
    BLOCK_32 = 2


class Tcgen05MMACollectorBBuffer(Enum):
    BUFFER_0 = 0
    BUFFER_1 = 1
    BUFFER_2 = 2
    BUFFER_3 = 3


class Tcgen05MMACollectorOp(Enum):
    DISCARD = 0
    LASTUSE = 1
    FILL = 2
    USE = 3


class Tcgen05LoadStoreShape(Enum):
    """Load/store shapes supported by tcgen05 tensor memory operations."""

    SHAPE_16X64B = "16x64b"
    SHAPE_16X128B = "16x128b"
    SHAPE_16X256B = "16x256b"
    SHAPE_32X32B = "32x32b"
    SHAPE_16X32BX2 = "16x32bx2"


class CTAGroup(Enum):
    """CTA group selection for tcgen05 tensor memory operations."""

    CTA_1 = "cg1"
    CTA_2 = "cg2"


class Tcgen05WaitKind(Enum):
    LOAD = 0
    STORE = 1


class Tcgen05CopyMulticast(Enum):
    WARPX2_02_13 = 1
    WARPX2_01_23 = 2
    WARPX4 = 3


class Tcgen05CopyShape(Enum):
    SHAPE_128x256b = 0
    SHAPE_4x256b = 1
    SHAPE_128x128b = 2
    SHAPE_64x128b = 3
    SHAPE_32x128b = 4


class Tcgen05CopySourceFormat(Enum):
    B6x16_P32 = 0
    B4x16_P64 = 1


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


__all__ = (
    "MemorySpace",
    "MemoryScope",
    "MemoryOrder",
    "RoundingMode",
    "SaturationMode",
    "SwizzleMode",
    "TensorMapL2Promotion",
    "MbarrierScope",
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
