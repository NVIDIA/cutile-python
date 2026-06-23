# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import enum
from cuda.tile import _cext
from cuda.tile._memory_model import MemorySpace


class TensorMapSwizzle(enum.Enum):
    """Swizzle modes for tiled tensor map descriptors."""

    SWIZZLE_NONE = _cext.CU_TENSOR_MAP_SWIZZLE_NONE
    SWIZZLE_32B = _cext.CU_TENSOR_MAP_SWIZZLE_32B
    SWIZZLE_64B = _cext.CU_TENSOR_MAP_SWIZZLE_64B
    SWIZZLE_128B = _cext.CU_TENSOR_MAP_SWIZZLE_128B
    SWIZZLE_128B_ATOM_32B = _cext.CU_TENSOR_MAP_SWIZZLE_128B_ATOM_32B
    SWIZZLE_128B_ATOM_32B_FLIP_8B = _cext.CU_TENSOR_MAP_SWIZZLE_128B_ATOM_32B_FLIP_8B
    SWIZZLE_128B_ATOM_64B = _cext.CU_TENSOR_MAP_SWIZZLE_128B_ATOM_64B


class MbarrierScope(enum.Enum):
    """Scope of the threads that observe an mbarrier operation."""

    BLOCK = "cta"
    CLUSTER = "cluster"


class TMALoadMode(enum.Enum):
    TILE = 0
    IM2COL = 1
    IM2COL_W = 2
    IM2COL_W_128 = 3
    TILE_GATHER4 = 4


class TMAStoreMode(enum.Enum):
    TILE = 0
    IM2COL = 1
    TILE_SCATTER4 = 2


class Tcgen05MMAKind(enum.Enum):
    F16 = 0
    TF32 = 1
    F8F6F4 = 2
    I8 = 3
    MXF8F6F4 = 4
    MXF4 = 5
    MXF4NVF4 = 6


class Tcgen05MMACollectorOp(enum.Enum):
    DISCARD = 0
    LASTUSE = 1
    FILL = 2
    USE = 3


class Tcgen05LdStShape(enum.Enum):
    """Load/store shapes supported by tcgen05 tensor memory operations."""

    SHAPE_16X64B = "16x64b"
    SHAPE_16X128B = "16x128b"
    SHAPE_16X256B = "16x256b"
    SHAPE_32X32B = "32x32b"
    SHAPE_16X32BX2 = "16x32bx2"


class CTAGroup(enum.Enum):
    """CTA group selection for tcgen05 tensor memory operations."""

    CTA_1 = "cg1"
    CTA_2 = "cg2"


__all__ = (
    "MemorySpace",
    "TensorMapSwizzle",
    "MbarrierScope",
    "TMALoadMode",
    "TMAStoreMode",
    "CTAGroup",
    "Tcgen05MMAKind",
    "Tcgen05MMACollectorOp",
    "Tcgen05LdStShape",
)
