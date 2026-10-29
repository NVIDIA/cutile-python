# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from . import ArrayAttr
from . import BoolAttr
from . import DenseI32ArrayAttr
from . import DictionaryAttr
from . import FloatType
from . import IntegerAttr
from . import IntegerType
from . import StringAttr
from . import TypeAttr
from . import UnitAttr
from . import VectorType
from . import gpu
from . import llvm
from . import ptr
from ._builtins import Attribute
from ._builtins import Type
from ._builtins import Value
from ._builtins import add_operation
from dataclasses import dataclass
from typing import Optional
from typing import Sequence
import enum


# ========= 'nvvm' dialect of MLIR ==========


# ---- Interfaces ----


class BasicPtxBuilderInterface:
    def __init__(self):
        raise NotImplementedError('Interfaces cannot be instantiated')


class RequiresSMInterface:
    def __init__(self):
        raise NotImplementedError('Interfaces cannot be instantiated')


# ---- Enums ----


class BarrierReduction(enum.Enum):
    POPC = 0
    AND = 1
    OR = 2

    def _print_mlir_unqualified(self, p):
        p(("popc", "and", "or",)[self._value_])


class BlockScaleFormat(enum.Enum):
    UE8M0 = 0
    UE4M3 = 1

    def _print_mlir_unqualified(self, p):
        p(("ue8m0", "ue4m3",)[self._value_])


class CTAGroupKind(enum.Enum):
    CTA_1 = 0
    CTA_2 = 1

    def _print_mlir_unqualified(self, p):
        p(("cta_1", "cta_2",)[self._value_])


class CacheEvictionPriority(enum.Enum):
    EvictNormal = 0
    EvictFirst = 1
    EvictLast = 2
    EvictUnchanged = 3
    NoAllocate = 4

    def _print_mlir_unqualified(self, p):
        p(("evict_normal", "evict_first", "evict_last", "evict_unchanged",
           "no_allocate",)[self._value_])


class ClusterLaunchControlQueryType(enum.Enum):
    IS_CANCELED = 0
    GET_FIRST_CTA_ID_X = 1
    GET_FIRST_CTA_ID_Y = 2
    GET_FIRST_CTA_ID_Z = 3

    def _print_mlir_unqualified(self, p):
        p(("is_canceled", "get_first_cta_id_x", "get_first_cta_id_y",
           "get_first_cta_id_z",)[self._value_])


class DotAccumulateType(enum.Enum):
    SIGNED = 1
    UNSIGNED = 0

    def _print_mlir_unqualified(self, p):
        p(("unsigned", "signed",)[self._value_])


class FPRoundingMode(enum.Enum):
    NONE = 0
    RN = 1
    RM = 2
    RP = 3
    RZ = 4
    RNA = 5
    RS = 6

    def _print_mlir_unqualified(self, p):
        p(("none", "rn", "rm", "rp", "rz", "rna", "rs",)[self._value_])


class GridDepActionKind(enum.Enum):
    wait = 0
    launch_dependents = 1

    def _print_mlir_unqualified(self, p):
        p(("wait", "launch_dependents",)[self._value_])


class LdStMatrixEltType(enum.Enum):
    B16 = 0
    B8 = 1
    B8X16_B6X16_P32 = 2
    B8X16_B4X16_P64 = 3

    def _print_mlir_unqualified(self, p):
        p(("b16", "b8", "b8x16.b6x16_p32", "b8x16.b4x16_p64",)[self._value_])


class LoadCacheModifierKind(enum.Enum):
    CA = 0
    CG = 1
    CS = 2
    LU = 3
    CV = 4

    def _print_mlir_unqualified(self, p):
        p(("ca", "cg", "cs", "lu", "cv",)[self._value_])


class MMAB1Op(enum.Enum):
    none = 0
    xor_popc = 1
    and_popc = 2

    def _print_mlir_unqualified(self, p):
        p(("none", "xor_popc", "and_popc",)[self._value_])


class MMABlockScaleKind(enum.Enum):
    MXF8F6F4 = 0
    MXF4 = 1
    MXF4NVF4 = 2

    def _print_mlir_unqualified(self, p):
        p(("mxf8f6f4", "mxf4", "mxf4nvf4",)[self._value_])


class MMAFrag(enum.Enum):
    a = 0
    b = 1
    c = 2

    def _print_mlir_unqualified(self, p):
        p(("a", "b", "c",)[self._value_])


class MMAIntOverflow(enum.Enum):
    satfinite = 1
    wrapped = 0

    def _print_mlir_unqualified(self, p):
        p(("wrapped", "satfinite",)[self._value_])


class MMAKind(enum.Enum):
    f8f6f4 = 0

    def _print_mlir_unqualified(self, p):
        p(("f8f6f4",)[self._value_])


class MMALayout(enum.Enum):
    row = 0
    col = 1

    def _print_mlir_unqualified(self, p):
        p(("row", "col",)[self._value_])


class MMATypes(enum.Enum):
    f16 = 0
    f32 = 1
    tf32 = 2
    bf16 = 9
    s8 = 4
    u8 = 3
    s32 = 5
    s4 = 8
    u4 = 7
    b1 = 6
    f64 = 10
    e4m3 = 11
    e5m2 = 12
    e3m2 = 13
    e2m3 = 14
    e2m1 = 15

    def _print_mlir_unqualified(self, p):
        p(("f16", "f32", "tf32", "u8", "s8", "s32", "b1", "u4", "s4", "bf16", "f64", "e4m3",
           "e5m2", "e3m2", "e2m3", "e2m1",)[self._value_])


class MatchSyncKind(enum.Enum):
    any = 0
    all = 1

    def _print_mlir_unqualified(self, p):
        p(("any", "all",)[self._value_])


class MemOrderKind(enum.Enum):
    WEAK = 0
    RELAXED = 1
    ACQUIRE = 2
    RELEASE = 3
    ACQ_REL = 4
    SC = 5
    MMIO = 6
    VOLATILE = 7

    def _print_mlir_unqualified(self, p):
        p(("weak", "relaxed", "acquire", "release", "acq_rel", "sc", "mmio",
           "volatile",)[self._value_])


class MemScopeKind(enum.Enum):
    CTA = 0
    CLUSTER = 1
    GPU = 2
    SYS = 3

    def _print_mlir_unqualified(self, p):
        p(("cta", "cluster", "gpu", "sys",)[self._value_])


class NVVMMemorySpace(enum.Enum):
    Generic = 0
    Global = 1
    Shared = 3
    Constant = 4
    Local = 5
    Tensor = 6
    SharedCluster = 7

    def _print_mlir_unqualified(self, p):
        p(("generic", "global", "", "shared", "constant", "local", "tensor",
           "shared_cluster",)[self._value_])


class PermuteMode(enum.Enum):
    DEFAULT = 0
    F4E = 1
    B4E = 2
    RC8 = 3
    ECL = 4
    ECR = 5
    RC16 = 6

    def _print_mlir_unqualified(self, p):
        p(("default", "f4e", "b4e", "rc8", "ecl", "ecr", "rc16",)[self._value_])


class PrefetchCacheLevel(enum.Enum):
    L1 = 0
    L2 = 1

    def _print_mlir_unqualified(self, p):
        p(("L1", "L2",)[self._value_])


class ProxyKind(enum.Enum):
    alias = 0
    async_ = 1
    async_global = 2
    async_shared = 3
    TENSORMAP = 4
    GENERIC = 5

    def _print_mlir_unqualified(self, p):
        p(("alias", "async", "async.global", "async.shared", "tensormap",
           "generic",)[self._value_])


class ReduxKind(enum.Enum):
    ADD = 1
    AND = 2
    MAX = 3
    MIN = 4
    OR = 5
    UMAX = 6
    UMIN = 7
    XOR = 8
    FMIN = 9
    FMAX = 10

    def _print_mlir_unqualified(self, p):
        p(("", "add", "and", "max", "min", "or", "umax", "umin", "xor", "fmin",
           "fmax",)[self._value_])


class SaturationMode(enum.Enum):
    NONE = 0
    SATFINITE = 1

    def _print_mlir_unqualified(self, p):
        p(("none", "satfinite",)[self._value_])


class ScaleVecSize(enum.Enum):
    X1 = 0
    X2 = 1
    X4 = 2

    def _print_mlir_unqualified(self, p):
        p(("x1", "x2", "x4",)[self._value_])


class SetMaxRegisterAction(enum.Enum):
    decrease = 1
    increase = 0

    def _print_mlir_unqualified(self, p):
        p(("increase", "decrease",)[self._value_])


class SharedSpace(enum.Enum):
    shared_cta = 0
    shared_cluster = 1

    def _print_mlir_unqualified(self, p):
        p(("cta", "cluster",)[self._value_])


class ShflKind(enum.Enum):
    bfly = 0
    up = 1
    down = 2
    idx = 3

    def _print_mlir_unqualified(self, p):
        p(("bfly", "up", "down", "idx",)[self._value_])


class TMALoadMode(enum.Enum):
    TILE = 0
    IM2COL = 1
    IM2COL_W = 2
    IM2COL_W_128 = 3
    TILE_GATHER4 = 4

    def _print_mlir_unqualified(self, p):
        p(("tile", "im2col", "im2col_w", "im2col_w_128", "tile_gather4",)[self._value_])


class TMAReduxKind(enum.Enum):
    ADD = 0
    MAX = 2
    MIN = 1
    INC = 3
    DEC = 4
    AND = 5
    OR = 6
    XOR = 7

    def _print_mlir_unqualified(self, p):
        p(("add", "min", "max", "inc", "dec", "and", "or", "xor",)[self._value_])


class TMAStoreMode(enum.Enum):
    TILE = 0
    IM2COL = 1
    TILE_SCATTER4 = 2

    def _print_mlir_unqualified(self, p):
        p(("tile", "im2col", "tile_scatter4",)[self._value_])


class Tcgen05CpMulticast(enum.Enum):
    NONE = 0
    WARPX2_02_13 = 1
    WARPX2_01_23 = 2
    WARPX4 = 3

    def _print_mlir_unqualified(self, p):
        p(("none", "warpx2_02_13", "warpx2_01_23", "warpx4",)[self._value_])


class Tcgen05CpShape(enum.Enum):
    SHAPE_128x256b = 0
    SHAPE_4x256b = 1
    SHAPE_128x128b = 2
    SHAPE_64x128b = 3
    SHAPE_32x128b = 4

    def _print_mlir_unqualified(self, p):
        p(("shape_128x256b", "shape_4x256b", "shape_128x128b", "shape_64x128b",
           "shape_32x128b",)[self._value_])


class Tcgen05CpSrcFormat(enum.Enum):
    B6x16_P32 = 0
    B4x16_P64 = 1

    def _print_mlir_unqualified(self, p):
        p(("b6x16_p32", "b4x16_p64",)[self._value_])


class Tcgen05FenceKind(enum.Enum):
    BEFORE_THREAD_SYNC = 0
    AFTER_THREAD_SYNC = 1

    def _print_mlir_unqualified(self, p):
        p(("before", "after",)[self._value_])


class Tcgen05LdStShape(enum.Enum):
    SHAPE_16X64B = 0
    SHAPE_16X128B = 1
    SHAPE_16X256B = 2
    SHAPE_32X32B = 3
    SHAPE_16X32BX2 = 4

    def _print_mlir_unqualified(self, p):
        p(("shape_16x64b", "shape_16x128b", "shape_16x256b", "shape_32x32b",
           "shape_16x32bx2",)[self._value_])


class Tcgen05MMABlockScale(enum.Enum):
    DEFAULT = 0
    BLOCK16 = 1
    BLOCK32 = 2

    def _print_mlir_unqualified(self, p):
        p(("default", "block16", "block32",)[self._value_])


class Tcgen05MMACollectorBBuffer(enum.Enum):
    B0 = 0
    B1 = 1
    B2 = 2
    B3 = 3

    def _print_mlir_unqualified(self, p):
        p(("b0", "b1", "b2", "b3",)[self._value_])


class Tcgen05MMACollectorOp(enum.Enum):
    DISCARD = 0
    LASTUSE = 1
    FILL = 2
    USE = 3

    def _print_mlir_unqualified(self, p):
        p(("discard", "lastuse", "fill", "use",)[self._value_])


class Tcgen05MMAKind(enum.Enum):
    F8F6F4 = 2
    I8 = 3
    F16 = 0
    TF32 = 1

    def _print_mlir_unqualified(self, p):
        p(("f16", "tf32", "f8f6f4", "i8",)[self._value_])


class Tcgen05WaitKind(enum.Enum):
    LOAD = 0
    STORE = 1

    def _print_mlir_unqualified(self, p):
        p(("load", "store",)[self._value_])


class ValidatePattern(enum.Enum):
    PER_16BYTES_80000000 = 1
    PER_16BYTES_8000 = 2
    PER_16BYTES_80 = 3
    PER_16BYTES_8 = 4
    PER_ELEMENT_FF = 5

    def _print_mlir_unqualified(self, p):
        p(("", "per_16bytes_80000000", "per_16bytes_8000", "per_16bytes_80",
           "per_16bytes_8", "per_element_ff",)[self._value_])


class VoteSyncKind(enum.Enum):
    any = 0
    all = 1
    ballot = 2
    uni = 3

    def _print_mlir_unqualified(self, p):
        p(("any", "all", "ballot", "uni",)[self._value_])


class WGMMAScaleIn(enum.Enum):
    one = 1
    neg = -1

    def _print_mlir_unqualified(self, p):
        p(("", "neg",)[self._value_])


class WGMMAScaleOut(enum.Enum):
    zero = 0
    one = 1

    def _print_mlir_unqualified(self, p):
        p(("zero", "one",)[self._value_])


class WGMMATypes(enum.Enum):
    f16 = 0
    tf32 = 1
    u8 = 2
    s8 = 3
    b1 = 4
    bf16 = 5
    e4m3 = 6
    e5m2 = 7
    f32 = 8
    s32 = 9

    def _print_mlir_unqualified(self, p):
        p(("f16", "tf32", "u8", "s8", "b1", "bf16", "e4m3", "e5m2", "f32", "s32",)[self._value_])


# ---- Attributes ----


@dataclass(kw_only=True)
class BarrierReductionAttr(Attribute, dialect='nvvm', mnemonic='reduction'):
    value: "BarrierReduction"

    def _print_mlir_unqualified(self, p):
        p("<")
        self.value._print_mlir_unqualified(p)
        p(">")


@dataclass(kw_only=True)
class BlockScaleFormatAttr(Attribute, dialect='nvvm', mnemonic='block_scale_format'):
    value: "BlockScaleFormat"

    def _print_mlir_unqualified(self, p):
        p("<")
        self.value._print_mlir_unqualified(p)
        p(">")


@dataclass(kw_only=True)
class CTAGroupKindAttr(Attribute, dialect='nvvm', mnemonic='cta_group'):
    value: "CTAGroupKind"

    def _print_mlir_unqualified(self, p):
        p("<")
        self.value._print_mlir_unqualified(p)
        p(">")


@dataclass(kw_only=True)
class CacheEvictionPriorityAttr(Attribute, dialect='nvvm', mnemonic='cache_eviction_priority'):
    value: "CacheEvictionPriority"

    def _print_mlir_unqualified(self, p):
        self.value._print_mlir_unqualified(p)


@dataclass(kw_only=True)
class ClusterLaunchControlQueryTypeAttr(Attribute, dialect='nvvm',
                                        mnemonic='cluster_launch_control_query_type'):
    value: "ClusterLaunchControlQueryType"

    def _print_mlir_unqualified(self, p):
        self.value._print_mlir_unqualified(p)


@dataclass(kw_only=True)
class DotAccumulateTypeAttr(Attribute, dialect='nvvm', mnemonic='dot_accumulate_type'):
    value: "DotAccumulateType"

    def _print_mlir_unqualified(self, p):
        p("<")
        self.value._print_mlir_unqualified(p)
        p(">")


@dataclass(kw_only=True)
class FPRoundingModeAttr(Attribute, dialect='nvvm', mnemonic='fp_rnd_mode'):
    value: "FPRoundingMode"

    def _print_mlir_unqualified(self, p):
        p("<")
        self.value._print_mlir_unqualified(p)
        p(">")


@dataclass(kw_only=True)
class GridDepActionKindAttr(Attribute, dialect='nvvm', mnemonic='grid_dep_action'):
    value: "GridDepActionKind"

    def _print_mlir_unqualified(self, p):
        self.value._print_mlir_unqualified(p)


@dataclass(kw_only=True)
class LdStMatrixEltTypeAttr(Attribute, dialect='nvvm', mnemonic='ld_st_matrix_elt_type'):
    value: "LdStMatrixEltType"

    def _print_mlir_unqualified(self, p):
        p("<")
        self.value._print_mlir_unqualified(p)
        p(">")


@dataclass(kw_only=True)
class LdStMatrixShapeAttr(Attribute, dialect='nvvm', mnemonic='ld_st_matrix_shape'):
    m: "int"
    n: "int"

    def _print_mlir_unqualified(self, p):
        p("<m = ")
        p(str(self.m))
        p(", n = ")
        p(str(self.n))
        p(">")


@dataclass(kw_only=True)
class LoadCacheModifierKindAttr(Attribute, dialect='nvvm', mnemonic='load_cache_modifier'):
    value: "LoadCacheModifierKind"

    def _print_mlir_unqualified(self, p):
        self.value._print_mlir_unqualified(p)


@dataclass(kw_only=True)
class MMAB1OpAttr(Attribute, dialect='nvvm', mnemonic='mma_b1op'):
    value: "MMAB1Op"

    def _print_mlir_unqualified(self, p):
        p("<")
        self.value._print_mlir_unqualified(p)
        p(">")


@dataclass(kw_only=True)
class MMABlockScaleKindAttr(Attribute, dialect='nvvm', mnemonic='block_scale_kind'):
    value: "MMABlockScaleKind"

    def _print_mlir_unqualified(self, p):
        p("<")
        self.value._print_mlir_unqualified(p)
        p(">")


@dataclass(kw_only=True)
class MMAFragAttr(Attribute, dialect='nvvm', mnemonic='mma_frag'):
    value: "MMAFrag"

    def _print_mlir_unqualified(self, p):
        p("<")
        self.value._print_mlir_unqualified(p)
        p(">")


@dataclass(kw_only=True)
class MMAIntOverflowAttr(Attribute, dialect='nvvm', mnemonic='mma_int_overflow'):
    value: "MMAIntOverflow"

    def _print_mlir_unqualified(self, p):
        p("<")
        self.value._print_mlir_unqualified(p)
        p(">")


@dataclass(kw_only=True)
class MMAKindAttr(Attribute, dialect='nvvm', mnemonic='mma_kind'):
    value: "MMAKind"

    def _print_mlir_unqualified(self, p):
        p("<")
        self.value._print_mlir_unqualified(p)
        p(">")


@dataclass(kw_only=True)
class MMALayoutAttr(Attribute, dialect='nvvm', mnemonic='mma_layout'):
    value: "MMALayout"

    def _print_mlir_unqualified(self, p):
        p("<")
        self.value._print_mlir_unqualified(p)
        p(">")


@dataclass(kw_only=True)
class MMAShapeAttr(Attribute, dialect='nvvm', mnemonic='shape'):
    m: "int"
    n: "int"
    k: "int"

    def _print_mlir_unqualified(self, p):
        p("<m = ")
        p(str(self.m))
        p(", n = ")
        p(str(self.n))
        p(", k = ")
        p(str(self.k))
        p(">")


@dataclass(kw_only=True)
class MMATypesAttr(Attribute, dialect='nvvm', mnemonic='mma_type'):
    value: "MMATypes"

    def _print_mlir_unqualified(self, p):
        p("<")
        self.value._print_mlir_unqualified(p)
        p(">")


@dataclass(kw_only=True)
class MatchSyncKindAttr(Attribute, dialect='nvvm', mnemonic='match_sync_kind'):
    value: "MatchSyncKind"

    def _print_mlir_unqualified(self, p):
        self.value._print_mlir_unqualified(p)


@dataclass(kw_only=True)
class MemOrderKindAttr(Attribute, dialect='nvvm', mnemonic='mem_order'):
    value: "MemOrderKind"

    def _print_mlir_unqualified(self, p):
        p("<")
        self.value._print_mlir_unqualified(p)
        p(">")


@dataclass(kw_only=True)
class MemScopeKindAttr(Attribute, dialect='nvvm', mnemonic='mem_scope'):
    value: "MemScopeKind"

    def _print_mlir_unqualified(self, p):
        p("<")
        self.value._print_mlir_unqualified(p)
        p(">")


@dataclass(kw_only=True)
class NVVMMemorySpaceAttr(Attribute, llvm.LLVMAddrSpaceAttrInterface,
                          ptr.MemorySpaceAttrInterface, dialect='nvvm', mnemonic='memory_space'):
    value: "NVVMMemorySpace"

    def _print_mlir_unqualified(self, p):
        p("<")
        self.value._print_mlir_unqualified(p)
        p(">")


@dataclass(kw_only=True)
class NVVMTargetAttr(Attribute, gpu.TargetAttrVerifyInterface, dialect='nvvm', mnemonic='target'):
    O: "int" = 2
    triple: "str" = "nvptx64-nvidia-cuda"
    chip: "str" = "sm_50"
    features: "str" = ""
    flags: Optional["DictionaryAttr"] = None
    link: Optional["ArrayAttr"] = None
    verifyTarget: "bool" = True

    def _print_mlir_unqualified(self, p):
        if (self.O != 2 or self.triple != "nvptx64-nvidia-cuda" or self.chip != "sm_50" or
                self.features != "" or self.flags is not None or self.link is not None or
                not self.verifyTarget):
            p("<")
            comma = ""
            if self.O != 2:
                p("O = ")
                p(str(self.O))
                comma = ", "
            if self.triple != "nvptx64-nvidia-cuda":
                p(comma)
                p("triple = ")
                p.print_escaped_string(self.triple)
                comma = ", "
            if self.chip != "sm_50":
                p(comma)
                p("chip = ")
                p.print_escaped_string(self.chip)
                comma = ", "
            if self.features != "":
                p(comma)
                p("features = ")
                p.print_escaped_string(self.features)
                comma = ", "
            if self.flags is not None:
                p(comma)
                p("flags = ")
                self.flags._print_mlir_unqualified(p)
                comma = ", "
            if self.link is not None:
                p(comma)
                p("link = ")
                self.link._print_mlir_unqualified(p)
                comma = ", "
            if not self.verifyTarget:
                p(comma)
                p("verifyTarget = ")
                p("true" if self.verifyTarget else "false")
            p(">")
        else:
            pass


@dataclass(kw_only=True)
class PermuteModeAttr(Attribute, dialect='nvvm', mnemonic='permute_mode'):
    value: "PermuteMode"

    def _print_mlir_unqualified(self, p):
        p("<")
        self.value._print_mlir_unqualified(p)
        p(">")


@dataclass(kw_only=True)
class PrefetchCacheLevelAttr(Attribute, dialect='nvvm', mnemonic='prefetch_cache_level'):
    value: "PrefetchCacheLevel"

    def _print_mlir_unqualified(self, p):
        self.value._print_mlir_unqualified(p)


@dataclass(kw_only=True)
class ProxyKindAttr(Attribute, dialect='nvvm', mnemonic='proxy_kind'):
    value: "ProxyKind"

    def _print_mlir_unqualified(self, p):
        p("<")
        self.value._print_mlir_unqualified(p)
        p(">")


@dataclass(kw_only=True)
class ReduxKindAttr(Attribute, dialect='nvvm', mnemonic='redux_kind'):
    value: "ReduxKind"

    def _print_mlir_unqualified(self, p):
        self.value._print_mlir_unqualified(p)


@dataclass(kw_only=True)
class SaturationModeAttr(Attribute, dialect='nvvm', mnemonic='sat_mode'):
    value: "SaturationMode"

    def _print_mlir_unqualified(self, p):
        p("<")
        self.value._print_mlir_unqualified(p)
        p(">")


@dataclass(kw_only=True)
class ScaleVecSizeAttr(Attribute, dialect='nvvm', mnemonic='scale_vec_size'):
    value: "ScaleVecSize"

    def _print_mlir_unqualified(self, p):
        p("<")
        self.value._print_mlir_unqualified(p)
        p(">")


@dataclass(kw_only=True)
class SetMaxRegisterActionAttr(Attribute, dialect='nvvm', mnemonic='action'):
    value: "SetMaxRegisterAction"

    def _print_mlir_unqualified(self, p):
        self.value._print_mlir_unqualified(p)


@dataclass(kw_only=True)
class SharedSpaceAttr(Attribute, dialect='nvvm', mnemonic='shared_space'):
    value: "SharedSpace"

    def _print_mlir_unqualified(self, p):
        p("<")
        self.value._print_mlir_unqualified(p)
        p(">")


@dataclass(kw_only=True)
class ShflKindAttr(Attribute, dialect='nvvm', mnemonic='shfl_kind'):
    value: "ShflKind"

    def _print_mlir_unqualified(self, p):
        self.value._print_mlir_unqualified(p)


@dataclass(kw_only=True)
class TMALoadModeAttr(Attribute, dialect='nvvm', mnemonic='tma_load_mode'):
    value: "TMALoadMode"

    def _print_mlir_unqualified(self, p):
        p("<")
        self.value._print_mlir_unqualified(p)
        p(">")


@dataclass(kw_only=True)
class TMAReduxKindAttr(Attribute, dialect='nvvm', mnemonic='tma_redux_kind'):
    value: "TMAReduxKind"

    def _print_mlir_unqualified(self, p):
        p("<")
        self.value._print_mlir_unqualified(p)
        p(">")


@dataclass(kw_only=True)
class TMAStoreModeAttr(Attribute, dialect='nvvm', mnemonic='tma_store_mode'):
    value: "TMAStoreMode"

    def _print_mlir_unqualified(self, p):
        p("<")
        self.value._print_mlir_unqualified(p)
        p(">")


@dataclass(kw_only=True)
class Tcgen05CpMulticastAttr(Attribute, dialect='nvvm', mnemonic='tcgen05_cp_multicast'):
    value: "Tcgen05CpMulticast"

    def _print_mlir_unqualified(self, p):
        p("<")
        self.value._print_mlir_unqualified(p)
        p(">")


@dataclass(kw_only=True)
class Tcgen05CpShapeAttr(Attribute, dialect='nvvm', mnemonic='tcgen05_cp_shape'):
    value: "Tcgen05CpShape"

    def _print_mlir_unqualified(self, p):
        p("<")
        self.value._print_mlir_unqualified(p)
        p(">")


@dataclass(kw_only=True)
class Tcgen05CpSrcFormatAttr(Attribute, dialect='nvvm', mnemonic='tcgen05_cp_src_fmt'):
    value: "Tcgen05CpSrcFormat"

    def _print_mlir_unqualified(self, p):
        p("<")
        self.value._print_mlir_unqualified(p)
        p(">")


@dataclass(kw_only=True)
class Tcgen05FenceKindAttr(Attribute, dialect='nvvm', mnemonic='tcgen05_fence'):
    value: "Tcgen05FenceKind"

    def _print_mlir_unqualified(self, p):
        p("<")
        self.value._print_mlir_unqualified(p)
        p(">")


@dataclass(kw_only=True)
class Tcgen05LdStShapeAttr(Attribute, dialect='nvvm', mnemonic='tcgen05_ldst_shape'):
    value: "Tcgen05LdStShape"

    def _print_mlir_unqualified(self, p):
        p("<")
        self.value._print_mlir_unqualified(p)
        p(">")


@dataclass(kw_only=True)
class Tcgen05MMABlockScaleAttr(Attribute, dialect='nvvm', mnemonic='tcgen05_mma_block_scale'):
    value: "Tcgen05MMABlockScale"

    def _print_mlir_unqualified(self, p):
        p("<")
        self.value._print_mlir_unqualified(p)
        p(">")


@dataclass(kw_only=True)
class Tcgen05MMACollectorBBufferAttr(Attribute, dialect='nvvm',
                                     mnemonic='tcgen05_mma_collectorb'):
    value: "Tcgen05MMACollectorBBuffer"

    def _print_mlir_unqualified(self, p):
        p("<")
        self.value._print_mlir_unqualified(p)
        p(">")


@dataclass(kw_only=True)
class Tcgen05MMACollectorOpAttr(Attribute, dialect='nvvm', mnemonic='tcgen05_mma_collectorop'):
    value: "Tcgen05MMACollectorOp"

    def _print_mlir_unqualified(self, p):
        p("<")
        self.value._print_mlir_unqualified(p)
        p(">")


@dataclass(kw_only=True)
class Tcgen05MMAKindAttr(Attribute, dialect='nvvm', mnemonic='tcgen05_mma_kind'):
    value: "Tcgen05MMAKind"

    def _print_mlir_unqualified(self, p):
        p("<")
        self.value._print_mlir_unqualified(p)
        p(">")


@dataclass(kw_only=True)
class Tcgen05WaitKindAttr(Attribute, dialect='nvvm', mnemonic='tcgen05_wait'):
    value: "Tcgen05WaitKind"

    def _print_mlir_unqualified(self, p):
        p("<")
        self.value._print_mlir_unqualified(p)
        p(">")


@dataclass(kw_only=True)
class ValidatePatternAttr(Attribute, dialect='nvvm', mnemonic='validate_pattern'):
    value: "ValidatePattern"

    def _print_mlir_unqualified(self, p):
        p("<")
        self.value._print_mlir_unqualified(p)
        p(">")


@dataclass(kw_only=True)
class VoteSyncKindAttr(Attribute, dialect='nvvm', mnemonic='vote_sync_kind'):
    value: "VoteSyncKind"

    def _print_mlir_unqualified(self, p):
        self.value._print_mlir_unqualified(p)


@dataclass(kw_only=True)
class WGMMAScaleInAttr(Attribute, dialect='nvvm', mnemonic='wgmma_scale_in'):
    value: "WGMMAScaleIn"

    def _print_mlir_unqualified(self, p):
        p("<")
        self.value._print_mlir_unqualified(p)
        p(">")


@dataclass(kw_only=True)
class WGMMAScaleOutAttr(Attribute, dialect='nvvm', mnemonic='wgmma_scale_out'):
    value: "WGMMAScaleOut"

    def _print_mlir_unqualified(self, p):
        p("<")
        self.value._print_mlir_unqualified(p)
        p(">")


@dataclass(kw_only=True)
class WGMMATypesAttr(Attribute, dialect='nvvm', mnemonic='wgmma_type'):
    value: "WGMMATypes"

    def _print_mlir_unqualified(self, p):
        p("<")
        self.value._print_mlir_unqualified(p)
        p(">")


# ---- Operators ----


def add_AggrSmemSize(
    *,
    res_type: Type,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    return add_operation(
        name="nvvm.read.ptx.sreg.aggr.smem.size",
        result_type=res_type,
        operands=[],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_Barrier0Op(
    *,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> None:
    all_props = []
    return add_operation(
        name="nvvm.barrier0",
        result_type=None,
        operands=[],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_BarrierArriveOp(
    *,
    barrierId: Optional[Value] = None,
    numberOfThreads: Value,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> None:
    all_props = []
    return add_operation(
        name="nvvm.barrier.arrive",
        result_type=None,
        operands=[*([] if barrierId is None else [barrierId]), numberOfThreads],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_BarrierOp(
    *,
    res_type: Optional[IntegerType],
    barrierId: Optional[Value] = None,
    numberOfThreads: Optional[Value] = None,
    reductionOp: Optional[BarrierReduction] = None,
    reductionPredicate: Optional[Value] = None,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Optional[Value]:
    all_props = []
    if reductionOp is not None:
        all_props.append(('reductionOp', BarrierReductionAttr(value=reductionOp)))
    all_props.append(('operandSegmentSizes',
                      DenseI32ArrayAttr([int(barrierId is not None),
                                         int(numberOfThreads is not None),
                                         int(reductionPredicate is not None)])))
    return add_operation(
        name="nvvm.barrier",
        result_type=res_type,
        operands=[*([] if barrierId is None else [barrierId]),
                  *([] if numberOfThreads is None else [numberOfThreads]),
                  *([] if reductionPredicate is None else [reductionPredicate])],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_BlockDimXOp(
    *,
    res_type: Type,
    range: Optional[llvm.ConstantRangeAttr] = None,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    if range is not None:
        all_props.append(('range', range))
    return add_operation(
        name="nvvm.read.ptx.sreg.ntid.x",
        result_type=res_type,
        operands=[],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_BlockDimYOp(
    *,
    res_type: Type,
    range: Optional[llvm.ConstantRangeAttr] = None,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    if range is not None:
        all_props.append(('range', range))
    return add_operation(
        name="nvvm.read.ptx.sreg.ntid.y",
        result_type=res_type,
        operands=[],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_BlockDimZOp(
    *,
    res_type: Type,
    range: Optional[llvm.ConstantRangeAttr] = None,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    if range is not None:
        all_props.append(('range', range))
    return add_operation(
        name="nvvm.read.ptx.sreg.ntid.z",
        result_type=res_type,
        operands=[],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_BlockIdXOp(
    *,
    res_type: Type,
    range: Optional[llvm.ConstantRangeAttr] = None,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    if range is not None:
        all_props.append(('range', range))
    return add_operation(
        name="nvvm.read.ptx.sreg.ctaid.x",
        result_type=res_type,
        operands=[],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_BlockIdYOp(
    *,
    res_type: Type,
    range: Optional[llvm.ConstantRangeAttr] = None,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    if range is not None:
        all_props.append(('range', range))
    return add_operation(
        name="nvvm.read.ptx.sreg.ctaid.y",
        result_type=res_type,
        operands=[],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_BlockIdZOp(
    *,
    res_type: Type,
    range: Optional[llvm.ConstantRangeAttr] = None,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    if range is not None:
        all_props.append(('range', range))
    return add_operation(
        name="nvvm.read.ptx.sreg.ctaid.z",
        result_type=res_type,
        operands=[],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_BlockInClusterIdXOp(
    *,
    res_type: Type,
    range: Optional[llvm.ConstantRangeAttr] = None,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    if range is not None:
        all_props.append(('range', range))
    return add_operation(
        name="nvvm.read.ptx.sreg.cluster.ctaid.x",
        result_type=res_type,
        operands=[],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_BlockInClusterIdYOp(
    *,
    res_type: Type,
    range: Optional[llvm.ConstantRangeAttr] = None,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    if range is not None:
        all_props.append(('range', range))
    return add_operation(
        name="nvvm.read.ptx.sreg.cluster.ctaid.y",
        result_type=res_type,
        operands=[],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_BlockInClusterIdZOp(
    *,
    res_type: Type,
    range: Optional[llvm.ConstantRangeAttr] = None,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    if range is not None:
        all_props.append(('range', range))
    return add_operation(
        name="nvvm.read.ptx.sreg.cluster.ctaid.z",
        result_type=res_type,
        operands=[],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_Breakpoint(
    *,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> None:
    all_props = []
    return add_operation(
        name="nvvm.breakpoint",
        result_type=None,
        operands=[],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_BulkStoreOp(
    *,
    addr: Value,
    size: Value,
    initVal: int = 0,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> None:
    all_props = []
    all_props.append(('initVal', IntegerAttr.make(IntegerType.signless(64), initVal)))
    return add_operation(
        name="nvvm.st.bulk",
        result_type=None,
        operands=[addr, size],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_Clock64Op(
    *,
    res_type: Type,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    return add_operation(
        name="nvvm.read.ptx.sreg.clock64",
        result_type=res_type,
        operands=[],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_ClockOp(
    *,
    res_type: Type,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    return add_operation(
        name="nvvm.read.ptx.sreg.clock",
        result_type=res_type,
        operands=[],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_ClusterArriveOp(
    *,
    aligned: Optional[bool] = None,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> None:
    all_props = []
    if aligned is not None:
        all_props.append(('aligned', UnitAttr()))
    return add_operation(
        name="nvvm.cluster.arrive",
        result_type=None,
        operands=[],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_ClusterArriveRelaxedOp(
    *,
    aligned: Optional[bool] = None,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> None:
    all_props = []
    if aligned is not None:
        all_props.append(('aligned', UnitAttr()))
    return add_operation(
        name="nvvm.cluster.arrive.relaxed",
        result_type=None,
        operands=[],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_ClusterDim(
    *,
    res_type: Type,
    range: Optional[llvm.ConstantRangeAttr] = None,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    if range is not None:
        all_props.append(('range', range))
    return add_operation(
        name="nvvm.read.ptx.sreg.cluster.nctarank",
        result_type=res_type,
        operands=[],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_ClusterDimBlocksXOp(
    *,
    res_type: Type,
    range: Optional[llvm.ConstantRangeAttr] = None,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    if range is not None:
        all_props.append(('range', range))
    return add_operation(
        name="nvvm.read.ptx.sreg.cluster.nctaid.x",
        result_type=res_type,
        operands=[],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_ClusterDimBlocksYOp(
    *,
    res_type: Type,
    range: Optional[llvm.ConstantRangeAttr] = None,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    if range is not None:
        all_props.append(('range', range))
    return add_operation(
        name="nvvm.read.ptx.sreg.cluster.nctaid.y",
        result_type=res_type,
        operands=[],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_ClusterDimBlocksZOp(
    *,
    res_type: Type,
    range: Optional[llvm.ConstantRangeAttr] = None,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    if range is not None:
        all_props.append(('range', range))
    return add_operation(
        name="nvvm.read.ptx.sreg.cluster.nctaid.z",
        result_type=res_type,
        operands=[],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_ClusterDimXOp(
    *,
    res_type: Type,
    range: Optional[llvm.ConstantRangeAttr] = None,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    if range is not None:
        all_props.append(('range', range))
    return add_operation(
        name="nvvm.read.ptx.sreg.nclusterid.x",
        result_type=res_type,
        operands=[],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_ClusterDimYOp(
    *,
    res_type: Type,
    range: Optional[llvm.ConstantRangeAttr] = None,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    if range is not None:
        all_props.append(('range', range))
    return add_operation(
        name="nvvm.read.ptx.sreg.nclusterid.y",
        result_type=res_type,
        operands=[],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_ClusterDimZOp(
    *,
    res_type: Type,
    range: Optional[llvm.ConstantRangeAttr] = None,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    if range is not None:
        all_props.append(('range', range))
    return add_operation(
        name="nvvm.read.ptx.sreg.nclusterid.z",
        result_type=res_type,
        operands=[],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_ClusterId(
    *,
    res_type: Type,
    range: Optional[llvm.ConstantRangeAttr] = None,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    if range is not None:
        all_props.append(('range', range))
    return add_operation(
        name="nvvm.read.ptx.sreg.cluster.ctarank",
        result_type=res_type,
        operands=[],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_ClusterIdXOp(
    *,
    res_type: Type,
    range: Optional[llvm.ConstantRangeAttr] = None,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    if range is not None:
        all_props.append(('range', range))
    return add_operation(
        name="nvvm.read.ptx.sreg.clusterid.x",
        result_type=res_type,
        operands=[],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_ClusterIdYOp(
    *,
    res_type: Type,
    range: Optional[llvm.ConstantRangeAttr] = None,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    if range is not None:
        all_props.append(('range', range))
    return add_operation(
        name="nvvm.read.ptx.sreg.clusterid.y",
        result_type=res_type,
        operands=[],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_ClusterIdZOp(
    *,
    res_type: Type,
    range: Optional[llvm.ConstantRangeAttr] = None,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    if range is not None:
        all_props.append(('range', range))
    return add_operation(
        name="nvvm.read.ptx.sreg.clusterid.z",
        result_type=res_type,
        operands=[],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_ClusterLaunchControlQueryCancelOp(
    *,
    res_type: Type,
    query_type: ClusterLaunchControlQueryType,
    try_cancel_response: Value,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    all_props.append(('query_type', ClusterLaunchControlQueryTypeAttr(value=query_type)))
    return add_operation(
        name="nvvm.clusterlaunchcontrol.query.cancel",
        result_type=res_type,
        operands=[try_cancel_response],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_ClusterLaunchControlTryCancelOp(
    *,
    multicast: bool = False,
    smemAddress: Value,
    mbarrier: Value,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> None:
    all_props = []
    if multicast:
        all_props.append(('multicast', UnitAttr()))
    return add_operation(
        name="nvvm.clusterlaunchcontrol.try.cancel",
        result_type=None,
        operands=[smemAddress, mbarrier],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_ClusterWaitOp(
    *,
    aligned: Optional[bool] = None,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> None:
    all_props = []
    if aligned is not None:
        all_props.append(('aligned', UnitAttr()))
    return add_operation(
        name="nvvm.cluster.wait",
        result_type=None,
        operands=[],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_ConvertBF16x2ToF8x2Op(
    *,
    dst_type: Type,
    a: Value,
    rnd: FPRoundingMode = FPRoundingMode(0),
    sat: SaturationMode = SaturationMode(0),
    dstTy: Type,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    all_props.append(('rnd', FPRoundingModeAttr(value=rnd)))
    all_props.append(('sat', SaturationModeAttr(value=sat)))
    all_props.append(('dstTy', TypeAttr(value=dstTy)))
    return add_operation(
        name="nvvm.convert.bf16x2.to.f8x2",
        result_type=dst_type,
        operands=[a],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_ConvertF16x2ToF8x2Op(
    *,
    dst_type: Type,
    a: Value,
    relu: bool = False,
    dstTy: Type,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    all_props.append(('relu', BoolAttr(value=relu)))
    all_props.append(('dstTy', TypeAttr(value=dstTy)))
    return add_operation(
        name="nvvm.convert.f16x2.to.f8x2",
        result_type=dst_type,
        operands=[a],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_ConvertF32x2ToBF16x2Op(
    *,
    dst_type: VectorType,
    src_hi: Value,
    src_lo: Value,
    random_bits: Optional[Value] = None,
    rnd: FPRoundingMode = FPRoundingMode(0),
    sat: SaturationMode = SaturationMode(0),
    relu: bool = False,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    all_props.append(('rnd', FPRoundingModeAttr(value=rnd)))
    all_props.append(('sat', SaturationModeAttr(value=sat)))
    all_props.append(('relu', BoolAttr(value=relu)))
    return add_operation(
        name="nvvm.convert.f32x2.to.bf16x2",
        result_type=dst_type,
        operands=[src_hi, src_lo, *([] if random_bits is None else [random_bits])],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_ConvertF32x2ToF16x2Op(
    *,
    dst_type: VectorType,
    src_hi: Value,
    src_lo: Value,
    random_bits: Optional[Value] = None,
    rnd: FPRoundingMode = FPRoundingMode(0),
    sat: SaturationMode = SaturationMode(0),
    relu: bool = False,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    all_props.append(('rnd', FPRoundingModeAttr(value=rnd)))
    all_props.append(('sat', SaturationModeAttr(value=sat)))
    all_props.append(('relu', BoolAttr(value=relu)))
    return add_operation(
        name="nvvm.convert.f32x2.to.f16x2",
        result_type=dst_type,
        operands=[src_hi, src_lo, *([] if random_bits is None else [random_bits])],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_ConvertF32x2ToF4x2Op(
    *,
    dst_type: IntegerType,
    a: Value,
    b: Value,
    relu: bool = False,
    dstTy: Type,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    all_props.append(('relu', BoolAttr(value=relu)))
    all_props.append(('dstTy', TypeAttr(value=dstTy)))
    return add_operation(
        name="nvvm.convert.f32x2.to.f4x2",
        result_type=dst_type,
        operands=[a, b],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_ConvertF32x2ToF6x2Op(
    *,
    dst_type: Type,
    a: Value,
    b: Value,
    relu: bool = False,
    dstTy: Type,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    all_props.append(('relu', BoolAttr(value=relu)))
    all_props.append(('dstTy', TypeAttr(value=dstTy)))
    return add_operation(
        name="nvvm.convert.f32x2.to.f6x2",
        result_type=dst_type,
        operands=[a, b],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_ConvertF32x2ToF8x2Op(
    *,
    dst_type: Type,
    a: Value,
    b: Value,
    rnd: FPRoundingMode = FPRoundingMode(0),
    sat: SaturationMode = SaturationMode(0),
    relu: bool = False,
    dstTy: Type,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    all_props.append(('rnd', FPRoundingModeAttr(value=rnd)))
    all_props.append(('sat', SaturationModeAttr(value=sat)))
    all_props.append(('relu', BoolAttr(value=relu)))
    all_props.append(('dstTy', TypeAttr(value=dstTy)))
    return add_operation(
        name="nvvm.convert.f32x2.to.f8x2",
        result_type=dst_type,
        operands=[a, b],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_ConvertF32x4ToF4x4Op(
    *,
    dst_type: IntegerType,
    src: Value,
    rbits: Value,
    relu: bool = False,
    dstTy: Type,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    all_props.append(('relu', BoolAttr(value=relu)))
    all_props.append(('dstTy', TypeAttr(value=dstTy)))
    return add_operation(
        name="nvvm.convert.f32x4.to.f4x4",
        result_type=dst_type,
        operands=[src, rbits],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_ConvertF32x4ToF6x4Op(
    *,
    dst_type: VectorType,
    src: Value,
    rbits: Value,
    relu: bool = False,
    dstTy: Type,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    all_props.append(('relu', BoolAttr(value=relu)))
    all_props.append(('dstTy', TypeAttr(value=dstTy)))
    return add_operation(
        name="nvvm.convert.f32x4.to.f6x4",
        result_type=dst_type,
        operands=[src, rbits],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_ConvertF32x4ToF8x4Op(
    *,
    dst_type: VectorType,
    src: Value,
    rbits: Value,
    relu: bool = False,
    dstTy: Type,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    all_props.append(('relu', BoolAttr(value=relu)))
    all_props.append(('dstTy', TypeAttr(value=dstTy)))
    return add_operation(
        name="nvvm.convert.f32x4.to.f8x4",
        result_type=dst_type,
        operands=[src, rbits],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_ConvertF4x2ToF16x2Op(
    *,
    dst_type: VectorType,
    src: Value,
    relu: bool = False,
    srcType: Type,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    all_props.append(('relu', BoolAttr(value=relu)))
    all_props.append(('srcType', TypeAttr(value=srcType)))
    return add_operation(
        name="nvvm.convert.f4x2.to.f16x2",
        result_type=dst_type,
        operands=[src],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_ConvertF6x2ToF16x2Op(
    *,
    dst_type: VectorType,
    src: Value,
    relu: bool = False,
    srcType: Type,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    all_props.append(('relu', BoolAttr(value=relu)))
    all_props.append(('srcType', TypeAttr(value=srcType)))
    return add_operation(
        name="nvvm.convert.f6x2.to.f16x2",
        result_type=dst_type,
        operands=[src],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_ConvertF8x2ToBF16x2Op(
    *,
    dst_type: VectorType,
    src: Value,
    srcType: Type,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    all_props.append(('srcType', TypeAttr(value=srcType)))
    return add_operation(
        name="nvvm.convert.f8x2.to.bf16x2",
        result_type=dst_type,
        operands=[src],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_ConvertF8x2ToF16x2Op(
    *,
    dst_type: VectorType,
    src: Value,
    relu: bool = False,
    srcType: Type,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    all_props.append(('relu', BoolAttr(value=relu)))
    all_props.append(('srcType', TypeAttr(value=srcType)))
    return add_operation(
        name="nvvm.convert.f8x2.to.f16x2",
        result_type=dst_type,
        operands=[src],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_ConvertFloatToTF32Op(
    *,
    res_type: IntegerType,
    src: Value,
    rnd: FPRoundingMode = FPRoundingMode(0),
    sat: SaturationMode = SaturationMode(0),
    relu: bool = False,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    all_props.append(('rnd', FPRoundingModeAttr(value=rnd)))
    all_props.append(('sat', SaturationModeAttr(value=sat)))
    all_props.append(('relu', BoolAttr(value=relu)))
    return add_operation(
        name="nvvm.convert.float.to.tf32",
        result_type=res_type,
        operands=[src],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_CpAsyncBulkCommitGroupOp(
    *,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> None:
    all_props = []
    return add_operation(
        name="nvvm.cp.async.bulk.commit.group",
        result_type=None,
        operands=[],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_CpAsyncBulkGlobalToSharedClusterOp(
    *,
    dstMem: Value,
    srcMem: Value,
    mbar: Value,
    size: Value,
    multicastMask: Optional[Value] = None,
    l2CacheHint: Optional[Value] = None,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> None:
    all_props = []
    all_props.append(('operandSegmentSizes',
                      DenseI32ArrayAttr([1, 1, 1, 1, int(multicastMask is not None),
                                         int(l2CacheHint is not None)])))
    return add_operation(
        name="nvvm.cp.async.bulk.shared.cluster.global",
        result_type=None,
        operands=[dstMem, srcMem, mbar, size, *([] if multicastMask is None else [multicastMask]),
                  *([] if l2CacheHint is None else [l2CacheHint])],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_CpAsyncBulkPrefetchOp(
    *,
    srcMem: Value,
    size: Value,
    l2CacheHint: Optional[Value] = None,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> None:
    all_props = []
    return add_operation(
        name="nvvm.cp.async.bulk.prefetch",
        result_type=None,
        operands=[srcMem, size, *([] if l2CacheHint is None else [l2CacheHint])],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_CpAsyncBulkSharedCTAToGlobalOp(
    *,
    dstMem: Value,
    srcMem: Value,
    size: Value,
    l2CacheHint: Optional[Value] = None,
    byteMask: Optional[Value] = None,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> None:
    all_props = []
    all_props.append(('operandSegmentSizes',
                      DenseI32ArrayAttr([1, 1, 1, int(l2CacheHint is not None),
                                         int(byteMask is not None)])))
    return add_operation(
        name="nvvm.cp.async.bulk.global.shared.cta",
        result_type=None,
        operands=[dstMem, srcMem, size, *([] if l2CacheHint is None else [l2CacheHint]),
                  *([] if byteMask is None else [byteMask])],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_CpAsyncBulkSharedCTAToSharedClusterOp(
    *,
    dstMem: Value,
    srcMem: Value,
    mbar: Value,
    size: Value,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> None:
    all_props = []
    return add_operation(
        name="nvvm.cp.async.bulk.shared.cluster.shared.cta",
        result_type=None,
        operands=[dstMem, srcMem, mbar, size],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_CpAsyncBulkTensorGlobalToSharedClusterOp(
    *,
    dstMem: Value,
    tmaDescriptor: Value,
    coordinates: Sequence[Value],
    mbar: Value,
    im2colOffsets: Sequence[Value],
    multicastMask: Optional[Value] = None,
    validatePattern: Optional[ValidatePattern] = None,
    l2CacheHint: Optional[Value] = None,
    mode: TMALoadMode = TMALoadMode(0),
    isCTAOnly: bool = False,
    group: Optional[CTAGroupKind] = None,
    predicate: Optional[Value] = None,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> None:
    all_props = []
    if validatePattern is not None:
        all_props.append(('validatePattern', ValidatePatternAttr(value=validatePattern)))
    all_props.append(('mode', TMALoadModeAttr(value=mode)))
    all_props.append(('isCTAOnly', BoolAttr(value=isCTAOnly)))
    if group is not None:
        all_props.append(('group', CTAGroupKindAttr(value=group)))
    all_props.append(('operandSegmentSizes',
                      DenseI32ArrayAttr([1, 1, len(coordinates), 1, len(im2colOffsets),
                                         int(multicastMask is not None),
                                         int(l2CacheHint is not None),
                                         int(predicate is not None)])))
    return add_operation(
        name="nvvm.cp.async.bulk.tensor.shared.cluster.global",
        result_type=None,
        operands=[dstMem, tmaDescriptor, *coordinates, mbar, *im2colOffsets,
                  *([] if multicastMask is None else [multicastMask]),
                  *([] if l2CacheHint is None else [l2CacheHint]),
                  *([] if predicate is None else [predicate])],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_CpAsyncBulkTensorPrefetchOp(
    *,
    tmaDescriptor: Value,
    coordinates: Sequence[Value],
    im2colOffsets: Sequence[Value],
    mode: TMALoadMode = TMALoadMode(0),
    l2CacheHint: Optional[Value] = None,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> None:
    all_props = []
    all_props.append(('mode', TMALoadModeAttr(value=mode)))
    all_props.append(('operandSegmentSizes',
                      DenseI32ArrayAttr([1, len(coordinates), len(im2colOffsets),
                                         int(l2CacheHint is not None)])))
    return add_operation(
        name="nvvm.cp.async.bulk.tensor.prefetch",
        result_type=None,
        operands=[tmaDescriptor, *coordinates, *im2colOffsets,
                  *([] if l2CacheHint is None else [l2CacheHint])],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_CpAsyncBulkTensorReduceOp(
    *,
    tmaDescriptor: Value,
    srcMem: Value,
    redKind: TMAReduxKind,
    mode: TMAStoreMode = TMAStoreMode(0),
    coordinates: Sequence[Value],
    l2CacheHint: Optional[Value] = None,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> None:
    all_props = []
    all_props.append(('redKind', TMAReduxKindAttr(value=redKind)))
    all_props.append(('mode', TMAStoreModeAttr(value=mode)))
    all_props.append(('operandSegmentSizes',
                      DenseI32ArrayAttr([1, 1, len(coordinates), int(l2CacheHint is not None)])))
    return add_operation(
        name="nvvm.cp.async.bulk.tensor.reduce",
        result_type=None,
        operands=[tmaDescriptor, srcMem, *coordinates,
                  *([] if l2CacheHint is None else [l2CacheHint])],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_CpAsyncBulkTensorSharedCTAToGlobalOp(
    *,
    tmaDescriptor: Value,
    srcMem: Value,
    coordinates: Sequence[Value],
    l2CacheHint: Optional[Value] = None,
    mode: TMAStoreMode = TMAStoreMode(0),
    predicate: Optional[Value] = None,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> None:
    all_props = []
    all_props.append(('mode', TMAStoreModeAttr(value=mode)))
    all_props.append(('operandSegmentSizes',
                      DenseI32ArrayAttr([1, 1, len(coordinates), int(l2CacheHint is not None),
                                         int(predicate is not None)])))
    return add_operation(
        name="nvvm.cp.async.bulk.tensor.global.shared.cta",
        result_type=None,
        operands=[tmaDescriptor, srcMem, *coordinates,
                  *([] if l2CacheHint is None else [l2CacheHint]),
                  *([] if predicate is None else [predicate])],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_CpAsyncBulkWaitGroupOp(
    *,
    group: int,
    read: Optional[bool] = None,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> None:
    all_props = []
    all_props.append(('group', IntegerAttr.make(IntegerType.signless(32), group)))
    if read is not None:
        all_props.append(('read', UnitAttr()))
    return add_operation(
        name="nvvm.cp.async.bulk.wait_group",
        result_type=None,
        operands=[],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_CpAsyncCommitGroupOp(
    *,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> None:
    all_props = []
    return add_operation(
        name="nvvm.cp.async.commit.group",
        result_type=None,
        operands=[],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_CpAsyncMBarrierArriveOp(
    *,
    addr: Value,
    noinc: bool = False,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> None:
    all_props = []
    all_props.append(('noinc', IntegerAttr.make(IntegerType.signless(1), int(noinc))))
    return add_operation(
        name="nvvm.cp.async.mbarrier.arrive",
        result_type=None,
        operands=[addr],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_CpAsyncOp(
    *,
    dst: Value,
    src: Value,
    size: int,
    modifier: LoadCacheModifierKind,
    cpSize: Optional[Value] = None,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> None:
    all_props = []
    all_props.append(('size', IntegerAttr.make(IntegerType.signless(32), size)))
    all_props.append(('modifier', LoadCacheModifierKindAttr(value=modifier)))
    return add_operation(
        name="nvvm.cp.async.shared.global",
        result_type=None,
        operands=[dst, src, *([] if cpSize is None else [cpSize])],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_CpAsyncWaitGroupOp(
    *,
    n: int,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> None:
    all_props = []
    all_props.append(('n', IntegerAttr.make(IntegerType.signless(32), n)))
    return add_operation(
        name="nvvm.cp.async.wait.group",
        result_type=None,
        operands=[],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_DotAccumulate2WayOp(
    *,
    res_type: IntegerType,
    a: Value,
    a_type: DotAccumulateType,
    b: Value,
    b_type: DotAccumulateType,
    c: Value,
    b_hi: bool,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    all_props.append(('a_type', DotAccumulateTypeAttr(value=a_type)))
    all_props.append(('b_type', DotAccumulateTypeAttr(value=b_type)))
    all_props.append(('b_hi', BoolAttr(value=b_hi)))
    return add_operation(
        name="nvvm.dot.accumulate.2way",
        result_type=res_type,
        operands=[a, b, c],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_DotAccumulate4WayOp(
    *,
    res_type: IntegerType,
    a: Value,
    a_type: DotAccumulateType,
    b: Value,
    b_type: DotAccumulateType,
    c: Value,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    all_props.append(('a_type', DotAccumulateTypeAttr(value=a_type)))
    all_props.append(('b_type', DotAccumulateTypeAttr(value=b_type)))
    return add_operation(
        name="nvvm.dot.accumulate.4way",
        result_type=res_type,
        operands=[a, b, c],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_DynamicSmemSize(
    *,
    res_type: Type,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    return add_operation(
        name="nvvm.read.ptx.sreg.dynamic.smem.size",
        result_type=res_type,
        operands=[],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_ElectSyncOp(
    *,
    pred_type: IntegerType,
    membermask: Optional[Value] = None,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    return add_operation(
        name="nvvm.elect.sync",
        result_type=pred_type,
        operands=[] if membermask is None else [membermask],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_EnvReg0Op(
    *,
    res_type: Type,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    return add_operation(
        name="nvvm.read.ptx.sreg.envreg0",
        result_type=res_type,
        operands=[],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_EnvReg10Op(
    *,
    res_type: Type,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    return add_operation(
        name="nvvm.read.ptx.sreg.envreg10",
        result_type=res_type,
        operands=[],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_EnvReg11Op(
    *,
    res_type: Type,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    return add_operation(
        name="nvvm.read.ptx.sreg.envreg11",
        result_type=res_type,
        operands=[],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_EnvReg12Op(
    *,
    res_type: Type,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    return add_operation(
        name="nvvm.read.ptx.sreg.envreg12",
        result_type=res_type,
        operands=[],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_EnvReg13Op(
    *,
    res_type: Type,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    return add_operation(
        name="nvvm.read.ptx.sreg.envreg13",
        result_type=res_type,
        operands=[],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_EnvReg14Op(
    *,
    res_type: Type,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    return add_operation(
        name="nvvm.read.ptx.sreg.envreg14",
        result_type=res_type,
        operands=[],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_EnvReg15Op(
    *,
    res_type: Type,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    return add_operation(
        name="nvvm.read.ptx.sreg.envreg15",
        result_type=res_type,
        operands=[],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_EnvReg16Op(
    *,
    res_type: Type,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    return add_operation(
        name="nvvm.read.ptx.sreg.envreg16",
        result_type=res_type,
        operands=[],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_EnvReg17Op(
    *,
    res_type: Type,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    return add_operation(
        name="nvvm.read.ptx.sreg.envreg17",
        result_type=res_type,
        operands=[],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_EnvReg18Op(
    *,
    res_type: Type,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    return add_operation(
        name="nvvm.read.ptx.sreg.envreg18",
        result_type=res_type,
        operands=[],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_EnvReg19Op(
    *,
    res_type: Type,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    return add_operation(
        name="nvvm.read.ptx.sreg.envreg19",
        result_type=res_type,
        operands=[],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_EnvReg1Op(
    *,
    res_type: Type,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    return add_operation(
        name="nvvm.read.ptx.sreg.envreg1",
        result_type=res_type,
        operands=[],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_EnvReg20Op(
    *,
    res_type: Type,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    return add_operation(
        name="nvvm.read.ptx.sreg.envreg20",
        result_type=res_type,
        operands=[],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_EnvReg21Op(
    *,
    res_type: Type,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    return add_operation(
        name="nvvm.read.ptx.sreg.envreg21",
        result_type=res_type,
        operands=[],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_EnvReg22Op(
    *,
    res_type: Type,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    return add_operation(
        name="nvvm.read.ptx.sreg.envreg22",
        result_type=res_type,
        operands=[],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_EnvReg23Op(
    *,
    res_type: Type,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    return add_operation(
        name="nvvm.read.ptx.sreg.envreg23",
        result_type=res_type,
        operands=[],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_EnvReg24Op(
    *,
    res_type: Type,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    return add_operation(
        name="nvvm.read.ptx.sreg.envreg24",
        result_type=res_type,
        operands=[],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_EnvReg25Op(
    *,
    res_type: Type,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    return add_operation(
        name="nvvm.read.ptx.sreg.envreg25",
        result_type=res_type,
        operands=[],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_EnvReg26Op(
    *,
    res_type: Type,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    return add_operation(
        name="nvvm.read.ptx.sreg.envreg26",
        result_type=res_type,
        operands=[],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_EnvReg27Op(
    *,
    res_type: Type,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    return add_operation(
        name="nvvm.read.ptx.sreg.envreg27",
        result_type=res_type,
        operands=[],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_EnvReg28Op(
    *,
    res_type: Type,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    return add_operation(
        name="nvvm.read.ptx.sreg.envreg28",
        result_type=res_type,
        operands=[],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_EnvReg29Op(
    *,
    res_type: Type,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    return add_operation(
        name="nvvm.read.ptx.sreg.envreg29",
        result_type=res_type,
        operands=[],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_EnvReg2Op(
    *,
    res_type: Type,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    return add_operation(
        name="nvvm.read.ptx.sreg.envreg2",
        result_type=res_type,
        operands=[],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_EnvReg30Op(
    *,
    res_type: Type,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    return add_operation(
        name="nvvm.read.ptx.sreg.envreg30",
        result_type=res_type,
        operands=[],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_EnvReg31Op(
    *,
    res_type: Type,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    return add_operation(
        name="nvvm.read.ptx.sreg.envreg31",
        result_type=res_type,
        operands=[],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_EnvReg3Op(
    *,
    res_type: Type,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    return add_operation(
        name="nvvm.read.ptx.sreg.envreg3",
        result_type=res_type,
        operands=[],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_EnvReg4Op(
    *,
    res_type: Type,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    return add_operation(
        name="nvvm.read.ptx.sreg.envreg4",
        result_type=res_type,
        operands=[],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_EnvReg5Op(
    *,
    res_type: Type,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    return add_operation(
        name="nvvm.read.ptx.sreg.envreg5",
        result_type=res_type,
        operands=[],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_EnvReg6Op(
    *,
    res_type: Type,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    return add_operation(
        name="nvvm.read.ptx.sreg.envreg6",
        result_type=res_type,
        operands=[],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_EnvReg7Op(
    *,
    res_type: Type,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    return add_operation(
        name="nvvm.read.ptx.sreg.envreg7",
        result_type=res_type,
        operands=[],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_EnvReg8Op(
    *,
    res_type: Type,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    return add_operation(
        name="nvvm.read.ptx.sreg.envreg8",
        result_type=res_type,
        operands=[],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_EnvReg9Op(
    *,
    res_type: Type,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    return add_operation(
        name="nvvm.read.ptx.sreg.envreg9",
        result_type=res_type,
        operands=[],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_Exit(
    *,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> None:
    all_props = []
    return add_operation(
        name="nvvm.exit",
        result_type=None,
        operands=[],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_FenceMbarrierInitOp(
    *,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> None:
    all_props = []
    return add_operation(
        name="nvvm.fence.mbarrier.init",
        result_type=None,
        operands=[],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_FenceProxyAcquireOp(
    *,
    scope: MemScopeKind,
    addr: Value,
    size: Value,
    fromProxy: ProxyKind = ProxyKind(5),
    toProxy: ProxyKind = ProxyKind(4),
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> None:
    all_props = []
    all_props.append(('scope', MemScopeKindAttr(value=scope)))
    all_props.append(('fromProxy', ProxyKindAttr(value=fromProxy)))
    all_props.append(('toProxy', ProxyKindAttr(value=toProxy)))
    return add_operation(
        name="nvvm.fence.proxy.acquire",
        result_type=None,
        operands=[addr, size],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_FenceProxyOp(
    *,
    kind: ProxyKind,
    space: Optional[SharedSpace] = None,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> None:
    all_props = []
    all_props.append(('kind', ProxyKindAttr(value=kind)))
    if space is not None:
        all_props.append(('space', SharedSpaceAttr(value=space)))
    return add_operation(
        name="nvvm.fence.proxy",
        result_type=None,
        operands=[],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_FenceProxyReleaseOp(
    *,
    scope: MemScopeKind,
    fromProxy: ProxyKind = ProxyKind(5),
    toProxy: ProxyKind = ProxyKind(4),
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> None:
    all_props = []
    all_props.append(('scope', MemScopeKindAttr(value=scope)))
    all_props.append(('fromProxy', ProxyKindAttr(value=fromProxy)))
    all_props.append(('toProxy', ProxyKindAttr(value=toProxy)))
    return add_operation(
        name="nvvm.fence.proxy.release",
        result_type=None,
        operands=[],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_FenceProxySyncRestrictOp(
    *,
    order: MemOrderKind,
    fromProxy: ProxyKind = ProxyKind(5),
    toProxy: ProxyKind = ProxyKind(1),
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> None:
    all_props = []
    all_props.append(('order', MemOrderKindAttr(value=order)))
    all_props.append(('fromProxy', ProxyKindAttr(value=fromProxy)))
    all_props.append(('toProxy', ProxyKindAttr(value=toProxy)))
    return add_operation(
        name="nvvm.fence.proxy.sync_restrict",
        result_type=None,
        operands=[],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_FenceScClusterOp(
    *,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> None:
    all_props = []
    return add_operation(
        name="nvvm.fence.sc.cluster",
        result_type=None,
        operands=[],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_FenceSyncRestrictOp(
    *,
    order: MemOrderKind,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> None:
    all_props = []
    all_props.append(('order', MemOrderKindAttr(value=order)))
    return add_operation(
        name="nvvm.fence.sync_restrict",
        result_type=None,
        operands=[],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_GlobalTimerLoOp(
    *,
    res_type: Type,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    return add_operation(
        name="nvvm.read.ptx.sreg.globaltimer.lo",
        result_type=res_type,
        operands=[],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_GlobalTimerOp(
    *,
    res_type: Type,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    return add_operation(
        name="nvvm.read.ptx.sreg.globaltimer",
        result_type=res_type,
        operands=[],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_GridDimXOp(
    *,
    res_type: Type,
    range: Optional[llvm.ConstantRangeAttr] = None,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    if range is not None:
        all_props.append(('range', range))
    return add_operation(
        name="nvvm.read.ptx.sreg.nctaid.x",
        result_type=res_type,
        operands=[],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_GridDimYOp(
    *,
    res_type: Type,
    range: Optional[llvm.ConstantRangeAttr] = None,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    if range is not None:
        all_props.append(('range', range))
    return add_operation(
        name="nvvm.read.ptx.sreg.nctaid.y",
        result_type=res_type,
        operands=[],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_GridDimZOp(
    *,
    res_type: Type,
    range: Optional[llvm.ConstantRangeAttr] = None,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    if range is not None:
        all_props.append(('range', range))
    return add_operation(
        name="nvvm.read.ptx.sreg.nctaid.z",
        result_type=res_type,
        operands=[],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_GridIdOp(
    *,
    res_type: Type,
    range: Optional[llvm.ConstantRangeAttr] = None,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    if range is not None:
        all_props.append(('range', range))
    return add_operation(
        name="nvvm.read.ptx.sreg.gridid",
        result_type=res_type,
        operands=[],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_GriddepcontrolOp(
    *,
    kind: GridDepActionKind,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> None:
    all_props = []
    all_props.append(('kind', GridDepActionKindAttr(value=kind)))
    return add_operation(
        name="nvvm.griddepcontrol",
        result_type=None,
        operands=[],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_InlinePtxOp(
    *,
    writeOnlyArgs_types: Sequence[Type],
    readOnlyArgs: Sequence[Value],
    readWriteArgs: Sequence[Value],
    ptxCode: str,
    predicate: Optional[Value] = None,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> tuple[Value, ...]:
    all_props = []
    all_props.append(('ptxCode', StringAttr(value=ptxCode)))
    all_props.append(('operandSegmentSizes',
                      DenseI32ArrayAttr([len(readOnlyArgs), len(readWriteArgs),
                                         int(predicate is not None)])))
    return add_operation(
        name="nvvm.inline_ptx",
        result_type=writeOnlyArgs_types,
        operands=[*readOnlyArgs, *readWriteArgs, *([] if predicate is None else [predicate])],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_LaneIdOp(
    *,
    res_type: Type,
    range: Optional[llvm.ConstantRangeAttr] = None,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    if range is not None:
        all_props.append(('range', range))
    return add_operation(
        name="nvvm.read.ptx.sreg.laneid",
        result_type=res_type,
        operands=[],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_LaneMaskEqOp(
    *,
    res_type: Type,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    return add_operation(
        name="nvvm.read.ptx.sreg.lanemask.eq",
        result_type=res_type,
        operands=[],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_LaneMaskGeOp(
    *,
    res_type: Type,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    return add_operation(
        name="nvvm.read.ptx.sreg.lanemask.ge",
        result_type=res_type,
        operands=[],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_LaneMaskGtOp(
    *,
    res_type: Type,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    return add_operation(
        name="nvvm.read.ptx.sreg.lanemask.gt",
        result_type=res_type,
        operands=[],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_LaneMaskLeOp(
    *,
    res_type: Type,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    return add_operation(
        name="nvvm.read.ptx.sreg.lanemask.le",
        result_type=res_type,
        operands=[],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_LaneMaskLtOp(
    *,
    res_type: Type,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    return add_operation(
        name="nvvm.read.ptx.sreg.lanemask.lt",
        result_type=res_type,
        operands=[],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_LdMatrixOp(
    *,
    res_type: Type,
    ptr: Value,
    num: int,
    layout: MMALayout,
    shape: LdStMatrixShapeAttr,
    eltType: LdStMatrixEltType,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    all_props.append(('num', IntegerAttr.make(IntegerType.signless(32), num)))
    all_props.append(('layout', MMALayoutAttr(value=layout)))
    all_props.append(('shape', shape))
    all_props.append(('eltType', LdStMatrixEltTypeAttr(value=eltType)))
    return add_operation(
        name="nvvm.ldmatrix",
        result_type=res_type,
        operands=[ptr],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_MBarrierArriveDropExpectTxOp(
    *,
    res_type: Optional[IntegerType],
    addr: Value,
    txcount: Value,
    scope: MemScopeKind = MemScopeKind(0),
    relaxed: bool = False,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Optional[Value]:
    all_props = []
    all_props.append(('scope', MemScopeKindAttr(value=scope)))
    all_props.append(('relaxed', BoolAttr(value=relaxed)))
    return add_operation(
        name="nvvm.mbarrier.arrive_drop.expect_tx",
        result_type=res_type,
        operands=[addr, txcount],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_MBarrierArriveDropNocompleteOp(
    *,
    res_type: IntegerType,
    addr: Value,
    count: Value,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    return add_operation(
        name="nvvm.mbarrier.arrive_drop.nocomplete",
        result_type=res_type,
        operands=[addr, count],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_MBarrierArriveDropOp(
    *,
    res_type: Optional[IntegerType],
    addr: Value,
    count: Optional[Value] = None,
    scope: MemScopeKind = MemScopeKind(0),
    relaxed: bool = False,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Optional[Value]:
    all_props = []
    all_props.append(('scope', MemScopeKindAttr(value=scope)))
    all_props.append(('relaxed', BoolAttr(value=relaxed)))
    return add_operation(
        name="nvvm.mbarrier.arrive_drop",
        result_type=res_type,
        operands=[addr, *([] if count is None else [count])],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_MBarrierArriveExpectTxOp(
    *,
    res_type: Optional[IntegerType],
    addr: Value,
    txcount: Value,
    scope: MemScopeKind = MemScopeKind(0),
    relaxed: bool = False,
    predicate: Optional[Value] = None,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Optional[Value]:
    all_props = []
    all_props.append(('scope', MemScopeKindAttr(value=scope)))
    all_props.append(('relaxed', BoolAttr(value=relaxed)))
    return add_operation(
        name="nvvm.mbarrier.arrive.expect_tx",
        result_type=res_type,
        operands=[addr, txcount, *([] if predicate is None else [predicate])],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_MBarrierArriveNocompleteOp(
    *,
    res_type: IntegerType,
    addr: Value,
    count: Value,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    return add_operation(
        name="nvvm.mbarrier.arrive.nocomplete",
        result_type=res_type,
        operands=[addr, count],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_MBarrierArriveOp(
    *,
    res_type: Optional[IntegerType],
    addr: Value,
    count: Optional[Value] = None,
    scope: MemScopeKind = MemScopeKind(0),
    relaxed: bool = False,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Optional[Value]:
    all_props = []
    all_props.append(('scope', MemScopeKindAttr(value=scope)))
    all_props.append(('relaxed', BoolAttr(value=relaxed)))
    return add_operation(
        name="nvvm.mbarrier.arrive",
        result_type=res_type,
        operands=[addr, *([] if count is None else [count])],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_MBarrierCompleteTxOp(
    *,
    addr: Value,
    txcount: Value,
    scope: MemScopeKind = MemScopeKind(0),
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> None:
    all_props = []
    all_props.append(('scope', MemScopeKindAttr(value=scope)))
    return add_operation(
        name="nvvm.mbarrier.complete_tx",
        result_type=None,
        operands=[addr, txcount],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_MBarrierExpectTxOp(
    *,
    addr: Value,
    txcount: Value,
    scope: MemScopeKind = MemScopeKind(0),
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> None:
    all_props = []
    all_props.append(('scope', MemScopeKindAttr(value=scope)))
    return add_operation(
        name="nvvm.mbarrier.expect_tx",
        result_type=None,
        operands=[addr, txcount],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_MBarrierInitOp(
    *,
    addr: Value,
    count: Value,
    predicate: Optional[Value] = None,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> None:
    all_props = []
    return add_operation(
        name="nvvm.mbarrier.init",
        result_type=None,
        operands=[addr, count, *([] if predicate is None else [predicate])],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_MBarrierInvalOp(
    *,
    addr: Value,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> None:
    all_props = []
    return add_operation(
        name="nvvm.mbarrier.inval",
        result_type=None,
        operands=[addr],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_MBarrierTestWaitOp(
    *,
    res_type: IntegerType,
    addr: Value,
    stateOrPhase: Value,
    scope: MemScopeKind = MemScopeKind(0),
    relaxed: bool = False,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    all_props.append(('scope', MemScopeKindAttr(value=scope)))
    all_props.append(('relaxed', BoolAttr(value=relaxed)))
    return add_operation(
        name="nvvm.mbarrier.test.wait",
        result_type=res_type,
        operands=[addr, stateOrPhase],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_MBarrierTryWaitOp(
    *,
    res_type: IntegerType,
    addr: Value,
    stateOrPhase: Value,
    ticks: Optional[Value] = None,
    scope: MemScopeKind = MemScopeKind(0),
    relaxed: bool = False,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    all_props.append(('scope', MemScopeKindAttr(value=scope)))
    all_props.append(('relaxed', BoolAttr(value=relaxed)))
    return add_operation(
        name="nvvm.mbarrier.try_wait",
        result_type=res_type,
        operands=[addr, stateOrPhase, *([] if ticks is None else [ticks])],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_MBarrierTryWaitParityOp(
    *,
    addr: Value,
    phase: Value,
    ticks: Value,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> None:
    all_props = []
    return add_operation(
        name="nvvm.mbarrier.try_wait.parity",
        result_type=None,
        operands=[addr, phase, ticks],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_MapaOp(
    *,
    res_type: Type,
    a: Value,
    b: Value,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    return add_operation(
        name="nvvm.mapa",
        result_type=res_type,
        operands=[a, b],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_MatchSyncOp(
    *,
    res_type: Type,
    thread_mask: Value,
    val: Value,
    kind: MatchSyncKind,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    all_props.append(('kind', MatchSyncKindAttr(value=kind)))
    return add_operation(
        name="nvvm.match.sync",
        result_type=res_type,
        operands=[thread_mask, val],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_MembarOp(
    *,
    scope: MemScopeKind,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> None:
    all_props = []
    all_props.append(('scope', MemScopeKindAttr(value=scope)))
    return add_operation(
        name="nvvm.memory.barrier",
        result_type=None,
        operands=[],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_MmaBlockScaleOp(
    *,
    res_type: Type,
    shape: MMAShapeAttr,
    multiplicandAPtxType: Optional[MMATypes] = None,
    multiplicandBPtxType: Optional[MMATypes] = None,
    scaleVecSize: ScaleVecSize,
    blockScaleFormat: BlockScaleFormat,
    kind: MMABlockScaleKind,
    operandA: Sequence[Value],
    operandB: Sequence[Value],
    operandC: Sequence[Value],
    scaleAData: Value,
    byteIdA: Value,
    threadIdA: Value,
    scaleBData: Value,
    byteIdB: Value,
    threadIdB: Value,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    all_props.append(('shape', shape))
    if multiplicandAPtxType is not None:
        all_props.append(('multiplicandAPtxType', MMATypesAttr(value=multiplicandAPtxType)))
    if multiplicandBPtxType is not None:
        all_props.append(('multiplicandBPtxType', MMATypesAttr(value=multiplicandBPtxType)))
    all_props.append(('scaleVecSize', ScaleVecSizeAttr(value=scaleVecSize)))
    all_props.append(('blockScaleFormat', BlockScaleFormatAttr(value=blockScaleFormat)))
    all_props.append(('kind', MMABlockScaleKindAttr(value=kind)))
    all_props.append(('operandSegmentSizes',
                      DenseI32ArrayAttr([len(operandA), len(operandB), len(operandC), 1, 1, 1, 1,
                                         1, 1])))
    return add_operation(
        name="nvvm.mma.block_scale",
        result_type=res_type,
        operands=[*operandA, *operandB, *operandC, scaleAData, byteIdA, threadIdA, scaleBData,
                  byteIdB, threadIdB],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_MmaOp(
    *,
    res_type: Type,
    shape: MMAShapeAttr,
    b1Op: Optional[MMAB1Op] = None,
    intOverflowBehavior: Optional[MMAIntOverflow] = None,
    layoutA: MMALayout,
    layoutB: MMALayout,
    multiplicandAPtxType: Optional[MMATypes] = None,
    multiplicandBPtxType: Optional[MMATypes] = None,
    operandA: Sequence[Value],
    operandB: Sequence[Value],
    operandC: Sequence[Value],
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    all_props.append(('shape', shape))
    if b1Op is not None:
        all_props.append(('b1Op', MMAB1OpAttr(value=b1Op)))
    if intOverflowBehavior is not None:
        all_props.append(('intOverflowBehavior', MMAIntOverflowAttr(value=intOverflowBehavior)))
    all_props.append(('layoutA', MMALayoutAttr(value=layoutA)))
    all_props.append(('layoutB', MMALayoutAttr(value=layoutB)))
    if multiplicandAPtxType is not None:
        all_props.append(('multiplicandAPtxType', MMATypesAttr(value=multiplicandAPtxType)))
    if multiplicandBPtxType is not None:
        all_props.append(('multiplicandBPtxType', MMATypesAttr(value=multiplicandBPtxType)))
    all_props.append(('operandSegmentSizes',
                      DenseI32ArrayAttr([len(operandA), len(operandB), len(operandC)])))
    return add_operation(
        name="nvvm.mma.sync",
        result_type=res_type,
        operands=[*operandA, *operandB, *operandC],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_MmaSpBlockScaleOp(
    *,
    res_type: Type,
    shape: MMAShapeAttr,
    multiplicandAPtxType: Optional[MMATypes] = None,
    multiplicandBPtxType: Optional[MMATypes] = None,
    scaleVecSize: ScaleVecSize,
    blockScaleFormat: BlockScaleFormat,
    kind: MMABlockScaleKind,
    orderedMetadata: bool = False,
    operandA: Sequence[Value],
    operandB: Sequence[Value],
    operandC: Sequence[Value],
    sparseMetadata: Value,
    sparsitySelector: Value,
    scaleAData: Value,
    byteIdA: Value,
    threadIdA: Value,
    scaleBData: Value,
    byteIdB: Value,
    threadIdB: Value,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    all_props.append(('shape', shape))
    if multiplicandAPtxType is not None:
        all_props.append(('multiplicandAPtxType', MMATypesAttr(value=multiplicandAPtxType)))
    if multiplicandBPtxType is not None:
        all_props.append(('multiplicandBPtxType', MMATypesAttr(value=multiplicandBPtxType)))
    all_props.append(('scaleVecSize', ScaleVecSizeAttr(value=scaleVecSize)))
    all_props.append(('blockScaleFormat', BlockScaleFormatAttr(value=blockScaleFormat)))
    all_props.append(('kind', MMABlockScaleKindAttr(value=kind)))
    if orderedMetadata:
        all_props.append(('orderedMetadata', UnitAttr()))
    all_props.append(('operandSegmentSizes',
                      DenseI32ArrayAttr([len(operandA), len(operandB), len(operandC), 1, 1, 1, 1,
                                         1, 1, 1, 1])))
    return add_operation(
        name="nvvm.mma.sp.block_scale",
        result_type=res_type,
        operands=[*operandA, *operandB, *operandC, sparseMetadata, sparsitySelector, scaleAData,
                  byteIdA, threadIdA, scaleBData, byteIdB, threadIdB],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_MmaSpOp(
    *,
    res_type: Type,
    shape: MMAShapeAttr,
    intOverflowBehavior: Optional[MMAIntOverflow] = None,
    multiplicandAPtxType: Optional[MMATypes] = None,
    multiplicandBPtxType: Optional[MMATypes] = None,
    orderedMetadata: bool = False,
    kind: Optional[MMAKind] = None,
    operandA: Sequence[Value],
    operandB: Sequence[Value],
    operandC: Sequence[Value],
    sparseMetadata: Value,
    sparsitySelector: Value,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    all_props.append(('shape', shape))
    if intOverflowBehavior is not None:
        all_props.append(('intOverflowBehavior', MMAIntOverflowAttr(value=intOverflowBehavior)))
    if multiplicandAPtxType is not None:
        all_props.append(('multiplicandAPtxType', MMATypesAttr(value=multiplicandAPtxType)))
    if multiplicandBPtxType is not None:
        all_props.append(('multiplicandBPtxType', MMATypesAttr(value=multiplicandBPtxType)))
    if orderedMetadata:
        all_props.append(('orderedMetadata', UnitAttr()))
    if kind is not None:
        all_props.append(('kind', MMAKindAttr(value=kind)))
    all_props.append(('operandSegmentSizes',
                      DenseI32ArrayAttr([len(operandA), len(operandB), len(operandC), 1, 1])))
    return add_operation(
        name="nvvm.mma.sp.sync",
        result_type=res_type,
        operands=[*operandA, *operandB, *operandC, sparseMetadata, sparsitySelector],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_NanosleepOp(
    *,
    duration: Value,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> None:
    all_props = []
    return add_operation(
        name="nvvm.nanosleep",
        result_type=None,
        operands=[duration],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_PMEventOp(
    *,
    maskedEventId: Optional[int] = None,
    eventId: Optional[int] = None,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> None:
    all_props = []
    if maskedEventId is not None:
        all_props.append(('maskedEventId',
                          IntegerAttr.make(IntegerType.signless(16), maskedEventId)))
    if eventId is not None:
        all_props.append(('eventId', IntegerAttr.make(IntegerType.signless(32), eventId)))
    return add_operation(
        name="nvvm.pmevent",
        result_type=None,
        operands=[],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_PermuteOp(
    *,
    res_type: IntegerType,
    lo: Value,
    hi: Optional[Value] = None,
    selector: Value,
    mode: PermuteMode,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    all_props.append(('mode', PermuteModeAttr(value=mode)))
    return add_operation(
        name="nvvm.prmt",
        result_type=res_type,
        operands=[lo, *([] if hi is None else [hi]), selector],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_PrefetchOp(
    *,
    cacheLevel: Optional[PrefetchCacheLevel] = None,
    evictPriority: Optional[CacheEvictionPriority] = None,
    addr: Value,
    predicate: Optional[Value] = None,
    tensormap: bool = False,
    uniform: bool = False,
    in_param_space: bool = False,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> None:
    all_props = []
    if cacheLevel is not None:
        all_props.append(('cacheLevel', PrefetchCacheLevelAttr(value=cacheLevel)))
    if evictPriority is not None:
        all_props.append(('evictPriority', CacheEvictionPriorityAttr(value=evictPriority)))
    if tensormap:
        all_props.append(('tensormap', UnitAttr()))
    if uniform:
        all_props.append(('uniform', UnitAttr()))
    if in_param_space:
        all_props.append(('in_param_space', UnitAttr()))
    return add_operation(
        name="nvvm.prefetch",
        result_type=None,
        operands=[addr, *([] if predicate is None else [predicate])],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_RcpApproxFtzF32Op(
    *,
    res_type: FloatType,
    arg: Value,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    return add_operation(
        name="nvvm.rcp.approx.ftz.f",
        result_type=res_type,
        operands=[arg],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_ReduxOp(
    *,
    res_type: Type,
    val: Value,
    kind: ReduxKind,
    mask_and_clamp: Value,
    abs: bool = False,
    nan: bool = False,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    all_props.append(('kind', ReduxKindAttr(value=kind)))
    all_props.append(('abs', BoolAttr(value=abs)))
    all_props.append(('nan', BoolAttr(value=nan)))
    return add_operation(
        name="nvvm.redux.sync",
        result_type=res_type,
        operands=[val, mask_and_clamp],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_SetMaxRegisterOp(
    *,
    regCount: int,
    action: SetMaxRegisterAction,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> None:
    all_props = []
    all_props.append(('regCount', IntegerAttr.make(IntegerType.signless(32), regCount)))
    all_props.append(('action', SetMaxRegisterActionAttr(value=action)))
    return add_operation(
        name="nvvm.setmaxregister",
        result_type=None,
        operands=[],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_ShflOp(
    *,
    res_type: Type,
    thread_mask: Value,
    val: Value,
    offset: Value,
    mask_and_clamp: Value,
    kind: ShflKind,
    return_value_and_is_valid: Optional[bool] = None,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    all_props.append(('kind', ShflKindAttr(value=kind)))
    if return_value_and_is_valid is not None:
        all_props.append(('return_value_and_is_valid', UnitAttr()))
    return add_operation(
        name="nvvm.shfl.sync",
        result_type=res_type,
        operands=[thread_mask, val, offset, mask_and_clamp],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_SmDimOp(
    *,
    res_type: Type,
    range: Optional[llvm.ConstantRangeAttr] = None,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    if range is not None:
        all_props.append(('range', range))
    return add_operation(
        name="nvvm.read.ptx.sreg.nsmid",
        result_type=res_type,
        operands=[],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_SmIdOp(
    *,
    res_type: Type,
    range: Optional[llvm.ConstantRangeAttr] = None,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    if range is not None:
        all_props.append(('range', range))
    return add_operation(
        name="nvvm.read.ptx.sreg.smid",
        result_type=res_type,
        operands=[],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_StMatrixOp(
    *,
    ptr: Value,
    sources: Sequence[Value],
    layout: MMALayout,
    shape: LdStMatrixShapeAttr,
    eltType: LdStMatrixEltType,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> None:
    all_props = []
    all_props.append(('layout', MMALayoutAttr(value=layout)))
    all_props.append(('shape', shape))
    all_props.append(('eltType', LdStMatrixEltTypeAttr(value=eltType)))
    return add_operation(
        name="nvvm.stmatrix",
        result_type=None,
        operands=[ptr, *sources],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_SyncWarpOp(
    *,
    mask: Value,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> None:
    all_props = []
    return add_operation(
        name="nvvm.bar.warp.sync",
        result_type=None,
        operands=[mask],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_Tcgen05AllocOp(
    *,
    addr: Value,
    nCols: Value,
    group: CTAGroupKind = CTAGroupKind(0),
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> None:
    all_props = []
    all_props.append(('group', CTAGroupKindAttr(value=group)))
    return add_operation(
        name="nvvm.tcgen05.alloc",
        result_type=None,
        operands=[addr, nCols],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_Tcgen05CommitOp(
    *,
    addr: Value,
    multicastMask: Optional[Value] = None,
    group: CTAGroupKind = CTAGroupKind(0),
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> None:
    all_props = []
    all_props.append(('group', CTAGroupKindAttr(value=group)))
    return add_operation(
        name="nvvm.tcgen05.commit",
        result_type=None,
        operands=[addr, *([] if multicastMask is None else [multicastMask])],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_Tcgen05CpOp(
    *,
    shape: Tcgen05CpShape,
    group: CTAGroupKind = CTAGroupKind(0),
    multicast: Tcgen05CpMulticast = Tcgen05CpMulticast(0),
    srcFormat: Optional[Tcgen05CpSrcFormat] = None,
    taddr: Value,
    smem_desc: Value,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> None:
    all_props = []
    all_props.append(('shape', Tcgen05CpShapeAttr(value=shape)))
    all_props.append(('group', CTAGroupKindAttr(value=group)))
    all_props.append(('multicast', Tcgen05CpMulticastAttr(value=multicast)))
    if srcFormat is not None:
        all_props.append(('srcFormat', Tcgen05CpSrcFormatAttr(value=srcFormat)))
    return add_operation(
        name="nvvm.tcgen05.cp",
        result_type=None,
        operands=[taddr, smem_desc],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_Tcgen05DeallocOp(
    *,
    taddr: Value,
    nCols: Value,
    group: CTAGroupKind = CTAGroupKind(0),
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> None:
    all_props = []
    all_props.append(('group', CTAGroupKindAttr(value=group)))
    return add_operation(
        name="nvvm.tcgen05.dealloc",
        result_type=None,
        operands=[taddr, nCols],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_Tcgen05FenceOp(
    *,
    kind: Tcgen05FenceKind,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> None:
    all_props = []
    all_props.append(('kind', Tcgen05FenceKindAttr(value=kind)))
    return add_operation(
        name="nvvm.tcgen05.fence",
        result_type=None,
        operands=[],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_Tcgen05LdOp(
    *,
    res_type: Type,
    pack: bool = False,
    shape: Tcgen05LdStShape,
    tmemAddr: Value,
    offset: Optional[Value] = None,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    if pack:
        all_props.append(('pack', UnitAttr()))
    all_props.append(('shape', Tcgen05LdStShapeAttr(value=shape)))
    return add_operation(
        name="nvvm.tcgen05.ld",
        result_type=res_type,
        operands=[tmemAddr, *([] if offset is None else [offset])],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_Tcgen05MMABlockScaleOp(
    *,
    kind: MMABlockScaleKind,
    ctaGroup: CTAGroupKind,
    blockScale: Tcgen05MMABlockScale = Tcgen05MMABlockScale(0),
    collectorOp: Tcgen05MMACollectorOp = Tcgen05MMACollectorOp(0),
    matrixD: Value,
    matrixA: Value,
    matrixB: Value,
    idesc: Value,
    enableInputD: Value,
    scaleA: Value,
    scaleB: Value,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> None:
    all_props = []
    all_props.append(('kind', MMABlockScaleKindAttr(value=kind)))
    all_props.append(('ctaGroup', CTAGroupKindAttr(value=ctaGroup)))
    all_props.append(('blockScale', Tcgen05MMABlockScaleAttr(value=blockScale)))
    all_props.append(('collectorOp', Tcgen05MMACollectorOpAttr(value=collectorOp)))
    return add_operation(
        name="nvvm.tcgen05.mma.block_scale",
        result_type=None,
        operands=[matrixD, matrixA, matrixB, idesc, enableInputD, scaleA, scaleB],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_Tcgen05MMAOp(
    *,
    kind: Tcgen05MMAKind,
    ctaGroup: CTAGroupKind,
    collectorOp: Tcgen05MMACollectorOp = Tcgen05MMACollectorOp(0),
    aShift: bool = False,
    matrixD: Value,
    matrixA: Value,
    matrixB: Value,
    idesc: Value,
    enableInputD: Value,
    scaleInputD: Optional[Value] = None,
    disableOutputLane: Optional[Value] = None,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> None:
    all_props = []
    all_props.append(('kind', Tcgen05MMAKindAttr(value=kind)))
    all_props.append(('ctaGroup', CTAGroupKindAttr(value=ctaGroup)))
    all_props.append(('collectorOp', Tcgen05MMACollectorOpAttr(value=collectorOp)))
    if aShift:
        all_props.append(('aShift', UnitAttr()))
    all_props.append(('operandSegmentSizes',
                      DenseI32ArrayAttr([1, 1, 1, 1, 1, int(scaleInputD is not None),
                                         int(disableOutputLane is not None)])))
    return add_operation(
        name="nvvm.tcgen05.mma",
        result_type=None,
        operands=[matrixD, matrixA, matrixB, idesc, enableInputD,
                  *([] if scaleInputD is None else [scaleInputD]),
                  *([] if disableOutputLane is None else [disableOutputLane])],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_Tcgen05MMASparseBlockScaleOp(
    *,
    kind: MMABlockScaleKind,
    ctaGroup: CTAGroupKind,
    blockScale: Tcgen05MMABlockScale = Tcgen05MMABlockScale(0),
    collectorOp: Tcgen05MMACollectorOp = Tcgen05MMACollectorOp(0),
    matrixD: Value,
    matrixA: Value,
    matrixB: Value,
    idesc: Value,
    enableInputD: Value,
    sparseMetadata: Value,
    scaleA: Value,
    scaleB: Value,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> None:
    all_props = []
    all_props.append(('kind', MMABlockScaleKindAttr(value=kind)))
    all_props.append(('ctaGroup', CTAGroupKindAttr(value=ctaGroup)))
    all_props.append(('blockScale', Tcgen05MMABlockScaleAttr(value=blockScale)))
    all_props.append(('collectorOp', Tcgen05MMACollectorOpAttr(value=collectorOp)))
    return add_operation(
        name="nvvm.tcgen05.mma.sp.block_scale",
        result_type=None,
        operands=[matrixD, matrixA, matrixB, idesc, enableInputD, sparseMetadata, scaleA, scaleB],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_Tcgen05MMASparseOp(
    *,
    kind: Tcgen05MMAKind,
    ctaGroup: CTAGroupKind,
    collectorOp: Tcgen05MMACollectorOp = Tcgen05MMACollectorOp(0),
    aShift: bool = False,
    matrixD: Value,
    matrixA: Value,
    matrixB: Value,
    idesc: Value,
    enableInputD: Value,
    sparseMetadata: Value,
    scaleInputD: Optional[Value] = None,
    disableOutputLane: Optional[Value] = None,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> None:
    all_props = []
    all_props.append(('kind', Tcgen05MMAKindAttr(value=kind)))
    all_props.append(('ctaGroup', CTAGroupKindAttr(value=ctaGroup)))
    all_props.append(('collectorOp', Tcgen05MMACollectorOpAttr(value=collectorOp)))
    if aShift:
        all_props.append(('aShift', UnitAttr()))
    all_props.append(('operandSegmentSizes',
                      DenseI32ArrayAttr([1, 1, 1, 1, 1, 1, int(scaleInputD is not None),
                                         int(disableOutputLane is not None)])))
    return add_operation(
        name="nvvm.tcgen05.mma.sp",
        result_type=None,
        operands=[matrixD, matrixA, matrixB, idesc, enableInputD, sparseMetadata,
                  *([] if scaleInputD is None else [scaleInputD]),
                  *([] if disableOutputLane is None else [disableOutputLane])],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_Tcgen05MMAWsOp(
    *,
    kind: Tcgen05MMAKind,
    collectorBBuffer: Tcgen05MMACollectorBBuffer = Tcgen05MMACollectorBBuffer(0),
    collectorOp: Tcgen05MMACollectorOp = Tcgen05MMACollectorOp(0),
    matrixD: Value,
    matrixA: Value,
    matrixB: Value,
    idesc: Value,
    enableInputD: Value,
    zeroColMask: Optional[Value] = None,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> None:
    all_props = []
    all_props.append(('kind', Tcgen05MMAKindAttr(value=kind)))
    all_props.append(('collectorBBuffer', Tcgen05MMACollectorBBufferAttr(value=collectorBBuffer)))
    all_props.append(('collectorOp', Tcgen05MMACollectorOpAttr(value=collectorOp)))
    return add_operation(
        name="nvvm.tcgen05.mma.ws",
        result_type=None,
        operands=[matrixD, matrixA, matrixB, idesc, enableInputD,
                  *([] if zeroColMask is None else [zeroColMask])],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_Tcgen05MMAWsSparseOp(
    *,
    kind: Tcgen05MMAKind,
    collectorBBuffer: Tcgen05MMACollectorBBuffer = Tcgen05MMACollectorBBuffer(0),
    collectorOp: Tcgen05MMACollectorOp = Tcgen05MMACollectorOp(0),
    matrixD: Value,
    matrixA: Value,
    matrixB: Value,
    idesc: Value,
    enableInputD: Value,
    sparseMetadata: Value,
    zeroColMask: Optional[Value] = None,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> None:
    all_props = []
    all_props.append(('kind', Tcgen05MMAKindAttr(value=kind)))
    all_props.append(('collectorBBuffer', Tcgen05MMACollectorBBufferAttr(value=collectorBBuffer)))
    all_props.append(('collectorOp', Tcgen05MMACollectorOpAttr(value=collectorOp)))
    return add_operation(
        name="nvvm.tcgen05.mma.ws.sp",
        result_type=None,
        operands=[matrixD, matrixA, matrixB, idesc, enableInputD, sparseMetadata,
                  *([] if zeroColMask is None else [zeroColMask])],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_Tcgen05MmaSmemDescOp(
    *,
    res_type: IntegerType,
    startAddr: Value,
    leadingDimOffset: Value,
    strideDimOffset: Value,
    baseOffset: Value,
    leadingDimMode: Value,
    swizzleMode: Value,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    return add_operation(
        name="nvvm.tcgen05.mma_smem_desc",
        result_type=res_type,
        operands=[startAddr, leadingDimOffset, strideDimOffset, baseOffset, leadingDimMode,
                  swizzleMode],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_Tcgen05RelinquishAllocPermitOp(
    *,
    group: CTAGroupKind = CTAGroupKind(0),
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> None:
    all_props = []
    all_props.append(('group', CTAGroupKindAttr(value=group)))
    return add_operation(
        name="nvvm.tcgen05.relinquish_alloc_permit",
        result_type=None,
        operands=[],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_Tcgen05ShiftOp(
    *,
    taddr: Value,
    group: CTAGroupKind = CTAGroupKind(0),
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> None:
    all_props = []
    all_props.append(('group', CTAGroupKindAttr(value=group)))
    return add_operation(
        name="nvvm.tcgen05.shift",
        result_type=None,
        operands=[taddr],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_Tcgen05StOp(
    *,
    unpack: bool = False,
    shape: Tcgen05LdStShape,
    tmemAddr: Value,
    val: Value,
    offset: Optional[Value] = None,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> None:
    all_props = []
    if unpack:
        all_props.append(('unpack', UnitAttr()))
    all_props.append(('shape', Tcgen05LdStShapeAttr(value=shape)))
    return add_operation(
        name="nvvm.tcgen05.st",
        result_type=None,
        operands=[tmemAddr, val, *([] if offset is None else [offset])],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_Tcgen05WaitOp(
    *,
    kind: Tcgen05WaitKind,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> None:
    all_props = []
    all_props.append(('kind', Tcgen05WaitKindAttr(value=kind)))
    return add_operation(
        name="nvvm.tcgen05.wait",
        result_type=None,
        operands=[],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_ThreadIdXOp(
    *,
    res_type: Type,
    range: Optional[llvm.ConstantRangeAttr] = None,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    if range is not None:
        all_props.append(('range', range))
    return add_operation(
        name="nvvm.read.ptx.sreg.tid.x",
        result_type=res_type,
        operands=[],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_ThreadIdYOp(
    *,
    res_type: Type,
    range: Optional[llvm.ConstantRangeAttr] = None,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    if range is not None:
        all_props.append(('range', range))
    return add_operation(
        name="nvvm.read.ptx.sreg.tid.y",
        result_type=res_type,
        operands=[],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_ThreadIdZOp(
    *,
    res_type: Type,
    range: Optional[llvm.ConstantRangeAttr] = None,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    if range is not None:
        all_props.append(('range', range))
    return add_operation(
        name="nvvm.read.ptx.sreg.tid.z",
        result_type=res_type,
        operands=[],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_TotalSmemSize(
    *,
    res_type: Type,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    return add_operation(
        name="nvvm.read.ptx.sreg.total.smem.size",
        result_type=res_type,
        operands=[],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_VoteSyncOp(
    *,
    res_type: Type,
    mask: Value,
    pred: Value,
    kind: VoteSyncKind,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    all_props.append(('kind', VoteSyncKindAttr(value=kind)))
    return add_operation(
        name="nvvm.vote.sync",
        result_type=res_type,
        operands=[mask, pred],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_WMMALoadOp(
    *,
    res_type: Type,
    ptr: Value,
    stride: Value,
    m: int,
    n: int,
    k: int,
    layout: MMALayout,
    eltype: MMATypes,
    frag: MMAFrag,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    all_props.append(('m', IntegerAttr.make(IntegerType.signless(32), m)))
    all_props.append(('n', IntegerAttr.make(IntegerType.signless(32), n)))
    all_props.append(('k', IntegerAttr.make(IntegerType.signless(32), k)))
    all_props.append(('layout', MMALayoutAttr(value=layout)))
    all_props.append(('eltype', MMATypesAttr(value=eltype)))
    all_props.append(('frag', MMAFragAttr(value=frag)))
    return add_operation(
        name="nvvm.wmma.load",
        result_type=res_type,
        operands=[ptr, stride],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_WMMAMmaOp(
    *,
    res_type: Type,
    m: int,
    n: int,
    k: int,
    layoutA: MMALayout,
    layoutB: MMALayout,
    eltypeA: MMATypes,
    eltypeB: MMATypes,
    args: Sequence[Value],
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    all_props.append(('m', IntegerAttr.make(IntegerType.signless(32), m)))
    all_props.append(('n', IntegerAttr.make(IntegerType.signless(32), n)))
    all_props.append(('k', IntegerAttr.make(IntegerType.signless(32), k)))
    all_props.append(('layoutA', MMALayoutAttr(value=layoutA)))
    all_props.append(('layoutB', MMALayoutAttr(value=layoutB)))
    all_props.append(('eltypeA', MMATypesAttr(value=eltypeA)))
    all_props.append(('eltypeB', MMATypesAttr(value=eltypeB)))
    return add_operation(
        name="nvvm.wmma.mma",
        result_type=res_type,
        operands=list(args),
        properties=all_props,
        attributes=extra_attributes,
    )


def add_WMMAStoreOp(
    *,
    ptr: Value,
    m: int,
    n: int,
    k: int,
    layout: MMALayout,
    eltype: MMATypes,
    args: Sequence[Value],
    stride: Value,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> None:
    all_props = []
    all_props.append(('m', IntegerAttr.make(IntegerType.signless(32), m)))
    all_props.append(('n', IntegerAttr.make(IntegerType.signless(32), n)))
    all_props.append(('k', IntegerAttr.make(IntegerType.signless(32), k)))
    all_props.append(('layout', MMALayoutAttr(value=layout)))
    all_props.append(('eltype', MMATypesAttr(value=eltype)))
    return add_operation(
        name="nvvm.wmma.store",
        result_type=None,
        operands=[ptr, *args, stride],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_WarpDimOp(
    *,
    res_type: Type,
    range: Optional[llvm.ConstantRangeAttr] = None,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    if range is not None:
        all_props.append(('range', range))
    return add_operation(
        name="nvvm.read.ptx.sreg.nwarpid",
        result_type=res_type,
        operands=[],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_WarpIdOp(
    *,
    res_type: Type,
    range: Optional[llvm.ConstantRangeAttr] = None,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    if range is not None:
        all_props.append(('range', range))
    return add_operation(
        name="nvvm.read.ptx.sreg.warpid",
        result_type=res_type,
        operands=[],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_WarpSizeOp(
    *,
    res_type: Type,
    range: Optional[llvm.ConstantRangeAttr] = None,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    if range is not None:
        all_props.append(('range', range))
    return add_operation(
        name="nvvm.read.ptx.sreg.warpsize",
        result_type=res_type,
        operands=[],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_WgmmaFenceAlignedOp(
    *,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> None:
    all_props = []
    return add_operation(
        name="nvvm.wgmma.fence.aligned",
        result_type=None,
        operands=[],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_WgmmaGroupSyncAlignedOp(
    *,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> None:
    all_props = []
    return add_operation(
        name="nvvm.wgmma.commit.group.sync.aligned",
        result_type=None,
        operands=[],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_WgmmaMmaAsyncOp(
    *,
    results_type: Type,
    inouts: Value,
    descriptorA: Value,
    descriptorB: Value,
    shape: MMAShapeAttr,
    typeA: WGMMATypes,
    typeB: WGMMATypes,
    typeD: WGMMATypes,
    scaleD: WGMMAScaleOut,
    scaleA: WGMMAScaleIn,
    scaleB: WGMMAScaleIn,
    layoutA: MMALayout,
    layoutB: MMALayout,
    satfinite: Optional[MMAIntOverflow] = None,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    all_props.append(('shape', shape))
    all_props.append(('typeA', WGMMATypesAttr(value=typeA)))
    all_props.append(('typeB', WGMMATypesAttr(value=typeB)))
    all_props.append(('typeD', WGMMATypesAttr(value=typeD)))
    all_props.append(('scaleD', WGMMAScaleOutAttr(value=scaleD)))
    all_props.append(('scaleA', WGMMAScaleInAttr(value=scaleA)))
    all_props.append(('scaleB', WGMMAScaleInAttr(value=scaleB)))
    all_props.append(('layoutA', MMALayoutAttr(value=layoutA)))
    all_props.append(('layoutB', MMALayoutAttr(value=layoutB)))
    if satfinite is not None:
        all_props.append(('satfinite', MMAIntOverflowAttr(value=satfinite)))
    return add_operation(
        name="nvvm.wgmma.mma_async",
        result_type=results_type,
        operands=[inouts, descriptorA, descriptorB],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_WgmmaWaitGroupSyncOp(
    *,
    group: int,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> None:
    all_props = []
    all_props.append(('group', IntegerAttr.make(IntegerType.signless(64), group)))
    return add_operation(
        name="nvvm.wgmma.wait.group.sync.aligned",
        result_type=None,
        operands=[],
        properties=all_props,
        attributes=extra_attributes,
    )
