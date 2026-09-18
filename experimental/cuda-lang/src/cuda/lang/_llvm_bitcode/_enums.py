# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0


import enum
from . import _codes as codes


class CallingConvention(enum.Enum):
    C = 0
    PTX_Kernel = 71
    PTX_Device = 72


class Linkage(enum.Enum):
    External = 0
    Internal = 3


# Matches `enum BinaryOpcodes` in LLVMBitCodes.h
class Binop(enum.Enum):
    ADD = 0
    SUB = 1
    MUL = 2
    UDIV = 3
    SDIV = 4
    UREM = 5
    SREM = 6
    SHL = 7
    LSHR = 8
    ASHR = 9
    AND = 10
    OR = 11
    XOR = 12


# Matches `enum UnaryOpcodes` in LLVMBitCodes.h
class Unop(enum.Enum):
    FNEG = 0


class CmpPredicate(enum.Enum):
    FCMP_FALSE = 0
    FCMP_OEQ = 1
    FCMP_OGT = 2
    FCMP_OGE = 3
    FCMP_OLT = 4
    FCMP_OLE = 5
    FCMP_ONE = 6
    FCMP_ORD = 7
    FCMP_UNO = 8
    FCMP_UEQ = 9
    FCMP_UGT = 10
    FCMP_UGE = 11
    FCMP_ULT = 12
    FCMP_ULE = 13
    FCMP_UNE = 14
    FCMP_TRUE = 15
    ICMP_EQ = 32
    ICMP_NE = 33
    ICMP_UGT = 34
    ICMP_UGE = 35
    ICMP_ULT = 36
    ICMP_ULE = 37
    ICMP_SGT = 38
    ICMP_SGE = 39
    ICMP_SLT = 40
    ICMP_SLE = 41


# Matches `enum CastOpcodes` in LLVMBitCodes.h
class Cast(enum.Enum):
    TRUNC = 0
    ZEXT = 1
    SEXT = 2
    FPTOUI = 3
    FPTOSI = 4
    UITOFP = 5
    SITOFP = 6
    FPTRUNC = 7
    FPEXT = 8
    PTRTOINT = 9
    INTTOPTR = 10
    BITCAST = 11
    ADDRSPACECAST = 12
    PTRTOADDR = 13


class FloatKind(enum.Enum):
    f16 = codes.TYPE_CODE_HALF
    bf16 = codes.TYPE_CODE_BFLOAT
    f32 = codes.TYPE_CODE_FLOAT
    f64 = codes.TYPE_CODE_DOUBLE


class FPClass(enum.IntEnum):
    SNan = 0x0001
    QNan = 0x0002
    NegInf = 0x0004
    NegNormal = 0x0008
    NegSubnormal = 0x0010
    NegZero = 0x0020
    PosZero = 0x0040
    PosSubnormal = 0x0080
    PosNormal = 0x0100
    PosInf = 0x0200

    Nan = SNan | QNan
    Inf = PosInf | NegInf
    Normal = PosNormal | NegNormal
    Subnormal = PosSubnormal | NegSubnormal
    Zero = PosZero | NegZero
    PosFinite = PosNormal | PosSubnormal | PosZero
    NegFinite = NegNormal | NegSubnormal | NegZero
    Finite = PosFinite | NegFinite
    Positive = PosFinite | PosInf
    Negative = NegFinite | NegInf

    AllFlags = Nan | Inf | Finite
