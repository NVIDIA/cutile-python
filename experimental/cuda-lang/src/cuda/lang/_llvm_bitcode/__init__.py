# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from ._enums import (
    Binop,
    CallingConvention,
    Cast,
    CmpPredicate,
    FloatKind,
    FPClass,
    Linkage,
    Unop,
)

from ._builder import (
    AnonStructType,
    BitcodeBuilder,
    FloatType,
    Function,
    FunctionType,
    IntegerType,
    Metadata,
    MetadataTable,
    PointerType,
    Type,
    TypeTable,
    Value,
)

DATALAYOUT_PTX = ("e-p:64:64:64-p3:32:32:32"
                  "-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-i128:128:128"
                  "-f32:32:32-f64:64:64"
                  "-v16:16:16-v32:32:32-v64:64:64-v128:128:128"
                  "-n16:32:64-a:8:8")


__all__ = [
    "Binop",
    "CallingConvention",
    "Cast",
    "CmpPredicate",
    "FloatKind",
    "Linkage",
    "AnonStructType",
    "BitcodeBuilder",
    "FloatType",
    "FPClass",
    "Function",
    "FunctionType",
    "IntegerType",
    "Metadata",
    "MetadataTable",
    "PointerType",
    "Type",
    "TypeTable",
    "Unop",
    "Value",
]
