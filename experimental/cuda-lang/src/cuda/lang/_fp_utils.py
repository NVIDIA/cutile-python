# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import math

from . import _datatype as datatype

_FLOAT_SMALLEST_NORMAL = {
    datatype.float16: 2**-14,
    datatype.bfloat16: 2**-126,
    datatype.float32: 2**-126,
    datatype.float64: 2**-1022,
}


def isnormal(x, dtype: datatype.DType):
    normal_min = _FLOAT_SMALLEST_NORMAL[dtype]
    return math.isfinite(x) and x != 0.0 and math.fabs(x) >= normal_min


__all__ = ("isnormal",)
