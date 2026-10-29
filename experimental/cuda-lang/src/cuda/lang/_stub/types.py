# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from cuda.tile._cext import OpaqueValue, Pointer, Scalar, Vector
from cuda.lang._execution import simt_stub


__all__ = ["OpaqueValue", "Pointer", "Scalar", "Vector", "pointer_add", "pointer_sub"]


@simt_stub(static_eval_ok=True)
def pointer_add(ptr, other):
    ...


@simt_stub(static_eval_ok=True)
def pointer_sub(ptr, other):
    ...
