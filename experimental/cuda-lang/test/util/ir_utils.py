# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from cuda.lang._ir.ir import IRContext
from cuda.lang._compile import get_function_hir, get_function_ir
from cuda.lang.compilation import ArrayConstraint, KernelSignature, ScalarConstraint
from cuda.lang import int32


def get_ir(func, constraints):
    ctx = IRContext(log_ir_on_error=False)
    func_hir = get_function_hir(func, entry_point=True)
    return get_function_ir(func_hir, KernelSignature(constraints), ctx)


def make_symbolic_scalar(dtype):
    return ScalarConstraint(dtype=dtype)


def make_symbolic_tensor(shape, dtype):
    if isinstance(shape, int):
        shape = [shape]
    return ArrayConstraint(
        dtype=dtype,
        ndim=len(shape),
        index_dtype=int32,
        stride_lower_bound_incl=0,
        alias_groups=(),
        may_alias_internally=False,
    )
