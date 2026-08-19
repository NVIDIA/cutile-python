# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from task_scheduling_test_requirements import cuda_lang as cl

from cuda.lang._compile import get_compute_capability
from cuda.lang.compilation import ArrayConstraint


def make_symbolic_tensor(shape, dtype):
    if isinstance(shape, int):
        shape = [shape]
    return ArrayConstraint(
        dtype=dtype,
        ndim=len(shape),
        index_dtype=cl.int32,
        stride_lower_bound_incl=0,
        alias_groups=(),
        may_alias_internally=False,
    )


def require_blackwell_cc100():
    compute_capability = get_compute_capability()
    return pytest.mark.skipif(
        compute_capability.major != 10,
        reason="feature requires Blackwell with compute capability 100",
    )
