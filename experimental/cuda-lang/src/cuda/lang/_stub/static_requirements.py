# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import cuda.lang as cl

from cuda.lang._execution import function


@function()
def require_constant_bool(var):
    cl.static_assert(
        var in (True, False),
        f"Expected constant of type bool but got {var}",
    )


@function()
def require_constant_enum(var, enum):
    cl.static_assert(
        var in tuple(enum),
        f"Expected enum constant of type {enum.__name__} but got {var}",
    )


@function()
def require_constant_int(var):
    cl.static_assert(
        isinstance(var, int),
        f"Expected constant of type int but got {var}",
    )
