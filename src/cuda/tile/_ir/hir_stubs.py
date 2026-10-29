# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from cuda.tile._execution import stub


def hir_stub(func):
    """Declare a compiler-internal helper shared by all compiled execution spaces."""
    return stub(func, host=False, tile=True, simt=True, compiled_host=True)


@hir_stub
def enter_context(manager, /): ...


@hir_stub
def pop_context(): ...


@hir_stub
def is_contained_in(x, y, /): ...  # "return x in y"


@hir_stub
def is_not_contained_in(x, y, /): ...  # return "x not in y"
