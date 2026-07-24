# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import cuda.lang as cl
from cuda.lang._ir.ir import IRContext
from cuda.lang._passes.flatten_cfg import flatten_cfg

from ..util import (
    filecheck,
    get_source,
    make_symbolic_scalar,
    make_symbolic_tensor,
    get_ir,
)


def test_flatten_ifelse():
    def test_kernel(A):
        if A[0]:
            A[0] = 1
        else:
            A[0] = 0

    # BEFORE: $[[ITEM:[0-9]+]]: int32 = load_pointer
    # BEFORE: $[[ITEM_CASTED:[0-9]+]]: bool_ = tile_astype(x=$[[ITEM]], rounding_mode=None)
    # BEFORE: if(cond=$[[ITEM_CASTED]])
    # BEFORE: then
    # BEFORE:   store_pointer
    # BEFORE:   yield
    # BEFORE: else
    # BEFORE:   store_pointer
    # BEFORE:   yield
    # BEFORE: return
    body = get_ir(test_kernel, [make_symbolic_tensor((1,), cl.int32)])
    filecheck(str(body), get_source(), ("BEFORE",))

    # AFTER: ^entry({{.+}}):
    # AFTER:   $[[ITEM:[0-9]+]]: int32 = load_pointer
    # AFTER:   $[[ITEM_CASTED:[0-9]+]]: bool_ = tile_astype(x=$[[ITEM]], rounding_mode=None)
    # AFTER:   cond_br $[[ITEM_CASTED]]: bool_ ^then() ^else()
    # AFTER: ^then():
    # AFTER:   store_pointer
    # AFTER:   br ^phi()
    # AFTER: ^else():
    # AFTER:   store_pointer
    # AFTER:   br ^phi()
    # AFTER: ^phi():
    # AFTER:   return
    ctx = IRContext()
    body = flatten_cfg(body, ctx)
    filecheck(str(body), get_source(), ("AFTER",))


def test_flatten_ifelse_nested():
    def test_kernel(cond1, cond2):
        # CHECK: cond_br cond1{{.+}} ^then() ^else()
        if cond1:
            # CHECK: cond_br cond2{{.+}} ^then.1() ^else.1()
            if cond2:
                # CHECK: br ^phi.1()
                pass
            else:
                # CHECK: br ^phi.1()
                pass
            # CHECK: ^phi.1():
            # CHECK: br ^phi()
        else:
            # CHECK: cond_br cond2{{.+}} ^then.2() ^else.2()
            if cond2:
                # CHECK: br ^phi.2()
                pass
            else:
                # CHECK: br ^phi.2()
                pass
            # CHECK: ^phi.2():
            # CHECK: br ^phi()
        # CHECK: ^phi():
        # CHECK: return

    body = get_ir(test_kernel, [make_symbolic_scalar(cl.bool_), make_symbolic_scalar(cl.bool_)])
    ctx = IRContext()
    body = flatten_cfg(body, ctx)
    filecheck(str(body), get_source())


def test_flatten_ifelse_phi_merge():
    def test_kernel(A):
        # CHECK: $[[ITEM:[0-9]+]]: int32 = load_pointer
        # CHECK: $[[ITEM_BOOL:[0-9]+]]: bool_ = tile_astype(x=$[[ITEM]], rounding_mode=None)
        # CHECK: cond_br $[[ITEM_BOOL]]: bool_ ^then() ^else()
        if A[0]:
            # CHECK: $[[ITEM_1:[0-9]+]]: int32 = load_pointer
            # CHECK: x{{.*}}: int32 = $[[ITEM_1]]
            # CHECK: br ^phi(x{{.*}}: int32)
            x = A[1]
        else:
            # CHECK: $[[ITEM_2:[0-9]+]]: int32 = load_pointer
            # CHECK: x{{.*}}: int32 = $[[ITEM_2]]
            # CHECK: br ^phi(x{{.*}}: int32)
            x = A[2]

        # CHECK: ^phi($[[RESULT:[0-9]+]]: int32):
        # CHECK: [[RESULT1:x\.[0-9]+]]: int32 = $[[RESULT]]
        # CHECK: store_pointer
        A[0] = x

    body = get_ir(test_kernel, [make_symbolic_tensor((3,), cl.int32)])
    ctx = IRContext()
    body = flatten_cfg(body, ctx)
    filecheck(str(body), get_source())


def test_flatten_ifelse_sibling_blocks():
    def test_kernel(cond1, cond2):
        # CHECK: cond_br cond1{{.+}} ^then() ^else()
        if cond1:
            # check: br ^phi()
            pass
        else:
            # check: br ^phi()
            pass
        # CHECK: ^phi():

        # CHECK: cond_br cond2{{.+}} ^then.1() ^else.1()
        if cond2:
            # check: br ^phi.1()
            pass
        else:
            # check: br ^phi.1()
            pass
        # CHECK: ^phi.1():

        # CHECK: return

    body = get_ir(test_kernel, [make_symbolic_scalar(cl.bool_), make_symbolic_scalar(cl.bool_)])
    ctx = IRContext()
    body = flatten_cfg(body, ctx)
    filecheck(str(body), get_source())
