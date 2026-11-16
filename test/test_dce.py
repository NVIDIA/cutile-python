# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import cuda.tile as ct
from cuda.tile._cext import CallingConvention
from cuda.tile._compile import compile_tile
from cuda.tile._ir.control_flow_ops import Loop, Continue, Break, MakeDummy
from cuda.tile._ir.ops import TileLoad
from cuda.tile._ir import ir
from cuda.tile.compilation import ArrayConstraint, KernelSignature


def _get_defining_op(var, root_block: ir.Block):
    for op in root_block.traverse():
        if var.name in (rv.name for rv in op.result_vars):
            return op
    return None


def get_ir(func) -> ir.Block:
    x = ArrayConstraint(dtype=ct.int32, ndim=1, index_dtype=ct.int32, stride_lower_bound_incl=0,
                        alias_groups=(), may_alias_internally=False)
    sig = KernelSignature([x], CallingConvention.cutile_python_v1(), symbol="kernel")
    [body] = compile_tile(func, [sig], return_final_ir=True, return_cubin=False).final_ir
    return body


def test_unused_loop_var():
    def kernel(x):
        a = 0   # can be pruned
        t = ct.load(x, (0,), (1,))
        for i in range(10):
            a = a + 1   # can be pruned
            t = t + 1
        ct.store(x, (1,), t)

    func_body = get_ir(kernel)
    loop, = [op for op in func_body if isinstance(op, Loop)]
    assert len(loop.body_vars) == 1
    assert loop.body_vars[0].name.startswith("t")


def test_unused_body_var():
    def kernel(x):
        t = ct.load(x, (0,), (1,))   # can be pruned
        i = 0
        while True:
            t = ct.ones((1,), x.dtype)
            if i > ct.bid(0):
                break
            t = t + 1   # can be pruned
            i = i + 1
        ct.store(x, (1,), t)

    func_body = get_ir(kernel)
    loop, = [op for op in func_body if isinstance(op, Loop)]

    # The initial variable's defining op should be replaced with a dummy MakeDummy
    t_idx = next(i for i, v in enumerate(loop.body.params) if v.name.startswith("t"))
    assert isinstance(_get_defining_op(loop.initial_values[t_idx], func_body), MakeDummy)

    # The yielded variable's defining op should also be replaced with a dummy MakeDummy
    continue_op = loop.body[-1]
    assert isinstance(continue_op, Continue)
    assert isinstance(_get_defining_op(continue_op.values[t_idx], func_body), MakeDummy)


def test_dead_acquire_load_is_kept():
    def kernel(x):
        ct.load(x, (0,), (1,), memory_order=ct.MemoryOrder.ACQUIRE,
                memory_scope=ct.MemoryScope.DEVICE)

    func_body = get_ir(kernel)
    assert any(isinstance(op, TileLoad) for op in func_body.traverse())


def test_dead_weak_load_is_removed():
    def kernel(x):
        ct.load(x, (0,), (1,))

    func_body = get_ir(kernel)
    assert not any(isinstance(op, TileLoad) for op in func_body.traverse())


def test_unused_result_var():
    def kernel(x):
        t = ct.load(x, (0,), (1,))
        i = 0
        while True:
            ct.store(x, (1,), t)
            t = t + 1
            if i > ct.bid(0):
                t = t + 2  # can be pruned
                break
            i = i + 1

    func_body = get_ir(kernel)
    loop, = [op for op in func_body if isinstance(op, Loop)]

    # The value yielded by "break" should be replaced with a dummy MakeDummy by DCE
    t_idx = next(i for i, v in enumerate(loop.body_vars) if v.name.startswith("t"))
    break_op, = [op for op in func_body.traverse() if isinstance(op, Break)]
    assert isinstance(_get_defining_op(break_op.values[t_idx], func_body), MakeDummy)
