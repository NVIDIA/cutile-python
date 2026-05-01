# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import re
import pytest
from math import ceil

import torch
from torch.testing import make_tensor

import cuda.tile as ct
from cuda.tile._bytecode.version import BytecodeVersion
from cuda.tile._exception import TileTypeError
from cuda.tile._ir.ops_utils import _is_implicit_cast_ok
from cuda.tile._ir.typing_support import to_dtype
from util import (
    assert_equal, filecheck, get_bytecode, int_32_64_dtypes,
    int_float_32_64_dtypes, is_hopper_or_newer, raises_if, AtomicOp,
    ref_atomic_arith, ref_atomic_bitwise
)
from conftest import arithmetic_dtypes, dtype_id, get_tileiras_version


_op_to_func = {
    AtomicOp.XCHG: ct.atomic_xchg,
    AtomicOp.ADD: ct.atomic_add,
    AtomicOp.MAX: ct.atomic_max,
    AtomicOp.MIN: ct.atomic_min,
    AtomicOp.AND: ct.atomic_and,
    AtomicOp.OR: ct.atomic_or,
    AtomicOp.XOR: ct.atomic_xor,
}


def xchg_func(x): return x.get_raw_memory().atomic_xchg_offset
def add_func(x): return x.get_raw_memory().atomic_add_offset
def max_func(x): return x.get_raw_memory().atomic_max_offset
def min_func(x): return x.get_raw_memory().atomic_min_offset
def and_func(x): return x.get_raw_memory().atomic_and_offset
def or_func(x): return x.get_raw_memory().atomic_or_offset
def xor_func(x): return x.get_raw_memory().atomic_xor_offset


_op_to_raw_memory_func = {
    AtomicOp.XCHG: xchg_func,
    AtomicOp.ADD: add_func,
    AtomicOp.MAX: max_func,
    AtomicOp.MIN: min_func,
    AtomicOp.AND: and_func,
    AtomicOp.OR: or_func,
    AtomicOp.XOR: xor_func,
}


@ct.kernel
def atomic_arith_kernel(x, y, z, TILE: ct.Constant[int], op_id: ct.Constant[int],
                        test_raw_memory: ct.Constant[int]):
    bid = ct.bid(0)
    offset = ct.arange(TILE, dtype=ct.int64)
    offset += bid*TILE
    val = ct.gather(y, offset)
    if not test_raw_memory:
        func = ct.static_eval(_op_to_func[AtomicOp(op_id)])
        old_val = func(x, offset, val,
                       memory_order=ct.MemoryOrder.ACQ_REL,
                       memory_scope=ct.MemoryScope.DEVICE)
    else:
        get_func = ct.static_eval(_op_to_raw_memory_func[AtomicOp(op_id)])
        func = get_func(x)
        old_val = func(offset, val,
                       memory_order=ct.MemoryOrder.ACQ_REL,
                       memory_scope=ct.MemoryScope.DEVICE)
    ct.scatter(z, offset, old_val)


@ct.kernel
def scalar_atomic_arith_kernel(x, y, z, op_id: ct.Constant[int], test_raw_memory: ct.Constant[int]):
    val = ct.gather(y, 0)
    if not test_raw_memory:
        func = ct.static_eval(_op_to_func[AtomicOp(op_id)])
        old_val = func(x, 0, val)
    else:
        get_func = ct.static_eval(_op_to_raw_memory_func[AtomicOp(op_id)])
        func = get_func(x)

        old_val = func(0, val)
    ct.scatter(z, 0, old_val)


def create_atomic_test_params(ops_config):
    params = []
    for op_name, torch_op, supported_dtypes in ops_config:
        for x_dtype in supported_dtypes:
            param_id = f"{op_name}-{dtype_id(x_dtype)}"
            params.append(pytest.param(op_name, torch_op, x_dtype, id=param_id))
    return params


atomic_arith_config = [
    (AtomicOp.XCHG, lambda _, y: y, int_float_32_64_dtypes),
    (AtomicOp.ADD, torch.add, int_float_32_64_dtypes + [torch.float16, torch.bfloat16]),
    (AtomicOp.MAX, torch.maximum, int_32_64_dtypes),
    (AtomicOp.MIN, torch.minimum, int_32_64_dtypes),
]


@pytest.mark.parametrize("op_name,torch_op,x_dtype",
                         create_atomic_test_params(atomic_arith_config))
@pytest.mark.parametrize("y_dtype", arithmetic_dtypes, ids=dtype_id)
@pytest.mark.parametrize("mode", ["array", "scalar"])
@pytest.mark.parametrize("test_raw_memory", [True, False])
def test_atomic_arith(op_name, torch_op, x_dtype, y_dtype, mode, test_raw_memory):
    if (x_dtype == torch.bfloat16
            and (not is_hopper_or_newer()
                 or get_tileiras_version() < BytecodeVersion.V_13_3)):
        pytest.skip("bfloat16 atomics require Hopper or newer and tileiras V_13_3+")

    if mode == "array":
        x = make_tensor((512,), dtype=x_dtype, device='cuda')
        y = make_tensor((512,), dtype=y_dtype, device='cuda')
        z = torch.zeros_like(x, device="cuda")
        grid = tuple(map(lambda d: ceil(d / 128), z.shape))

        def launch():
            ct.launch(torch.cuda.current_stream(), grid, atomic_arith_kernel,
                      (x, y, z, 128, op_name.value, test_raw_memory))
    else:
        x = make_tensor((1,), dtype=x_dtype, device='cuda')
        y = make_tensor((1,), dtype=y_dtype, device='cuda')
        z = torch.zeros_like(x, device="cuda")
        grid = (1,)

        def launch():
            ct.launch(torch.cuda.current_stream(), grid, scalar_atomic_arith_kernel,
                      (x, y, z, op_name.value, test_raw_memory))

    invalid_cast = not _is_implicit_cast_ok(to_dtype(y_dtype), to_dtype(x_dtype))
    msg = "cannot implicitly cast"
    with raises_if(invalid_cast, TileTypeError, match=re.escape(msg)):
        ref_x, ref_z = ref_atomic_arith(x, y, torch_op)
        launch()
        assert_equal(x, ref_x)
        assert_equal(z, ref_z)


atomic_bitwise_config = [
    (AtomicOp.AND, lambda x, y: x & y, int_float_32_64_dtypes),
    (AtomicOp.OR, lambda x, y: x | y, int_float_32_64_dtypes),
    (AtomicOp.XOR, lambda x, y: x ^ y, int_float_32_64_dtypes),
]


@pytest.mark.parametrize("op_name,torch_op,x_dtype",
                         create_atomic_test_params(atomic_bitwise_config))
@pytest.mark.parametrize("y_dtype", arithmetic_dtypes, ids=dtype_id)
@pytest.mark.parametrize("mode", ["array", "scalar"])
@pytest.mark.parametrize("test_raw_memory", [True, False])
def test_atomic_bitwise(op_name, torch_op, x_dtype, y_dtype, mode, test_raw_memory):
    if mode == "array":
        x = make_tensor((512,), dtype=x_dtype, device='cuda')
        y = make_tensor((512,), dtype=y_dtype, device='cuda')
        z = torch.zeros_like(x, device="cuda")
        grid = tuple(map(lambda d: ceil(d / 128), z.shape))

        def launch():
            ct.launch(torch.cuda.current_stream(), grid, atomic_arith_kernel,
                      (x, y, z, 128, op_name.value, test_raw_memory))
    else:
        x = make_tensor((1,), dtype=x_dtype, device='cuda')
        y = make_tensor((1,), dtype=y_dtype, device='cuda')
        z = torch.zeros_like(x, device="cuda")
        grid = (1,)

        def launch():
            ct.launch(torch.cuda.current_stream(), grid, scalar_atomic_arith_kernel,
                      (x, y, z, op_name.value, test_raw_memory))

    x_dtype = to_dtype(x_dtype)
    y_dtype = to_dtype(y_dtype)
    if x_dtype in (ct.float32, ct.float64):
        with pytest.raises(TileTypeError, match="Unsupported array dtype"):
            launch()
    elif x_dtype != y_dtype:
        msg = re.escape(f"Bitwise atomic read-modify-write operations require the "
                        f"update dtype ({y_dtype}) to exactly match the target dtype ({x_dtype})")
        with pytest.raises(TileTypeError, match=msg):
            launch()
    else:
        ref_x, ref_z = ref_atomic_bitwise(x, y, torch_op)
        launch()
        assert_equal(x, ref_x)
        assert_equal(z, ref_z)


@ct.kernel
def atomic_cas(x, y, z, TILE: ct.Constant[int], test_raw_memory: ct.Constant[int]):
    bid = ct.bid(0)
    offset = ct.arange(TILE, dtype=ct.int64)
    offset += bid*TILE
    cmp = ct.gather(x, offset)
    val = ct.gather(y, offset)
    if not test_raw_memory:
        old_val = ct.atomic_cas(
            x, offset, cmp, val,
            memory_order=ct.MemoryOrder.ACQ_REL,
            memory_scope=ct.MemoryScope.DEVICE)
    else:
        old_val = x.get_raw_memory().atomic_cas_offset(
            offset, cmp, val,
            memory_order=ct.MemoryOrder.ACQ_REL,
            memory_scope=ct.MemoryScope.DEVICE)
    ct.scatter(z, offset, old_val)


@ct.kernel
def scalar_atomic_cas(x, y, z, test_raw_memory: ct.Constant[int]):
    cmp = ct.gather(x, 0)
    val = ct.gather(y, 0)
    if not test_raw_memory:
        old_val = ct.atomic_cas(x, 0, cmp, val)
    else:
        old_val = x.get_raw_memory().atomic_cas_offset(0, cmp, val)
    ct.scatter(z, 0, old_val)


def ref_atomic_cas(x, y):
    ref_x = y.to(x.dtype)
    ref_z = x.clone()
    return ref_x, ref_z


atomic_cas_dtypes = [torch.uint32, torch.uint64, torch.int32, torch.int64,
                     torch.float32, torch.float64]


@pytest.mark.parametrize("x_dtype", atomic_cas_dtypes, ids=dtype_id)
@pytest.mark.parametrize("y_dtype", arithmetic_dtypes, ids=dtype_id)
@pytest.mark.parametrize("mode", ["array", "scalar"])
@pytest.mark.parametrize("test_raw_memory", [True, False])
def test_atomic_cas(x_dtype, y_dtype, mode, test_raw_memory):
    if mode == "array":
        x = make_tensor((512,), dtype=x_dtype, device='cuda')
        y = make_tensor((512,), dtype=y_dtype, device='cuda')
        z = torch.zeros_like(x, device="cuda")
        grid = tuple(map(lambda d: ceil(d / 128), z.shape))

        def launch():
            ct.launch(torch.cuda.current_stream(), grid,
                      atomic_cas, (x, y, z, 128, test_raw_memory))
    else:
        x = make_tensor((1,), dtype=x_dtype, device='cuda')
        y = make_tensor((1,), dtype=y_dtype, device='cuda')
        z = torch.zeros_like(x, device="cuda")
        grid = (1,)

        def launch():
            ct.launch(torch.cuda.current_stream(), grid,
                      scalar_atomic_cas, (x, y, z, test_raw_memory))

    invalid_cast = not _is_implicit_cast_ok(to_dtype(y_dtype), to_dtype(x_dtype))
    msg = "cannot implicitly cast"
    with raises_if(invalid_cast, TileTypeError, match=re.escape(msg)):
        ref_x, ref_z = ref_atomic_cas(x, y)
        launch()
        assert_equal(x, ref_x)
        assert_equal(z, ref_z)


ct_scope_to_tileir_scope = {
    ct.MemoryScope.BLOCK: "tl_blk",
    ct.MemoryScope.DEVICE: "device",
    ct.MemoryScope.SYS: "sys"
}


@pytest.mark.use_mlir
@pytest.mark.parametrize(
    "order",
    [
        None,
        ct.MemoryOrder.ACQ_REL,
        ct.MemoryOrder.ACQUIRE,
        ct.MemoryOrder.RELEASE,
        ct.MemoryOrder.RELAXED,
    ],
)
@pytest.mark.parametrize(
    "scope",
    [
        None,
        ct.MemoryScope.BLOCK,
        ct.MemoryScope.DEVICE,
        ct.MemoryScope.SYS,
    ],
)
@pytest.mark.parametrize("test_raw_memory", [True, False])
def test_atomic_order_scope(order, scope, test_raw_memory):
    @ct.kernel
    def atomic_kernel_for_order_scope(x, TILE: ct.Constant[int]):
        bid = ct.bid(0)
        offset = ct.arange(TILE, dtype=ct.int64)
        offset += bid*TILE
        val = ct.full((TILE,), 1, dtype=ct.int32)

        if not test_raw_memory:
            if order is None and scope is None:
                ct.atomic_add(x, offset, val)
            elif order is None:
                ct.atomic_add(x, offset, val, memory_scope=scope)
            elif scope is None:
                ct.atomic_add(x, offset, val, memory_order=order)
            else:
                ct.atomic_add(x, offset, val, memory_order=order, memory_scope=scope)
        else:
            raw_memory = x.get_raw_memory()
            if order is None and scope is None:
                raw_memory.atomic_add_offset(offset, val)
            elif order is None:
                raw_memory.atomic_add_offset(offset, val,
                                             memory_scope=scope)
            elif scope is None:
                raw_memory.atomic_add_offset(offset, val,
                                             memory_order=order)
            else:
                raw_memory.atomic_add_offset(offset, val,
                                             memory_order=order,
                                             memory_scope=scope)

    check_directive = "// CHECK: atomic_rmw_tko"

    # set up expected order
    memory_order = order if order is not None else ct.MemoryOrder.ACQ_REL
    check_directive += f" {memory_order.value}"

    # set up expected scope
    memory_scope = scope if scope is not None else ct.MemoryScope.DEVICE
    check_directive += f" {ct_scope_to_tileir_scope[memory_scope]}"

    x = make_tensor((512,), dtype=torch.int32, device='cuda')
    bytecode = get_bytecode(atomic_kernel_for_order_scope, (x, 128))
    filecheck(bytecode, check_directive)


@ct.kernel
def mixed_scalar_tile_atomic(x, y):
    cmp = ct.gather(x, 0)
    val = ct.gather(y, 0)
    ct.atomic_cas(x, 0, cmp, val)
    ct.atomic_xchg(x, 1, val)
    ct.atomic_add(x, 2, val)
    ct.atomic_xor(x, 3, val)
    ct.atomic_max(x, 4, val)


@ct.kernel
def raw_memory_mixed_scalar_tile_atomic(x, y):
    mem_x = x.get_raw_memory()
    cmp = ct.gather(x, 0)
    val = ct.gather(y, 0)
    mem_x.atomic_cas_offset(0, cmp, val)
    mem_x.atomic_xchg_offset(1, val)
    mem_x.atomic_add_offset(2, val)
    mem_x.atomic_xor_offset(3, val)
    mem_x.atomic_max_offset(4, val)


@pytest.mark.parametrize("test_raw_memory", [True, False])
def test_mixed_scalar_tile_atomic(test_raw_memory):
    x = make_tensor((1,), dtype=torch.int32, device="cuda")
    y = make_tensor((1,), dtype=torch.int32, device="cuda")
    kernel = (mixed_scalar_tile_atomic if not test_raw_memory
              else raw_memory_mixed_scalar_tile_atomic)
    ct.launch(torch.cuda.current_stream(), (1,), kernel, (x, y))


class TestInvalidAtomicMemoryOrderAndScope:
    @pytest.mark.parametrize("test_raw_memory", [True, False])
    def test_atomic_cas_weak_ordering(self, test_raw_memory):
        if not test_raw_memory:
            @ct.kernel
            def kernel(x):
                ct.atomic_cas(x, 0, 0, 0, memory_order=ct.MemoryOrder.WEAK,
                              memory_scope=ct.MemoryScope.DEVICE)
        else:
            @ct.kernel
            def kernel(x):
                mem_x = x.get_raw_memory()
                mem_x.atomic_cas_offset(0, 0, 0, memory_order=ct.MemoryOrder.WEAK,
                                        memory_scope=ct.MemoryScope.DEVICE)
        x = make_tensor((1,), dtype=torch.int32, device="cuda")
        with pytest.raises(TileTypeError, match="Invalid memory order for tile_atomic_cas"):
            ct.launch(torch.cuda.current_stream(), (1,), kernel, (x,))

    @pytest.mark.parametrize("test_raw_memory", [True, False])
    def test_atomic_rmw_weak_ordering(self, test_raw_memory):
        if not test_raw_memory:
            @ct.kernel
            def kernel(x):
                ct.atomic_add(x, 0, 0, memory_order=ct.MemoryOrder.WEAK,
                              memory_scope=ct.MemoryScope.DEVICE)
        else:
            @ct.kernel
            def kernel(x):
                mem_x = x.get_raw_memory()
                mem_x.atomic_add_offset(0, 0, memory_order=ct.MemoryOrder.WEAK,
                                        memory_scope=ct.MemoryScope.DEVICE)
        x = make_tensor((1,), dtype=torch.int32, device="cuda")
        with pytest.raises(TileTypeError, match="Invalid memory order for tile_atomic_rmw"):
            ct.launch(torch.cuda.current_stream(), (1,), kernel, (x,))

    @pytest.mark.parametrize("test_raw_memory", [True, False])
    @pytest.mark.parametrize(
        "memory_order",
        [
            ct.MemoryOrder.ACQ_REL,
            ct.MemoryOrder.ACQUIRE,
            ct.MemoryOrder.RELEASE,
            ct.MemoryOrder.RELAXED,
        ],
    )
    def test_atomic_rmw_none_scope(self, memory_order, test_raw_memory):
        if not test_raw_memory:
            @ct.kernel
            def kernel(x):
                ct.atomic_add(x, 0, 0, memory_order=memory_order,
                              memory_scope=ct.MemoryScope.NONE)
        else:
            @ct.kernel
            def kernel(x):
                mem_x = x.get_raw_memory()
                mem_x.atomic_add_offset(0, 0, memory_order=memory_order,
                                        memory_scope=ct.MemoryScope.NONE)
        x = make_tensor((1,), dtype=torch.int32, device="cuda")
        with pytest.raises(
            TileTypeError,
            match="tile_atomic_rmw with (.+) memory ordering requires a memory scope",
        ):
            ct.launch(torch.cuda.current_stream(), (1,), kernel, (x,))

    @pytest.mark.parametrize("test_raw_memory", [True, False])
    @pytest.mark.parametrize(
        "memory_order",
        [
            ct.MemoryOrder.ACQ_REL,
            ct.MemoryOrder.ACQUIRE,
            ct.MemoryOrder.RELEASE,
            ct.MemoryOrder.RELAXED,
        ],
    )
    def test_atomic_cas_none_scope(self, memory_order, test_raw_memory):
        if not test_raw_memory:
            @ct.kernel
            def kernel(x):
                ct.atomic_cas(x, 0, 0, 0, memory_order=memory_order,
                              memory_scope=ct.MemoryScope.NONE)
        else:
            @ct.kernel
            def kernel(x):
                mem_x = x.get_raw_memory()
                mem_x.atomic_cas_offset(0, 0, 0, memory_order=memory_order,
                                        memory_scope=ct.MemoryScope.NONE)
        x = make_tensor((1,), dtype=torch.int32, device="cuda")
        with pytest.raises(
            TileTypeError,
            match="tile_atomic_cas with (.+) memory ordering requires a memory scope",
        ):
            ct.launch(torch.cuda.current_stream(), (1,), kernel, (x,))


@ct.kernel
def offset_atomic_add_with_mask(x, update, TILE: ct.Constant[int]):
    bid = ct.bid(0)
    offset = ct.arange(TILE, dtype=ct.int64)
    offset += bid*TILE
    val = ct.gather(update, offset)
    mem_x = x.get_raw_memory()
    mask = (offset % 2) == 0
    mem_x.atomic_add_offset(offset, val, mask=mask)


def test_atomic_offset_mask():
    n = 512
    x = torch.zeros(n, dtype=torch.int32, device='cuda')
    update = torch.ones(n, dtype=torch.int32, device='cuda')
    tile = 128
    grid = (ceil(n / tile),)
    ct.launch(torch.cuda.current_stream(), grid, offset_atomic_add_with_mask,
              (x, update, tile))
    x_cpu = x.cpu()
    assert (x_cpu[0::2] == 1).all(), "Even elements should have been incremented"
    assert (x_cpu[1::2] == 0).all(), "Odd elements should be unchanged (masked)"


@ct.kernel
def offset_atomic_cas_with_mask(x, expected, desired, out, TILE: ct.Constant[int],
                                N_HALF: ct.Constant[int]):
    bid = ct.bid(0)
    offset = ct.arange(TILE, dtype=ct.int64)
    offset += bid*TILE
    exp_val = ct.gather(expected, offset)
    des_val = ct.gather(desired, offset)
    mem_x = x.get_raw_memory()
    mask = offset < N_HALF
    old_val = mem_x.atomic_cas_offset(offset, exp_val, des_val, mask=mask)
    ct.scatter(out, offset, old_val)


def test_atomic_offset_cas_mask():
    """When mask is False, atomic_cas_offset returns expected"""
    n = 512
    dtype = torch.int32
    x = torch.zeros(n, dtype=dtype, device='cuda')
    expected = torch.full((n,), -1, dtype=dtype, device='cuda')
    desired = torch.ones(n, dtype=dtype, device='cuda')
    out = torch.zeros(n, dtype=dtype, device='cuda')
    tile = 128
    grid = (ceil(n / tile),)
    ct.launch(torch.cuda.current_stream(), grid, offset_atomic_cas_with_mask,
              (x, expected, desired, out, tile, n // 2))

    assert (out.cpu()[:n // 2] == 0).all(), "First half: CAS attempted, old value returned"
    assert (out.cpu()[n // 2:] == -1).all(), "Second half: masked out, 'expected' value returned"
