# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import cuda.tile as ct
import pytest
from cuda.tile._ir.ops import AssumeDivBy, StorePointer, MakeTensorView
from cuda.tile._ir.ir import Block
from cuda.tile._compile import compile_tile
from cuda.tile.compilation import (
        ParameterConstraint, ArrayConstraint, ListConstraint,
        ConstantConstraint, ScalarConstraint, KernelSignature
)
from cuda.tile._cext import CallingConvention
from cuda.tile._exception import TileTypeError
from typing import Sequence
import torch


def get_ir(func, args: Sequence[ParameterConstraint]) -> Block:
    sig = KernelSignature(args, CallingConvention.cutile_python_v1())
    [body] = compile_tile(func, [sig], return_final_ir=True, return_cubin=False).final_ir
    return body


def flattened_inputs(op):
    if isinstance(op, MakeTensorView):
        yield ('base_ptr', op.base_ptr)
        yield from ((f'shape[{i}]', s) for i, s in enumerate(op.shape))
        yield from ((f'stride[{i}]', s) for i, s in enumerate(op.dynamic_strides))
    elif isinstance(op, StorePointer):
        yield ('base_ptr', op.pointer)
    else:
        raise NotImplementedError()


def get_op_divby(block: Block, op_class) -> list[dict[str, int]]:
    """For each op of `op_class` get a dict of `field_name -> divby`
    """
    assumes = {op.result_var.name: op.divisor
               for op in block.traverse() if isinstance(op, AssumeDivBy)}
    result = []
    for op in block.traverse():
        if isinstance(op, op_class):
            input_divby = {}
            for input_name, var in flattened_inputs(op):
                if var.name in assumes:
                    input_divby[input_name] = assumes[var.name]
            result.append(input_divby)
    return result


def array_arg(dtype: ct.DType = ct.float32,
              ndim: int = 1,
              base_div: int = 1,
              stride_div: Sequence[int] | None = None,
              shape_div: Sequence[int] | None = None,
              stride_const: Sequence[int | None] | None = None,
              ) -> ArrayConstraint:
    if stride_div is None:
        stride_div = (1,) * ndim
    if shape_div is None:
        shape_div = (1,) * ndim
    return ArrayConstraint(dtype, ndim,
                           index_dtype=ct.int32,
                           base_addr_divisible_by=base_div,
                           stride_lower_bound_incl=0,
                           stride_constant=stride_const,
                           stride_divisible_by=stride_div,
                           shape_divisible_by=shape_div,
                           alias_groups=[],
                           may_alias_internally=False)


def list_arg(dtype: ct.DType = ct.float32,
             ndim: int = 1,
             arr_base_div: int = 1,
             arr_stride_div: Sequence[int] | None = None,
             arr_shape_div: Sequence[int] | None = None,
             arr_stride_const: Sequence[int | None] | None = None,
             ) -> ListConstraint:
    elem_constraint = array_arg(dtype, ndim, arr_base_div,
                                arr_stride_div, arr_shape_div, arr_stride_const)
    return ListConstraint(elem_constraint, alias_groups=[], elements_may_alias=False)


def const_arg(val):
    return ConstantConstraint(val)


# --- Seeding from kernel args ---

def test_seed_from_array_arg():
    def kernel(x):
        ct.store(x, (0, 0), 0)

    body = get_ir(kernel, (array_arg(ndim=2, base_div=16, stride_div=(8, 1), shape_div=(4, 1)),))
    assert get_op_divby(body, MakeTensorView) == [{'base_ptr': 16, 'shape[0]': 4, 'stride[0]': 8}]


def test_unconstarined_array():
    def kernel(x):
        t = ct.load(x, (0,), (1,))
        ct.store(x, (0,), t)

    body = get_ir(kernel, (array_arg(),))
    assert get_op_divby(body, MakeTensorView) == [{}]


# --- ct.assume_divisible_by ---

def test_assume_divisible_by_emits_op():
    def kernel(x, n: int):
        n = ct.assume_divisible_by(n, 16)
        ct.store(x, (n,), 0)

    body = get_ir(kernel, (
        array_arg(ndim=1, base_div=1, stride_const=(1,)),
        ScalarConstraint(ct.int32),
    ))
    ops = [op.divisor for op in body.traverse() if isinstance(op, AssumeDivBy)]
    assert ops == [16]


def test_assume_divisible_by_propagates_to_dynamic_slice():
    def kernel(x, start_factor: int, extent: int):
        start_factor = ct.assume_divisible_by(start_factor, 32)
        extent = ct.assume_divisible_by(extent, 32)
        start = ct.bid(0) * start_factor
        stop = start + extent
        sub_x = x.slice(axis=0, start=start, stop=stop)
        tile = ct.load(sub_x, index=(0,), shape=(1,))
        ct.store(sub_x, index=(0,), tile=tile)

    body = get_ir(kernel, (
        array_arg(dtype=ct.bfloat16, base_div=16, stride_const=(1,)),
        ScalarConstraint(ct.int32),
        ScalarConstraint(ct.int32),
    ))

    divby = get_op_divby(body, MakeTensorView)[0]
    assert divby.get('base_ptr') == 16 and divby.get('shape[0]') == 32


def test_assume_divisible_by_divisor_one_is_noop():
    def kernel(x, n: int):
        n = ct.assume_divisible_by(n, 1)
        ct.store(x, (n,), 0)

    body = get_ir(kernel, (
        array_arg(ndim=1, base_div=1, stride_const=(1,)),
        ScalarConstraint(ct.int32),
    ))
    ops = [op for op in body.traverse() if isinstance(op, AssumeDivBy)]
    assert ops == []


def test_assume_divisible_by_non_power_of_two_divisor():
    # divisor=12 has largest power-of-2 factor 4.
    # The propagate_divby pass extracts that power-of-2 when inserting
    # AssumeDivBy before MakeTensorView, so shape[0] ends up with divisor=4.
    def kernel(x, extent: int):
        extent = ct.assume_divisible_by(extent, 12)
        sub_x = x.slice(axis=0, start=0, stop=extent)
        tile = ct.load(sub_x, index=(0,), shape=(1,))
        ct.store(sub_x, index=(0,), tile=tile)

    body = get_ir(kernel, (
        array_arg(ndim=1, base_div=1, stride_const=(1,)),
        ScalarConstraint(ct.int32),
    ))
    divby = get_op_divby(body, MakeTensorView)[0]
    assert divby.get('shape[0]') == 4


def test_assume_divisible_by_type_error_on_float():
    def kernel(x, f: float):
        f = ct.assume_divisible_by(f, 16)
        ct.store(x, (0,), 0)

    with pytest.raises(TileTypeError, match="integer scalar"):
        get_ir(kernel, (
            array_arg(ndim=1, stride_const=(1,)),
            ScalarConstraint(ct.float32),
        ))


def test_assume_divisible_by_error_on_nonconstant_divisor():
    def kernel(x, n: int, d: int):
        n = ct.assume_divisible_by(n, d)
        ct.store(x, (n,), 0)

    with pytest.raises(TileTypeError, match="integer constant"):
        get_ir(kernel, (
            array_arg(ndim=1, stride_const=(1,)),
            ScalarConstraint(ct.int32),
            ScalarConstraint(ct.int32),
        ))


def test_assume_divisible_by_error_on_nonpositive_divisor():
    def kernel(x, n: int):
        n = ct.assume_divisible_by(n, 0)
        ct.store(x, (n,), 0)

    with pytest.raises(TileTypeError, match="positive divisor"):
        get_ir(kernel, (
            array_arg(ndim=1, stride_const=(1,)),
            ScalarConstraint(ct.int32),
        ))


# --- Control flow propagation ---

def test_if_else():
    def kernel(x, y):
        if ct.bid(0) == 0:
            z = x
        else:
            z = y
        ct.scatter(z, 0, 0)

    body = get_ir(kernel, (
        array_arg(base_div=32, stride_const=(1,)),
        array_arg(base_div=16, stride_const=(1,)),
    ))
    assert get_op_divby(body, StorePointer) == [{'base_ptr': 16}]


def test_for_loop_same_var():
    def kernel(x):
        a = x
        for _ in range(5):
            a = x
        ct.scatter(a, 0, 0)

    body = get_ir(kernel, (array_arg(base_div=16, stride_const=(1,)),))
    assert get_op_divby(body, StorePointer) == [{'base_ptr': 16}]


def test_for_loop_different_vars():
    def kernel(x, y):
        a = x
        for _ in range(5):
            a = y
        t = ct.load(a, (0,), (1,))
        ct.store(a, (0,), t)

    body = get_ir(kernel, (
        array_arg(base_div=32),
        array_arg(base_div=8),
    ))
    assert get_op_divby(body, MakeTensorView) == [{'base_ptr': 8}]


def test_while_loop_different_vars():
    def kernel(x, y, z):
        a = x
        while True:
            a = y
            if ct.bid(0) == 0:
                break
            else:
                a = z
                ct.store(a, (0,), 0)

        t = ct.load(a, (0,), (1,))
        ct.store(a, (0,), t)

    body = get_ir(kernel, (
        array_arg(base_div=32),
        array_arg(base_div=8),
        array_arg(base_div=4),
    ))
    assert get_op_divby(body, MakeTensorView) == [{'base_ptr': 4}, {'base_ptr': 8}]


# --- Slice propagation ---


@pytest.mark.parametrize("stride_const", [(1,), (None,)])
def test_slice_offset_zero(stride_const):
    def kernel(x):
        y = x.slice(axis=0, start=0, stop=4)
        ct.store(y, (0,), ct.load(y, (0,), (1,)))

    body = get_ir(kernel, (array_arg(base_div=32, stride_const=stride_const),))
    assert get_op_divby(body, MakeTensorView) == [{"base_ptr": 32, "shape[0]": 4}]


@pytest.mark.parametrize("stride_const", [(1,), (None,)])
def test_slice_offset_aligned(stride_const):
    def kernel(x):
        y = x.slice(axis=0, start=8, stop=16)
        ct.store(y, (0,), ct.load(y, (0,), (1,)))

    body = get_ir(kernel, (array_arg(base_div=32, stride_const=stride_const),))
    assert get_op_divby(body, MakeTensorView) == [{"base_ptr": 32, "shape[0]": 8}]


@pytest.mark.parametrize("stride_const", [(1,), (None,)])
def test_slice_offset_partially_aligned(stride_const):
    def kernel(x):
        y = x.slice(axis=0, start=2, stop=16)
        ct.store(y, (0,), ct.load(y, (0,), (1,)))

    body = get_ir(kernel, (array_arg(base_div=32, stride_const=stride_const),))
    assert get_op_divby(body, MakeTensorView) == [{"base_ptr": 8, "shape[0]": 2}]


@pytest.mark.parametrize("stride_const", [(1,), (None,)])
def test_slice_offset_unaligned(stride_const):
    def kernel(x):
        y = x.slice(axis=0, start=1, stop=16)
        ct.store(y, (0,), ct.load(y, (0,), (1,)))

    body = get_ir(kernel, (array_arg(base_div=32, stride_const=stride_const),))
    assert get_op_divby(body, MakeTensorView) == [{"base_ptr": 4}]


@pytest.mark.parametrize("stride_const", [(1,), (None,)])
def test_slice_array_dynamic_offset(stride_const):
    def kernel(x, y):
        # start y.shape[0] divby 4, which is 16 bytes offset
        z = x.slice(axis=0, start=y.shape[0], stop=x.shape[0])
        ct.store(z, 0, 0)

        # start has no divby, which is 4 bytes offset
        z2 = x.slice(axis=0, start=y.shape[1], stop=x.shape[0])
        ct.store(z2, 0, 0)

    body = get_ir(kernel, (
        array_arg(ndim=1, base_div=32, stride_const=stride_const),
        array_arg(ndim=2, base_div=1, shape_div=(4, 1)),
    ))
    assert get_op_divby(body, MakeTensorView) == [{'base_ptr': 16}, {'base_ptr': 4}]


# --- Binary op and uniary op propagation---

def test_divby_add():
    def kernel(x, y):
        # start = x.shape[0] + y.shape[0], divby gcd(8,4) = 4
        # 4 elements * 4 bytes = 16 byte offset, gcd(32, 16) = 16
        z = x.slice(axis=0, start=x.shape[0] + y.shape[0], stop=x.shape[0] * 2)
        ct.store(z, 0, 0)

    body = get_ir(kernel, (
        array_arg(base_div=32, stride_const=(1,), shape_div=(8,)),
        array_arg(shape_div=(4,)),
    ))
    assert get_op_divby(body, MakeTensorView) == [{'base_ptr': 16, 'shape[0]': 4}]


def test_divby_sub():
    def kernel(x, y):
        # start = x.shape[0] - y.shape[0], divby gcd(8,4) = 4
        z = x.slice(axis=0, start=x.shape[0] - y.shape[0], stop=x.shape[0])
        ct.store(z, 0, 0)

    body = get_ir(kernel, (
        array_arg(base_div=32, stride_const=(1,), shape_div=(8,)),
        array_arg(shape_div=(4,)),
    ))
    assert get_op_divby(body, MakeTensorView) == [{'base_ptr': 16, 'shape[0]': 4}]


def test_divby_mul():
    def kernel(x, y):
        # start = x.shape[0] * y.shape[0], divby 2*2 = 4
        # 4 elements * 4 bytes = 16, gcd(32, 16) = 16
        z = x.slice(axis=0, start=x.shape[0] * y.shape[0], stop=x.shape[0] * 2)
        ct.store(z, 0, 0)

    body = get_ir(kernel, (
        array_arg(base_div=32, stride_const=(1,), shape_div=(2,)),
        array_arg(shape_div=(2,)),
    ))
    assert get_op_divby(body, MakeTensorView) == [{'base_ptr': 16, 'shape[0]': 4}]


def test_divby_neg():
    def kernel(x):
        # -x.shape[0] divby 8
        # 8 elements * 4 bytes = 32 byte offset, gcd(32, 32) = 32
        z = x.slice(axis=0, start=-x.shape[0], stop=0)
        ct.store(z, 0, 0)

    body = get_ir(kernel, (
        array_arg(base_div=32, stride_const=(1,), shape_div=(8,)),
    ))
    assert get_op_divby(body, MakeTensorView) == [{'base_ptr': 32, 'shape[0]': 8}]


# --- List array divisibility ---
def test_list_array_divby():
    def kernel(xs, n: ct.Constant[int]):
        for i in range(n):
            item = xs[i]
            t = ct.load(item, (0,), (1,))
            ct.store(item, (0,), t)

    body = get_ir(kernel, (list_arg(arr_base_div=8), const_arg(10)))
    assert get_op_divby(body, MakeTensorView) == [{'base_ptr': 8}]


# --- List divby through block boundaries ---

def test_list_divby_through_if_else():
    def kernel(xs, ys):
        if ct.bid(0) == 0:
            zs = xs
        else:
            zs = ys
        item = zs[0]
        t = ct.load(item, (0,), (1,))
        ct.store(item, (0,), t)

    body = get_ir(kernel, (
        list_arg(arr_base_div=32),
        list_arg(arr_base_div=16),
    ))
    assert get_op_divby(body, MakeTensorView) == [{'base_ptr': 16}]


def test_list_divby_through_loop():
    def kernel(xs, ys, n: ct.Constant[int]):
        zs = xs
        for _ in range(n):
            zs = ys
        item = zs[0]
        t = ct.load(item, (0,), (1,))
        ct.store(item, (0,), t)

    body = get_ir(kernel, (
        list_arg(arr_base_div=32),
        list_arg(arr_base_div=8),
        const_arg(5),
    ))
    assert get_op_divby(body, MakeTensorView) == [{'base_ptr': 8}]


def test_use_assumed_var_after_block():
    @ct.kernel
    def kern(x):
        c = 16
        if ct.bid(0) == 0:
            ct.scatter(x, 0, c)
        ct.scatter(x, 0, c)

    x = torch.zeros(8, dtype=torch.int32, device="cuda")
    ct.launch(torch.cuda.current_stream(), (1,), kern, (x,))
