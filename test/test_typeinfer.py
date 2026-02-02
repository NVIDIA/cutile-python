# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import cuda.tile as ct
import re

from cuda.tile._compiler_options import CompilerOptions
from cuda.tile._exception import TileTypeError, TileValueError
from cuda.tile._compile import compile_tile

from util import raises_if, is_ampere_or_ada


def nd_tensor(nd: int, dtype=None):
    return torch.rand((4,) * nd, dtype=dtype, device='cuda')


def compile(pyfunc, args):
    return compile_tile(pyfunc, args, CompilerOptions())


# ===== Failure cases ==========
def test_invalid_shape_rank():

    def kernel(x):
        ct.load(x, (0, 0), shape=(2, 2, 4))

    msg = re.escape('Expected shape length to be 2, got 3')
    with pytest.raises(TileTypeError, match=msg):
        compile(kernel, (nd_tensor(2),))


def test_invalid_shape_dtype():

    def kernel(x):
        ct.load(x, (0, 0), shape=(2, 2.0))

    msg = re.escape('Invalid argument "shape" of load(): Expected a tuple of integers,'
                    ' but element #1 has type Tile[float32,()]')

    with pytest.raises(TileTypeError, match=msg):
        compile(kernel, (nd_tensor(2),))


def test_invalid_shape_tuple():

    def kernel(x):
        ct.load(x, (0, 0), shape=2)

    msg = re.escape('Invalid argument "shape" of load(): Expected shape length to be 2, got 1')

    with pytest.raises(TileTypeError, match=msg):
        compile(kernel, (nd_tensor(2),))


def test_invalid_shape_sign():

    def kernel(x):
        ct.load(x, (0, 0), shape=(-1, -2))

    msg = re.escape('Invalid argument "shape" of load():'
                    ' Dimension #0 of shape (-1, -2) is not positive')
    with pytest.raises(TileTypeError, match=msg):
        compile(kernel, (nd_tensor(2),))


def test_invalid_shape_const():

    def kernel(x, i):
        ct.load(x, (0, 0), shape=(2, i))

    # TODO: improve error message to show which index is not const
    msg = re.escape(
        'Invalid argument "shape" of load(): '
        'Expected a constant integer tuple, but given value is not constant'
    )

    with pytest.raises(TileTypeError, match=msg):
        compile(kernel, (nd_tensor(2), 0))


def test_zero_shape():

    def kernel():
        ct.full((0, 0), 1, dtype=torch.float32)

    msg = re.escape('Invalid argument "shape" of full():'
                    ' Dimension #0 of shape (0, 0) is not positive')

    with pytest.raises(TileTypeError, match=msg):
        compile(kernel, ())


def test_non_power_of_2_shape():

    def kernel():
        ct.full((2, 3), 1, dtype=torch.float32)

    msg = re.escape('Invalid argument "shape" of full():'
                    ' Dimension #1 of shape (2, 3) is not a power of two')
    with pytest.raises(TileTypeError, match=msg):
        compile(kernel, ())


def test_invalid_index_rank():
    def kernel(x):
        ct.load(x, (0, 0, 0), shape=(2, 2))

    msg = re.escape('Index size 3 does not match the array rank 2')
    with pytest.raises(TileTypeError, match=msg):
        compile(kernel, (nd_tensor(2),))


def test_invalid_order_literal():
    def kernel(x):
        ct.load(x, (0, 0), shape=(2, 2), order='A')

    msg = r'Invalid argument "order" of load\(\): Expected \'C\' or \'F\', got \'A\''
    with pytest.raises(TileTypeError, match=msg):
        compile(kernel, (nd_tensor(2),))


def test_invalid_order_range():
    def kernel(x):
        ct.load(x, (0, 0), shape=(2, 2), order=(0, 3))

    msg = re.escape('Invalid argument "order" of load(): Axis 3 is out of range for rank 2')
    with pytest.raises(TileTypeError, match=msg):
        compile(kernel, (nd_tensor(2),))


def test_invalid_tile_shape():
    def kernel(x, y):
        tx = ct.load(x, (0, 0), shape=(2, 2))
        ty = ct.load(x, (0, 0), shape=(2, 2, 2))
        tx + ty

    msg = re.escape('Invalid argument "shape" of load(): Expected shape length to be 2, got 3')
    with pytest.raises(TileTypeError, match=msg):
        compile(kernel, (nd_tensor(2), nd_tensor(3)))


def test_invalid_tile_arg():
    def kernel(x):
        ct.permute(x, (1, 0))

    msg = re.escape('Invalid argument #1 of permute(): '
                    'Expected a tile, but given value has type Array[float32,(?,?):(?,1)]')
    with pytest.raises(TileTypeError, match=msg):
        compile(kernel, (nd_tensor(2),))


def test_invalid_scalar():
    def kernel(x):
        ct.full((4, 4), "foo", dtype=torch.int32)

    msg = re.escape('Invalid argument "fill_value" of full(): Expected a scalar')
    with pytest.raises(TileTypeError, match=msg):
        compile(kernel, (nd_tensor(2),))


def test_invalid_dtype():
    def kernel(x):
        ct.full((4, 4), 1, dtype="foo")

    msg = re.escape('Invalid argument "dtype" of full(): Expected a dtype constant')
    with pytest.raises(TileTypeError, match=msg):
        compile(kernel, (nd_tensor(2),))


def test_invalid_constant_arg_format():

    def kernel():
        x = ct.full((1,), 0, dtype=ct.float32)
        ct.printf(x)

    msg = re.escape("Invalid argument \"format\" of printf(): "
                    "Expected a string constant, but given value is not constant")
    with pytest.raises(TileTypeError, match=msg):
        compile(kernel, ())


def test_invalid_constant_arg_keepdims():
    def kernel(keepdims: bool):
        x = ct.full((1,), 0, dtype=ct.float32)
        ct.sum(x, 0, keepdims=keepdims)

    msg = re.escape("Invalid argument \"keepdims\" of sum(): "
                    "Expected a boolean constant, but given value is not constant")
    with pytest.raises(TileTypeError, match=msg):
        compile(kernel, (True,))


def test_arith_on_bool():
    def kernel():
        x = ct.full((1,), 0, dtype=ct.bool_)
        y = ct.full((1,), 0, dtype=ct.bool_)
        x + y

    msg = r'Binary arithmetic op `add` does not support bool, please cast bool to int'
    with pytest.raises(TileTypeError, match=msg):
        compile(kernel, ())


def test_printf_format():
    SKIP_FP8 = is_ampere_or_ada()

    def print_kernel():
        # signed
        ct.printf("%d", -1)
        ct.printf("%d", ct.int32(-1))
        ct.printf("%d", ct.int64(-1))
        ct.printf("%ld", ct.int32(-1))
        ct.printf("%lld", ct.int64(-1))
        # unsigned
        ct.printf("%u", 123)
        ct.printf("%u", ct.uint32(1))
        ct.printf("%u", ct.uint64(1))
        ct.printf("%lu", ct.uint32(-1))
        ct.printf("%llu", ct.uint64(-1))
        # float
        ct.printf("%f", 3.14)
        ct.printf("%f", ct.bfloat16(3.14))
        ct.printf("%f", ct.float16(3.14))
        ct.printf("%f", ct.float32(3.14))
        ct.printf("%f", ct.float64(3.14))
        if not SKIP_FP8:
            ct.printf("%f", ct.float8_e5m2(3.14))
            ct.printf("%f", ct.float8_e4m3fn(3.14))
        ct.printf("%f", ct.tfloat32(3.14))
        # others
        ct.printf("escape %% %d", 123)
        ct.printf("escape %%%% %d", 123)
        ct.printf("ints %d %i %u %o %x %X",
                  1, 2, 3, 4, 5, 6)
        ct.printf("floats %f %e %E %f %F %g %G %a %A",
                  1., 2., 3., 4., 5., 6., 7., 8., 9.)
        ct.printf("floats percent %+3.5f%%", 3.14)
        ct.printf("pad zero %010d", 1977)
        ct.printf("hex %#x", 255)

    compile(print_kernel, ())

    # Format specifier doesn't match input tile dtype
    def mix_int_float():
        ct.printf("%d", -1.0)

    def mix_float_int():
        ct.printf("%f", 1)

    for f in [mix_int_float, mix_float_int]:
        msg = r"Format .* for arg #0 got unexpected type of .*"
        with pytest.raises(TileTypeError, match=msg):
            compile(f, ())

    # Format specifier ill-formed
    def invalid_format_1():
        ct.printf("%%%+3", 1)

    def invalid_format_2():
        ct.printf("%!")

    for f in [invalid_format_1, invalid_format_2]:
        with pytest.raises(TileTypeError, match=r'Invalid format string'):
            compile(f, ())

    # Specifier not supported
    def invalid_specifier_1():
        ct.printf("%c", 1)

    def invalid_specifier_2():
        ct.printf("%s", 1)

    def invalid_specifier_3():
        ct.printf("%p", 1)

    def invalid_specifier_4():
        ct.printf("%n", 1)

    for f in [invalid_specifier_1, invalid_specifier_2, invalid_specifier_3, invalid_specifier_4]:
        with pytest.raises(TileTypeError, match=r'Specifier .* in .* is not supported'):
            compile(f, ())

    def not_enough_args():
        ct.printf("prefix: %d, %d", 1)

    with pytest.raises(TileTypeError, match=r'Not enough arguments for format string'):
        compile(not_enough_args, ())

    def too_many_args():
        ct.printf("prefix: %d", 1, 2, 3)

    with pytest.raises(TileTypeError, match=r'Too many arguments for format string'):
        compile(too_many_args, ())


def kernel_if_else(x):
    if ct.bid(0) == 0:
        a = 1
    else:
        a = 2.0
    ct.store(x, (0,), a)


def kernel_for_loop(x):
    a = 1
    for _ in range(10):
        a *= 2.0
    ct.store(x, (0,), a)


def kernel_while_loop(x):
    a = 1
    i = 0
    while i < 10:
        a *= 2.0
        i += 1
        if i >= 5:
            break
    ct.store(x, (0,), a)


@pytest.mark.parametrize("kernel", [kernel_if_else, kernel_for_loop, kernel_while_loop])
def test_control_flow_type_mismatch(kernel):
    x = torch.zeros(1, dtype=torch.float32, device='cuda')
    msg = re.escape('Type of `a` depends on path taken')
    with pytest.raises(TileTypeError, match=msg):
        compile(kernel, (x, ))


def test_unused_type_mismatch_inside_loop():
    @ct.kernel
    def kernel(x, y):
        for i in range(2):
            if ct.bid(0) == 0:
                t = ct.gather(x, i)  # t is an int32
                ct.scatter(x, i, t + 1)
            else:
                t = ct.gather(y, i)  # t is a float32
                ct.scatter(y, i, t + 3.0)
        # There should be no type error because `t` is never used

    x = torch.tensor([10, 20], dtype=torch.int32, device="cuda")
    y = torch.tensor([10.0, 20.0], dtype=torch.float32, device="cuda")
    ct.launch(torch.cuda.current_stream(), (2,), kernel, (x, y))
    assert x.tolist() == [11, 21]
    assert y.tolist() == [13.0, 23.0]


@pytest.mark.parametrize("val, int32_raises, int64_raises, uint64_raises", [
    (5, False, False, True),
    (-2**31, False, False, True),
    (-2**31-1, True, False, True),
    (2**31, True, False, True),
    (2**63, True, True, False),
    ])
def test_typeof_constant_int_arg(val, int32_raises, int64_raises, uint64_raises):
    @ct.kernel
    def kernel(n: ct.Constant[int], x):
        t = n
        # Using `t` as a loop variable materializes the constant's type
        for i in range(2):
            t += 1
        # Attempt to store `t` in the arrays, possibly triggering an implicit cast error
        ct.scatter(x, (), t)

    def run(n, x_dtype, raises):
        x = torch.zeros((), dtype=x_dtype, device="cuda")
        with raises_if(raises, TileTypeError, match="cannot implicitly cast"):
            ct.launch(torch.cuda.current_stream(), (1,), kernel, (n, x))
            assert x.cpu().item() == n + 2

    # Control: `t` is at least int32, so attempting it to store in an uint16 array is an error
    run(val, torch.int16, True)

    run(val, torch.int32, int32_raises)
    run(val, torch.int64, int64_raises)
    run(val, torch.uint64, uint64_raises)


def test_typeof_constant_too_big():
    @ct.kernel
    def kernel(x):
        t = 18446744073709551616   # 2**64
        # Using `t` as a loop variable materializes the constant's type
        for i in range(2):
            t += 1
        ct.scatter(x, (), t)

    x = torch.zeros((), dtype=torch.uint64, device="cuda")
    with pytest.raises(TileValueError, match="is out of range of any supported integer type"):
        ct.launch(torch.cuda.current_stream(), (1,), kernel, (x,))


def test_allow_type_hints_on_assignment():
    @ct.kernel
    def kernel(x):
        a: int = ct.gather(x, ())  # the hint is intentionally wrong -- should still work
        a: float
        ct.scatter(x, (), a + 3.0)

    x = torch.ones((), dtype=torch.float32, device="cuda")
    ct.launch(torch.cuda.current_stream(), (1,), kernel, (x,))
    assert x.item() == 4.0
