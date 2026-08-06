# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import cuda.tile as ct
from util import assert_equal, require_blackwell_or_newer, require_rubin_cc107
from cuda.tile._bytecode import BytecodeVersion
from cuda.tile._exception import TileTypeError
from conftest import arithmetic_dtypes, dtype_id, float8_dtypes, requires_tileiras


def _shape(v):
    return (len(v),) + _shape(v[0]) if isinstance(v, tuple) else ()


@pytest.mark.parametrize("value", [
    (1,),
    (((1,),),),
    (1, 2),
    ((1, 2), (3, 4)),
    (((1, 2), (3, 4)), ((5, 6), (7, 8)))
])
def test_astile_shape(value):
    shape = _shape(value)

    @ct.kernel
    def kernel(X):
        t = ct.astile(value, dtype=ct.int32)
        idx = ct.static_eval((0,) * len(shape))
        ct.store(X, idx, t)

    x = torch.zeros(shape, dtype=torch.int32, device="cuda")
    ref = torch.tensor(value, dtype=torch.int32, device="cuda")
    ct.launch(torch.cuda.current_stream(), (1,), kernel, (x,))
    assert_equal(x, ref)


def test_astile_scalar_const():
    @ct.kernel
    def kernel(X):
        s = ct.astile(5, dtype=ct.int32)
        ct.store(X, 0, s)

    x = torch.zeros((1,), dtype=torch.int32, device="cuda")
    ref = torch.tensor([5], dtype=torch.int32, device="cuda")
    ct.launch(torch.cuda.current_stream(), (1,), kernel, (x,))
    assert_equal(x, ref)


@require_rubin_cc107()
@requires_tileiras(BytecodeVersion.V_13_4)
def test_float8_e5m3fnu_constructor():
    @ct.kernel
    def kernel(X):
        value = ct.float8_e5m3fnu(1.5)
        t = ct.astile((value,), dtype=ct.float8_e5m3fnu)
        t = t.astype(ct.float32)
        ct.store(X, (0,), t)

    x = torch.zeros((1,), dtype=torch.float32, device="cuda")
    ref = torch.tensor([1.5], dtype=torch.float32, device="cuda")
    ct.launch(torch.cuda.current_stream(), (1,), kernel, (x,))
    assert_equal(x, ref)


@pytest.mark.parametrize("dtype", arithmetic_dtypes + float8_dtypes + [torch.float64], ids=dtype_id)
def test_astile_dtype_const(dtype):
    value = (0, 1, 0, 4)

    @ct.kernel
    def kernel(X):
        t = ct.astile(value, dtype=dtype)
        ct.store(X, (0,), t)

    x = torch.zeros((4,), dtype=dtype, device="cuda")
    ref = torch.tensor(value, dtype=dtype, device="cuda")
    ct.launch(torch.cuda.current_stream(), (1,), kernel, (x,))
    assert_equal(x, ref)


bool_tuple = ((True, False), (True, True), (False, False), (True, False))


@pytest.mark.parametrize("dtype", arithmetic_dtypes + float8_dtypes + [torch.float64], ids=dtype_id)
def test_astile_bool_const(dtype):

    @ct.kernel
    def kernel(X):
        t = ct.astile(bool_tuple, dtype=dtype)
        ct.store(X, (0, 0), t)

    x = torch.zeros((4, 2), dtype=dtype, device="cuda")
    ref = torch.tensor(bool_tuple, dtype=dtype, device="cuda")
    ct.launch(torch.cuda.current_stream(), (1,), kernel, (x,))
    assert_equal(x, ref)


@pytest.mark.parametrize("value", [0, 1, (0.0,), (2.5,)], ids=str)
def test_astile_bool_from_numeric_const(value):
    @ct.kernel
    def kernel(X):
        t = ct.astile(value, dtype=ct.bool_)
        ct.store(X, (0,), t)

    x = torch.zeros((1,), dtype=torch.bool, device="cuda")
    ref = torch.tensor(value, dtype=torch.bool, device="cuda").reshape((1,))
    ct.launch(torch.cuda.current_stream(), (1,), kernel, (x,))
    assert_equal(x, ref)


@pytest.mark.parametrize("dtype", [
    pytest.param(ct.tfloat32, id="tf32"),
    pytest.param(ct.float4_e2m1fn, id="f4e2m1fn",
                 marks=(require_blackwell_or_newer(),
                        requires_tileiras(BytecodeVersion.V_13_3))),
])
def test_astile_tf32_f4_const(dtype):
    value = (1.0, 2.0, 4.0, 6.0)

    @ct.kernel
    def kernel(X):
        t = ct.astile(value, dtype=dtype).astype(ct.float32)
        ct.store(X, (0,), t)

    x = torch.zeros((4,), dtype=torch.float32, device="cuda")
    ref = torch.tensor(value, dtype=torch.float32, device="cuda")
    ct.launch(torch.cuda.current_stream(), (1,), kernel, (x,))
    assert_equal(x, ref)


def test_astile_nested_const():
    @ct.kernel
    def kernel(X):
        tp = (ct.int8(1), ct.astile(2, dtype=ct.int64),
              ct.full((), 3, dtype=ct.float32), 4)
        t = ct.astile(tp, dtype=ct.int32)
        ct.store(X, (0,), t)

    x = torch.zeros((4,), dtype=torch.float32, device="cuda")
    ref = torch.tensor((1, 2, 3, 4), dtype=torch.float32, device="cuda")
    ct.launch(torch.cuda.current_stream(), (1,), kernel, (x,))
    assert_equal(x, ref)


def test_astile_scalar_runtime():
    @ct.kernel
    def kernel(X, a: float):
        t = ct.astile(a, dtype=ct.int32)
        ct.store(X, 0, t)

    x = torch.zeros((1,), dtype=torch.int32, device="cuda")
    ct.launch(torch.cuda.current_stream(), (1,), kernel, (x, 42.0))
    assert x.item() == 42


def test_astile_1d_runtime():
    @ct.kernel
    def kernel(X, a: bool):
        t = ct.astile((a,), dtype=ct.int32)
        ct.store(X, 0, t)

    x = torch.zeros((1,), dtype=torch.int32, device="cuda")
    ref = torch.tensor([1], dtype=torch.int32, device="cuda")
    ct.launch(torch.cuda.current_stream(), (1,), kernel, (x, True))
    assert_equal(x, ref)


def test_astile_2d_runtime():
    @ct.kernel
    def kernel(X, a: int, b: int, c: float, d: bool):
        t = ct.astile(((a, b), (c, d)), dtype=ct.bool_)
        ct.store(X, (0, 0), t)

    x = torch.zeros((2, 2), dtype=torch.bool, device="cuda")
    ref = torch.tensor([[0, 1], [0.0, True]], dtype=torch.bool, device="cuda")
    ct.launch(torch.cuda.current_stream(), (1,), kernel, (x, 0, 1, 0.0, True))
    assert_equal(x, ref)


def test_astile_3d_mixed():
    @ct.kernel
    def kernel(X, a: int, b: int, c: float, d: bool):
        t = ct.astile((((1, a), (2, b)), ((3, c), (4, d))), dtype=ct.float32)
        ct.store(X, (0, 0, 0), t)

    x = torch.zeros((2, 2, 2), dtype=torch.float32, device="cuda")
    ct.launch(torch.cuda.current_stream(), (1,), kernel, (x, 10, 20, 3.14, False))
    assert_equal(x, torch.tensor([[[1, 10], [2, 20]], [[3, 3.14], [4, False]]],
                                 dtype=torch.float32, device="cuda"))


def test_astile_empty_tuple():
    @ct.kernel
    def kernel(X):
        t = ct.astile((), dtype=ct.int32)
        ct.store(X, 0, t)

    x = torch.zeros((1,), dtype=torch.int32, device="cuda")
    with pytest.raises(TileTypeError, match="Tuple length 0 at value is not a power of 2"):
        ct.launch(torch.cuda.current_stream(), (1,), kernel, (x,))


def test_astile_non_scalar_leaf():
    @ct.kernel
    def kernel(X):
        t = ct.astile((1, "a"), dtype=ct.int32)
        ct.store(X, (0, 0), t)

    x = torch.zeros((2, 2), dtype=torch.int32, device="cuda")
    with pytest.raises(TileTypeError, match=r"Expected scalar elements at value\[1\]"):
        ct.launch(torch.cuda.current_stream(), (1,), kernel, (x,))


def test_astile_2d_tile_leaf():
    @ct.kernel
    def kernel(X):
        leaf = ct.full((2, 2), 1, dtype=ct.int32)
        t = ct.astile(((1,), (leaf,)), dtype=ct.int32)
        ct.store(X, (0, 0), t)

    x = torch.zeros((2, 2), dtype=torch.int32, device="cuda")
    with pytest.raises(TileTypeError, match=r"Expected scalar elements at value\[1\]\[0\]"):
        ct.launch(torch.cuda.current_stream(), (1,), kernel, (x,))


ragged_tuple = ((1, 2), (3,))


def test_astile_ragged_shape():
    @ct.kernel
    def kernel(X):
        t = ct.astile(ragged_tuple, dtype=ct.int32)
        ct.store(X, (0, 0), t)

    x = torch.zeros((2, 2), dtype=torch.int32, device="cuda")
    with pytest.raises(TileTypeError, match=r"Tuple has non-uniform inner shapes at value"):
        ct.launch(torch.cuda.current_stream(), (1,), kernel, (x,))


def test_astile_top_level_not_supported():
    @ct.kernel
    def kernel():
        ct.astile(ct.full((4,), 1, dtype=ct.int32), dtype=ct.int32)
    with pytest.raises(TileTypeError,
                       match=r"Expected a scalar or \(possibly nested\) tuple of scalars"):
        ct.launch(torch.cuda.current_stream(), (1,), kernel, ())
