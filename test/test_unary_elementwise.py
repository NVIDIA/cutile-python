# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import math
from torch.testing import make_tensor

from util import launch_unary, assert_equal, assert_close, jit_kernel, filecheck, get_bytecode
from conftest import float_dtypes, int_dtypes, bool_dtypes, dtype_id
from cuda.tile._exception import TileTypeError
from cuda.tile._numeric_semantics import RoundingMode as RMd


# === Helpers ===
kernel_cache = {}


array_kernel_template = """
def {name}(x, y, TILE: ct.Constant[int]):
    bid = ct.bid(0)
    tx = ct.load(x, index=(bid,), shape=(TILE,))
    {body}
    ct.store(y, index=(bid,), tile=ty)"""


def array_kernel(name: str, body: str, tmp_path, globals: dict = None):
    name = 'array_' + name
    source = array_kernel_template.format(name=name, body=body)
    if source not in kernel_cache:
        kernel_cache[source] = jit_kernel(name, source, tmp_path, globals)
    return kernel_cache[source]


scalar_kernel_template = """
def {name}(x, y, TILE: ct.Constant[int]):
    bid = ct.bid(0)
    {body}
    ty = ct.full((TILE,), c, dtype=y.dtype)
    ct.store(y, index=(bid,), tile=ty)"""


def scalar_kernel(name: str, body: str, tmp_path):
    name = 'scalar_' + name
    source = scalar_kernel_template.format(name=name, body=body)
    if source not in kernel_cache:
        kernel_cache[source] = jit_kernel(name, source, tmp_path)
    return kernel_cache[source]


const_scalar_kernel_template = """
def {name}(x: ct.Constant[{dtype}], y, TILE: ct.Constant[int]):
    bid = ct.bid(0)
    {body}
    ty = ct.full((TILE,), c, dtype=y.dtype)
    ct.store(y, index=(bid,), tile=ty)"""


def const_scalar_kernel(name: str, dtype: str, body: str, tmp_path):
    name = 'const_scalar_' + name
    source = const_scalar_kernel_template.format(name=name, dtype=dtype, body=body)
    if source not in kernel_cache:
        kernel_cache[source] = jit_kernel(name, source, tmp_path)
    return kernel_cache[source]


@pytest.fixture
def shape():
    return (512, )


@pytest.fixture
def tile():
    return 64


# === End of Helpers ===


@pytest.mark.parametrize("dtype", bool_dtypes + int_dtypes + float_dtypes, ids=dtype_id)
@pytest.mark.parametrize("op", ['sqrt', 'rsqrt'], ids=['sqrt', 'rsqrt'])
def test_array_root_ops(shape, tile, dtype, op, tmp_path):
    x = make_tensor(shape, dtype=dtype, low=0, high=100, device='cuda')
    y_ref = getattr(torch, op)(x)
    y = torch.zeros_like(y_ref, device="cuda")
    kernel = array_kernel(op, f"ty = ct.{op}(tx)", tmp_path)
    launch_unary(kernel, x, y, tile)
    assert_equal(y, y_ref)


@pytest.mark.use_mlir
@pytest.mark.parametrize("dtype", bool_dtypes + int_dtypes + float_dtypes, ids=dtype_id)
@pytest.mark.parametrize("op", ['sqrt', 'rsqrt'], ids=['sqrt', 'rsqrt'])
@pytest.mark.parametrize("flush_to_zero", [True, False])
def test_array_root_ops_flush_to_zero(shape, tile, dtype, op, flush_to_zero, tmp_path):
    should_raise = flush_to_zero and dtype in float_dtypes and dtype != torch.float32
    x = make_tensor(shape, dtype=dtype, low=0, high=100, device='cuda')
    y_ref = getattr(torch, op)(x)
    y = torch.zeros_like(y_ref, device="cuda")
    kernel = array_kernel(f"{op}_flush_to_zero",
                          f"ty = ct.{op}(tx, flush_to_zero={flush_to_zero})",
                          tmp_path)
    if should_raise:
        with pytest.raises(TileTypeError,
                           match=r"Flush to zero can only be used for float32 type"):
            launch_unary(kernel, x, y, tile)
    else:
        bytecode = get_bytecode(kernel, (x, y, tile))
        if flush_to_zero:
            check_directive = f"// CHECK: %[[RES:.*]] = {op} %[[A:.*]] flush_to_zero :"
        else:
            check_directive = f"// CHECK: %[[RES:.*]] = {op} %[[A:.*]]{{{{[[:space:]]*}}}}:"
        filecheck(bytecode, check_directive)
        launch_unary(kernel, x, y, tile)


@pytest.mark.use_mlir
@pytest.mark.parametrize("dtype", float_dtypes, ids=dtype_id)
@pytest.mark.parametrize("rounding_mode",
                         [RMd.RN, RMd.RZ, RMd.RM, RMd.RP, RMd.FULL, RMd.APPROX, RMd.RZI])
def test_array_sqrt_rounding_mode(shape, tile, dtype, rounding_mode, tmp_path):
    should_raise_rounding_mode = rounding_mode in [RMd.RZI, RMd.FULL]
    should_raise_dtype = (rounding_mode in [RMd.APPROX]
                          and dtype in float_dtypes and dtype != torch.float32)
    x = make_tensor(shape, dtype=dtype, low=0, high=100, device='cuda')
    y = torch.zeros_like(x, device="cuda")
    kernel = array_kernel("sqrt_rounding_mode",
                          f"ty = ct.sqrt(tx, rounding_mode={rounding_mode})", tmp_path,
                          globals={"RoundingMode": RMd})
    if should_raise_rounding_mode:
        with pytest.raises(TileTypeError,
                           match=fr"Rounding mode {rounding_mode.value} is not supported"):
            launch_unary(kernel, x, y, tile)
    elif should_raise_dtype:
        with pytest.raises(TileTypeError,
                           match=fr"Rounding mode {rounding_mode.value} can only be used for "
                           "float32 type"):
            launch_unary(kernel, x, y, tile)
    else:
        bytecode = get_bytecode(kernel, (x, y, tile))
        if rounding_mode is RMd.RN:
            # Rmd.RN as the default rounding mode is not included in the mlir text
            check_directive = "// CHECK-NOT: rounding<{{[^>]*}}>"
        else:
            check_directive = (
                f"// CHECK: %[[RES:.*]] = sqrt %[[A:.*]] rounding<{rounding_mode.value}>"
            )
        filecheck(bytecode, check_directive)
        launch_unary(kernel, x, y, tile)


@pytest.mark.parametrize("op", ['log', 'log2'], ids=['log', 'log2'])
@pytest.mark.parametrize("dtype", bool_dtypes + int_dtypes + float_dtypes, ids=dtype_id)
def test_array_log(shape, tile, dtype, op, tmp_path):
    x = make_tensor(shape, dtype=dtype, low=0, high=100, device='cuda')
    y_ref = getattr(torch, op)(x)
    y = torch.zeros_like(y_ref, device="cuda")
    kernel = array_kernel('log', f"ty = ct.{op}(tx)", tmp_path)
    launch_unary(kernel, x, y, tile)
    assert_equal(y, y_ref)


@pytest.mark.parametrize("op", ['sin', 'cos', 'tan', 'sinh', 'cosh', 'tanh'],
                         ids=['sin', 'cos', 'tan', 'sinh', 'cosh', 'tanh'])
@pytest.mark.parametrize("dtype", bool_dtypes + int_dtypes + float_dtypes, ids=dtype_id)
def test_array_trig(shape, tile, dtype, op, tmp_path):
    x = make_tensor(shape, dtype=dtype, device='cuda')
    y_ref = getattr(torch, op)(x)
    y = torch.zeros_like(y_ref, device="cuda")
    kernel = array_kernel('trig', f"ty = ct.{op}(tx)", tmp_path)
    launch_unary(kernel, x, y, tile)
    assert_equal(y, y_ref)


@pytest.mark.parametrize("neg_func", ['-', 'ct.negative'])
@pytest.mark.parametrize("dtype", bool_dtypes + int_dtypes + float_dtypes, ids=dtype_id)
def test_array_neg(shape, tile, dtype, tmp_path, neg_func):
    x = make_tensor(shape, dtype=dtype, device='cuda')
    y_ref = -x.to(torch.int32) if dtype == torch.bool else -x
    y = torch.zeros_like(y_ref, device="cuda")
    kernel = array_kernel('neg',
                          "ty = -tx" if neg_func == "-" else f"ty = {neg_func}(tx)",
                          tmp_path)
    launch_unary(kernel, x, y, tile)
    assert_equal(y, y_ref)


@pytest.mark.parametrize("is_constant", [False, True])
@pytest.mark.parametrize("dtype", int_dtypes + float_dtypes, ids=dtype_id)
def test_scalar_neg(shape, tile, is_constant, dtype, tmp_path):
    if dtype in int_dtypes:
        x = 5
        dtype_str = "int"
    else:
        x = 5.0
        dtype_str = "float"
    y = torch.zeros(shape, dtype=dtype, device='cuda')
    if not is_constant:
        kernel = scalar_kernel('neg', 'c = -x', tmp_path)
    else:
        kernel = const_scalar_kernel('neg', dtype_str, 'c = -x', tmp_path)
    launch_unary(kernel, x, y, tile)
    assert_equal(y, -x)


@pytest.mark.parametrize("dtype", bool_dtypes + int_dtypes + float_dtypes, ids=dtype_id)
def test_array_pos(shape, tile, dtype, tmp_path):
    x = make_tensor(shape, dtype=dtype, device='cuda')
    y_ref = x.to(torch.int32) if dtype == torch.bool else +x
    y = torch.zeros_like(y_ref, device="cuda")
    kernel = array_kernel('pos', "ty = +tx", tmp_path)
    launch_unary(kernel, x, y, tile)
    assert_equal(y, y_ref)


@pytest.mark.parametrize("dtype", bool_dtypes + int_dtypes + float_dtypes, ids=dtype_id)
def test_array_abs(shape, tile, dtype, tmp_path):
    x = make_tensor(shape, dtype=dtype, device='cuda')
    y = torch.zeros_like(x, device="cuda")
    kernel = array_kernel('abs', "ty = abs(tx)", tmp_path)
    launch_unary(kernel, x, y, tile)
    assert_equal(y, abs(x))


@pytest.mark.parametrize("dtype", bool_dtypes + int_dtypes + float_dtypes, ids=dtype_id)
def test_array_ct_abs(shape, tile, dtype, tmp_path):
    x = make_tensor(shape, dtype=dtype, device='cuda')
    y = torch.zeros_like(x, device="cuda")
    kernel = array_kernel('ct_abs', "ty = ct.abs(tx)", tmp_path)
    launch_unary(kernel, x, y, tile)
    assert_equal(y, abs(x))


@pytest.mark.parametrize("is_constant", [False, True])
@pytest.mark.parametrize("dtype", int_dtypes + float_dtypes, ids=dtype_id)
def test_scalar_abs(shape, tile, is_constant, dtype, tmp_path):
    if dtype in int_dtypes:
        x = -5
        dtype_str = "int"
    else:
        x = -5.0
        dtype_str = "float"
    y = torch.zeros(shape, dtype=dtype, device='cuda')
    if not is_constant:
        kernel = scalar_kernel('abs', 'c = abs(x)', tmp_path)
    else:
        kernel = const_scalar_kernel('abs', dtype_str, 'c = abs(x)', tmp_path)
    launch_unary(kernel, x, y, tile)
    assert_equal(y, abs(x))


@pytest.mark.parametrize("is_constant", [False, True])
@pytest.mark.parametrize("dtype", int_dtypes + float_dtypes, ids=dtype_id)
def test_scalar_ct_abs(shape, tile, is_constant, dtype, tmp_path):
    if dtype in int_dtypes:
        x = -5
        dtype_str = "int"
    else:
        x = -5.0
        dtype_str = "float"
    y = torch.zeros(shape, dtype=dtype, device='cuda')
    if not is_constant:
        kernel = scalar_kernel('ct_abs', 'c = ct.abs(x)', tmp_path)
    else:
        kernel = const_scalar_kernel('ct_abs', dtype_str, 'c = ct.abs(x)', tmp_path)
    launch_unary(kernel, x, y, tile)
    assert_equal(y, abs(x))


@pytest.mark.parametrize("bitwise_not_func", ['~', 'ct.bitwise_not'])
@pytest.mark.parametrize("dtype", int_dtypes + bool_dtypes, ids=dtype_id)
def test_array_bitwise_not(shape, tile, dtype, tmp_path, bitwise_not_func):
    x = make_tensor(shape, dtype=dtype, device='cuda')
    y = torch.zeros_like(x, device="cuda")
    kernel = array_kernel('bitwise_not',
                          "ty = ~tx" if bitwise_not_func == "~" else f"ty = {bitwise_not_func}(tx)",
                          tmp_path)
    launch_unary(kernel, x, y, tile)
    assert_equal(y, ~x)


@pytest.mark.parametrize("is_constant", [False, True])
@pytest.mark.parametrize("dtype", int_dtypes, ids=dtype_id)
def test_scalar_bitwise_not(shape, tile, is_constant, dtype, tmp_path):
    x = 5
    y = torch.zeros(shape, dtype=torch.int32, device='cuda')
    if not is_constant:
        kernel = scalar_kernel('bitwise_not', 'c = ~x', tmp_path)
    else:
        kernel = const_scalar_kernel('bitwise_not', "int", 'c = ~x', tmp_path)
    launch_unary(kernel, x, y, tile)
    assert_equal(y, ~x)


@pytest.mark.parametrize("op", ['exp', 'exp2'],
                         ids=['exp', 'exp2'])
@pytest.mark.parametrize("dtype", bool_dtypes + int_dtypes + float_dtypes, ids=dtype_id)
def test_array_exp(shape, tile, dtype, op, tmp_path):
    x = make_tensor(shape, dtype=dtype, device='cuda')
    y_ref = getattr(torch, op)(x)
    y = torch.zeros_like(y_ref, device="cuda")
    kernel = array_kernel('exp', f"ty = ct.{op}(tx)", tmp_path)
    launch_unary(kernel, x, y, tile)
    assert_close(y, y_ref)


@pytest.mark.use_mlir
@pytest.mark.parametrize("dtype", float_dtypes, ids=dtype_id)
@pytest.mark.parametrize("flush_to_zero", [True, False])
def test_array_exp2_flush_to_zero(shape, tile, dtype, flush_to_zero, tmp_path):
    should_raise = flush_to_zero and (dtype != torch.float32)
    x = make_tensor(shape, dtype=dtype, device='cuda')
    y = torch.zeros_like(x, device="cuda")
    kernel = array_kernel("exp2_flush_to_zero",
                          f"ty = ct.exp2(tx, flush_to_zero={flush_to_zero})", tmp_path)
    if should_raise:
        with pytest.raises(TileTypeError,
                           match=r"Flush to zero can only be used for float32 type"):
            launch_unary(kernel, x, y, tile)
    else:
        bytecode = get_bytecode(kernel, (x, y, tile))
        if flush_to_zero:
            check_directive = "// CHECK: %[[RES:.*]] = exp2 %[[A:.*]] flush_to_zero :"
        else:
            check_directive = "// CHECK: %[[RES:.*]] = exp2 %[[A:.*]]{{[[:space:]]*}}:"
        filecheck(bytecode, check_directive)
        launch_unary(kernel, x, y, tile)


@pytest.mark.parametrize("dtype", float_dtypes, ids=dtype_id)
@pytest.mark.parametrize("op", ['floor', 'ceil'], ids=['floor', 'ceil'])
def test_array_rounding(shape, tile, dtype, op, tmp_path):
    x = make_tensor(shape, dtype=dtype, device='cuda')
    y = torch.zeros_like(x, device="cuda")
    kernel = array_kernel(op, f"ty = ct.{op}(tx)", tmp_path)
    launch_unary(kernel, x, y, tile)
    assert_equal(y, getattr(torch, op)(x))


@pytest.mark.parametrize("is_constant", [False, True])
@pytest.mark.parametrize("dtype", float_dtypes, ids=dtype_id)
@pytest.mark.parametrize("op", ['floor', 'ceil'], ids=['floor', 'ceil'])
def test_scalar_rounding(shape, tile, is_constant, dtype, op, tmp_path):
    if dtype in int_dtypes:
        x = -5
        dtype_str = "int"
    else:
        x = -5.5
        dtype_str = "float"
    y = torch.zeros(shape, dtype=dtype, device='cuda')
    if not is_constant:
        kernel = scalar_kernel(op, f'c = ct.{op}(x)', tmp_path)
    else:
        kernel = const_scalar_kernel(op, dtype_str, f'c = ct.{op}(x)', tmp_path)
    launch_unary(kernel, x, y, tile)
    assert_equal(y, getattr(math, op)(x))
