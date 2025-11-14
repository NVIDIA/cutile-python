# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from math import ceil

import torch
from torch.testing import make_tensor

import cuda.tile as ct
from util import (
    launch_binary, assert_equal, assert_close, jit_kernel, filecheck,
    get_bytecode, raises_if
)
from conftest import float_dtypes, int_dtypes, bool_dtypes, dtype_id
from cuda.tile._exception import TileTypeError
from cuda.tile._ir.typing_support import to_dtype
from cuda.tile._numeric_semantics import RoundingMode as RMd


# === Helpers ===
kernel_cache = {}


array_kernel_template = """
def {name}(x, y, z, TILE: ct.Constant[int]):
    bid = ct.bid(0)
    tx = ct.load(x, index=(bid,), shape=(TILE,))
    ty = ct.load(y, index=(bid,), shape=(TILE,))
    {body}
    ct.store(z, index=(bid,), tile=tz)"""


def array_kernel(name: str, body: str, tmp_path, globals: dict = None):
    name = 'array_' + name
    source = array_kernel_template.format(name=name, body=body)
    if source not in kernel_cache:
        kernel_cache[source] = jit_kernel(name, source, tmp_path, globals)
    return kernel_cache[source]


scalar_kernel_template = """
def {name}(x, y, z, TILE: ct.Constant[int]):
    bid = ct.bid(0)
    {body}
    tz = ct.full((TILE,), c, dtype=z.dtype)
    ct.store(z, index=(bid,), tile=tz)"""


def scalar_kernel(name: str, body: str, tmp_path):
    name = 'scalar_' + name
    source = scalar_kernel_template.format(name=name, body=body)
    if source not in kernel_cache:
        kernel_cache[source] = jit_kernel(name, source, tmp_path)
    return kernel_cache[source]


const_scalar_kernel_template = """
def {name}(x: ct.Constant[{dtype_x}], y: ct.Constant[{dtype_y}], z, TILE: ct.Constant[int]):
    bid = ct.bid(0)
    {body}
    tz = ct.full((TILE,), c, dtype=z.dtype)
    ct.store(z, index=(bid,), tile=tz)"""


def const_scalar_kernel(name: str, dtype_x: str, dtype_y: str, body: str, tmp_path):
    name = 'const_scalar_' + name
    source = const_scalar_kernel_template.format(name=name,
                                                 dtype_x=dtype_x,
                                                 dtype_y=dtype_y,
                                                 body=body)
    if source not in kernel_cache:
        kernel_cache[source] = jit_kernel(name, source, tmp_path)
    return kernel_cache[source]


array_scalar_kernel_template = """
def {name}(x, y, z, TILE: ct.Constant[int]):
    bid = ct.bid(0)
    tx = ct.load(x, index=(bid,), shape=(TILE,))
    {body}
    ct.store(z, index=(bid,), tile=tz)"""


def array_scalar_kernel(name: str, body: str, tmp_path, globals: dict = None):
    name = 'array_scalar_' + name
    source = array_scalar_kernel_template.format(name=name, body=body)
    if source not in kernel_cache:
        kernel_cache[source] = jit_kernel(name, source, tmp_path, globals)
    return kernel_cache[source]


@pytest.fixture
def shape():
    return (512, )


@pytest.fixture
def tile():
    return 64


# === End of Helpers ===


core_arithmetic_cases = [
    pytest.param("+", "ct.add", id="+"),
    pytest.param("-", "ct.sub", id="-"),
    pytest.param("*", "ct.mul", id="*"),
]


@pytest.mark.parametrize("x_dtype", int_dtypes + float_dtypes, ids=dtype_id)
@pytest.mark.parametrize("y_dtype", int_dtypes + float_dtypes, ids=dtype_id)
@pytest.mark.parametrize("op_symbol, op_func", core_arithmetic_cases)
def test_array_core_arithmetic(shape, tile, x_dtype, y_dtype, tmp_path, op_symbol, op_func):
    x = make_tensor(shape, dtype=x_dtype, device='cuda')
    y = make_tensor(shape, dtype=y_dtype, device='cuda')
    should_raise = {x_dtype, y_dtype} == {torch.float16, torch.bfloat16}
    z = torch.zeros_like(x, device="cuda").to(torch.promote_types(x.dtype, y.dtype))
    for expr in [f"tz = tx {op_symbol} ty", f"tz = {op_func}(tx, ty)"]:
        kernel = array_kernel("core_arithmetic", expr, tmp_path)
        if should_raise:
            with pytest.raises(TileTypeError,
                               match=r"Implicit promotion of .* and .* is not supported"):
                launch_binary(kernel, x, y, z, tile)
        else:
            launch_binary(kernel, x, y, z, tile)
            assert_equal(z, eval(f"x {op_symbol} y"))


@pytest.mark.parametrize("is_constant", [False, True])
def test_scalar_add(shape, tile, is_constant, float_dtype, tmp_path):
    z = torch.zeros(shape, dtype=float_dtype, device='cuda')
    if not is_constant:
        kernel = scalar_kernel("add", "c = x + y", tmp_path)
    else:
        kernel = const_scalar_kernel("add", "float", "float", "c = x + y", tmp_path)
    launch_binary(kernel, 1, 1.0, z, tile)
    assert_equal(z, 2.0)


@pytest.mark.parametrize("dtype", int_dtypes + float_dtypes, ids=dtype_id)
def test_array_scalar_add(shape, tile, dtype, tmp_path):
    x = make_tensor(shape, dtype=dtype, device='cuda')
    y = 5.0
    res_dtype = torch.promote_types(x.dtype, torch.float32)
    z = torch.zeros_like(x, dtype=res_dtype)
    kernel = array_scalar_kernel("add", "tz = tx + y", tmp_path)
    launch_binary(kernel, x, y, z, tile)
    assert_equal(z, (x.to(res_dtype) + y))


@ct.kernel
def explicit_broadcast_add(x, y, z, TILE: ct.Constant[int]):
    bidx = ct.bid(0)
    bidy = ct.bid(1)
    tx = ct.load(x, index=(bidx,), shape=(TILE,))
    ty = ct.load(y, index=(bidy,), shape=(TILE,))
    tz = tx[:, None] + ty[None, :]
    ct.store(z, index=(bidx, bidy), tile=tz)


@ct.kernel
def implicit_broadcast_add(x, y, z, TILE: ct.Constant[int]):
    bidx = ct.bid(0)
    bidy = ct.bid(1)
    tx = ct.load(x, index=(bidx,), shape=(TILE,))
    ty = ct.load(y, index=(bidy,), shape=(TILE,))
    tz = tx[:, None] + ty[None, :]
    ct.store(z, index=(bidx, bidy), tile=tz)


@pytest.mark.parametrize("fn", [explicit_broadcast_add,
                                implicit_broadcast_add])
def test_broadcast(shape, tile, float_dtype, fn):
    x = torch.randn(shape, dtype=float_dtype, device='cuda')
    y = torch.randn(shape, dtype=float_dtype, device='cuda')
    ref = x[:, None] + y[None, :]
    z = torch.zeros_like(ref)
    launch_binary(fn, x, y, z, tile)
    assert_equal(z, ref)


@pytest.mark.use_mlir
@pytest.mark.parametrize("dtype", float_dtypes + int_dtypes, ids=dtype_id)
@pytest.mark.parametrize("rounding_mode",
                         [RMd.RN, RMd.RZ, RMd.RM, RMd.RP, RMd.FULL, RMd.APPROX, RMd.RZI])
@pytest.mark.parametrize("op_func, tile_op",
                         [("ct.add", "addf"), ("ct.sub", "subf"), ("ct.mul", "mulf")])
def test_array_core_arithmetic_rounding_mode(
    tile, dtype, rounding_mode, op_func, tile_op, tmp_path
):
    should_raise_rounding_mode = rounding_mode in [RMd.FULL, RMd.APPROX, RMd.RZI]
    should_raise_dtype = dtype in int_dtypes
    x = make_tensor((1,), dtype=dtype, device='cuda')
    y = make_tensor((1,), dtype=dtype, device='cuda')
    z = torch.zeros_like(x, device="cuda")
    kernel = array_kernel("core_arithmetic_rounding_mode",
                          f"tz = {op_func}(tx, ty, rounding_mode={rounding_mode})",
                          tmp_path,
                          globals={"RoundingMode": RMd})
    if should_raise_rounding_mode:
        with pytest.raises(TileTypeError,
                           match=fr"Rounding mode {rounding_mode.value} is not supported"):
            launch_binary(kernel, x, y, z, tile)
    elif should_raise_dtype:
        with pytest.raises(TileTypeError,
                           match=r"Rounding mode can only be used for float types"):
            launch_binary(kernel, x, y, z, tile)
    else:
        bytecode = get_bytecode(kernel, (x, y, z, tile))
        if rounding_mode is RMd.RN:
            # Rmd.RN as the default rounding mode is not included in the mlir text
            check_directive = "// CHECK-NOT: rounding<{{[^>]*}}>"
        else:
            check_directive = (
                f"// CHECK: %[[RES:.*]] = {tile_op} %[[A:.*]], %[[B:.*]] "
                f"rounding<{rounding_mode.value}>"
            )
        filecheck(bytecode, check_directive)
        launch_binary(kernel, x, y, z, tile)


@pytest.mark.use_mlir
@pytest.mark.parametrize("dtype", float_dtypes, ids=dtype_id)
@pytest.mark.parametrize("flush_to_zero", [True, False])
@pytest.mark.parametrize("op_func, tile_op",
                         [("ct.add", "addf"), ("ct.sub", "subf"), ("ct.mul", "mulf")])
def test_core_arithmetic_flush_to_zero(tile, dtype, flush_to_zero, op_func, tile_op, tmp_path):
    should_raise_dtype = flush_to_zero and (dtype != torch.float32)
    x = make_tensor((1,), dtype=dtype, device='cuda')
    y = make_tensor((1,), dtype=dtype, device='cuda')
    z = torch.zeros_like(x, device="cuda")
    kernel = array_kernel("core_arithmetic_flush_to_zero",
                          f"tz = {op_func}(tx, ty, flush_to_zero={flush_to_zero})",
                          tmp_path)
    if should_raise_dtype:
        with pytest.raises(TileTypeError,
                           match=r"Flush to zero can only be used for float32 type"):
            launch_binary(kernel, x, y, z, tile)
    else:
        bytecode = get_bytecode(kernel, (x, y, z, tile))
        if flush_to_zero:
            check_directive = (
                f"// CHECK: %[[RES:.*]] = {tile_op} %[[A:.*]], %[[B:.*]] flush_to_zero :"
            )
        else:
            check_directive = (
                f"// CHECK: %[[RES:.*]] = {tile_op} %[[A:.*]], %[[B:.*]]{{{{[[:space:]]*}}}}:"
            )
        filecheck(bytecode, check_directive)
        launch_binary(kernel, x, y, z, tile)


compare_cases = [
    pytest.param(">", "ct.greater", id="gt"),
    pytest.param("<", "ct.less", id="lt"),
    pytest.param(">=", "ct.greater_equal", id="ge"),
    pytest.param("<=", "ct.less_equal", id="le"),
    pytest.param("==", "ct.equal", id="eq"),
    pytest.param("!=", "ct.not_equal", id="ne"),
]


@pytest.mark.parametrize("op_symbol, op_func", compare_cases)
@pytest.mark.parametrize("dtype", bool_dtypes + int_dtypes + float_dtypes, ids=dtype_id)
def test_array_compare(shape, tile, dtype, op_symbol, op_func, tmp_path):
    x = make_tensor(shape, dtype=dtype, device='cuda')
    y = torch.zeros_like(x)
    y[::2] = x[::2]
    z = torch.zeros_like(x).to(torch.bool)
    for expr in [f"tz = tx {op_symbol} ty", f"tz = {op_func}(tx, ty)"]:
        kernel = array_kernel('cmp', expr, tmp_path)
        ref = eval(f'x {op_symbol} y')
        launch_binary(kernel, x, y, z, tile)
        assert_equal(z, ref)


@pytest.mark.parametrize("max_func", ["max", "ct.maximum"])
@pytest.mark.parametrize("dtype", int_dtypes + float_dtypes, ids=dtype_id)
def test_array_max(shape, tile, dtype, tmp_path, max_func):
    x = make_tensor(shape, dtype=dtype, device='cuda')
    y = torch.zeros_like(x)
    z = torch.zeros_like(x)
    ref = torch.maximum(x, y)
    kernel = array_kernel("max", f"tz = {max_func}(tx, ty)", tmp_path)
    launch_binary(kernel, x, y, z, tile)
    assert_equal(z, ref)


@pytest.mark.parametrize("max_func", ["max", "ct.maximum"])
@pytest.mark.parametrize("is_constant", [False, True])
def test_scalar_max(shape, tile, is_constant, tmp_path, max_func):
    x = 1
    y = 4.2
    z = torch.zeros(shape, dtype=torch.float32, device='cuda')
    if not is_constant:
        kernel = scalar_kernel("max", f"c = {max_func}(x, y)", tmp_path)
    else:
        kernel = const_scalar_kernel("max", "int", "float", f"c = {max_func}(x, y)", tmp_path)
    launch_binary(kernel, x, y, z, tile)
    assert_equal(z, max(x, y))


@pytest.mark.parametrize("max_func", ["max", "ct.maximum"])
@pytest.mark.parametrize("dtype", int_dtypes + float_dtypes, ids=dtype_id)
def test_array_scalar_max(shape, tile, dtype, tmp_path, max_func):
    x = make_tensor(shape, dtype=dtype, device='cuda')
    y = 5.0
    res_dtype = torch.promote_types(dtype, torch.float32)
    ref = torch.maximum(x.to(res_dtype), torch.tensor(y, device="cuda"))
    z = torch.zeros_like(ref)
    kernel = array_scalar_kernel("max", f"tz = {max_func}(tx, y)", tmp_path)
    launch_binary(kernel, x, y, z, tile)
    assert_equal(z, ref)


@pytest.mark.parametrize("min_func", ["min", "ct.minimum"])
@pytest.mark.parametrize("dtype", int_dtypes + float_dtypes, ids=dtype_id)
def test_array_min(shape, tile, dtype, tmp_path, min_func):
    x = make_tensor(shape, dtype=dtype, device='cuda')
    y = torch.zeros_like(x)
    z = torch.zeros_like(x)
    ref = torch.minimum(x, y)
    kernel = array_kernel('min', f'tz = {min_func}(tx, ty)', tmp_path)
    launch_binary(kernel, x, y, z, tile)
    assert_equal(z, ref)


@pytest.mark.use_mlir
@pytest.mark.parametrize("op_func, tile_op", [("ct.maximum", "maxf"), ("ct.minimum", "minf")])
@pytest.mark.parametrize("dtype", float_dtypes, ids=dtype_id)
@pytest.mark.parametrize("flush_to_zero", [True, False])
def test_array_maxmin_flush_to_zero(shape, tile, dtype, op_func, tile_op, flush_to_zero, tmp_path):
    should_raise = flush_to_zero and (dtype != torch.float32)
    x = make_tensor(shape, dtype=dtype, device='cuda')
    y = torch.zeros_like(x)
    z = torch.zeros_like(x)
    kernel = array_kernel('min', f'tz = {op_func}(tx, ty, flush_to_zero={flush_to_zero})', tmp_path)
    if should_raise:
        with pytest.raises(TileTypeError,
                           match=r"Flush to zero can only be used for float32 type"):
            launch_binary(kernel, x, y, z, tile)
    else:
        bytecode = get_bytecode(kernel, (x, y, z, tile))
        if flush_to_zero:
            check_directive = (
                f"// CHECK: %[[RES:.*]] = {tile_op} %[[A:.*]], %[[B:.*]] flush_to_zero :"
            )
        else:
            check_directive = (
                f"// CHECK: %[[RES:.*]] = {tile_op} %[[A:.*]], %[[B:.*]]{{{{[[:space:]]*}}}}:"
            )
        filecheck(bytecode, check_directive)
        launch_binary(kernel, x, y, z, tile)


@pytest.mark.parametrize("is_constant", [False, True])
@pytest.mark.parametrize("x", [100, 100.0])
@pytest.mark.parametrize("y", [23, 2.3])
def test_scalar_mod(x, y, shape, tile, is_constant, tmp_path):
    z = torch.zeros(shape, dtype=torch.int32, device='cuda')
    if not is_constant:
        kernel = scalar_kernel('mod', 'c = x % y', tmp_path)
    else:
        kernel = const_scalar_kernel('mod',
                                     type(x).__name__, type(y).__name__,
                                     "c = x % y",
                                     tmp_path)
    launch_binary(kernel, x, y, z, tile)
    assert_equal(z, x % y)


@pytest.mark.parametrize("y", [23, 2.3, -23, -2.3, -1, 1])
def test_array_scalar_mod(y, shape, tile, tmp_path):
    x = torch.randint(-100, 100, shape).to('cuda')
    ref = x % y
    z = torch.zeros_like(x).to(ref.dtype)
    kernel = array_scalar_kernel('mod', 'tz = tx % y', tmp_path)
    launch_binary(kernel, x, y, z, tile)
    assert_equal(z, ref)


@pytest.mark.parametrize("mod_func", ["%", "ct.mod"])
@pytest.mark.parametrize("x_dtype", int_dtypes + float_dtypes, ids=dtype_id)
@pytest.mark.parametrize("y_dtype", int_dtypes + float_dtypes, ids=dtype_id)
def test_array_mod(shape, tile, x_dtype, y_dtype, tmp_path, mod_func):
    x = (torch.rand(*shape, device="cuda") * 100).to(x_dtype)
    y = (torch.rand(*shape, device="cuda") * 100 + 1).to(y_dtype)
    should_raise = (
        (x_dtype == torch.float16 and y_dtype == torch.bfloat16) or
        (x_dtype == torch.bfloat16 and y_dtype == torch.float16)
    )
    result_type = torch.promote_types(x_dtype, y_dtype)
    z = torch.zeros_like(x, device="cuda").to(result_type)
    ref = x % y
    kernel = array_kernel('mod',
                          f"tz = tx {mod_func} ty" if mod_func == "%" else
                          f"tz = {mod_func}(tx, ty)",
                          tmp_path)
    if should_raise:
        with pytest.raises(TileTypeError,
                           match=r"Implicit promotion of .* and .* is not supported"):
            launch_binary(kernel, x, y, z, tile)
    else:
        launch_binary(kernel, x, y, z, tile)
        assert_equal(z, ref)


@pytest.mark.parametrize("is_constant", [False, True])
@pytest.mark.parametrize("x", [100, -30])
@pytest.mark.parametrize("y", [23, -13])
def test_scalar_cdiv(shape, tile, x, y, is_constant, tmp_path):
    z = torch.zeros(shape, dtype=torch.int32, device='cuda')
    if not is_constant:
        kernel = scalar_kernel('cdiv', 'c = ct.cdiv(x, y)', tmp_path)
    else:
        kernel = const_scalar_kernel('cdiv', "int", "int", "c = ct.cdiv(x, y)", tmp_path)
    launch_binary(kernel, x, y, z, tile)
    assert_equal(z, ceil(x / y))


@pytest.mark.parametrize("op_symbol, ref_impl, force_float", [
    ("ct.cdiv", lambda x, y: torch.ceil(x / y), False),
    ("/", lambda x, y: x / y, True),
    ("ct.truediv", lambda x, y: x / y, True),
    ("//", lambda x, y: x // y, False),
    ("ct.floordiv", lambda x, y: x // y, False),
    ])
def test_array_scalar_div(shape, tile, int_dtype, tmp_path, op_symbol, ref_impl, force_float):
    x = torch.randint(0, 100, shape, dtype=int_dtype, device='cuda')
    y = 23
    result_type = torch.float32 if force_float else torch.promote_types(x.dtype, torch.int32)
    z = torch.zeros_like(x, dtype=result_type)
    # TODO: torch.ceil always return f32, should we align?
    ref = ref_impl(x, y).to(result_type)
    kernel = array_scalar_kernel('div',
                                 f'tz = {op_symbol}(tx, y)' if op_symbol.startswith("ct.") else
                                 f'tz = tx {op_symbol} y',
                                 tmp_path)
    launch_binary(kernel, x, y, z, tile)
    if force_float:
        assert_close(z, ref)
    else:
        assert_equal(z, ref)


@pytest.mark.parametrize("op_symbol, ref_impl", [
    ("/", lambda x, y: x / y),
    ("ct.truediv", lambda x, y: x / y),
    ])
def test_array_scalar_truediv_float(shape, tile, float_dtype, tmp_path, op_symbol, ref_impl):
    x = make_tensor(shape, dtype=float_dtype, device='cuda')
    y = 23.0
    res_dtype = torch.promote_types(x.dtype, torch.float32)
    ref = ref_impl(x.to(res_dtype), y)
    z = torch.zeros_like(ref)
    kernel = array_scalar_kernel('truediv',
                                 f'tz = {op_symbol}(tx, y)' if op_symbol.startswith("ct.") else
                                 f'tz = tx {op_symbol} y',
                                 tmp_path)
    launch_binary(kernel, x, y, z, tile)
    assert_close(z, ref)


@pytest.mark.parametrize("op_symbol, ref_impl, force_float", [
    ("ct.cdiv", lambda x, y: torch.ceil(x / y), False),
    ("/", lambda x, y: x / y, True),
    ("ct.truediv", lambda x, y: x / y, True),
    ("//", lambda x, y: x // y, False),
    ("ct.floordiv", lambda x, y: x // y, False),
    ])
@pytest.mark.parametrize("x_dtype", int_dtypes, ids=dtype_id)
@pytest.mark.parametrize("y_dtype", int_dtypes, ids=dtype_id)
def test_array_div(shape, tile, x_dtype, y_dtype, tmp_path, op_symbol, ref_impl, force_float):
    x = (torch.rand(*shape, device="cuda") * 100).to(dtype=x_dtype)
    y = (torch.rand(*shape, device="cuda") * 100 + 1).to(dtype=y_dtype)
    result_type = torch.promote_types(x.dtype, y.dtype) if not force_float else torch.float32
    z = torch.zeros_like(x).to(result_type)
    # TODO: torch.ceil always return f32, should we align?
    ref = ref_impl(x, y).to(result_type)
    kernel = array_kernel('div',
                          f"tz = {op_symbol}(tx, ty)" if op_symbol.startswith("ct.") else
                          f"tz = tx {op_symbol} ty",
                          tmp_path)
    launch_binary(kernel, x, y, z, tile)
    assert_equal(z, ref)


@pytest.mark.parametrize("op_symbol, ref_impl", [
    ("/", lambda x, y: x / y),
    ("ct.truediv", lambda x, y: x / y),
    ])
@pytest.mark.parametrize("x_dtype", float_dtypes, ids=dtype_id)
@pytest.mark.parametrize("y_dtype", float_dtypes, ids=dtype_id)
def test_array_truediv_float(shape, tile, x_dtype, y_dtype, tmp_path, op_symbol, ref_impl):
    should_raise = {x_dtype, y_dtype} == {torch.float16, torch.bfloat16}
    x = (torch.rand(*shape, device="cuda") * 100).to(dtype=x_dtype)
    y = (torch.rand(*shape, device="cuda") * 100 + 1).to(dtype=y_dtype)
    result_type = torch.promote_types(x.dtype, y.dtype)
    z = torch.zeros_like(x).to(result_type)
    ref = ref_impl(x, y).to(result_type)
    kernel = array_kernel('truediv',
                          f"tz = {op_symbol}(tx, ty)" if op_symbol.startswith("ct.") else
                          f"tz = tx {op_symbol} ty",
                          tmp_path)
    if should_raise:
        with pytest.raises(TileTypeError,
                           match=r"Implicit promotion of .* and .* is not supported"):
            launch_binary(kernel, x, y, z, tile)
    else:
        launch_binary(kernel, x, y, z, tile)
        assert_close(z, ref)


@pytest.mark.use_mlir
@pytest.mark.parametrize("rounding_mode",
                         [RMd.RN, RMd.RZ, RMd.RM, RMd.RP, RMd.FULL, RMd.APPROX, RMd.RZI])
def test_array_scalar_truediv_float_rounding_mode(
    shape, tile, float_dtype, tmp_path, rounding_mode
):
    should_raise_rounding_mode = rounding_mode in [RMd.RZI]
    x = make_tensor(shape, dtype=float_dtype, device='cuda')
    y = 23.0
    result_type = torch.promote_types(x.dtype, torch.float32)
    should_raise_dtype = rounding_mode in [RMd.APPROX, RMd.FULL] and result_type != torch.float32
    z = torch.zeros_like(x, dtype=result_type)
    kernel = array_scalar_kernel('truediv_rounding_mode',
                                 f'tz = ct.truediv(tx, y, rounding_mode={rounding_mode})',
                                 tmp_path,
                                 globals={"RoundingMode": RMd})
    if should_raise_rounding_mode:
        with pytest.raises(TileTypeError,
                           match=fr"Rounding mode {rounding_mode.value} is not supported"):
            launch_binary(kernel, x, y, z, tile)
    elif should_raise_dtype:
        with pytest.raises(TileTypeError,
                           match=fr"Rounding mode {rounding_mode.value} can only be used for "
                           "float32 type"):
            launch_binary(kernel, x, y, z, tile)
    else:
        bytecode = get_bytecode(kernel, (x, y, z, tile))
        if rounding_mode is RMd.RN:
            # Rmd.RN as the default rounding mode is not included in the mlir text
            check_directive = "// CHECK-NOT: rounding<{{[^>]*}}>"
        else:
            check_directive = (
                f"// CHECK: %[[RES:.*]] = divf %[[A:.*]], %[[B:.*]] "
                f"rounding<{rounding_mode.value}>"
            )
        filecheck(bytecode, check_directive)
        launch_binary(kernel, x, y, z, tile)


@pytest.mark.use_mlir
@pytest.mark.parametrize("dtype", float_dtypes, ids=dtype_id)
@pytest.mark.parametrize("flush_to_zero", [True, False])
def test_truediv_float_flush_to_zero(tile, dtype, flush_to_zero, tmp_path):
    should_raise_dtype = flush_to_zero and (dtype != torch.float32)
    x = make_tensor((1,), dtype=dtype, device='cuda')
    y = make_tensor((1,), dtype=dtype, device='cuda')
    z = torch.zeros_like(x, device="cuda")
    kernel = array_kernel("truediv_flush_to_zero",
                          f"tz = ct.truediv(tx, ty, flush_to_zero={flush_to_zero})",
                          tmp_path)
    if should_raise_dtype:
        with pytest.raises(TileTypeError,
                           match=r"Flush to zero can only be used for float32 type"):
            launch_binary(kernel, x, y, z, tile)
    else:
        bytecode = get_bytecode(kernel, (x, y, z, tile))
        if flush_to_zero:
            check_directive = "// CHECK: %[[RES:.*]] = divf %[[A:.*]], %[[B:.*]] flush_to_zero :"
        else:
            check_directive = (
                "// CHECK: %[[RES:.*]] = divf %[[A:.*]], %[[B:.*]]{{[[:space:]]*}}:"
            )
        filecheck(bytecode, check_directive)
        launch_binary(kernel, x, y, z, tile)


@pytest.mark.parametrize("power_func", ["**", "ct.pow"])
@pytest.mark.parametrize("x_dtype", float_dtypes, ids=dtype_id)
@pytest.mark.parametrize("y_dtype", float_dtypes, ids=dtype_id)
def test_array_pow(shape, tile, x_dtype, y_dtype, tmp_path, power_func):
    should_raise = {x_dtype, y_dtype} == {torch.float16, torch.bfloat16}
    x = torch.rand(shape, dtype=x_dtype, device='cuda')
    y = torch.rand(shape, dtype=y_dtype, device='cuda')
    z = torch.zeros_like(x).to(torch.promote_types(x_dtype, y_dtype))
    kernel = array_kernel('pow',
                          "tz = tx ** ty" if power_func == "**" else
                          f"tz = {power_func}(tx, ty)",
                          tmp_path)
    if should_raise:
        with pytest.raises(TileTypeError,
                           match=r"Implicit promotion of .* and .* is not supported"):
            launch_binary(kernel, x, y, z, tile)
    else:
        launch_binary(kernel, x, y, z, tile)
        torch.testing.assert_close(z, torch.pow(x, y))


@pytest.mark.parametrize("is_constant", [False, True])
def test_scalar_pow(shape, tile, is_constant, tmp_path):
    x = 5
    y = 2.0
    ref = torch.full(shape, x ** y, device="cuda")
    z = torch.zeros(shape, dtype=torch.float32, device='cuda')
    if not is_constant:
        kernel = scalar_kernel('pow', 'c = x ** y', tmp_path)
    else:
        kernel = const_scalar_kernel('pow', "int", "float", "c = x ** y", tmp_path)
    launch_binary(kernel, x, y, z, tile)
    torch.testing.assert_close(z, ref)


def test_array_scalar_pow(shape, tile, float_dtype, tmp_path):
    x = torch.rand(shape, dtype=float_dtype, device='cuda')
    y = 5.0
    res_dtype = torch.promote_types(x.dtype, torch.float32)
    z = torch.zeros_like(x, dtype=res_dtype)
    kernel = array_scalar_kernel('pow', 'tz = tx ** y', tmp_path)
    launch_binary(kernel, x, y, z, tile)
    torch.testing.assert_close(z, (x.to(res_dtype) ** y))


def bitwise_reference(op_symbol: str, x: torch.Tensor, y: torch.Tensor | int):
    res_dtype = torch.promote_types(x.dtype,
                                    y.dtype if isinstance(y, torch.Tensor) else torch.int32)
    # Workaround for the missing kernel in torch
    if res_dtype == torch.uint32:
        impl_dtype = torch.int32
    elif res_dtype == torch.uint64:
        impl_dtype = torch.int64
    else:
        impl_dtype = res_dtype

    x = x.to(impl_dtype)
    y = torch.tensor(y).to(impl_dtype)
    res = eval(f'x {op_symbol} y', None, dict(x=x, y=y))
    return res.to(res_dtype)


bitwise_logcal_cases = [
    pytest.param("&", "ct.bitwise_and", id="bit_and"),
    pytest.param("|", "ct.bitwise_or", id="bit_or"),
    pytest.param("^", "ct.bitwise_xor", id="bit_xor"),
]
bitwise_logical_dtypes = [torch.uint32, torch.uint64, torch.int32, torch.int64,
                          torch.float32, torch.float64, torch.bool, torch.int16,
                          torch.int8]


@pytest.mark.parametrize("x_dtype", bitwise_logical_dtypes, ids=dtype_id)
@pytest.mark.parametrize("y_dtype", bitwise_logical_dtypes, ids=dtype_id)
@pytest.mark.parametrize("op_symbol, op_func", bitwise_logcal_cases)
def test_array_bitwise_logical(shape, tile, x_dtype, y_dtype, op_symbol, op_func, tmp_path):
    x = make_tensor(shape, dtype=x_dtype, device='cuda')
    y = make_tensor(shape, dtype=y_dtype, device='cuda')
    z = torch.zeros_like(x)
    for expr in [f"tz = tx {op_symbol} ty", f"tz = {op_func}(tx, ty)"]:
        if x_dtype.is_floating_point or y_dtype.is_floating_point:
            with pytest.raises(TileTypeError, match=r'Bitwise operations require integers'):
                kernel = array_kernel('bitwise', expr, tmp_path)
                launch_binary(kernel, x, y, z, tile)
        elif to_dtype(x_dtype) != to_dtype(y_dtype):
            with pytest.raises(TileTypeError, match=r'Bitwise operands must have same data type'):
                kernel = array_kernel('bitwise', expr, tmp_path)
                launch_binary(kernel, x, y, z, tile)
        else:
            kernel = array_kernel('bitwise', expr, tmp_path)
            launch_binary(kernel, x, y, z, tile)
            ref = bitwise_reference(op_symbol, x, y)
            assert_equal(z, ref)


def bitwise_shift_reference(op_symbol: str, x: torch.Tensor, y: torch.Tensor | int):
    return eval(f'x {op_symbol} y')


bitwise_shift_cases = [
    pytest.param("<<", "ct.bitwise_lshift", id="bit_lshift"),
    pytest.param(">>", "ct.bitwise_rshift", id="bit_rshift"),
]
bitwise_shift_dtypes = [torch.int32, torch.int64, torch.int16, torch.int8,
                        torch.float32, torch.float64]


@pytest.mark.parametrize("x_dtype", bitwise_shift_dtypes, ids=dtype_id)
@pytest.mark.parametrize("y_dtype", bitwise_shift_dtypes, ids=dtype_id)
@pytest.mark.parametrize("op_symbol, op_func", bitwise_shift_cases)
def test_array_bitwise_shift(shape, tile, x_dtype, y_dtype, op_symbol, op_func, tmp_path):
    x = make_tensor(shape, dtype=x_dtype, device='cuda')
    y = make_tensor(shape, dtype=y_dtype, device='cuda', low=0, high=8)
    if x_dtype.is_floating_point or y_dtype.is_floating_point:
        res_type = torch.uint64  # Doesn't matter, we should error out anyway
    else:
        res_type = (x << y).dtype

    z = torch.zeros_like(x, dtype=res_type)
    for expr in [f"tz = tx {op_symbol} ty", f"tz = {op_func}(tx, ty)"]:
        if x_dtype.is_floating_point or y_dtype.is_floating_point:
            with pytest.raises(TileTypeError,
                               match=r'Bitwise shift requires an integer'):
                kernel = array_kernel('bitwise', expr, tmp_path)
                launch_binary(kernel, x, y, z, tile)
        else:
            kernel = array_kernel('bitwise', expr, tmp_path)
            launch_binary(kernel, x, y, z, tile)
            ref = bitwise_shift_reference(op_symbol, x, y)
            assert_equal(z, ref)


@pytest.mark.parametrize("is_constant", [False, True])
@pytest.mark.parametrize('op', ['&', '|', '^', '<<', '>>'],
                         ids=['bit_and', 'bit_or', 'bit_xor', 'bit_lshift', 'bit_rshift'])
def test_scalar_bitwise(shape, tile, is_constant, op, tmp_path):
    x = 5
    y = 2
    z = torch.zeros(shape, dtype=torch.int32, device='cuda')
    if not is_constant:
        kernel = scalar_kernel('bitwise', f'c = x {op} y', tmp_path)
    else:
        kernel = const_scalar_kernel('bitwise', "int", "int", f'c = x {op} y', tmp_path)
    launch_binary(kernel, x, y, z, tile)
    assert_equal(z, eval(f'x {op} y'))


@pytest.mark.parametrize('dtype', [torch.int32, torch.int64, torch.uint32], ids=dtype_id)
@pytest.mark.parametrize('op', ['&', '|', '^'],
                         ids=['bit_and', 'bit_or', 'bit_xor'])
def test_array_scalar_bitwise(shape, dtype, tile, op, tmp_path):
    x = make_tensor(shape, dtype=dtype, device='cuda')
    y = 5
    z = torch.zeros_like(x)
    kernel = array_scalar_kernel('bitwise', f'tz = tx {op} y', tmp_path)
    with raises_if(dtype != torch.int32, TileTypeError,
                   match="Bitwise operands must have same data type"):
        launch_binary(kernel, x, y, z, tile)
        ref = bitwise_reference(op, x, y)
        assert_equal(z, ref)


@pytest.mark.parametrize('dtype', [torch.int32, torch.int64, torch.uint32], ids=dtype_id)
@pytest.mark.parametrize('op', ['<<', '>>'],
                         ids=['bit_lshift', 'bit_rshift'])
def test_array_scalar_shift(shape, dtype, tile, op, tmp_path):
    x = make_tensor(shape, dtype=dtype, device='cuda')
    y = 5
    z = torch.zeros_like(x)
    kernel = array_scalar_kernel('bitwise', f'tz = tx {op} y', tmp_path)
    with raises_if(not dtype.is_signed, TileTypeError,
                   match="Implicit promotion of .* and int32 is not supported"):
        launch_binary(kernel, x, y, z, tile)
        ref = bitwise_reference(op, x, y)
        assert_equal(z, ref)


def test_array_implicit_cast_happy(tmp_path):
    x = make_tensor((2,), dtype=torch.int32, device='cuda')
    y = make_tensor((2,), dtype=torch.float32, device='cuda')
    z = torch.zeros_like(y)
    kernel = array_kernel('inplace_bin', 'ty *= tx; tz = ty', tmp_path)
    launch_binary(kernel, x, y, z, 1)


def test_array_implicit_cast_unhappy(tmp_path):
    x = make_tensor((2,), dtype=torch.float32, device='cuda')
    y = make_tensor((2,), dtype=torch.int32, device='cuda')
    z = torch.zeros_like(y)
    kernel = array_kernel('inplace_bin', 'ty *= tx; tz = ty', tmp_path)
    with pytest.raises(TileTypeError):
        launch_binary(kernel, x, y, z, 1)


def test_array_scalar_implicit_cast_happy(tmp_path):
    x = make_tensor((2,), dtype=torch.int32, device='cuda')
    z = torch.zeros_like(x)
    kernel = array_scalar_kernel('inplace_bin', 'tx *= 3; tz = tx', tmp_path)
    launch_binary(kernel, x, x, z, 1)


def test_array_scalar_implicit_cast_unhappy(tmp_path):
    x = make_tensor((2,), dtype=torch.int32, device='cuda')
    z = torch.zeros_like(x)
    kernel = array_scalar_kernel('inplace_bin', 'tx *= 3.0; tz = tx', tmp_path)
    with pytest.raises(TileTypeError):
        launch_binary(kernel, x, x, z, 1)
