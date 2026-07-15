# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
import dataclasses
import math
import ctypes
import re
import sys
import traceback
from dataclasses import dataclass
from enum import Enum
from typing import Any

import torch
import numpy as np
import pytest

from util import FdCaptureRunner
import cuda.tile as ct
from cuda.tile._bytecode.version import BytecodeVersion
from cuda.tile._compiler_options import CompilerOptions
from conftest import get_tileiras_version
from cuda.tile._datatype import (bool_,
                                 float16, float32, float64, bfloat16,
                                 int64, uint8, uint16, uint32, uint64,
                                 is_float, numeric_dtype_category,
                                 NumericDTypeCategory, _DTypePromotionImpl)

# opt_level=0 required for correct print ordering in tileiras < 13.2
_DEFAULT_OPT_LEVEL = CompilerOptions.__dataclass_fields__['opt_level'].default
_OPT_LEVEL = 0 if get_tileiras_version() < BytecodeVersion.V_13_2 else _DEFAULT_OPT_LEVEL
_libc = ctypes.CDLL('ucrtbase' if sys.platform == 'win32' else None)

# === Helpers ===
_kernel_runner = None


def _get_kernel_runner():
    global _kernel_runner
    if _kernel_runner is None or not _kernel_runner.is_alive():
        _kernel_runner = FdCaptureRunner(__file__, "start_kernel_runner")
    return _kernel_runner


def _close_kernel_runner():
    global _kernel_runner
    if _kernel_runner is not None and _kernel_runner.is_alive():
        _kernel_runner.close()
    _kernel_runner = None


@pytest.fixture(scope="module", autouse=True)
def _module_teardown():
    yield
    _close_kernel_runner()


def _kernel_runner_main():
    while True:
        args = FdCaptureRunner.get_cmd_args(sys.stdin.buffer)
        if args is None:
            break
        try:
            FdCaptureRunner.write_begin_marker()
            kernel_name, shape_str, dtype_str, tile_str = args
            kernel = globals()[kernel_name]
            shape = tuple(int(d) for d in shape_str.split(","))
            dtype = getattr(torch, dtype_str)
            tile = int(tile_str)
            x = torch.arange(math.prod(shape), device='cuda').reshape(shape).to(dtype)
            grid = (math.ceil(shape[0] / tile), 1, 1)
            ct.launch(torch.cuda.current_stream(), grid, kernel, (x, tile))
            torch.cuda.synchronize()
        except Exception:
            sys.stderr.write(traceback.format_exc())
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            _libc.fflush(None)
            FdCaptureRunner.write_end_marker()


def _run_kernel(kernel, shape, dtype_name, tile):
    stdout_lines, stderr_lines = _get_kernel_runner().run_cmd(
        kernel._pyfunc.__name__,
        ",".join(str(d) for d in shape),
        dtype_name,
        str(tile))
    if stderr_lines:
        raise RuntimeError("Kernel raised an exception\n" + "\n".join(stderr_lines))
    return stdout_lines
# === End of helpers ===


@ct.kernel(opt_level=_OPT_LEVEL)
def kernel_printf(x, TILE: ct.Constant[int]):
    bid = ct.bid(0)
    tx = ct.load(x, index=(bid,), shape=(TILE,))
    if x.dtype == float16 or x.dtype == float32 or x.dtype == bfloat16:
        ct.printf("tile[%d]:%.5f\n", bid, tx)
    elif x.dtype == float64:
        ct.printf("tile[%d]:%.5lf\n", bid, tx)
    elif x.dtype == int64:
        ct.printf("tile[%d]:%lld\n", bid, tx)
    elif x.dtype == uint8 or x.dtype == uint16 or x.dtype == uint32:
        ct.printf("tile[%d]:%u\n", bid, tx)
    elif x.dtype == uint64:
        ct.printf("tile[%d]:%llu\n", bid, tx)
    else:
        ct.printf("tile[%d]:%d\n", bid, tx)


@ct.kernel(opt_level=_OPT_LEVEL)
def kernel_print(x, TILE: ct.Constant[int]):
    bid = ct.bid(0)
    tx = ct.load(x, index=(bid,), shape=(TILE,))
    if x.dtype == float16 or x.dtype == float32 or x.dtype == float64 or x.dtype == bfloat16:
        ct.print(f"tile[{bid}]:{tx:.5f}")
    else:
        ct.print(f"tile[{bid}]:{tx}")


@ct.kernel(opt_level=_OPT_LEVEL)
def kernel_print_sep(x, TILE: ct.Constant[int]):
    bid = ct.bid(0)
    tx = ct.load(x, index=(bid,), shape=(TILE,))
    ct.print("tile:", tx, sep='')


@ct.kernel(opt_level=_OPT_LEVEL)
def kernel_print_two_vars_with_expr(x, TILE: ct.Constant[int]):
    bid = ct.bid(0)
    tx = ct.load(x, index=(bid,), shape=(TILE,))
    ct.print(f"tile[{bid}]: a={tx:.6f} b={tx + tx:.6f}")


@ct.kernel(opt_level=_OPT_LEVEL)
def kernel_print_no_end(x, TILE: ct.Constant[int]):
    bid = ct.bid(0)
    tx = ct.load(x, index=(bid,), shape=(TILE,))
    ct.print(tx, end='')


@ct.kernel(opt_level=_OPT_LEVEL)
def kernel_builtin_print(x, TILE: ct.Constant[int]):
    bid = ct.bid(0)
    tx = ct.load(x, index=(bid,), shape=(TILE,))
    if x.dtype == float16 or x.dtype == float32 or x.dtype == float64 or x.dtype == bfloat16:
        print(f"tile[{bid}]:{tx:.5f}")
    else:
        print(f"tile[{bid}]:{tx}")


@ct.kernel(opt_level=_OPT_LEVEL)
def kernel_fstring_nested(x, TILE: ct.Constant[int]):
    bid = ct.bid(0)
    tx = ct.load(x, index=(bid,), shape=(TILE,))
    msg = f"tile[{bid}]:{tx:10.4f}"
    ct.print(msg)               # f-string stored in variable
    ct.print(f"msg={msg}")      # nested f-string


@ct.kernel(opt_level=_OPT_LEVEL)
def kernel_print_aliases(x, TILE: ct.Constant[int]):
    bid = ct.bid(0)
    tx = ct.load(x, index=(bid,), shape=(TILE,))
    p_ct = ct.print
    p_ct(f"ct%:{tx}")
    p_builtin = print
    p_builtin(f"builtin%:{tx}")
    p_printf = ct.printf
    p_printf("printf%%[%d]:%d\n", bid, tx)


@ct.kernel(opt_level=_OPT_LEVEL)
def kernel_print_tuple(x, TILE: ct.Constant[int]):
    ct.print(x.shape)
    print(x.shape)
    ct.print(f"shape = {x.shape}")
    ct.print("shape:", x.shape)


@ct.kernel(opt_level=_OPT_LEVEL)
def kernel_print_nested_tuple(x, TILE: ct.Constant[int]):
    ct.print(f"nested tuple: {((x.shape, x.shape), ct.bid(0))}")
    ct.print(((x.shape, x.shape), ct.bid(0)))


@ct.kernel(opt_level=_OPT_LEVEL)
def kernel_print_tuple_escaped_str(x, TILE: ct.Constant[int]):
    bid = ct.bid(0)
    ct.print(f"tuple w/ escaped str: {((x.shape, '%d'), '%%d')}")
    ct.print(("foo", "'''foo'''", "\"foo'"))
    ct.print((f"foo{bid}", f"'''foo{bid}'''", f"\"foo{bid}\'"))
    single = "'"
    double = '"'
    both = '\'"'
    ct.print((f'{single}', f"{double}", f"{both}"))


@ct.kernel(opt_level=_OPT_LEVEL)
def kernel_print_tuple_fstring(x, TILE: ct.Constant[int]):
    bid = ct.bid(0)
    msg = f"bid={bid}"
    ct.print((msg, bid))


@dataclass(frozen=True)
class Foo:
    x: Any
    y: Any = dataclasses.field(repr=False)
    z: Any


@ct.kernel(opt_level=_OPT_LEVEL)
def kernel_print_dataclass(x, TILE: ct.Constant[int]):
    t = ct.load(x, 0, (TILE,))
    foo = Foo(t, t + 1, "hello")
    ct.print(foo)
    print(foo)


@dataclass(frozen=True)
class DataclassCustomStr:
    x: Any
    y: Any

    def __str__(self):
        return f"CUSTOM{{{self.x}|{self.y}}}"


@ct.kernel(opt_level=_OPT_LEVEL)
def kernel_print_dataclass_custom_str(x, TILE: ct.Constant[int]):
    t = ct.load(x, 0, (TILE,))
    foo = DataclassCustomStr(t, t + 1)
    ct.print(foo)
    print(foo)


@ct.kernel(opt_level=_OPT_LEVEL)
def kernel_print_dataclass_repr_ignores_custom_str(x, TILE: ct.Constant[int]):
    t = ct.load(x, 0, (TILE,))
    foo = DataclassCustomStr(t, t + 1)
    ct.print((foo,))
    print((foo,))


@dataclass(frozen=True)
class DataclassCustomRepr:
    x: Any
    y: Any

    def __repr__(self):
        return f"CUSTOMREPR({self.x}|{self.y})"


@ct.kernel(opt_level=_OPT_LEVEL)
def kernel_print_dataclass_custom_repr(x, TILE: ct.Constant[int]):
    t = ct.load(x, 0, (TILE,))
    foo = DataclassCustomRepr(t, t + 1)
    ct.print(foo)
    print(foo)


@dataclass(frozen=True)
class DataclassCustomStrAndRepr:
    x: Any

    def __str__(self):
        return f"CUSTOM_STR({self.x})"

    def __repr__(self):
        return f"CUSTOM_REPR({self.x})"


@ct.kernel(opt_level=_OPT_LEVEL)
def kernel_print_dataclass_custom_str_and_repr_pick_str(x, TILE: ct.Constant[int]):
    t = ct.load(x, 0, (TILE,))
    foo = DataclassCustomStrAndRepr(t)
    ct.print(foo)
    print(foo)


@ct.kernel(opt_level=_OPT_LEVEL)
def kernel_print_dataclass_custom_str_and_repr_pick_repr(x, TILE: ct.Constant[int]):
    t = ct.load(x, 0, (TILE,))
    foo = DataclassCustomStrAndRepr(t)
    ct.print((foo,))
    print((foo,))


@dataclass(frozen=True, repr=False)
class DataclassDisabledRepr:
    x: Any


@ct.kernel(opt_level=_OPT_LEVEL)
def kernel_print_dataclass_disabled_repr(x, TILE: ct.Constant[int]):
    t = ct.load(x, 0, (TILE,))
    foo = DataclassDisabledRepr(t)
    ct.print(foo)
    print(foo)


@dataclass(frozen=True, repr=False)
class DataclassDisabledReprCustomStr:
    x: Any

    def __str__(self):
        return f"NO_REPR_BUT_CUSTOM_STR({self.x})"


@ct.kernel(opt_level=_OPT_LEVEL)
def kernel_print_dataclass_disabled_repr_but_custom_str(x, TILE: ct.Constant[int]):
    t = ct.load(x, 0, (TILE,))
    foo = DataclassDisabledReprCustomStr(t)
    ct.print(foo)
    print(foo)


@ct.kernel(opt_level=_OPT_LEVEL)
def kernel_print_dataclass_disabled_repr_ignore_custom_str(x, TILE: ct.Constant[int]):
    t = ct.load(x, 0, (TILE,))
    foo = DataclassDisabledReprCustomStr(t)
    ct.print((foo,))
    print((foo,))


@dataclass(frozen=True, repr=False)
class DataclassNoDefaultButCustomRepr:
    x: Any

    def __repr__(self):
        return f"NO_DEFAULT_BUT_CUSTOM_REPR({self.x})"


@ct.kernel(opt_level=_OPT_LEVEL)
def kernel_print_dataclass_no_default_but_custom_repr(x, TILE: ct.Constant[int]):
    t = ct.load(x, 0, (TILE,))
    foo = DataclassNoDefaultButCustomRepr(t)
    ct.print(foo)
    print(foo)


@ct.kernel(opt_level=_OPT_LEVEL)
def kernel_print_empty_tuple(x, TILE: ct.Constant[int]):
    ct.print(f"empty tuple: {()}")
    ct.print(())


@ct.kernel(opt_level=_OPT_LEVEL)
def kernel_print_single_tuple(x, TILE: ct.Constant[int]):
    bid = ct.bid(0)
    ct.print(f"single tuple: {(bid,)}")
    ct.print((bid,))


@ct.kernel(opt_level=_OPT_LEVEL)
def kernel_print_tuple_w_tile(x, TILE: ct.Constant[int]):
    bid = ct.bid(0)
    tx = ct.load(x, index=(bid,), shape=(TILE,))
    ct.print(f"tuple w/ tile: {(tx,)}")
    ct.print((tx,))


@ct.kernel(opt_level=_OPT_LEVEL)
def kernel_print_tuple_tile_shape(x, TILE: ct.Constant[int]):
    bid = ct.bid(0)
    tx = ct.load(x, index=(bid,), shape=(TILE,))
    ct.print(f"Tile shape: {tx.shape}")
    ct.print(tx.shape)


@ct.kernel(opt_level=_OPT_LEVEL)
def kernel_print_dtype(x, TILE: ct.Constant[int]):
    bid = ct.bid(0)
    tx = ct.load(x, index=(bid,), shape=(TILE,))
    ct.print(x.dtype)
    print(x.dtype)
    ct.print(f"dtype = {tx.dtype}")
    print(f"dtype = {tx.dtype}")


class _PrintColor(Enum):
    RED = 0
    GREEN = 1


@ct.kernel(opt_level=_OPT_LEVEL)
def kernel_print_enum(x, TILE: ct.Constant[int]):
    ct.print(_PrintColor.GREEN)
    print(_PrintColor.GREEN)
    ct.print(f"color = {_PrintColor.RED}")
    print(f"color = {_PrintColor.RED}")


@ct.kernel(opt_level=_OPT_LEVEL)
def kernel_print_ordering(x, TILE: ct.Constant[int]):
    bid = ct.bid(0)
    tx = ct.load(x, index=(bid,), shape=(TILE,)).reshape((2, 4))

    print("original:", tx)
    print("reduce over all axes:", ct.sum(tx, None))
    print("reduce over axis 1:", ct.sum(tx, 1))
    print("reduce over axis 0 and keep dims:", ct.sum(tx, 1, keepdims=True))

    print(f"original: {tx}")
    print(f"reduce over all axes: {ct.sum(tx, None)}")
    print(f"reduce over axis 1: {ct.sum(tx, 1)}")
    print(f"reduce over axis 0 and keep dims: {ct.sum(tx, 1, keepdims=True)}")

    ct.print("original:", tx)
    ct.print("reduce over all axes:", ct.sum(tx, None))
    ct.print("reduce over axis 1:", ct.sum(tx, 1))
    ct.print("reduce over axis 0 and keep dims:", ct.sum(tx, 1, keepdims=True))

    ct.print(f"original: {tx}")
    ct.print(f"reduce over all axes: {ct.sum(tx, None)}")
    ct.print(f"reduce over axis 1: {ct.sum(tx, 1)}")
    ct.print(f"reduce over axis 0 and keep dims: {ct.sum(tx, 1, keepdims=True)}")

    ct.printf("original: %i\n", tx)
    ct.printf("reduce over all axes: %i\n", ct.sum(tx, None))
    ct.printf("reduce over axis 1: %i\n", ct.sum(tx, 1))
    ct.printf("reduce over axis 0 and keep dims: %i\n", ct.sum(tx, 1, keepdims=True))


def _test_print_1d(shape, tile, kernel, dtype):
    dtype_name = 'bool' if dtype.__name__ == 'bool_' else dtype.__name__
    actual_outs = _run_kernel(kernel, shape, dtype_name, tile)

    # Numpy does not support bfloat16
    np_dtype = dtype.__name__ if dtype.__name__ != 'bfloat16' else 'float16'
    x = np.arange(np.prod(shape)).reshape(shape).astype(np_dtype)
    num_tiles = math.ceil(shape[0] / tile)
    for i in range(num_tiles):
        start_idx, end_idx = tile * i, tile * (i + 1)
        if is_float(dtype):
            formatted_x = ', '.join([f"{elem:.5f}" for elem in x[start_idx:end_idx]])
        else:
            formatted_x = ', '.join([f"{int(elem)}" for elem in x[start_idx:end_idx]])
        expected_out = f"tile[{i}]:[{formatted_x}]"
        assert expected_out in actual_outs


all_int_and_float_dtypes = [
    x for x in _DTypePromotionImpl._order
    if numeric_dtype_category(x) in (NumericDTypeCategory.Integral, NumericDTypeCategory.Float)
]


@pytest.mark.parametrize("shape", [(8,), (16,)])
@pytest.mark.parametrize("tile", [8])
@pytest.mark.parametrize("dtype", all_int_and_float_dtypes)
def test_printf(shape, tile, dtype):
    _test_print_1d(shape, tile, kernel_printf, dtype)


@pytest.mark.parametrize("shape", [(8,), (16,)])
@pytest.mark.parametrize("tile", [8])
@pytest.mark.parametrize("dtype", all_int_and_float_dtypes)
@pytest.mark.parametrize("kernel", [kernel_print, kernel_builtin_print],
                         ids=["ct_print", "builtin_print"])
def test_ct_print_and_builtin_print(shape, tile, kernel, dtype):
    _test_print_1d(shape, tile, kernel, dtype)


@pytest.mark.parametrize("shape", [(8,), (16,)])
@pytest.mark.parametrize("tile", [8])
@pytest.mark.parametrize("kernel", [kernel_printf, kernel_print, kernel_builtin_print],
                         ids=["ct_printf", "ct_print", "builtin_print"])
def test_bool(shape, tile, kernel):
    _test_print_1d(shape, tile, kernel, bool_)


@pytest.mark.parametrize("shape", [(8,),])
@pytest.mark.parametrize("tile", [8])
def test_ct_print_sep(shape, tile):
    actual_outs = _run_kernel(kernel_print_sep, shape, "int32", tile)
    x = np.arange(np.prod(shape)).reshape(shape).astype(np.int32)
    formatted_x = ', '.join([f"{elem}" for elem in x[:tile]])
    expected = f"tile:[{formatted_x}]"
    assert expected in actual_outs


@pytest.mark.parametrize("shape", [(8,), (16,)])
@pytest.mark.parametrize("tile", [8])
def test_ct_print_two_vars(shape, tile):
    actual_outs = _run_kernel(kernel_print_two_vars_with_expr, shape, "float32", tile)
    x = np.arange(np.prod(shape)).reshape(shape).astype(np.float32)
    num_tiles = math.ceil(shape[0] / tile)
    for i in range(num_tiles):
        start_idx, end_idx = tile * i, tile * (i + 1)
        formatted_a = ', '.join([f"{elem:.6f}" for elem in x[start_idx:end_idx]])
        formatted_b = ', '.join([f"{elem * 2:.6f}" for elem in x[start_idx:end_idx]])
        expected = f"tile[{i}]: a=[{formatted_a}] b=[{formatted_b}]"
        assert expected in actual_outs


@pytest.mark.parametrize("shape", [(8,),])
@pytest.mark.parametrize("tile", [8])
def test_ct_print_no_end(shape, tile):
    actual_outs = _run_kernel(kernel_print_no_end, shape, "int32", tile)
    x = np.arange(np.prod(shape)).reshape(shape).astype(np.int32)
    formatted_x = ', '.join([f"{elem}" for elem in x[:tile]])
    assert f"[{formatted_x}]" in actual_outs


def test_ct_print_error_conversion():
    from cuda.tile._exception import TileSyntaxError

    @ct.kernel(opt_level=_OPT_LEVEL)
    def bad_kernel(x, TILE: ct.Constant[int]):
        tx = ct.load(x, index=(0,), shape=(TILE,))
        ct.print(f"{tx!r}")

    x = torch.zeros(8, device='cuda', dtype=torch.int32)
    with pytest.raises(TileSyntaxError, match="!r, !s, !a"):
        ct.launch(torch.cuda.current_stream(), (1, 1, 1), bad_kernel, (x, 8))


def test_ct_print_error_dynamic_format_spec():
    from cuda.tile._exception import TileSyntaxError

    @ct.kernel(opt_level=_OPT_LEVEL)
    def bad_kernel(x, TILE: ct.Constant[int]):
        width = 5
        tx = ct.load(x, index=(0,), shape=(TILE,))
        ct.print(f"{tx:{width}}")

    x = torch.zeros(8, device='cuda', dtype=torch.int32)
    with pytest.raises(TileSyntaxError, match="format spec must be a literal string"):
        ct.launch(torch.cuda.current_stream(), (1, 1, 1), bad_kernel, (x, 8))


@pytest.mark.parametrize("shape", [(8,)])
@pytest.mark.parametrize("tile", [8])
def test_fstring_nested(shape, tile):
    actual_outs = _run_kernel(kernel_fstring_nested, shape, "float32", tile)
    x = np.arange(np.prod(shape)).reshape(shape).astype(np.float32)
    num_tiles = math.ceil(shape[0] / tile)
    for i in range(num_tiles):
        start_idx, end_idx = tile * i, tile * (i + 1)
        formatted_x = ', '.join(
            [f"{elem:10.4f}" for elem in x[start_idx:end_idx]])
        assert f"tile[{i}]:[{formatted_x}]" in actual_outs
        assert f"msg=tile[{i}]:[{formatted_x}]" in actual_outs


@pytest.mark.parametrize("shape", [(8,)])
@pytest.mark.parametrize("tile", [8])
def test_print_aliases(shape, tile):
    actual_outs = _run_kernel(kernel_print_aliases, shape, "int32", tile)
    x = np.arange(np.prod(shape)).reshape(shape).astype(np.int32)
    num_tiles = math.ceil(shape[0] / tile)
    for i in range(num_tiles):
        start_idx, end_idx = tile * i, tile * (i + 1)
        formatted_x = ', '.join([f"{elem}" for elem in x[start_idx:end_idx]])
        assert f"ct%:[{formatted_x}]" in actual_outs
        assert f"builtin%:[{formatted_x}]" in actual_outs
        assert f"printf%[{i}]:[{formatted_x}]" in actual_outs


def test_ct_print_tuple():
    actual_outs = _run_kernel(kernel_print_tuple, (2, 4), "int32", 2)
    assert actual_outs[0] == "(2, 4)"          # ct.print(x.shape)
    assert actual_outs[1] == "(2, 4)"          # print(x.shape)
    assert actual_outs[2] == "shape = (2, 4)"  # ct.print(f"shape = {x.shape}")
    assert actual_outs[3] == "shape: (2, 4)"   # ct.print("shape:", x.shape)


def test_ct_print_nested_tuple():
    actual_outs = _run_kernel(kernel_print_nested_tuple, (2, 4), "int32", 2)
    assert actual_outs[0] == "nested tuple: (((2, 4), (2, 4)), 0)"
    assert actual_outs[1] == "(((2, 4), (2, 4)), 0)"


def test_ct_print_tuple_with_escaped_str():
    actual_outs = _run_kernel(kernel_print_tuple_escaped_str, (2, 4), "int32", 2)
    assert actual_outs[0] == "tuple w/ escaped str: (((2, 4), '%d'), '%%d')"
    assert actual_outs[1] == "('foo', \"'''foo'''\", '\"foo\\\'')"
    assert actual_outs[2] == "('foo0', \"'''foo0'''\", '\"foo0\\\'')"
    assert actual_outs[3] == "(\"'\", '\"', '\\'\"')"


def test_ct_print_empty_tuple():
    actual_outs = _run_kernel(kernel_print_empty_tuple, (8,), "int32", 8)
    assert actual_outs[0] == "empty tuple: ()"
    assert actual_outs[1] == "()"


def test_ct_print_single_tuple():
    actual_outs = _run_kernel(kernel_print_single_tuple, (8,), "int32", 8)
    assert actual_outs[0] == "single tuple: (0,)"
    assert actual_outs[1] == "(0,)"


def test_ct_print_tuple_w_tile():
    shape = (8,)
    tile = 8
    actual_outs = _run_kernel(kernel_print_tuple_w_tile, shape, "int32", tile)
    x = np.arange(np.prod(shape)).reshape(shape).astype(np.int32)
    formatted_x = ', '.join([f"{elem}" for elem in x[:tile]])
    assert actual_outs[0] == f"tuple w/ tile: ([{formatted_x}],)"
    assert actual_outs[1] == f"([{formatted_x}],)"


def test_ct_print_tuple_tile_shape():
    actual_outs = _run_kernel(kernel_print_tuple_tile_shape, (8,), "int32", 8)
    assert actual_outs[0] == "Tile shape: (8,)"
    assert actual_outs[1] == "(8,)"


def test_ct_print_tuple_fstring():
    actual_outs = _run_kernel(kernel_print_tuple_fstring, (8,), "int32", 8)
    assert actual_outs[0] == "('bid=0', 0)"


def test_ct_print_tuple_format_spec_error():
    from cuda.tile._exception import TileTypeError

    @ct.kernel(opt_level=_OPT_LEVEL)
    def bad_kernel(x, TILE: ct.Constant[int]):
        ct.print(f"{x.shape:10}")

    x = torch.zeros(8, device='cuda', dtype=torch.int32)
    with pytest.raises(TileTypeError, match="cannot apply format spec to a tuple value"):
        ct.launch(torch.cuda.current_stream(), (1, 1, 1), bad_kernel, (x, 8))


def test_ct_print_dataclass():
    [actual_ct, actual_builtin] = _run_kernel(kernel_print_dataclass, (8,), "int32", 8)
    assert actual_ct == actual_builtin == "Foo(x=[0, 1, 2, 3, 4, 5, 6, 7], z='hello')"


def test_ct_print_dataclass_custom_str():
    [actual_ct, actual_builtin] = _run_kernel(kernel_print_dataclass_custom_str, (8,), "int32", 8)
    assert (actual_ct == actual_builtin
            == "CUSTOM{[0, 1, 2, 3, 4, 5, 6, 7]|[1, 2, 3, 4, 5, 6, 7, 8]}")


def test_ct_print_dataclass_repr_ignores_custom_str():
    [actual_ct, actual_builtin] = _run_kernel(kernel_print_dataclass_repr_ignores_custom_str,
                                              (8,), "int32", 8)
    assert actual_ct == actual_builtin == ("(DataclassCustomStr(x=[0, 1, 2, 3, 4, 5, 6, 7],"
                                           " y=[1, 2, 3, 4, 5, 6, 7, 8]),)")


def test_ct_print_dataclass_custom_repr():
    [actual_ct, actual_builtin] = _run_kernel(kernel_print_dataclass_custom_repr, (8,), "int32", 8)
    assert (actual_ct == actual_builtin
            == "CUSTOMREPR([0, 1, 2, 3, 4, 5, 6, 7]|[1, 2, 3, 4, 5, 6, 7, 8])")


def test_ct_print_dataclass_custom_str_and_repr_pick_str():
    [actual_ct, actual_builtin] = _run_kernel(kernel_print_dataclass_custom_str_and_repr_pick_str,
                                              (8,), "int32", 8)
    assert actual_ct == actual_builtin == "CUSTOM_STR([0, 1, 2, 3, 4, 5, 6, 7])"


def test_ct_print_dataclass_custom_str_and_repr_pick_repr():
    [actual_ct, actual_builtin] = _run_kernel(kernel_print_dataclass_custom_str_and_repr_pick_repr,
                                              (8,), "int32", 8)
    assert actual_ct == actual_builtin == "(CUSTOM_REPR([0, 1, 2, 3, 4, 5, 6, 7]),)"


def test_ct_print_dataclass_disabled_repr():
    [actual_ct, actual_builtin] = _run_kernel(kernel_print_dataclass_disabled_repr,
                                              (8,), "int32", 8)
    assert actual_ct == actual_builtin == "<DataclassDisabledRepr object>"


def test_ct_print_dataclass_disabled_repr_but_custom_str():
    [actual_ct, actual_builtin] = _run_kernel(kernel_print_dataclass_disabled_repr_but_custom_str,
                                              (8,), "int32", 8)
    assert actual_ct == actual_builtin == "NO_REPR_BUT_CUSTOM_STR([0, 1, 2, 3, 4, 5, 6, 7])"


def test_ct_print_dataclass_disabled_repr_ignore_custom_str():
    [actual_ct, actual_builtin] = _run_kernel(
            kernel_print_dataclass_disabled_repr_ignore_custom_str, (8,), "int32", 8)
    assert actual_ct == actual_builtin == "(<DataclassDisabledReprCustomStr object>,)"


def test_ct_print_dataclass_no_default_but_custom_repr():
    [actual_ct, actual_builtin] = _run_kernel(
        kernel_print_dataclass_no_default_but_custom_repr, (8,), "int32", 8)
    assert actual_ct == actual_builtin == "NO_DEFAULT_BUT_CUSTOM_REPR([0, 1, 2, 3, 4, 5, 6, 7])"


def test_ct_print_reject_dataclass_with_custom_format():
    @dataclass(frozen=True)
    class CustomFormat:
        x: int

        def __format__(self, spec):
            return "Custom!"

    @ct.kernel
    def kern():
        print(CustomFormat(123))

    with pytest.raises(ct.TileTypeError,
                       match=re.escape("Printing dataclasses"
                                       " with custom __format__() is not supported")):
        ct.launch(torch.cuda.current_stream(), (1,), kern, ())


def test_ct_print_dtype():
    actual_outs = _run_kernel(kernel_print_dtype, (8,), "float32", 8)
    assert actual_outs[0] == "float32"          # ct.print(tx.dtype)
    assert actual_outs[1] == "float32"          # print(tx.dtype)
    assert actual_outs[2] == "dtype = float32"  # ct.print(f"dtype = {tx.dtype}")
    assert actual_outs[3] == "dtype = float32"  # print(f"dtype = {tx.dtype}")


def test_ct_print_enum():
    actual_outs = _run_kernel(kernel_print_enum, (8,), "float32", 8)
    assert actual_outs[0] == "_PrintColor.GREEN"          # ct.print(_PrintColor.GREEN)
    assert actual_outs[1] == "_PrintColor.GREEN"          # print(_PrintColor.GREEN)
    assert actual_outs[2] == "color = _PrintColor.RED"   # ct.print(f"color = {_PrintColor.RED}")
    assert actual_outs[3] == "color = _PrintColor.RED"   # print(f"color = {_PrintColor.RED}")


def test_ct_print_ordering():
    actual_outs = _run_kernel(kernel_print_ordering, (8,), "int32", 8)

    assert actual_outs[0] == "original: [[0, 1, 2, 3], [4, 5, 6, 7]]"
    assert actual_outs[1] == "reduce over all axes: 28"
    assert actual_outs[2] == "reduce over axis 1: [6, 22]"
    assert actual_outs[3] == "reduce over axis 0 and keep dims: [[6], [22]]"

    assert actual_outs[4] == "original: [[0, 1, 2, 3], [4, 5, 6, 7]]"
    assert actual_outs[5] == "reduce over all axes: 28"
    assert actual_outs[6] == "reduce over axis 1: [6, 22]"
    assert actual_outs[7] == "reduce over axis 0 and keep dims: [[6], [22]]"

    assert actual_outs[8] == "original: [[0, 1, 2, 3], [4, 5, 6, 7]]"
    assert actual_outs[9] == "reduce over all axes: 28"
    assert actual_outs[10] == "reduce over axis 1: [6, 22]"
    assert actual_outs[11] == "reduce over axis 0 and keep dims: [[6], [22]]"

    assert actual_outs[12] == "original: [[0, 1, 2, 3], [4, 5, 6, 7]]"
    assert actual_outs[13] == "reduce over all axes: 28"
    assert actual_outs[14] == "reduce over axis 1: [6, 22]"
    assert actual_outs[15] == "reduce over axis 0 and keep dims: [[6], [22]]"

    assert actual_outs[16] == "original: [[0, 1, 2, 3], [4, 5, 6, 7]]"
    assert actual_outs[17] == "reduce over all axes: 28"
    assert actual_outs[18] == "reduce over axis 1: [6, 22]"
    assert actual_outs[19] == "reduce over axis 0 and keep dims: [[6], [22]]"


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "start_kernel_runner":
        _kernel_runner_main()
