# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import ctypes
import math
import sys
import os
import tempfile
import torch
import numpy as np
import pytest

import cuda.tile as ct
from cuda.tile._bytecode.version import BytecodeVersion
from cuda.tile._compiler_options import CompilerOptions
from conftest import get_tileiras_version
from cuda.tile._datatype import (bool_,
                                 float16, float32, float64,
                                 int64, uint8, uint16, uint32, uint64,
                                 NumericDTypeCategories, is_float)

# opt_level=0 required for correct print ordering in tileiras < 13.2
_DEFAULT_OPT_LEVEL = CompilerOptions.__dataclass_fields__['opt_level'].default
_OPT_LEVEL = 0 if get_tileiras_version() < BytecodeVersion.V_13_2 else _DEFAULT_OPT_LEVEL
_libc = ctypes.CDLL('msvcrt' if sys.platform == 'win32' else None)


def _run_kernel(kernel, x, tile):
    """Pipe stdout to a tempfile and return the kernel's output as a list of lines."""

    grid = (math.ceil(x.shape[0] / tile), 1, 1)
    with tempfile.TemporaryFile(mode='w+b') as tmpfile:
        saved_fd = os.dup(1)
        os.dup2(tmpfile.fileno(), 1)
        try:
            ct.launch(torch.cuda.current_stream(), grid, kernel, (x, tile))
            torch.cuda.synchronize()
            _libc.fflush(None)
            tmpfile.seek(0)
            out = tmpfile.read().decode('utf-8', errors='replace')
        finally:
            os.dup2(saved_fd, 1)
            os.close(saved_fd)
    return [line for line in out.splitlines() if line]


@ct.kernel(opt_level=_OPT_LEVEL)
def kernel_printf(x, TILE: ct.Constant[int]):
    bid = ct.bid(0)
    tx = ct.load(x, index=(bid,), shape=(TILE,))
    if x.dtype == float16 or x.dtype == float32:
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
    if x.dtype == float16 or x.dtype == float32 or x.dtype == float64:
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
    if x.dtype == float16 or x.dtype == float32 or x.dtype == float64:
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


def _test_print_1d(shape, tile, kernel, dtype):
    torch_dtype = getattr(torch, 'bool' if dtype.__name__ == 'bool_' else dtype.__name__)
    data = torch.arange(int(np.prod(shape)), device='cuda').reshape(shape).to(torch_dtype)
    actual_outs = _run_kernel(kernel, data, tile)

    x = np.arange(np.prod(shape)).reshape(shape).astype(dtype.__name__)
    num_tiles = math.ceil(shape[0] / tile)
    for i in range(num_tiles):
        start_idx, end_idx = tile * i, tile * (i + 1)
        if is_float(dtype):
            formatted_x = ', '.join([f"{elem:.5f}" for elem in x[start_idx:end_idx]])
        else:
            formatted_x = ', '.join([f"{int(elem)}" for elem in x[start_idx:end_idx]])
        expected_out = f"tile[{i}]:[{formatted_x}]"
        assert expected_out in actual_outs


@pytest.mark.parametrize("shape", [(8,), (16,)])
@pytest.mark.parametrize("tile", [8])
@pytest.mark.parametrize("dtype", [*NumericDTypeCategories.Integral, float16, float32, float64])
def test_printf(shape, tile, dtype):
    _test_print_1d(shape, tile, kernel_printf, dtype)


@pytest.mark.parametrize("shape", [(8,), (16,)])
@pytest.mark.parametrize("tile", [8])
@pytest.mark.parametrize("dtype", [*NumericDTypeCategories.Integral, float16, float32, float64])
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
    data = torch.arange(int(np.prod(shape)), device='cuda').reshape(shape).to(torch.int32)
    actual_outs = _run_kernel(kernel_print_sep, data, tile)
    x = np.arange(np.prod(shape)).reshape(shape).astype(np.int32)
    formatted_x = ', '.join([f"{elem}" for elem in x[:tile]])
    expected = f"tile:[{formatted_x}]"
    assert expected in actual_outs


@pytest.mark.parametrize("shape", [(8,), (16,)])
@pytest.mark.parametrize("tile", [8])
def test_ct_print_two_vars(shape, tile):
    data = torch.arange(int(np.prod(shape)), device='cuda').reshape(shape).to(torch.float32)
    actual_outs = _run_kernel(kernel_print_two_vars_with_expr, data, tile)
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
    data = torch.arange(int(np.prod(shape)), device='cuda').reshape(shape).to(torch.int32)
    actual_outs = _run_kernel(kernel_print_no_end, data, tile)
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
    data = torch.arange(int(np.prod(shape)), device='cuda').reshape(shape).to(torch.float32)
    actual_outs = _run_kernel(kernel_fstring_nested, data, tile)
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
    data = torch.arange(int(np.prod(shape)), device='cuda').reshape(shape).to(torch.int32)
    actual_outs = _run_kernel(kernel_print_aliases, data, tile)
    x = np.arange(np.prod(shape)).reshape(shape).astype(np.int32)
    num_tiles = math.ceil(shape[0] / tile)
    for i in range(num_tiles):
        start_idx, end_idx = tile * i, tile * (i + 1)
        formatted_x = ', '.join([f"{elem}" for elem in x[start_idx:end_idx]])
        assert f"ct%:[{formatted_x}]" in actual_outs
        assert f"builtin%:[{formatted_x}]" in actual_outs
        assert f"printf%[{i}]:[{formatted_x}]" in actual_outs


def test_ct_print_tuple():
    x = torch.arange(8, device='cuda', dtype=torch.int32).reshape(2, 4)
    actual_outs = _run_kernel(kernel_print_tuple, x, 2)
    assert actual_outs[0] == "(2, 4)"          # ct.print(x.shape)
    assert actual_outs[1] == "(2, 4)"          # print(x.shape)
    assert actual_outs[2] == "shape = (2, 4)"  # ct.print(f"shape = {x.shape}")
    assert actual_outs[3] == "shape: (2, 4)"   # ct.print("shape:", x.shape)


def test_ct_print_nested_tuple():
    x = torch.arange(8, device='cuda', dtype=torch.int32).reshape(2, 4)
    actual_outs = _run_kernel(kernel_print_nested_tuple, x, 2)
    assert actual_outs[0] == "nested tuple: (((2, 4), (2, 4)), 0)"
    assert actual_outs[1] == "(((2, 4), (2, 4)), 0)"


def test_ct_print_tuple_with_escaped_str():
    x = torch.arange(8, device='cuda', dtype=torch.int32).reshape(2, 4)
    actual_outs = _run_kernel(kernel_print_tuple_escaped_str, x, 2)
    assert actual_outs[0] == "tuple w/ escaped str: (((2, 4), '%d'), '%%d')"
    assert actual_outs[1] == "('foo', \"'''foo'''\", '\"foo\\\'')"
    assert actual_outs[2] == "('foo0', \"'''foo0'''\", '\"foo0\\\'')"
    assert actual_outs[3] == "(\"'\", '\"', '\\'\"')"


def test_ct_print_empty_tuple():
    x = torch.arange(8, device='cuda', dtype=torch.int32)
    actual_outs = _run_kernel(kernel_print_empty_tuple, x, 8)
    assert actual_outs[0] == "empty tuple: ()"
    assert actual_outs[1] == "()"


def test_ct_print_single_tuple():
    x = torch.arange(8, device='cuda', dtype=torch.int32)
    actual_outs = _run_kernel(kernel_print_single_tuple, x, 8)
    assert actual_outs[0] == "single tuple: (0,)"
    assert actual_outs[1] == "(0,)"


def test_ct_print_tuple_w_tile():
    shape = (8,)
    tile = 8
    data = torch.arange(int(np.prod(shape)), device='cuda', dtype=torch.int32).reshape(shape)
    actual_outs = _run_kernel(kernel_print_tuple_w_tile, data, tile)
    x = np.arange(np.prod(shape)).reshape(shape).astype(np.int32)
    formatted_x = ', '.join([f"{elem}" for elem in x[:tile]])
    assert actual_outs[0] == f"tuple w/ tile: ([{formatted_x}],)"
    assert actual_outs[1] == f"([{formatted_x}],)"


def test_ct_print_tuple_tile_shape():
    x = torch.arange(8, device='cuda', dtype=torch.int32)
    actual_outs = _run_kernel(kernel_print_tuple_tile_shape, x, 8)
    assert actual_outs[0] == "Tile shape: (8,)"
    assert actual_outs[1] == "(8,)"


def test_ct_print_tuple_fstring():
    x = torch.arange(8, device='cuda', dtype=torch.int32)
    actual_outs = _run_kernel(kernel_print_tuple_fstring, x, 8)
    assert actual_outs[0] == "('bid=0', 0)"


def test_ct_print_tuple_format_spec_error():
    from cuda.tile._exception import TileTypeError

    @ct.kernel(opt_level=_OPT_LEVEL)
    def bad_kernel(x, TILE: ct.Constant[int]):
        ct.print(f"{x.shape:10}")

    x = torch.zeros(8, device='cuda', dtype=torch.int32)
    with pytest.raises(TileTypeError, match="cannot apply format spec to a tuple value"):
        ct.launch(torch.cuda.current_stream(), (1, 1, 1), bad_kernel, (x, 8))
