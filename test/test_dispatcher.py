# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import re
import pytest
import torch

from math import ceil
from contextlib import contextmanager
from unittest.mock import patch
import cuda.tile
import cuda.tile as ct
from util import (
    assert_equal,
    get_ptr_16_byte_divisible_view,
    get_ptr_16_byte_non_divisible_view
)
from torch.testing import make_tensor
from util import jit_kernel


@contextmanager
def clear_kernel_cache(old_kernel):
    # Create a new dispatcher for clean slate multi-level caches
    yield cuda.tile.kernel(old_kernel._pyfunc)


@ct.kernel
def array_inc_1d(x, TILE: ct.Constant[int]):
    bid = ct.bid(0)
    tx = ct.load(x, index=(bid,), shape=(TILE,))
    tx += 1
    ct.store(x, index=(bid,), tile=tx)


def launch_array_inc_1d(kernel, shape, tile):
    x = make_tensor(shape, dtype=torch.float32, device="cuda")
    ref = x + 1
    grid = (ceil(shape[0] / tile), 1, 1)
    ct.launch(torch.cuda.current_stream(), grid, kernel, (x, tile))
    assert_equal(x, ref)


def launch_array_inc_1d_stream(kernel, shape, tile):
    x = make_tensor(shape, dtype=torch.float32, device="cuda")
    ref = x + 1
    torch.cuda.synchronize()
    stream = torch.cuda.Stream()
    grid = (ceil(shape[0] / tile), 1, 1)
    ct.launch(stream, grid, kernel, (x, tile))
    stream.synchronize()
    assert_equal(x, ref)


kernel_cache = {}
array_add_n_kernel_template = """
def {name}(x, TILE: ct.Constant[int], N: {annotation}):
    bid = ct.bid(0)
    tx = ct.load(x, index=(bid,), shape=(TILE,))
    tx += N
    ct.store(x, index=(bid,), tile=tx)"""


def array_add_n_kernel(name: str, annotation: str, tmp_path):
    name = 'array_add_n_' + name
    source = array_add_n_kernel_template.format(name=name, annotation=annotation)
    if source not in kernel_cache:
        kernel_cache[source] = jit_kernel(name, source, tmp_path)
    return kernel_cache[source]


def launch_array_add_n(kernel, shape, tile, n):
    x = make_tensor(shape, dtype=torch.float32, device="cuda")
    ref = x + n
    grid = (ceil(shape[0] / tile), 1, 1)
    ct.launch(torch.cuda.current_stream(), grid, kernel, (x, tile, n))
    assert_equal(x, ref)


def test_multi_launch_single_compile():
    launch_funcs = [launch_array_inc_1d,
                    launch_array_inc_1d_stream]
    shapes = ((128,), (256,), (384,), (128,))
    tile = 128
    with patch(
        'cuda.tile._compile.compile_tile',
        side_effect=cuda.tile._compile.compile_tile
    ) as mock_compile_tile:
        with clear_kernel_cache(array_inc_1d) as kernel:
            for fn in launch_funcs:
                for shape in shapes:
                    fn(kernel, shape, tile)
            assert mock_compile_tile.call_count == 1


@pytest.mark.parametrize("annotation", ["ct.Constant[int]", "int"])
def test_int_arg_compile_count(annotation, tmp_path):
    shape = (256,)
    tile = 128
    ints = [0, 1, 2, 3, 4]
    expected = len(ints) if annotation == "ct.Constant[int]" else 1
    with patch(
        'cuda.tile._compile.compile_tile',
        side_effect=cuda.tile._compile.compile_tile
    ) as mock_compile_tile:
        kernel = array_add_n_kernel("int_arg", annotation, tmp_path)
        for i in ints:
            launch_array_add_n(kernel, shape, tile, i)
        assert mock_compile_tile.call_count == expected


@pytest.mark.parametrize("annotation", ["ct.Constant[float]", "float"])
def test_float_arg_compile_count(annotation, tmp_path):
    shape = (256,)
    tile = 128
    floats = [0.0, float('-inf'), float('inf'), 3.14, (3.14 + 1e-6)]
    expected = len(floats) if annotation == "ct.Constant[float]" else 1
    with patch(
        'cuda.tile._compile.compile_tile',
        side_effect=cuda.tile._compile.compile_tile
    ) as mock_compile_tile:
        kernel = array_add_n_kernel("float_arg", annotation, tmp_path)
        for f in floats:
            launch_array_add_n(kernel, shape, tile, f)
        assert mock_compile_tile.call_count == expected


@pytest.mark.parametrize("shape", ((128,), (256,), (384,), (400,)))
@pytest.mark.parametrize("tile", (128,))
def test_launch_grid_padding(shape, tile):
    num_tiles = ceil(shape[0] / tile)
    # 1D, 2D, 3D grid tuples
    grids = [(num_tiles,), (num_tiles, 1), (num_tiles, 1, 1)]
    if num_tiles == 1:
        # 0D grid to test grid (1, 1, 1)
        grids.append(())
    x = make_tensor(shape, dtype=torch.float32, device="cuda")
    ref = x.clone()
    for grid in grids:
        ct.launch(torch.cuda.current_stream(), grid, array_inc_1d, (x, tile))
        ref += 1
        assert_equal(x, ref)


def test_stride_static_one_launch_check():
    dtype = torch.float16  # 2 bytes
    tile = 64

    with patch(
        'cuda.tile._compile.compile_tile',
        side_effect=cuda.tile._compile.compile_tile
    ) as mock_compile_tile:
        with clear_kernel_cache(array_inc_1d) as kernel:
            # First compilation: stride is (1,)
            A0 = torch.zeros(tile, dtype=dtype, device='cuda')
            assert A0.stride() == (1,)
            ref0 = A0 + 1
            grid = (1, 1, 1)
            ct.launch(torch.cuda.current_stream(), grid, kernel, (A0, tile))
            assert_equal(A0, ref0)
            assert mock_compile_tile.call_count == 1

            # Second compilation: stride is (2,), shape is (64,)
            A0.zero_()
            A1 = A0[::2]
            assert A1.stride() == (2,)
            ref1 = A1 + 1
            ct.launch(torch.cuda.current_stream(), grid, kernel, (A1, tile))
            assert_equal(A1, ref1)
            assert mock_compile_tile.call_count == 2

            # No re-compilation: stride is (4,), shape is (32,)
            A0.zero_()
            A2 = A0[::4]
            assert A2.stride() == (4,)
            ref2 = A2 + 1
            ct.launch(torch.cuda.current_stream(), grid, kernel, (A2, tile))
            assert_equal(A2, ref2)
            assert mock_compile_tile.call_count == 2


def test_stride_divisibility_launch_check():
    dtype = torch.float16  # 2 bytes
    tile = 64

    with patch(
        'cuda.tile._compile.compile_tile',
        side_effect=cuda.tile._compile.compile_tile
    ) as mock_compile_tile:
        with clear_kernel_cache(array_inc_1d) as kernel:
            A0 = torch.zeros(tile, dtype=dtype, device='cuda')

            # First compilation: stride is (8,), divisible by 16 bytes
            A1 = A0[::8]
            assert A1.stride() == (8,)
            ref1 = A1 + 1
            ct.launch(torch.cuda.current_stream(), (1,), kernel, (A1, tile))
            assert_equal(A1, ref1)
            assert mock_compile_tile.call_count == 1

            # Second compilation:
            # stride is (4,), not divisible by 16 bytes; shape is (32,), divisible by 16
            A0.zero_()
            A2 = A0[::4]
            assert A2.stride() == (4,)
            ref2 = A2 + 1
            ct.launch(torch.cuda.current_stream(), (1,), kernel, (A2, tile))
            assert_equal(A2, ref2)
            assert mock_compile_tile.call_count == 2

            # No re-compilation:
            # stride is (2,), not divisible by 16 bytes; shape is (64,), divisible by 16
            A0.zero_()
            A3 = A0[::2]
            assert A3.stride() == (2,)
            ref3 = A3 + 1
            ct.launch(torch.cuda.current_stream(), (1,), kernel, (A3, tile))
            assert_equal(A3, ref3)
            assert mock_compile_tile.call_count == 2

            # No re-compilation:
            # stride is (8,), divisible by 16 bytes; shape is (16,), divisible by 16
            A0.zero_()
            A4 = A0[::8]
            assert A4.stride() == (8,)
            ref4 = A4 + 1
            ct.launch(torch.cuda.current_stream(), (1,), kernel, (A4, tile))
            assert_equal(A4, ref4)
            assert mock_compile_tile.call_count == 2


def test_base_ptr_divisibility_launch_check():
    dtype = torch.float16  # 2 bytes
    tile = 64

    with patch(
        'cuda.tile._compile.compile_tile',
        side_effect=cuda.tile._compile.compile_tile
    ) as mock_compile_tile:
        with clear_kernel_cache(array_inc_1d) as kernel:
            A0 = torch.zeros(tile, dtype=dtype, device='cuda')

            # First compilation: base ptr is divisible by 16
            A1 = get_ptr_16_byte_divisible_view(A0)
            ref1 = A1 + 1
            ct.launch(torch.cuda.current_stream(), (1,), kernel, (A1, tile))
            assert_equal(A1, ref1)
            assert mock_compile_tile.call_count == 1

            # Second compilation: base ptr is not divisible by 16
            A0.zero_()
            A2 = get_ptr_16_byte_non_divisible_view(A0)
            ref2 = A2 + 1
            ct.launch(torch.cuda.current_stream(), (1,), kernel, (A2, tile))
            assert_equal(A2, ref2)
            assert mock_compile_tile.call_count == 2

            # No re-compilation: base ptr is divisible by 16
            A0.zero_()
            A3 = get_ptr_16_byte_divisible_view(A0)
            ref3 = A3 + 1
            ct.launch(torch.cuda.current_stream(), (1,), kernel, (A3, tile))
            assert_equal(A3, ref3)
            assert mock_compile_tile.call_count == 2


def test_max_grid_size():
    pytest.skip("Skipping test_max_grid_size as it has been hidden with the 24-bit limit")
    max_grid_size = cuda.tile._cext._get_max_grid_size(0)
    tile = 128
    x = make_tensor(tile, dtype=torch.float32, device="cuda")
    grid = (max_grid_size[0] + 1, 1, 1)

    expected_msg = f"Grid[0] is too big: max={max_grid_size[0]}, got={grid[0]}"
    with pytest.raises(ValueError, match=re.escape(expected_msg)):
        ct.launch(torch.cuda.current_stream(), grid, array_inc_1d, (x, tile))


def test_max_grid_size_24bit():
    max_grid_size = 2**24 - 1
    tile = 128
    x = make_tensor(tile, dtype=torch.float32, device="cuda")
    grid = (max_grid_size + 1, 1, 1)

    expected_msg = (
        f"Grid[0] exceeds 24-bit limit: max={max_grid_size}, got={grid[0]}. "
        "Use multiple kernel launches for larger workloads."
    )
    with pytest.raises(ValueError, match=re.escape(expected_msg)):
        ct.launch(torch.cuda.current_stream(), grid, array_inc_1d, (x, tile))


def test_wrong_number_of_kernel_arguments():
    @ct.kernel
    def kern_one_arg(x):
        print(x)

    @ct.kernel
    def kern_two_args(x, y):
        print(x, y)

    with pytest.raises(TypeError, match="Kernel expects 1 argument but 2 were given"):
        ct.launch(torch.cuda.current_stream(), (1,), kern_one_arg, (123, 456))

    with pytest.raises(TypeError, match="Kernel expects 2 arguments but 1 was given"):
        ct.launch(torch.cuda.current_stream(), (1,), kern_two_args, (123,))

    with pytest.raises(TypeError, match="Kernel expects 2 arguments but 3 were given"):
        ct.launch(torch.cuda.current_stream(), (1,), kern_two_args, (123, 456, 789))
