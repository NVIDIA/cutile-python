# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
import re

import pytest
from unittest.mock import patch

import cuda.tile
import cuda.tile as ct
import torch

from util import assert_equal


# ============================================================
# Basic cases
# ============================================================

@ct.kernel
def kernel_scalar_tuple(a, out, addends):
    # Load a tile and add both scalar tuple elements to it.
    t = ct.load(a, (0,), (8,))
    result = t + addends[0] + addends[1]
    ct.store(out, (0,), result)


def test_tuple_scalar_arg():
    a = torch.zeros(8, dtype=torch.int32, device="cuda")
    out = torch.zeros(8, dtype=torch.int32, device="cuda")
    ct.launch(torch.cuda.current_stream(), (1,), kernel_scalar_tuple, (a, out, (3, 7)))
    assert_equal(out, torch.full((8,), 10, dtype=torch.int32, device="cuda"))


@ct.kernel
def kernel_array_tuple(pair, out):
    a = ct.load(pair[0], (0, 0), (4, 4))
    b = ct.load(pair[1], (0, 0), (4, 4))
    ct.store(out, (0, 0), a + b)


def test_tuple_array_arg():
    # Pass a tuple[Tensor, Tensor].
    a = torch.ones(4, 4, dtype=torch.float32, device="cuda")
    b = torch.full((4, 4), 2.0, dtype=torch.float32, device="cuda")
    out = torch.zeros(4, 4, dtype=torch.float32, device="cuda")
    ct.launch(torch.cuda.current_stream(), (1,), kernel_array_tuple, ((a, b), out))
    assert_equal(out, a + b)


@ct.kernel
def kernel_mixed_tuple(pair, out):
    t = ct.load(pair[0], (0,), (8,))
    result = t + pair[1]
    ct.store(out, (0,), result)


def test_tuple_mixed_arg():
    # Pass a tuple[Tensor, int].
    data = torch.ones(8, dtype=torch.int32, device="cuda")
    out = torch.zeros(8, dtype=torch.int32, device="cuda")
    ct.launch(torch.cuda.current_stream(), (1,), kernel_mixed_tuple, ((data, 5), out))
    assert_equal(out, torch.full((8,), 6, dtype=torch.int32, device="cuda"))


def make_i64_index_tuple_kernel(annotation):
    @ct.kernel
    def k(pair: annotation, out):
        a = ct.load(pair[0], (0,), (16,))
        b = ct.load(pair[1], (0,), (16,))
        ct.store(out, (0,), a + b)
    return k


@pytest.mark.parametrize("annotation", [
    pytest.param(tuple[ct.IndexedWithInt64, ct.IndexedWithInt64], id="both_i64"),
    pytest.param(tuple[ct.IndexedWithInt64, torch.Tensor],       id="first_i64"),
    pytest.param(tuple[torch.Tensor, ct.IndexedWithInt64],       id="second_i64"),
])
def test_tuple_i64_index_arg(annotation):
    # Pass a tuple with ct.IndexedWithInt64.
    a = torch.ones(16, dtype=torch.float32, device="cuda")
    b = torch.full((16,), 2.0, dtype=torch.float32, device="cuda")
    out = torch.zeros(16, dtype=torch.float32, device="cuda")
    k = make_i64_index_tuple_kernel(annotation)
    ct.launch(torch.cuda.current_stream(), (1,), k, ((a, b), out))
    assert_equal(out, a + b)


def make_mixed_scalar_kernel(annotation):
    @ct.kernel
    def k(a, scalars: annotation, out):
        t = ct.load(a, (0,), (8,))
        result = t + scalars[0] - scalars[1]
        ct.store(out, (0,), result)
    return k


@pytest.mark.parametrize("annotation,scalars", [
    pytest.param(tuple[ct.ScalarInt64, int], (2**33 + 7, 5), id="i64_first"),
    pytest.param(tuple[int, ct.ScalarInt64], (5, 2**33 + 7), id="i32_first"),
])
def test_mixed_scalar_tuple_arg(annotation, scalars):
    a = torch.zeros(8, dtype=torch.int64, device="cuda")
    out = torch.zeros(8, dtype=torch.int64, device="cuda")
    k = make_mixed_scalar_kernel(annotation)
    ct.launch(torch.cuda.current_stream(), (1,), k, (a, scalars, out))
    assert_equal(out, torch.full((8,), scalars[0] - scalars[1], dtype=torch.int64, device="cuda"))


@ct.kernel
def kernel_constant_tuple(a, out, shape: ct.Constant[tuple]):
    t = ct.load(a, (0,), (shape[0],))
    ct.store(out, (0,), t)


def test_constant_tuple_arg():
    N = 8
    a = torch.arange(N, dtype=torch.float32, device="cuda")
    out = torch.zeros(N, dtype=torch.float32, device="cuda")
    ct.launch(torch.cuda.current_stream(), (1,), kernel_constant_tuple, (a, out, (N,)))
    assert_equal(out, a)


@ct.kernel
def kernel_constant_i64_scalar_tuple(a, out, addends: ct.Constant[tuple[ct.ScalarInt64, int]]):
    t = ct.load(a, (0,), (8,))
    result = t + addends[0] - addends[1]
    ct.store(out, (0,), result)


def test_constant_i64_scalar_tuple_arg():
    i64_val = 2**33 + 7
    i32_val = 5
    a = torch.zeros(8, dtype=torch.int64, device="cuda")
    out = torch.zeros(8, dtype=torch.int64, device="cuda")

    expected_message = re.escape(
            "Constant annotation cannot be combined with ScalarAnnotation/ScalarInt64")
    with pytest.raises(TypeError, match=expected_message):
        ct.launch(torch.cuda.current_stream(), (1,), kernel_constant_i64_scalar_tuple,
                  (a, out, (i64_val, i32_val)))


@ct.kernel
def kernel_partial_const_first(a, out, cfg: tuple[ct.Constant[int], int]):
    t = ct.load(a, (0,), (cfg[0],))
    ct.store(out, (0,), t + cfg[1])


def test_partial_const_tuple_first():
    N, M = 8, 5
    a = torch.arange(N, dtype=torch.int32, device="cuda")
    out = torch.zeros(N, dtype=torch.int32, device="cuda")
    ct.launch(torch.cuda.current_stream(), (1,), kernel_partial_const_first, (a, out, (N, M)))
    assert_equal(out, torch.arange(N, dtype=torch.int32, device="cuda") + M)


@ct.kernel
def kernel_partial_const_second(a, out, cfg: tuple[int, ct.Constant[int]]):
    t = ct.load(a, (0,), (cfg[1],))
    ct.store(out, (0,), t + cfg[0])


def test_partial_const_tuple_second():
    N, M = 8, 3
    a = torch.arange(N, dtype=torch.int32, device="cuda")
    out = torch.zeros(N, dtype=torch.int32, device="cuda")
    ct.launch(torch.cuda.current_stream(), (1,), kernel_partial_const_second, (a, out, (M, N)))
    assert_equal(out, torch.arange(N, dtype=torch.int32, device="cuda") + M)


def test_tuple_arg_empty():
    @ct.kernel
    def k(out, empty):
        ct.scatter(out, (), len(empty) + 1)

    out = torch.zeros((), dtype=torch.int32, device="cuda")
    ct.launch(torch.cuda.current_stream(), (1,), k, (out, ()))
    assert out.item() == 1


# ============================================================
# Nested tuple arguments
# ============================================================

@ct.kernel
def kernel_nested_scalar_tuple(a, out, cfg):
    t = ct.load(a, (0,), (8,))
    result = t * cfg[0][1] + cfg[0][0] + cfg[1]
    ct.store(out, (0,), result)


def test_nested_scalar_tuple_arg():
    a = torch.ones(8, dtype=torch.int32, device="cuda")
    out = torch.zeros(8, dtype=torch.int32, device="cuda")
    # 1 * 3 + 2 + 5 = 10
    ct.launch(torch.cuda.current_stream(), (1,), kernel_nested_scalar_tuple,
              (a, out, ((2, 3), 5)))
    assert_equal(out, torch.full((8,), 10, dtype=torch.int32, device="cuda"))


@ct.kernel
def kernel_nested_mixed_tuple(pair, out):
    t = ct.load(pair[0], (0,), (8,))
    result = t + pair[1][0] + pair[1][1]
    ct.store(out, (0,), result)


def test_nested_mixed_tuple_arg():
    data = torch.ones(8, dtype=torch.int32, device="cuda")
    out = torch.zeros(8, dtype=torch.int32, device="cuda")
    ct.launch(torch.cuda.current_stream(), (1,), kernel_nested_mixed_tuple,
              ((data, (3, 7)), out))
    assert_equal(out, torch.full((8,), 11, dtype=torch.int32, device="cuda"))


def test_tuple_arg_contains_list():
    @ct.kernel
    def k(pair, out):
        res = ct.zeros((8,), dtype=out.dtype)
        for i in range(len(pair[0])):
            t = ct.load(pair[0][i], (0,), (8,))
            res = res + t
        ct.store(out, (0,), res + pair[1])

    a = torch.ones(8, dtype=torch.float32, device="cuda")
    b = torch.full((8,), 2.0, dtype=torch.float32, device="cuda")
    out = torch.zeros(8, dtype=torch.float32, device="cuda")
    ct.launch(torch.cuda.current_stream(), (1,), k, (([a, b], 3), out))
    assert_equal(out, torch.full((8,), 6.0, dtype=torch.float32, device="cuda"))


@ct.kernel
def kernel_array_const_tuple(pair: tuple[torch.Tensor, ct.Constant[int]], out):
    t = ct.load(pair[0], (0,), (pair[1],))
    ct.store(out, (0,), t)


def test_tuple_annotation_array_and_const():
    N = 8
    a = torch.arange(N, dtype=torch.float32, device="cuda")
    out = torch.zeros(N, dtype=torch.float32, device="cuda")
    ct.launch(torch.cuda.current_stream(), (1,), kernel_array_const_tuple, ((a, N), out))
    assert_equal(out, a)


@ct.kernel
def kernel_nested_partial_const(a, out, cfg: tuple[tuple[ct.Constant[int], int], int]):
    t = ct.load(a, (0,), (cfg[0][0],))
    ct.store(out, (0,), t + cfg[0][1] + cfg[1])


def test_nested_tuple_partial_const():
    N, M1, M2 = 8, 3, 5
    a = torch.arange(N, dtype=torch.int32, device="cuda")
    out = torch.zeros(N, dtype=torch.int32, device="cuda")
    ct.launch(torch.cuda.current_stream(), (1,), kernel_nested_partial_const,
              (a, out, ((N, M1), M2)))
    assert_equal(out, torch.arange(N, dtype=torch.int32, device="cuda") + M1 + M2)


def test_nested_tuple_partial_const_recompilation():
    stream = torch.cuda.current_stream()
    N = 8
    a = torch.arange(N, dtype=torch.int32, device="cuda")
    out = torch.zeros(N, dtype=torch.int32, device="cuda")

    kernel = cuda.tile.kernel(kernel_nested_partial_const._pyfunc)

    with patch('cuda.tile._compile.compile_tile',
               side_effect=cuda.tile._compile.compile_tile) as mock:
        # First call
        ct.launch(stream, (1,), kernel, (a, out, ((N, 3), 5)))
        assert mock.call_count == 1

        # Runtime values change — no recompilation.
        ct.launch(stream, (1,), kernel, (a, out, ((N, 7), 5)))
        assert mock.call_count == 1

        # Constant changes — recompilation.
        ct.launch(stream, (1,), kernel, (a, out, ((16, 7), 5)))
        assert mock.call_count == 2


def test_nested_tuple_different_structures():
    # Both kernels receive 4 scalar leaves but with different nesting structures.
    @ct.kernel
    def k(cfg, out):
        ct.scatter(out, (0,), cfg[0][0] + cfg[0][1])

    stream = torch.cuda.current_stream()
    out = torch.zeros(1, dtype=torch.int32, device="cuda")

    ct.launch(stream, (1,), k, (((1, 2), 3, 4), out))
    assert out[0] == 3

    ct.launch(stream, (1,), k, (((1, 2, 3), 4), out))
    assert out[0] == 3


def test_tuple_with_variable_length_annotation():
    @ct.kernel
    def k(out, addends: tuple[ct.Constant[int], ...]):
        ct.scatter(out, (0,), addends[0] + addends[1])

    stream = torch.cuda.current_stream()
    out = torch.zeros(1, dtype=torch.int32, device="cuda")

    with patch('cuda.tile._compile.compile_tile',
               side_effect=cuda.tile._compile.compile_tile) as mock:
        ct.launch(stream, (1,), k, (out, (1, 2)))
        assert out[0] == 3
        assert mock.call_count == 1

        # Same constants — no recompilation.
        ct.launch(stream, (1,), k, (out, (1, 2)))
        assert mock.call_count == 1

        # Element 0 changes — recompilation triggered.
        ct.launch(stream, (1,), k, (out, (3, 2)))
        assert mock.call_count == 2

        # Element 1 changes — recompilation triggered.
        ct.launch(stream, (1,), k, (out, (3, 4)))
        assert mock.call_count == 3


def test_nested_tuple_bare_tuple_annotation():
    @ct.kernel
    def k(out, addends: tuple[tuple, ct.Constant[int]]):
        ct.scatter(out, (0,), addends[0][0] + addends[1])

    stream = torch.cuda.current_stream()
    out = torch.zeros(1, dtype=torch.int32, device="cuda")

    with patch('cuda.tile._compile.compile_tile',
               side_effect=cuda.tile._compile.compile_tile) as mock:
        ct.launch(stream, (1,), k, (out, ((1, 2), 3)))
        assert out[0] == 4
        assert mock.call_count == 1

        # Same constant — no recompilation.
        ct.launch(stream, (1,), k, (out, ((5, 6), 3)))
        assert mock.call_count == 1

        # Different constant — recompilation.
        ct.launch(stream, (1,), k, (out, ((5, 6), 7)))
        assert mock.call_count == 2


def test_nested_tuple_variable_length_tuple_annotation():
    @ct.kernel
    def k(out, addends: tuple[tuple[int, ...], ct.Constant[int]]):
        ct.scatter(out, (0,), addends[0][0] + addends[1])

    stream = torch.cuda.current_stream()
    out = torch.zeros(1, dtype=torch.int32, device="cuda")

    with patch('cuda.tile._compile.compile_tile',
               side_effect=cuda.tile._compile.compile_tile) as mock:
        ct.launch(stream, (1,), k, (out, ((1, 2), 3)))
        assert out[0] == 4
        assert mock.call_count == 1

        # Same constant — no recompilation.
        ct.launch(stream, (1,), k, (out, ((5, 6), 3)))
        assert mock.call_count == 1

        # Different constant — recompilation.
        ct.launch(stream, (1,), k, (out, ((5, 6), 7)))
        assert mock.call_count == 2


def test_variable_length_tuple_structured_element():
    @ct.kernel
    def k(out, addends: tuple[tuple[int, ct.Constant[int]], ...]):
        ct.scatter(out, (0,), addends[0][0] + addends[1][0])

    stream = torch.cuda.current_stream()
    out = torch.zeros(1, dtype=torch.int32, device="cuda")

    with patch('cuda.tile._compile.compile_tile',
               side_effect=cuda.tile._compile.compile_tile) as mock:
        ct.launch(stream, (1,), k, (out, ((1, 2), (3, 4))))
        assert out[0] == 4
        assert mock.call_count == 1

        # Same constants — no recompilation.
        ct.launch(stream, (1,), k, (out, ((5, 2), (6, 4))))
        assert out[0] == 11
        assert mock.call_count == 1

        # Constant changes — recompilation triggered.
        ct.launch(stream, (1,), k, (out, ((5, 10), (6, 20))))
        assert out[0] == 11
        assert mock.call_count == 2


# ============================================================
# Error cases
# ============================================================


def test_constant_tuple_array_element_rejected():
    @ct.kernel
    def k(a, out, c: ct.Constant[tuple]):
        t = ct.load(a, (0,), (8,))
        ct.store(out, (0,), t)

    a = torch.zeros(8, dtype=torch.float32, device="cuda")
    out = torch.zeros(8, dtype=torch.float32, device="cuda")
    expected_message = re.escape(
        "Invalid item #0 of kernel argument #2:"
        " Could not interpret object of type 'Tensor' as a constant."
    )
    with pytest.raises(TypeError, match=expected_message):
        ct.launch(torch.cuda.current_stream(), (1,), k, (a, out, (a, 1)))


def test_tuple_more_than_annotation_size():
    @ct.kernel
    def k(out, addends: tuple[int, ct.Constant[int]]):
        ct.scatter(out, (0,), addends[0] + addends[1])

    out = torch.zeros(1, dtype=torch.int32, device="cuda")
    with pytest.raises(TypeError, match=r"Received a tuple of length 3"
                                        r" for a parameter annotated as a tuple of length 2"):
        ct.launch(torch.cuda.current_stream(), (1,), k, (out, ((1, 2, 3))))


def test_tuple_fewer_than_annotation_size():
    @ct.kernel
    def k(out, addends: tuple[int, ct.Constant[int]]):
        ct.scatter(out, (0,), addends[0] + addends[1])

    out = torch.zeros(1, dtype=torch.int32, device="cuda")
    with pytest.raises(TypeError, match=r"Received a tuple of length 1"
                                        r" for a parameter annotated as a tuple of length 2"):
        ct.launch(torch.cuda.current_stream(), (1,), k, (out, ((1, ))))


def test_nested_tuple_wrong_annotation_size():
    @ct.kernel
    def k(out, addends: tuple[tuple[int, ct.Constant[int]], ct.Constant[int]]):
        ct.scatter(out, (0,), addends[0][1] + addends[1])

    out = torch.zeros(1, dtype=torch.int32, device="cuda")
    with pytest.raises(TypeError, match=r"Received a tuple of length 3"
                                        r" for a parameter annotated as a tuple of length 2"):
        ct.launch(torch.cuda.current_stream(), (1,), k, (out, ((1, 2, 3), 4)))


def test_namedtuple_rejected():
    from collections import namedtuple
    Point = namedtuple('Point', ['x', 'y'])

    @ct.kernel
    def k(out, p):
        ct.scatter(out, (0,), p[0] + p[1])

    out = torch.zeros(1, dtype=torch.int32, device="cuda")
    with pytest.raises(TypeError,
                       match=re.escape("Invalid kernel argument #1:"
                                       " 'Point' is a subclass of 'tuple'."
                                       " Only plain tuples are accepted.")):
        ct.launch(torch.cuda.current_stream(), (1,), k, (out, Point(1, 2)))


def test_namedtuple_nested_rejected():
    from collections import namedtuple
    Point = namedtuple('Point', ['x', 'y'])

    @ct.kernel
    def k(out, pair):
        ct.scatter(out, (0,), pair[0] + pair[1][0])

    out = torch.zeros(1, dtype=torch.int32, device="cuda")
    with pytest.raises(TypeError,
                       match=re.escape("Invalid item #1 of kernel argument #1:"
                                       " 'Point' is a subclass of 'tuple'."
                                       " Only plain tuples are accepted.")):
        ct.launch(torch.cuda.current_stream(), (1,), k, (out, (1, Point(2, 3))))
