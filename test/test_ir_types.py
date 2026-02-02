# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from cuda.tile import TileValueError
from cuda.tile._exception import TileTypeError
from cuda.tile._ir.type import (
    TupleTy, TileTy, ArrayTy, SizeTy, NONE, LooselyTypedScalar, make_tile_ty
)
from cuda.tile._datatype import (
    DType,
    float16, float32, float64, bool_,
    int64, int32, int16, int8,
    uint64, uint32, uint16, uint8, bfloat16,
    tfloat32, float8_e4m3fn, float8_e5m2,
    is_boolean, is_integral, is_float, is_restricted_float, is_signed,
)
from cuda.tile._ir.ops_utils import promote_dtypes, check_implicit_cast
from cuda.tile._ir.typing_support import to_dtype, typeof_pyval
import torch
import numpy as np
from util import get_ptr_16_byte_divisible_view, get_ptr_16_byte_non_divisible_view


def test_builtin_types():
    assert str(float16) == 'float16'
    assert str(int64) == 'int64'
    assert str(int32) == 'int32'
    assert str(int16) == 'int16'
    assert str(int8) == 'int8'
    assert str(bool_) == 'bool_'
    assert str(bfloat16) == 'bfloat16'
    assert str(uint64) == 'uint64'
    assert str(uint32) == 'uint32'
    assert str(float32) == 'float32'
    assert str(float64) == 'float64'
    assert str(tfloat32) == 'tfloat32'
    assert str(float8_e4m3fn) == 'float8_e4m3fn'
    assert str(float8_e5m2) == 'float8_e5m2'
    assert is_integral(int32)
    assert is_signed(int32)
    assert is_signed(float32)
    assert not is_signed(uint32)
    assert not is_signed(bool_)
    assert is_boolean(bool_)
    assert is_float(bfloat16)
    assert not is_float(uint32)
    assert is_restricted_float(tfloat32)
    assert is_restricted_float(float8_e4m3fn)
    assert is_restricted_float(float8_e5m2)

    # type equality
    assert float16 == DType('float16', 16, float, None)


def test_tuple_type():
    # tuple type
    shape = TupleTy((SizeTy(5), SizeTy(4)))
    assert len(shape) == 2
    assert shape[0].value == 5
    assert shape[1].value == 4


def test_tile_type():
    # tile type
    shape = TupleTy((SizeTy(5), SizeTy(4)))
    tile = TileTy(float16, shape)
    assert tile.dtype == float16
    assert tile.shape == shape

    # tile type equality
    tile2 = TileTy(float16, shape)
    assert tile == tile2

    with pytest.raises(TypeError):
        TileTy(float16, TupleTy((int32, SizeTy(3))))


def test_array_type():
    # array with dynamic shape
    arr = ArrayTy(bfloat16, shape=TupleTy((SizeTy(), SizeTy())),
                  strides=TupleTy((SizeTy(), SizeTy())),
                  elements_disjoint=True,
                  base_ptr_div_by=None,
                  stride_div_by=(None, None),
                  shape_div_by=(None, None))
    assert arr.dtype == bfloat16
    assert arr.elements_disjoint
    assert len(arr.shape) == 2
    assert len(arr.strides) == 2
    with pytest.raises(TypeError):
        arr.shape[0].value
    with pytest.raises(TypeError):
        arr.strides[0].value


def test_promote_dtypes():

    def assert_no_promote(t1, t2):
        with pytest.raises(TileTypeError):
            promote_dtypes(t1, t2)

    def assert_out_of_range(t1, t2):
        with pytest.raises(TileValueError, match="is out of range"):
            promote_dtypes(t1, t2)

    # Bool
    assert promote_dtypes(bool_, uint8) == uint8
    assert promote_dtypes(bool_, int16) == int16
    assert promote_dtypes(bool_, float32) == float32
    assert promote_dtypes(bool_, bfloat16) == bfloat16
    assert_no_promote(bool_, tfloat32)
    assert_no_promote(bool_, float8_e5m2)

    # Int
    assert promote_dtypes(int8, int16) == int16
    assert promote_dtypes(int32, float16) == float16
    assert promote_dtypes(int64, bfloat16) == bfloat16
    assert_no_promote(int32, tfloat32)
    assert_no_promote(int32, float8_e5m2)

    # Uint
    assert promote_dtypes(uint8, uint16) == uint16
    assert promote_dtypes(uint32, float16) == float16
    assert promote_dtypes(uint32, bfloat16) == bfloat16
    assert_no_promote(uint32, int32)
    assert_no_promote(uint32, int64)
    assert_no_promote(uint32, tfloat32)
    assert_no_promote(uint32, float8_e5m2)

    # float
    assert promote_dtypes(float16, float32) == float32
    assert promote_dtypes(bfloat16, float32) == float32
    assert_no_promote(float16, bfloat16)
    assert_no_promote(float16, tfloat32)
    assert_no_promote(float16, float8_e5m2)

    # Loosely typed scalars
    assert promote_dtypes(int16, LooselyTypedScalar(5)) == int16
    assert promote_dtypes(LooselyTypedScalar(5), int8) == int8
    assert promote_dtypes(LooselyTypedScalar(5), LooselyTypedScalar(7)) == int32
    assert promote_dtypes(LooselyTypedScalar(5), LooselyTypedScalar(7.0)) == float32
    assert promote_dtypes(int16, LooselyTypedScalar(5.0)) == float32
    assert promote_dtypes(float16, LooselyTypedScalar(5.0)) == float16
    assert promote_dtypes(bool_, LooselyTypedScalar(5)) == int32
    assert_out_of_range(int8, LooselyTypedScalar(128))


def test_check_implicit_cast():

    def allow(src, dst):
        if isinstance(src, DType):
            src = make_tile_ty(src, ())
        check_implicit_cast(src, dst)

    def disallow(src, dst):
        if isinstance(src, DType):
            src = make_tile_ty(src, ())
        with pytest.raises((TileTypeError, TileValueError)):
            check_implicit_cast(src, dst)

    # same category
    allow(int8, int8)
    allow(uint8, uint8)

    allow(int8, int16)
    disallow(int16, int8)

    allow(uint8, uint16)
    disallow(uint16, uint8)

    allow(float16, float32)
    disallow(float32, float16)

    allow(bfloat16, float32)
    disallow(float32, bfloat16)

    disallow(float16, bfloat16)
    disallow(bfloat16, float16)

    # bool -> int or float
    allow(bool_, int32)
    disallow(int32, bool_)

    allow(bool_, uint32)
    disallow(uint32, bool_)

    allow(bool_, float32)
    disallow(float32, bool_)

    allow(bool_, bfloat16)
    disallow(bfloat16, bool_)

    disallow(bool_, tfloat32)
    disallow(tfloat32, bool_)

    disallow(bool_, float8_e5m2)
    disallow(float8_e5m2, bool_)

    # int -> float
    allow(uint32, float16)
    disallow(float16, uint32)

    allow(int32, float16)
    disallow(float16, int32)

    disallow(uint32, tfloat32)
    disallow(tfloat32, uint32)

    disallow(uint32, float8_e5m2)
    disallow(float8_e5m2, uint32)

    disallow(int32, tfloat32)
    disallow(tfloat32, int32)

    disallow(int32, float8_e5m2)
    disallow(float8_e5m2, int32)

    # signed <> unsigned not allowed
    disallow(int32, uint32)
    disallow(uint32, int32)

    # restricted float not allowed
    disallow(float32, tfloat32)
    disallow(tfloat32, float32)

    disallow(float8_e5m2, tfloat32)
    disallow(tfloat32, float8_e5m2)

    # Loosely typed scalars
    allow(LooselyTypedScalar(10), int8)
    disallow(LooselyTypedScalar(128), int8)

    allow(LooselyTypedScalar(10), float32)
    allow(LooselyTypedScalar(4.0), float32)
    allow(LooselyTypedScalar(4.0), float16)

    allow(LooselyTypedScalar(False), bool_)
    allow(LooselyTypedScalar(True), bool_)
    allow(LooselyTypedScalar(1), bool_)
    allow(LooselyTypedScalar(0), bool_)
    disallow(LooselyTypedScalar(1.0), bool_)
    disallow(LooselyTypedScalar(0.0), bool_)


def test_np_dtype_support():
    assert to_dtype(np.float64) == float64
    assert to_dtype(np.float32) == float32
    assert to_dtype(np.float16) == float16
    assert to_dtype(np.int64) == int64
    assert to_dtype(np.int32) == int32
    assert to_dtype(np.int16) == int16
    assert to_dtype(np.int8) == int8
    assert to_dtype(np.uint64) == uint64
    assert to_dtype(np.uint32) == uint32
    assert to_dtype(np.uint16) == uint16
    assert to_dtype(np.uint8) == uint8
    assert to_dtype(np.bool_) == bool_

    assert to_dtype(np.dtype('float64')) == float64
    assert to_dtype(np.dtype('float32')) == float32
    assert to_dtype(np.dtype('float16')) == float16
    assert to_dtype(np.dtype('int64')) == int64
    assert to_dtype(np.dtype('int32')) == int32
    assert to_dtype(np.dtype('int16')) == int16
    assert to_dtype(np.dtype('int8')) == int8
    assert to_dtype(np.dtype('uint64')) == uint64
    assert to_dtype(np.dtype('uint32')) == uint32
    assert to_dtype(np.dtype("uint16")) == uint16
    assert to_dtype(np.dtype("uint8")) == uint8
    assert to_dtype(np.dtype('bool_')) == bool_


def test_torch_dtype_support():
    assert to_dtype(torch.float64) == float64
    assert to_dtype(torch.float32) == float32
    assert to_dtype(torch.float16) == float16
    assert to_dtype(torch.int64) == int64
    assert to_dtype(torch.int32) == int32
    assert to_dtype(torch.int16) == int16
    assert to_dtype(torch.int8) == int8
    assert to_dtype(torch.uint64) == uint64
    assert to_dtype(torch.uint32) == uint32
    assert to_dtype(torch.uint16) == uint16
    assert to_dtype(torch.uint8) == uint8
    assert to_dtype(torch.bool) == bool_
    assert to_dtype(torch.bfloat16) == bfloat16
    assert to_dtype(torch.float8_e4m3fn) == float8_e4m3fn
    assert to_dtype(torch.float8_e5m2) == float8_e5m2


def _array_base_equal(arryty: ArrayTy, dtype, shape, strides):
    return (arryty.dtype == dtype and
            arryty.shape == shape and
            arryty.strides == strides)


def test_typeof_pyval():
    tp = typeof_pyval
    assert tp(1) == make_tile_ty(int32, ())
    assert tp(1.) == make_tile_ty(float32, ())
    assert tp(np.int16(1)) == make_tile_ty(int16, ())
    assert tp(np.float64(1.0)) == make_tile_ty(float64, ())
    assert tp(True) == make_tile_ty(bool_, ())
    assert tp(None) == NONE

    # 0D tensor
    t = torch.tensor(0, device='cuda', dtype=torch.int32)
    assert _array_base_equal(tp(t), int32,
                             shape=TupleTy(()),
                             strides=TupleTy(()))

    # 1D tensor
    t = torch.zeros(4, device='cuda', dtype=torch.int32)
    assert _array_base_equal(tp(t), int32,
                             shape=TupleTy((SizeTy(None),)),
                             strides=TupleTy((SizeTy(1),)))

    # 2D transposed tensor, dim[0] contiguous
    t = torch.zeros(2, 4, device='cuda', dtype=torch.int32).t()
    assert t.stride() == (1, 4)
    assert _array_base_equal(tp(t), int32,
                             shape=TupleTy((SizeTy(None), SizeTy(None))),
                             strides=TupleTy((SizeTy(1), SizeTy(None))))

    # 2D transposed tensor, no dim contiguous
    t = torch.zeros(4, 6, device='cuda', dtype=torch.int32)[::2, ::3]
    assert t.stride() == (12, 3)
    assert _array_base_equal(tp(t), int32,
                             shape=TupleTy((SizeTy(None), SizeTy(None))),
                             strides=TupleTy((SizeTy(None), SizeTy(None))))

    # 2D tensor, dim[1] stride 0
    t = torch.zeros(3, 1, device='cuda', dtype=torch.int32).broadcast_to((3, 3))
    assert t.stride() == (1, 0)
    assert _array_base_equal(tp(t), int32,
                             shape=TupleTy((SizeTy(None), SizeTy(None))),
                             strides=TupleTy((SizeTy(1), SizeTy(None))))

    # 2D tensor, contiguous, dim[0] and dim[1] stride 1
    t = torch.zeros(1, 1, device='cuda', dtype=torch.int32)
    assert t.stride() == (1, 1)
    assert _array_base_equal(tp(t), int32,
                             shape=TupleTy((SizeTy(None), SizeTy(None))),
                             strides=TupleTy((SizeTy(1), SizeTy(1))))


def test_type_of_pyval_numba(numba_cuda):
    tp = typeof_pyval
    t = numba_cuda.to_device(np.zeros(4, dtype=np.int32))
    assert _array_base_equal(tp(t), int32,
                             shape=TupleTy((SizeTy(None),)),
                             strides=TupleTy((SizeTy(1),)))

    t = numba_cuda.to_device(np.transpose(np.zeros((2, 4), dtype=np.int32)))
    assert _array_base_equal(tp(t), int32,
                             shape=TupleTy((SizeTy(None), SizeTy(None))),
                             strides=TupleTy((SizeTy(1), SizeTy(None))))

    t = numba_cuda.to_device(np.zeros((4, 6), dtype=np.int32))[::2, ::3]
    assert _array_base_equal(tp(t), int32,
                             shape=TupleTy((SizeTy(None), SizeTy(None))),
                             strides=TupleTy((SizeTy(None), SizeTy(None))))


def _assert_array_specialization_equal(arryty: ArrayTy, elements_disjoint,
                                       base_ptr_div_by, stride_div_by, shape_div_by):
    assert arryty.elements_disjoint == elements_disjoint, (
        f'elements_disjoint: {arryty.elements_disjoint} != {elements_disjoint}'
    )
    assert arryty.base_ptr_div_by == base_ptr_div_by, (
        f'base_ptr_div_by:{arryty.base_ptr_div_by} != {base_ptr_div_by}'
    )
    assert arryty.stride_div_by == stride_div_by, (
        f'stride_div_by: {arryty.stride_div_by} != {stride_div_by}'
    )
    assert arryty.shape_div_by == shape_div_by, (
        f'shape_div_by: {arryty.shape_div_by} != {shape_div_by}'
    )


def test_typeof_pyval_array_specialization():
    tp = typeof_pyval

    four_byte_dtype = torch.int32
    div_by_16 = 16
    two_byte_dtype = torch.float16

    def get_base_ptr_div_by(data_ptr):
        return 16 if data_ptr % 16 == 0 else None

    # Test base ptr div by

    # 1D tensor, contiguous, no base ptr div by
    t = get_ptr_16_byte_non_divisible_view(
        torch.zeros(32, device='cuda', dtype=four_byte_dtype))
    assert t.shape == (31,)
    _assert_array_specialization_equal(tp(t), True, None, (None,), (None,))
    # 1D tensor, contiguous, base ptr div by 16
    t = get_ptr_16_byte_divisible_view(
        torch.zeros(32, device='cuda', dtype=four_byte_dtype))
    _assert_array_specialization_equal(tp(t), True, 16, (None,), (div_by_16,))

    # Test stride div by

    # 0D tensor, contiguous, no stride div by
    t = torch.tensor(0, device='cuda', dtype=four_byte_dtype)
    assert t.stride() == tuple()
    _assert_array_specialization_equal(tp(t), True, get_base_ptr_div_by(t.data_ptr()),
                                       tuple(), tuple())

    # 2D tensor, contiguous, no stride div by, no shape div by
    t = torch.zeros(4, 2, device='cuda', dtype=four_byte_dtype)
    assert t.stride() == (2, 1)
    _assert_array_specialization_equal(tp(t), True, get_base_ptr_div_by(t.data_ptr()),
                                       (None, None), (None, None))

    # 2D tensor, contiguous, 4 byte dtype, dim[0] stride div by 16
    t = torch.zeros(4, 4, device='cuda', dtype=four_byte_dtype)
    assert t.stride() == (4, 1)
    _assert_array_specialization_equal(tp(t), True, get_base_ptr_div_by(t.data_ptr()),
                                       (4, None),
                                       (None, None))

    # 2D tensor, contiguous, 2 byte dtype, no stride div by
    t = torch.zeros(4, 4, device='cuda', dtype=two_byte_dtype)
    assert t.stride() == (4, 1)
    _assert_array_specialization_equal(tp(t), True, get_base_ptr_div_by(t.data_ptr()),
                                       (None, None), (None, None))

    # 1D tensor, non-contiguous, no stride div by
    t = torch.zeros(32, device='cuda', dtype=four_byte_dtype)[::2]
    assert t.stride() == (2,)
    _assert_array_specialization_equal(tp(t), True, get_base_ptr_div_by(t.data_ptr()),
                                       (None,), (div_by_16,))

    # 2D tensor, non-contiguous, dim[0] stride div by 16
    t = torch.zeros(4, 2, device='cuda', dtype=four_byte_dtype)[::2, ::2]
    assert t.stride() == (4, 2)
    assert t.shape == (2, 1)
    _assert_array_specialization_equal(tp(t), True, get_base_ptr_div_by(t.data_ptr()),
                                       (4, None), (None, None))

    # 2D tensor, dim[1] 0 stride div by 16
    t = torch.zeros(3, 1, device='cuda', dtype=four_byte_dtype).broadcast_to((3, 3))
    assert t.stride() == (1, 0)
    _assert_array_specialization_equal(tp(t), False, get_base_ptr_div_by(t.data_ptr()),
                                       (None, 4), (None, None))


def test_typeof_pyval_array_specialization_numba(numba_cuda):
    tp = typeof_pyval

    def get_base_ptr_div_by(data_ptr):
        return 16 if data_ptr % 16 == 0 else None

    # stride (1,), 4 byte dtype
    t = numba_cuda.to_device(np.zeros(4, dtype=np.int32))
    _assert_array_specialization_equal(
        tp(t),
        True,
        get_base_ptr_div_by(t.__cuda_array_interface__['data'][0]),
        (None,), (None,))

    # stride (1, 4), 4 byte dtype
    t = numba_cuda.to_device(np.transpose(np.zeros((2, 4), dtype=np.int32)))
    _assert_array_specialization_equal(
        tp(t),
        True,
        get_base_ptr_div_by(t.__cuda_array_interface__['data'][0]),
        (None, 4), (None, None))

    # stride (12, 3), 4 byte dtype
    t = numba_cuda.to_device(np.zeros((4, 6), dtype=np.int32))[::2, ::3]
    _assert_array_specialization_equal(
        tp(t),
        True,
        get_base_ptr_div_by(t.__cuda_array_interface__['data'][0]),
        (4, None), (None, None))
