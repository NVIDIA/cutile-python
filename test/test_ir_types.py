# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
import copy
import functools
import pickle

import pytest

from cuda.tile import TileValueError
from cuda.tile._exception import TileTypeError
from cuda.tile._ir.type import (
    TupleTy, TileTy, ArrayTy, NONE, LooselyTypedScalar
)
from cuda.tile._datatype import (
    DType,
    float16, float32,
    float4_e2m1fn, float64, bool_,
    float8_e8m0fnu,
    int64, int32, int16, int8,
    uint64, uint32, uint16, uint8, bfloat16,
    tfloat32, float8_e4m3fn, float8_e5m2,
    is_boolean, is_integral, is_float, is_unrestricted_float, is_restricted_float, is_signed,
    IntegerInfo, opaque_pointer_dtype, pointer_dtype, PointerInfo,
)
from cuda.tile._ir.ops_utils import promote_dtypes
from cuda.tile._ir.typing_support import to_dtype
from cuda.tile._ir.cast_ops import check_implicit_cast
import torch
import numpy as np

from cuda.tile._memory_model import MemorySpace


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
    assert str(float8_e8m0fnu) == "float8_e8m0fnu"
    assert str(float4_e2m1fn) == "float4_e2m1fn"
    assert is_integral(int32)
    assert is_signed(int32)
    assert is_signed(float32)
    assert not is_signed(uint32)
    assert not is_signed(bool_)
    assert is_boolean(bool_)
    assert is_float(bfloat16)
    assert not is_float(uint32)
    assert is_unrestricted_float(bfloat16)
    assert not is_unrestricted_float(tfloat32)
    assert is_float(tfloat32)
    assert is_restricted_float(tfloat32)
    assert is_restricted_float(float8_e4m3fn)
    assert is_restricted_float(float8_e5m2)
    assert is_restricted_float(float8_e8m0fnu)
    assert is_restricted_float(float4_e2m1fn)

    # Pickle-unpickle roundtrip
    assert pickle.loads(pickle.dumps(float16)) is float16

    # Deep copy roundtrip
    assert copy.deepcopy(float16) is float16


def test_tuple_type():
    # tuple type
    shape = TupleTy((5, 4))
    assert len(shape) == 2
    assert shape[0] == 5
    assert shape[1] == 4


def test_tile_type():
    # tile type
    shape = (5, 4)
    tile = TileTy(float16, shape)
    assert tile.dtype == float16
    assert tile.shape == shape

    # tile type equality
    tile2 = TileTy(float16, shape)
    assert tile == tile2


def test_array_type():
    # array with dynamic shape
    arr = ArrayTy(bfloat16, shape=(None, None), strides=(None, None))
    assert arr.dtype == bfloat16
    assert len(arr.shape) == 2
    assert len(arr.strides) == 2
    assert arr.shape[0] is None
    assert arr.strides[0] is None


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
    assert_no_promote(bool_, float8_e8m0fnu)
    assert_no_promote(bool_, float4_e2m1fn)

    # Int
    assert promote_dtypes(int8, int16) == int16
    assert promote_dtypes(int32, float16) == float16
    assert promote_dtypes(int64, bfloat16) == bfloat16
    assert_no_promote(int32, tfloat32)
    assert_no_promote(int32, float8_e5m2)
    assert_no_promote(int32, float8_e8m0fnu)
    assert_no_promote(int32, float4_e2m1fn)

    # Uint
    assert promote_dtypes(uint8, uint16) == uint16
    assert promote_dtypes(uint32, float16) == float16
    assert promote_dtypes(uint32, bfloat16) == bfloat16
    assert_no_promote(uint32, int32)
    assert_no_promote(uint32, int64)
    assert_no_promote(uint32, tfloat32)
    assert_no_promote(uint32, float8_e5m2)
    assert_no_promote(uint32, float8_e8m0fnu)
    assert_no_promote(uint32, float4_e2m1fn)

    # float
    assert promote_dtypes(float16, float32) == float32
    assert promote_dtypes(bfloat16, float32) == float32
    assert_no_promote(float16, bfloat16)
    assert_no_promote(float16, tfloat32)
    assert_no_promote(float16, float8_e5m2)
    assert_no_promote(float16, float8_e8m0fnu)
    assert_no_promote(float16, float4_e2m1fn)

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
            src = TileTy(src)
        check_implicit_cast(src, dst)

    def disallow(src, dst):
        if isinstance(src, DType):
            src = TileTy(src)
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

    disallow(bool_, float8_e8m0fnu)
    disallow(float8_e8m0fnu, bool_)

    disallow(bool_, float4_e2m1fn)
    disallow(float4_e2m1fn, bool_)

    # int -> float
    allow(uint32, float16)
    disallow(float16, uint32)

    allow(int32, float16)
    disallow(float16, int32)

    disallow(uint32, tfloat32)
    disallow(tfloat32, uint32)

    disallow(uint32, float8_e5m2)
    disallow(float8_e5m2, uint32)

    disallow(uint32, float8_e8m0fnu)
    disallow(float8_e8m0fnu, uint32)

    disallow(uint32, float4_e2m1fn)
    disallow(float4_e2m1fn, uint32)

    disallow(int32, tfloat32)
    disallow(tfloat32, int32)

    disallow(int32, float8_e5m2)
    disallow(float8_e5m2, int32)

    disallow(int32, float8_e8m0fnu)
    disallow(float8_e8m0fnu, int32)

    disallow(int32, float4_e2m1fn)
    disallow(float4_e2m1fn, int32)

    # signed <> unsigned not allowed
    disallow(int32, uint32)
    disallow(uint32, int32)

    # restricted float not allowed
    disallow(float32, tfloat32)
    disallow(tfloat32, float32)

    disallow(float8_e5m2, tfloat32)
    disallow(tfloat32, float8_e5m2)

    disallow(float8_e8m0fnu, tfloat32)
    disallow(tfloat32, float8_e8m0fnu)

    disallow(float4_e2m1fn, tfloat32)
    disallow(tfloat32, float4_e2m1fn)

    disallow(float8_e8m0fnu, float8_e5m2)
    disallow(float8_e5m2, float8_e8m0fnu)

    disallow(float4_e2m1fn, float8_e5m2)
    disallow(float8_e5m2, float4_e2m1fn)

    disallow(float4_e2m1fn, float8_e8m0fnu)
    disallow(float8_e8m0fnu, float4_e2m1fn)

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
    assert to_dtype(torch.float8_e8m0fnu) == float8_e8m0fnu


def test_type_of_constant_python_value():
    from cuda.tile._ir.typing_support import type_of_constant_python_value
    from cuda.tile._compile import _TileTypingHooks
    tp = functools.partial(type_of_constant_python_value, typing_hooks=_TileTypingHooks())
    assert tp(1) == TileTy(int32)
    assert tp(1.) == TileTy(float32)
    assert tp(True) == TileTy(bool_)
    assert tp(None) == NONE


def test_integer_info_equality():
    assert IntegerInfo(int32) == IntegerInfo(int32)
    assert IntegerInfo(int32) != IntegerInfo(int64)


def test_pointer_info_equality():
    dtypes = [
        opaque_pointer_dtype(),
        opaque_pointer_dtype(MemorySpace.SHARED),
        pointer_dtype(int32),
        pointer_dtype(int32, MemorySpace.SHARED),
        pointer_dtype(int64),
        pointer_dtype(int64, MemorySpace.SHARED)
    ]

    for i, a in enumerate(dtypes):
        for j, b in enumerate(dtypes):
            assert (PointerInfo(a) == PointerInfo(b)) == (i == j)
