# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import cuda.tile as ct
from cuda.tile._bytecode.version import BytecodeVersion
from util import assert_equal, make_test_tensor, require_hopper_or_newer, require_blackwell_or_newer
from cuda.tile._exception import TileTypeError
from conftest import float_dtypes, int_dtypes, requires_tileiras, uint_dtypes, dtype_id

# TODO: remove when feature is out of development only
from cuda.tile._datatype import float4_e2m1fn
from cuda.tile._stub import pack_to_bytes, unpack_from_bytes
ct.pack_to_bytes = pack_to_bytes
ct.unpack_from_bytes = unpack_from_bytes

pytestmark = requires_tileiras(BytecodeVersion.V_13_3)

float8_dtypes = [
    pytest.param(torch.float8_e5m2, marks=require_hopper_or_newer()),
    pytest.param(torch.float8_e4m3fn, marks=require_hopper_or_newer()),
    pytest.param(torch.float8_e8m0fnu, marks=require_blackwell_or_newer()),
]

test_dtypes = (float_dtypes + int_dtypes + uint_dtypes +
               [torch.float64] + float8_dtypes)


@ct.kernel
def pack_unpack_1d(x, y, TILE: ct.Constant[int]):
    tx = ct.load(x, index=(0,), shape=(TILE,))
    packed = ct.pack_to_bytes(tx)
    ty = ct.unpack_from_bytes(packed, y.dtype)
    ct.store(y, index=(0,), tile=ty)


@pytest.mark.parametrize("dtype", test_dtypes, ids=dtype_id)
def test_pack_to_bytes(dtype):
    @ct.kernel
    def kernel(x, y, TILE: ct.Constant[int]):
        tx = ct.load(x, index=(0,), shape=(TILE,))
        ty = ct.pack_to_bytes(tx)
        ct.store(y, index=(0,), tile=ty)

    tile = 128
    x = make_test_tensor((tile,), dtype=dtype, device='cuda')
    nbytes = tile * x.element_size()
    y = torch.zeros(nbytes, dtype=torch.uint8, device='cuda')
    ct.launch(torch.cuda.current_stream(), (1,), kernel, (x, y, tile))
    ref = x.view(torch.uint8)
    assert_equal(y, ref)


@pytest.mark.parametrize("dtype", test_dtypes, ids=dtype_id)
def test_unpack_from_bytes(dtype):
    @ct.kernel
    def kernel(x, y, TILE: ct.Constant[int]):
        tx = ct.load(x, index=(0,), shape=(TILE,))
        ty = ct.unpack_from_bytes(tx, y.dtype)
        ct.store(y, index=(0,), tile=ty)

    ref = make_test_tensor((32,), dtype=dtype, device='cuda')
    x = ref.view(torch.uint8)
    y = torch.zeros_like(ref)
    tile = x.shape[0]
    ct.launch(torch.cuda.current_stream(), (1,), kernel, (x, y, tile))
    assert_equal(y, ref)


@pytest.mark.parametrize("dtype", test_dtypes, ids=dtype_id)
def test_pack_unpack_roundtrip(dtype):
    tile = 128
    x = make_test_tensor((tile,), dtype=dtype, device='cuda')
    y = torch.zeros_like(x)
    ct.launch(torch.cuda.current_stream(), (1,), pack_unpack_1d, (x, y, tile))
    assert_equal(y, x)


@pytest.mark.parametrize("dtype", test_dtypes, ids=dtype_id)
def test_pack_unpack_roundtrip_0d(dtype):
    @ct.kernel
    def kernel(x, y):
        tx = ct.gather(x, ())
        packed = ct.pack_to_bytes(tx)
        ty = ct.unpack_from_bytes(packed, x.dtype)
        ty = ty.reshape(())
        ct.scatter(y, (), ty)

    x = make_test_tensor((), dtype=dtype, device='cuda')
    y = torch.zeros_like(x)
    ct.launch(torch.cuda.current_stream(), (1,), kernel, (x, y))
    assert_equal(y, x)


@pytest.mark.parametrize("dtype", test_dtypes, ids=dtype_id)
def test_pack_unpack_roundtrip_2d(dtype):
    @ct.kernel
    def kernel(x, y, TILE_M: ct.Constant[int], TILE_N: ct.Constant[int]):
        bidm = ct.bid(0)
        bidn = ct.bid(1)
        tx = ct.load(x, index=(bidm, bidn), shape=(TILE_M, TILE_N))
        packed = ct.pack_to_bytes(tx)
        ty = ct.unpack_from_bytes(packed, x.dtype)
        ty = ct.reshape(ty, (TILE_M, TILE_N))
        ct.store(y, index=(bidm, bidn), tile=ty)

    shape = (64, 128)
    tiles = (32, 64)
    x = make_test_tensor(shape, dtype=dtype, device='cuda')
    y = torch.zeros_like(x)
    grid = (ct.cdiv(shape[0], tiles[0]), ct.cdiv(shape[1], tiles[1]))
    ct.launch(torch.cuda.current_stream(), grid,
              kernel, (x, y, tiles[0], tiles[1]))
    assert_equal(y, x)


@pytest.mark.parametrize("dtype_x", test_dtypes, ids=dtype_id)
@pytest.mark.parametrize("dtype_y", test_dtypes, ids=dtype_id)
def test_cross_type_pack_unpack(dtype_x, dtype_y):
    tile = 128
    x = make_test_tensor((tile,), dtype=dtype_x, device='cuda')
    ref = x.view(torch.uint8).view(dtype_y)
    y = torch.zeros_like(ref)
    ct.launch(torch.cuda.current_stream(), (1,), pack_unpack_1d, (x, y, tile))
    assert_equal(y, ref)


@pytest.mark.parametrize("dtype", test_dtypes + [
    pytest.param(float4_e2m1fn, marks=require_blackwell_or_newer()),
])
def test_unpack_pack_roundtrip(dtype):
    @ct.kernel
    def kernel(x, y, TILE: ct.Constant[int]):
        tx = ct.load(x, index=(0,), shape=(TILE,))
        unpacked = ct.unpack_from_bytes(tx, dtype)
        packed = ct.pack_to_bytes(unpacked)
        ct.store(y, index=(0,), tile=packed)

    tile = 128
    x = torch.randint(0, 256, (tile,), dtype=torch.uint8, device='cuda')
    y = torch.zeros(tile, dtype=torch.uint8, device='cuda')
    ct.launch(torch.cuda.current_stream(), (1,), kernel, (x, y, tile))
    assert_equal(y, x)


def test_unpack_from_bytes_not_divisible():
    @ct.kernel
    def kernel(x, y):
        tx = ct.load(x, index=(0,), shape=(2,))
        ct.unpack_from_bytes(tx, y.dtype)

    x = torch.ones(2, dtype=torch.uint8, device='cuda')
    y = torch.zeros(1, dtype=torch.int32, device='cuda')
    with pytest.raises(TileTypeError, match="not divisible by 32"):
        ct.launch(torch.cuda.current_stream(), (1,), kernel, (x, y))


def test_unpack_from_bytes_wrong_input_dtype():
    @ct.kernel
    def kernel(x, y):
        tx = ct.load(x, index=(0,), shape=(4,))
        ct.unpack_from_bytes(tx, y.dtype)

    x = torch.ones(4, dtype=torch.int32, device='cuda')
    y = torch.zeros(4, dtype=torch.int32, device='cuda')
    with pytest.raises(TileTypeError, match="unpack_from_bytes requires uint8 tile"):
        ct.launch(torch.cuda.current_stream(), (1,), kernel, (x, y))


def test_unpack_from_bytes_not_1d():
    @ct.kernel
    def kernel(x, y):
        tx = ct.load(x, index=(0, 0), shape=(4, 4))
        ct.unpack_from_bytes(tx, y.dtype)

    x = torch.ones((4, 4), dtype=torch.uint8, device='cuda')
    y = torch.zeros(4, dtype=torch.int32, device='cuda')
    with pytest.raises(TileTypeError, match="unpack_from_bytes requires a 1D tile"):
        ct.launch(torch.cuda.current_stream(), (1,), kernel, (x, y))


def test_pack_to_bytes_bool():
    @ct.kernel
    def kernel(x, y, TILE: ct.Constant[int]):
        tx = ct.load(x, index=(0,), shape=(TILE,))
        ct.pack_to_bytes(tx)

    x = torch.ones(4, dtype=torch.bool, device='cuda')
    y = torch.zeros(4, dtype=torch.uint8, device='cuda')
    with pytest.raises(TileTypeError, match="pack_to_bytes from a bool_ tile"):
        ct.launch(torch.cuda.current_stream(), (1,), kernel, (x, y, 4))


def test_unpack_from_bytes_bool():
    @ct.kernel
    def kernel(x, y):
        tx = ct.load(x, index=(0,), shape=(4,))
        ct.unpack_from_bytes(tx, y.dtype)

    x = torch.ones(4, dtype=torch.uint8, device='cuda')
    y = torch.zeros(4, dtype=torch.bool, device='cuda')
    with pytest.raises(TileTypeError, match="unpack_from_bytes to a bool_ tile"):
        ct.launch(torch.cuda.current_stream(), (1,), kernel, (x, y))
