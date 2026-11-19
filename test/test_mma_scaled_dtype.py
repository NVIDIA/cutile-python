# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from cuda.tile._datatype import (
    float4_e2m1fn, float8_e4m3fn, float8_e5m2, float8_e8m0fnu,
    float8_e5m3fnu, float16, float32, _resolve_mma_scaled_supported_dtype,
)
from cuda.tile._exception import TileTypeError
from cuda.tile._ir.ops import _verify_scaling_block_size
from cuda.tile._ir.type import TileTy


@pytest.mark.parametrize("x_dtype,y_dtype", [
    (float8_e4m3fn, float4_e2m1fn),
    (float4_e2m1fn, float8_e4m3fn),
    (float8_e5m2, float4_e2m1fn),
    (float4_e2m1fn, float8_e5m2),
], ids=str)
def test_mma_scaled_mixed_f8_f4(x_dtype, y_dtype):
    acc_dtype, block_sizes = _resolve_mma_scaled_supported_dtype(
        x_dtype, float8_e8m0fnu, y_dtype, float8_e8m0fnu, float32)
    assert acc_dtype == float32
    assert block_sizes == (32,)


@pytest.mark.parametrize("scale_dtype", [
    float8_e4m3fn,
    float8_e5m3fnu,
], ids=str)
def test_mma_scaled_mixed_f8_f4_rejects_unsupported_scale(scale_dtype):
    with pytest.raises(TileTypeError, match="Unsupported scale dtype"):
        _resolve_mma_scaled_supported_dtype(
            float8_e4m3fn, scale_dtype, float4_e2m1fn, scale_dtype, float32)


def test_mma_scaled_mixed_f8_f4_rejects_mismatched_scales():
    with pytest.raises(TileTypeError, match="x_scale and y_scale must have the same dtype"):
        _resolve_mma_scaled_supported_dtype(
            float8_e4m3fn, float8_e8m0fnu, float4_e2m1fn, float8_e4m3fn, float32)


def test_mma_scaled_mixed_f8_f4_rejects_unsupported_acc():
    with pytest.raises(TileTypeError, match="Unsupported acc dtype"):
        _resolve_mma_scaled_supported_dtype(
            float8_e4m3fn, float8_e8m0fnu, float4_e2m1fn, float8_e8m0fnu, float16)


def test_mma_scaled_same_width_f8_mix_still_rejected():
    with pytest.raises(TileTypeError, match="x and y must have the same dtype"):
        _resolve_mma_scaled_supported_dtype(
            float8_e4m3fn, float8_e8m0fnu, float8_e5m2, float8_e8m0fnu, float32)


def test_mma_scaled_mixed_f8_f4_requires_block_size_32():
    x = TileTy(float8_e4m3fn, (16, 64))
    x_scale = TileTy(float8_e8m0fnu, (16, 2))
    y = TileTy(float4_e2m1fn, (64, 16))
    y_scale = TileTy(float8_e8m0fnu, (2, 16))

    _verify_scaling_block_size(
        x, x_scale, k_axis=-1, name="x", scale_name="x_scale", allowed=(32,))
    _verify_scaling_block_size(
        y, y_scale, k_axis=-2, name="y", scale_name="y_scale", allowed=(32,))

    x_scale_16 = TileTy(float8_e8m0fnu, (16, 4))
    y_scale_16 = TileTy(float8_e8m0fnu, (4, 16))
    with pytest.raises(TileTypeError, match="scaling block size B = K // K_s"):
        _verify_scaling_block_size(
            x, x_scale_16, k_axis=-1, name="x", scale_name="x_scale", allowed=(32,))
    with pytest.raises(TileTypeError, match="scaling block size B = K // K_s"):
        _verify_scaling_block_size(
            y, y_scale_16, k_axis=-2, name="y", scale_name="y_scale", allowed=(32,))


def test_mma_scaled_same_dtype_f4_still_allows_block_size_16():
    _, block_sizes = _resolve_mma_scaled_supported_dtype(
        float4_e2m1fn, float8_e8m0fnu,
        float4_e2m1fn, float8_e8m0fnu,
        float32)
    assert block_sizes == (16, 32)

    x = TileTy(float4_e2m1fn, (16, 64))
    x_scale = TileTy(float8_e8m0fnu, (16, 4))
    _verify_scaling_block_size(
        x, x_scale, k_axis=-1, name="x", scale_name="x_scale", allowed=block_sizes)
