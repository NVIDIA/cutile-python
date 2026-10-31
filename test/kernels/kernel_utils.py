# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import torch
import cuda.tile as ct
from cuda.tile._cext import dev_features_enabled
from cuda.tile._compile import _get_max_supported_bytecode_version
from cuda.tile._cext import encode_float, decode_float
from functools import cache
import tempfile


@cache
def get_tileiras_version():
    return _get_max_supported_bytecode_version(tempfile.gettempdir(),
                                               allow_dev=dev_features_enabled())


def swizzle_2d_from_bid(M, N, tm, tn, GROUP_SIZE_M, bid):
    # Get the global IDs of a given block in a 1D grid.
    num_bid_m = ct.cdiv(M, tm)
    num_bid_n = ct.cdiv(N, tn)
    num_bid_in_group = GROUP_SIZE_M * num_bid_n
    group_id = bid // num_bid_in_group
    first_bid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_bid_m - first_bid_m, GROUP_SIZE_M)
    bid_m = first_bid_m + (bid % group_size_m)
    bid_n = (bid % num_bid_in_group) // group_size_m
    return bid_m, bid_n


def swizzle_2d(M, N, tm, tn, GROUP_SIZE_M):
    # Get the global IDs of the current block in a 1D grid.
    bid = ct.bid(0)
    return swizzle_2d_from_bid(M, N, tm, tn, GROUP_SIZE_M, bid)


def swizzle_32_4_4(scale):
    '''
    Prepare the original scale tensor to align with the expected tmem layout.
    With the innermost dimensions being (m1=32, m2=4, k1=4), and the outer dimensions
    being (m0=(M // (m1 * m2)), k0=(K_s // k1)).

    Reference: PTX ISA tcgen05.mma scale factor layout.
    https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-mma-scale-factor-a-layout-1x
    '''
    m1, m2, k1 = 32, 4, 4

    M, K_s = scale.shape
    m0 = M // (m1 * m2)
    k0 = K_s // k1
    scale = scale.reshape(m0, m2, m1, k0, k1).permute(0, 3, 2, 1, 4).contiguous()
    return scale.reshape(m0, k0, 32, 16)


def unswizzle_32_4_4(tile_swizzled_scale):
    '''
    Kernel-side inverse of ``swizzle_32_4_4``: take a tile loaded
    from the host swizzled scale tensor and recover the ``(M, K_s)``
    view that ``ct.mma_scaled`` expects.
    '''
    m1, m2, k1 = 32, 4, 4
    m0, k0, _, _ = tile_swizzled_scale.shape

    return (tile_swizzled_scale.reshape((m0, k0, m1, m2, k1))
                               .permute((0, 3, 2, 1, 4))
                               .reshape((m0 * m2 * m1, k0 * k1)))


def bits_to_torch(x: torch.Tensor, dtype: ct.DType):
    x_out = [decode_float(int(i), dtype) for i in x.view(-1)]
    x_out = torch.tensor(x_out, dtype=torch.float32, device=x.device).reshape(x.shape)
    return x_out


def torch_to_bits(x: torch.Tensor, dtype: ct.DType):
    x_out = [encode_float(i, dtype) for i in x.view(-1)]
    x_out = torch.tensor(x_out, dtype=torch.uint8, device=x.device).reshape(x.shape)
    return x_out


def _pad_tensor(x: torch.Tensor, block_size: int):
    if not x.dim():
        raise ValueError('The input tensor must at least have 1 dimension')

    if x.shape[-1] % block_size != 0:
        print('[WARNING] the innermost dimension of the input tensor')
        print('          is not a multiple of block_size padding the input tensor...')
        pad_len = (block_size - (x.shape[-1] % block_size)) % block_size
        x = torch.nn.functional.pad(x, (0, pad_len), value=0)
    return x


def block_quantize_f8e4m3fn_f8e8m0fnu(x: torch.Tensor, block_size: int):
    F8E4M3FN_MAX = 448
    x = _pad_tensor(x, block_size)

    x_block = x.reshape(*x.shape[:-1], x.shape[-1] // block_size, block_size)
    block_amax = x_block.abs().amax(dim=-1, keepdim=True)
    block_scale = block_amax / F8E4M3FN_MAX
    block_scale = torch.pow(2.0, torch.ceil(torch.log2(block_scale)))
    block_scale_fp8 = block_scale.to(torch.float8_e8m0fnu)
    effective_block_scale = block_scale_fp8.to(x.dtype)

    x_block_scaled = x_block / torch.where(effective_block_scale == 0,
                                           torch.ones_like(effective_block_scale),
                                           effective_block_scale)
    x_block_scaled = x_block_scaled.to(torch.float8_e4m3fn).reshape(x.shape)
    block_scale_fp8 = block_scale_fp8.squeeze(-1)
    return x_block_scaled, block_scale_fp8


def block_quantize_f4e2m1fn_f8e4m3fn(x: torch.Tensor, block_size: int):
    F4E2M1FN_MAX = 6
    F8E4M3FN_MAX = 448
    x = _pad_tensor(x, block_size)

    x_block = x.reshape(*x.shape[:-1], x.shape[-1] // block_size, block_size)
    block_amax = x_block.abs().amax(dim=-1, keepdim=True)
    global_amax = x.abs().amax()
    global_scale = global_amax / (F4E2M1FN_MAX * F8E4M3FN_MAX)
    block_scale = block_amax / (F4E2M1FN_MAX * torch.where(global_scale == 0,
                                                           torch.ones_like(global_scale),
                                                           global_scale))

    block_scale_fp8 = block_scale.to(torch.float8_e4m3fn)
    effective_block_scale = block_scale_fp8.to(x.dtype)
    effective_block_scale *= global_scale
    x_block_scaled = x_block / torch.where(effective_block_scale == 0,
                                           torch.ones_like(effective_block_scale),
                                           effective_block_scale)

    x_block_scaled_fp4_bits_packed = pack_e2m1(x_block_scaled)
    x_block_scaled = x_block_scaled_fp4_bits_packed.reshape(*x.shape[:-1], x.shape[-1] // 2)
    block_scale_fp8 = block_scale_fp8.squeeze(-1)
    return x_block_scaled, block_scale_fp8, global_scale.view((1,))


def pack_e2m1(x: torch.Tensor) -> torch.Tensor:
    """
    Input shape:  (..., K)
    Output shape: (..., K // 2)
    """
    assert x.dtype == torch.float32
    assert x.shape[-1] % 2 == 0

    mag = x.abs()

    lookup = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
    midpoints = ((i + j) / 2 for i, j in zip(lookup[:-1], lookup[1:]))

    # the alternating > and >= comparisons select the even encoding
    # when a value lies exactly halfway between two values (round to nearest even encoding)
    encoding = 0
    for index, e in enumerate(midpoints):
        if index % 2:
            encoding += (mag >= e).to(torch.uint8)
        else:
            encoding += (mag > e).to(torch.uint8)

    # add the E2M1 sign bit
    encoding |= torch.signbit(x).to(torch.uint8) << 3

    # pack two E2M1 values into each byte
    packed = encoding[..., 0::2] | (encoding[..., 1::2] << 4)
    return packed.contiguous().view(torch.float4_e2m1fn_x2)


def unpack_e2m1_bytes_to_float(fp4_bytes: torch.Tensor) -> torch.Tensor:
    lookup = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
        dtype=torch.float32,
        device=fp4_bytes.device,
    )
    low = fp4_bytes & 0x0F
    high = (fp4_bytes >> 4) & 0x0F
    low_values = lookup[(low & 0x7).long()]
    high_values = lookup[(high & 0x7).long()]
    low_values = torch.where((low & 0x8) != 0, -low_values, low_values)
    high_values = torch.where((high & 0x8) != 0, -high_values, high_values)
    values = torch.empty(
        (*fp4_bytes.shape[:-1], fp4_bytes.shape[-1] * 2),
        dtype=torch.float32,
        device=fp4_bytes.device,
    )
    values[..., 0::2] = low_values
    values[..., 1::2] = high_values
    return values
