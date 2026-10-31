# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import cuda.tile as ct

from kernels.kernel_utils import swizzle_2d, unswizzle_32_4_4

ConstInt = ct.Constant[int]


@ct.function
def quantize_mxfp8_fn(tx, x_dim: ConstInt, y_dim: ConstInt,
                      block_size: ConstInt):
    F8E4M3FN_MAX = 448
    y_dim_block = y_dim // block_size
    block = tx.reshape(shape=(x_dim, y_dim_block, block_size))
    block_max = ct.max(ct.abs(block), axis=-1, keepdims=True)
    block_scale = block_max / F8E4M3FN_MAX
    block_scale_fp8 = block_scale.astype(ct.float8_e8m0fnu, rounding_mode=ct.RoundingMode.RP)
    effective_block_scale = block_scale_fp8.astype(tx.dtype)
    block_scaled = block / ct.where(effective_block_scale == 0,
                                    ct.ones(shape=effective_block_scale.shape, dtype=tx.dtype),
                                    effective_block_scale)

    block_scaled = block_scaled.astype(ct.float8_e4m3fn).reshape(shape=(x_dim, y_dim))
    block_scale_fp8 = block_scale_fp8.reshape(shape=(x_dim, y_dim_block))

    m1, m2, k1 = 32, 4, 4
    M, K_s = x_dim, y_dim_block
    m0 = M // (m1 * m2)
    k0 = K_s // k1
    block_scale_fp8 = block_scale_fp8.reshape(shape=(m0, m2, m1, k0, k1)).permute((0, 3, 2, 1, 4))
    block_scale_fp8 = block_scale_fp8.reshape(shape=(m0, k0, 32, 16))
    return block_scaled, block_scale_fp8


@ct.kernel
def mxfp8_quantize_kernel(x, x_q, x_s, x_dim: ConstInt, y_dim: ConstInt,
                          block_size: ConstInt):
    bid_x = ct.bid(0)
    bid_y = ct.bid(1)
    tx = ct.load(x, index=(bid_x, bid_y), shape=(x_dim, y_dim), padding_mode=ct.PaddingMode.ZERO)

    block_scaled, block_scale_fp8 = quantize_mxfp8_fn(tx, x_dim, y_dim, block_size)
    ct.store(x_q, index=(bid_x, bid_y), tile=block_scaled)
    ct.store(x_s, index=(bid_x, bid_y, 0, 0), tile=block_scale_fp8)


@ct.function
def quantize_nvfp4_fn(global_scale, tx, x_dim: ConstInt, y_dim: ConstInt,
                      block_size: ConstInt):
    F4E2M1FN_MAX = 6
    y_dim_block = y_dim // block_size
    tx_block = tx.reshape(shape=(x_dim, y_dim // block_size, block_size))
    block_amax = ct.max(ct.abs(tx_block), axis=-1, keepdims=True)
    block_scale = block_amax / (F4E2M1FN_MAX * ct.where(global_scale == 0,
                                                        ct.ones(shape=global_scale.shape,
                                                                dtype=global_scale.dtype),
                                                        global_scale))

    block_scale_fp8 = block_scale.astype(ct.float8_e4m3fn)
    effective_block_scale = block_scale_fp8.astype(tx.dtype)
    effective_block_scale *= global_scale
    tx_block_scaled = tx_block / ct.where(effective_block_scale == 0,
                                          ct.ones(shape=effective_block_scale.shape,
                                                  dtype=effective_block_scale.dtype),
                                          effective_block_scale)

    tx_block_scaled_fp4_bits = tx_block_scaled.astype(ct.float4_e2m1fn)
    tx_block_scaled_fp4_bits_packed = ct.pack_to_bytes(tx_block_scaled_fp4_bits)
    tx_block_scaled = tx_block_scaled_fp4_bits_packed.reshape(shape=(x_dim, y_dim // 2))
    block_scale_fp8 = block_scale_fp8.reshape(shape=(x_dim, y_dim_block))

    m1, m2, k1 = 32, 4, 4
    m0 = x_dim // (m1 * m2)
    k0 = y_dim_block // k1
    block_scale_fp8 = block_scale_fp8.reshape(shape=(m0, m2, m1, k0, k1)).permute((0, 3, 2, 1, 4))
    block_scale_fp8 = block_scale_fp8.reshape(shape=(m0, k0, 32, 16))
    return tx_block_scaled, block_scale_fp8


@ct.kernel
def nvfp4_quantize_kernel(x, x_q, x_s, g_scale, gmax_bits, x_dim: ConstInt,
                          y_dim: ConstInt, block_size: ConstInt, done, vbid):
    num_tiles_m = ct.num_tiles(x, axis=0, shape=(x_dim, y_dim))
    num_tiles_k = ct.num_tiles(x, axis=1, shape=(x_dim, y_dim))
    num_tile_blocks = num_tiles_m * num_tiles_k
    virtual_bid = ct.atomic_add(vbid, 0, 1, memory_order=ct.MemoryOrder.RELAXED)

    while virtual_bid < num_tile_blocks:
        vbid_x, vbid_y = virtual_bid // num_tiles_k, virtual_bid % num_tiles_k
        tx = ct.load(x, index=(vbid_x, vbid_y), shape=(x_dim, y_dim),
                     padding_mode=ct.PaddingMode.ZERO)
        vblock_max_bits = ct.bitcast(ct.max(ct.abs(tx)), ct.uint32)
        ct.atomic_max(gmax_bits, 0, vblock_max_bits, memory_order=ct.MemoryOrder.RELAXED)
        ct.atomic_add(done, 0, 1, memory_order=ct.MemoryOrder.RELEASE)
        virtual_bid = ct.atomic_add(vbid, 0, 1, memory_order=ct.MemoryOrder.RELAXED)

    while ct.atomic_add(done, 0, 0, memory_order=ct.MemoryOrder.ACQUIRE) < num_tile_blocks:
        pass

    F4E2M1FN_MAX = 6
    F8E4M3FN_MAX = 448
    bid_x = ct.bid(0)
    bid_y = ct.bid(1)
    tx = ct.load(x, index=(bid_x, bid_y), shape=(x_dim, y_dim), padding_mode=ct.PaddingMode.ZERO)

    global_amax_bits = ct.load(gmax_bits, 0, ())
    global_amax = ct.bitcast(global_amax_bits, ct.float32)
    global_scale = global_amax / (F4E2M1FN_MAX * F8E4M3FN_MAX)

    tx_block_scaled, block_scale_fp8 = quantize_nvfp4_fn(global_scale, tx, x_dim, y_dim, block_size)
    ct.store(x_q, index=(bid_x, bid_y), tile=tx_block_scaled)
    ct.store(x_s, index=(bid_x, bid_y, 0, 0), tile=block_scale_fp8)

    if bid_x == bid_y == 0:
        ct.store(g_scale, index=(0,), tile=global_scale)


@ct.function
def grid_sync_helper(a, b,
                     a_x_dim: ConstInt, b_x_dim: ConstInt, y_dim: ConstInt,
                     vbid, done, func, func_args_1, func_args_2):
    num_tiles_m = ct.num_tiles(a, axis=0, shape=(a_x_dim, y_dim))
    num_tiles_k = ct.num_tiles(a, axis=1, shape=(a_x_dim, y_dim))
    num_tiles_n = ct.num_tiles(b, axis=0, shape=(b_x_dim, y_dim))

    a_num_tile_blocks = num_tiles_m * num_tiles_k
    b_num_tile_blocks = num_tiles_n * num_tiles_k
    num_tile_blocks = a_num_tile_blocks + b_num_tile_blocks
    virtual_bid = ct.atomic_add(vbid, 0, 1, memory_order=ct.MemoryOrder.RELAXED)

    while virtual_bid < num_tile_blocks:

        if virtual_bid < a_num_tile_blocks:
            func(virtual_bid, *func_args_1)
        else:
            func(virtual_bid - a_num_tile_blocks, *func_args_2)

        ct.atomic_add(done, 0, 1, memory_order=ct.MemoryOrder.RELEASE)
        virtual_bid = ct.atomic_add(vbid, 0, 1, memory_order=ct.MemoryOrder.RELAXED)

    while ct.atomic_add(done, 0, 0, memory_order=ct.MemoryOrder.ACQUIRE) < num_tile_blocks:
        pass


@ct.function
def quantize_mxfp8_fn_helper(virtual_bid, x, x_q, x_s,
                             x_dim: ConstInt, y_dim: ConstInt, block_size: ConstInt, num_tiles_k):
    vbid_x, vbid_y = virtual_bid // num_tiles_k, virtual_bid % num_tiles_k
    tx = ct.load(x, index=(vbid_x, vbid_y), shape=(x_dim, y_dim), padding_mode=ct.PaddingMode.ZERO)
    tx_q, tx_s = quantize_mxfp8_fn(tx, x_dim, y_dim, block_size)

    ct.store(x_q, index=(vbid_x, vbid_y), tile=tx_q)
    ct.store(x_s, index=(vbid_x, vbid_y, 0, 0), tile=tx_s)


@ct.kernel
def qunatize_mxfp8_matmul_kernel(A, A_q, A_s, B, B_q, B_s, C, vbid, done,
                                 tm: ConstInt, tn: ConstInt, tk: ConstInt,
                                 tm_q: ConstInt, tn_q: ConstInt, tk_q: ConstInt,
                                 block_size: ConstInt):
    GROUP_SIZE_M = 8
    M = A.shape[0]
    N = B.shape[0]
    bidx, bidy = swizzle_2d(M, N, tm, tn, GROUP_SIZE_M)
    tks = tk // block_size
    num_tiles_k = ct.num_tiles(A, axis=1, shape=(tm, tk))
    num_q_tiles_k = ct.num_tiles(A, axis=1, shape=(tm_q, tk_q))

    accumulator = ct.full((tm, tn), 0, dtype=ct.float32)
    zero_pad = ct.PaddingMode.ZERO

    grid_sync_helper(
        A, B, tm_q, tn_q, tk_q, vbid, done,
        quantize_mxfp8_fn_helper,
        (A, A_q, A_s, tm_q, tk_q, block_size, num_q_tiles_k),
        (B, B_q, B_s, tn_q, tk_q, block_size, num_q_tiles_k)
    )

    for k in range(num_tiles_k):
        a = ct.load(A_q, index=(bidx, k), shape=(tm, tk), padding_mode=zero_pad)
        a_scale = ct.load(
            A_s, index=(bidx, k, 0, 0),
            shape=(tm // 32 // 4, tks // 4, 32, 16), padding_mode=zero_pad
        )
        a_scale = unswizzle_32_4_4(a_scale)

        b = ct.load(B_q, index=(bidy, k), shape=(tn, tk), padding_mode=zero_pad)
        b = b.permute((1, 0))
        b_scale = ct.load(
            B_s, index=(bidy, k, 0, 0),
            shape=(tn // 32 // 4, tks // 4, 32, 16), padding_mode=zero_pad
        )
        b_scale = unswizzle_32_4_4(b_scale)
        b_scale = b_scale.permute((1, 0))

        accumulator = ct.mma_scaled(a, a_scale, b, b_scale, accumulator)

    ct.store(C, index=(bidx, bidy), tile=accumulator)


@ct.function
def nvfp4_gmax(virtual_bid, gmax_bits, x, x_dim: ConstInt, y_dim: ConstInt, num_tiles_k):
    vbid_x, vbid_y = virtual_bid // num_tiles_k, virtual_bid % num_tiles_k
    tx = ct.load(x, index=(vbid_x, vbid_y), shape=(x_dim, y_dim), padding_mode=ct.PaddingMode.ZERO)
    vblock_max_bits = ct.bitcast(ct.max(ct.abs(tx)), ct.uint32)
    ct.atomic_max(gmax_bits, 0, vblock_max_bits, memory_order=ct.MemoryOrder.RELAXED)


@ct.function
def quantize_nvfp4_fn_helper(virtual_bid, global_scale, x, x_q, x_s,
                             x_dim: ConstInt, y_dim: ConstInt, block_size: ConstInt, num_tiles_k):
    vbid_x, vbid_y = virtual_bid // num_tiles_k, virtual_bid % num_tiles_k
    tx = ct.load(x, index=(vbid_x, vbid_y), shape=(x_dim, y_dim), padding_mode=ct.PaddingMode.ZERO)
    tx_q, tx_s = quantize_nvfp4_fn(global_scale, tx, x_dim, y_dim, block_size)

    ct.store(x_q, index=(vbid_x, vbid_y), tile=tx_q)
    ct.store(x_s, index=(vbid_x, vbid_y, 0, 0), tile=tx_s)


@ct.kernel
def qunatize_nvfp4_matmul_kernel(A, A_q, A_s, A_gmax_bits,
                                 B, B_q, B_s, B_gmax_bits,
                                 C,
                                 vbid_1, done_1, vbid_2, done_2,
                                 tm: ConstInt, tn: ConstInt, tk: ConstInt,
                                 tm_q: ConstInt, tn_q: ConstInt, tk_q: ConstInt,
                                 block_size: ConstInt):
    GROUP_SIZE_M = 8
    M = A.shape[0]
    N = B.shape[0]
    F4E2M1FN_MAX = 6
    F8E4M3FN_MAX = 448
    bidx, bidy = swizzle_2d(M, N, tm, tn, GROUP_SIZE_M)
    tks = tk // block_size
    num_tiles_k = ct.num_tiles(A, axis=1, shape=(tm, tk))
    num_q_tiles_k = ct.num_tiles(A, axis=1, shape=(tm_q, tk_q))

    accumulator = ct.full((tm, tn), 0, dtype=ct.float32)
    zero_pad = ct.PaddingMode.ZERO

    grid_sync_helper(
        A, B, tm_q, tn_q, tk_q, vbid_1, done_1, nvfp4_gmax,
        (A_gmax_bits, A, tm_q, tk_q, num_q_tiles_k),
        (B_gmax_bits, B, tn_q, tk_q, num_q_tiles_k)
    )

    A_global_amax_bits = ct.load(A_gmax_bits, 0, ())
    A_global_amax = ct.bitcast(A_global_amax_bits, ct.float32)
    A_global_scale = A_global_amax / (F4E2M1FN_MAX * F8E4M3FN_MAX)

    B_global_amax_bits = ct.load(B_gmax_bits, 0, ())
    B_global_amax = ct.bitcast(B_global_amax_bits, ct.float32)
    B_global_scale = B_global_amax / (F4E2M1FN_MAX * F8E4M3FN_MAX)

    grid_sync_helper(
        A, B, tm_q, tn_q, tk_q, vbid_2, done_2, quantize_nvfp4_fn_helper,
        (A_global_scale, A, A_q, A_s, tm_q, tk_q, block_size, num_q_tiles_k),
        (B_global_scale, B, B_q, B_s, tn_q, tk_q, block_size, num_q_tiles_k)
    )

    for k in range(num_tiles_k):
        a = ct.load(A_q, index=(bidx, k), shape=(tm, tk // 2), padding_mode=zero_pad)
        a = ct.unpack_from_bytes(a.reshape((-1,)), ct.float4_e2m1fn).reshape((tm, tk))
        a_scale = ct.load(
            A_s, index=(bidx, k, 0, 0),
            shape=(tm // 32 // 4, tks // 4, 32, 16), padding_mode=zero_pad
        )
        a_scale = unswizzle_32_4_4(a_scale)

        b = ct.load(B_q, index=(bidy, k), shape=(tn, tk // 2), padding_mode=zero_pad)
        b = ct.unpack_from_bytes(b.reshape((-1,)), ct.float4_e2m1fn).reshape((tn, tk))
        b = b.permute((1, 0))
        b_scale = ct.load(
            B_s, index=(bidy, k, 0, 0),
            shape=(tn // 32 // 4, tks // 4, 32, 16), padding_mode=zero_pad
        )
        b_scale = unswizzle_32_4_4(b_scale)
        b_scale = b_scale.permute((1, 0))

        accumulator = ct.mma_scaled(a, a_scale, b, b_scale, accumulator)

    accumulator = accumulator * A_global_scale * B_global_scale
    ct.store(C, index=(bidx, bidy), tile=accumulator)
