# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Device address helpers shared by decode resources."""

import cuda.lang as cl

NEG_MAX = -3.4028234663852886e38


def smem_array(stage_info, offset, dtype, count):
    ptr = stage_info.context.smem_base.pointer() + offset
    ptr = cl.bitcast(ptr, cl.pointer_dtype(dtype, ptr.memory_space))
    return cl.Array.from_parts(ptr, (count,))


def tmem_ptr(stage_info, column, row=0):
    return cl.tcgen05_tmem_offset(
        stage_info.context.tasks_inputs.tmem_base,
        column_offset=column,
        lane_offset=row,
    )


def smem_descriptor(ptr, leading):
    return cl.int64(
        cl.Tcgen05SharedMemoryDescriptor(
            matrix_start_address=ptr,
            leading_dimension_byte_offset=leading,
            stride_dimension_byte_offset=1024,
            swizzle_mode=cl.SwizzleMode.SWIZZLE_128B,
        ).encode()
    )


def warp_and_lane(stage_info):
    return stage_info.context.warp_index % 4, cl.thread_index(0) % 32


def work_coords(stage_info):
    tile = stage_info.work_tile.tile_idx
    return tile[0], tile[1], tile[2]


def q_group_token_base(group, cfg):
    if cfg.groups_tokens_heads_q:
        return group * cfg.q_tokens_per_cta
    return group // ((cfg.heads_q_per_kv + 7) // 8)


def q_row_token_and_local_head(group, row, cfg):
    if cfg.groups_tokens_heads_q:
        return group * cfg.q_tokens_per_cta + row // cfg.heads_q_per_kv, row % cfg.heads_q_per_kv
    head_ctas = (cfg.heads_q_per_kv + 7) // 8
    return group // head_ctas, (group % head_ctas) * 8 + row


def kv_tile_bounds(length, seq_len_q, group, cfg):
    """Union of the causal/window spans for the live Q rows in one CTA."""
    first = cl.int32(0)
    last = length
    if cfg.use_variable_seqlens_q:
        q_begin = q_group_token_base(group, cfg)
        if cfg.mask_type == "causal":
            q_end = cl.minimum(q_begin + cfg.q_tokens_per_cta, seq_len_q)
            last = cl.minimum(cl.maximum(length - seq_len_q + q_end, 0), length)
        if cfg.window_left >= 0:
            first = cl.maximum(length - seq_len_q + q_begin - cfg.window_left, 0) >> 7
    elif cfg.window_left >= 0:
        first = cl.maximum(length - cfg.window_left - 1, 0) // 128
    if cfg.use_variable_seqlens_q:
        return first, cl.int32(cl.uint32(last + 127) >> 7)
    return first, (last + 127) // 128


def transform_ragged_coords(coords, box, extent):
    """Bound a token box using two synthetic TMA dimensions."""
    extent = cl.minimum(cl.maximum(extent, 0), box)
    shift = (box - extent) % box + box * cl.int32(extent == 0)
    return coords[0], coords[1], shift, cl.int32(1 << 30), coords[2] + (1 << 30) - shift


def kv_tile_idx(stage_info, inst_idx, section, is_v, cfg):
    if section == 0:
        tile = cl.int32(inst_idx)
    elif section == 1:
        tile = 2 * stage_info.loop_offset + inst_idx + (0 if is_v else 2)
    else:
        tile = 2 * (num_pairs(stage_info, cfg) - 1) + inst_idx
    inputs = stage_info.context.tasks_inputs
    start, _ = kv_tile_bounds(inputs.seq_len, inputs.seq_len_q, work_coords(stage_info)[0], cfg)
    return tile + start


def num_pairs(stage_info, cfg):
    inputs = stage_info.context.tasks_inputs
    start, end = kv_tile_bounds(inputs.seq_len, inputs.seq_len_q, work_coords(stage_info)[0], cfg)
    if cfg.use_variable_seqlens_q:
        return (end - start + 1) >> 1
    return (end - start + 1) // 2


def load_o(stage_info, column):
    warp, _ = warp_and_lane(stage_info)
    chunks = tuple(
        cl.tcgen05_load(
            cl.Tcgen05LoadStoreShape.SHAPE_16X256B,
            tmem_ptr(stage_info, column, warp * 32 + chunk * 16),
            element_count=4,
            dtype=cl.float32,
        )
        for chunk in cl.static_iter(range(2))
    )
    cl.tcgen05_wait_load()
    return cl.Vector(
        *tuple(chunks[i // 4][i % 4] for i in cl.static_iter(range(8))), dtype=cl.float32
    )
