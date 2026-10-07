# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Swapped-operand QK/PV, online softmax, and the two-instance epilogue."""

from dataclasses import dataclass

import cuda.lang as cl
import task_scheduling as ts

from .helpers_common import (
    NEG_MAX,
    load_o,
    smem_array,
    smem_descriptor,
    tmem_ptr,
    warp_and_lane,
    work_coords,
    num_pairs,
    kv_tile_bounds,
    q_group_token_base,
    q_row_token_and_local_head,
)


def pair_vector(value):
    return cl.Vector(cl.float32(value), cl.float32(value))


def max_encode(value):
    bits = cl.bitcast(value, cl.int32)
    return cl.bitcast(bits ^ ((bits >> 31) | cl.int32(-2147483648)), cl.uint32)


def max_decode(value):
    bits = cl.bitcast(value, cl.int32)
    return cl.bitcast(bits ^ ((~(bits >> 31)) | cl.int32(-2147483648)), cl.float32)


def rescale(old, new):
    return (
        cl.float32(0)
        if old == NEG_MAX
        else cl.exp2((old - new) * (128**-0.5 * 1.4426950408889634), flush_to_zero=True)
    )


def safe_norm_rcp(value):
    """Clamp the denominator before taking an approximate reciprocal."""
    denominator = cl._nvvm.fmax_ftz_f(value, cl.float32(1.0e-12))
    return cl._inline_ptx("rcp.approx.f32 %0, %1;", cl.float32, denominator)[0]


def store_o(stage_info, column, values):
    warp, _ = warp_and_lane(stage_info)
    for chunk in cl.static_iter(range(2)):
        cl.tcgen05_store(
            cl.Tcgen05LoadStoreShape.SHAPE_16X256B,
            tmem_ptr(stage_info, column, warp * 32 + chunk * 16),
            cl.Vector(*tuple(values[chunk * 4 + i] for i in cl.static_iter(range(4)))),
        )
    cl.tcgen05_wait_store()


@dataclass(kw_only=True, eq=False)
class TmemSResource(ts.MemoryResource):
    @ts.producer_work
    @staticmethod
    def qk_mma(stage_info, q_desc, k_desc, column, cfg):
        instr = cl.Tcgen05InstructionDescriptor(
            d_type=cl.float32,
            a_type=cfg.q_dtype,
            b_type=cfg.q_dtype,
            m=128,
            n=8,
        ).encode()
        if cl.elect_sync():
            for ki in cl.static_iter(range(8)):
                cl.tcgen05_mma(
                    cl.Tcgen05MMAKind.F16,
                    tmem_ptr(stage_info, column),
                    k_desc + ki * 2 + (ki // 4) * 1016,
                    q_desc + ki * 2 + (ki // 4) * 56,
                    instr,
                    accumulate=ki > 0,
                    cta_group=cl.CTAGroup.CTA_1,
                )

    @ts.consumer_work(outputs=2, work_attrs=ts.WorkAttr.AUXILIARY)
    @staticmethod
    def init_state(stage_info):
        return pair_vector(NEG_MAX), pair_vector(0)

    @ts.consumer_work(outputs=1, work_attrs=ts.WorkAttr.AUXILIARY)
    @staticmethod
    def reduce_sums(stage_info, old_max, new_max, old_sum, local_sum):
        scales = cl.Vector(rescale(old_max[0], new_max[0]), rescale(old_max[1], new_max[1]))
        return cl.fma(old_sum, scales, local_sum)

    @ts.consumer_work(outputs=2)
    @staticmethod
    def compute_softmax_loop(stage_info, old_max, column, scratch_offset, inst_idx, cfg):
        warp, lane = warp_and_lane(stage_info)
        group, _, _ = work_coords(stage_info)
        inputs = stage_info.context.tasks_inputs
        scratch = smem_array(stage_info, scratch_offset, cl.uint32, 8)
        if not cfg.use_variable_seqlens_q and stage_info.loop_offset == 0:
            if warp == 0 and lane < 8:
                scratch[lane] = max_encode(cl.float32(NEG_MAX))
            cl.barrier_sync_block(number_of_threads=128, barrier_id=inst_idx + 1)
        values = load_o(stage_info, column)
        length = inputs.seq_len
        if not cfg.use_variable_seqlens_q:
            tile = 2 * stage_info.loop_offset + inst_idx
            lower = cl.int32(0)
            if cfg.window_left >= 0:
                lower = cl.maximum(length - cfg.window_left - 1, 0)
                tile += lower // 128
            scores = values
            if ((tile + 1) * 128 > length) | (tile * 128 < lower):
                token_base = tile * 128 + warp * 32 + lane // 4
                scores = cl.Vector(
                    *tuple(
                        values[i]
                        if (token_base + (i // 2) * 8 < length)
                        & (token_base + (i // 2) * 8 >= lower)
                        else cl.float32(NEG_MAX)
                        for i in cl.static_iter(range(8))
                    )
                )
            if group * 8 + 8 > inputs.head_ratio:
                scores = cl.Vector(
                    *tuple(
                        scores[i]
                        if group * 8 + (lane % 4) * 2 + i % 2 < inputs.head_ratio
                        else cl.float32(NEG_MAX)
                        for i in cl.static_iter(range(8))
                    )
                )
        if cfg.use_variable_seqlens_q:
            start, _ = kv_tile_bounds(length, inputs.seq_len_q, group, cfg)
            tile = 2 * stage_info.loop_offset + inst_idx + start
            q_base = q_group_token_base(group, cfg)
            visible_end, visible_begin = length, cl.int32(0)
            if cfg.mask_type == "causal":
                visible_end = length - inputs.seq_len_q + q_base + 1
                if cfg.window_left >= 0:
                    q_end = cl.minimum(q_base + cfg.q_tokens_per_cta, inputs.seq_len_q)
                    visible_begin = cl.maximum(
                        length - inputs.seq_len_q + q_end - cfg.window_left - 1, 0
                    )
            scores = values
            if ((tile + 1) * 128 > visible_end) | (tile * 128 < visible_begin):
                masked = ()
                for i in cl.static_iter(range(8)):
                    q_token, _ = q_row_token_and_local_head(
                        group, (lane % 4) * 2 + i % 2, cfg
                    )
                    upper, lower = length, cl.int32(0)
                    if cfg.mask_type == "causal":
                        upper = length - inputs.seq_len_q + q_token + 1
                        if cfg.window_left >= 0:
                            lower = cl.maximum(upper - cfg.window_left - 1, 0)
                    token = tile * 128 + warp * 32 + lane // 4 + (i // 2) * 8
                    masked += (
                        values[i] if (token < upper) & (token >= lower) else cl.float32(NEG_MAX),
                    )
                scores = cl.Vector(*masked)
            valid_scores = ()
            if cfg.groups_tokens_heads_q:
                valid_tokens = cl.minimum(
                    cl.maximum(inputs.seq_len_q - q_base, 0), cfg.q_tokens_per_cta
                )
                valid_rows = valid_tokens * cfg.heads_q_per_kv
                for i in cl.static_iter(range(8)):
                    row = (lane % 4) * 2 + i % 2
                    valid_scores += (
                        scores[i] if row < valid_rows else cl.float32(NEG_MAX),
                    )
            else:
                for i in cl.static_iter(range(8)):
                    q_token, head = q_row_token_and_local_head(
                        group, (lane % 4) * 2 + i % 2, cfg
                    )
                    valid_scores += (
                        scores[i] if (q_token < inputs.seq_len_q) & (head < cfg.heads_q_per_kv)
                        else cl.float32(NEG_MAX),
                    )
            scores = cl.Vector(*valid_scores)
        maxima = ()
        if cfg.use_variable_seqlens_q:
            local_maxima = ()
            for pair in cl.static_iter(range(2)):
                first = cl._nvvm.fmax_ftz_f(scores[pair], scores[pair + 2])
                second = cl._nvvm.fmax_ftz_f(scores[pair + 4], scores[pair + 6])
                value = cl._nvvm.fmax_ftz_f(first, second)
                local_maxima += (cl._nvvm.fmax_ftz_f(value, old_max[pair]),)
            warp_maxima = ()
            for pair in cl.static_iter(range(2)):
                value = local_maxima[pair]
                shuffled, _ = cl.shuffle_sync(cl.ShuffleKind.XOR, value, 16)
                value = cl._nvvm.fmax_ftz_f(value, shuffled)
                shuffled, _ = cl.shuffle_sync(cl.ShuffleKind.XOR, value, 8)
                value = cl._nvvm.fmax_ftz_f(value, shuffled)
                warp_maxima += (value,)
            if stage_info.loop_offset == 0:
                if warp == 0 and lane < 8:
                    scratch[lane] = max_encode(cl.float32(NEG_MAX))
                cl.barrier_sync_block(number_of_threads=128, barrier_id=inst_idx + 1)
            if lane < 8:
                for pair in cl.static_iter(range(2)):
                    cl.atomic_rmw(
                        cl.AtomicOp.MAX, scratch.pointer((lane % 4) * 2 + pair),
                        max_encode(warp_maxima[pair]),
                    )
        else:
            for pair in cl.static_iter(range(2)):
                value = old_max[pair]
                for k in cl.static_iter(range(4)):
                    value = cl.maximum(value, scores[2 * k + pair])
                shuffled, _ = cl.shuffle_sync(cl.ShuffleKind.XOR, value, 16)
                value = cl.maximum(value, shuffled)
                shuffled, _ = cl.shuffle_sync(cl.ShuffleKind.XOR, value, 8)
                value = cl.maximum(value, shuffled)
                if lane < 8:
                    cl.atomic_rmw(
                        cl.AtomicOp.MAX, scratch.pointer((lane % 4) * 2 + pair), max_encode(value)
                    )
        cl.barrier_sync_block(number_of_threads=128, barrier_id=inst_idx + 1)
        for pair in cl.static_iter(range(2)):
            maxima += (max_decode(scratch[(lane % 4) * 2 + pair]),)
        return scores, cl.Vector(*maxima)


@dataclass(kw_only=True, eq=False)
class SmemPResource(ts.MemoryResource):
    @ts.producer_work(outputs=1)
    @staticmethod
    def compute_p(stage_info, scores, new_max, offset, cfg):
        warp, lane = warp_and_lane(stage_info)
        probabilities = ()
        for k in cl.static_iter(range(4)):
            score_pair = cl.Vector(scores[2 * k], scores[2 * k + 1])
            shifted = cl.sub(score_pair, new_max, rounding_mode=cl.RoundingMode.RN)
            p_pair = cl.exp2(
                shifted * pair_vector(128**-0.5 * 1.4426950408889634), flush_to_zero=True
            )
            probabilities += (
                p_pair[0] if new_max[0] != NEG_MAX else cl.float32(0),
                p_pair[1] if new_max[1] != NEG_MAX else cl.float32(0),
            )
        local_sum = pair_vector(0)
        for k in cl.static_iter((0, 2, 1, 3)):
            local_sum = cl.add(
                local_sum,
                cl.Vector(probabilities[2 * k], probabilities[2 * k + 1]),
                rounding_mode=cl.RoundingMode.RN,
            )
        registers = ()
        for k in cl.static_iter(range(4)):
            packed = cl.Vector(
                cfg.q_dtype(probabilities[2 * k]), cfg.q_dtype(probabilities[2 * k + 1])
            )
            registers += (packed.reinterpret_as_scalar(cl.int32),)
        smem = smem_array(stage_info, offset, cl.int32, 512)
        row, matrix = lane % 8, lane // 8
        col = (warp % 2) * 4 + matrix
        byte_offset = (warp // 2) * 1024 + row * 128 + ((col ^ row) * 16)
        cl.store_matrix(
            smem.pointer(byte_offset // 4),
            cl.Vector(*registers),
            shape=cl.MatrixStoreShape.M8N8,
            transpose=True,
        )
        cl.fence_proxy_bidirectional(
            cl.FenceProxy.ASYNC, restriction=cl.FenceRestriction.shared_block()
        )
        return local_sum

    @ts.consumer_work(outputs=1)
    @staticmethod
    def p_desc(stage_info, offset, cfg):
        return smem_descriptor(smem_array(stage_info, offset, cfg.q_dtype, 1024).pointer(), 1024)


@dataclass(kw_only=True, eq=False)
class TmemSoftmaxStatsResource(ts.MemoryResource):
    @ts.producer_work
    @staticmethod
    def store_stats(stage_info, values, maxima, column):
        warp, _ = warp_and_lane(stage_info)
        cl.tcgen05_store(
            cl.Tcgen05LoadStoreShape.SHAPE_32X32B,
            tmem_ptr(stage_info, column, warp * 32),
            cl.Vector(values[0], values[1], maxima[0], maxima[1]),
        )
        cl.tcgen05_wait_store()

    @ts.consumer_work(outputs=1)
    @staticmethod
    def load_stats(stage_info, column):
        warp, _ = warp_and_lane(stage_info)
        values = cl.tcgen05_load(
            cl.Tcgen05LoadStoreShape.SHAPE_32X32B,
            tmem_ptr(stage_info, column, warp * 32),
            element_count=4,
            dtype=cl.float32,
        )
        cl.tcgen05_wait_load()
        return values


@dataclass(kw_only=True, eq=False)
class TmemOResource(ts.MemoryResource):
    @ts.producer_work
    @staticmethod
    def pv_mma(stage_info, v_desc, p_desc, section, cfg):
        instr = cl.Tcgen05InstructionDescriptor(
            d_type=cl.float32,
            a_type=cfg.q_dtype,
            b_type=cfg.q_dtype,
            m=128,
            n=8,
            transpose_a=True,
        ).encode()
        accumulate = stage_info.loop_offset > 0 if section == 1 else num_pairs(stage_info, cfg) > 1
        if cl.elect_sync():
            for ki in cl.static_iter(range(8)):
                cl.tcgen05_mma(
                    cl.Tcgen05MMAKind.F16,
                    tmem_ptr(stage_info, 80 + stage_info.stage_idx * 8),
                    v_desc + ki * 128,
                    p_desc + ki * 2 + (ki // 4) * 56,
                    instr,
                    accumulate=accumulate if ki == 0 else True,
                    cta_group=cl.CTAGroup.CTA_1,
                )

    @ts.consumer_work
    @staticmethod
    def correction(stage_info, stats):
        unchanged = (stats[0] == stats[2]) & (stats[1] == stats[3])
        if not cl.vote_all_sync(unchanged):
            scales = cl.Vector(
                cl.float32(1) if stats[0] == stats[2] else rescale(stats[0], stats[2]),
                cl.float32(1) if stats[1] == stats[3] else rescale(stats[1], stats[3]),
            )
            column = 80 + stage_info.stage_idx * 8
            values = load_o(stage_info, column)
            corrected = values * cl.Vector(*tuple(scales[i % 2] for i in cl.static_iter(range(8))))
            store_o(stage_info, column, corrected)
        cl.tcgen05_wait_store()

    @ts.consumer_work
    @staticmethod
    def tail_epilogue(stage_info, stats0, stats1, scratch_offset, output_offset, cfg):
        warp, lane = warp_and_lane(stage_info)
        group, kv_head, batch = work_coords(stage_info)
        inputs = stage_info.context.tasks_inputs
        scratch = smem_array(stage_info, scratch_offset, cl.float32, 32)
        exp0, exp1 = (), ()
        for pair in cl.static_iter(range(2)):
            maximum = cl.maximum(stats0[pair + 2], stats1[pair + 2])
            if cfg.use_variable_seqlens_q:
                maximum = cl._nvvm.fmax_ftz_f(stats0[pair + 2], stats1[pair + 2])
            e0 = rescale(stats0[pair + 2], maximum)
            e1 = rescale(stats1[pair + 2], maximum)
            exp0 += (e0,)
            exp1 += (e1,)
            total = stats0[pair] * e0 + stats1[pair] * e1
            for mask in cl.static_iter((16, 8, 4)):
                shuffled, _ = cl.shuffle_sync(cl.ShuffleKind.XOR, total, mask)
                total += shuffled
            if lane < 4:
                scratch[warp * 8 + lane * 2 + pair] = total
        cl.barrier_sync_block(number_of_threads=128, barrier_id=3)
        denom = ()
        for pair in cl.static_iter(range(2)):
            total = cl.float32(0)
            for w in cl.static_iter(range(4)):
                total += scratch[w * 8 + (lane % 4) * 2 + pair]
            denom += (total,)
        o0, o1 = load_o(stage_info, 80), load_o(stage_info, 88)
        inverse = cl.Vector(safe_norm_rcp(denom[0]), safe_norm_rcp(denom[1]))
        scale0 = cl.Vector(*exp0) * inverse
        scale1 = cl.Vector(*exp1) * inverse
        registers = ()
        for k in cl.static_iter(range(4)):
            pair0 = cl.Vector(o0[2 * k], o0[2 * k + 1])
            pair1 = cl.Vector(o1[2 * k], o1[2 * k + 1])
            final_pair = cl.fma(pair0, scale0, pair1 * scale1)
            registers += (final_pair.astype(cfg.out_dtype).reinterpret_as_scalar(cl.int32),)
        smem = smem_array(stage_info, output_offset, cl.int32, 512)
        row = lane % 8
        col = (warp % 2) * 4 + lane // 8
        byte_offset = (warp // 2) * 1024 + row * 128 + ((col ^ row) * 16)
        cl.store_matrix(
            smem.pointer(byte_offset // 4),
            cl.Vector(*registers),
            shape=cl.MatrixStoreShape.M8N8,
            transpose=True,
        )
        cl.fence_proxy_bidirectional(
            cl.FenceProxy.ASYNC, restriction=cl.FenceRestriction.shared_block()
        )
        cl.barrier_sync_block(number_of_threads=128, barrier_id=3)
        base_offset = (warp * 32 + lane) * 16
        smem_row = base_offset // 128
        load_offset = base_offset ^ ((smem_row % 8) * 16)
        head = group * 8 + smem_row % 8
        q_valid = head < inputs.head_ratio
        if cfg.use_variable_seqlens_q:
            token, head = q_row_token_and_local_head(group, smem_row % 8, cfg)
            batch = inputs.q_token_offset + token
            q_valid = (token < inputs.seq_len_q) & (head < cfg.heads_q_per_kv)
        column = (smem_row // 8) * 64 + (base_offset % 128) // 2
        if q_valid:
            packed = smem.pointer(load_offset // 4).load(count=4, alignment=16)
            dst = inputs.o.pointer((batch, kv_head * inputs.head_ratio + head, column))
            cl.bitcast(dst, cl.pointer_dtype(cl.int32, cl.MemorySpace.GLOBAL)).store(
                packed, alignment=16
            )
