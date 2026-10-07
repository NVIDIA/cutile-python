# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Two alternating K/V instances with peeled QK head and PV tail."""

from contextlib import contextmanager
from dataclasses import dataclass, field, replace

import cuda.lang as cl
import task_scheduling as ts
from task_scheduling.task import DeviceWorkTileLoop

from .fmha_decode_config import FmhaDecodeConfig
from .fmha_decode_resources.helpers_common import kv_tile_bounds, q_group_token_base


@dataclass(frozen=True)
class _ResolvedPackedDecodeWorkQueue:
    cfg: FmhaDecodeConfig
    qo_indptr: object


@dataclass(frozen=True)
class _DevicePackedDecodeWorkQueue:
    cfg: FmhaDecodeConfig

    def bind_inputs(self, inputs):
        return _ResolvedPackedDecodeWorkQueue(self.cfg, inputs.qo_indptr)


@dataclass(kw_only=True, eq=False)
class PackedDecodeWorkQueue(ts.WorkQueue):
    """Skip inactive packed-Q tiles without skipping scheduler handoffs."""

    cfg: FmhaDecodeConfig = field(init=False, default=None)

    def __init__(self, cfg, **kwargs):
        super().__init__(**kwargs)
        self.cfg = cfg

    def _freeze_skip_context(self):
        return _DevicePackedDecodeWorkQueue(self.cfg)

    def skip_work_tile_if(self, work_tile):
        group, _, batch = work_tile.tile_idx
        length = self.qo_indptr[batch + 1] - self.qo_indptr[batch]
        return q_group_token_base(group, self.cfg) >= length


@dataclass(frozen=True)
class _RefreshDecodeInputs:
    """Bind request metadata once per logical tile, before its pipeline work."""

    packed_q: bool

    def __call__(self, context):
        inputs = context.tasks_inputs
        batch = context.work_tile.tile_idx[2]
        begin = inputs.paged_kv_indptr[batch]
        q_begin, q_length = inputs.q_token_offset, inputs.seq_len_q
        if self.packed_q:
            q_begin = inputs.qo_indptr[batch]
            q_length = inputs.qo_indptr[batch + 1] - q_begin
        return replace(
            context,
            tasks_inputs=replace(
                inputs,
                seq_len=inputs.seq_lens[batch],
                page_begin=begin,
                page_count=inputs.paged_kv_indptr[batch + 1] - begin,
                q_token_offset=q_begin,
                seq_len_q=q_length,
            ),
        )


@dataclass(kw_only=True, eq=False)
class ScheduleTokenThrottleResource(ts.MemoryResource):
    """Prevent the scheduler from recycling a token before Load owns it."""

    @ts.producer_work
    @staticmethod
    def publish_schedule_token(stage_info):
        pass

    @ts.consumer_work
    @staticmethod
    def consume_schedule_token(stage_info):
        pass


def _work_queue_tail(work_queue):
    work_queue.wait()
    work_queue.get_and_advance_work_tile()
    work_queue.release()


def _schedule_token_throttle_head(throttle):
    if throttle is not None:
        throttle.acquire()
        throttle.publish_schedule_token()
        throttle.commit()


@contextmanager
def _work_tile_schedule(work_queue, cfg, throttle=None):
    if work_queue is None:
        yield
    elif cfg.use_variable_seqlens_q:
        with ts.work_tile_loop(
            work_queue, skip_if=PackedDecodeWorkQueue.skip_work_tile_if
        ) as work_tiles:
            _schedule_token_throttle_head(throttle)
            with work_tiles.skippable():
                yield
            _work_queue_tail(work_queue)
    else:
        with ts.work_tile_loop(work_queue):
            _schedule_token_throttle_head(throttle)
            yield
            _work_queue_tail(work_queue)


@dataclass(frozen=True)
class _ResolvedDecodeDomain:
    seq_len: object
    seq_len_q: object
    offset: int
    cfg: FmhaDecodeConfig


@dataclass(frozen=True)
class _DeviceDecodeDomain:
    offset: int
    cfg: FmhaDecodeConfig

    def bind_inputs(self, inputs):
        return _ResolvedDecodeDomain(inputs.seq_len, inputs.seq_len_q, self.offset, self.cfg)


class DecodeDomainTask(ts.Task):
    def __init__(self, *args, cfg, offset=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.offset = offset
        self.cfg = cfg

    def _freeze_domain_task(self):
        return _DeviceDecodeDomain(self.offset, self.cfg)

    def to_device(self, *args, **kwargs):
        device_task = super().to_device(*args, **kwargs)
        # Keep request metadata in SSA values across all resource/domain steps.
        # Pipeline phases remain live when a persistent CTA takes another tile.
        return replace(
            device_task,
            body=tuple(
                replace(
                    node, body=(_RefreshDecodeInputs(self.cfg.use_variable_seqlens_q),) + node.body
                )
                if isinstance(node, DeviceWorkTileLoop)
                else node
                for node in device_task.body
            ),
        )

    def get_domain(self, tile_coord):
        start, end = kv_tile_bounds(self.seq_len, self.seq_len_q, tile_coord[0], self.cfg)
        if self.cfg.use_variable_seqlens_q:
            return ((end - start + 1) >> 1) - self.offset
        return (end - start + 1) // 2 - self.offset

    def get_last_iteration(self, tile_coord):
        return cl.maximum(DecodeDomainTask.get_domain(self, tile_coord) - 1, 0)


def create_load_task(sq, skv, pages, offsets, cfg, work_queue=None, throttle=None):
    def schedule_body(sq, skv, pages, wq, throttle):
        with _work_tile_schedule(wq, cfg, throttle):

            def load(inst, section, is_v):
                pages.wait()
                ids = pages.page_ids(offsets["pages"], inst, section, is_v, cfg)
                skv.acquire()
                skv.tma_load(ids, offsets["kv"], is_v, cfg)
                skv.commit()
                pages.release()

            sq.acquire()
            sq.tma_load(offsets["q"], cfg)
            sq.commit()
            load(0, 0, False)
            load(1, 0, False)

            def body():
                load(0, 1, False)
                load(0, 1, True)
                load(1, 1, False)
                load(1, 1, True)

            ts.domain_loop(0, DecodeDomainTask.get_domain, 1, body)
            load(0, 2, True)
            load(1, 2, True)

    @ts.schedule
    def direct_schedule(stage_info, sq, skv, pages):
        schedule_body(sq, skv, pages, None, None)

    @ts.schedule
    def persistent_schedule(stage_info, sq, skv, pages, wq, throttle):
        schedule_body(sq, skv, pages, wq, throttle)

    captured = (
        persistent_schedule(sq, skv, pages, work_queue, throttle)
        if work_queue is not None
        else direct_schedule(sq, skv, pages)
    )

    return DecodeDomainTask(
        name="LoadTask",
        warp_idx=cfg.clc_load_warp_idx if cfg.use_persistent_scheduler else cfg.load_warp_idx,
        num_warps=1,
        schedule=captured,
        offset=1,
        cfg=cfg,
    )


def create_page_offsets_task(pages, offset, cfg, work_queue=None):
    def schedule_body(pages, wq):
        with _work_tile_schedule(wq, cfg):
            def load(inst, section, is_v):
                pages.acquire()
                pages.load_page_offsets(offset, inst, section, is_v, cfg)
                pages.commit()

            load(0, 0, False)
            load(1, 0, False)

            def body():
                load(0, 1, False)
                load(0, 1, True)
                load(1, 1, False)
                load(1, 1, True)

            ts.domain_loop(0, DecodeDomainTask.get_domain, 1, body)
            load(0, 2, True)
            load(1, 2, True)

    @ts.schedule
    def direct_schedule(stage_info, pages):
        schedule_body(pages, None)

    @ts.schedule
    def persistent_schedule(stage_info, pages, wq):
        schedule_body(pages, wq)

    captured = (
        persistent_schedule(pages, work_queue)
        if work_queue is not None
        else direct_schedule(pages)
    )

    return DecodeDomainTask(
        name="PageOffsetsTask",
        warp_idx=14,
        num_warps=1,
        schedule=captured,
        offset=1,
        cfg=cfg,
    )


def create_mma_task(sq, skv, s0, s1, p0, p1, o, offsets, cfg, work_queue=None):
    def schedule_body(sq, skv, s0, s1, p0, p1, o, wq):
        with _work_tile_schedule(wq, cfg):
            sq.wait()
            q_desc = sq.q_desc(offsets["q"], cfg)

            def qk(s, column):
                s.acquire()
                skv.wait()
                k_desc = skv.kv_desc(offsets["kv"], cfg)
                s.qk_mma(q_desc, k_desc, column, cfg)
                skv.release()
                s.commit()

            def pv(p, p_offset, section):
                p.wait()
                p_desc = p.p_desc(p_offset, cfg)
                o.acquire()
                skv.wait()
                v_desc = skv.kv_desc(offsets["kv"], cfg)
                o.pv_mma(v_desc, p_desc, section, cfg)
                skv.release()
                o.commit()
                p.release()

            qk(s0, 0)
            qk(s1, 8)

            def body():
                qk(s0, 0)
                pv(p0, offsets["p0"], 1)
                qk(s1, 8)
                pv(p1, offsets["p1"], 1)

            ts.domain_loop(0, DecodeDomainTask.get_domain, 1, body)
            pv(p0, offsets["p0"], 2)
            pv(p1, offsets["p1"], 2)
            sq.release()

    @ts.schedule
    def direct_schedule(stage_info, sq, skv, s0, s1, p0, p1, o):
        schedule_body(sq, skv, s0, s1, p0, p1, o, None)

    @ts.schedule
    def persistent_schedule(stage_info, sq, skv, s0, s1, p0, p1, o, wq):
        schedule_body(sq, skv, s0, s1, p0, p1, o, wq)

    captured = (
        persistent_schedule(sq, skv, s0, s1, p0, p1, o, work_queue)
        if work_queue is not None
        else direct_schedule(sq, skv, s0, s1, p0, p1, o)
    )

    return DecodeDomainTask(
        name="MmaTask",
        warp_idx=12,
        num_warps=1,
        schedule=captured,
        offset=1,
        cfg=cfg,
    )


def create_softmax_task(s, p, stats, inst, offsets, cfg, work_queue=None):
    def schedule_body(s, p, stats, wq):
        with _work_tile_schedule(wq, cfg):
            maximum, total = s.init_state()

            def body(maximum, total):
                s.wait()
                scores, new_max = s.compute_softmax_loop(
                    maximum, inst * 8, offsets["max" + str(inst)], inst, cfg
                )
                s.release()
                stats.acquire()
                stats.store_stats(maximum, new_max, 16 + inst * 32)
                stats.commit()
                p.acquire()
                local_sum = p.compute_p(scores, new_max, offsets["p" + str(inst)], cfg)
                p.commit()
                total = s.reduce_sums(maximum, new_max, total, local_sum)
                return new_max, total

            # Peel the final iteration to keep its stats handoff out of the steady-state loop.
            maximum, total = ts.domain_loop(
                0, DecodeDomainTask.get_last_iteration, 1, body, maximum, total, unroll=1
            )

            def last_body(maximum, total):
                new_max, total = body(maximum, total)
                with ts.last_iter():
                    stats.acquire()
                    stats.store_stats(total, new_max, 16 + inst * 32)
                    stats.commit()
                return new_max, total

            ts.domain_loop(
                DecodeDomainTask.get_last_iteration,
                DecodeDomainTask.get_domain,
                1,
                last_body,
                maximum,
                total,
                unroll=1,
            )

    @ts.schedule
    def direct_schedule(stage_info, s, p, stats):
        schedule_body(s, p, stats, None)

    @ts.schedule
    def persistent_schedule(stage_info, s, p, stats, wq):
        schedule_body(s, p, stats, wq)

    captured = (
        persistent_schedule(s, p, stats, work_queue)
        if work_queue is not None
        else direct_schedule(s, p, stats)
    )

    return DecodeDomainTask(
        name="Softmax" + str(inst) + "Task",
        warp_idx=inst * 4,
        num_warps=4,
        schedule=captured,
        cfg=cfg,
    )


def create_correction_task(stats0, stats1, o, offsets, cfg, work_queue=None):
    def schedule_body(stats0, stats1, o, wq):
        with _work_tile_schedule(wq, cfg):
            stats0.wait()
            stats0.release()
            stats1.wait()
            stats1.release()

            def correct(stats, column):
                stats.wait()
                values = stats.load_stats(column)
                stats.release()
                o.wait()
                o.correction(values)
                o.release()

            def body():
                correct(stats0, 16)
                correct(stats1, 48)

            ts.domain_loop(0, DecodeDomainTask.get_domain, 1, body)
            stats0.wait()
            values0 = stats0.load_stats(16)
            stats0.release()
            o.wait()
            stats1.wait()
            values1 = stats1.load_stats(48)
            stats1.release()
            o.wait()
            o.tail_epilogue(values0, values1, offsets["sum"], offsets["output"], cfg)
            o.release()
            o.release()

    @ts.schedule
    def direct_schedule(stage_info, stats0, stats1, o):
        schedule_body(stats0, stats1, o, None)

    @ts.schedule
    def persistent_schedule(stage_info, stats0, stats1, o, wq):
        schedule_body(stats0, stats1, o, wq)

    captured = (
        persistent_schedule(stats0, stats1, o, work_queue)
        if work_queue is not None
        else direct_schedule(stats0, stats1, o)
    )

    return DecodeDomainTask(
        name="CorrectionTask",
        warp_idx=8,
        num_warps=4,
        schedule=captured,
        offset=1,
        cfg=cfg,
    )


def create_scheduler_task(work_queue, throttle, cfg):
    @ts.schedule
    def schedule(stage_info, wq, throttle):
        with ts.work_tile_loop(wq):
            with ts.domain_loop(0):
                pass
            throttle.wait()
            throttle.consume_schedule_token()
            throttle.release()
            wq.acquire()
            wq.fetch_work_tile()
            wq.commit()
            _work_queue_tail(wq)

    return ts.Task(
        name="SchedulerTask",
        warp_idx=cfg.scheduler_warp_idx,
        num_warps=1,
        schedule=schedule(work_queue, throttle),
    )
