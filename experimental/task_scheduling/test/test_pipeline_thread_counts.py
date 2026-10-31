# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cooperative-group sizes must match the arrivals each barrier stage receives."""

from dataclasses import dataclass

import pytest

from task_scheduling_test_requirements import task_scheduling as ts


@dataclass(kw_only=True, eq=False)
class Pipe(ts.MemoryResource):
    @ts.producer_work
    @staticmethod
    def write(stage_info):
        stage_info

    @ts.consumer_work
    @staticmethod
    def read(stage_info):
        stage_info


@ts.schedule
def produce(pipe):
    pipe.acquire()
    pipe.write()
    pipe.commit()


@ts.schedule
def consume(pipe):
    pipe.wait()
    pipe.read()
    pipe.release()


def producer(pipe, warp_idx, num_warps=1, **kwargs):
    return ts.Task(warp_idx, num_warps, schedule=produce(pipe), name=f"prod{warp_idx}", **kwargs)


def consumer(pipe, warp_idx, num_warps=1, **kwargs):
    return ts.Task(warp_idx, num_warps, schedule=consume(pipe), name=f"cons{warp_idx}", **kwargs)


def manager(tasks, **kwargs):
    return ts.TaskManager(tasks, {}, exhaustive_deadlock_race_check=False, **kwargs)


def group(size):
    return ts.CooperativeGroup(size)


# ── Async sides ──────────────────────────────────────────────────────────


def test_async_sides_count_every_thread_of_every_signaling_task():
    config = ts.PipelineConfig.create_async_async_pipeline_cfg(2, group(64), group(96))
    pipe = Pipe(name="pipe", pipeline_config=config)
    manager([producer(pipe, 0, 2), consumer(pipe, 2), consumer(pipe, 3, 2)])


def test_async_producer_size_mismatch_names_tasks_and_formula():
    config = ts.PipelineConfig.create_async_async_pipeline_cfg(2, group(32), group(32))
    pipe = Pipe(name="pipe", pipeline_config=config)
    expected = r"producer_group.size=32 does not match expected 64"
    with pytest.raises(ValueError, match=expected) as info:
        manager([producer(pipe, 0, 2), consumer(pipe, 2)])
    message = str(info.value)
    assert "Resource 'pipe' (pipeline=AsyncAsync)" in message
    assert "Σ task.num_warps × 32" in message
    assert "'prod0' (2 warps)" in message


def test_two_consumers_need_the_sum_of_their_threads():
    config = ts.PipelineConfig.create_async_async_pipeline_cfg(1, group(32), group(32))
    pipe = Pipe(name="pipe", pipeline_config=config)
    with pytest.raises(ValueError, match=r"consumer_group.size=32 does not match expected 64"):
        manager([producer(pipe, 0), consumer(pipe, 1), consumer(pipe, 2)])


def test_elected_lane_accepts_size_one_for_a_single_warp_only():
    config = ts.PipelineConfig.create_async_async_pipeline_cfg(1, group(1), group(1))
    pipe = Pipe(name="pipe", pipeline_config=config)
    manager([producer(pipe, 0), consumer(pipe, 1)])
    with pytest.raises(ValueError, match=r"consumer_group.size=1 does not match expected 64"):
        manager([producer(pipe, 0), consumer(pipe, 1, 2)])


def test_sides_without_signaling_tasks_are_left_to_the_barrier_model():
    config = ts.PipelineConfig.create_async_async_pipeline_cfg(1, group(1000), group(32))
    pipe = Pipe(name="pipe", pipeline_config=config)
    manager([consumer(pipe, 0)])


def test_skip_validation_downgrades_the_mismatch_to_a_warning():
    config = ts.PipelineConfig.create_async_async_pipeline_cfg(1, group(32), group(1000))
    pipe = Pipe(name="pipe", pipeline_config=config)
    with pytest.warns(UserWarning, match="consumer_group.size=1000"):
        manager([producer(pipe, 0), consumer(pipe, 1)], skip_validation=True)


# ── TMA producers ────────────────────────────────────────────────────────


def test_tma_producer_arrives_once_per_warp():
    config = ts.PipelineConfig.create_tma_async_pipeline_cfg(2, 128, group(2), group(64))
    pipe = Pipe(name="pipe", pipeline_config=config)
    manager([producer(pipe, 0), producer(pipe, 1), consumer(pipe, 2, 2)])
    with pytest.raises(ValueError, match=r"producer_group.size=2 does not match expected 1"):
        manager([producer(pipe, 0), consumer(pipe, 2, 2)])


def test_multi_warp_tma_producer_needs_task_warp_leader_or_interleaving():
    config = ts.PipelineConfig.create_tma_async_pipeline_cfg(2, 128, group(2), group(32))
    pipe = Pipe(name="pipe", pipeline_config=config)
    with pytest.raises(ValueError, match="more than one warp"):
        manager([producer(pipe, 0, 2), consumer(pipe, 2)])

    leader = ts.PipelineConfig.create_tma_async_pipeline_cfg(
        2, 128, group(1), group(32),
        producer_signaling_threads=ts.SignalingThreads.TaskWarpLeader,
    )
    pipe = Pipe(name="pipe", pipeline_config=leader)
    manager([producer(pipe, 0, 2), consumer(pipe, 2)])
    with pytest.raises(ValueError, match=r"producer_group.size=1 does not match expected 2"):
        manager([producer(pipe, 0, 2), producer(pipe, 2, 2), consumer(pipe, 4)])


def test_interleaved_tma_sides_are_signaled_by_one_elected_thread():
    config = ts.PipelineConfig.create_tma_async_pipeline_cfg(
        4, 128, group(2), group(1), interleave_stride=2
    )
    pipe = Pipe(name="pipe", pipeline_config=config)
    with pytest.raises(ValueError, match=r"producer_group.size=2 but interleave_stride=2"):
        manager([producer(pipe, 0, 2), consumer(pipe, 2)])
    config = ts.PipelineConfig.create_tma_async_pipeline_cfg(
        4, 128, group(1), group(32), interleave_stride=2
    )
    pipe = Pipe(name="pipe", pipeline_config=config)
    with pytest.raises(ValueError, match=r"consumer_group.size=32 but interleave_stride=2"):
        manager([producer(pipe, 0, 2), consumer(pipe, 2)])


def test_task_warp_leader_is_restricted_to_tma_producers():
    config = ts.PipelineConfig.create_async_async_pipeline_cfg(
        1, group(32), group(32),
        producer_signaling_threads=ts.SignalingThreads.TaskWarpLeader,
    )
    pipe = Pipe(name="pipe", pipeline_config=config)
    with pytest.raises(ValueError, match="only supported for TMA producer"):
        manager([producer(pipe, 0), consumer(pipe, 1)])
    config = ts.PipelineConfig.create_tma_async_pipeline_cfg(
        1, 128, group(1), group(32),
        consumer_signaling_threads=ts.SignalingThreads.TaskWarpLeader,
    )
    pipe = Pipe(name="pipe", pipeline_config=config)
    with pytest.raises(ValueError, match="producer-side only"):
        manager([producer(pipe, 0), consumer(pipe, 1)])


# ── TmaAsync consumers ───────────────────────────────────────────────────


def test_tma_async_consumer_accepts_per_warp_or_all_thread_release():
    for size in (2, 64):
        config = ts.PipelineConfig.create_tma_async_pipeline_cfg(2, 128, group(1), group(size))
        pipe = Pipe(name="pipe", pipeline_config=config)
        manager([producer(pipe, 0), consumer(pipe, 1, 2)])
    config = ts.PipelineConfig.create_tma_async_pipeline_cfg(2, 128, group(1), group(32))
    pipe = Pipe(name="pipe", pipeline_config=config)
    with pytest.raises(ValueError, match=r"expected 2 for a per-warp lane release .* or 64 for"):
        manager([producer(pipe, 0), consumer(pipe, 1, 2)])


@pytest.mark.parametrize(
    "mcast_mode_mn, lanes",
    [((1, 0), 3), ((0, 1), 2), ((1, 1), 4)],
)
def test_clustered_tma_async_release_selects_one_lane_per_multicast_peer(mcast_mode_mn, lanes):
    def config(size):
        return ts.PipelineConfig.create_tma_async_pipeline_cfg(
            1, 128, group(1), group(size),
            cta_layout_vmnk=(1, 2, 3, 1),
            mcast_mode_mn=mcast_mode_mn,
        )

    pipe = Pipe(name="pipe", pipeline_config=config(lanes))
    manager([producer(pipe, 0), consumer(pipe, 1)])
    pipe = Pipe(name="pipe", pipeline_config=config(32))
    manager([producer(pipe, 0), consumer(pipe, 1)])
    pipe = Pipe(name="pipe", pipeline_config=config(1))
    with pytest.raises(ValueError, match=f"lanes-per-warp\\({lanes}\\)"):
        manager([producer(pipe, 0), consumer(pipe, 1)])


# ── tcgen05 sides ────────────────────────────────────────────────────────


def test_umma_sides_commit_once_per_warp():
    config = ts.PipelineConfig.create_umma_async_pipeline_cfg(1, group(1), group(128))
    pipe = Pipe(name="pipe", pipeline_config=config)
    manager([producer(pipe, 0), consumer(pipe, 1, 4)])
    config = ts.PipelineConfig.create_umma_async_pipeline_cfg(1, group(32), group(128))
    pipe = Pipe(name="pipe", pipeline_config=config)
    with pytest.raises(ValueError, match=r"producer_group.size=32 does not match expected 1"):
        manager([producer(pipe, 0), consumer(pipe, 1, 4)])


def test_umma_async_consumers_route_to_the_v_group_leader():
    def config(size, **kwargs):
        return ts.PipelineConfig.create_umma_async_pipeline_cfg(
            1, group(1), group(size), cta_layout_vmnk=(2, 1, 1, 1),
            producer_signaling_threads=ts.SignalingThreads.CtaLeader, **kwargs,
        )

    pipe = Pipe(name="tmem", pipeline_config=config(256))
    manager([producer(pipe, 0), consumer(pipe, 1, 4)])
    pipe = Pipe(name="tmem", pipeline_config=config(128))
    with pytest.raises(ValueError, match=r"expected 256 \(= Σ task.num_warps × 32 × ctas\(2\)\)"):
        manager([producer(pipe, 0), consumer(pipe, 1, 4)])
    pipe = Pipe(
        name="tmem",
        pipeline_config=config(128, consumer_signaling_threads=ts.SignalingThreads.CtaLeader),
    )
    manager([producer(pipe, 0), consumer(pipe, 1, 4)])


def test_umma_producer_in_a_cta_pair_commits_from_each_running_cta():
    def config(**kwargs):
        return ts.PipelineConfig.create_umma_async_pipeline_cfg(
            1, group(1), group(256), cta_layout_vmnk=(2, 1, 1, 1), **kwargs,
        )

    pipe = Pipe(name="tmem", pipeline_config=config())
    with pytest.raises(ValueError, match=r"producer_group.size=1 does not match expected 2"):
        manager([producer(pipe, 0), consumer(pipe, 1, 4)])
    manager([producer(pipe, 0, run_only_on_cta_id=0), consumer(pipe, 1, 4)])
    pipe = Pipe(
        name="tmem",
        pipeline_config=config(producer_signaling_threads=ts.SignalingThreads.CtaLeader),
    )
    manager([producer(pipe, 0), consumer(pipe, 1, 4)])


def test_tma_umma_release_follows_the_vmnk_multicast_mask():
    def config(size, layout, mcast_mode_mn, **kwargs):
        return ts.PipelineConfig.create_tma_umma_pipeline_cfg(
            1, 128, group(1), group(size),
            cta_layout_vmnk=layout, mcast_mode_mn=mcast_mode_mn, **kwargs,
        )

    leader = dict(consumer_signaling_threads=ts.SignalingThreads.CtaLeader)
    # One CTA pair: the leader commits once.
    pipe = Pipe(name="smem", pipeline_config=config(1, (2, 1, 1, 1), (1, 1), **leader))
    manager([producer(pipe, 0), consumer(pipe, 1)])
    # Both CTAs of the pair commit when every CTA signals.
    pipe = Pipe(name="smem", pipeline_config=config(2, (2, 1, 1, 1), (1, 1)))
    manager([producer(pipe, 0), consumer(pipe, 1)])
    # Eight CTAs, multicast along N: the two V-group leaders sharing N commit.
    pipe = Pipe(name="smem", pipeline_config=config(2, (2, 2, 2, 1), (0, 1), **leader))
    manager([producer(pipe, 0), consumer(pipe, 1)])
    pipe = Pipe(name="smem", pipeline_config=config(4, (2, 2, 2, 1), (0, 1), **leader))
    with pytest.raises(ValueError, match=r"consumer_group.size=4 does not match expected 2"):
        manager([producer(pipe, 0), consumer(pipe, 1)])
    # Multicast along both M and N without leader election: 2 × (2 + 2 - 1).
    pipe = Pipe(name="smem", pipeline_config=config(6, (2, 2, 2, 1), (1, 1)))
    manager([producer(pipe, 0), consumer(pipe, 1)])


# ── CLC work queues ──────────────────────────────────────────────────────


def clc_queue(consumer_size, layout):
    return ts.WorkQueue(
        name="queue",
        pipeline_config=ts.PipelineConfig.create_clc_fetch_async_pipeline_cfg(
            num_stages=2,
            num_bytes=16,
            producer_group=group(1),
            consumer_group=group(consumer_size),
            cta_layout_vmnk=layout,
            producer_signaling_threads=ts.SignalingThreads.CtaLeader,
            consumer_signaling_threads=ts.SignalingThreads.All,
        ),
    )


@ts.schedule
def scheduler(wq):
    with ts.work_tile_loop(wq):
        wq.try_acquire()
        wq.acquire()
        wq.fetch_work_tile()
        wq.commit()
        wq.try_wait()
        wq.wait()
        wq.get_and_advance_work_tile()
        wq.release()


@ts.schedule
def worker(wq):
    with ts.work_tile_loop(wq):
        wq.try_wait()
        wq.wait()
        wq.get_and_advance_work_tile()
        wq.release()


def test_clc_consumers_release_into_the_cluster_leader_from_every_cta():
    queue = clc_queue(5 * 32 * 2, (2, 1, 1, 1))
    tasks = [
        ts.Task(0, 1, schedule=scheduler(queue), name="scheduler"),
        ts.Task(1, 4, schedule=worker(queue), name="worker"),
    ]
    manager(tasks, infer_allocators=False)
    # A task pinned to one CTA arrives from that CTA only.
    queue = clc_queue(1 * 32 * 2 + 4 * 32, (2, 1, 1, 1))
    tasks = [
        ts.Task(0, 1, schedule=scheduler(queue), name="scheduler"),
        ts.Task(1, 4, schedule=worker(queue), name="worker", run_only_on_cta_id=0),
    ]
    manager(tasks, infer_allocators=False)
    queue = clc_queue(5 * 32, (2, 1, 1, 1))
    tasks = [
        ts.Task(0, 1, schedule=scheduler(queue), name="scheduler"),
        ts.Task(1, 4, schedule=worker(queue), name="worker"),
    ]
    with pytest.raises(ValueError, match=r"consumer_group.size=160 does not match expected 320"):
        manager(tasks, infer_allocators=False)


def test_clc_producer_arms_each_cta_barrier_once():
    queue = ts.WorkQueue(
        name="queue",
        pipeline_config=ts.PipelineConfig.create_clc_fetch_async_pipeline_cfg(
            num_stages=2,
            num_bytes=16,
            producer_group=group(32),
            consumer_group=group(32),
            producer_signaling_threads=ts.SignalingThreads.CtaLeader,
        ),
    )
    task = ts.Task(0, 1, schedule=scheduler(queue), name="scheduler")
    with pytest.raises(ValueError, match=r"producer_group.size=32 but the scheduler thread"):
        manager([task], infer_allocators=False)
