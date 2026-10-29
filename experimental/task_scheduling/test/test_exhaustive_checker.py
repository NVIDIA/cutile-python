# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Barrier-phase model of the exhaustive checker.

Participant counts: a full barrier flips after every producer task of the
resource has committed a stage and an empty barrier after every consumer task
has released it.  UMMA operand liveness: the read hold of a ``TmaUmma`` /
``AsyncUmma`` / ``UmmaUmma`` consumer persists past ``release()`` until the
next ``acquire()`` of that stage or a paired completion wait.
"""

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

    @ts.consumer_work(work_attrs=ts.WorkAttr.AUXILIARY, outputs=1)
    @staticmethod
    def probe(stage_info):
        return True


def async_config(stages, producers=32, consumers=32):
    return ts.PipelineConfig.create_async_async_pipeline_cfg(
        stages, ts.CooperativeGroup(producers), ts.CooperativeGroup(consumers)
    )


def tma_umma_config(stages):
    return ts.PipelineConfig.create_tma_umma_pipeline_cfg(
        stages, 128, ts.CooperativeGroup(1), ts.CooperativeGroup(32)
    )


def umma_async_config(stages):
    return ts.PipelineConfig.create_umma_async_pipeline_cfg(
        stages, ts.CooperativeGroup(1), ts.CooperativeGroup(128)
    )


def smem(name, size_bytes, offset=0):
    allocation = ts.SmemAllocation(name, size_bytes, 16)
    allocation.offset = offset
    return allocation


def racing_resources(result):
    names = set()
    for race in result.race_states:
        names.add(race.writer_resource)
        names.add(race.victim_resource)
    return names


def task(index, schedule, name):
    return ts.Task(index, 1, schedule=schedule, name=name)


# ── Participant counts ────────────────────────────────────────────────────


def test_one_producer_feeding_two_consumers_is_safe():
    pipe = Pipe(name="pipe", pipeline_config=async_config(2, consumers=64))

    @ts.schedule
    def producer(pipe):
        with ts.domain_loop(2):
            pipe.acquire()
            pipe.write()
            pipe.commit()

    @ts.schedule
    def consumer(pipe):
        with ts.domain_loop(2):
            pipe.wait()
            pipe.read()
            pipe.release()

    tasks = [
        task(0, producer(pipe), "producer"),
        task(1, consumer(pipe), "cons_a"),
        task(2, consumer(pipe), "cons_b"),
    ]
    result = ts.check_all_interleavings(tasks)
    assert result.is_safe
    assert result.complete_count >= 1
    ts.TaskManager(tasks, {})


def test_broadcast_ring_wraps_in_lockstep():
    """A one-stage ring refills only after both consumers released it."""
    pipe = Pipe(name="pipe", pipeline_config=async_config(1))

    @ts.schedule
    def producer(pipe):
        with ts.domain_loop(3):
            pipe.acquire()
            pipe.write()
            pipe.commit()

    @ts.schedule
    def consumer(pipe):
        with ts.domain_loop(3):
            pipe.wait()
            pipe.read()
            pipe.release()

    tasks = [
        task(0, producer(pipe), "producer"),
        task(1, consumer(pipe), "cons_a"),
        task(2, consumer(pipe), "cons_b"),
    ]
    result = ts.check_all_interleavings(tasks)
    assert result.is_safe
    assert result.complete_count >= 1


def test_consumer_with_fewer_releases_deadlocks_the_ring_wrap():
    pipe = Pipe(name="pipe", pipeline_config=async_config(1))

    @ts.schedule
    def producer(pipe):
        with ts.domain_loop(3):
            pipe.acquire()
            pipe.write()
            pipe.commit()

    @ts.schedule
    def consumer(pipe):
        with ts.domain_loop(3):
            pipe.wait()
            pipe.read()
            pipe.release()

    @ts.schedule
    def short_consumer(pipe):
        pipe.wait()
        pipe.read()
        pipe.release()

    tasks = [
        task(0, producer(pipe), "producer"),
        task(1, consumer(pipe), "cons_a"),
        task(2, short_consumer(pipe), "cons_b"),
    ]
    result = ts.check_all_interleavings(tasks)
    assert not result.is_safe
    assert result.complete_count == 0
    lines = result.deadlock_states[0].format_lines()
    assert "blocked tasks producer" in lines[0]
    assert any(
        "ProducerAcquire(pipe) on stage 0" in line
        and "empty barrier" in line
        and "1 of 2 arrivals" in line
        for line in lines[1:]
    )


def test_two_producers_must_both_arrive():
    pipe = Pipe(name="pipe", pipeline_config=async_config(2))

    @ts.schedule
    def producer(pipe):
        with ts.domain_loop(2):
            pipe.acquire()
            pipe.write()
            pipe.commit()

    @ts.schedule
    def consumer(pipe):
        with ts.domain_loop(2):
            pipe.wait()
            pipe.read()
            pipe.release()

    tasks = [
        task(0, producer(pipe), "prod_a"),
        task(1, producer(pipe), "prod_b"),
        task(2, consumer(pipe), "consumer"),
    ]
    result = ts.check_all_interleavings(tasks)
    assert result.is_safe
    assert result.complete_count >= 1
    pipe.pipeline_config = async_config(2, producers=64)
    ts.TaskManager(tasks, {})


def test_producer_with_fewer_commits_deadlocks_the_consumer():
    """The second round never fills: only one of two producers commits it."""
    pipe = Pipe(name="pipe", pipeline_config=async_config(2, producers=64))

    @ts.schedule
    def producer(pipe):
        with ts.domain_loop(2):
            pipe.acquire()
            pipe.write()
            pipe.commit()

    @ts.schedule
    def short_producer(pipe):
        pipe.acquire()
        pipe.write()
        pipe.commit()

    @ts.schedule
    def consumer(pipe):
        with ts.domain_loop(2):
            pipe.wait()
            pipe.read()
            pipe.release()

    tasks = [
        task(0, producer(pipe), "prod_a"),
        task(1, short_producer(pipe), "prod_b"),
        task(2, consumer(pipe), "consumer"),
    ]
    result = ts.check_all_interleavings(tasks)
    assert not result.is_safe
    assert result.complete_count == 0
    lines = result.deadlock_states[0].format_lines()
    assert lines[0].endswith("blocked tasks consumer")
    assert any(
        "ConsumerWait(pipe) on stage 1" in line
        and "full barrier" in line
        and "1 of 2 arrivals" in line
        for line in lines[1:]
    )
    with pytest.raises(ValueError, match="1 of 2 arrivals"):
        ts.TaskManager(tasks, {})


def test_participants_are_counted_from_the_captured_schedule():
    """A producer whose commit is guarded off still owns an arrival slot."""
    pipe = Pipe(name="pipe", pipeline_config=async_config(1))

    @ts.schedule
    def producer(pipe):
        pipe.acquire()
        pipe.write()
        pipe.commit()

    @ts.schedule
    def guarded_producer(pipe):
        flag = pipe.probe()
        with ts.when_true(flag, key="b"):
            pipe.acquire()
            pipe.write()
            pipe.commit()

    @ts.schedule
    def consumer(pipe):
        pipe.wait()
        pipe.read()
        pipe.release()

    tasks = [
        task(0, producer(pipe), "prod_a"),
        task(1, guarded_producer(pipe), "prod_b"),
        task(2, consumer(pipe), "consumer"),
    ]
    assert ts.check_all_interleavings(tasks, opaque_assignment={"b": True}).is_safe
    skipped = ts.check_all_interleavings(tasks, opaque_assignment={"b": False})
    assert not skipped.is_safe
    assert any(
        "1 of 2 arrivals" in line
        for line in skipped.deadlock_states[0].format_lines()
    )


def test_deadlock_witness_reports_stage_phase_and_arrivals():
    pipe = Pipe(name="pipe", pipeline_config=async_config(1))

    @ts.schedule
    def consumer(pipe):
        pipe.wait()
        pipe.read()
        pipe.release()

    blocked = task(0, consumer(pipe), "blocked")
    result = ts.check_all_interleavings([blocked])
    assert not result.is_safe
    assert result.deadlock_states[0].format_lines() == [
        "deadlock witness at cursors (0,): blocked tasks blocked",
        "blocked: ConsumerWait(pipe) on stage 0 waits for the full barrier to "
        "leave phase 0 (0 of 1 arrivals)",
    ]
    with pytest.raises(ValueError, match="0 of 1 arrivals"):
        ts.TaskManager([blocked], {})


def test_shared_state_is_a_function_of_the_cursors():
    pipe = Pipe(name="pipe", pipeline_config=async_config(2))

    @ts.schedule
    def producer(pipe):
        with ts.domain_loop(3):
            pipe.acquire()
            pipe.write()
            pipe.commit()

    @ts.schedule
    def consumer(pipe):
        with ts.domain_loop(3):
            pipe.wait()
            pipe.read()
            pipe.release()

    tasks = [
        task(0, producer(pipe), "producer"),
        task(1, consumer(pipe), "cons_a"),
        task(2, consumer(pipe), "cons_b"),
    ]
    compact = ts.check_all_interleavings(tasks)
    full = ts.check_all_interleavings(tasks, cursor_only_visited=False)
    assert compact.is_safe and full.is_safe
    assert compact.states_explored == full.states_explored
    assert compact.complete_count == full.complete_count


# ── UMMA operand liveness ─────────────────────────────────────────────────


def _signal(name):
    return Pipe(name=name, pipeline_config=async_config(1))


def test_umma_operand_hold_persists_past_release():
    operand = Pipe(
        name="R", pipeline_config=tma_umma_config(1), smem_requirements=[smem("r", 128)]
    )
    aliased = Pipe(
        name="S", pipeline_config=async_config(1), smem_requirements=[smem("s", 4)]
    )
    done = _signal("Q")

    @ts.schedule
    def producer(r):
        r.acquire()
        r.write()
        r.commit()

    @ts.schedule
    def mma(r, q):
        r.wait()
        r.read()
        r.release()
        q.acquire()
        q.commit()

    @ts.schedule
    def writer(q, s):
        q.wait()
        q.release()
        s.acquire()
        s.write()
        s.commit()

    tasks = [
        task(0, producer(operand), "Prod"),
        task(1, mma(operand, done), "Mma"),
        task(2, writer(done, aliased), "Writer"),
    ]
    result = ts.check_all_interleavings(tasks)
    assert not result.is_safe
    assert racing_resources(result) == {"R", "S"}


def test_async_consumer_hold_ends_at_release():
    operand = Pipe(
        name="R", pipeline_config=async_config(1), smem_requirements=[smem("r", 128)]
    )
    aliased = Pipe(
        name="S", pipeline_config=async_config(1), smem_requirements=[smem("s", 4)]
    )
    done = _signal("Q")

    @ts.schedule
    def producer(r):
        r.acquire()
        r.write()
        r.commit()

    @ts.schedule
    def consumer(r, q):
        r.wait()
        r.read()
        r.release()
        q.acquire()
        q.commit()

    @ts.schedule
    def writer(q, s):
        q.wait()
        q.release()
        s.acquire()
        s.write()
        s.commit()

    tasks = [
        task(0, producer(operand), "Prod"),
        task(1, consumer(operand, done), "Cons"),
        task(2, writer(done, aliased), "Writer"),
    ]
    result = ts.check_all_interleavings(tasks)
    assert result.is_safe
    assert result.complete_count >= 1


def test_next_acquire_releases_the_umma_operand_hold():
    operand = Pipe(
        name="R", pipeline_config=tma_umma_config(1), smem_requirements=[smem("r", 128)]
    )
    scratch = Pipe(name="S", smem_requirements=[smem("s", 4)])

    @ts.schedule
    def producer(r, s):
        r.acquire()
        r.write()
        r.commit()
        r.acquire()
        s.write()
        r.write()
        r.commit()

    @ts.schedule
    def mma(r):
        with ts.domain_loop(2):
            r.wait()
            r.read()
            r.release()

    tasks = [task(0, producer(operand, scratch), "Prod"), task(1, mma(operand), "Mma")]
    result = ts.check_all_interleavings(tasks)
    assert result.is_safe
    assert result.complete_count >= 1


def test_next_acquire_frees_every_consumers_hold():
    operand = Pipe(
        name="R", pipeline_config=tma_umma_config(1), smem_requirements=[smem("r", 128)]
    )
    scratch = Pipe(name="S", smem_requirements=[smem("s", 4)])

    @ts.schedule
    def producer(r, s):
        r.acquire()
        r.write()
        r.commit()
        r.acquire()
        s.write()
        r.write()
        r.commit()

    @ts.schedule
    def mma(r):
        with ts.domain_loop(2):
            r.wait()
            r.read()
            r.release()

    tasks = [
        task(0, producer(operand, scratch), "Prod"),
        task(1, mma(operand), "MmaA"),
        task(2, mma(operand), "MmaB"),
    ]
    result = ts.check_all_interleavings(tasks)
    assert result.is_safe
    assert result.complete_count >= 1


def test_umma_operand_hold_without_a_next_acquire_races():
    operand = Pipe(
        name="R", pipeline_config=tma_umma_config(1), smem_requirements=[smem("r", 128)]
    )
    scratch = Pipe(name="S", smem_requirements=[smem("s", 4)])
    done = _signal("Q")

    @ts.schedule
    def producer(r, q, s):
        r.acquire()
        r.write()
        r.commit()
        q.wait()
        q.release()
        s.write()

    @ts.schedule
    def mma(r, q):
        r.wait()
        r.read()
        r.release()
        q.acquire()
        q.commit()

    tasks = [
        task(0, producer(operand, done, scratch), "Prod"),
        task(1, mma(operand, done), "Mma"),
    ]
    result = ts.check_all_interleavings(tasks)
    assert not result.is_safe
    assert racing_resources(result) == {"R", "S"}


def _completion_pipeline(operand_stages, accumulator_stages):
    operand = Pipe(
        name="SmemAb",
        pipeline_config=tma_umma_config(operand_stages),
        smem_requirements=[smem("ab", 128)],
    )
    accumulator = Pipe(name="TmemC", pipeline_config=umma_async_config(accumulator_stages))
    return operand, accumulator


def test_completion_wait_frees_the_umma_operands():
    operand, accumulator = _completion_pipeline(1, 1)
    aliased = Pipe(
        name="S", pipeline_config=async_config(1), smem_requirements=[smem("s", 4)]
    )

    @ts.schedule
    def load(smem_ab):
        smem_ab.acquire()
        smem_ab.write()
        smem_ab.commit()

    @ts.schedule
    def mma(smem_ab, tmem_c):
        tmem_c.acquire()
        smem_ab.wait()
        smem_ab.read()
        smem_ab.release()
        tmem_c.commit()

    @ts.schedule
    def epilogue(tmem_c, s):
        tmem_c.wait()
        tmem_c.read()
        tmem_c.release()
        s.acquire()
        s.write()
        s.commit()

    tasks = [
        task(0, load(operand), "Load"),
        task(1, mma(operand, accumulator), "Mma"),
        task(2, epilogue(accumulator, aliased), "Epilogue"),
    ]
    result = ts.check_all_interleavings(tasks)
    assert result.is_safe
    assert result.complete_count >= 1


def test_aliased_write_before_the_completion_wait_races():
    operand, accumulator = _completion_pipeline(1, 1)
    aliased = Pipe(
        name="S", pipeline_config=async_config(1), smem_requirements=[smem("s", 4)]
    )
    released = _signal("Q")

    @ts.schedule
    def load(smem_ab):
        smem_ab.acquire()
        smem_ab.write()
        smem_ab.commit()

    @ts.schedule
    def mma(smem_ab, tmem_c, q):
        tmem_c.acquire()
        smem_ab.wait()
        smem_ab.read()
        smem_ab.release()
        q.acquire()
        q.commit()
        tmem_c.commit()

    @ts.schedule
    def writer(q, s):
        q.wait()
        q.release()
        s.acquire()
        s.write()
        s.commit()

    tasks = [
        task(0, load(operand), "Load"),
        task(1, mma(operand, accumulator, released), "Mma"),
        task(2, writer(released, aliased), "Writer"),
    ]
    result = ts.check_all_interleavings(tasks)
    assert not result.is_safe
    assert racing_resources(result) == {"SmemAb", "S"}


def test_single_completion_wait_frees_every_paired_stage():
    operand, accumulator = _completion_pipeline(2, 1)
    output = Pipe(name="GmemC", smem_requirements=[smem("c", 4)])

    @ts.schedule
    def load(smem_ab):
        with ts.domain_loop(2):
            smem_ab.acquire()
            smem_ab.write()
            smem_ab.commit()

    @ts.schedule
    def mma(smem_ab, tmem_c):
        tmem_c.acquire()
        with ts.domain_loop(2):
            smem_ab.wait()
            smem_ab.read()
            tmem_c.write()
            smem_ab.release()
        tmem_c.commit()

    @ts.schedule
    def epilogue(tmem_c, gmem_c):
        tmem_c.wait()
        tmem_c.read()
        gmem_c.write()
        tmem_c.release()

    tasks = [
        task(0, load(operand), "Load"),
        task(1, mma(operand, accumulator), "Mma"),
        task(2, epilogue(accumulator, output), "Epilogue"),
    ]
    result = ts.check_all_interleavings(tasks)
    assert result.is_safe
    assert result.complete_count >= 1


def test_partial_completion_wait_leaves_later_reads_live():
    operand, accumulator = _completion_pipeline(4, 2)
    output = Pipe(name="GmemC", smem_requirements=[smem("c", 4)])
    loaded = _signal("W")

    @ts.schedule
    def load(smem_ab, w):
        with ts.domain_loop(4):
            smem_ab.acquire()
            smem_ab.write()
            smem_ab.commit()
        w.acquire()
        w.commit()

    @ts.schedule
    def mma(smem_ab, tmem_c):
        with ts.domain_loop(2):
            tmem_c.acquire()
            smem_ab.wait()
            smem_ab.read()
            tmem_c.write()
            smem_ab.release()
            smem_ab.wait()
            smem_ab.read()
            tmem_c.write()
            smem_ab.release()
            tmem_c.commit()

    @ts.schedule
    def epilogue(w, tmem_c, gmem_c):
        w.wait()
        w.release()
        tmem_c.wait()
        tmem_c.read()
        gmem_c.write()
        tmem_c.release()

    tasks = [
        task(0, load(operand, loaded), "Load"),
        task(1, mma(operand, accumulator), "Mma"),
        task(2, epilogue(loaded, accumulator, output), "Epilogue"),
    ]
    result = ts.check_all_interleavings(tasks, early_exit=False)
    assert not result.is_safe
    assert racing_resources(result) == {"SmemAb", "GmemC"}


def test_completion_wait_frees_only_the_committing_tasks_reads():
    operand, accumulator = _completion_pipeline(1, 1)
    aliased = Pipe(
        name="S", pipeline_config=async_config(1), smem_requirements=[smem("s", 4)]
    )
    released = _signal("Q")

    @ts.schedule
    def load(smem_ab):
        smem_ab.acquire()
        smem_ab.write()
        smem_ab.commit()

    @ts.schedule
    def mma(smem_ab, tmem_c):
        tmem_c.acquire()
        smem_ab.wait()
        smem_ab.read()
        smem_ab.release()
        tmem_c.commit()

    @ts.schedule
    def other_mma(smem_ab, q):
        smem_ab.wait()
        smem_ab.read()
        smem_ab.release()
        q.acquire()
        q.commit()

    @ts.schedule
    def writer(q, tmem_c, s):
        q.wait()
        q.release()
        tmem_c.wait()
        tmem_c.read()
        tmem_c.release()
        s.acquire()
        s.write()
        s.commit()

    tasks = [
        task(0, load(operand), "Load"),
        task(1, mma(operand, accumulator), "Mma"),
        task(2, other_mma(operand, released), "OtherMma"),
        task(3, writer(released, accumulator, aliased), "Writer"),
    ]
    result = ts.check_all_interleavings(tasks)
    assert not result.is_safe
    assert racing_resources(result) == {"SmemAb", "S"}
