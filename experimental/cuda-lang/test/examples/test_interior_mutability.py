# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import torch
import cuda.lang as cl
from contextlib import contextmanager
from dataclasses import dataclass


@dataclass(frozen=True)
class Counter:
    memref: cl.Pointer
    count_by: int

    def iter(self):
        self.memref[0] += self.count_by

    @property
    def total(self):
        return self.memref[0]


@contextmanager
def count_by(n):
    '''
    The user must handle the allocation and the class is able to mutate the data
    held by the allocation.
    '''
    with cl.local_array(1, cl.int32) as arr:
        arr[0] = 0
        c = Counter(memref=arr.pointer(), count_by=n)
        yield c


def test_interior_mutability():
    @cl.kernel
    def kernel(m, n, out):
        with count_by(m) as c:
            for i in range(n):
                c.iter()
            out[0] = c.total

    out = torch.tensor([0], dtype=torch.int32).cuda(0)
    m, n = 3, 4
    cl.launch(torch.cuda.current_stream(), (1,), (1,), kernel, (m, n, out))
    torch.cuda.synchronize()
    assert out.cpu().tolist() == [m * n]


def test_block_mutex_example():
    @dataclass(frozen=True)
    class Mutex:
        # The pointer stays fixed, but its contents can change.
        storage: cl.Pointer

        def try_lock(self):
            return cl.atomic_rmw(
                "CAS", self.storage, 0, 1,
                memory_order="ACQUIRE", memory_scope=cl.MemoryScope.BLOCK,
            ) == 0

        def unlock(self):
            cl.atomic_store(
                self.storage, 0,
                memory_order="RELEASE", memory_scope=cl.MemoryScope.BLOCK,
            )

    @cl.kernel
    def kernel(out: cl.Array[int]):
        thread = cl.thread_index(0)
        mutex_storage = cl.shared_array(1, cl.int32)
        counter = cl.shared_array(1, cl.int32)
        if thread == 0:
            mutex_storage[0] = 0
            counter[0] = 0
        mutex = Mutex(mutex_storage.pointer())
        cl.barrier_sync_block_aligned()
        acquired = mutex.try_lock()
        if acquired:
            counter[0] += 1
            mutex.unlock()
        out[thread + 1] = cl.int32(acquired)
        cl.barrier_sync_block_aligned()
        if thread == 0:
            out[0] = counter[0]
            # All owners have released the mutex, so it must be available again.
            acquired = mutex.try_lock()
            out[cl.thread_count(0) + 1] = cl.int32(acquired)
            if acquired:
                mutex.unlock()

    stream = torch.cuda.current_stream()
    num_threads = 32
    out = torch.zeros(num_threads + 2, dtype=torch.int32).cuda(0)
    cl.launch(stream, (1,), (num_threads,), kernel, (out,))
    stream.synchronize()
    out = out.cpu().tolist()
    assert 0 < out[0] == sum(out[1:-1])
    assert out[-1] == 1
