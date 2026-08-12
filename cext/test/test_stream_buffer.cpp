// SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0

#include "../stream_buffer.cpp"
#include "../check.h"
#include "../py.h"

#include <algorithm>
#include <iostream>


static bool hold_events = false;

static CUresult (*orig_cuEventQuery)(CUevent);

static CUresult patched_cuEventQuery(CUevent ev) {
    if (hold_events) return CUDA_ERROR_NOT_READY;
    return orig_cuEventQuery(ev);
}

DriverApi g_driver;

static void run_transaction(StreamBufferPool* pool, CUstream stream, size_t size, size_t reps = 1) {
    for (size_t i = 0; i < reps; ++i) {
        CHECK(g_driver.cuStreamSynchronize(stream) == CUDA_SUCCESS);
        StreamBufferTransaction tx = stream_buffer_transaction_open(&g_driver, pool, stream);
        DualPointer ptr = tx.allocate(size);
        CHECK(ptr.device);
        CHECK(ptr.host);
    }
}

static StreamBuffer* find_stream_buffer(StreamBufferPool* pool, CUstream s) {
    unsigned long long stream_id;
    CHECK(g_driver.cuStreamGetId(s, &stream_id) == CUDA_SUCCESS);
    auto it = std::find(pool->stream_ids.begin(), pool->stream_ids.end(), stream_id);
    CHECK(it != pool->stream_ids.end());
    return pool->stream_buffers[it - pool->stream_ids.begin()];
}

static size_t list_len(Chunk* head) {
    size_t ret = 0;
    while (head) {
        ++ret;
        head = head->next;
    }
    return ret;
}

static size_t sb_len(StreamBufferPool* pool, CUstream s) {
    return list_len(find_stream_buffer(pool, s)->head);
}

static cuGetProcAddress_v2_t load_libcuda() {
    PyPtr ctypes = steal(PyImport_ImportModule("ctypes"));
    CHECK(ctypes);

#ifdef _WIN32
    const char* name = "nvcuda.dll";
#else
    const char* name = "libcuda.so.1";
#endif

    static PyObject* dll;
    if (!dll) {
        dll = PyObject_CallMethod(ctypes.get(), "CDLL", "(s)", name);
        CHECK(dll);
    }

    PyPtr symbol = to_pyunicode("cuGetProcAddress_v2");
    CHECK(symbol);

    PyPtr func = steal(PyObject_GetItem(dll, symbol.get()));
    CHECK(func);

    PyPtr pp = steal(PyObject_CallMethod(ctypes.get(), "addressof", "(O)", func.get()));
    CHECK(pp);

    cuGetProcAddress_v2_t* cuGetProcAddr_pp = reinterpret_cast<cuGetProcAddress_v2_t*>(
            PyLong_AsSize_t(pp.get()));
    CHECK(!PyErr_Occurred());
    CHECK(cuGetProcAddr_pp);
    return *cuGetProcAddr_pp;
}

int main() {
    // Initialize Pyhton & CUDA
    Py_Initialize();
    CHECK(driver_api_init(&g_driver, load_libcuda()));
    CHECK(g_driver.cuInit(0) == CUDA_SUCCESS);
    // Patch cuEventQuery() so that we can simulate uncompleted events
    orig_cuEventQuery = g_driver.cuEventQuery;
    g_driver.cuEventQuery = patched_cuEventQuery;

    // Create & activate a CUDA context
    CUdevice dev;
    CHECK(g_driver.cuDeviceGet(&dev, 0) == CUDA_SUCCESS);
    CUcontext ctx;
    CHECK(g_driver.cuDevicePrimaryCtxRetain(&ctx, dev) == CUDA_SUCCESS);
    CHECK(g_driver.cuCtxSetCurrent(ctx) == CUDA_SUCCESS);

    // Create up a StreamBufferPool
    StreamBufferPool* pool = stream_buffer_pool_new();
    CHECK(pool);

    // Create a stream
    CUstream stream;
    CHECK(g_driver.cuStreamCreate(&stream, 0) == CUDA_SUCCESS);

    // Allocate a "small" block of memory. This shouldn't fill a chunk.
    static constexpr size_t kSmallRequest = kInitialChunkCapacity / 16;
    static_assert(kSmallRequest < kInitialChunkCapacity / kMinChunkToAllocationRatio);
    run_transaction(pool, stream, kSmallRequest);
    CHECK(sb_len(pool, stream) == 1);
    CHECK(list_len(pool->chunk_freelist) == 0);
    hold_events = true;

    // Allocate a bunch of small blocks to exhaust our initial chunk
    run_transaction(pool, stream, kSmallRequest, 17);
    CHECK(sb_len(pool, stream) == 2);
    CHECK(list_len(pool->chunk_freelist) == 0);

    // Fill two more chunks. We can't reclaim any chunks yet because the work
    // submitted by the very first transaction hasn't been completed
    // (this is simulated by `hold_events`);
    run_transaction(pool, stream, kSmallRequest, 32);
    CHECK(sb_len(pool, stream) == 4);
    CHECK(list_len(pool->chunk_freelist) == 0);

    // Pretend that all the work has been done now and fill another chunk.
    // We should be able to reclaim the first three chunks.
    hold_events = false;
    run_transaction(pool, stream, kSmallRequest, 16);
    CHECK(sb_len(pool, stream) == 1);
    CHECK(list_len(pool->chunk_freelist) == 3);

    // Run many transactions to make sure we can maintain a steady state
    for (int i = 0; i < 100; ++i)
        run_transaction(pool, stream, kSmallRequest, 16);
    CHECK(sb_len(pool, stream) == 1);
    CHECK(list_len(pool->chunk_freelist) == 3);

    // Now create another stream and start using it exclusively for a while
    CUstream other_stream;
    CHECK(g_driver.cuStreamCreate(&other_stream, 0) == CUDA_SUCCESS);

    // At first, both streams should be kept in the pool
    run_transaction(pool, other_stream, kSmallRequest);
    CHECK(pool->stream_ids.size() == 2);

    // Exhaust all free chunks
    hold_events = true;
    run_transaction(pool, other_stream, kSmallRequest, 16 * 5);
    CHECK(sb_len(pool, stream) == 1);
    CHECK(sb_len(pool, other_stream) == 6);
    CHECK(list_len(pool->chunk_freelist) == 0);
    hold_events = false;

    // Run another series of transactions to trigger garbage collection
    run_transaction(pool, other_stream, kSmallRequest, 16);

    // The original stream must be garbage collected by now
    CHECK(pool->stream_ids.size() == 1);
    CHECK(sb_len(pool, other_stream) == 1);
    CHECK(list_len(pool->chunk_freelist) == 6);

    // Now run a transaction on the original stream: it should be inserted back
    run_transaction(pool, stream, kSmallRequest, 1);
    CHECK(pool->stream_ids.size() == 2);
    CHECK(sb_len(pool, stream) == 1);
    CHECK(sb_len(pool, other_stream) == 1);
    CHECK(list_len(pool->chunk_freelist) == 5);

    // Now request a bigger buffer so that we increase the chunk size.
    // All the existing chunks from the freelist should be deleted.
    static constexpr size_t kLargeRequest = kInitialChunkCapacity;
    run_transaction(pool, stream, kLargeRequest, 1);
    CHECK(list_len(pool->chunk_freelist) == 0);

    // There should still be an unreclaimed chunk of previous size
    // hanging around on the second stream.
    CHECK(sb_len(pool, other_stream) == 1);

    // Run a bunch of transactions to make sure it gets garbage collected
    hold_events = true;
    run_transaction(pool, stream, kLargeRequest, kMinChunkToAllocationRatio);
    hold_events = false;
    run_transaction(pool, stream, kLargeRequest, kMinChunkToAllocationRatio);
    CHECK(pool->stream_ids.size() == 1);
    CHECK(list_len(pool->chunk_freelist) == 1);
    return 0;
}


