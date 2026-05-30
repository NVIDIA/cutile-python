/*
 * SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "py.h"
#include <cuda.h>

#define FOREACH_CUDA_FUNCTION_TO_LOAD(X) \
    X(cuInit, 2000) \
    X(cuLibraryLoadData, 12000) \
    X(cuLibraryUnload, 12000) \
    X(cuLibraryGetKernel, 12000) \
    X(cuGetErrorString, 6000) \
    X(cuLaunchKernel, 7000) \
    X(cuLaunchKernelEx, 11060) \
    X(cuPointerGetAttribute, 4000) \
    X(cuCtxSynchronize, 2000) \
    X(cuCtxPushCurrent, 4000) \
    X(cuCtxPopCurrent, 4000) \
    X(cuCtxGetCurrent, 4000) \
    X(cuCtxSetCurrent, 4000) \
    X(cuCtxGetDevice, 2000) \
    X(cuCtxGetId, 12000) \
    X(cuDeviceGet, 2000) \
    X(cuDeviceGetAttribute, 2000) \
    X(cuDevicePrimaryCtxRetain, 7000) \
    X(cuDriverGetVersion, 2020) \
    X(cuEventCreate, 2000) \
    X(cuEventDestroy, 2000) \
    X(cuEventQuery, 2000) \
    X(cuEventRecord, 2000) \
    X(cuEventSynchronize, 2000) \
    X(cuKernelGetFunction, 12000) \
    X(cuMemAlloc, 3020) \
    X(cuMemAllocHost, 3020) \
    X(cuMemFree, 3020) \
    X(cuMemFreeHost, 2000) \
    X(cuMemcpyHtoDAsync, 3020) \
    X(cuStreamCreate, 2000) \
    X(cuStreamDestroy, 4000) \
    X(cuStreamGetCtx, 9020) \
    X(cuStreamGetId, 12000) \
    X(cuStreamIsCapturing, 10000) \
    X(cuStreamSynchronize, 7000) \
    X(cuStreamWaitEvent, 7000) \
    X(cuEventElapsedTime, 12080) \
    X(cuGraphCreate, 10000) \
    X(cuGraphDestroy, 10000) \
    X(cuGraphAddEventRecordNode, 11010) \
    X(cuGraphAddKernelNode, 12000) \
    X(cuGraphAddMemsetNode, 10000) \
    X(cuGraphAddMemAllocNode, 11040) \
    X(cuGraphAddMemFreeNode, 11040) \
    X(cuGraphInstantiateWithFlags, 11040) \
    X(cuGraphExecDestroy, 10000) \
    X(cuGraphLaunch, 10000) \
    X(cuTensorMapEncodeTiled, 12000)


#define DECLARE_CUDA_FUNC_EXTERN(name, _cuda_version) \
    decltype(::name)* name;

struct DriverApi {
    FOREACH_CUDA_FUNCTION_TO_LOAD(DECLARE_CUDA_FUNC_EXTERN)
};

Result<const DriverApi*> get_driver_api();


class CudaGraph {
    const DriverApi* d;
    CUgraph graph;
public:
    CudaGraph(const CudaGraph&) = delete;
    void operator=(const CudaGraph&) = delete;

    explicit CudaGraph(const DriverApi* d) : d(d), graph(nullptr) {}

    CUresult create() {
        CHECK(!graph);
        return d->cuGraphCreate(&graph, 0);
    }

    CUgraph get() const {
        return graph;
    }

    ~CudaGraph() {
        if (graph) d->cuGraphDestroy(graph);
    }
};

class CudaGraphExec {
    const DriverApi* d;
    CUgraphExec exec;
public:
    CudaGraphExec(const CudaGraphExec&) = delete;
    void operator=(const CudaGraphExec&) = delete;

    explicit CudaGraphExec(const DriverApi* d) : d(d), exec(nullptr) {}

    CUresult instantiate(const CudaGraph& graph) {
        CHECK(!exec);
        return d->cuGraphInstantiateWithFlags(&exec, graph.get(), 0);
    }

    CUgraphExec get() const {
        return exec;
    }

    ~CudaGraphExec() {
        if (exec) d->cuGraphExecDestroy(exec);
    }
};

class CudaEvent {
    const DriverApi* d;
    CUevent event;
public:
    CudaEvent(const CudaEvent&) = delete;
    void operator=(const CudaEvent&) = delete;

    explicit CudaEvent(const DriverApi* d) : d(d), event(nullptr) {}

    CUresult create() {
        CHECK(!event);
        return d->cuEventCreate(&event, CU_EVENT_DEFAULT);
    }

    CUevent get() const {
        return event;
    }

    ~CudaEvent() {
        if (event) d->cuEventDestroy(event);
    }
};
