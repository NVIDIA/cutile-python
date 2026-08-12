/*
 * SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "py.h"
#include <cuda.h>

#define FOREACH_CUDA_FUNCTION_TO_LOAD(X) \
    X(cuInit, "cuInit", 2000) \
    X(cuLibraryLoadData, "cuLibraryLoadData", 12000) \
    X(cuLibraryUnload, "cuLibraryUnload", 12000) \
    X(cuLibraryGetKernel, "cuLibraryGetKernel", 12000) \
    X(cuGetErrorString, "cuGetErrorString", 6000) \
    X(cuLaunchKernel, "cuLaunchKernel", 7000) \
    X(cuLaunchKernelEx, "cuLaunchKernelEx", 11060) \
    X(cuPointerGetAttribute, "cuPointerGetAttribute", 4000) \
    X(cuCtxSynchronize, "cuCtxSynchronize", 2000) \
    X(cuCtxPushCurrent, "cuCtxPushCurrent", 4000) \
    X(cuCtxPopCurrent, "cuCtxPopCurrent", 4000) \
    X(cuCtxGetCurrent, "cuCtxGetCurrent", 4000) \
    X(cuCtxSetCurrent, "cuCtxSetCurrent", 4000) \
    X(cuCtxGetDevice, "cuCtxGetDevice", 2000) \
    X(cuCtxGetDevice_v2, "cuCtxGetDevice", 13000) \
    X(cuCtxGetId, "cuCtxGetId", 12000) \
    X(cuDeviceGet, "cuDeviceGet", 2000) \
    X(cuDeviceGetCount, "cuDeviceGetCount", 2000) \
    X(cuDeviceGetAttribute, "cuDeviceGetAttribute", 2000) \
    X(cuDevicePrimaryCtxRetain, "cuDevicePrimaryCtxRetain", 7000) \
    X(cuDriverGetVersion, "cuDriverGetVersion", 2020) \
    X(cuEventCreate, "cuEventCreate", 2000) \
    X(cuEventDestroy, "cuEventDestroy", 2000) \
    X(cuEventQuery, "cuEventQuery", 2000) \
    X(cuEventRecord, "cuEventRecord", 2000) \
    X(cuKernelGetFunction, "cuKernelGetFunction", 12000) \
    X(cuKernelGetAttribute, "cuKernelGetAttribute", 12000) \
    X(cuKernelSetAttribute, "cuKernelSetAttribute", 12000) \
    X(cuMemAlloc, "cuMemAlloc", 3020) \
    X(cuMemAllocHost, "cuMemAllocHost", 3020) \
    X(cuMemFree, "cuMemFree", 3020) \
    X(cuMemFreeHost, "cuMemFreeHost", 2000) \
    X(cuMemGetAddressRange, "cuMemGetAddressRange", 3020) \
    X(cuIpcGetMemHandle, "cuIpcGetMemHandle", 4010) \
    X(cuIpcOpenMemHandle, "cuIpcOpenMemHandle", 4010) \
    X(cuIpcCloseMemHandle, "cuIpcCloseMemHandle", 4010) \
    X(cuMemcpyHtoDAsync, "cuMemcpyHtoDAsync", 3020) \
    X(cuStreamCreate, "cuStreamCreate", 2000) \
    X(cuStreamDestroy, "cuStreamDestroy", 4000) \
    X(cuStreamGetCtx, "cuStreamGetCtx", 9020) \
    X(cuStreamGetId, "cuStreamGetId", 12000) \
    X(cuStreamIsCapturing, "cuStreamIsCapturing", 10000) \
    X(cuStreamSynchronize, "cuStreamSynchronize", 7000) \
    X(cuStreamWaitEvent, "cuStreamWaitEvent", 7000) \
    X(cuEventElapsedTime, "cuEventElapsedTime", 12080) \
    X(cuGraphCreate, "cuGraphCreate", 10000) \
    X(cuGraphDestroy, "cuGraphDestroy", 10000) \
    X(cuGraphAddEventRecordNode, "cuGraphAddEventRecordNode", 11010) \
    X(cuGraphAddKernelNode, "cuGraphAddKernelNode", 12000) \
    X(cuGraphAddMemsetNode, "cuGraphAddMemsetNode", 10000) \
    X(cuGraphAddMemAllocNode, "cuGraphAddMemAllocNode", 11040) \
    X(cuGraphAddMemFreeNode, "cuGraphAddMemFreeNode", 11040) \
    X(cuGraphInstantiateWithFlags, "cuGraphInstantiateWithFlags", 11040) \
    X(cuGraphExecDestroy, "cuGraphExecDestroy", 10000) \
    X(cuGraphLaunch, "cuGraphLaunch", 10000) \
    X(cuTensorMapEncodeTiled, "cuTensorMapEncodeTiled", 12000)


#define DECLARE_CUDA_FUNC_EXTERN(name, _key, _cuda_version) \
    decltype(::name)* name;

struct DriverApi {
    FOREACH_CUDA_FUNCTION_TO_LOAD(DECLARE_CUDA_FUNC_EXTERN)
};


typedef CUresult (*cuGetProcAddress_v2_t)
    (const char *symbol, void **funcPtr, int cudaVersion,
     cuuint64_t flags, CUdriverProcAddressQueryResult *symbolStatus);

Status driver_api_init(DriverApi* driver_api, cuGetProcAddress_v2_t _cuGetProcAddress);

Result<const DriverApi*> get_driver_api();


class CudaContextGuard {
    const DriverApi* driver;
    bool need_to_pop = false;
public:
    CudaContextGuard(const CudaContextGuard&) = delete;
    void operator=(const CudaContextGuard&) = delete;

    explicit CudaContextGuard(const DriverApi* driver) : driver(driver) {}

    Status switch_to(CUcontext target);

    ~CudaContextGuard();
};


// RAII wrapper around a CUDA library loaded into a context.
class CudaLibrary {
public:
    explicit CudaLibrary(const DriverApi* driver, CUlibrary lib);
    CudaLibrary(CudaLibrary&& other);
    CudaLibrary(const CudaLibrary&) = delete;
    void operator=(const CudaLibrary&) = delete;
    ~CudaLibrary();

    const CUlibrary& get() const;

private:
    const DriverApi* driver_;
    CUlibrary lib_;
};


struct CudaKernel {
    CudaLibrary lib;
    CUkernel kernel;
};


Result<CudaKernel> load_cuda_kernel(
        const DriverApi* driver,
        const char* cubin_data,
        size_t cubin_size,
        const char* func_name);


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
