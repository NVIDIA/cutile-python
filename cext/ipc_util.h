// SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "cuda_helper.h"
#include "cuda_loader.h"
#include "hash_map.h"
#include "launch_helper.h"
#include "py.h"
#include "vec.h"

#include <cuda.h>

#include <cstdint>
#include <cstddef>


struct IpcDevicePtrRef {
    size_t ipc_mem_handle_index;
    uint64_t offset;
};

struct IpcArrayPtrPatch {
    ArenaOffset arena_offset;
    IpcDevicePtrRef array_ptr;
};

struct IpcBenchmarkPayload {
    uint32_t grid_dims[3];
    int device_id;
    unsigned dynamic_smem_bytes;
    Vec<IpcArrayPtrPatch> arena_array_ptrs;
    Vec<CUipcMemHandle> ipc_mem_handles;
    Vec<char> cubin;
    Vec<char> symbol;
};

// Used in main process to get shareable IPC memory handles.
struct IpcHandleExporter {
    Vec<CUipcMemHandle> ipc_mem_handles;

    explicit IpcHandleExporter(const DriverApi* driver) : driver(driver) {}

    Result<bool> check_ipc_supported(CUdevice device) {
        int supported = 0;
        CUresult res = driver->cuDeviceGetAttribute(
                &supported, CU_DEVICE_ATTRIBUTE_IPC_EVENT_SUPPORTED, device);
        if (res != CUDA_SUCCESS) {
            return raise(PyExc_RuntimeError, "cuDeviceGetAttribute: ",
                         get_cuda_error(driver, res));
        }
        return supported != 0;
    }

    Result<bool> is_legacy_ipc_capable(CUdeviceptr dptr) {
        if (!dptr)
            return false;

        int capable = 0;
        CUresult res = driver->cuPointerGetAttribute(
                &capable, CU_POINTER_ATTRIBUTE_IS_LEGACY_CUDA_IPC_CAPABLE, dptr);
        if (res != CUDA_SUCCESS) {
            // This is a real pointer/context/driver error, not an unsupported
            // allocation that should silently fall back.
            return raise(
                    PyExc_RuntimeError,
                    "cuPointerGetAttribute(IS_LEGACY_CUDA_IPC_CAPABLE): ",
                    get_cuda_error(driver, res));
        }

        return capable != 0;
    }

    Result<IpcDevicePtrRef> get_ipc_pointer(CUdeviceptr dptr) {
        if (!dptr)
            return raise(PyExc_ValueError, "Cannot export a null pointer through CUDA IPC");

        CUdeviceptr base = 0;
        size_t allocation_size = 0;
        CUresult res = driver->cuMemGetAddressRange(&base, &allocation_size, dptr);
        if (res != CUDA_SUCCESS) {
            return raise(PyExc_RuntimeError, "cuMemGetAddressRange: ",
                         get_cuda_error(driver, res));
        }
        if (dptr < base)
            return raise(PyExc_RuntimeError, "Invalid CUDA allocation range");

        size_t handle_index;
        HashMap<CUdeviceptr, size_t>::Item* existing_handle_index =
                ipc_mem_handle_index_by_base.find(base);
        if (existing_handle_index) {
            handle_index = existing_handle_index->value;
        } else {
            CUipcMemHandle handle;
            res = driver->cuIpcGetMemHandle(&handle, base);
            if (res != CUDA_SUCCESS) {
                return raise(PyExc_RuntimeError, "cuIpcGetMemHandle: ",
                             get_cuda_error(driver, res));
            }
            handle_index = ipc_mem_handles.size();
            ipc_mem_handles.push_back(handle);
            ipc_mem_handle_index_by_base.insert(base, handle_index);
        }

        return IpcDevicePtrRef{handle_index, static_cast<uint64_t>(dptr - base)};
    }

private:
    const DriverApi* driver;
    HashMap<CUdeviceptr, size_t> ipc_mem_handle_index_by_base;
};

// Used in sub-process to open IPC memory handles and manage their lifecycle.
struct IpcHandleCreator {
    const DriverApi* driver;
    Vec<CUdeviceptr> mapped_handles;

    explicit IpcHandleCreator(const DriverApi* driver) : driver(driver) {}

    Status open_handles(const Vec<CUipcMemHandle>& ipc_mem_handles) {
        CHECK(mapped_handles.empty());
        mapped_handles.reserve(ipc_mem_handles.size());
        for (const CUipcMemHandle& ipc_mem_handle : ipc_mem_handles) {
            CUdeviceptr mapped_ptr;
            CUresult res = driver->cuIpcOpenMemHandle(
                    &mapped_ptr, ipc_mem_handle, CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS);
            if (res != CUDA_SUCCESS) {
                return raise(PyExc_RuntimeError, "cuIpcOpenMemHandle: ",
                             get_cuda_error(driver, res));
            }
            mapped_handles.push_back(mapped_ptr);
        }
        return OK;
    }

    IpcHandleCreator(const IpcHandleCreator&) = delete;
    void operator=(const IpcHandleCreator&) = delete;

    ~IpcHandleCreator() {
        for (CUdeviceptr ptr : mapped_handles) {
            CUresult res = driver->cuIpcCloseMemHandle(ptr);
            CHECK(res == CUDA_SUCCESS);
        }
    }
};

PyPtr serialize_ipc_benchmark_payload(const uint32_t grid_dims[3],
                                      int device_id,
                                      unsigned dynamic_smem_bytes,
                                      const Arena& arena,
                                      const Vec<ArenaOffset>& cuarg_offsets,
                                      const Vec<ListArg>& list_args,
                                      size_t total_list_data_size_words,
                                      const Vec<IpcArrayPtrPatch>& arena_array_ptrs,
                                      const Vec<CUipcMemHandle>& ipc_mem_handles,
                                      const char* cubin,
                                      size_t cubin_size,
                                      const char* symbol,
                                      size_t symbol_size);

Result<IpcBenchmarkPayload> deserialize_ipc_benchmark_payload(const char* data,
                                                              size_t nbytes,
                                                              LaunchHelper& helper);
