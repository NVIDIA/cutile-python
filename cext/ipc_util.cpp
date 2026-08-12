// SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0

#include "ipc_util.h"

#include "memory.h"

#include <climits>


static constexpr uint32_t kSerializationHeader = 0x48454144;
static constexpr uint32_t kSerializationVersion = 1;

namespace {

struct IpcPayloadWriter {
    Vec<char> data;

    template <typename T>
    void write(T value) {
        size_t offset = data.size();
        data.resize(offset + sizeof(T));
        mem_copy(data.data() + offset, &value, sizeof(T));
    }

    template <typename T, typename Allocation>
    void write_vec(const Vec<T, Allocation>& vec) {
        write_span(vec.data(), vec.size());
    }

    template <typename T>
    void write_span(const T* span_data, size_t count) {
        write<size_t>(count);
        size_t nbytes = count * sizeof(T);
        if (nbytes == 0) return;
        size_t offset = data.size();
        data.resize(offset + nbytes);
        mem_copy(data.data() + offset, span_data, nbytes);
    }
};

struct IpcPayloadReader {
    const char* data;
    size_t size;
    size_t offset;

    template <typename T>
    Status read(const char* field_name, T* out) {
        if (sizeof(T) > size - offset) {
            return raise(PyExc_ValueError,
                         "Truncated IPC benchmark payload while reading ", field_name);
        }
        mem_copy(out, data + offset, sizeof(T));
        offset += sizeof(T);
        return OK;
    }

    template <typename T, typename Allocation>
    Status read_vec(const char* field_name, Vec<T, Allocation>* out) {
        size_t count;
        if (!read(field_name, &count)) return ErrorRaised;
        if (count > SIZE_MAX / sizeof(T)) {
            return raise(PyExc_OverflowError,
                         "IPC benchmark payload ", field_name, " is too large");
        }
        size_t nbytes = count * sizeof(T);
        if (nbytes > size - offset) {
            return raise(PyExc_ValueError,
                         "Truncated IPC benchmark payload while reading ", field_name);
        }
        out->resize(count);
        if (nbytes)
            mem_copy(out->data(), data + offset, nbytes);
        offset += nbytes;
        return OK;
    }

    Status check_finished() const {
        if (offset != size)
            return raise(PyExc_ValueError, "IPC benchmark payload has trailing bytes");
        return OK;
    }
};
}

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
                                      size_t symbol_size) {
    IpcPayloadWriter writer;
    writer.write<uint32_t>(kSerializationHeader);
    writer.write<uint32_t>(kSerializationVersion);
    for (size_t i = 0; i < 3; ++i)
        writer.write<uint32_t>(grid_dims[i]);
    writer.write<uint32_t>(static_cast<uint32_t>(device_id));
    writer.write<uint32_t>(static_cast<uint32_t>(dynamic_smem_bytes));

    writer.write_vec(arena);
    writer.write_vec(cuarg_offsets);
    writer.write_vec(list_args);
    writer.write<size_t>(total_list_data_size_words);
    writer.write_vec(arena_array_ptrs);
    writer.write_vec(ipc_mem_handles);
    writer.write_span(cubin, cubin_size);
    writer.write_span(symbol, symbol_size);

    return steal(PyBytes_FromStringAndSize(writer.data.data(), writer.data.size()));
}

Result<IpcBenchmarkPayload> deserialize_ipc_benchmark_payload(const char* data,
                                                              size_t nbytes,
                                                              LaunchHelper& helper) {
    IpcPayloadReader reader{data, nbytes, 0};

    uint32_t header;
    if (!reader.read("serialization_header", &header)) return ErrorRaised;
    if (header != kSerializationHeader)
        return raise(PyExc_ValueError, "Invalid IPC benchmark payload header");

    uint32_t version;
    if (!reader.read("serialization_version", &version)) return ErrorRaised;
    if (version != kSerializationVersion)
        return raise(PyExc_ValueError, "Unsupported IPC benchmark payload version");

    IpcBenchmarkPayload payload = {};
    for (size_t i = 0; i < 3; ++i) {
        if (!reader.read("grid", &payload.grid_dims[i])) return ErrorRaised;
    }

    uint32_t py_device_id;
    if (!reader.read("device_id", &py_device_id)) return ErrorRaised;
    payload.device_id = static_cast<int>(py_device_id);

    uint32_t dynamic_smem_bytes;
    if (!reader.read("dynamic_smem_bytes", &dynamic_smem_bytes)) return ErrorRaised;
    payload.dynamic_smem_bytes = static_cast<unsigned>(dynamic_smem_bytes);

    if (!reader.read_vec("arena", &helper.arena)) return ErrorRaised;
    if (!reader.read_vec("cuarg_offsets", &helper.cuarg_offsets)) return ErrorRaised;
    if (!reader.read_vec("list_args", &helper.list_args)) return ErrorRaised;
    if (!reader.read("total_list_data_size_words",
                     &helper.total_list_data_size_words))
        return ErrorRaised;
    if (!reader.read_vec("arena_array_ptrs", &payload.arena_array_ptrs))
        return ErrorRaised;
    if (!reader.read_vec("ipc_mem_handles", &payload.ipc_mem_handles))
        return ErrorRaised;
    if (!reader.read_vec("cubin", &payload.cubin))
        return ErrorRaised;
    if (!reader.read_vec("symbol", &payload.symbol))
        return ErrorRaised;
    if (!reader.check_finished()) return ErrorRaised;

    return payload;
}
