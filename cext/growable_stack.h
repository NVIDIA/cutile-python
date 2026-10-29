// SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include "memory.h"

class GrowableStack {
public:
    struct Watermark {
        size_t chunk_idx;
        size_t chunk_available;
    };

private:
    static constexpr size_t kLog2InitialCapacity = 14;  // 16 KB
    // Since chunks quadruple in size, this puts the largest one at 2^(14 + 2*15) = 2^44 bytes,
    // i.e. 16 TB.
    static constexpr size_t kMaxChunks = 16;

    std::unique_ptr<char[]> chunks_[kMaxChunks];
    Watermark cur_ = {SIZE_MAX, 0};

    char* _allocate(size_t bytes, size_t alignment_mask) {
        ensure_available(bytes);
        cur_.chunk_available -= bytes;
        cur_.chunk_available &= ~alignment_mask;
        return chunks_[cur_.chunk_idx].get() + cur_.chunk_available;
    }

    static inline constexpr size_t chunk_capacity(size_t chunk_idx) {
        return size_t{1} << (kLog2InitialCapacity + chunk_idx * 2);
    }

    void ensure_available(size_t bytes) {
        if (bytes > cur_.chunk_available) {
            for (size_t i = cur_.chunk_idx + 1; i < kMaxChunks; ++i) {
                size_t new_capacity = chunk_capacity(i);
                if (bytes <= new_capacity) {
                    if (!chunks_[i]) chunks_[i].reset(new char[new_capacity]);
                    cur_.chunk_idx = i;
                    cur_.chunk_available = new_capacity;
                    return;
                }
            }
            CHECK_UNREACHABLE;
        }
    }

public:
    Watermark current_watermark() const {
        return cur_;
    }

    void reset_to(Watermark watermark) {
        cur_ = watermark;
    }

    void clear() {
        cur_ = {SIZE_MAX, 0};
    }

    template <typename T>
    T* allocate(size_t count) {
        static_assert(alignof(T) <= 16);
        if (!count) count = 1;  // Allocate unique addresses for zero-sized requests
        size_t size = count * sizeof(T);
        CHECK(size / sizeof(T) == count);
        return reinterpret_cast<T*>(_allocate(size, alignof(T) - 1));
    }

    template <typename T, typename... Args>
    T* emplace(Args&&... args) {
        T* ret = allocate<T>(1);
        new (ret) T(std::forward<Args>(args)...);
        return ret;
    }

    template <typename T>
    T* allocate_zeroed(size_t count) {
        T* ret = allocate<T>(count);
        mem_zero(ret, sizeof(T) * count);
        return ret;
    }
};


