// SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>

template <typename T, size_t SmallCapacity>
class TemporaryArray {
    union {
        T small_[SmallCapacity];
    };
    T* large_;

    TemporaryArray(TemporaryArray&&) = delete;
    void operator=(TemporaryArray&&) = delete;

public:
    static_assert(std::is_trivially_destructible_v<T>);

    TemporaryArray() : large_(nullptr) {}

    ~TemporaryArray() {
        mem_free(large_);
    }

    T* allocate(size_t len) {
        if (len <= std::extent_v<decltype(small_)>)
            return small_;
        else
            return (large_ = xcalloc<T>(len));
    }
};

