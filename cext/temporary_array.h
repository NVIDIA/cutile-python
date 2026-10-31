// SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>

template <typename T, size_t SmallCapacity>
class TemporaryArray {
    T small_[SmallCapacity];
    std::unique_ptr<T[]> large_;

    TemporaryArray(TemporaryArray&&) = delete;
    void operator=(TemporaryArray&&) = delete;

public:
    TemporaryArray() = default;

     T* allocate(size_t len) {
        if (len <= std::extent_v<decltype(small_)>) {
            return small_;
        } else {
            large_.reset(new T[len]);
            return large_.get();
        }
    }
};

