// SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#if defined(_MSC_VER)
#include <atomic>
#endif


template <typename T>
static inline T atomic_load_acquire(const T* ptr) {
#if defined(_MSC_VER)
    // atomic_ref<const T> is not allowed in C++20
    return std::atomic_ref<T>(*const_cast<T*>(ptr)).load(std::memory_order_acquire);
#else
    return __atomic_load_n(ptr, __ATOMIC_ACQUIRE);
#endif
}

template <typename T>
static inline void atomic_store_release(T* ptr, T value) {
#if defined(_MSC_VER)
    std::atomic_ref<T>(*ptr).store(value, std::memory_order_release);
#else
    __atomic_store_n(ptr, value, __ATOMIC_RELEASE);
#endif
}

// If `*ptr` is equal to `*expected`, replaces it with `desired` and returns true.
// Otherwise, stores the current value of `*ptr` into `*expected` and returns false.
template <typename T>
static inline bool atomic_compare_exchange_acq_rel(T* ptr, T* expected, T desired) {
#if defined(_MSC_VER)
    return std::atomic_ref<T>(*ptr).compare_exchange_strong(
            *expected, desired, std::memory_order_acq_rel, std::memory_order_acquire);
#else
    return __atomic_compare_exchange_n(ptr, expected, desired, false,
                                       __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE);
#endif
}
