// SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "check.h"

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


// A pointer that is set at most once and is never released afterwards. Multiple threads may
// race to compute the value: the first set() wins, and the other values are released.
// Accepts owning smart pointers, such as RefPtr<T> or std::unique_ptr<T>.
template <typename T>
class OncePointer {
    T* ptr_ = nullptr;  // Accessed atomically

public:
    // Returns a borrowed reference, or nullptr if not set yet
    T* get() const {
        return atomic_load_acquire(&ptr_);
    }

    // Stores `value` if not set yet. Otherwise, releases `value`.
    // Returns a borrowed reference to the stored value.
    template <typename Ptr>
    T* set(Ptr value) {
        CHECK(value);
        T* expected = nullptr;
        if (atomic_compare_exchange_acq_rel(&ptr_, &expected, value.get()))
            return value.release();
        return expected;
    }
};
