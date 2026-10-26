/*
 * SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>

#include "atomics.h"
#include "check.h"
#include "hash_map.h"

#if defined(Py_GIL_DISABLED) && defined(_MSC_VER)
#include <atomic>
#endif


// A reference count that starts at 1. In a free-threaded build, it is updated atomically.
struct ReferenceCounter {
    ReferenceCounter() = default;

    ReferenceCounter(const ReferenceCounter&) = delete;
    void operator=(const ReferenceCounter&) = delete;

    void increment() {
#if !defined(Py_GIL_DISABLED)
        ++count_;
#elif defined(_MSC_VER)
        std::atomic_ref<size_t>(count_).fetch_add(1, std::memory_order_relaxed);
#else
        __atomic_fetch_add(&count_, 1, __ATOMIC_RELAXED);
#endif
    }

    // Returns true if the count has dropped to zero
    [[nodiscard]] bool decrement() {
#if !defined(Py_GIL_DISABLED)
        size_t old = count_--;
#elif defined(_MSC_VER)
        size_t old = std::atomic_ref<size_t>(count_).fetch_sub(1, std::memory_order_acq_rel);
#else
        size_t old = __atomic_fetch_sub(&count_, 1, __ATOMIC_ACQ_REL);
#endif
        CHECK(old);
        return old == 1;
    }

    // Once decrement() has returned true, the count is no longer needed,
    // and its storage can be reused, e.g. for an intrusive linked list
    size_t count_ = 1;
};


template <typename Derived>  // curiously recurring template
struct SimpleRefcount {
    ReferenceCounter refcount;
};

template <typename T>
void reference_add(SimpleRefcount<T>& obj) {
    obj.refcount.increment();
}

template <typename T>
void reference_remove(SimpleRefcount<T>& obj) {
    if (obj.refcount.decrement()) delete static_cast<T*>(&obj);
}


template <typename T>
class RefPtr {
public:
    RefPtr() : ptr_(nullptr) {}

    RefPtr(const RefPtr& that) : ptr_(that.ptr_) {
        if (ptr_) reference_add(*ptr_);
    }

    RefPtr(RefPtr&& that) : ptr_(that.ptr_) {
        that.ptr_ = nullptr;
    }

    template <typename T2>
    RefPtr(const RefPtr<T2>& that) : ptr_(that.ptr_) {
        if (ptr_) reference_add(*ptr_);
    }

    template <typename Y>
    RefPtr(RefPtr<Y>&& that) : ptr_(that.ptr_) {
        that.ptr_ = nullptr;
    }

    RefPtr& operator= (const RefPtr& that) {
        if (this != &that) {
            if (that.ptr_) reference_add(*that.ptr_);
            if (ptr_) reference_remove(*ptr_);
            ptr_ = that.ptr_;
        }
        return *this;
    }

    RefPtr& operator= (RefPtr&& that) {
        if (this != &that) {
            if (ptr_) reference_remove(*ptr_);
            ptr_ = that.ptr_;
            that.ptr_ = nullptr;
        }
        return *this;
    }

    template <typename T2>
    RefPtr& operator= (RefPtr<T2>&& that) {
        return (*this = RefPtr(std::move(that)));
    }

    T& operator* () const {
        return *ptr_;
    }

    T* operator-> () const {
        return ptr_;
    }

    T* get() const {
        return ptr_;
    }

    explicit operator bool() const {
        return ptr_;
    }

    bool operator== (const RefPtr& that) const {
        return ptr_ == that.ptr_;
    }

    bool operator!= (const RefPtr& that) const {
        return ptr_ != that.ptr_;
    }

    T* release() {
        T* ret = ptr_;
        ptr_ = nullptr;
        return ret;
    }

    ~RefPtr() {
        if (ptr_) reference_remove(*ptr_);
    }

private:
    T* ptr_;

    // For clarity, use steal() instead
    explicit RefPtr(T* ptr) : ptr_(ptr) {}

    template <typename T2> friend RefPtr<T2> steal(T2* ptr);
    template <typename T2> friend RefPtr<T2> newref(T2* ptr);
    template <typename T2> friend class RefPtr;
};

template <typename T>
RefPtr<T> steal(T* ptr) {
    return RefPtr<T>(ptr);
}

template <typename T>
RefPtr<T> newref(T* ptr) {
    CHECK(ptr);
    reference_add(*ptr);
    return RefPtr<T>(ptr);
}


// A reference that is set at most once and is never released afterwards. Multiple threads may
// race to compute the value: the first set() wins, and the other values are released.
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
    T* set(RefPtr<T> value) {
        CHECK(value);
        T* expected = nullptr;
        if (atomic_compare_exchange_acq_rel(&ptr_, &expected, value.get()))
            return value.release();
        return expected;
    }
};


template <typename T>
struct Hash<RefPtr<T>> {
    static void hash(const RefPtr<T>& ptr, Hasher& h) {
        Hash<T*>::hash(ptr.get(), h);
    }
};

// Allow heterogeneous lookup for T* vs. RefPtr<T>
template <typename T>
struct CompareKey <T*, RefPtr<T>> {
    static bool equals(T* a, const RefPtr<T>& b) {
        return a == b.get();
    }
};
