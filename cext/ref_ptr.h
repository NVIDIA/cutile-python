/*
 * SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>

#include "check.h"
#include "hash_map.h"


template <typename Derived>  // curiously recurring template
struct SimpleRefcount {
    size_t refcount = 1;
};

template <typename T>
void reference_add(SimpleRefcount<T>& obj) {
    ++obj.refcount;
}

template <typename T>
void reference_remove(SimpleRefcount<T>& obj) {
    CHECK(obj.refcount);
    if (!--obj.refcount) delete static_cast<T*>(&obj);
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
        if (this != &that) {
            if (ptr_) reference_remove(*ptr_);
            ptr_ = that.ptr_;
            that.ptr_ = nullptr;
        }
        return *this;
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
