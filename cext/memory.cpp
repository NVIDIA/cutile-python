/*
 * SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "memory.h"
#include <Python.h>


void* xcalloc(size_t nmemb, size_t size) {
    void* ret = PyMem_RawCalloc(nmemb, size);
    CHECK(ret);
    return ret;
}

void* operator new (size_t len) {
    void* ret = PyMem_RawMalloc(len);
    CHECK(ret);
    return ret;
}

void* operator new[] (size_t len) {
    void* ret = PyMem_RawMalloc(len);
    CHECK(ret);
    return ret;
}

void operator delete (void* ptr, size_t) {
    PyMem_RawFree(ptr);
}

void operator delete[] (void* ptr) {
    PyMem_RawFree(ptr);
}

void mem_free(void* p) {
    PyMem_RawFree(p);
}

#ifdef _WIN32
// Prevent msvc to optimize this into a memcpy call
// because there isn't equivalent of -fno-builtin
#pragma optimize("", off)
#endif
void mem_copy(void* RESTRICT dst, const void* RESTRICT src, size_t n) {
    char* dst_c = static_cast<char*>(dst);
    const char* src_c = static_cast<const char*>(src);
    for (size_t i = 0; i < n; ++i)
        dst_c[i] = src_c[i];
}
#ifdef _WIN32
#pragma optimize("", on)
#endif

#ifdef _WIN32
// We don't link the C runtime, but MSVC may still emit calls to these functions,
// e.g. for large struct copies or zero-initialization.
#pragma function(memset, memcpy, memmove, memcmp)
// Prevent msvc from turning the loops below back into calls to themselves
#pragma optimize("", off)

extern "C" void* memcpy(void* dst, const void* src, size_t n) {
    mem_copy(dst, src, n);
    return dst;
}

extern "C" void* memset(void* dst, int c, size_t n) {
    unsigned char* dst_c = static_cast<unsigned char*>(dst);
    for (size_t i = 0; i < n; ++i)
        dst_c[i] = static_cast<unsigned char>(c);
    return dst;
}

extern "C" void* memmove(void* dst, const void* src, size_t n) {
    unsigned char* dst_c = static_cast<unsigned char*>(dst);
    const unsigned char* src_c = static_cast<const unsigned char*>(src);
    if (reinterpret_cast<uintptr_t>(dst_c) < reinterpret_cast<uintptr_t>(src_c)) {
        for (size_t i = 0; i < n; ++i)
            dst_c[i] = src_c[i];
    } else {
        for (size_t i = n; i > 0; --i)
            dst_c[i - 1] = src_c[i - 1];
    }
    return dst;
}

extern "C" int memcmp(const void* a, const void* b, size_t n) {
    const unsigned char* a_c = static_cast<const unsigned char*>(a);
    const unsigned char* b_c = static_cast<const unsigned char*>(b);
    for (size_t i = 0; i < n; ++i) {
        if (a_c[i] != b_c[i])
            return a_c[i] < b_c[i] ? -1 : 1;
    }
    return 0;
}

#pragma optimize("", on)
#endif
