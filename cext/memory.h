/*
 * SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "check.h"


void* operator new (size_t len);

void* operator new[] (size_t len);

void operator delete (void* ptr, size_t);

void operator delete[] (void* ptr);

void* xcalloc(size_t nmemb, size_t size);


template <typename T>
T* xcalloc(size_t nmemb) {
    return static_cast<T*>(xcalloc(nmemb, sizeof(T)));
}

void mem_free(void* p);

#if defined(_MSC_VER)
  #define RESTRICT __restrict
#else
  #define RESTRICT __restrict__
#endif

void mem_copy(void* RESTRICT dst, const void* RESTRICT src, size_t n);

void mem_zero(void* dst, size_t n);

