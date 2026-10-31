// SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "atomics.h"
#include "check.h"
#include "memory.h"

#include <cstdint>
#include <utility>

#if defined(_MSC_VER)
#include <intrin.h>
#endif


// A table of elements addressed by 32-bit index, allocated in pages on first access.
// Elements are value-initialized when their page is allocated. They never move,
// and are never destroyed. The table doesn't track its size: that's up to the user.
// Thread-safe. Synchronizing access to the elements themselves is up to the user.
//
// Intended to be used as a static/global.
template <typename T, uint32_t kFirstPageLog2 = 9>
struct PagedTable {
    static_assert(kFirstPageLog2 < 32);
    static constexpr uint32_t kMaxPages = 33 - kFirstPageLog2;
    static constexpr uint32_t kFirstPageSize = 1u << kFirstPageLog2;

    T* pages_[kMaxPages];  // Accessed atomically

    T& operator[](uint32_t index) {
        auto [page, offset] = index_to_page_and_offset(index);
        T* data = atomic_load_acquire(&pages_[page]);
        if (!data) data = allocate_page(page);
        return data[offset];
    }

    // Like operator[], but never allocates: aborts the program if the page containing `index`
    // hasn't been allocated yet.
    const T& expect(uint32_t index) const {
        auto [page, offset] = index_to_page_and_offset(index);
        const T* data = atomic_load_acquire(&pages_[page]);
        CHECK(data);
        return data[offset];
    }

private:
    T* allocate_page(uint32_t page) {
        uint32_t capacity = page == 0 ? kFirstPageSize : kFirstPageSize << (page - 1);
        T* fresh = new T[capacity]();
        T* expected = nullptr;
        if (atomic_compare_exchange_acq_rel(&pages_[page], &expected, fresh)) return fresh;
        // Another thread has allocated the page first. Ours has never been published,
        // so no one else can be using it.
        delete[] fresh;
        return expected;
    }

    static std::pair<uint32_t, uint32_t> index_to_page_and_offset(uint32_t index) {
        if (index < kFirstPageSize) return {0, index};
        uint32_t log2 = floor_log2(index);
        return {log2 - kFirstPageLog2 + 1, index - (1u << log2)};
    }

    static uint32_t floor_log2(uint32_t x) {
#if defined(_MSC_VER)
        unsigned long ret;
        _BitScanReverse(&ret, x);
        return static_cast<uint32_t>(ret);
#else
        return 31 - static_cast<uint32_t>(__builtin_clz(x));
#endif
    }
};
