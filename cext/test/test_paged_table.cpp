// SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0

#include "../paged_table.h"
#include "../check.h"

struct Item {
    uint32_t a;
    uint64_t b;
};

// Has a non-trivial default constructor
struct WithDefault {
    uint32_t value = 7;
};

static PagedTable<Item, 2> g_small_table;
static PagedTable<Item, 0> g_tiny_table;
static PagedTable<Item> g_default_table;
static PagedTable<WithDefault, 2> g_with_default_table;

template <uint32_t kFirstPageLog2>
static void test_write_and_read(PagedTable<Item, kFirstPageLog2>& table, uint32_t n) {
    // Elements are zero-initialized
    for (uint32_t i = 0; i < n; ++i)
        CHECK(table[i].a == 0 && table[i].b == 0);

    for (uint32_t i = 0; i < n; ++i)
        table[i] = Item{i, uint64_t(i) * 3};

    for (uint32_t i = 0; i < n; ++i) {
        CHECK(table[i].a == i);
        CHECK(table[i].b == uint64_t(i) * 3);
    }
}

int main() {
    // Small first page: exercises many page boundaries (4, 4, 8, 16, ...)
    test_write_and_read(g_small_table, 10000);

    // Default first page: the boundaries around 512, 1024 and 2048
    test_write_and_read(g_default_table, 2049);

    // First page of size 1
    test_write_and_read(g_tiny_table, 1000);

    // Elements don't move when new pages are allocated
    Item* first = &g_small_table[0];
    Item* boundary = &g_small_table[4];
    g_small_table[100000].a = 1;
    CHECK(&g_small_table[0] == first);
    CHECK(&g_small_table[4] == boundary);

    // expect() returns the same element as operator[]
    CHECK(&g_small_table.expect(0) == first);
    CHECK(&g_small_table.expect(100000) == &g_small_table[100000]);
    CHECK(g_small_table.expect(100000).a == 1);

    // Elements are value-initialized using the default constructor
    for (uint32_t i = 0; i < 100; ++i)
        CHECK(g_with_default_table[i].value == 7);

    return 0;
}
