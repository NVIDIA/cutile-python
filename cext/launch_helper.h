// SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "vec.h"

#include <cuda.h>

#include <cstddef>
#include <cstdint>


using ArenaOffset = size_t;

union Word {
    void* device_ptr;
    int32_t i32;
    int64_t i64;
    size_t size;
    float f32;
    ArenaOffset arena_offset;
};

static_assert(sizeof(Word) == 8);

using Arena = Vec<Word, AlignedAllocation<Word, alignof(CUtensorMap)>>;

struct ListArg {
    ArenaOffset base_ptr_cuarg;
    size_t length;
    ArenaOffset item_offsets;  // [length], each word contains an `ArenaOffset` points to the item
    size_t item_size_words;
};

struct LaunchHelper {
    Vec<PyTypeObject*> pyarg_types_breadth_first;
    Vec<PyObject*> pyarg_objs_breadth_first;
    Vec<PyObject*> leaf_pyarg_objs;
    Vec<PyPtr> pyarg_refs;  // extra references to parsed arguments, e.g. unpacked dataclass fields
    Arena arena;
    Vec<ArenaOffset> cuarg_offsets;  // offsets into `arena`
    Vec<ArenaOffset> array_ptr_arena_offsets;
    Vec<void*> launch_params;
    Vec<ListArg> list_args;
    size_t total_list_data_size_words;
    Vec<int64_t> constants;
    CUcontext cuda_context;
    LaunchHelper* next_free;

    LaunchHelper()
      : total_list_data_size_words(0)
      , cuda_context(nullptr)
      , next_free(nullptr)
    {}
};
