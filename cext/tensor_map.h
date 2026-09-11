/*
 * SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "py.h"

#include <cuda.h>
#include <dlpack.h>

#include <cstdint>


inline constexpr uint32_t kTensorMapMaxRank = 5;

PyPtr tensor_map_tiled(
        DLDataType dtype,
        void* global_address,
        size_t rank,
        const int64_t* array_dimensions,
        const int64_t* array_strides,
        const uint32_t* shape,
        const uint32_t* order,
        int interleave,
        int swizzle,
        int l2_promotion,
        int oob_fill,
        GlobalLock& lock);

Result<uint32_t> tensor_map_data_type_bitwidth(CUtensorMapDataType dtype);

Status tensor_map_validate_tile(
        CUtensorMapDataType data_type,
        size_t rank,
        const uint32_t* shape,
        CUtensorMapInterleave interleave,
        CUtensorMapSwizzle swizzle);

Status tensor_map_validate_global_address(
        CUtensorMapDataType data_type,
        CUtensorMapInterleave interleave,
        const void* global_address);

Status tensor_map_encode_global_dimensions(
        CUtensorMapDataType data_type,
        int32_t rank,
        const int64_t* global_dimensions,
        uint64_t* encoded_dimensions);

Status tensor_map_encode_global_strides(
        CUtensorMapDataType data_type,
        CUtensorMapInterleave interleave,
        int32_t rank,
        const int64_t* global_element_strides,
        uint64_t* encoded_byte_strides);

int32_t native_tensor_map_encode_tiled(
        GlobalLock& lock,
        void* descriptor,
        int32_t data_type,
        int32_t rank,
        void* global_address,
        const int64_t* global_dimensions,
        const int64_t* global_element_strides,
        const int64_t* tile_dimensions,
        int32_t interleave,
        int32_t swizzle,
        int32_t l2_promotion,
        int32_t oob_fill);

Status tensor_map_init(PyObject* m);
