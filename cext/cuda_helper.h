/*
 * SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "py.h"
#include <cstdint>
#include <cuda.h>

struct DriverApi;

Status cuda_helper_init(PyObject* m);

const char* get_cuda_error(const DriverApi*, CUresult res);

void try_cuInit(const DriverApi*);

Status check_driver_version(const DriverApi*, int minimum_version);

struct ComputeCapability {
    int32_t major;
    int32_t minor;

    int64_t as_key() const {
        return (static_cast<int64_t>(major) << 32) | static_cast<uint32_t>(minor);
    }
};

Result<ComputeCapability> get_device_compute_capability(const DriverApi*, int device_id);
