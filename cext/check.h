/*
 * SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <Python.h>

#define _FATAL_ERROR3(msg, file, lineno) Py_FatalError(file ":" #lineno ": " msg)
#define _FATAL_ERROR2(msg, file, lineno) _FATAL_ERROR3(msg, file, lineno)
#define _FATAL_ERROR(msg) _FATAL_ERROR2(msg, __FILE__, __LINE__)

// Like assert() but can't be disabled
#define CHECK(cond) do { \
        if (!(cond)) _FATAL_ERROR("CHECK FAILED: " #cond); \
    } while (0)

#define CHECK_UNREACHABLE _FATAL_ERROR("Unreachable code has been reached")
