// SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "py.h"


bool compiled_host_program_check(PyObject* object);
PyObject* compiled_host_program_invoke(PyObject* program, void** arguments, GlobalLock& lock);
Status compiled_host_init(PyObject* module);
