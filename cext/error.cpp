// SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0

#include "error.h"

static PyObject* g_type_checking_error_cls;
static PyObject* g_invalid_value_error_cls;
static PyObject* g_unsupported_feature_error_cls;

ErrorRaised_t Error::raise_builtin_exception() {
    PyObject* exc_type = nullptr;
    switch (tag()) {
    case kOk: CHECK_UNREACHABLE;
    case kRaised: return ErrorRaised;
    case kTypeError: exc_type = PyExc_TypeError; break;
    case kValueError: exc_type = PyExc_ValueError; break;
    case kOverflowError: exc_type = PyExc_OverflowError; break;
    case kZeroDivisionError: exc_type = PyExc_ZeroDivisionError; break;
    case kIndexError: exc_type = PyExc_IndexError; break;
    case kNotImplementedError: exc_type = PyExc_NotImplementedError; break;
    }
    PyErr_SetObject(exc_type, message());
    *this = Error(ErrorRaised);
    return ErrorRaised;
}

ErrorRaised_t Error::raise_compile_time_error() {
    PyObject* exc_type = nullptr;
    switch (tag()) {
    case Error::kOk: CHECK_UNREACHABLE;
    case Error::kRaised: return ErrorRaised;
    case Error::kTypeError: exc_type = g_type_checking_error_cls; break;
    case Error::kValueError: exc_type = g_invalid_value_error_cls; break;
    case Error::kOverflowError: exc_type = g_invalid_value_error_cls; break;
    case Error::kZeroDivisionError: exc_type = g_invalid_value_error_cls; break;
    case Error::kIndexError: exc_type = g_invalid_value_error_cls; break;
    case Error::kNotImplementedError: exc_type = g_unsupported_feature_error_cls; break;
    }
    PyErr_SetObject(exc_type, message());
    *this = Error(ErrorRaised);
    return ErrorRaised;
}

Status error_module_init(PyObject* m) {
    PyPtr exception_mod = steal(PyImport_ImportModule("cuda.tile._exception"));
    if (!exception_mod) return ErrorRaised;
    g_type_checking_error_cls = getattr(exception_mod, "TypeCheckingError").release();
    if (!g_type_checking_error_cls) return ErrorRaised;
    g_invalid_value_error_cls = getattr(exception_mod, "InvalidValueError").release();
    if (!g_invalid_value_error_cls) return ErrorRaised;
    g_unsupported_feature_error_cls = getattr(exception_mod, "UnsupportedFeatureError").release();
    if (!g_unsupported_feature_error_cls) return ErrorRaised;

    return OK;
}

