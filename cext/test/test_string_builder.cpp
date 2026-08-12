// SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0

#include "../py.h"

int main() {
    Py_Initialize();

    // Smoke test for to_pyunicode()
    PyPtr abc = steal(PyUnicode_FromString("abc"));
    CHECK(abc);

    PyPtr ptr = to_pyunicode("hello", 123, '.',
                             -9223372036854775807LL, abc, 18446744073709551615ULL, use_repr(abc));

    CHECK(0 == PyUnicode_CompareWithASCIIString(
                ptr.get(), "hello123.-9223372036854775807abc18446744073709551615'abc'"));

    // Smoke test for raise()
    raise(PyExc_TypeError, "hello", 123);
    CHECK(PyErr_ExceptionMatches(PyExc_TypeError));

    SavedException ex = save_raised_exception();
    ex.normalize();
    CHECK(ex.value);
    PyPtr ex_str = steal(PyObject_Str(ex.value.get()));
    CHECK(ex_str);

    CHECK(0 == PyUnicode_CompareWithASCIIString(ex_str.get(), "hello123"));

    // Smoke test for println() -- we don't actually check the output
    println(123, "hello");
    println_err(456, "hello");
    return 0;
}
