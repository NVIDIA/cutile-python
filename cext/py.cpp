// SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0

#include "py.h"
#include <cstdarg>

void log_python_error(const char* filename, int line, const char* level, SavedException& exc,
                      const char* fmt, ...) {
    ErrorGuard guard;

    static PyObject* kwnames = NULL;
    if (!kwnames) {
        kwnames = Py_BuildValue("(s)", "exc_info");
        if (!kwnames) return;
    }

    static PyObject* logging = NULL;
    if (!logging) {
        logging = try_import("logging").release();
        if (!logging) return;
    }

    va_list a;
    va_start(a, fmt);
    char buf[1024];
    PyOS_vsnprintf(buf, sizeof buf, fmt, a);
    va_end(a);

    PyPtr formatted_message = to_pyunicode(filename, ":", line, ": ", buf);
    if (!formatted_message) return;

    PyPtr func = getattr(logging, level);
    if (!func) return;

    // Pretend that we have caught the exception so that the logger can print it
    PyObject *old_excinfo_type, *old_excinfo_value, *old_excinfo_tb;
    PyErr_GetExcInfo(&old_excinfo_type, &old_excinfo_value, &old_excinfo_tb);
    exc.normalize();
    PyErr_SetExcInfo(Py_XNewRef(exc.type.get()), Py_XNewRef(exc.value.get()),
                     Py_XNewRef(exc.traceback.get()));

    PyObject* args[2] = {formatted_message.get(), Py_True};
    PyObject* res = PyObject_Vectorcall(func.get(), args, 1, kwnames);
    Py_XDECREF(res);

    PyErr_SetExcInfo(old_excinfo_type, old_excinfo_value, old_excinfo_tb);
}

