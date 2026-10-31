// SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0

#include "py.h"
#include "vec.h"

static void best_effort_cleanup_on_internal_error(Vec<PyPtr>& stack) {
    if (stack.empty())
        return;
    CHECK(PyErr_Occurred());
    PyPtr frame = try_getattr(stack.back(), "cr_frame");
    if (frame && PyFrame_Check(frame.get()))
        PyTraceBack_Here(reinterpret_cast<PyFrameObject*>(frame.get()));

    ErrorGuard guard;
    while (!stack.empty()) {
        PyPtr coro = stack.back();
        stack.pop_back();
        Py_XDECREF(PyObject_CallMethod(coro.get(), "close", ""));
        if (PyErr_Occurred())
            PyErr_Print();
    }
}


// Run a coroutine using a software stack to bypass the Python's recursion limit.
// Use resume_after() to break the call chain and push a new frame to the software stack.
static PyObject* run_coroutine(PyObject* self, PyObject* main_coro) {
    if (!PyCoro_CheckExact(main_coro)) {
        raise(PyExc_TypeError, "Expected a coroutine");
        return nullptr;
    }

    Vec<PyPtr> stack;
    stack.push_back(newref(main_coro));

    PyPtr ret = newref(Py_None);
    SavedException exc;

    while (!stack.empty()) {
        PyObject* coro = stack.back().get();
        if (!exc) {
            // Happy path: use PyIter_Send() C API for efficiency
            PyObject* res = nullptr;
            PySendResult send_res = PyIter_Send(coro, ret.get(), &res);
            if (send_res == PYGEN_RETURN) {
                ret = steal(res);
                stack.pop_back();
            } else if (send_res == PYGEN_NEXT) {
                PyPtr next_value = steal(res);
                ret = newref(Py_None);
                if (!PyCoro_CheckExact(next_value.get())) {
                    raise(PyExc_TypeError, "Expected a continuation coroutine");
                    best_effort_cleanup_on_internal_error(stack);
                    return nullptr;
                }
                stack.push_back(std::move(next_value));
            } else {
                CHECK(send_res == PYGEN_ERROR);
                exc = save_raised_exception();
                exc.normalize();
                ret = newref(Py_None);
                stack.pop_back();
            }
            continue;
        }

        // Slow path: need to call .throw() since there is no public C API to _gen_throw()
        PyPtr continuation = steal(PyObject_CallMethod(coro, "throw", "(O)", exc.value.get()));
        if (continuation) {
            ret = newref(Py_None);
            exc = {};
            if (!PyCoro_CheckExact(continuation.get())) {
                raise(PyExc_TypeError, "Expected a continuation coroutine");
                best_effort_cleanup_on_internal_error(stack);
                return nullptr;
            }
            stack.push_back(std::move(continuation));
        } else {
            CHECK(PyErr_Occurred());
            exc = save_raised_exception();
            exc.normalize();
            CHECK(exc.value);
            if (PyErr_GivenExceptionMatches(exc.value.get(), PyExc_StopIteration)) {
                ret = getattr(exc.value, "value");
                if (!ret) {
                    best_effort_cleanup_on_internal_error(stack);
                    return nullptr;
                }
                exc = {};
            } else {
                ret = newref(Py_None);
            }
            stack.pop_back();
        }
    }

    if (exc) {
        exc.restore();
        return nullptr;
    }

    return ret.release();
}


static PyMethodDef functions[] = {
    {"run_coroutine", run_coroutine, METH_O, ""},
    {}
};


Status coroutine_util_init(PyObject* m) {
    if (PyModule_AddFunctions(m, functions) < 0)
        return ErrorRaised;

    return OK;
}

