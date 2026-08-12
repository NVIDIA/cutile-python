/*
 * SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "check.h"
#include "ref_ptr.h"
#include "vec.h"
#include <Python.h>
#include <optional>


using PyPtr = RefPtr<PyObject>;

static inline void reference_add(PyObject& obj) {
    Py_INCREF(&obj);
}

static inline void reference_remove(PyObject& obj) {
    Py_DECREF(&obj);
}

static inline int pylong_as_int(PyObject* obj) {
    int overflow;
    long val = PyLong_AsLongAndOverflow(obj, &overflow);
    if (PyErr_Occurred()) return -1;
    if (overflow || val > INT_MAX || val < INT_MIN) {
        PyErr_SetString(PyExc_OverflowError,
            "Python int too large to convert to C int");
        return -1;
    }
    return static_cast<int>(val);
}

static inline unsigned pylong_as_uint(PyObject* obj) {
    unsigned long val = PyLong_AsUnsignedLong(obj);
    if (PyErr_Occurred()) return -1;
    if (val > UINT_MAX) {
        PyErr_SetString(PyExc_OverflowError,
            "Python int too large to convert to C unsigned int");
        return -1;
    }
    return static_cast<unsigned>(val);
}

template <typename T>
T pylong_as(PyObject* obj) {
    if constexpr (std::is_same_v<T, int>) {
        return pylong_as_int(obj);
    } else if constexpr (std::is_same_v<T, unsigned>) {
        return pylong_as_uint(obj);
    } else if constexpr (std::is_same_v<T, long>) {
        return PyLong_AsLong(obj);
    } else if constexpr (std::is_same_v<T, long long>) {
        return PyLong_AsLongLong(obj);
    } else if constexpr (std::is_same_v<T, unsigned long>) {
        return PyLong_AsUnsignedLong(obj);
    } else if constexpr (std::is_same_v<T, unsigned long long>) {
        return PyLong_AsUnsignedLongLong(obj);
    } else {
        static_assert(!sizeof(T*), "pylong_as<T> not implemented for given T");
    }
}

template <typename T>
T pylong_as(const PyPtr& ptr) {
    return pylong_as<T>(ptr.get());
}

template <typename T>
T pylong_as_overflow_and(PyObject* obj, int* overflow) {
    if constexpr (std::is_same_v<T, int>) {
        return pylong_as_int(obj);
    } else if constexpr (std::is_same_v<T, long>) {
        return PyLong_AsLongAndOverflow(obj, overflow);
    } else if constexpr (std::is_same_v<T, long long>) {
        return PyLong_AsLongLongAndOverflow(obj, overflow);
    } else {
        static_assert(!sizeof(T*), "pylong_as_overflow_and<T> not implemented for given T");
    }
}

template <typename T>
struct PythonWrapper {
    PyObject_HEAD
    T object;
};

template <typename T>
T& py_unwrap(PyObject* pyobj) {
    PythonWrapper<T>* wrapper = reinterpret_cast<PythonWrapper<T>*>(pyobj);
    return wrapper->object;
}

template <typename T>
PyObject* pywrapper_new(PyTypeObject* type, PyObject*, PyObject*) {
    PyObject* ret = type->tp_alloc(type, 0);
    if (!ret) return nullptr;

    T& obj = py_unwrap<T>(ret);
    new (&obj) T();
    return ret;
}

template <typename T>
void pywrapper_dealloc(PyObject* self) {
    PythonWrapper<T>* wrapper = reinterpret_cast<PythonWrapper<T>*>(self);
    wrapper->object.~T();
    Py_TYPE(self)->tp_free(self);
}

template <typename T>
PyObject* pywrapper_richcompare_via_operator_equals(PyObject* self, PyObject* other, int op) {
    if (!PyObject_TypeCheck(self, &T::pytype) || !PyObject_TypeCheck(other, &T::pytype))
        return Py_NewRef(Py_NotImplemented);

    T& a = py_unwrap<T>(self);
    T& b = py_unwrap<T>(other);

    switch (op) {
    case Py_EQ: return Py_NewRef(a == b ? Py_True : Py_False);
    case Py_NE: return Py_NewRef(a == b ? Py_False : Py_True);
    default: return Py_NewRef(Py_NotImplemented);
    }
}

struct OK_t{};
struct ErrorRaised_t{};

class [[nodiscard]] Status {
public:
    Status(OK_t) : ok_(true) {}
    Status(ErrorRaised_t) : ok_(false) {}

    explicit operator bool() const {
        return ok_;
    }

private:
    bool ok_;
};

static constexpr OK_t OK = {};
static constexpr ErrorRaised_t ErrorRaised = {};


template <typename T>
class [[nodiscard]] Result {
public:
    Result(ErrorRaised_t) : opt_(std::nullopt) {}

    Result(const T& val) : opt_(val) {}

    Result(T&& val) : opt_(std::move(val)) {}

    Result(const Result& other) : opt_(other.opt_) {}

    Result(Result&& other) : opt_(std::move(other.opt_)) {}

    Result& operator= (const Result& other) {
        opt_ = other.opt_;
        return *this;
    }

    Result& operator= (Result&& other) {
        opt_ = std::move(other.opt_);
        return *this;
    }

    bool is_ok() const {
        return opt_.has_value();
    }

    T& operator* () {
        return *opt_;
    }

    const T& operator* () const {
        return *opt_;
    }

    T* operator-> () {
        return &*opt_;
    }

    const T* operator-> () const {
        return &*opt_;
    }

private:
    std::optional<T> opt_;
};

struct UseRepr {
    PyObject* obj;
};

// Wraps a PyObject* argument to raise()/println()/to_pyunicode()/etc. to indicate that
// repr() should be used rather than the default str(), e.g.:
//
//     println(use_repr(obj));
static inline UseRepr use_repr(PyObject* obj) {
    return UseRepr{ obj };
}

static inline UseRepr use_repr(const PyPtr& obj) {
    return use_repr(obj.get());
}


#if PY_VERSION_HEX >= 0x030E0000
class StringBuilderImpl {
protected:
    Status init() {
        writer_ = PyUnicodeWriter_Create(0);
        if (!writer_) return ErrorRaised;
        return OK;
    }

    int write_ascii(const char* str, Py_ssize_t size = -1) {
        return PyUnicodeWriter_WriteASCII(writer_, str, size);
    }

    int write_char(Py_UCS4 c) {
        return PyUnicodeWriter_WriteChar(writer_, c);
    }

    int write_str(PyObject* obj) {
        return PyUnicodeWriter_WriteStr(writer_, obj);
    }

    int write_repr(PyObject* obj) {
        return PyUnicodeWriter_WriteRepr(writer_, obj);
    }

    void discard() {
        PyUnicodeWriter_Discard(writer_);
        writer_ = nullptr;
    }

    PyObject* finish() {
        PyObject* ret = PyUnicodeWriter_Finish(writer_);
        writer_ = nullptr;
        return ret;
    }
private:
    PyUnicodeWriter* writer_;
};
#else  // PY_VERSION_HEX >= 0x030E0000
class StringBuilderImpl {
protected:
    Status init() {
        _PyUnicodeWriter_Init(&writer_);
        return OK;
    }

    int write_ascii(const char* str, Py_ssize_t size = -1) {
        return _PyUnicodeWriter_WriteASCIIString(&writer_, str, size);
    }

    int write_char(Py_UCS4 c) {
        return _PyUnicodeWriter_WriteChar(&writer_, c);
    }

    int write_str(PyObject* obj) {
        PyPtr str = steal(PyObject_Str(obj));
        if (!str) return -1;
        return _PyUnicodeWriter_WriteStr(&writer_, str.get());
    }

    int write_repr(PyObject* obj) {
        PyPtr str = steal(PyObject_Repr(obj));
        if (!str) return -1;
        return _PyUnicodeWriter_WriteStr(&writer_, str.get());
    }

    void discard() {
        _PyUnicodeWriter_Dealloc(&writer_);
    }

    PyObject* finish() {
        return _PyUnicodeWriter_Finish(&writer_);
    }
private:
    _PyUnicodeWriter writer_;
};
#endif  // PY_VERSION_HEX >= 0x030E0000


class StringBuilder : StringBuilderImpl {
public:
    StringBuilder() {
        error_ = !init();
    }

    StringBuilder(const StringBuilder&) = delete;
    void operator=(const StringBuilder&) = delete;

    ~StringBuilder() {
        discard();
    }

    void append(const char* s) {
        handle_error([=] { return write_ascii(s); });
    }

    void append(char c) {
        Py_UCS4 usc4 = static_cast<unsigned char>(c);
        handle_error([=] { return write_char(usc4); });
    }

    void append(unsigned int x) {
        append_sprintf<30>("%u", x);
    }

    void append(int x) {
        append_sprintf<30>("%d", x);
    }

    void append(unsigned long x) {
        append_sprintf<30>("%lu", x);
    }

    void append(long x) {
        append_sprintf<30>("%ld", x);
    }

    void append(unsigned long long x) {
        append_sprintf<30>("%llu", x);
    }

    void append(long long x) {
        append_sprintf<30>("%lld", x);
    }

    void append(PyObject* obj) {
        handle_error([=] { return obj ? write_str(obj) : write_ascii("(null)"); });
    }

    void append(UseRepr u) {
        handle_error([=] { return u.obj ? write_repr(u.obj) : write_ascii("(null)"); });
    }

    void append(const PyPtr& obj) {
        append(obj.get());
    }

    template <typename T, typename = std::enable_if_t<std::is_enum_v<T>>>
    void append(const T& value) {
        append(static_cast<std::underlying_type_t<T>>(value));
    }

    template <typename T>
    void append(const T* ptr) {
        append_sprintf<30>("%p", ptr);
    }

    template <typename T>
    void append(const Vec<T>& vec) {
        append("[");
        const char* comma = "";
        for (const T& x : vec) {
            append(comma);
            append(x);
            comma = ", ";
        }
        append("]");
    }

    template <typename T>
    void append(const Result<T>& res) {
        if (res.is_ok()) {
            append("OK(");
            append(*res);
            append(")");
        } else {
            append("ErrorRaised");
        }
    }

    template <typename T>
    void append(const std::optional<T>& opt) {
        if (opt.has_value()) {
            append("std::optional{");
            append(*opt);
            append("}");
        } else {
            append("std::nullopt");
        }
    }

    template <typename... Args>
    void append_many(Args&&... args) {
        (append(std::forward<Args>(args)), ...);
    }

    PyPtr build() {
        if (error_) return {};
        return steal(finish());
    }

    ErrorRaised_t raise(PyObject* exctype) {
        PyPtr message = build();
        if (message)
            PyErr_Format(exctype, "%U", message.get());  // noqa
        else
            PyErr_SetString(exctype, "Failed to format error message");
        return ErrorRaised;
    }

private:
    bool error_;

    template <size_t BufSize, typename... Args>
    void append_sprintf(const char* fmt, Args&&... args) {
        handle_error([&] {
            char buf[BufSize];
            int r = PyOS_snprintf(buf, sizeof buf, fmt, std::forward<Args>(args)...);
            if (r < 0 || r >= (int) sizeof buf) {
                PyErr_SetString(PyExc_RuntimeError, "snprintf() failed");
                return -1;
            }
            return write_ascii(buf, r);
        });
    }

    template <typename F>
    void handle_error(F&& func) {
        if (!error_) {
            if (func() < 0)
                error_ = true;
        }
    }
};

template <typename... Args>
ErrorRaised_t raise(PyObject* exctype, Args&&... message) {
    StringBuilder builder;
    builder.append_many(std::forward<Args>(message)...);
    return builder.raise(exctype);
}

template <typename... Args>
PyPtr to_pyunicode(Args&&... pieces) {
    StringBuilder builder;
    builder.append_many(std::forward<Args>(pieces)...);
    return builder.build();
}

template <typename... Args>
void println(Args&&... message) {
    StringBuilder builder;
    builder.append_many(std::forward<Args>(message)...);
    PyPtr s = builder.build();
    CHECK(s);
    PySys_FormatStdout("%U\n", s.get());
}

template <typename... Args>
void println_err(Args&&... message) {
    StringBuilder builder;
    builder.append_many(std::forward<Args>(message)...);
    PyPtr s = builder.build();
    CHECK(s);
    PySys_FormatStderr("%U\n", s.get());
}

struct SavedException {
    PyPtr type, value, traceback;

    operator bool() const {
        return bool(type);
    }

    void normalize() {
        PyObject* tmp_type = type.release();
        PyObject* tmp_value = value.release();
        PyObject* tmp_traceback = traceback.release();
        PyErr_NormalizeException(&tmp_type, &tmp_value, &tmp_traceback);
        type = steal(tmp_type);
        value = steal(tmp_value);
        traceback = steal(tmp_traceback);
        if (traceback)
            PyException_SetTraceback(value.get(), traceback.get());
    }

    void restore() {
        PyObject* tmp_type = type.release();
        PyObject* tmp_value = value.release();
        PyObject* tmp_traceback = traceback.release();
        PyErr_Restore(tmp_type, tmp_value, tmp_traceback);
    }
};

static inline SavedException save_raised_exception() {
    PyObject *type, *value, *traceback;
    PyErr_Fetch(&type, &value, &traceback);
    return SavedException{steal(type), steal(value), steal(traceback)};
}

#ifdef _MSC_VER
#define PRINTF_LIKE(a, b)
#else
#define PRINTF_LIKE(a, b) __attribute__(( format(printf, a, b) ))
#endif

void log_python_error(const char* filename, int line, const char* level, SavedException& exc,
                      const char* fmt, ...) PRINTF_LIKE(5, 6);

#define LOG_PYTHON_ERROR(level, exc, ...) \
        log_python_error(__FILE__, __LINE__, level, exc, __VA_ARGS__)


static inline PyPtr getattr(PyObject* obj, PyObject* attrname) {
    return steal(PyObject_GetAttr(obj, attrname));
}

static inline PyPtr getattr(PyObject* obj, const char* attrname) {
    return steal(PyObject_GetAttrString(obj, attrname));
}

static inline PyPtr getattr(const PyPtr& obj, const char* attrname) {
    return getattr(obj.get(), attrname);
}

static inline void pyunicode_intern_in_place(PyPtr* s) {
    PyObject* raw = s->release();
    PyUnicode_InternInPlace(&raw);
    *s = steal(raw);
}

struct ErrorGuard {
    SavedException exc;

    ErrorGuard() {
        exc = save_raised_exception();
    }

    ErrorGuard(const ErrorGuard&) = delete;
    void operator=(const ErrorGuard&) = delete;

    ~ErrorGuard() {
        exc.restore();
    }
};

static inline PyPtr try_getattr(PyObject* obj, const char* attrname,
                                SavedException* exc = nullptr) {
    ErrorGuard guard;
    PyPtr ret = getattr(obj, attrname);
    if (!ret && exc) *exc = save_raised_exception();
    return ret;
}

static inline PyPtr try_getattr(const PyPtr& obj, const char* attrname,
                                SavedException* exc = nullptr) {
    return try_getattr(obj.get(), attrname, exc);
}

static inline PyPtr try_import(const char* modname, SavedException* exc = nullptr) {
    ErrorGuard guard;
    PyPtr ret = steal(PyImport_ImportModule(modname));
    if (!ret && exc) *exc = save_raised_exception();
    return ret;
}

class GILGuard {
public:
    GILGuard(const GILGuard&) = delete;
    void operator=(const GILGuard&) = delete;

    GILGuard() {
        gstate = PyGILState_Ensure();
    }

    ~GILGuard() {
        PyGILState_Release(gstate);
    }
private:
    PyGILState_STATE gstate;
};

#ifdef Py_GIL_DISABLED
class PyCriticalSectionGuard {
    public:
        explicit PyCriticalSectionGuard(PyMutex* mutex) {
            CHECK(mutex);
            PyCriticalSection_BeginMutex(&_py_cs, mutex);
        }

        ~PyCriticalSectionGuard() {
            PyCriticalSection_End(&_py_cs);
        }

        PyCriticalSectionGuard(const PyCriticalSectionGuard&) = delete;
        void operator=(const PyCriticalSectionGuard&) = delete;
    private:
        PyCriticalSection _py_cs;
};
#endif
