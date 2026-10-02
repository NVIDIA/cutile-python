// SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "py.h"

#include <cstdint>

// Success or an error, packed into a single word. Unlike Status, an error other than kRaised
// is carried explicitly instead of being set as the Python exception, so that the entry point
// that returns it to Python can choose the exception class.
class [[nodiscard]] Error {
public:
    enum Tag : uint8_t {
        kOk = 0,
        kRaised,  // the Python exception has already been set
        kTypeError,
        kValueError,
        kOverflowError,
        kZeroDivisionError,
        kIndexError,
        kNotImplementedError,
    };

    Error(OK_t) : data_(0) {}

    Error(ErrorRaised_t) : data_(encode(kRaised, nullptr)) {}

    // Takes ownership of `message`, which must be a str
    Error(Tag tag, PyObject* message) : data_(encode(tag, message)) {
        CHECK(tag != kOk && tag != kRaised);
        CHECK(message);
    }

    Error(const Error&) = delete;
    void operator=(const Error&) = delete;

    // A moved-from error becomes kRaised, so that it no longer owns the message
    Error(Error&& other) : data_(other.data_) {
        if (!other.is_ok()) other.data_ = encode(kRaised, nullptr);
    }

    Error& operator= (Error&& other) {
        if (this != &other) {
            Py_XDECREF(message());
            data_ = other.data_;
            if (!other.is_ok()) other.data_ = encode(kRaised, nullptr);
        }
        return *this;
    }

    ~Error() {
        Py_XDECREF(message());
    }

    constexpr Tag tag() const {
        return static_cast<Tag>(data_ >> 56);
    }

    constexpr bool is_ok() const {
        return tag() == kOk;
    }

    PyObject* message() const {
        return reinterpret_cast<PyObject*>(data_ & 0x00ff'ffff'ffff'ffff);
    }

    ErrorRaised_t raise_builtin_exception();
    ErrorRaised_t raise_compile_time_error();

    ErrorRaised_t raise(bool compile_time_error) {
        return compile_time_error ? raise_compile_time_error() : raise_builtin_exception();
    }


private:
    uint64_t data_;  // Tag in the high 8 bits, the message in the low 56 bits

    static uint64_t encode(Tag tag, PyObject* message) {
        return (static_cast<uint64_t>(tag) << 56) | reinterpret_cast<uint64_t>(message);
    }
};

// Returns an error with a message built like raise(), e.g.
//     return make_error(Error::kTypeError, "Shape must be a tuple, not ", use_type(val));
template <typename... Args>
Error make_error(Error::Tag tag, Args&&... message) {
    StringBuilder builder;
    builder.append_many(std::forward<Args>(message)...);
    PyPtr str = builder.build();
    if (!str) return ErrorRaised;
    return Error(tag, str.release());
}

// Either a value of type T or an error
template <typename T>
class [[nodiscard]] Expected {
public:
    Expected(const T& value) : error_(OK) {
        new (&value_) T(value);
    }

    Expected(T&& value) : error_(OK) {
        new (&value_) T(std::move(value));
    }

    Expected(ErrorRaised_t) : error_(ErrorRaised) {}

    Expected(Result<T>&& result) : error_(OK) {
        if (result.is_ok())
            new (&value_) T(std::move(*result));
        else
            error_ = Error(ErrorRaised);
    }

    Expected(Error&& error) : error_(std::move(error)) {
        CHECK(!error_.is_ok());
    }

    Expected(const Expected&) = delete;
    void operator=(const Expected&) = delete;

    Expected(Expected&& other) : error_(std::move(other.error_)) {
        if (is_ok()) new (&value_) T(std::move(other.value_));
    }

    Expected& operator= (Expected&& other) {
        if (this != &other) {
            if (is_ok()) value_.~T();
            error_ = std::move(other.error_);
            if (is_ok()) new (&value_) T(std::move(other.value_));
        }
        return *this;
    }

    Expected& operator= (Error&& error) {
        CHECK(!error.is_ok());
        if (is_ok()) value_.~T();
        error_ = std::move(error);
        return *this;
    }

    ~Expected() {
        if (is_ok()) value_.~T();
    }

    constexpr bool is_ok() const {
        return error_.is_ok();
    }

    Error& error() {
        return error_;
    }

    const Error& error() const {
        return error_;
    }

    T& operator* () {
        return value_;
    }

    const T& operator* () const {
        return value_;
    }

    T* operator-> () {
        return &value_;
    }

    const T* operator-> () const {
        return &value_;
    }

private:
    Error error_;
    union {
        T value_;
    };
};

// Converts an error to the corresponding built-in Python exception, e.g. at an entry point
template <typename T>
Result<T> to_result(Expected<T>&& expected) {
    if (!expected.is_ok()) return expected.error().raise_builtin_exception();
    return std::move(*expected);
}

static inline Status to_status(Error&& error) {
    if (!error.is_ok()) return error.raise_builtin_exception();
    return OK;
}

Status error_module_init(PyObject* m);

