// SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "check.h"
#include "py.h"
#include "enums.h"
#include "error.h"

#include <cuda.h>
#include <cstdint>
#include <type_traits>

// It may look like this file is overusing X macros, which it probably is.

class StringBuilder;

static inline uint32_t memory_space_pointer_bitwidth(MemorySpace memory_space) {
    switch (memory_space) {
        #define MEMORY_SPACE_POINTER_BITWIDTH(name, _id, ptrwidth) \
            case MemorySpace::name: return ptrwidth;
        FOREACH_MEMORY_SPACE(MEMORY_SPACE_POINTER_BITWIDTH);
        #undef MEMORY_SPACE_POINTER_BITWIDTH
    }
    CHECK_UNREACHABLE;
}

// ----- Integer -----

union Integer {
    uint64_t u;
    int64_t s;

    PyPtr to_pylong(bool is_signed) const {
        return steal(is_signed ? PyLong_FromLongLong(s) : PyLong_FromUnsignedLongLong(u));
    }

    void append_to(StringBuilder* sb, bool is_signed) {
        if (is_signed)
            sb->append(s);
        else
            sb->append(u);
    }
};


// ----- FloatFormat -----

struct FloatFormat {
    enum NonFiniteBehavior : uint8_t {
        IEEE,
        NanOnlyAllOnes,
        FiniteOnly,
    };

    int16_t emin;
    int16_t emax;
    uint8_t bitwidth;
    uint8_t exp_bits;
    uint8_t precision;
    NonFiniteBehavior non_finite_behavior;
    bool have_sign;
    bool have_zero_and_subnormals;
};

// ----- DType -----

struct DType {
    uint32_t dtype_id;

    static DType invalid() {
        return {0};
    }

    explicit operator bool() const {
        return dtype_id != 0;
    }

    inline bool operator== (DType other) const {
        return dtype_id == other.dtype_id;
    }

    inline bool operator!= (DType other) const {
        return dtype_id != other.dtype_id;
    }
};

struct DTypeObject {
    DType dtype;
    static PyTypeObject pytype;
};



// (name, bitwdth, signed?, docstring)
#define FOREACH_UNSIGNED_INTEGRAL_DTYPE(X) \
    X(uint8, 8, false, nullptr) \
    X(uint16, 16, false, nullptr) \
    X(uint32, 32, false, nullptr) \
    X(uint64, 64, false, nullptr)

#define FOREACH_SIGNED_INTEGRAL_DTYPE(X) \
    X(int8, 8, true, nullptr) \
    X(int16, 16, true, nullptr) \
    X(int32, 32, true, nullptr) \
    X(int64, 64, true, nullptr)

#define FOREACH_UNRESTRICTED_FLOAT_DTYPE(X) \
    X(float16, 16, true, \
        "A IEEE 754 half-precision (16-bit) binary floating-point |arithmetic dtype| " \
        "(see |IEEE 754-2019|).") \
    X(float32, 32, true, \
        "A IEEE 754 single-precision (32-bit) binary floating-point |arithmetic dtype| " \
        "(see |IEEE 754-2019|).") \
    X(float64, 64, true, \
        "A IEEE 754 double-precision (64-bit) binary floating-point |arithmetic dtype| " \
        "(see |IEEE 754-2019|).") \
    X(bfloat16, 16, true, \
        "A 16-bit floating-point |arithmetic dtype| with 1 sign bit, 8 exponent bits, " \
        "and 7 mantissa bits.")

#define FOREACH_RESTRICTED_FLOAT_DTYPE(X) \
    X(tfloat32, 32, true, \
        "A 32-bit tensor floating-point |numeric dtype| with 1 sign bit, 8 exponent bits, " \
        "and 10 mantissa bits (19-bit representation stored in 32-bit container).") \
    X(float8_e4m3fn, 8, true, \
        "An 8-bit floating-point |numeric dtype| with 1 sign bit, " \
        "4 exponent bits, and 3 mantissa bits.") \
    X(float8_e5m3fnu, 8, false, \
        "An 8-bit floating-point |numeric dtype| with no sign bit, " \
        "5 exponent bits, and 3 mantissa bits.") \
    X(float8_e5m2, 8, true, \
        "An 8-bit floating-point |numeric dtype| with 1 sign bit, " \
        "5 exponent bits, and 2 mantissa bits.") \
    X(float8_e8m0fnu, 8, false, \
        "An 8-bit floating-point |numeric dtype| with no sign bit, " \
        "8 exponent bits, and 0 mantissa bits.") \
    X(float4_e2m1fn, 4, true, \
        "A 4-bit floating-point |numeric dtype| with 1 sign bit, " \
        "2 exponent bits, and 1 mantissa bit.") \
    X(float6_e2m3fn, 6, true, \
        "A 6-bit floating-point numeric dtype with 1 sign bit, " \
        "2 exponent bits, and 3 mantissa bits.") \
    X(float6_e3m2fn, 6, true, \
        "A 6-bit floating-point numeric dtype with 1 sign bit, " \
        "3 exponent bits, and 2 mantissa bits.")


#define FOREACH_NUMERIC_DTYPE(X) \
    X(bool_, 8, false, "An 8-bit boolean |arithmetic dtype|.") \
    FOREACH_UNSIGNED_INTEGRAL_DTYPE(X) \
    FOREACH_SIGNED_INTEGRAL_DTYPE(X) \
    FOREACH_UNRESTRICTED_FLOAT_DTYPE(X) \
    FOREACH_RESTRICTED_FLOAT_DTYPE(X)

#define FOREACH_BASIC_DTYPE(X) \
    FOREACH_NUMERIC_DTYPE(X) \
    X(mbarrier, 64, false, "An opaque dtype representing an mbarrier state.") \
    X(cluster_launch_control_token, 128, false, \
            "An opaque dtype representing a cluster launch control token value.") \
    X(tensor_map_descriptor, 8 * sizeof(CUtensorMap), false, \
            "An opaque dtype representing a tensor map descriptor.")


namespace _dtype_detail {
    enum class BasicDTypeEnum : uint32_t {
        Invalid = 0,
        #define BASIC_DTYPE_ENUM_ENTRY(name, _bitwidth, _signed, _doc) \
            name,
        FOREACH_BASIC_DTYPE(BASIC_DTYPE_ENUM_ENTRY)
        #undef BASIC_DTYPE_ENUM_ENTRY
    };

    // Define constants like `unsigned_int_first` and `unsigned_int_last`,
    // for quickly checking a dtype category.
    #define BASIC_DTYPE_ENUM(name, _bitwidth, _signed, _doc) \
        BasicDTypeEnum::name,

    #define BASIC_DTYPE_FIRST_LAST(category, xmacro) \
        static constexpr BasicDTypeEnum all_##category##_dtypes[] = { \
            xmacro(BASIC_DTYPE_ENUM) \
        }; \
        static constexpr uint32_t category##_first = static_cast<uint32_t>( \
                all_##category##_dtypes[0]); \
        static constexpr uint32_t category##_last = static_cast<uint32_t>( \
                all_##category##_dtypes[std::extent_v<decltype(all_##category##_dtypes)> - 1]);

    BASIC_DTYPE_FIRST_LAST(unsigned_int, FOREACH_UNSIGNED_INTEGRAL_DTYPE);
    BASIC_DTYPE_FIRST_LAST(signed_int, FOREACH_SIGNED_INTEGRAL_DTYPE);
    BASIC_DTYPE_FIRST_LAST(unrestricted_float, FOREACH_UNRESTRICTED_FLOAT_DTYPE);
    BASIC_DTYPE_FIRST_LAST(restricted_float, FOREACH_RESTRICTED_FLOAT_DTYPE);
    BASIC_DTYPE_FIRST_LAST(numeric, FOREACH_NUMERIC_DTYPE);
    BASIC_DTYPE_FIRST_LAST(basic, FOREACH_BASIC_DTYPE);

    #undef BASIC_DTYPE_ENUM
    #undef BASIC_DTYPE_FIRST_LAST

    extern const uint32_t basic_dtype_bitwidth[basic_last + 1];
    extern const bool basic_dtype_is_signed[basic_last + 1];

    bool is_derived_dtype_pointer(DType dtype, GlobalLock& lock);
    bool is_derived_dtype_foreign_pointer(DType dtype, GlobalLock& lock);
    uint32_t derived_dtype_bitwidth(DType dtype, GlobalLock& lock);
} // namespace _dtype_detail

// Define constants like
//     static constexpr DType k_int32 = ...;
#define BASIC_DTYPE_CONSTANT(name, _bitwidth, _signed, _doc) \
    static constexpr DType k_##name = {static_cast<uint32_t>(_dtype_detail::BasicDTypeEnum::name)};
FOREACH_BASIC_DTYPE(BASIC_DTYPE_CONSTANT)
#undef BASIC_DTYPE_CONSTANT

static constexpr uint32_t kFirstDerivedDTypeId = _dtype_detail::basic_last + 1;

static inline bool is_unsigned_integer_dtype(DType dtype) {
    return dtype.dtype_id >= _dtype_detail::unsigned_int_first
        && dtype.dtype_id <= _dtype_detail::unsigned_int_last;
}

static inline bool is_signed_integer_dtype(DType dtype) {
    return dtype.dtype_id >= _dtype_detail::signed_int_first
        && dtype.dtype_id <= _dtype_detail::signed_int_last;
}

static inline bool is_integer_dtype(DType dtype) {
    return is_unsigned_integer_dtype(dtype) || is_signed_integer_dtype(dtype);
}

Integer integer_dtype_min(DType dtype);

Integer integer_dtype_max(DType dtype);

const FloatFormat& float_dtype_format(DType dtype);

static inline bool is_unrestricted_float_dtype(DType dtype) {
    return dtype.dtype_id >= _dtype_detail::unrestricted_float_first
        && dtype.dtype_id <= _dtype_detail::unrestricted_float_last;
}

static inline bool is_restricted_float_dtype(DType dtype) {
    return dtype.dtype_id >= _dtype_detail::restricted_float_first
        && dtype.dtype_id <= _dtype_detail::restricted_float_last;
}

static inline bool is_float_dtype(DType dtype) {
    return is_unrestricted_float_dtype(dtype) || is_restricted_float_dtype(dtype);
}

static inline bool is_arithmetic_numeric_dtype(DType dtype) {
    return dtype == k_bool_ || is_integer_dtype(dtype) || is_unrestricted_float_dtype(dtype);
}

static inline bool is_pointer_dtype(DType dtype, GlobalLock& lock) {
    return dtype.dtype_id >= kFirstDerivedDTypeId
        && _dtype_detail::is_derived_dtype_pointer(dtype, lock);
}

static inline bool is_foreign_pointer_dtype(DType dtype, GlobalLock& lock) {
    return dtype.dtype_id >= kFirstDerivedDTypeId
        && _dtype_detail::is_derived_dtype_foreign_pointer(dtype, lock);
}

static inline bool is_numeric_dtype(DType dtype) {
    return dtype.dtype_id >= _dtype_detail::numeric_first
        && dtype.dtype_id <= _dtype_detail::numeric_last;
}

// Basic dtypes that are not numeric, e.g. mbarrier
static inline bool is_opaque_dtype(DType dtype) {
    return dtype.dtype_id > _dtype_detail::numeric_last
        && dtype.dtype_id <= _dtype_detail::basic_last;
}

enum class NumericDTypeCategory {
    Boolean,
    Integer,
    Float,
};

static inline NumericDTypeCategory numeric_dtype_category(DType dtype) {
    if (dtype == k_bool_) {
        return NumericDTypeCategory::Boolean;
    } else if (is_integer_dtype(dtype)) {
        return NumericDTypeCategory::Integer;
    } else if (is_float_dtype(dtype)) {
        return NumericDTypeCategory::Float;
    } else {
        CHECK_UNREACHABLE;
    }
}

static inline bool is_signed_numeric_dtype(DType dtype) {
    return is_numeric_dtype(dtype) && _dtype_detail::basic_dtype_is_signed[dtype.dtype_id];
}

std::optional<DType> promote_dtypes(DType a, DType b);

enum class DTypeConversionStatus {
    OK,
    NegativeNotRepresentable,
    NanNotRepresentable,
    InfinityNotRepresentable,
    IntegerOutOfRange,
};

DTypeConversionStatus encode_float(double value, DType dtype, uint64_t* out);

DTypeConversionStatus encode_int_as_float(uint64_t bits, bool is_signed, DType dtype,
                                          uint64_t* out);

Result<DTypeConversionStatus> encode_pylong_as_float(PyObject* pylong, DType dtype, uint64_t* out);

double decode_float(uint64_t bits, DType dtype);

DType pointer_dtype_pointee(DType pointer_dtype, GlobalLock& lock);

DType foreign_pointer_dtype_pointee(DType foreign_pointer_dtype, GlobalLock& lock);

MemorySpace pointer_dtype_memory_space(DType pointer_dtype, GlobalLock& lock);

static inline uint32_t basic_dtype_bitwidth(DType dtype) {
    CHECK(dtype && dtype.dtype_id < kFirstDerivedDTypeId);
    return _dtype_detail::basic_dtype_bitwidth[dtype.dtype_id];
}

static inline uint32_t dtype_bitwidth(DType dtype, GlobalLock& lock) {
    CHECK(dtype);
    if (dtype.dtype_id < kFirstDerivedDTypeId)
        return _dtype_detail::basic_dtype_bitwidth[dtype.dtype_id];

    return _dtype_detail::derived_dtype_bitwidth(dtype, lock);
}

int append_to_string_builder(DType dtype, StringBuilder* sb);

PyObject* dtype_to_pyobject(DType dtype, GlobalLock& lock);

// Returns std::nullopt if `obj` is not a DType object
std::optional<DType> dtype_from_pyobject(PyObject* obj);

Status dtype_init(PyObject* m);
