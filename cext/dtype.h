// SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "check.h"
#include "py.h"

#include <cuda.h>
#include <cstdint>
#include <type_traits>

// It may look like this file is overusing X macros, which it probably is.

class StringBuilder;

// ----- Memory space -----

#define FOREACH_MEMORY_SPACE(X) \
    X(GENERIC, 0, 64) \
    X(GLOBAL, 1, 64) \
    X(SHARED, 3, 32) \
    X(CONSTANT, 4, 64) \
    X(LOCAL, 5, 64) \
    X(TENSOR, 6, 32) \
    X(SHARED_CLUSTER, 7, 32)

enum class MemorySpace : uint8_t {
    #define MEMORY_SPACE_ENUM_ENTRY(name, id, _ptrwidth) \
        name = id,
    FOREACH_MEMORY_SPACE(MEMORY_SPACE_ENUM_ENTRY)
    #undef MEMORY_SPACE_ENUM_ENTRY
};

constexpr uint8_t kMemorySpaceMax = static_cast<uint8_t>(MemorySpace::SHARED_CLUSTER);

const char* memory_space_str(MemorySpace space);


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

struct Integer {
    uint64_t bits;
    bool is_signed;

    static inline constexpr Integer from_u64(uint64_t x) {
        return Integer{x, false};
    }

    static inline constexpr Integer from_i64(int64_t x) {
        return Integer{static_cast<uint64_t>(x), true};
    }

    PyPtr to_pylong() const {
        return steal(is_signed ? PyLong_FromLongLong(static_cast<int64_t>(bits))
                        : PyLong_FromUnsignedLongLong(bits));
    }

};

int append_to_string_builder(Integer integer, StringBuilder* sb);


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
    X(bfloat16, 16, true, \
        "A 16-bit floating-point |arithmetic dtype| with 1 sign bit, 8 exponent bits, " \
        "and 7 mantissa bits.") \
    X(float32, 32, true, \
        "A IEEE 754 single-precision (32-bit) binary floating-point |arithmetic dtype| " \
        "(see |IEEE 754-2019|).") \
    X(float64, 64, true, \
        "A IEEE 754 double-precision (64-bit) binary floating-point |arithmetic dtype| " \
        "(see |IEEE 754-2019|).")

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

static inline bool is_signed_numeric_dtype(DType dtype) {
    return is_numeric_dtype(dtype) && _dtype_detail::basic_dtype_is_signed[dtype.dtype_id];
}

DType pointer_dtype_pointee(DType pointer_dtype, GlobalLock& lock);

DType foreign_pointer_dtype_pointee(DType foreign_pointer_dtype, GlobalLock& lock);

MemorySpace pointer_dtype_memory_space(DType pointer_dtype, GlobalLock& lock);

static inline uint32_t dtype_bitwidth(DType dtype, GlobalLock& lock) {
    CHECK(dtype);
    if (dtype.dtype_id < kFirstDerivedDTypeId)
        return _dtype_detail::basic_dtype_bitwidth[dtype.dtype_id];

    return _dtype_detail::derived_dtype_bitwidth(dtype, lock);
}

int append_to_string_builder(DType dtype, StringBuilder* sb);

Status dtype_init(PyObject* m);
