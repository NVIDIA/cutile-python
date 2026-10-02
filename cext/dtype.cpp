// SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0

#include "dtype.h"
#include "vec.h"
#include "py.h"

#if defined(_MSC_VER)
#include <intrin.h>
#endif

static PyObject* g_module;
static PyObject* g_pointee_dtype_pyunicode;
static PyObject* g_memory_space_pyunicode;


// ----- FloatFormat constants -----

#define IEEE_LIKE(emin, emax, bitwidth, exp_bits, precision) \
    { emin, emax, bitwidth, exp_bits, precision, FloatFormat::IEEE, true, true }


// The bitwidth of tfloat32 is that of its encoding (19 bits), not its 32-bit container.
static constexpr FloatFormat
    k_format_float16        = IEEE_LIKE(-14, 15, 16, 5, 10),
    k_format_float32        = IEEE_LIKE(-126, 127, 32, 8, 23),
    k_format_float64        = IEEE_LIKE(-1022, 1023, 64, 11, 52),
    k_format_bfloat16       = IEEE_LIKE(-126, 127, 16, 8, 7),
    k_format_tfloat32       = IEEE_LIKE(-126, 127, 19, 8, 10),
    k_format_float8_e5m2    = IEEE_LIKE(-14, 15, 8, 5, 2),
    k_format_float8_e4m3fn  = { -6, 8, 8, 4, 3, FloatFormat::NanOnlyAllOnes, true, true },
    k_format_float8_e5m3fnu = { -14, 16, 8, 5, 3, FloatFormat::NanOnlyAllOnes, false, true },
    k_format_float8_e8m0fnu = { -127, 127, 8, 8, 0, FloatFormat::NanOnlyAllOnes, false, false },
    k_format_float4_e2m1fn  = { 0, 2, 4, 2, 1, FloatFormat::FiniteOnly, true, true },
    k_format_float6_e2m3fn  = { 0, 2, 6, 2, 3, FloatFormat::FiniteOnly, true, true },
    k_format_float6_e3m2fn  = { -2, 4, 6, 3, 2, FloatFormat::FiniteOnly, true, true };

#undef IEEE_LIKE

static constexpr FloatFormat float_formats[] = {
    #define FLOAT_FORMAT_ENTRY(name, _bitwidth, _signed, _doc) k_format_##name,
    FOREACH_UNRESTRICTED_FLOAT_DTYPE(FLOAT_FORMAT_ENTRY)
    FOREACH_RESTRICTED_FLOAT_DTYPE(FLOAT_FORMAT_ENTRY)
    #undef FLOAT_FORMAT_ENTRY
};

static_assert(std::extent_v<decltype(float_formats)>
        == _dtype_detail::restricted_float_last - _dtype_detail::unrestricted_float_first + 1);

static constexpr bool float_format_bitwidths_are_consistent() {
    for (const FloatFormat& f : float_formats) {
        if (f.have_sign + f.exp_bits + f.precision != f.bitwidth)
            return false;
    }
    return true;
}

static_assert(float_format_bitwidths_are_consistent());


// ----- DType implementation -----

namespace _dtype_detail {
    const uint32_t basic_dtype_bitwidth[basic_last + 1] = {
        0,
        #define BASIC_DTYPE_BITWIDTH(_name, bitwidth, _signed, _doc) \
            bitwidth,
        FOREACH_BASIC_DTYPE(BASIC_DTYPE_BITWIDTH)
        #undef BASIC_DTYPE_BITWIDTH
    };
    const bool basic_dtype_is_signed[basic_last + 1] = {
        false,
        #define BASIC_DTYPE_IS_SIGNED(_name, _bitwidth, signed, _doc) \
            signed,
        FOREACH_BASIC_DTYPE(BASIC_DTYPE_IS_SIGNED)
        #undef BASIC_DTYPE_IS_SIGNED
    };
} // namespace _dtype_detail

enum class DerivedDTypeKind : uint8_t {
    Pointer = 1,
    ForeignPointer,
};

struct DerivedDType {
    DerivedDTypeKind kind;
    MemorySpace memory_space;  // when kind is `Pointer`
    DType pointee_dtype;  // when kind is `Pointer` or `ForeignPointer`
};

struct DTypeRegistry {
    Vec<DType> pointer_dtypes[enum_max_value<MemorySpace> + 1];
    Vec<DType> foreign_pointer_dtypes;
    Vec<DerivedDType> derived_dtypes;
    Vec<PyObject*> names;
    Vec<PyObject*> pyobjects;

    static DTypeRegistry& get(GlobalLock& lock) {
        DTypeRegistry*& ptr = instance_.get(lock);
        if (!ptr) ptr = new DTypeRegistry;
        return *ptr;
    }

    static ProtectedByGlobalLock<DTypeRegistry*> instance_;
private:
    DTypeRegistry() = default;
    ~DTypeRegistry();  // DTypeRegistry is an immortal singleton
};

ProtectedByGlobalLock<DTypeRegistry*> DTypeRegistry::instance_;


Integer integer_dtype_min(DType dtype) {
    if (is_signed_integer_dtype(dtype)) {
        uint32_t width = _dtype_detail::basic_dtype_bitwidth[dtype.dtype_id];
        CHECK(width <= 64);
        uint64_t bits = static_cast<uint64_t>(-1) << (width - 1);
        return Integer{.s = static_cast<int64_t>(bits)};
    } else {
        CHECK(is_unsigned_integer_dtype(dtype));
        return Integer{.u = 0};
    }
}

Integer integer_dtype_max(DType dtype) {
    uint32_t width = _dtype_detail::basic_dtype_bitwidth[dtype.dtype_id];
    CHECK(width <= 64);
    if (is_signed_integer_dtype(dtype)) {
        uint64_t bits = (static_cast<uint64_t>(1) << (width - 1)) - 1;
        return Integer{.s = static_cast<int64_t>(bits)};
    } else {
        CHECK(is_unsigned_integer_dtype(dtype));
        uint64_t bits = (static_cast<uint64_t>(-1)) >> (64 - width);
        return Integer{.u = bits};
    }
}

const FloatFormat& float_dtype_format(DType dtype) {
    CHECK(is_float_dtype(dtype));
    return float_formats[dtype.dtype_id - _dtype_detail::unrestricted_float_first];
}

DType pointer_dtype(DType pointee_dtype, MemorySpace memory_space, GlobalLock& lock) {
    uint32_t space_idx = static_cast<uint32_t>(memory_space);
    CHECK(space_idx <= enum_max_value<MemorySpace>);
    DTypeRegistry& reg = DTypeRegistry::get(lock);
    Vec<DType>& pointer_dtypes = reg.pointer_dtypes[space_idx];

    if (pointee_dtype.dtype_id >= pointer_dtypes.size())
        pointer_dtypes.resize(pointee_dtype.dtype_id + 1);

    DType& ret = pointer_dtypes[pointee_dtype.dtype_id];
    if (!ret) {
        uint64_t dtype_id = static_cast<uint64_t>(reg.derived_dtypes.size()) + kFirstDerivedDTypeId;
        CHECK(dtype_id <= UINT32_MAX);
        reg.derived_dtypes.push_back(DerivedDType{
                DerivedDTypeKind::Pointer, memory_space, pointee_dtype});
        ret.dtype_id = static_cast<uint32_t>(dtype_id);
    }
    return ret;
}

DType foreign_pointer_dtype(DType pointee_dtype, GlobalLock& lock) {
    DTypeRegistry& reg = DTypeRegistry::get(lock);
    if (pointee_dtype.dtype_id >= reg.foreign_pointer_dtypes.size())
        reg.foreign_pointer_dtypes.resize(pointee_dtype.dtype_id + 1);

    DType& ret = reg.foreign_pointer_dtypes[pointee_dtype.dtype_id];
    if (!ret) {
        uint64_t dtype_id = static_cast<uint64_t>(reg.derived_dtypes.size()) + kFirstDerivedDTypeId;
        CHECK(dtype_id <= UINT32_MAX);
        reg.derived_dtypes.push_back(DerivedDType{
                DerivedDTypeKind::ForeignPointer, MemorySpace::GENERIC, pointee_dtype});
        ret.dtype_id = static_cast<uint32_t>(dtype_id);
    }
    return ret;
}

static const DerivedDType& get_derived_dtype(DType dtype, GlobalLock& lock) {
    DTypeRegistry& reg = DTypeRegistry::get(lock);
    uint32_t derived_idx = dtype.dtype_id - kFirstDerivedDTypeId;
    CHECK(derived_idx < reg.derived_dtypes.size());
    return reg.derived_dtypes[derived_idx];
}

bool _dtype_detail::is_derived_dtype_pointer(DType dtype, GlobalLock& lock) {
    return get_derived_dtype(dtype, lock).kind == DerivedDTypeKind::Pointer;
}

bool _dtype_detail::is_derived_dtype_foreign_pointer(DType dtype, GlobalLock& lock) {
    return get_derived_dtype(dtype, lock).kind == DerivedDTypeKind::ForeignPointer;
}

DType pointer_dtype_pointee(DType pointer_dtype, GlobalLock& lock) {
    const DerivedDType& derived = get_derived_dtype(pointer_dtype, lock);
    CHECK(derived.kind == DerivedDTypeKind::Pointer);
    return derived.pointee_dtype;
}

MemorySpace pointer_dtype_memory_space(DType pointer_dtype, GlobalLock& lock) {
    const DerivedDType& derived = get_derived_dtype(pointer_dtype, lock);
    CHECK(derived.kind == DerivedDTypeKind::Pointer);
    return derived.memory_space;
}

DType foreign_pointer_dtype_pointee(DType foreign_pointer_dtype, GlobalLock& lock) {
    const DerivedDType& derived = get_derived_dtype(foreign_pointer_dtype, lock);
    CHECK(derived.kind == DerivedDTypeKind::ForeignPointer);
    return derived.pointee_dtype;
}

uint32_t _dtype_detail::derived_dtype_bitwidth(DType dtype, GlobalLock& lock) {
    const DerivedDType& derived = get_derived_dtype(dtype, lock);
    switch (derived.kind) {
    case DerivedDTypeKind::Pointer:
        return memory_space_pointer_bitwidth(derived.memory_space);
    case DerivedDTypeKind::ForeignPointer:
        return 64;
    }
    CHECK_UNREACHABLE;
}

static const char* basic_dtype_name(_dtype_detail::BasicDTypeEnum basic) {
    switch (basic) {
        case _dtype_detail::BasicDTypeEnum::Invalid: return "<Invalid DType>";

        #define BASIC_DTYPE_NAME(name, _width, _signed, _doc) \
            case _dtype_detail::BasicDTypeEnum::name: return #name;
        FOREACH_BASIC_DTYPE(BASIC_DTYPE_NAME);
        #undef BASIC_DTYPE_NAME
    }
    CHECK_UNREACHABLE;
}

static const char* basic_dtype_doc(_dtype_detail::BasicDTypeEnum basic) {
    switch (basic) {
        case _dtype_detail::BasicDTypeEnum::Invalid: return nullptr;

        #define BASIC_DTYPE_NAME(name, _width, _signed, doc) \
            case _dtype_detail::BasicDTypeEnum::name: return doc;
        FOREACH_BASIC_DTYPE(BASIC_DTYPE_NAME);
        #undef BASIC_DTYPE_NAME
    }
    CHECK_UNREACHABLE;
}

static PyPtr make_dtype_name(DType dtype, GlobalLock& lock) {
    if (dtype.dtype_id < kFirstDerivedDTypeId) {
        return steal(PyUnicode_FromString(basic_dtype_name(
                static_cast<_dtype_detail::BasicDTypeEnum>(dtype.dtype_id))));
    }

    const DerivedDType& derived = get_derived_dtype(dtype, lock);
    switch (derived.kind) {
    case DerivedDTypeKind::Pointer:
        {
            StringBuilder builder;
            DType pointee = pointer_dtype_pointee(dtype, lock);
            int num_params = 0;
            if (pointee) {
                CHECK(pointee.dtype_id < dtype.dtype_id);
                builder.append("pointer[");
                builder.append(pointee);
                num_params = 1;
            } else {
                builder.append("opaque_pointer");
            }

            MemorySpace space = pointer_dtype_memory_space(dtype, lock);
            if (space != MemorySpace::GENERIC) {
                builder.append(++num_params == 1 ? "[" : ", ");
                builder.append("MemorySpace.");
                builder.append(enum_str(space));
            }

            if (num_params) builder.append("]");
            return builder.build();
        }
    case DerivedDTypeKind::ForeignPointer:
        {
            DType pointee = foreign_pointer_dtype_pointee(dtype, lock);
            return to_pyunicode("foreign_pointer[", pointee, "]");
        }
    }
    CHECK_UNREACHABLE;
}

PyObject* dtype_name(DType dtype, GlobalLock& lock) {
    DTypeRegistry& reg = DTypeRegistry::get(lock);
    // Just fill all the names sequentially up to our type ID,
    // so that we don't need to deal with recursion.
    while (dtype.dtype_id >= reg.names.size()) {
        uint32_t next_id = reg.names.size();
        PyPtr name = make_dtype_name(DType{next_id}, lock);
        if (!name) return nullptr;
        reg.names.push_back(name.release());
    }
    return reg.names[dtype.dtype_id];
}

int append_to_string_builder(DType dtype, StringBuilder* sb) {
    GlobalLock lock;
    PyObject* name = dtype_name(dtype, lock);
    if (!name) return -1;
    sb->append(name);
    return 0;
}


// ----- DType promotion -----

namespace {
namespace promotion_table {
    // The table stores dtype IDs as uint8_t to save space
    static_assert(_dtype_detail::unrestricted_float_last <= UINT8_MAX);

    constexpr uint8_t b1 = k_bool_.dtype_id;
    constexpr uint8_t u8 = k_uint8.dtype_id, u16 = k_uint16.dtype_id,
                      u32 = k_uint32.dtype_id, u64 = k_uint64.dtype_id;
    constexpr uint8_t i8 = k_int8.dtype_id, i16 = k_int16.dtype_id,
                      i32 = k_int32.dtype_id, i64 = k_int64.dtype_id;
    constexpr uint8_t f16 = k_float16.dtype_id, f32 = k_float32.dtype_id,
                      f64 = k_float64.dtype_id, bf = k_bfloat16.dtype_id;
    constexpr uint8_t na = 0;

    // Arithmetic dtype IDs, in increasing order
    constexpr uint8_t dtypes[] = {b1, u8, u16, u32, u64, i8, i16, i32, i64, f16, f32, f64, bf};
    constexpr size_t N = std::size(dtypes);

    constexpr uint8_t table[N][N] = {
        //b1   u8   u16  u32  u64  i8   i16  i32  i64  f16  f32  f64  bf
        { b1 , u8 , u16, u32, u64, i8 , i16, i32, i64, f16, f32, f64, bf },  // b1
        { u8 , u8 , u16, u32, u64, na , na , na , na , f16, f32, f64, bf },  // u8
        { u16, u16, u16, u32, u64, na , na , na , na , f16, f32, f64, bf },  // u16
        { u32, u32, u32, u32, u64, na , na , na , na , f16, f32, f64, bf },  // u32
        { u64, u64, u64, u64, u64, na , na , na , na , f16, f32, f64, bf },  // u64
        { i8 , na , na , na , na , i8 , i16, i32, i64, f16, f32, f64, bf },  // i8
        { i16, na , na , na , na , i16, i16, i32, i64, f16, f32, f64, bf },  // i16
        { i32, na , na , na , na , i32, i32, i32, i64, f16, f32, f64, bf },  // i32
        { i64, na , na , na , na , i64, i64, i64, i64, f16, f32, f64, bf },  // i64
        { f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f32, f64, na },  // f16
        { f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f64, f32},  // f32
        { f64, f64, f64, f64, f64, f64, f64, f64, f64, f64, f64, f64, f64},  // f64
        { bf , bf , bf , bf , bf , bf , bf , bf , bf , na , f32, f64, bf },  // bf
    };

    constexpr bool dtypes_are_contiguous() {
        for (size_t i = 0; i < N; ++i) {
            if (dtypes[i] != dtypes[0] + i)
                return false;
        }
        return true;
    }

    static_assert(dtypes_are_contiguous());
    static_assert(dtypes[0] == _dtype_detail::numeric_first);
    static_assert(dtypes[N - 1] == _dtype_detail::unrestricted_float_last);
} // namespace promotion_table
} // namespace

std::optional<DType> promote_dtypes(DType a, DType b) {
    if (a == b) return a;
    // Unsigned wraparound maps dtype IDs below the table's range to large indices
    uint32_t first = promotion_table::dtypes[0];
    uint32_t i = a.dtype_id - first;
    uint32_t j = b.dtype_id - first;
    if (i >= promotion_table::N || j >= promotion_table::N)
        return std::nullopt;
    uint8_t res = promotion_table::table[i][j];
    if (res == promotion_table::na) return std::nullopt;
    return DType{res};
}


// ----- Float encoding -----

static int bit_length(uint64_t x) {
#if defined(_MSC_VER)
    unsigned long index;
    return _BitScanReverse64(&index, x) ? int(index) + 1 : 0;
#else
    return x ? 64 - __builtin_clzll(x) : 0;
#endif
}

static DTypeConversionStatus double_nonfinite_to_bits(uint64_t f64_bits, DType dtype,
                                                      uint64_t* out) {
    const FloatFormat& fmt = float_dtype_format(dtype);
    uint64_t f64_sign = f64_bits & 0x8000'0000'0000'0000;
    uint64_t sign_bit = f64_sign >> (64 - fmt.bitwidth);
    bool is_nan = f64_bits & 0x000f'ffff'ffff'ffff;
    if (f64_sign && !is_nan && !fmt.have_sign)
        return DTypeConversionStatus::NegativeNotRepresentable;
    uint64_t all_ones_but_sign = (1ull << (fmt.bitwidth - 1)) - 1;

    switch (fmt.non_finite_behavior) {
    case FloatFormat::NanOnlyAllOnes:
        // Infinity also becomes NaN
        *out = fmt.have_sign ? (sign_bit | all_ones_but_sign) : (1ull << fmt.bitwidth) - 1;
        return DTypeConversionStatus::OK;
    case FloatFormat::FiniteOnly:
        // Infinity saturates to the largest magnitude
        if (is_nan)
            return DTypeConversionStatus::NanNotRepresentable;
        *out = sign_bit | all_ones_but_sign;
        return DTypeConversionStatus::OK;
    case FloatFormat::IEEE:
        {
            // Keep the sign and the all-ones exponent, and truncate the NaN payload.
            // A signaling NaN could lose all of its payload bits and become an infinity,
            // so it gets a nonzero payload instead.
            uint64_t sign_and_exp = (f64_bits >> (63 - fmt.exp_bits)) << fmt.precision;
            uint64_t payload;
            if (is_nan && !(f64_bits & (1ull << 51)))
                payload = 1;
            else
                payload = (f64_bits >> (52 - fmt.precision)) & ((1ull << fmt.precision) - 1);
            *out = sign_and_exp | payload;
            return DTypeConversionStatus::OK;
        }
    }
    CHECK_UNREACHABLE;
}


static uint64_t first_invalid_float_value(const FloatFormat& fmt) {
    switch (fmt.non_finite_behavior) {
    case FloatFormat::IEEE:
        return ((1ull << fmt.exp_bits) - 1) << fmt.precision;
    case FloatFormat::NanOnlyAllOnes:
        return (1ull << (fmt.exp_bits + fmt.precision)) - 1;
    case FloatFormat::FiniteOnly:
        return 1ull << (fmt.exp_bits + fmt.precision);
    }
    CHECK_UNREACHABLE;
}


// 64-bit unsigned integer interpreted as a binary fraction in [0, 2):
//     y.xxxxxxx xxxxxxxx xxxxxxxx xxxxxxxx xxxxxxxx xxxxxxxx xxxxxxxx xxxxxxxx
//       `--------------------------- 63 bits --------------------------------'
struct U64Fraction {
    uint64_t val_;

    static constexpr uint64_t kOne = 1ull << 63;

    explicit U64Fraction(uint64_t val) : val_(val) {
        CHECK(val_ >= kOne);
    }

    inline uint64_t lshift(size_t n) {
        if (n == 0) {
            return 0;
        } else if (n == 64) {
            uint64_t ret = val_;
            val_ = 0;
            return ret;
        } else {
            uint64_t ret = val_ >> (64 - n);
            val_ <<= n;
            return ret;
        }
    }

    bool greater_than_one() const { return val_ > kOne; }
    bool equals_one() const { return val_ == kOne; }
};

static uint64_t uint8_split(uint8_t* val, size_t bits_to_keep) {
    size_t ret = (*val) >> bits_to_keep;
    *val &= (1 << bits_to_keep) - 1;
    return ret;
}

// Like U64Fraction, but with arbitrarily many bits
struct LongFraction {
    Vec<uint8_t> data_;
    uint64_t num_bits_;

    // Number of bits in data_.back(), between 1 and 8
    size_t top_byte_bits() const {
        return ((num_bits_ - 1) & 7) + 1;
    }

    uint64_t lshift(size_t n) {
        if (num_bits_ == 0)
            return 0;

        if (n > num_bits_)
            n = num_bits_;
        size_t num_hi_bits = top_byte_bits();
        num_bits_ -= n;
        if (n < num_hi_bits)
            return uint8_split(&data_.back(), num_hi_bits - n);

        uint64_t ret = data_.back();
        data_.pop_back();
        n -= num_hi_bits;
        while (n >= 8) {
            ret = (ret << 8) | data_.back();
            data_.pop_back();
            n -= 8;
        }

        if (n > 0)
            ret = (ret << n) | uint8_split(&data_.back(), 8 - n);
        return ret;
    }

    bool greater_than_one() const {
        if (!num_bits_) return false;
        unsigned half = 1u << (top_byte_bits() - 1);
        if (data_.back() != half) return data_.back() > half;
        for (size_t i = 0; i + 1 < data_.size(); ++i) {
            if (data_[i]) return true;
        }
        return false;
    }

    bool equals_one() const {
        if (!num_bits_) return false;
        for (size_t i = 0; i + 1 < data_.size(); ++i) {
            if (data_[i]) return false;
        }
        return data_.back() == (1u << (top_byte_bits() - 1));
    }
};


// The top bit of the significand must be explicitly set to 1.
// Rounding mode is "to nearest, ties to even".
template <typename Frac>
static DTypeConversionStatus encode_float_parts(bool sign, Py_ssize_t e, Frac&& significand,
                                                DType dtype, uint64_t* out) {
    const FloatFormat& fmt = float_dtype_format(dtype);
    if (sign && !fmt.have_sign)
        return DTypeConversionStatus::NegativeNotRepresentable;

    uint64_t f64_inf = (uint64_t(sign) << 63) | 0x7ff0'0000'0000'0000;
    if (e > fmt.emax)
        return double_nonfinite_to_bits(f64_inf, dtype, out);

    // Round the significand to `fmt.precision` (plus one leading 1) bits
    Py_ssize_t bits_to_keep = fmt.precision + 1;
    if (e < fmt.emin) {
        if (!fmt.have_zero_and_subnormals) {
            *out = 0;  // round up to the smallest value
            return DTypeConversionStatus::OK;
        }
        // Subnormal
        bits_to_keep -= fmt.emin - e;
        e = fmt.emin;
    }

    uint64_t u64_significand;
    if (bits_to_keep < 0) {
        u64_significand = 0;  // less than half of the smallest subnormal
    } else {
        u64_significand = significand.lshift(bits_to_keep);
        // Round to nearest, ties to even
        if (significand.greater_than_one() || (significand.equals_one() && (u64_significand & 1)))
            ++u64_significand;
    }

    // These two lines are doing a lot of work. We intentionally encode the biased exponent
    // as one less the actual value, so that the explicit leading 1 in the `u64_significand` will
    // increment it by one. This automatically takes care of the subnormals, as well as the
    // rounding to the next power of two.
    uint64_t biased_exp_minus_one = e - fmt.emin - !fmt.have_zero_and_subnormals;
    uint64_t enc = (biased_exp_minus_one << fmt.precision) + u64_significand;

    // Check that rounding didn't overflow into the non-finite encodings
    if (enc >= first_invalid_float_value(fmt))
        return double_nonfinite_to_bits(f64_inf, dtype, out);

    *out = (uint64_t(sign) << (fmt.bitwidth - 1)) | enc;
    return DTypeConversionStatus::OK;
}

static double bitcast_u64_to_double(uint64_t bits) {
    union { uint64_t u64; double dbl; } u = {.u64 = bits};
    return u.dbl;
}

static uint64_t bitcast_double_to_u64(double val) {
    union { double dbl; uint64_t u64; } u = {.dbl = val};
    return u.u64;
}

DTypeConversionStatus encode_float(double value, DType dtype, uint64_t* out) {
    uint64_t f64_bits = bitcast_double_to_u64(value);
    if (dtype == k_float64) {
        *out = f64_bits;
        return DTypeConversionStatus::OK;
    }

    const FloatFormat& fmt = float_dtype_format(dtype);

    int e = (f64_bits >> 52) & 0x7ff;
    if (e == 0x7ff)
        return double_nonfinite_to_bits(f64_bits, dtype, out);

    bool sign = f64_bits >> 63;
    if (sign && !fmt.have_sign)
        return DTypeConversionStatus::NegativeNotRepresentable;

    // Calculate a 53-bit significand with the bit #52 set
    uint64_t significand = f64_bits & 0x000f'ffff'ffff'ffff;
    if (e) {
        // Normal float64: add the implicit bit
        significand |= 0x0010'0000'0000'0000;
        e -= 1023;
    } else {
        if (!significand) {
            // Zero or negative zero. Without a zero, round up to the smallest value.
            *out = fmt.have_zero_and_subnormals ? uint64_t(sign) << (fmt.bitwidth - 1) : 0;
            return DTypeConversionStatus::OK;
        }
        // Subnormal float64: there is no implicit one, so we need to "normalize" the significand.
        int shift = 53 - bit_length(significand);
        significand <<= shift;
        e = -1022 - shift;
    }
    // Turn the significand into a 64-bit value with the top bit set,
    // so that it's compatible with encode_float_parts().
    U64Fraction frac{significand << 11};
    return encode_float_parts(sign, e, frac, dtype, out);
}

DTypeConversionStatus encode_int_as_float(uint64_t bits, bool is_signed, DType dtype,
                                          uint64_t* out) {
    bool negative = is_signed && int64_t(bits) < 0;
    uint64_t significand = negative ? 0 - bits : bits;  // works for INT64_MIN
    if (!significand) {
        *out = 0;  // +0.0, or the smallest value in a format without a zero
        return DTypeConversionStatus::OK;
    }
    int e = 63;

    // Normalize the significand so that the top bit is set
    int shift = 64 - bit_length(significand);
    significand <<= shift;
    e -= shift;
    U64Fraction frac(significand);
    return encode_float_parts(negative, e, frac, dtype, out);
}

Result<DTypeConversionStatus> encode_pylong_as_float(PyObject* pylong, DType dtype, uint64_t* out) {
    int overflow;
    int64_t i64_val = pylong_as_overflow_and<int64_t>(pylong, &overflow);
    if (PyErr_Occurred()) return ErrorRaised;
    if (!overflow)
        return encode_int_as_float(i64_val, true, dtype, out);

    PyPtr absval;
    bool sign = false;
    if (overflow < 0) {
        sign = true;
        absval = steal(PyNumber_Negative(pylong));
        if (!absval) return ErrorRaised;
        pylong = absval.get();
    }
    Result<Vec<uint8_t>> bytes_res = pylong_as_unsigned_bytes(pylong);
    if (!bytes_res.is_ok()) return ErrorRaised;
    uint64_t nbits = pylong_num_bits(pylong);
    CHECK(nbits > 0);  // we wouldn't have overflown otherwise
    LongFraction frac = { std::move(*bytes_res), nbits };
    CHECK((nbits + 7) / 8 == frac.data_.size());
    Py_ssize_t e = static_cast<Py_ssize_t>(nbits) - 1;
    return encode_float_parts(sign, e, frac, dtype, out);
}

double decode_float(uint64_t bits, DType dtype) {
    uint64_t enc = bits;
    if (dtype == k_float64)
        return bitcast_u64_to_double(enc);

    const FloatFormat& fmt = float_dtype_format(dtype);
    uint64_t exp_mask = (1ull << fmt.exp_bits) - 1;
    uint64_t biased_exp = (enc >> fmt.precision) & exp_mask;
    uint64_t mantissa_mask = (1ull << fmt.precision) - 1;
    uint64_t mantissa = enc & mantissa_mask;

    uint64_t f64_bits = fmt.have_sign ? (enc << (64 - fmt.bitwidth)) & 0x8000'0000'0000'0000 : 0;

    bool nonfinite = biased_exp == exp_mask;
    switch (fmt.non_finite_behavior) {
    case FloatFormat::IEEE:
        break;
    case FloatFormat::NanOnlyAllOnes:
        if (mantissa != mantissa_mask) nonfinite = false;
        break;
    case FloatFormat::FiniteOnly:
        nonfinite = false;
        break;
    }

    uint64_t f64_exp;
    if (nonfinite) {
        f64_exp = 0x7ff;
        if (fmt.precision == 0)
            f64_bits |= 0x0008'0000'0000'0000;
    } else if (biased_exp == 0 && fmt.have_zero_and_subnormals) {
        if (mantissa != 0) {
            // Subnormal number becomes normal
            int shift = fmt.precision - bit_length(mantissa) + 1;
            mantissa = (mantissa << shift) & mantissa_mask;
            f64_exp = fmt.emin + 1023 - shift;
        } else {
            f64_exp = 0;
        }
    } else {
        f64_exp = (biased_exp - fmt.have_zero_and_subnormals) + fmt.emin + 1023;
    }

    f64_bits |= (f64_exp << 52) | (mantissa << (52 - fmt.precision));
    return bitcast_u64_to_double(f64_bits);
}


// ----- DType Python wrapper object -----

static PyPtr dtype_pyobject_create(DType dtype) {
    PyPtr obj = py_create_object<DTypeObject>();
    if (obj) py_unwrap<DTypeObject>(obj).dtype = dtype;
    return obj;
}


PyObject* dtype_to_pyobject(DType dtype, GlobalLock& lock) {
    if (!dtype) {
        raise(PyExc_ValueError, "Invalid dtype");
        return nullptr;
    }
    DTypeRegistry& reg = DTypeRegistry::get(lock);
    if (dtype.dtype_id >= reg.pyobjects.size())
        reg.pyobjects.resize(dtype.dtype_id + 1);
    PyObject*& obj = reg.pyobjects[dtype.dtype_id];
    if (!obj) obj = dtype_pyobject_create(dtype).release();
    return obj;
}


std::optional<DType> dtype_from_pyobject(PyObject* obj) {
    if (!PyObject_TypeCheck(obj, &DTypeObject::pytype))
        return std::nullopt;
    return py_unwrap<DTypeObject>(obj).dtype;
}


static PyObject* DType_reduce(PyObject* self, PyObject*) {
    GlobalLock lock;
    DType dtype = py_unwrap<DTypeObject>(self).dtype;
    if (dtype.dtype_id < kFirstDerivedDTypeId) {
        PyPtr func = getattr(g_module, "_unpickle_basic_dtype");
        if (!func) return nullptr;
        PyObject* name = dtype_name(dtype, lock);
        if (!name) return nullptr;
        return Py_BuildValue("(O(O))", func.get(), name);
    } else {
        const DerivedDType& derived = get_derived_dtype(dtype, lock);
        switch (derived.kind) {
        case DerivedDTypeKind::Pointer:
            {
                PyPtr func = getattr(g_module, "_get_pointer_dtype");
                if (!func) return nullptr;
                PyObject* pointee = derived.pointee_dtype
                    ? dtype_to_pyobject(derived.pointee_dtype, lock) : Py_None;
                if (!pointee) return nullptr;
                PyObject* space = enum_to_pyobject(derived.memory_space);
                if (!space) return nullptr;
                return Py_BuildValue("(O(OO))", func.get(), pointee, space);
            }
        case DerivedDTypeKind::ForeignPointer:
            {
                PyPtr func = getattr(g_module, "_get_foreign_pointer_dtype");
                if (!func) return nullptr;
                PyObject* pointee = dtype_to_pyobject(derived.pointee_dtype, lock);
                if (!pointee) return nullptr;
                return Py_BuildValue("(O(O))", func.get(), pointee);
            }
        }
        CHECK_UNREACHABLE;
    }
}

static PyMethodDef DType_methods[] = {
    {"__reduce__", DType_reduce, METH_NOARGS, nullptr},
    {}
};

static PyObject* DType_get_name(PyObject* self, void* closure) {
    GlobalLock lock;
    return Py_XNewRef(dtype_name(py_unwrap<DTypeObject>(self).dtype, lock));
}

static PyObject* DType_get_module(PyObject* self, void* closure) {
    return PyUnicode_FromString("cuda.tile");
}

static PyObject* DType_get_bitwidth(PyObject* self, void* closure) {
    GlobalLock lock;
    DType dtype = py_unwrap<DTypeObject>(self).dtype;
    return PyLong_FromUnsignedLong(dtype_bitwidth(dtype, lock));
}

static PyObject* DType_repr(PyObject* self) {
    return to_pyunicode("<DType '", py_unwrap<DTypeObject>(self).dtype, "'>").release();
}

static PyObject* DType_str(PyObject* self) {
    GlobalLock lock;
    return Py_XNewRef(dtype_name(py_unwrap<DTypeObject>(self).dtype, lock));
}

static PyObject* DType_get_doc(PyObject* self, void* closure) {
    GlobalLock lock;
    DType dtype = py_unwrap<DTypeObject>(self).dtype;
    if (is_integer_dtype(dtype)) {
        StringBuilder sb;
        const char* signedness = is_signed_integer_dtype(dtype) ? "signed" : "unsigned";

        bool is_signed = is_signed_integer_dtype(dtype);
        sb.append_many(dtype_bitwidth(dtype, lock), "-bit ", signedness,
                       " |arithmetic dtype| with values on the interval [");
        integer_dtype_min(dtype).append_to(&sb, is_signed);
        sb.append(", +");
        integer_dtype_max(dtype).append_to(&sb, is_signed);
        sb.append("]");
        return sb.build().release();
    }

    if (dtype.dtype_id < kFirstDerivedDTypeId) {
        const char* doc = basic_dtype_doc(
                static_cast<_dtype_detail::BasicDTypeEnum>(dtype.dtype_id));
        if (doc) return PyUnicode_FromString(doc);
    }
    raise(PyExc_AttributeError, "__doc__");
    return nullptr;
}

static PyObject *DType_call([[maybe_unused]] PyObject *self,
                            [[maybe_unused]] PyObject *args,
                            [[maybe_unused]] PyObject *kwargs) {
    raise(PyExc_TypeError, "DType cannot be constructed in pure Python");
    return nullptr;
}

static PyGetSetDef DType_getsetters[] = {
    {"name", DType_get_name, nullptr, "The name of the |data type|"},
    {"__name__", DType_get_name},
    {"__module__", DType_get_module},
    {"bitwidth", DType_get_bitwidth, nullptr,
     "The number of bits in an element of the |data type|"},
    {"__doc__", DType_get_doc},
    {}
};

PyTypeObject DTypeObject::pytype = {
    .tp_name = "cuda.tile.DType",
    .tp_basicsize = sizeof(PythonWrapper<DTypeObject>),
    .tp_dealloc = pywrapper_dealloc<DTypeObject>,
    .tp_repr = DType_repr,
    .tp_call = DType_call,
    .tp_str = DType_str,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_doc =
        "A *data type* (or *dtype*) describes the type of the objects of an |array|, |tile|, "
        "or operation.\n"
        "\n"
        "|Dtypes| determine how values are stored in memory and how operations on those values are"
        "performed.\n",
    .tp_methods = DType_methods,
    .tp_getset = DType_getsetters,
};


// ----- Python wrappers for is_xxx(dtype) predicates -----

using DTypePredicate = bool(DType);
using DTypePredicateWithLock = bool(DType, GlobalLock&);

template <typename P>
static PyObject* is_whatever_impl(PyObject* dtype_obj, P&& pred) {
    if (!PyObject_TypeCheck(dtype_obj, &DTypeObject::pytype)) {
        raise(PyExc_TypeError, "Expected a DType object");
        return nullptr;
    }
    bool res = pred(py_unwrap<DTypeObject>(dtype_obj).dtype);
    return Py_NewRef(res ? Py_True : Py_False);
}

template <DTypePredicate P>
static PyObject* is_whatever(PyObject* module_self, PyObject* dtype_obj) {
    return is_whatever_impl(dtype_obj, P);
}

template <DTypePredicateWithLock P>
static PyObject* is_whatever(PyObject* module_self, PyObject* dtype_obj) {
    GlobalLock lock;
    return is_whatever_impl(dtype_obj, [&lock](DType x){return P(x, lock);});
}

static bool is_boolean_dtype(DType dtype) {
    return dtype == k_bool_;
}

// ----- Python wrappers for derived dtype construction -----

static PyObject* py_get_pointer_dtype(PyObject* module_self,
                                      PyObject* const* args, Py_ssize_t nargs) {
    if (nargs != 2) {
        raise(PyExc_TypeError, "Expected 2 positional arguments");
        return nullptr;
    }

    PyObject *py_pointee = args[0], *py_memory_space = args[1];

    DType pointee_dtype;
    if (py_pointee == Py_None) {
        pointee_dtype = DType::invalid();
    } else if (PyObject_TypeCheck(py_pointee, &DTypeObject::pytype)) {
        pointee_dtype = py_unwrap<DTypeObject>(py_pointee).dtype;
    } else {
        raise(PyExc_TypeError, "Expected a DType object or None for the first argument, got ",
              use_repr(py_pointee));
        return nullptr;
    }

    std::optional<MemorySpace> space = enum_from_pyobject<MemorySpace>(py_memory_space);
    if (!space.has_value()) {
        raise(PyExc_TypeError, "Expected a MemorySpace as the second argument, got ",
              use_repr(py_memory_space));
        return nullptr;
    }

    GlobalLock lock;
    DType res = pointer_dtype(pointee_dtype, *space, lock);
    return Py_XNewRef(dtype_to_pyobject(res, lock));
}

static PyObject* py_get_foreign_pointer_dtype(PyObject* module_self, PyObject* py_pointee) {
    if (!PyObject_TypeCheck(py_pointee, &DTypeObject::pytype)) {
        raise(PyExc_TypeError, "Expected a DType object for the first argument, got ",
              use_repr(py_pointee));
        return nullptr;
    }
    DType pointee_dtype = py_unwrap<DTypeObject>(py_pointee).dtype;
    GlobalLock lock;
    DType res = foreign_pointer_dtype(pointee_dtype, lock);
    return Py_XNewRef(dtype_to_pyobject(res, lock));
}


// ----- Python wrappers for dtype queries -----

static PyObject* py_integer_dtype_min(PyObject* module_self, PyObject* dtype_obj) {
    if (!PyObject_TypeCheck(dtype_obj, &DTypeObject::pytype)) {
        raise(PyExc_TypeError, "Expected a DType object, got ", use_repr(dtype_obj));
        return nullptr;
    }
    DType dtype = py_unwrap<DTypeObject>(dtype_obj).dtype;
    if (!is_integer_dtype(dtype)) {
        raise(PyExc_ValueError, "Expected an integer DType, got '", dtype, "'");
        return nullptr;
    }

    return integer_dtype_min(dtype).to_pylong(is_signed_integer_dtype(dtype)).release();
}

static PyObject* py_integer_dtype_max(PyObject* module_self, PyObject* dtype_obj) {
    if (!PyObject_TypeCheck(dtype_obj, &DTypeObject::pytype)) {
        raise(PyExc_TypeError, "Expected a DType object, got ", use_repr(dtype_obj));
        return nullptr;
    }
    DType dtype = py_unwrap<DTypeObject>(dtype_obj).dtype;
    if (!is_integer_dtype(dtype)) {
        raise(PyExc_ValueError, "Expected an integer DType, got '", dtype, "'");
        return nullptr;
    }

    return integer_dtype_max(dtype).to_pylong(is_signed_integer_dtype(dtype)).release();
}

static PyObject* py_promote_dtypes(PyObject* module_self,
                                   PyObject* const* args, Py_ssize_t nargs) {
    if (nargs != 2) {
        raise(PyExc_TypeError, "Expected 2 positional arguments");
        return nullptr;
    }
    for (Py_ssize_t i = 0; i < nargs; ++i) {
        if (!PyObject_TypeCheck(args[i], &DTypeObject::pytype)) {
            raise(PyExc_TypeError, "Expected a DType object, got ", use_repr(args[i]));
            return nullptr;
        }
    }
    std::optional<DType> res = promote_dtypes(py_unwrap<DTypeObject>(args[0]).dtype,
                                              py_unwrap<DTypeObject>(args[1]).dtype);
    if (!res.has_value()) return Py_NewRef(Py_None);
    GlobalLock lock;
    return Py_XNewRef(dtype_to_pyobject(*res, lock));
}

static Result<DType> parse_pointer_dtype(PyObject* py_pointer, GlobalLock& lock) {
    if (!PyObject_TypeCheck(py_pointer, &DTypeObject::pytype))
        return raise(PyExc_TypeError, "Expected a DType object for the first argument, got ",
                     use_repr(py_pointer));
    DType pointer_dtype = py_unwrap<DTypeObject>(py_pointer).dtype;
    if (!is_pointer_dtype(pointer_dtype, lock))
        return raise(PyExc_ValueError, pointer_dtype, " is not a pointer dtype");
    return pointer_dtype;
}

static PyObject* py_pointer_pointee_dtype(PyObject* module_self, PyObject* py_pointer) {
    GlobalLock lock;
    Result<DType> pointer_dtype = parse_pointer_dtype(py_pointer, lock);
    if (!pointer_dtype.is_ok()) return nullptr;
    DType pointee_dtype = pointer_dtype_pointee(*pointer_dtype, lock);
    if (!pointee_dtype)
        return Py_NewRef(Py_None);
    return Py_XNewRef(dtype_to_pyobject(pointee_dtype, lock));
}

static PyObject* py_pointer_memory_space(PyObject* module_self, PyObject* py_pointer) {
    GlobalLock lock;
    Result<DType> pointer_dtype = parse_pointer_dtype(py_pointer, lock);
    if (!pointer_dtype.is_ok()) return nullptr;
    MemorySpace space = pointer_dtype_memory_space(*pointer_dtype, lock);
    return Py_NewRef(enum_to_pyobject(space));
}

static PyObject* py_foreign_pointer_pointee_dtype(PyObject* module_self, PyObject* py_pointer) {
    GlobalLock lock;
    if (!PyObject_TypeCheck(py_pointer, &DTypeObject::pytype)) {
        raise(PyExc_TypeError, "Expected a DType object for the first argument, got ",
                use_repr(py_pointer));
        return nullptr;
    }
    DType foreign_pointer_dtype = py_unwrap<DTypeObject>(py_pointer).dtype;
    if (!is_foreign_pointer_dtype(foreign_pointer_dtype, lock)) {
        raise(PyExc_ValueError, foreign_pointer_dtype, " is not a pointer dtype");
        return nullptr;
    }
    DType pointee_dtype = foreign_pointer_dtype_pointee(foreign_pointer_dtype, lock);
    return Py_XNewRef(dtype_to_pyobject(pointee_dtype, lock));
}


// ----- Pickle/unpickle helper -----

static PyObject* _unpickle_basic_dtype(PyObject* module_self, PyObject* name) {
    return PyObject_GetAttr(module_self, name);
}


// ----- Module initialization -----

static PyMethodDef functions[] = {
    {"is_numeric", is_whatever<is_numeric_dtype>, METH_O, nullptr},
    {"is_boolean", is_whatever<is_boolean_dtype>, METH_O, nullptr},
    {"is_integral", is_whatever<is_integer_dtype>, METH_O, nullptr},
    {"is_signed", is_whatever<is_signed_numeric_dtype>, METH_O, nullptr},
    {"is_float", is_whatever<is_float_dtype>, METH_O, nullptr},
    {"is_unrestricted_float", is_whatever<is_unrestricted_float_dtype>, METH_O, nullptr},
    {"is_restricted_float", is_whatever<is_restricted_float_dtype>, METH_O, nullptr},
    {"is_arithmetic", is_whatever<is_arithmetic_numeric_dtype>, METH_O, nullptr},
    {"_is_pointer_dtype", is_whatever<is_pointer_dtype>, METH_O, nullptr},
    {"_is_foreign_pointer_dtype", is_whatever<is_foreign_pointer_dtype>, METH_O, nullptr},
    {"_get_pointer_dtype", reinterpret_cast<PyCFunction>(py_get_pointer_dtype),
     METH_FASTCALL, nullptr},
    {"_get_foreign_pointer_dtype", py_get_foreign_pointer_dtype, METH_O, nullptr},
    {"integer_dtype_min", py_integer_dtype_min, METH_O, nullptr},
    {"integer_dtype_max", py_integer_dtype_max, METH_O, nullptr},
    {"_promote_dtypes", reinterpret_cast<PyCFunction>(py_promote_dtypes),
     METH_FASTCALL, nullptr},
    {"_pointer_pointee_dtype", py_pointer_pointee_dtype, METH_O, nullptr},
    {"_pointer_memory_space", py_pointer_memory_space, METH_O, nullptr},
    {"_foreign_pointer_pointee_dtype", py_foreign_pointer_pointee_dtype, METH_O, nullptr},
    {"_unpickle_basic_dtype", _unpickle_basic_dtype, METH_O, nullptr},
    {}
};

Status dtype_init(PyObject* m) {
    GlobalLock lock;

    g_module = Py_NewRef(m);

    INIT_STRING_IDENT(pointee_dtype);
    INIT_STRING_IDENT(memory_space);

    if (PyType_Ready(&DTypeObject::pytype) < 0)
        return ErrorRaised;

    if (PyModule_AddObjectRef(m, "DType", reinterpret_cast<PyObject*>(&DTypeObject::pytype)) < 0)
        return ErrorRaised;

    if (PyModule_AddFunctions(m, functions) < 0)
        return ErrorRaised;

    PyObject* mod_dict = PyModule_GetDict(m);
    if (!mod_dict) return ErrorRaised;

    // Add dtypes by name
    for (_dtype_detail::BasicDTypeEnum basic_dtype : _dtype_detail::all_basic_dtypes) {
        DType dtype = {static_cast<uint32_t>(basic_dtype)};
        PyObject* name = dtype_name(dtype, lock);
        if (!name) return ErrorRaised;
        PyObject* obj = dtype_to_pyobject(dtype, lock);
        if (!obj) return ErrorRaised;
        if (PyDict_SetItem(mod_dict, name, obj) < 0)
            return ErrorRaised;
    }

    return OK;
}
