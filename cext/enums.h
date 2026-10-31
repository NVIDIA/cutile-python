// SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <cstdint>
#include <optional>
#include <utility>

#include <Python.h>


#define ENUM_ENTRY_AUTO(name, ...) name,
#define ENUM_ENTRY_WITH_ID(name, id, ...) name = id,

template <typename E>
const char* enum_name();

template <typename E>
const char* enum_str(E value);

template <typename E>
PyTypeObject* enum_pytype();

// Always returns non-null
template <typename E>
PyObject* enum_to_pyobject(E value);

template <typename E>
std::optional<E> enum_from_pyobject(PyObject* pyobj);

template <typename E>
std::optional<E> enum_from_pyunicode(PyObject* pyobj);


#define FOREACH_MEMORY_SPACE(X) \
    X(GENERIC, 0, 64) \
    X(GLOBAL, 1, 64) \
    X(SHARED, 3, 32) \
    X(CONSTANT, 4, 64) \
    X(LOCAL, 5, 64) \
    X(TENSOR, 6, 32) \
    X(SHARED_CLUSTER, 7, 32)

enum class MemorySpace : uint8_t {
    FOREACH_MEMORY_SPACE(ENUM_ENTRY_WITH_ID)
};


#define FOREACH_ROUNDING_MODE(X) \
    X(RN) \
    X(RZ) \
    X(RM) \
    X(RP) \
    X(RA) \
    X(FULL) \
    X(APPROX) \
    X(RZI)

enum class RoundingMode : uint8_t {
    FOREACH_ROUNDING_MODE(ENUM_ENTRY_AUTO)
};

#define FOREACH_ENUM(X) \
    X(RoundingMode, FOREACH_ROUNDING_MODE, "cuda.tile._numeric_semantics") \
    X(MemorySpace, FOREACH_MEMORY_SPACE, "cuda.tile._memory_model")

// Reflection helpers
namespace _enum_detail {
    template <typename E> struct is_reflection_enabled_enum : std::false_type {};

    template <typename E> struct EnumValues;

    #define ENUM_INTEGER_SEQ_HELPER(name, ...) , static_cast<T>(E::name)
    #define ENUM_VALUES_SPECIALIZATION(name, macro, py_module_name) \
        template <> struct EnumValues<name> { \
            using E = name; \
            using T = std::underlying_type_t<E>; \
            using Seq = std::integer_sequence<T macro(ENUM_INTEGER_SEQ_HELPER)>; \
        }; \
        template <> struct is_reflection_enabled_enum<name> : std::true_type {};


    FOREACH_ENUM(ENUM_VALUES_SPECIALIZATION)

    #undef ENUM_INTEGER_SEQ_HELPER
    #undef ENUM_VALUES_SPECIALIZATION

    template <typename S> struct intseq_head;

    template <typename T, T First, T... Tail>
    struct intseq_head<std::integer_sequence<T, First, Tail...>> {
        static constexpr T value = First;
    };

    template <typename T, T... Ints>
    constexpr T intseq_max_value(std::integer_sequence<T, Ints...>) {
        static_assert(sizeof...(Ints) > 0);
        return std::max({Ints...});
    }
} // namespace _enum_detail


template <typename E>
constexpr std::underlying_type_t<E> enum_first_value =
        _enum_detail::intseq_head<typename _enum_detail::EnumValues<E>::Seq>::value;

template <typename E>
constexpr std::underlying_type_t<E> enum_max_value =
        _enum_detail::intseq_max_value( typename _enum_detail::EnumValues<E>::Seq{});

template <typename E>
constexpr size_t enum_count = _enum_detail::EnumValues<E>::Seq::size();


int enums_module_init();

