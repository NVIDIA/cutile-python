// SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0

#include "enums.h"
#include "py.h"


static Status intern_strings(const char* const strings[], size_t count, PyObject** out) {
    for (size_t i = 0; i < count; ++i) {
        if (!(out[i] = PyUnicode_InternFromString(strings[i])))
            return ErrorRaised;
    }
    return OK;
}

static PyPtr import_enum_cls(const char* module_name, const char* enum_name) {
    PyPtr mod = steal(PyImport_ImportModule(module_name));
    if (!mod) return {};
    PyPtr ret = getattr(mod, enum_name);
    if (!ret) return {};
    if (!PyType_Check(ret.get())) {
        raise(PyExc_TypeError, module_name, ".", enum_name, " is not a type");
        return {};
    }
    return ret;
}

static Status get_enum_pyobjects(PyObject* cls, PyObject* item_names[], size_t count,
                                 PyObject** out) {

    for (size_t i = 0; i < count; ++i) {
        if (out[i]) continue;
        out[i] = getattr(cls, item_names[i]).release();
        if (!out[i]) return ErrorRaised;
    }
    return OK;
}

namespace _enum_detail {
namespace {
    template <typename E> struct EnumStrings;

    template <typename S> struct intseq_is_sequential;

    template <typename T, T Item>
    struct intseq_is_sequential<std::integer_sequence<T, Item>> : std::true_type {};

    template <typename T, T First, T Second, T... Tail>
    struct intseq_is_sequential<std::integer_sequence<T, First, Second, Tail...>> {
        static constexpr bool value = First + 1 == Second
                && intseq_is_sequential<std::integer_sequence<T, Second, Tail...>>::value;
    };

    template <typename T, T... Ints>
    constexpr std::array<T, sizeof...(Ints)> intseq_to_array(std::integer_sequence<T, Ints...>) {
        return {Ints...};
    }

    template <typename E>
    constexpr std::array<std::underlying_type_t<E>, enum_count<E>> all_values = intseq_to_array(
            typename EnumValues<E>::Seq{});

    template <typename E>
    constexpr bool kIsSequential = intseq_is_sequential<typename EnumValues<E>::Seq>::value;

    template <typename E>
    struct EnumImpl {
        static PyTypeObject* pytype;
        static PyObject* interned_names[enum_count<E>];
        static PyObject* pyobjects[enum_count<E>];

        // At module init time, thus no lock required
        static Status initialize() {
            if (!intern_strings(EnumStrings<E>::names, enum_count<E>, interned_names))
                return ErrorRaised;
            PyPtr cls = import_enum_cls(EnumStrings<E>::python_module_name,
                                        EnumStrings<E>::enum_name);
            if (!cls) return ErrorRaised;
            if (!get_enum_pyobjects(cls.get(), interned_names, enum_count<E>, pyobjects))
                return ErrorRaised;
            pytype = reinterpret_cast<PyTypeObject*>(cls.release());
            return OK;
        }
    };

    template <typename E>
    PyTypeObject* EnumImpl<E>::pytype;

    template <typename E>
    PyObject* EnumImpl<E>::interned_names[enum_count<E>];

    template <typename E>
    PyObject* EnumImpl<E>::pyobjects[enum_count<E>];

    #define ENUM_ITEM_NAME_STR(name, ...) #name,
    #define ENUM_STRINGS_SPECIALIZATION(name, macro, py_module_name) \
        template <> struct EnumStrings<name> { \
            static constexpr const char* names[] = { macro(ENUM_ITEM_NAME_STR) }; \
            static constexpr const char* enum_name = #name; \
            static constexpr const char* python_module_name = py_module_name; \
        }; \

    FOREACH_ENUM(ENUM_STRINGS_SPECIALIZATION)

    #undef ENUM_ITEM_NAME_STR
    #undef ENUM_STRINGS_SPECIALIZATION

} // anonymous namespace
} // namespace _enum_detail


template <typename E>
static size_t enum_value_to_index(E value) {
    using namespace _enum_detail;
    auto v = static_cast<std::underlying_type_t<E>>(value);
    if constexpr (kIsSequential<E>) {
        size_t ret = v - enum_first_value<E>;
        CHECK(ret < enum_count<E>);
        return ret;
    } else {
        for (size_t i = 0; i < enum_count<E>; ++i) {
            if (all_values<E>[i] == v)
                return i;
        }
        CHECK_UNREACHABLE;
    }
}

template <typename E>
static E enum_index_to_value(size_t index) {
    using namespace _enum_detail;
    if constexpr (kIsSequential<E>) {
        return static_cast<E>(enum_first_value<E> + index);
    } else {
        return static_cast<E>(all_values<E>[index]);
    }
}

template <typename E>
const char* enum_name() {
    return _enum_detail::EnumStrings<E>::enum_name;
}

template <typename E>
const char* enum_str(E value) {
    size_t idx = enum_value_to_index(value);
    return _enum_detail::EnumStrings<E>::names[idx];
}

template <typename E>
PyTypeObject* enum_pytype() {
    return _enum_detail::EnumImpl<E>::pytype;
}

template <typename E>
PyObject* enum_to_pyobject(E value) {
    size_t idx = enum_value_to_index(value);
    return _enum_detail::EnumImpl<E>::pyobjects[idx];
}

template <typename E>
std::optional<E> enum_from_pyobject(PyObject* pyobj) {
    using namespace _enum_detail;
    PyObject* const* pyobjects = _enum_detail::EnumImpl<E>::pyobjects;
    for (size_t i = 0; i < enum_count<E>; ++i) {
        if (pyobjects[i] == pyobj)
            return enum_index_to_value<E>(i);
    }
    return std::nullopt;
}

template <typename E>
std::optional<E> enum_from_pyunicode(PyObject* pyobj) {
    using namespace _enum_detail;
    PyPtr p = newref(pyobj);
    pyunicode_intern_in_place(&p);
    PyObject* const* interned_names = _enum_detail::EnumImpl<E>::interned_names;
    for (size_t i = 0; i < enum_count<E>; ++i) {
        if (interned_names[i] == p.get())
            return enum_index_to_value<E>(i);
    }
    return std::nullopt;
}

#define ENUM_INSTANTIATE(name, _macro, _module_name) \
    template const char* enum_name<name>(); \
    template const char* enum_str<name>(name); \
    template PyTypeObject* enum_pytype<name>(); \
    template PyObject* enum_to_pyobject<name>(name); \
    template std::optional<name> enum_from_pyobject<name>(PyObject*); \
    template std::optional<name> enum_from_pyunicode<name>(PyObject*);

FOREACH_ENUM(ENUM_INSTANTIATE)
#undef ENUM_INSTANTIATE


int enums_module_init() {
    #define ENUM_INIT(name, _macro, _module_name) \
        if (!_enum_detail::EnumImpl<name>::initialize()) return -1;
    FOREACH_ENUM(ENUM_INIT)
    #undef ENUM_INIT

    return 0;
}

