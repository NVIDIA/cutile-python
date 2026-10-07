// SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0

#include "dtype.h"
#include "vec.h"
#include "py.h"

static PyObject* g_module;
static PyObject* g_pointee_dtype_pyunicode;
static PyObject* g_memory_space_pyunicode;


int append_to_string_builder(Integer integer, StringBuilder* sb) {
    if (integer.is_signed) {
        sb->append(static_cast<int64_t>(integer.bits));
    } else {
        sb->append(static_cast<uint64_t>(integer.bits));
    }
    return 0;
}


// ----- Memory space implementation -----

const char* memory_space_str(MemorySpace space) {
    switch (space) {
        #define MEMORY_SPACE_STR_CASE(name, _id, _ptrwidth) \
            case MemorySpace::name: return #name;
        FOREACH_MEMORY_SPACE(MEMORY_SPACE_STR_CASE)
        #undef MEMORY_SPACE_STR_CASE
    }
    CHECK_UNREACHABLE;
}

static constexpr MemorySpace all_memory_spaces[] = {
    #define MEMORY_SPACE_ENUMERATE(name, _id, _ptrwidth) \
        MemorySpace::name,
    FOREACH_MEMORY_SPACE(MEMORY_SPACE_ENUMERATE)
    #undef MEMORY_SPACE_ENUMERATE
};

static PyObject* const* get_memory_space_pyobjects(GlobalLock&) {
    static bool is_cached;
    static PyObject* cache[kMemorySpaceMax + 1];
    if (!is_cached) {
        PyPtr mod = steal(PyImport_ImportModule("cuda.tile._memory_model"));
        if (!mod) return nullptr;
        PyPtr cls = getattr(mod, "MemorySpace");
        if (!cls) return nullptr;

        for (MemorySpace space : all_memory_spaces) {
            uint8_t i = static_cast<uint8_t>(space);
            if (cache[i]) continue;
            cache[i] = getattr(cls, memory_space_str(space)).release();
            if (!cache[i]) return nullptr;
        }
        is_cached = true;
    }
    return cache;
}

static PyObject* memory_space_to_pyobject(MemorySpace space, GlobalLock& lock) {
    PyObject* const* objects = get_memory_space_pyobjects(lock);
    if (!objects) return nullptr;
    return objects[static_cast<uint8_t>(space)];
}

static std::optional<MemorySpace> memory_space_from_pyobject(PyObject* space_obj,
                                                             GlobalLock& lock) {
    PyObject* const* objects = get_memory_space_pyobjects(lock);
    for (MemorySpace space : all_memory_spaces) {
        if (objects[static_cast<uint8_t>(space)] == space_obj)
            return space;
    }
    return std::nullopt;
}


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
    Vec<DType> pointer_dtypes[kMemorySpaceMax + 1];
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
        return Integer::from_i64(static_cast<int64_t>(bits));
    } else {
        CHECK(is_unsigned_integer_dtype(dtype));
        return Integer::from_u64(0);
    }
}

Integer integer_dtype_max(DType dtype) {
    uint32_t width = _dtype_detail::basic_dtype_bitwidth[dtype.dtype_id];
    CHECK(width <= 64);
    if (is_signed_integer_dtype(dtype)) {
        uint64_t bits = (static_cast<uint64_t>(1) << (width - 1)) - 1;
        return Integer::from_i64(static_cast<int64_t>(bits));
    } else {
        CHECK(is_unsigned_integer_dtype(dtype));
        uint64_t bits = (static_cast<uint64_t>(-1)) >> (64 - width);
        return Integer::from_u64(bits);
    }
}

DType pointer_dtype(DType pointee_dtype, MemorySpace memory_space, GlobalLock& lock) {
    uint32_t space_idx = static_cast<uint32_t>(memory_space);
    CHECK(space_idx <= kMemorySpaceMax);
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
                builder.append(memory_space_str(space));
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


// ----- DType Python wrapper object -----

struct DTypeObject {
    DType dtype;
    static PyTypeObject pytype;
};


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
                PyObject* space = memory_space_to_pyobject(derived.memory_space, lock);
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
    return Py_NewRef(dtype_name(py_unwrap<DTypeObject>(self).dtype, lock));
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
    return Py_NewRef(dtype_name(py_unwrap<DTypeObject>(self).dtype, lock));
}

static PyObject* DType_get_doc(PyObject* self, void* closure) {
    GlobalLock lock;
    DType dtype = py_unwrap<DTypeObject>(self).dtype;
    if (is_integer_dtype(dtype)) {
        StringBuilder sb;
        const char* signedness = is_signed_integer_dtype(dtype) ? "signed" : "unsigned";

        sb.append_many(dtype_bitwidth(dtype, lock), "-bit ", signedness,
                       " |arithmetic dtype| with values on the interval [",
                       integer_dtype_min(dtype), ", +", integer_dtype_max(dtype), "]");
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

    GlobalLock lock;
    std::optional<MemorySpace> space = memory_space_from_pyobject(py_memory_space, lock);
    if (!space.has_value()) {
        raise(PyExc_TypeError, "Expected a MemorySpace as the second argument, got ",
              use_repr(py_memory_space));
        return nullptr;
    }

    DType res = pointer_dtype(pointee_dtype, *space, lock);
    return Py_NewRef(dtype_to_pyobject(res, lock));
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
    return Py_NewRef(dtype_to_pyobject(res, lock));
}


// ----- Python wrappers for dtype queries -----

static PyObject* py_integer_dtype_min(PyObject* module_self, PyObject* dtype_obj) {
    if (!PyObject_TypeCheck(dtype_obj, &DTypeObject::pytype)) {
        raise(PyExc_TypeError, "Expected a DType object, got ", use_repr(dtype_obj));
        return nullptr;
    }
    DType dtype = py_unwrap<DTypeObject>(dtype_obj).dtype;
    return integer_dtype_min(dtype).to_pylong().release();
}

static PyObject* py_integer_dtype_max(PyObject* module_self, PyObject* dtype_obj) {
    if (!PyObject_TypeCheck(dtype_obj, &DTypeObject::pytype)) {
        raise(PyExc_TypeError, "Expected a DType object, got ", use_repr(dtype_obj));
        return nullptr;
    }
    DType dtype = py_unwrap<DTypeObject>(dtype_obj).dtype;
    return integer_dtype_max(dtype).to_pylong().release();
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
    return Py_NewRef(dtype_to_pyobject(pointee_dtype, lock));
}

static PyObject* py_pointer_memory_space(PyObject* module_self, PyObject* py_pointer) {
    GlobalLock lock;
    Result<DType> pointer_dtype = parse_pointer_dtype(py_pointer, lock);
    if (!pointer_dtype.is_ok()) return nullptr;
    MemorySpace space = pointer_dtype_memory_space(*pointer_dtype, lock);
    return Py_NewRef(memory_space_to_pyobject(space, lock));
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
    return Py_NewRef(dtype_to_pyobject(pointee_dtype, lock));
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
    {"_pointer_pointee_dtype", py_pointer_pointee_dtype, METH_O, nullptr},
    {"_pointer_memory_space", py_pointer_memory_space, METH_O, nullptr},
    {"_foreign_pointer_pointee_dtype", py_foreign_pointer_pointee_dtype, METH_O, nullptr},
    {"_unpickle_basic_dtype", _unpickle_basic_dtype, METH_O, nullptr},
    {}
};

#define INIT_STRING_CONSTANT(name, value) \
    if (!(name = PyUnicode_InternFromString(value))) return ErrorRaised

#define INIT_STRING_IDENT(ident) INIT_STRING_CONSTANT(g_##ident##_pyunicode, #ident)

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
