// SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0

#include "tile_kernel.h"

#include "check.h"
#include "cuda_loader.h"
#include "cuda_helper.h"
#include "hash_map.h"
#include "ipc_util.h"
#include "launch_helper.h"
#include "py.h"
#include "ref_ptr.h"
#include "stream_buffer.h"
#include "vec.h"

#include <cuda.h>
#include <dlpack.h>

#include <array>
#include <memory>
#include <algorithm>
#include <optional>
#include <utility>


static PyObject* g___cuda_array_interface___pyunicode;
static PyObject* g_typestr_pyunicode;
static PyObject* g_shape_pyunicode;
static PyObject* g_data_pyunicode;
static PyObject* g_strides_pyunicode;
static PyObject* g___dlpack___pyunicode;
static PyObject* g_compile_pyunicode;
static PyObject* g_dynamic_shared_memory_bytes_pyunicode;
static PyObject* g_cooperative_pyunicode;
static PyObject* g_block_in_cluster_count_pyunicode;
static PyObject* g_preferred_block_in_cluster_count_pyunicode;
static PyObject* g_programmatic_dependent_launch_pyunicode;
static PyObject* g___dataclass_fields___pyunicode;
static PyObject* g_torch_pyunicode;
static PyObject* g_cupy_pyunicode;
static PyObject* g_cuda_stream_pyunicode;
static PyObject* g_ptr_pyunicode;
static PyObject* g_numba_cuda_pyunicode;
static PyObject* g_cuda_bindings_driver_pyunicode;

static PyObject* g_enum_Enum_type;

static PyObject* g_default_tile_context;


static PyObject* get_datatype_module() {
    static PyObject* m;
    if (!m) m = PyImport_ImportModule("cuda.tile._datatype");
    return m;
}


static PyTypeObject* get_dtype_class() {
    static PyTypeObject* c;
    static bool cached;
    if (!cached) {
        cached = true;
        PyObject* datatype_mod = get_datatype_module();
        if (!datatype_mod) return nullptr;
        PyPtr dtype_class = getattr(datatype_mod, "DType");
        if (!dtype_class) return nullptr;
        if (!PyType_Check(dtype_class.get())) return nullptr;
        c = reinterpret_cast<PyTypeObject*>(dtype_class.release());
    }
    return c;
}


static PyObject* get_signature_module() {
    static PyObject* m;
    if (!m) m = PyImport_ImportModule("cuda.tile.compilation._signature");
    return m;
}

namespace { struct ImportedTypeChecker {
    PyTypeObject* cached_super_type_;
    bool is_cached_;

    // Check whether `sub_ty` is a subtype of
    //     "super_module_name[.super_submodule_name].super_type_name"
    // without importing the `super_module_name`.
    bool is_subtype_of(PyTypeObject* sub_ty,
                       PyObject* super_module_name,
                       const char* super_submodule_name,  // may be null
                       const char* super_type_name) {
        if (!is_cached_) {
            // Use PyImport_GetModule() rather than PyImport_Import() to avoid importing the module.
            // If the module is not in sys.modules, then there is no way there can be a subtype
            // of a type defined in that module.
            ErrorGuard guard;
            PyPtr mod = steal(PyImport_GetModule(super_module_name));
            // Can't flip is_cached_ to true just yet -- the module may get imported later.
            if (!mod) return false;

            is_cached_ = true;

            if (super_submodule_name) {
                mod = try_getattr(mod, super_submodule_name);
                if (!mod) return false;
            }

            PyPtr item = try_getattr(mod, super_type_name);
            if (!item || !PyType_Check(item.get()))
                return false;

            cached_super_type_ = reinterpret_cast<PyTypeObject*>(item.get());
        }
        return cached_super_type_ && PyType_IsSubtype(sub_ty, cached_super_type_);
    }
}; }

// Must be holding GIL or g_launch_mutex to call this
static bool is_torch_tensor_subtype(PyTypeObject* ty) {
    static ImportedTypeChecker checker;
    return checker.is_subtype_of(ty, g_torch_pyunicode, nullptr, "Tensor");
}

// Must be holding GIL or g_launch_mutex to call this
static bool is_torch_cuda_stream_subtype(PyTypeObject* ty) {
    static ImportedTypeChecker checker;
    return checker.is_subtype_of(ty, g_torch_pyunicode, "cuda", "Stream");
}

// Must be holding GIL or g_launch_mutex to call this
static bool is_cupy_cuda_stream_subtype(PyTypeObject* ty) {
    static ImportedTypeChecker checker;
    return checker.is_subtype_of(ty, g_cupy_pyunicode, "cuda", "Stream");
}

// Must be holding GIL or g_launch_mutex to call this
static bool is_numba_cuda_driver_stream_subtype(PyTypeObject* ty) {
    static ImportedTypeChecker checker;
    return checker.is_subtype_of(ty, g_numba_cuda_pyunicode, "driver", "Stream");
}

// Must be holding GIL or g_launch_mutex to call this
static bool is_cuda_bindings_driver_custream_subtype(PyTypeObject* ty) {
    static ImportedTypeChecker checker;
    return checker.is_subtype_of(ty, g_cuda_bindings_driver_pyunicode, nullptr, "CUstream");
}

// Must be holding GIL or g_launch_mutex to call this
static PyObject* try_get_torch_to_dlpack_func() {
    static PyObject* func;
    static bool cached;
    if (!cached) {
        cached = true;
        if (PyPtr torch_C = try_import("torch._C"))
            func = try_getattr(torch_C, "_to_dlpack").release();
    }
    return func;
}


constexpr uint8_t BYTE_BITWIDTH = 8;

constexpr uint8_t DIVISOR_16 = 16;

constexpr uint8_t TMA_MAX_NDIM = 5;

namespace { union ArraySpecializationBits {
    struct {
        bool baseptr_16byte_aligned : 1;
        bool disjoint_elements : 1;
        unsigned stride_16byte_divisible : TMA_MAX_NDIM;
        unsigned stride_one : TMA_MAX_NDIM;
        unsigned shape_divisible_by_16 : TMA_MAX_NDIM;
    };
    uint64_t u64;

    bool is_stride_16byte_divisible(size_t dim) const {
        return dim < TMA_MAX_NDIM && ((stride_16byte_divisible >> dim) & 1);
    }

    bool is_stride_one(size_t dim) const {
        return dim < TMA_MAX_NDIM && ((stride_one >> dim) & 1);
    }

    bool is_shape_divisible_by_16(size_t dim) const {
        return dim < TMA_MAX_NDIM && ((shape_divisible_by_16 >> dim) & 1);
    }
}; }

static_assert(sizeof(ArraySpecializationBits) == 8);

#ifdef CUDA_TILE_ENABLE_DEV_FEATURES
#define ENABLE_CCONV_V3
#endif

enum class CallConvVersion {
    CutilePython_V1 = 1,
    CutilePython_V2 = 2,
#ifdef ENABLE_CCONV_V3
    CutilePython_V3 = 3,
#endif
};

namespace { struct CallingConvention {
    CallConvVersion version;

    inline bool operator== (const CallingConvention& other) const {
        return version == other.version;
    }

    static PyTypeObject pytype;
}; }

static PyObject* CallingConvention_get_name(PyObject* self, void*) {
    CallingConvention& cconv = py_unwrap<CallingConvention>(self);
    return PyUnicode_FromFormat("cutile_python_v%d", cconv.version);
}

static PyObject* CallingConvention_get_code(PyObject* self, void*) {
    CallingConvention& cconv = py_unwrap<CallingConvention>(self);
    return PyUnicode_FromFormat("t%d", cconv.version);
}

static PyObject* CallingConvention_get_version(PyObject* self, void*) {
    CallingConvention& cconv = py_unwrap<CallingConvention>(self);
    return PyLong_FromLong(static_cast<long>(cconv.version));
}

static PyObject* CallingConvention_repr(PyObject* self) {
    PyPtr name = steal(PyObject_GetAttrString(self, "name"));
    if (!name) return nullptr;
    PyPtr code = steal(PyObject_GetAttrString(self, "code"));
    if (!code) return nullptr;
    return PyUnicode_FromFormat("CallingConvention(%R, %R)", name.get(), code.get());
}

static PyGetSetDef CallingConvention_getsetters[] = {
    {"name", CallingConvention_get_name, nullptr},
    {"code", CallingConvention_get_code, nullptr},
    {"version", CallingConvention_get_version, nullptr},
    {}  // sentinel
};

static PyPtr get_cconv(CallConvVersion version) {
    PyObject* ret = CallingConvention::pytype.tp_new(&CallingConvention::pytype, nullptr, nullptr);
    if (!ret) return {};

    CallingConvention& cconv = py_unwrap<CallingConvention>(ret);
    cconv.version = version;
    return steal(ret);
}

static PyObject* get_cached_cconv(CallConvVersion version, PyObject** cache) {
    if (!*cache) {
        PyPtr cconv = get_cconv(version);
        if (!cconv) return nullptr;
        *cache = cconv.release();
    }
    return Py_NewRef(*cache);
}

static PyObject* CallingConvention_cutile_python_v1(PyObject*, PyObject*) {
    static PyObject* c;
    return get_cached_cconv(CallConvVersion::CutilePython_V1, &c);
}

static PyObject* CallingConvention_cutile_python_v2(PyObject*, PyObject*) {
    static PyObject* c;
    return get_cached_cconv(CallConvVersion::CutilePython_V2, &c);
}

#ifdef ENABLE_CCONV_V3
static PyObject* CallingConvention_cutile_python_v3(PyObject*, PyObject*) {
    static PyObject* c;
    return get_cached_cconv(CallConvVersion::CutilePython_V3, &c);
}
#endif

static PyPtr parse_cutile_python_calling_convention(const char* s) {
    if (s[0] == '1' && !s[1])
        return get_cconv(CallConvVersion::CutilePython_V1);
    if (s[0] == '2' && !s[1])
        return get_cconv(CallConvVersion::CutilePython_V2);
#ifdef ENABLE_CCONV_V3
    if (s[0] == '3' && !s[1])
        return get_cconv(CallConvVersion::CutilePython_V3);
#endif
    return {};
}


static PyObject* CallingConvention_from_code(PyObject*, PyObject* args) {
    const char* code;
    if (!PyArg_ParseTuple(args, "s", &code))
        return nullptr;
    if (code[0] == 't') {
        PyPtr ret = parse_cutile_python_calling_convention(code + 1);
        if (ret) return ret.release();
    }
    return PyErr_Format(PyExc_ValueError, "Unknown calling convention code '%s'", code);
}


static PyMethodDef CallingConvention_methods[] = {
    {"from_code", CallingConvention_from_code, METH_VARARGS | METH_STATIC, nullptr},
    {"cutile_python_v1", CallingConvention_cutile_python_v1, METH_NOARGS | METH_STATIC,
       "cutile_python_v1()\n"
        "--\n\n"
        "Returns the ``cutile_python_v1`` calling convention.\n\n"
    },
    {"cutile_python_v2", CallingConvention_cutile_python_v2, METH_NOARGS | METH_STATIC,
       "cutile_python_v2()\n"
        "--\n\n"
        "Returns the ``cutile_python_v2`` calling convention.\n\n"
    },
#ifdef ENABLE_CCONV_V3
    {"cutile_python_v3", CallingConvention_cutile_python_v3, METH_NOARGS | METH_STATIC,
       "cutile_python_v3()\n"
        "--\n\n"
        "Returns the ``cutile_python_v3`` calling convention.\n\n"
    },
#endif
    {}  // sentinel
};

PyTypeObject CallingConvention::pytype = {
    .tp_name = "cuda.tile.compilation.CallingConvention",
    .tp_basicsize = sizeof(PythonWrapper<CallingConvention>),
    .tp_dealloc = pywrapper_dealloc<CallingConvention>,
    .tp_repr = CallingConvention_repr,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    .tp_richcompare = pywrapper_richcompare_via_operator_equals<CallingConvention>,
    .tp_methods = CallingConvention_methods,
    .tp_getset = CallingConvention_getsetters,
    .tp_new = pywrapper_new<CallingConvention>,
};


static Status enable_maximum_dynamic_shared_memory(const DriverApi *driver,
                                                   const CUkernel kernel,
                                                   const char *func_name) {
    int device_count;
    CUresult res = driver->cuDeviceGetCount(&device_count);
    if (res != CUDA_SUCCESS) {
        return raise(PyExc_RuntimeError, "Failed to get device count: %s",
                     get_cuda_error(driver, res));
    }

    for (int device_ordinal = 0; device_ordinal < device_count;
         device_ordinal++) {
        CUdevice device;
        res = driver->cuDeviceGet(&device, device_ordinal);
        if (res != CUDA_SUCCESS) {
            return raise(PyExc_RuntimeError, "Failed to get device %d: %s",
                         device_ordinal, get_cuda_error(driver, res));
        }

        int max_smem;
        res = driver->cuDeviceGetAttribute(
            &max_smem, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
            device);
        if (res != CUDA_SUCCESS) {
            return raise(PyExc_RuntimeError,
                         "Failed to get maximum shared memory for device %d: %s",
                         device_ordinal, get_cuda_error(driver, res));
        }

        int static_smem;
        res = driver->cuKernelGetAttribute(
            &static_smem, CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, kernel, device);
        if (res != CUDA_SUCCESS) {
            return raise(PyExc_RuntimeError,
                         "Failed to get static shared memory for kernel %s: %s",
                         func_name, get_cuda_error(driver, res));
        }

        if (max_smem < static_smem) {
            // If the user's program uses more static shared memory than the
            // current device has available, then we cannot request enough
            // shared memory. If the user has another device capable of running
            // their program, they must run on that device and errors will be
            // reported at launch time.
            continue;
        }
        int largest_possible_dynamic_smem = max_smem - static_smem;
        res = driver->cuKernelSetAttribute(
            CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
            largest_possible_dynamic_smem, kernel, device);
        if (res != CUDA_SUCCESS) {
            return raise(PyExc_RuntimeError,
                         "Failed to set dynamic shared memory for kernel %s: %s",
                         func_name, get_cuda_error(driver, res));
        }
    }

    return OK;
}

// X(Name, #Attrs, MinStack, StackEffect)
#define FOREACH_SIZE_OPCODE(X) \
    X(Const, 1, 0, 1) \
    X(KernelArgI32, 1, 0, 1) \
    X(KernelArgI64, 1, 0, 1) \
    X(Add, 0, 2, -1) \
    X(Mul, 0, 2, -1) \
    X(RoundUpToPow2, 1, 1, 0)

#define SIZE_OPCODE_ENUM_ENTRY(name, _nattr, _min_st, _stack_eff) \
    name,

enum class SizeOpcode : uint8_t {
    FOREACH_SIZE_OPCODE(SIZE_OPCODE_ENUM_ENTRY)
};

#define SIZE_OPCODE_PARSE(name, nattr, min_st, stack_eff) \
    if (!PyUnicode_CompareWithASCIIString(opcode_str, #name)) { \
        *num_attrs = nattr; \
        *min_stack = min_st; \
        *stack_effect = stack_eff; \
        return SizeOpcode::name; \
    }

static Result<SizeOpcode> size_opcode_parse(PyObject* opcode_str,
                                            int* num_attrs, int* min_stack, int* stack_effect) {
    FOREACH_SIZE_OPCODE(SIZE_OPCODE_PARSE);
    return raise(PyExc_ValueError, "Invalid opcode string %R", opcode_str);
}

namespace { struct HostProgram {
    enum { kMaxStackDepth = 32 };

    Vec<SizeOpcode> opcodes;
    Vec<int64_t> op_attrs;
}; }

namespace { struct HoistedTensorMap {
    enum { kMaxRank = 5 };

    CUtensorMapDataType dtype;
    uint32_t item_size;
    uint32_t rank;
    uint32_t base_ptr_param_idx;
    HostProgram shape_stride_program;
    uint32_t box_dim[kMaxRank];
    uint32_t traversal_steps[kMaxRank];
    CUtensorMapInterleave interleave;
    CUtensorMapSwizzle swizzle;
    CUtensorMapL2promotion l2_promotion;
    CUtensorMapFloatOOBfill oob_fill;
}; }

struct TileKernel {
    CudaKernel cukernel;
    HostProgram dyn_smem_size_prog;
    Vec<HoistedTensorMap> hoisted_tensor_maps;

    // For an identity constant (e.g., Enum values), the key of KernelMap contains the constant's
    // address encoded as an int64_t. Just by itself, this is prone to the ABA problem:
    // if the constant is freed, another object could be allocated at the same address.
    // Thus, for each such constant, we store a reference to it.
    Vec<PyPtr> constant_refs;
};

struct KernelImage {
    PyPtr cubin;
    PyPtr symbol;
};

using KernelMap = HashMap<Vec<int64_t>, TileKernel>;


static ArenaOffset arena_alloc_words(Arena& arena, size_t count) {
    ArenaOffset offset = arena.size();
    arena.resize(offset + count);
    return offset;
}

static void** make_launch_params(LaunchHelper& helper) {
    helper.launch_params.clear();
    helper.launch_params.reserve(helper.cuarg_offsets.size());
    for (ArenaOffset offset : helper.cuarg_offsets)
        helper.launch_params.push_back(&helper.arena[offset]);
    return helper.launch_params.data();
}

template <size_t AlignmentBytes>
static void arena_pad_to_alignment(Arena& arena) {
    static_assert(AlignmentBytes % sizeof(Word) == 0);
    constexpr size_t AlignmentWords = AlignmentBytes / sizeof(Word);
    size_t padded_size = ((arena.size() + AlignmentWords - 1) / AlignmentWords)
            * AlignmentWords;
    arena.resize(padded_size);
}

static ArenaOffset push_single_word_cuarg(LaunchHelper& helper, Word word) {
    ArenaOffset offset = arena_alloc_words(helper.arena, 1);
    helper.arena[offset] = word;
    helper.cuarg_offsets.push_back(offset);
    return offset;
}

static LaunchHelper* g_helper_freelist;  // protected by the GIL or g_launch_mutex

namespace { struct LaunchHelperDeleter {
    void operator() (LaunchHelper* helper) const {
        helper->pyarg_refs.clear();
        helper->next_free = g_helper_freelist;
        g_helper_freelist = helper;
    }
}; }

#ifdef Py_GIL_DISABLED
static PyMutex g_launch_mutex = {0};
#endif

using LaunchHelperPtr = std::unique_ptr<LaunchHelper, LaunchHelperDeleter>;


static LaunchHelperPtr launch_helper_get() {
    if (g_helper_freelist) {
        LaunchHelper* ret = g_helper_freelist;
        g_helper_freelist = ret->next_free;
        ret->pyarg_types_breadth_first.clear();
        ret->pyarg_objs_breadth_first.clear();
        ret->leaf_pyarg_objs.clear();
        return LaunchHelperPtr(ret);
    } else {
        return LaunchHelperPtr(new LaunchHelper());
    }
}


namespace { struct DataclassInfo : SimpleRefcount<DataclassInfo> {
    PyPtr dataclass;
    Vec<PyPtr> field_names;

    DataclassInfo(PyPtr dataclass, Vec<PyPtr> field_names)
        : dataclass(std::move(dataclass)), field_names(std::move(field_names))
    { }
}; }


struct AggregateArgType {
    enum Kind {
        Tuple,
#ifdef ENABLE_CCONV_V3
        Dataclass
#endif
    };

    Kind kind;
    RefPtr<DataclassInfo> dataclass_info;

    bool operator== (const AggregateArgType& other) const {
        if (kind != other.kind) return false;
        switch (kind) {
        case Kind::Tuple:
            return true;
#ifdef ENABLE_CCONV_V3
        case Kind::Dataclass:
            return dataclass_info->dataclass == other.dataclass_info->dataclass;
#endif
        }
        CHECK(false);
    }
};


// Kinds of constant values that are allowed as kernel arguments.
#define FOREACH_CONSTANT_KIND(X) \
    X(Bool) \
    X(Int) \
    X(Float) \
    X(None_) \
    X(String) \
    X(Enum) \
    X(NativeDType) \
    X(ForeignDType)

#define CONSTANT_KIND_ENTRY(name) name,

enum class ConstantKind : uint8_t {
    FOREACH_CONSTANT_KIND(CONSTANT_KIND_ENTRY)
};


#define CONSTANT_KIND_NAME_STR(name) #name,
static const char g_constant_kind_names[][16] = {
    FOREACH_CONSTANT_KIND(CONSTANT_KIND_NAME_STR)
};


static PyObject* g_constant_kind_enum;

static PyPtr define_constant_kind_enum() {
    PyPtr entries = steal(PyDict_New());
    if (!entries) return {};

    for (size_t i = 0; i < std::extent_v<decltype(g_constant_kind_names)>; ++i) {
        PyPtr value = steal(PyLong_FromUnsignedLongLong(i));
        if (!value) return {};

        if (PyDict_SetItemString(entries.get(), g_constant_kind_names[i], value.get()))
            return {};
    }

    return steal(PyObject_CallFunction(
            g_enum_Enum_type, "sO", "ConstantKind", entries.get()));
}



struct ParameterKind {
    enum Category : uint8_t {
        ConstantBool,
        ConstantInt,
        ConstantFloat,
        ConstantNone,
        IdentityConstant,  // constant that can be compared via object identity, e.g. an Enum value
        Array,
        Boolean,
        Integer,
        Float,
        List,
        AggregateBegin,
        AggregateEnd,
    };
    Category category;
    AggregateArgType agg_type;  // Only set when `category == AggregateBegin`

    bool operator== (const ParameterKind& other) const {
        if (category != other.category) return false;
        return category == AggregateBegin ? agg_type == other.agg_type : true;
    }

    bool operator!= (const ParameterKind& other) const {
        return !(*this == other);
    }
};

struct KernelFamily : SimpleRefcount<KernelFamily> {
    Vec<ParameterKind> param_kinds;
    KernelMap kernels_by_constants;

    explicit KernelFamily(Vec<ParameterKind>&& param_kinds) : param_kinds(std::move(param_kinds)) {}
};


enum class PythonArgKind : uint8_t {
    ConstantBool,
    ConstantInt,
    ConstantFloat,
    ConstantNone,
    ConstantString,
    IdentityConstant,
    ForeignDTypeConstant,
    // A torch.Tensor that we can access via torch._C._to_dlpack
    TorchTensorDlpack,
    // An object with __dlpack__ method
    DlpackArray,
    // An object with __cuda_array_interface__
    CudaArray,
    // Python `bool`,
    PyBool,
    // Python `int`,
    PyLong,
    // Python `float`
    PyFloat,
    // Python `list`
    PyList,
};

static inline PythonArgKind constant_kind_as_arg_kind(ConstantKind kind) {
    switch (kind) {
    case ConstantKind::Bool: return PythonArgKind::ConstantBool;
    case ConstantKind::Int: return PythonArgKind::ConstantInt;
    case ConstantKind::Float: return PythonArgKind::ConstantFloat;
    case ConstantKind::None_: return PythonArgKind::ConstantNone;
    case ConstantKind::String: return PythonArgKind::ConstantString;
    case ConstantKind::Enum: return PythonArgKind::IdentityConstant;
    case ConstantKind::NativeDType: return PythonArgKind::IdentityConstant;
    case ConstantKind::ForeignDType: return PythonArgKind::ForeignDTypeConstant;
    }
    CHECK_UNREACHABLE;
}


static ParameterKind::Category param_category_from_pyarg_kind(PythonArgKind k) {
    switch (k) {
    case PythonArgKind::ConstantBool: return ParameterKind::ConstantBool;
    case PythonArgKind::ConstantInt: return ParameterKind::ConstantInt;
    case PythonArgKind::ConstantFloat: return ParameterKind::ConstantFloat;
    case PythonArgKind::ConstantNone: return ParameterKind::ConstantNone;
    case PythonArgKind::ConstantString: return ParameterKind::IdentityConstant;
    case PythonArgKind::IdentityConstant: return ParameterKind::IdentityConstant;
    case PythonArgKind::ForeignDTypeConstant: return ParameterKind::IdentityConstant;
    case PythonArgKind::TorchTensorDlpack: return ParameterKind::Array;
    case PythonArgKind::DlpackArray: return ParameterKind::Array;
    case PythonArgKind::CudaArray: return ParameterKind::Array;
    case PythonArgKind::PyBool: return ParameterKind::Boolean;
    case PythonArgKind::PyLong: return ParameterKind::Integer;
    case PythonArgKind::PyFloat: return ParameterKind::Float;
    case PythonArgKind::PyList: return ParameterKind::List;
    }
    CHECK_UNREACHABLE;
}

static constexpr int u8_pair(uint8_t x, uint8_t y) {
    return x | (y << 8);
}

#define FOREACH_DLPACK_DTYPE(X) \
    X(kDLBool, 8, "bool_") \
    \
    X(kDLInt, 8, "int8") \
    X(kDLInt, 16, "int16") \
    X(kDLInt, 32, "int32") \
    X(kDLInt, 64, "int64") \
    \
    X(kDLUInt, 8, "uint8") \
    X(kDLUInt, 16, "uint16") \
    X(kDLUInt, 32, "uint32") \
    X(kDLUInt, 64, "uint64") \
    \
    X(kDLFloat, 16, "float16") \
    X(kDLFloat, 32, "float32") \
    X(kDLFloat, 64, "float64") \
    \
    X(kDLBfloat, 16, "bfloat16") \
    \
    X(kDLFloat8_e4m3fn, 8, "float8_e4m3fn") \
    X(kDLFloat8_e5m2, 8, "float8_e5m2") \
    X(kDLFloat8_e8m0fnu, 8, "float8_e8m0fnu")


#define DLPACK_DTYPE_NAME_CASE(code, bits, name) \
    case u8_pair(code, bits): return name;

static Result<const char*> dtype_name(DLDataType dtype) {
    if (dtype.lanes != 1)
        return raise(PyExc_TypeError, "Array dtypes with multiple lanes are not supported");

    switch (u8_pair(dtype.code, dtype.bits)) {
        FOREACH_DLPACK_DTYPE(DLPACK_DTYPE_NAME_CASE)
    default:
        return raise(PyExc_TypeError, "Unsupported array dtype");
    }
}

static PyPtr dtype_to_python(DLDataType dtype) {
    PyObject* dtype_module = get_datatype_module();
    if (!dtype_module) return {};

    Result<const char*> name = dtype_name(dtype);
    if (!name.is_ok()) return {};

    return getattr(dtype_module, *name);
}


#define DLPACK_DTYPE_NAME(_code, _bits, name) name,
#define DLPACK_DTYPE_TYPE(code, bits, _name) {code, bits, 1},

static constexpr char dlpack_dtype_names[][16] = { FOREACH_DLPACK_DTYPE(DLPACK_DTYPE_NAME) };
static constexpr DLDataType dlpack_dtypes[] = { FOREACH_DLPACK_DTYPE(DLPACK_DTYPE_TYPE) };

static Result<std::optional<DLDataType>> dtype_from_python(PyObject* dtype) {
    PyPtr py_name = getattr(dtype, "name");
    if (!py_name) return ErrorRaised;

    constexpr size_t n = std::extent_v<decltype(dlpack_dtype_names)>;
    static_assert(n == std::extent_v<decltype(dlpack_dtypes)>);

    for (size_t i = 0; i < n; ++i) {
        if (!PyUnicode_CompareWithASCIIString(py_name.get(), dlpack_dtype_names[i]))
            return {dlpack_dtypes[i]};
    }
    return {std::nullopt};
}


struct ForeignDTypeInfo {
    std::optional<DLDataType> dlpack_dtype;
    PyPtr native_dtype;
};


enum ForeignDtypeKind {
    kTorch = 1,
    kNumpy = 2,
    kMlDTypes = 8
};

static constexpr
struct { char name[16]; DLDataType type; int lib_mask; } foreign_dtype_table[] = {
    {"bool", {kDLBool, 8, 1}, kTorch },
    {"bool_", {kDLBool, 8, 1}, kNumpy },
    {"uint8", {kDLUInt, 8, 1}, kTorch | kNumpy },
    {"uint16", {kDLUInt, 16, 1}, kTorch | kNumpy },
    {"uint32", {kDLUInt, 32, 1}, kTorch | kNumpy},
    {"uint64", {kDLUInt, 64, 1}, kTorch | kNumpy},
    {"int8", {kDLInt, 8, 1}, kTorch | kNumpy},
    {"int16", {kDLInt, 16, 1}, kTorch | kNumpy},
    {"int32", {kDLInt, 32, 1}, kTorch | kNumpy},
    {"int64", {kDLInt, 64, 1}, kTorch | kNumpy},
    {"float16", {kDLFloat, 16, 1}, kTorch | kNumpy},
    {"float32", {kDLFloat, 32, 1}, kTorch | kNumpy},
    {"float64", {kDLFloat, 64, 1}, kTorch | kNumpy},
    {"bfloat16", {kDLBfloat, 16, 1}, kTorch | kMlDTypes},
    {"float8_e4m3fn", {kDLFloat8_e4m3fn, 8, 1}, kTorch | kMlDTypes},
    {"float8_e5m2", {kDLFloat8_e5m2, 8, 1}, kTorch | kMlDTypes},
    {"float8_e8m0fnu", {kDLFloat8_e8m0fnu, 8, 1}, kTorch | kMlDTypes}
};

static void register_foreign_dtypes_common(PyObject* foreign_module,
                                           PyObject* numpy_dtype_class,
                                           ForeignDtypeKind bit,
                                           HashMap<PyPtr, ForeignDTypeInfo>* registry) {
    for (const auto& entry : foreign_dtype_table) {
        if (!(entry.lib_mask & bit)) continue;

        PyPtr foreign_pyobj = try_getattr(foreign_module, entry.name);
        if (!foreign_pyobj) continue;

        ErrorGuard guard;
        PyPtr native_pyobj = dtype_to_python(entry.type);
        if (!native_pyobj) continue;

        registry->insert(foreign_pyobj, ForeignDTypeInfo{entry.type, native_pyobj});

        if (numpy_dtype_class) {
            PyPtr numpy_dtype_desc = steal(
                    PyObject_CallOneArg(numpy_dtype_class, foreign_pyobj.get()));
            if (numpy_dtype_desc)
                registry->insert(std::move(numpy_dtype_desc),
                                 ForeignDTypeInfo{entry.type, std::move(native_pyobj)});
        }
    }
}


static void register_torch_dtypes(HashMap<PyPtr, ForeignDTypeInfo>* registry) {
    PyPtr torch = try_import("torch");
    if (torch)
        register_foreign_dtypes_common(torch.get(), nullptr, kTorch, registry);
}


static void register_numpy_and_ml_dtypes(HashMap<PyPtr, ForeignDTypeInfo>* registry) {
    PyPtr numpy = try_import("numpy");
    if (!numpy) return;
    PyPtr numpy_dtype_class = try_getattr(numpy, "dtype");

    register_foreign_dtypes_common(numpy.get(), numpy_dtype_class.get(), kNumpy, registry);

    PyPtr ml_dtypes = try_import("ml_dtypes");
    if (ml_dtypes)
        register_foreign_dtypes_common(ml_dtypes.get(), numpy_dtype_class.get(), kMlDTypes,
                                       registry);
}


// Must hold g_launch_mutex or GIL to call this
static HashMap<PyPtr, ForeignDTypeInfo>* get_foreign_dtype_registry() {
    static HashMap<PyPtr, ForeignDTypeInfo>* registry;
    if (!registry) {
        auto reg = std::make_unique<HashMap<PyPtr, ForeignDTypeInfo>>();
        register_torch_dtypes(reg.get());
        register_numpy_and_ml_dtypes(reg.get());
        registry = reg.release();
    }
    return registry;
}


static PyObject* foreign_dtype_object_register(PyObject* self, PyObject* args) {
    PyObject* foreign_dtype;
    PyObject* native_dtype;
    if (!PyArg_ParseTuple(args, "OO", &foreign_dtype, &native_dtype))
        return nullptr;

    Result<std::optional<DLDataType>> dtype_res = dtype_from_python(native_dtype);
    if (!dtype_res.is_ok()) return nullptr;

#ifdef Py_GIL_DISABLED
    PyCriticalSectionGuard guard(&g_launch_mutex);
#endif

    get_foreign_dtype_registry()->insert(newref(foreign_dtype),
                                         ForeignDTypeInfo{*dtype_res, newref(native_dtype)});
    return Py_NewRef(Py_None);
}


static PyObject* foreign_dtype_object_to_native(PyObject* self, PyObject* object) {
#ifdef Py_GIL_DISABLED
    PyCriticalSectionGuard guard(&g_launch_mutex);
#endif
    HashMap<PyPtr, ForeignDTypeInfo>::Item* item = get_foreign_dtype_registry()->find(object);
    return Py_NewRef(item ? item->value.native_dtype.get() : Py_None);
}


static std::optional<ConstantKind> classify_constant(PyObject* obj, bool kernel_arg) {
    if (PyBool_Check(obj))
        return ConstantKind::Bool;

    if (PyLong_Check(obj))
        return ConstantKind::Int;

    if (PyFloat_Check(obj))
        return ConstantKind::Float;

#ifndef ENABLE_CCONV_V3
    if (!kernel_arg) {
#endif

    if (obj == Py_None)
        return ConstantKind::None_;

    if (PyUnicode_CheckExact(obj))
        return ConstantKind::String;

    if (PyObject_TypeCheck(obj, reinterpret_cast<PyTypeObject*>(g_enum_Enum_type)))
        return ConstantKind::Enum;

    if (Py_IS_TYPE(obj, get_dtype_class()))
        return ConstantKind::NativeDType;

    if (get_foreign_dtype_registry()->find(obj))
        return ConstantKind::ForeignDType;

#ifndef ENABLE_CCONV_V3
    }
#endif

    return std::nullopt;
}

static PyObject* py_classify_constant(PyObject* self, PyObject* args) {
    PyObject* obj;
    int kernel_arg;
    if (!PyArg_ParseTuple(args, "Op", &obj, &kernel_arg))
        return nullptr;

    std::optional<ConstantKind> res = classify_constant(obj, kernel_arg);
    if (!res.has_value())
        return Py_NewRef(Py_None);

    return PyObject_GetAttrString(g_constant_kind_enum,
                                  g_constant_kind_names[static_cast<size_t>(*res)]);
}


static std::optional<PythonArgKind> classify_nonconstant_arg(PyObject* arg) {
    if (PyBool_Check(arg))
        return PythonArgKind::PyBool;

    if (PyLong_Check(arg))
        return PythonArgKind::PyLong;

    if (PyFloat_Check(arg))
        return PythonArgKind::PyFloat;

    if (PyList_Check(arg))
        return PythonArgKind::PyList;

    if (is_torch_tensor_subtype(Py_TYPE(arg))) {
        // Calling torch._C._to_dlpack(arg) is much faster than calling arg.__dlpack__()
        // because it goes straight into C++ code, with no Python in between.
        // So we always prefer that.
        if (try_get_torch_to_dlpack_func())
            return PythonArgKind::TorchTensorDlpack;
    }

    if (PyObject_HasAttr(arg, g___dlpack___pyunicode))
        return PythonArgKind::DlpackArray;

    if (PyObject_HasAttr(arg, g___cuda_array_interface___pyunicode))
        return PythonArgKind::CudaArray;

    return {};
}

struct LeafAnnotationNode;
struct ParameterAnnotationNode;


namespace { struct ExpandAggregates; }

// ProfileMap enables us to quickly find a KernelFamily given the Python types
// of the kernel arguments (expressed as a sequence of PyTypeObject*).
// It is organized as a tree, with ProfileMapNode serving as a base class for the node.
// Each node is either a leaf (PythonArgProfile) or an intermediate node (ExpandAggregates).
// The tree structure is needed to support unpacking (flattening) aggregate arguments.
//
// For example, say the kernel takes two arguments: an `int` and a tuple of two `int`s.
// Our initial lookup key is therefore (&PyLong_Type, &PyTuple_Type). This is not sufficient
// information to determine the whole signature, because we don't know yet what's inside the tuple.
// Thus, the initial lookup will yield an ExpandAggregates node that will intruct us to expand
// the second argument. Looking inside the tuple will produce the next lookup key
// (&PyLong_Type, &PyLong_Type) that we will use to search in the ExpandAggregate::children
// hash map, which will lead us to a leaf PythonArgProfile node. In other words, the depth
// of the PythonArgProfile leaf matches the nesting depth of the arguments.
//
namespace { struct ProfileMapNode : SimpleRefcount<ProfileMapNode> {
    Vec<PyPtr> arg_types;  // the lookup key for the hash table
    ExpandAggregates* parent;  // null if and only if depth == 0
    int depth;  // 0 for top-level node
    bool leaf;  // if true, this is a PythonArgProfile; else an ExpandAggregates.

    explicit ProfileMapNode(Vec<PyPtr> arg_types,
                            ExpandAggregates* parent,
                            int depth,
                            bool leaf)
        : arg_types(std::move(arg_types)), parent(parent), depth(depth), leaf(leaf)
    { }

    virtual ~ProfileMapNode() {}
}; }

// Wrapper around ProfileMapNode pointer to make it usable as a key of a HashMap.
namespace { struct ProfileMapKey {
    RefPtr<ProfileMapNode> node;

    size_t size() const {
        return node->arg_types.size();
    }

    PyTypeObject* operator[] (size_t i) const {
        return reinterpret_cast<PyTypeObject*>(node->arg_types[i].get());
    }
}; }

// We use the HashMap as a hash set by providing a dummy value type.
using ProfileMap = HashMap<ProfileMapKey, int /*dummy*/>;



#ifdef ENABLE_CCONV_V3
static Status get_dataclass_field_names(PyObject* cls, Vec<PyPtr>* field_names) {
    PyPtr fields = steal(PyObject_GetAttr(cls, g___dataclass_fields___pyunicode));
    if (!fields) return ErrorRaised;

    PyPtr field_iter = steal(PyObject_GetIter(fields.get()));
    if (!field_iter) return ErrorRaised;

    while (PyPtr name = steal(PyIter_Next(field_iter.get()))) {
        field_names->push_back(std::move(name));
    }
    if (PyErr_Occurred())
        return ErrorRaised;
    return OK;
}


static RefPtr<DataclassInfo> get_dataclass_info(PyTypeObject* ty) {
    static HashMap<PyPtr, RefPtr<DataclassInfo>>* cache;
    if (!cache) cache = new HashMap<PyPtr, RefPtr<DataclassInfo>>();

    PyPtr cls_ref = newref(reinterpret_cast<PyObject*>(ty));
    HashMap<PyPtr, RefPtr<DataclassInfo>>::Item* cached = cache->find(cls_ref);
    if (cached) return cached->value;

    Vec<PyPtr> field_names;
    if (!get_dataclass_field_names(cls_ref.get(), &field_names))
        return {};

    RefPtr<DataclassInfo> info = newref(new DataclassInfo(cls_ref, std::move(field_names)));
    cache->insert(cls_ref, info);
    return info;
}
#endif  // ENABLE_CCONV_V3


namespace {struct AggregateArgInfo {
    size_t breadth_first_index;
    AggregateArgType type;
}; }

// Intermediate node of a ProfileMap tree.
namespace { struct ExpandAggregates : ProfileMapNode {
    // Indicates which arguments are aggregate (and therefore need to be expanded).
    // Must be non-empty.
    Vec<AggregateArgInfo> aggregate_args;

    // Maps the expanded argument types to the next node to follow.
    ProfileMap children;

    explicit ExpandAggregates(Vec<PyPtr> arg_types,
                              ExpandAggregates* parent,
                              int depth,
                              Vec<AggregateArgInfo> aggregate_args)
        : ProfileMapNode(std::move(arg_types), parent, depth, false)
        , aggregate_args(std::move(aggregate_args))
    {
        CHECK(!this->aggregate_args.empty());
    }
}; }

// Leaf node of a ProfileMap tree.
namespace { struct PythonArgProfile : ProfileMapNode {
    RefPtr<KernelFamily> family;

    // Indices into `pyarg_objs_breadth_first` to obtain leaf arguments in the depth-first order.
    Vec<size_t> leaf_pyarg_breadth_first_indices;

    Vec<PythonArgKind> arg_kinds;

    Vec<RefPtr<LeafAnnotationNode>> flat_param_annotations;

    explicit PythonArgProfile(Vec<PyPtr> arg_types,
                              ExpandAggregates* parent,
                              int depth,
                              KernelFamily* family,
                              Vec<size_t> leaf_pyarg_breadth_first_indices,
                              Vec<PythonArgKind> arg_kinds,
                              Vec<RefPtr<LeafAnnotationNode>> flat_param_annotations)
        : ProfileMapNode(std::move(arg_types), parent, depth, true)
        , family(newref(family))
        , leaf_pyarg_breadth_first_indices(std::move(leaf_pyarg_breadth_first_indices))
        , arg_kinds(std::move(arg_kinds))
        , flat_param_annotations(std::move(flat_param_annotations))
    {
        CHECK(this->leaf_pyarg_breadth_first_indices.size() == this->arg_kinds.size());
        CHECK(this->arg_kinds.size() == this->flat_param_annotations.size());
    }
}; }

// View into subsequence vec[offset:end_offset] of Vec<PyTypeObject*>.
// Used as a hash table key during lookup into ProfileMap,
// in order to avoid materializing a Vec<PyPtr> for the subsequence.
namespace { struct ProfileMapQuery {
    Vec<PyTypeObject*>* vec;
    size_t offset;
    size_t end_offset;

    PyTypeObject* operator[] (size_t i) const {
        return (*vec)[offset + i];
    }

    size_t size() const {
        return end_offset - offset;
    }

    void mark_start() {
        offset = vec->size();
    }

    void mark_end() {
        end_offset = vec->size() - 1;  // exclude the last sentinel
    }

    Vec<PyPtr> to_owned() const {
        Vec<PyPtr> ret;
        ret.reserve(size());
        for (size_t i = offset; i < end_offset; ++i) {
            PyTypeObject* ty = (*vec)[i];
            ret.push_back(ty ? newref(reinterpret_cast<PyObject*>(ty)) : PyPtr{});
        }
        return ret;
    }
}; }

// Compare any combination of ProfileMapQuery and ProfileMapKey.
// Generic to make sure the logic is consistent for all possible combinations.
template <typename T1, typename T2>
static bool profile_map_key_equal(T1&& key1, T2&& key2) {
    size_t n1 = key1.size();
    size_t n2 = key2.size();
    if (n1 != n2) return false;
    for (size_t i = 0; i < n1; ++i) {
        if (key1[i] != key2[i])
            return false;
    }
    return true;
}

namespace { bool operator==(const ProfileMapKey& key1, const ProfileMapKey& key2) {
    return profile_map_key_equal(key1, key2);
} }

// Allow heterogeneous lookup for ProfileMap
template <>
struct CompareKey <ProfileMapQuery, ProfileMapKey> {
    static bool equals(const ProfileMapQuery& a, const ProfileMapKey& b) {
        return profile_map_key_equal(a, b);
    }
};

// Compute the hash from ProfileMapQuery or ProfileMapKey.
// Generic to make sure the logic is consistent for both.
template <typename T>
static void profile_map_key_hash(T&& key, Hasher& h) {
    size_t n = key.size();
    h.hash(n);
    for (size_t i = 0; i < n; ++i)
        Hash<PyTypeObject*>::hash(key[i], h);
}

template <>
struct Hash<ProfileMapQuery> {
    static void hash(const ProfileMapQuery& query, Hasher& h) {
        profile_map_key_hash(query, h);
    }
};

template <>
struct Hash<ProfileMapKey> {
    static void hash(const ProfileMapKey& key, Hasher& h) {
        profile_map_key_hash(key, h);
    }
};


// Concatenate values of two chars in a single unsigned integer
static constexpr unsigned char_pair(char x, char y) {
    unsigned xu = static_cast<unsigned char>(x);
    unsigned yu = static_cast<unsigned char>(y);
    return ((xu << 8) | yu);
}

static Result<DLDataType> parse_typestr(PyObject* typestr) {
    if (!PyUnicode_Check(typestr)) {
        PyErr_SetString(PyExc_TypeError, "__cuda_array_interface__['typestr'] is not a string");
        return ErrorRaised;
    }

    Py_ssize_t len;
    const char* str = PyUnicode_AsUTF8AndSize(typestr, &len);
    if (!str) return ErrorRaised;

    if (len < 3) {
        PyErr_Format(PyExc_TypeError, "__cuda_array_interface__['typestr'] has invalid value %S",
                     typestr);
        return ErrorRaised;
    }

    // TODO: support big endian one day?
    if (str[0] != '<' && str[0] != '|') {
        PyErr_SetString(PyExc_TypeError, "Only little-endian types are supported");
        return ErrorRaised;
    }

    DLDataType ret;
    ret.lanes = 1;

    switch (str[1]) {
    case 'b': ret.code = kDLBool; break;
    case 'i': ret.code = kDLInt; break;
    case 'u': ret.code = kDLUInt; break;
    case 'f': ret.code = kDLFloat; break;
    case 'V': ret.code = kDLBfloat; break;
    case 'c': ret.code = kDLComplex; break;
    default:
        PyErr_Format(PyExc_TypeError, "Unsupported type code %c", str[1]);
        return ErrorRaised;
    }

    // str[3] is safe to index because there is always a NUL byte at the end
    switch (char_pair(str[2], str[3])) {
    case char_pair('1', '\0'): ret.bits = 8; break;
    case char_pair('2', '\0'): ret.bits = 16; break;
    case char_pair('4', '\0'): ret.bits = 32; break;
    case char_pair('8', '\0'): ret.bits = 64; break;
    case char_pair('1', '6'):
        if (!str[4]) {
            ret.bits = 64;
            break;
        }
        [[fallthrough]];
    default:
        PyErr_Format(PyExc_TypeError, "Unsupported byte size in typestr: %s", str + 2);
        return ErrorRaised;
    }

    return ret;
}

struct ArrayType {
    DLDataType dtype;
    size_t ndim;
    unsigned index_bitwidth;
};

struct ArrayRepr {
    ArrayType arrty;
    ArenaOffset repr;
};

// This should compile to a no-op
static inline uint32_t dtype_as_uint(DLDataType dtype) {
    return static_cast<uint32_t>(dtype.code)
        | (static_cast<uint32_t>(dtype.bits) << 8)
        | (static_cast<uint32_t>(dtype.lanes) << 16);
}

static inline DLDataType dtype_from_uint(uint32_t u) {
    return DLDataType{
        .code = static_cast<uint8_t>(u & 0xff),
        .bits = static_cast<uint8_t>((u >> 8) & 0xff),
        .lanes = static_cast<uint16_t>((u >> 16) & 0xffff),
    };
}

// Pack data type, array rank, and index bitwidth in a single int64_t so it
// could be used as a single constant for looking up the kernel in a family.
// Layout: [63: index_bitwidth (0=32, 1=64)] [62..32: ndim] [31..0: dtype]
static int64_t pack_array_type(const ArrayType& a) {
    uint64_t dtype_u = static_cast<uint64_t>(dtype_as_uint(a.dtype));
    uint64_t ndim_u = static_cast<uint64_t>(a.ndim);
    uint64_t ibw_bit = (a.index_bitwidth == 64) ? 1ULL : 0ULL;
    return static_cast<int64_t>(dtype_u | (ndim_u << 32) | (ibw_bit << 63));
}

static ArrayType unpack_array_type(int64_t c) {
    uint64_t u = c;
    uint32_t dtype = u & 0xffffffff;
    uint32_t ndim = (u >> 32) & 0x7fffffff;
    unsigned ibw = ((u >> 63) & 1) ? 64 : 32;
    return {dtype_from_uint(dtype), ndim, ibw};
}

static Status fill_row_major_strides(unsigned index_bitwidth, Word* repr, size_t ndim) {
    if (ndim == 0) return OK;

    Word* shape = repr + 1 + ndim;
    Word* stride = shape + ndim;
    uint64_t prev_stride = 1;
    (--stride)->i64 = 1;

    for (size_t i = 0; i < ndim - 1; ++i) {
        uint64_t new_stride = prev_stride * static_cast<uint64_t>((--shape)->i64);
        if (index_bitwidth != 64 && new_stride > INT32_MAX)
            return raise(PyExc_OverflowError, "stride is too big");
        (--stride)->i64 = new_stride;
        prev_stride = new_stride;
    }
    return OK;
}

static ArraySpecializationBits compute_array_specialization_bits(
        const Word* array_repr, size_t ndim, unsigned dtype_bitwidth, unsigned index_bitwidth,
        bool set_stride_one) {

    ArraySpecializationBits ret = {};
    void* data_ptr = array_repr[0].device_ptr;
    const Word* shape_words = array_repr + 1;
    const Word* stride_words = shape_words + ndim;

    // Only specialize stride divisibility, stride 1 and shape divisibility for ndim <= TMA_MAX_NDIM
    if (ndim <= TMA_MAX_NDIM) {
        for (size_t i = 0; i < ndim; ++i) {
            int64_t stride = stride_words[i].i64;
            int64_t shape = shape_words[i].i64;
            int64_t stride_bitwidth = stride * dtype_bitwidth;
            int64_t shape_bitwidth = shape * dtype_bitwidth;
            bool is_stride_byte_aligned = stride_bitwidth % BYTE_BITWIDTH == 0;
            bool is_stride_16_byte_divisible =
                    (stride_bitwidth / BYTE_BITWIDTH) % DIVISOR_16 == 0;
            bool is_shape_byte_aligned = shape_bitwidth % BYTE_BITWIDTH == 0;
            bool is_shape_divisible_by_16 = shape % DIVISOR_16 == 0;

            // A size-one axis has no address contribution for dense indexing.
            // Keep its physical stride dynamic, but preserve the 16-byte
            // alignment specialization used by layout/vectorization analysis.
            bool is_singleton_stride_one = shape == 1 && stride == 1;
            if ((is_stride_byte_aligned && is_stride_16_byte_divisible) ||
                    (set_stride_one && is_singleton_stride_one))
                ret.stride_16byte_divisible |= 1u << i;

            // A `static_stride_dims` annotation makes stride *values* authoritative, so the
            // dispatcher must not infer stride==1 for any dim (annotated dims get their exact
            // value pushed as a constant; the rest stay dynamic). Divisibility above is a
            // separate assumption and is unaffected.
            if (set_stride_one && stride == 1 && !is_singleton_stride_one)
                ret.stride_one |= 1u << i;

            if (is_shape_byte_aligned && is_shape_divisible_by_16)
                ret.shape_divisible_by_16 |= 1u << i;
        }
    }

    // extract base pointer divisibility
    intptr_t data_ptr_int = reinterpret_cast<intptr_t>(data_ptr);
    ret.baseptr_16byte_aligned = data_ptr_int % DIVISOR_16 == 0;

    // check elements disjoint.
    // sort by stride. the smallest stride indicates the contiguous axis
    // of the underlying array.
    Vec<std::pair<int64_t, int64_t>> strides_and_shape(ndim);
    for (size_t i = 0; i < ndim; ++i) {
        strides_and_shape[i] = {stride_words[i].i64, shape_words[i].i64};
    }
    std::sort(strides_and_shape.begin(), strides_and_shape.end());

    // disjointness check:
    // - 0 dimension array elements are always disjoint.
    // - >0 dimension array elements are disjoint if every stride is positive
    //    and greater than or equal to the product of the previous stride and
    //    the previous shape.
    bool elems_disjoint = (ndim == 0) || (strides_and_shape[0].first > 0);
    for (size_t i = 0; i + 1 < ndim; ++i) {
        int64_t prev_stride = strides_and_shape[i].first;
        int64_t prev_shape = strides_and_shape[i].second;
        int64_t cur_stride = strides_and_shape[i + 1].first;
        elems_disjoint &= (
            cur_stride > 0 && cur_stride >= prev_stride * prev_shape);
    }
    ret.disjoint_elements = elems_disjoint;

    return ret;
}


template <typename T>
struct Cursor {
    const T* data;
    size_t len;

    /* implicit */ Cursor(const Vec<T>& vec)
        : data(vec.data()), len(vec.size())
    {}

    Cursor(const T* data, size_t len)
        : data(data), len(len)
    {}

    const T& peek() const {
        CHECK(len);
        return *data;
    }

    const T& next() {
        CHECK(len);
        const T& ret = *data;
        ++data, --len;
        return ret;
    }
};

using ConstantCursor = Cursor<int64_t>;

static Result<size_t> normalize_dim(int64_t signed_dim, size_t ndim, const char* annotation) {
    int64_t ndim_i64 = static_cast<int64_t>(ndim);
    if (signed_dim < 0)
        signed_dim += ndim_i64;
    if (signed_dim < 0 || signed_dim >= ndim_i64)
        return raise(PyExc_ValueError,
                     "%s contains axis %lld, but array rank is %zu",
                     annotation, static_cast<long long>(signed_dim), ndim);
    return static_cast<size_t>(signed_dim);
}


struct ArrayTypeConstantBuilder {
    void* device_ptr = nullptr;
    uint64_t bits = -1;
    bool has_first_array = false;
    ArenaOffset first_array_repr = 0;

    // Verify that `words` (the current array's shape or stride words) agree with the
    // first array's homologous words at every annotated dim. `word_offset` is the repr
    // offset of the first word (1 for shape, 1 + ndim for stride).
    Status check_agreement(const Arena& arena,
                           const Word* words, size_t ndim, size_t word_offset,
                           const Vec<int64_t>& dims, const char* annotation,
                           const char* what) const {
        const Word* first_words = arena.data() + first_array_repr + word_offset;
        for (int64_t dim : dims) {
            Result<size_t> normalized_dim = normalize_dim(dim, ndim, annotation);
            if (!normalized_dim.is_ok()) return ErrorRaised;

            int64_t expected = first_words[*normalized_dim].i64;
            int64_t actual = words[*normalized_dim].i64;
            if (expected != actual)
                return raise(PyExc_ValueError,
                             "Arrays in list vary in static %s at axis %zu (%lld vs %lld)",
                             what, *normalized_dim,
                             static_cast<long long>(expected),
                             static_cast<long long>(actual));
        }
        return OK;
    }

    Status update(const Arena& arena, const ArrayRepr& ar,
                  const Vec<int64_t>& static_shape_dims,
                  const Vec<int64_t>& static_stride_dims) {
        const Word* repr = arena.data() + ar.repr;
        device_ptr = repr[0].device_ptr;
        bits &= compute_array_specialization_bits(
                    repr, ar.arrty.ndim, ar.arrty.dtype.bits * ar.arrty.dtype.lanes,
                    ar.arrty.index_bitwidth,
                    /*set_stride_one=*/static_stride_dims.empty()).u64;

        if (!has_first_array) {
            has_first_array = true;
            first_array_repr = ar.repr;
        } else {
            // When several arrays share one constraint (e.g. the items of a
            // list), the kernel is specialized to a single set of compile-time
            // shape/stride values, so every array must agree on those dims.
            const Word* shape_words = repr + 1;
            const Word* stride_words = shape_words + ar.arrty.ndim;
            if (!check_agreement(arena, shape_words, ar.arrty.ndim, 1,
                                 static_shape_dims, "static_shape_dims", "shape"))
                return ErrorRaised;
            if (!check_agreement(arena, stride_words, ar.arrty.ndim,
                                 1 + ar.arrty.ndim, static_stride_dims, "static_stride_dims",
                                 "stride"))
                return ErrorRaised;
        }
        return OK;
    }

    // Range-validates each annotated axis against the (runtime) rank -- this is the single
    // place the annotation is checked, so it returns ErrorRaised (with a Python exception set)
    // for an out-of-range axis rather than relying on a CHECK invariant established elsewhere.
    Status finalize(const DriverApi* driver, const ArrayType& arrty,
                    const Vec<int64_t>& static_shape_dims,
                    const Vec<int64_t>& static_stride_dims,
                    LaunchHelper& helper) {
        CHECK(has_first_array);
        const Word* repr = helper.arena.data() + first_array_repr;
        const Word* shape_words = repr + 1;
        const Word* stride_words = shape_words + arrty.ndim;
        // Written as [packed_array_type, specialization_bits, static_shape_value...,
        // static_stride_value...]; parse_array_constraint() must read in the same order.
        helper.constants.push_back(pack_array_type(arrty));
        helper.constants.push_back(bits);
        for (int64_t dim : static_shape_dims) {
            Result<size_t> normalized_dim = normalize_dim(dim, arrty.ndim, "static_shape_dims");
            if (!normalized_dim.is_ok()) return ErrorRaised;
            helper.constants.push_back(shape_words[*normalized_dim].i64);
        }
        for (int64_t dim : static_stride_dims) {
            Result<size_t> normalized_dim = normalize_dim(dim, arrty.ndim, "static_stride_dims");
            if (!normalized_dim.is_ok()) return ErrorRaised;
            helper.constants.push_back(stride_words[*normalized_dim].i64);
        }
        if (!helper.cuda_context) {
            driver->cuPointerGetAttribute(&helper.cuda_context, CU_POINTER_ATTRIBUTE_CONTEXT,
                    reinterpret_cast<CUdeviceptr>(device_ptr));
        }
        return OK;
    }
};


static Status read_static_axis_values(const Vec<int64_t>& dims, PyObject* target,
                                      ConstantCursor& cursor, size_t ndim,
                                      const char* annotation) {
    for (int64_t dim : dims) {
        Result<size_t> axis = normalize_dim(dim, ndim, annotation);
        CHECK(axis.is_ok());  // finalize() already range-checked this axis

        if (PyList_GET_ITEM(target, *axis) != Py_None)
            return raise(PyExc_ValueError, "%s contains duplicate axis %lld",
                         annotation, static_cast<long long>(dim));

        int64_t value = cursor.next();
        PyPtr value_obj = steal(PyLong_FromLongLong(value));
        if (!value_obj) return ErrorRaised;
        if (PySequence_SetItem(target, *axis, value_obj.get()) < 0)
            return ErrorRaised;
    }
    return OK;
}


// Parse the constants generated by ArrayTypeConstantBuilder.finalize()
// into an ArrayConstraint object.
static PyPtr parse_array_constraint(ConstantCursor& cursor, const Vec<int64_t>& static_shape_dims,
                                    const Vec<int64_t>& static_stride_dims) {
    ArrayType arrty = unpack_array_type(cursor.next());
    ArraySpecializationBits special_bits;
    special_bits.u64 = cursor.next();
    unsigned index_bitwidth = arrty.index_bitwidth;
    // Only int32 or int64 are supported now.
    CHECK(index_bitwidth == 32 || index_bitwidth == 64);

    PyObject* signature_module = get_signature_module();
    if (!signature_module) return {};

    PyPtr constraint_class = getattr(signature_module, "ArrayConstraint");
    if (!constraint_class) return {};

    PyPtr args = steal(PyTuple_New(0));
    if (!args) return {};

    PyPtr dtype = dtype_to_python(arrty.dtype);
    if (!dtype) return {};

    PyPtr index_dtype = dtype_to_python(
            DLDataType{kDLInt, static_cast<uint8_t>(index_bitwidth), 1});
    if (!index_dtype) return {};

    PyPtr constant_strides = steal(PyList_New(arrty.ndim));
    if (!constant_strides) return {};

    PyPtr constant_shape = steal(PyList_New(arrty.ndim));
    if (!constant_shape) return {};

    PyPtr stride_divisible_by = steal(PyTuple_New(arrty.ndim));
    if (!stride_divisible_by) return {};

    PyPtr shape_divisible_by = steal(PyTuple_New(arrty.ndim));
    if (!shape_divisible_by) return {};

    PyPtr zero = steal(PyLong_FromLong(0));
    if (!zero) return {};

    PyPtr one = steal(PyLong_FromLong(1));
    if (!one) return {};

    PyPtr sixteen = steal(PyLong_FromLong(DIVISOR_16));
    if (!sixteen) return {};

    PyPtr stride_divisor = one;
    constexpr unsigned divisor16_bits = DIVISOR_16 * BYTE_BITWIDTH;
    if (divisor16_bits % arrty.dtype.bits == 0) {
        stride_divisor = steal(PyLong_FromLong(divisor16_bits / arrty.dtype.bits));
        if (!stride_divisor) return {};
    }

    for (size_t i = 0; i < arrty.ndim; ++i) {
        PyObject* obj = special_bits.is_stride_one(i) ? one.get() : Py_None;
        PyList_SET_ITEM(constant_strides.get(), i, Py_NewRef(obj));

        obj = special_bits.is_stride_16byte_divisible(i) ? stride_divisor.get() : one.get();
        PyTuple_SET_ITEM(stride_divisible_by.get(), i, Py_NewRef(obj));

        obj = special_bits.is_shape_divisible_by_16(i) ? sixteen.get() : one.get();
        PyTuple_SET_ITEM(shape_divisible_by.get(), i, Py_NewRef(obj));

        PyList_SET_ITEM(constant_shape.get(), i, Py_NewRef(Py_None));
    }

    // Read in the same order finalize() pushed them: static_shape_value... then
    // static_stride_value....
    if (!read_static_axis_values(static_shape_dims, constant_shape.get(), cursor, arrty.ndim,
                                 "static_shape_dims"))
        return {};
    // When static_stride_dims is set, the stride==1 inference was
    // skipped, so constant_strides is all-None here and the duplicate check is meaningful.
    if (!read_static_axis_values(static_stride_dims, constant_strides.get(), cursor, arrty.ndim,
                                 "static_stride_dims"))
        return {};

    PyPtr kwargs = steal(Py_BuildValue(
            "{sO sI sO sO sO sO s() sO sO sO sO}",
            "dtype", dtype.get(),
            "ndim", static_cast<unsigned>(arrty.ndim),
            "index_dtype", index_dtype.get(),
            "stride_constant", constant_strides.get(),
            "shape_constant", constant_shape.get(),
            "stride_lower_bound_incl", zero.get(),
            "alias_groups",
            "may_alias_internally", special_bits.disjoint_elements ? Py_False : Py_True,
            "stride_divisible_by", stride_divisible_by.get(),
            "shape_divisible_by", shape_divisible_by.get(),
            "base_addr_divisible_by",
                special_bits.baseptr_16byte_aligned ? sixteen.get() : one.get()));
    if (!kwargs) return {};

    return steal(PyObject_Call(constraint_class.get(), args.get(), kwargs.get()));
}

#define UNPACK_ARRAY_INTERFACE(dict, key) \
    PyObject* key = PyDict_GetItemWithError((dict).get(), g_##key##_pyunicode); \
    if (!key) { \
        if (!PyErr_Occurred()) \
            PyErr_SetString(PyExc_TypeError, \
                            "__cuda_array_interface__ is missing the '" #key "' key"); \
        return ErrorRaised; \
    }


#define ASSERT_NDIM(ndim) \
    if (static_cast<uintmax_t>(ndim) > INT32_MAX) \
        return raise(PyExc_TypeError, "Input array exceeds max supported dimensions: %ld > %u", \
                     ndim, INT32_MAX);


static Result<ArrayRepr> arrayrepr_cuda_array_iface(PyObject* pyobj, unsigned index_bitwidth,
                                                    Arena& arena) {
    PyPtr dict = steal(PyObject_GetAttr(pyobj, g___cuda_array_interface___pyunicode));
    if (!PyDict_Check(dict.get())) {
        PyErr_SetString(PyExc_TypeError,
                        "__cuda_array_interface__ returned a non-dictionary object");
        return ErrorRaised;
    }

    UNPACK_ARRAY_INTERFACE(dict, typestr);
    UNPACK_ARRAY_INTERFACE(dict, shape);
    UNPACK_ARRAY_INTERFACE(dict, data);

    // Parse the dtype
    Result<DLDataType> dtype = parse_typestr(typestr);
    if (!dtype.is_ok()) return ErrorRaised;

    // Parse the data pointer
    if (!PyTuple_Check(data) || PyTuple_GET_SIZE(data) != 2) {
        PyErr_SetString(PyExc_TypeError,
                        "__cuda_array_interface['data'] is not a tuple of length 2");
        return ErrorRaised;
    }

    PyObject* data_ptr_pylong = PyTuple_GET_ITEM(data, 0);
    if (!PyLong_Check(data_ptr_pylong)) {
        PyErr_SetString(PyExc_TypeError, "__cuda_array_interface['data'][0] is not an integer");
        return ErrorRaised;
    }

    intptr_t data_ptr_int = pylong_as<intptr_t>(data_ptr_pylong);
    if (PyErr_Occurred()) return ErrorRaised;

    Py_ssize_t ndim = PyTuple_GET_SIZE(shape);
    ASSERT_NDIM(ndim);

    ArenaOffset repr_offset = arena_alloc_words(arena, 1 + 2 * ndim);
    arena[repr_offset].device_ptr = reinterpret_cast<void*>(data_ptr_int);

    // Parse the shape
    if (!PyTuple_Check(shape))
        return raise(PyExc_TypeError, "__cuda_array_interface['shape'] is not a tuple");

    for (Py_ssize_t i = 0; i < ndim; ++i) {
        int64_t size = pylong_as<int64_t>(PyTuple_GET_ITEM(shape, i));
        if (PyErr_Occurred()) return ErrorRaised;
        arena[repr_offset + 1 + i].i64 = size;
    }

    // Parse the strides
    PyObject* strides = PyDict_GetItem(dict.get(), g_strides_pyunicode);
    if (PyErr_Occurred()) return ErrorRaised;
    if (!strides || strides == Py_None) {
        if (!fill_row_major_strides(index_bitwidth, arena.data() + repr_offset, ndim))
            return ErrorRaised;
    } else if (PyTuple_Check(strides)) {
        // Only byte-aligned types should be supported by __cuda_array_interface__
        uint8_t dtype_bytewidth = dtype->bits / BYTE_BITWIDTH;
        for (Py_ssize_t i = 0; i < ndim; ++i) {
            int64_t stride = pylong_as<int64_t>(PyTuple_GET_ITEM(strides, i));
            if (PyErr_Occurred()) return ErrorRaised;
            arena[repr_offset + 1 + ndim + i].i64 = static_cast<int64_t>(
                    stride / dtype_bytewidth);
        }
    } else {
        return raise(PyExc_TypeError, "__cuda_array_interface['strides'] can only be"
                                      " absent, None, or a tuple");
    }

    return ArrayRepr {
        .arrty = {
            .dtype = *dtype,
            .ndim = static_cast<size_t>(ndim),
            .index_bitwidth = index_bitwidth,
        },
        .repr = repr_offset
    };
}

static Result<ArrayRepr> arrayrepr_dlpack_common(PyObject* dlpack_capsule, unsigned index_bitwidth,
                                                 Arena& arena) {
    void* ptr = PyCapsule_GetPointer(dlpack_capsule, "dltensor");
    if (!ptr) return ErrorRaised;
    DLManagedTensor* tensor = static_cast<DLManagedTensor*>(ptr);

    if (tensor->dl_tensor.device.device_type != kDLCUDA)
        return raise(PyExc_ValueError, "Input array is not on a CUDA device");

    // TODO: check device ID

    void* data_ptr = static_cast<char*>(tensor->dl_tensor.data) + tensor->dl_tensor.byte_offset;

    uint32_t ndim = tensor->dl_tensor.ndim;
    ASSERT_NDIM(ndim);

    ArenaOffset repr_offset = arena_alloc_words(arena, 1 + 2 * ndim);
    arena[repr_offset].device_ptr = data_ptr;

    for (uint32_t i = 0; i < ndim; ++i) {
        if (index_bitwidth != 64 && (tensor->dl_tensor.shape[i] < INT32_MIN
            || tensor->dl_tensor.shape[i] > INT32_MAX))
            return raise(PyExc_OverflowError, "shape is too big");
        arena[repr_offset + 1 + i].i64 = tensor->dl_tensor.shape[i];
    }

    if (!tensor->dl_tensor.strides) {
        if (!fill_row_major_strides(index_bitwidth, arena.data() + repr_offset, ndim))
            return ErrorRaised;
    } else {
        for (uint32_t i = 0; i < ndim; ++i) {
            if(index_bitwidth != 64 && (tensor->dl_tensor.strides[i] < INT32_MIN
                || tensor->dl_tensor.strides[i] > INT32_MAX))
                return raise(PyExc_OverflowError, "stride is too big");
            arena[repr_offset + 1 + ndim + i].i64 = tensor->dl_tensor.strides[i];
        }
    }

    ArrayRepr ret = {
        .arrty = {
            .dtype = tensor->dl_tensor.dtype,
            .ndim = ndim,
            .index_bitwidth = index_bitwidth,
        },
        .repr = repr_offset
    };

    PyCapsule_SetName(dlpack_capsule, "used_dltensor");

    // We assume that __dlpack__ returns a view of the tensor,
    // so we release the capsule immediately. This should be OK for using with PyTorch
    // since it always returns a view.
    //
    // This is technically an incorrect implementation. To do it correctly, we would
    // need to implement a mechanism similar to the one found in Torch's CUDACachingAllocator:
    // instead of calling the deleter immediately, we would push a cudaEvent to the stream
    // after we launch the kernel, and only call the deleter once the event is ready.
    tensor->deleter(tensor);
    return ret;
}


static Result<DLDataType> dtype_from_torch_dtype(PyObject* torch_dtype) {
    HashMap<PyPtr, ForeignDTypeInfo>::Item* item = get_foreign_dtype_registry()->find(torch_dtype);
    if (!item || !item->value.dlpack_dtype)
        return raise(PyExc_TypeError, "dtype is not supported");
    return *item->value.dlpack_dtype;
}

static Result<ArrayRepr> arrayrepr_torch_tensor_pymethod(PyObject* tensor, unsigned index_bitwidth,
                                                         Arena& arena) {
    PyPtr data_ptr = steal(PyObject_CallMethod(tensor, "data_ptr", nullptr));
    if (!data_ptr) return ErrorRaised;

    PyPtr shape_ptr = steal(PyObject_GetAttrString(tensor, "shape"));
    if (!shape_ptr) return ErrorRaised;

    PyPtr dtype_ptr = steal(PyObject_GetAttrString(tensor, "dtype"));
    if (!dtype_ptr) return ErrorRaised;

    PyPtr stride_ptr = steal(PyObject_CallMethod(tensor, "stride", nullptr));
    if (!stride_ptr) return ErrorRaised;

    if (!PyLong_Check(data_ptr.get()))
        return raise(PyExc_TypeError, "data_ptr cannot be converted to int");
    long long addr = PyLong_AsLongLong(data_ptr.get());
    if (PyErr_Occurred()) return ErrorRaised;

    // Extract shape
    if (!PyTuple_Check(shape_ptr.get()))
        return raise(PyExc_TypeError, "expect shape to be an tuple");
    Py_ssize_t len = PyTuple_GET_SIZE(shape_ptr.get());
    if (len == -1) return ErrorRaised;
    if (len > INT32_MAX)
        return raise(PyExc_OverflowError, "rank is too big");
    uint32_t ndim = len;
    ASSERT_NDIM(ndim);

    ArenaOffset repr_offset = arena_alloc_words(arena, 1 + 2 * ndim);
    arena[repr_offset].device_ptr = reinterpret_cast<void*>(addr);

    for (uint32_t i = 0; i < ndim; ++i) {
        PyObject* item_ptr = PyTuple_GetItem(shape_ptr.get(), i);
        if (!item_ptr) return ErrorRaised;
        if (!PyLong_Check(item_ptr))
            return raise(PyExc_TypeError, "unexpected type from .shape");
        long long si = PyLong_AsLongLong(item_ptr);
        if (PyErr_Occurred()) return ErrorRaised;

        if (index_bitwidth != 64 && (si < INT32_MIN || si > INT32_MAX))
            return raise(PyExc_OverflowError, "shape is too big");
        arena[repr_offset + 1 + i].i64 = static_cast<int64_t>(si);
    }

    // Extract stride
    if (!PyTuple_Check(stride_ptr.get()))
        return raise(PyExc_TypeError, "expect stride to be an tuple");
    Py_ssize_t stride_len = PyTuple_GET_SIZE(stride_ptr.get());
    if (stride_len == -1) return ErrorRaised;
    if (stride_len != ndim)
        return raise(PyExc_ValueError, "shape and stride have different length");

    for (uint32_t i = 0; i < ndim; ++i) {
        PyObject* item_ptr = PyTuple_GetItem(stride_ptr.get(), i);
        if (!item_ptr) return ErrorRaised;
        if (!PyLong_Check(item_ptr))
            return raise(PyExc_TypeError, "unexpected type of .stride");
        long long si = PyLong_AsLongLong(item_ptr);
        if (PyErr_Occurred()) return ErrorRaised;
        if (index_bitwidth != 64 && (si < INT32_MIN || si > INT32_MAX))
            return raise(PyExc_OverflowError, "stride is too big");
        arena[repr_offset + 1 + ndim + i].i64 = static_cast<int64_t>(si);
    }


    Result<DLDataType> dtype_res = dtype_from_torch_dtype(dtype_ptr.get());
    if (!dtype_res.is_ok())
        return ErrorRaised;

    return ArrayRepr{
        .arrty = {
            .dtype = *dtype_res,
            .ndim = ndim,
            .index_bitwidth = index_bitwidth,
        },
        .repr = repr_offset,
    };
}

static Result<ArrayRepr> arrayrepr_torch_tensor_dlpack(PyObject* pyobj, unsigned index_bitwidth,
                                                       Arena& arena) {
    // Safe to assume try_get_torch_to_dlpack_func() is not null because we wouldn't have produced
    // a PythonArgKind::TorchTensorDlpack value otherwise.
    PyPtr dlpack_capsule = steal(PyObject_CallFunctionObjArgs(
                try_get_torch_to_dlpack_func(), pyobj, nullptr));

    if (!dlpack_capsule) {
        SavedException exc = save_raised_exception();
        LOG_PYTHON_ERROR("debug", exc, "Fail to convert to dlpack, use fallback path");
        return arrayrepr_torch_tensor_pymethod(pyobj, index_bitwidth, arena);
    }

    return arrayrepr_dlpack_common(dlpack_capsule.get(), index_bitwidth, arena);
}

static Result<ArrayRepr> arrayrepr_dlpack(PyObject* pyobj, unsigned index_bitwidth,
                                          Arena& arena) {
    PyPtr dlpack_method = steal(PyObject_GetAttr(pyobj, g___dlpack___pyunicode));
    if (!dlpack_method) return ErrorRaised;

    PyPtr empty_args = steal(PyTuple_New(0));
    if (!empty_args) return ErrorRaised;

    PyPtr kwargs = steal(PyDict_New());
    if (!kwargs) return ErrorRaised;

    // stream -1 signals "producer must not perform any synchronization"
    PyPtr stream_value = steal(PyLong_FromLong(-1));
    if (!stream_value) return ErrorRaised;
    PyDict_SetItemString(kwargs.get(), "stream", stream_value.get());

    PyPtr dlpack_capsule = steal(PyObject_Call(
                dlpack_method.get(), empty_args.get(), kwargs.get()));
    if (!dlpack_capsule) return ErrorRaised;

    return arrayrepr_dlpack_common(dlpack_capsule.get(), index_bitwidth, arena);
}


struct ScalarAnnotation {
    unsigned bitwidth = 32;
};

struct ArrayAnnotation {
    unsigned index_bitwidth = 32;
    Vec<int64_t> static_shape_dims; // array shape dims specialized to launch-time values.
    Vec<int64_t> static_stride_dims; // array stride dims specialized to launch-time values.
};

struct ListAnnotation {
    ArrayAnnotation element;
};


typedef Result<ArrayRepr> (*ArrayReprFunc)(PyObject*, unsigned, Arena&);


template <ArrayReprFunc F>
static Status extract_array(const DriverApi* driver, PyObject* pyobj,
                            const ArrayAnnotation& array_ann,
                            LaunchHelper& helper) {
    Result<ArrayRepr> ar = F(pyobj, array_ann.index_bitwidth, helper.arena);
    if (!ar.is_ok()) return ErrorRaised;

    size_t num_words = 1 + 2 * ar->arrty.ndim;
    helper.array_ptr_arena_offsets.push_back(ar->repr);
    for (size_t i = 0; i < num_words; ++i)
        helper.cuarg_offsets.push_back(ar->repr + i);

    ArrayTypeConstantBuilder builder;
    if (!builder.update(helper.arena, *ar, array_ann.static_shape_dims,
                        array_ann.static_stride_dims))
        return ErrorRaised;
    if (!builder.finalize(driver, ar->arrty, array_ann.static_shape_dims,
                          array_ann.static_stride_dims, helper))
        return ErrorRaised;
    return OK;
}

enum class PylongConstantEncoding : int64_t {
    I64,
    U64
};

static inline Status extract_bool_constant(PyObject* pyobj, Vec<int64_t>* constants) {
    int val = PyObject_IsTrue(pyobj);
    if (val < 0) return ErrorRaised;
    constants->push_back(val);
    return OK;
}

static inline Status extract_py_bool(PyObject* pyobj, LaunchHelper& helper) {
    int val = PyObject_IsTrue(pyobj);
    if (val < 0) return ErrorRaised;
    push_single_word_cuarg(helper, {.i32 = val});
    return OK;
}

static PyPtr make_scalar_constraint(DLDataType dtype) {
    PyObject* signature_module = get_signature_module();
    if (!signature_module) return {};

    PyPtr py_dtype = dtype_to_python(dtype);
    if (!py_dtype) return {};

    return steal(PyObject_CallMethod(
                signature_module, "ScalarConstraint", "(O)", py_dtype.get()));
}

static PyPtr make_constant_constraint(PyObject* value) {
    PyObject* signature_module = get_signature_module();
    if (!signature_module) return {};

    return steal(PyObject_CallMethod(signature_module, "ConstantConstraint", "(O)", value));
}

static PyPtr parse_bool_constant_constraint(ConstantCursor& cursor) {
    int64_t val = cursor.next();
    return make_constant_constraint(val ? Py_True : Py_False);
}

static inline Status extract_int_constant(PyObject* pyobj, Vec<int64_t>* constants) {
    int overflow;
    int64_t value = pylong_as_overflow_and<int64_t>(pyobj, &overflow);
    if (PyErr_Occurred()) return ErrorRaised;
    if (overflow) {
        // TODO: support big values by extracting all digits
        constants->push_back(static_cast<int64_t>(PylongConstantEncoding::U64));
        uint64_t uval = pylong_as<uint64_t>(pyobj);
        if (PyErr_Occurred()) return ErrorRaised;
        constants->push_back(uval);
    } else {
        constants->push_back(static_cast<int64_t>(PylongConstantEncoding::I64));
        constants->push_back(value);
    }
    return OK;
}

static PyPtr parse_int_constant_constraint(ConstantCursor& cursor) {
    int64_t format = cursor.next();
    PyPtr value;
    if (format == static_cast<int64_t>(PylongConstantEncoding::I64)) {
        value = steal(PyLong_FromLongLong(cursor.next()));
    } else if (format == static_cast<int64_t>(PylongConstantEncoding::U64)) {
        value = steal(PyLong_FromUnsignedLongLong(cursor.next()));
    } else {
        CHECK_UNREACHABLE;
    }
    if (!value) return {};
    return make_constant_constraint(value.get());
}

static inline Status extract_py_long(PyObject* pyobj, unsigned bitwidth, LaunchHelper& helper) {
    if (bitwidth == 64) {
        int64_t value = pylong_as<int64_t>(pyobj);
        if (PyErr_Occurred()) return ErrorRaised;
        push_single_word_cuarg(helper, {.i64 = value});
    } else {
        CHECK(bitwidth == 32);
        int32_t value = pylong_as<int32_t>(pyobj);
        if (PyErr_Occurred()) return ErrorRaised;
        push_single_word_cuarg(helper, {.i32 = value});
    }
    return OK;
}

static void extract_float_constant(PyObject* pyobj, Vec<int64_t>* constants) {
    double value = PyFloat_AS_DOUBLE(pyobj);
    int64_t i64_val = 0;
    static_assert(sizeof(i64_val) == sizeof(value));
    mem_copy(&i64_val, &value, sizeof(i64_val));
    constants->push_back(i64_val);
}

static void extract_py_float(PyObject* pyobj, LaunchHelper& helper) {
    double value = PyFloat_AS_DOUBLE(pyobj);
    push_single_word_cuarg(helper, {.f32 = static_cast<float>(value)});
}

static PyPtr parse_float_constant_constraint(ConstantCursor& cursor) {
    union { int64_t i64; double f64; } u;
    u.i64 = cursor.next();
    PyPtr value = steal(PyFloat_FromDouble(u.f64));
    return make_constant_constraint(value.get());
}

static void extract_identity_constant(PyObject* object, Vec<int64_t>* constants,
                                      Vec<PyObject*>* identity_constants) {
    constants->push_back(reinterpret_cast<int64_t>(object));
    identity_constants->push_back(object);
}


static PyPtr parse_identity_constant_constraint(ConstantCursor& cursor,
                                                const Vec<PyObject*>& identity_constants) {
    int64_t address = cursor.next();
    for (PyObject* obj : identity_constants) {
        if (reinterpret_cast<int64_t>(obj) == address)
            return make_constant_constraint(obj);
    }
    CHECK_UNREACHABLE;
}

static Status extract_string_constant(PyObject* pyobj, Vec<int64_t>* constants,
                                      Vec<PyObject*>* identity_constants,
                                      Vec<PyPtr>* pyarg_refs) {
    if (!PyUnicode_CHECK_INTERNED(pyobj)) {
        PyPtr ref = newref(pyobj);
        pyunicode_intern_in_place(&ref);
        if (!PyUnicode_CHECK_INTERNED(ref.get()))
            return raise(PyExc_RuntimeError, "Failed to intern a string kernel argument");
        pyobj = ref.get();
        pyarg_refs->push_back(std::move(ref));
    }
    extract_identity_constant(pyobj, constants, identity_constants);
    return OK;
}

static Status extract_foreign_dtype_constant(PyObject* object, Vec<int64_t>* constants,
                                             Vec<PyObject*>* identity_constants) {
    HashMap<PyPtr, ForeignDTypeInfo>::Item* item = get_foreign_dtype_registry()->find(object);
    if (!item)
        return raise(PyExc_ValueError, "Received an unregistered foreign dtype object");
    extract_identity_constant(item->value.native_dtype.get(), constants, identity_constants);
    return OK;
}

static Result<ArrayRepr> get_array_repr(PythonArgKind kind, PyObject* pyobj,
                                        unsigned index_bitwidth, Arena& arena) {
    switch (kind) {
        case PythonArgKind::TorchTensorDlpack:
            return arrayrepr_torch_tensor_dlpack(pyobj, index_bitwidth, arena);
        case PythonArgKind::DlpackArray:
            return arrayrepr_dlpack(pyobj, index_bitwidth, arena);
        case PythonArgKind::CudaArray:
            return arrayrepr_cuda_array_iface(pyobj, index_bitwidth, arena);
        default:
            return raise(PyExc_AssertionError, "Unexpected argument kind for array: %d",
                         static_cast<int>(kind));
    }
}

static Result<PythonArgKind> classify_list_item(PyObject* item, size_t index) {
    std::optional<PythonArgKind> res = classify_nonconstant_arg(item);
    if (!res.has_value()) {
        return raise(PyExc_TypeError, "Invalid list item #%zu: unsupported object type '%s'",
                index, Py_TYPE(item)->tp_name);
    }
    return *res;
}

static Status extract_py_list(const DriverApi* driver, PyObject* pyobj,
                              const ListAnnotation& list_ann, LaunchHelper& helper) {
    size_t len = PyList_GET_SIZE(pyobj);
    if (len > INT32_MAX)
        return raise(PyExc_TypeError, "List is too long");

    // TODO: support empty list as its own type?
    if (!len)
        return raise(PyExc_TypeError, "Empty lists are not supported as kernel arguments");

    // Handle the first item separately in order to determine the item type

    PyObject* first_item = PyList_GET_ITEM(pyobj, 0);
    Result<PythonArgKind> first_item_res = classify_list_item(first_item, 0);
    if (!first_item_res.is_ok()) return ErrorRaised;

    if (param_category_from_pyarg_kind(*first_item_res) != ParameterKind::Array) {
        return raise(PyExc_TypeError, "Expected list items to be arrays, got %s",
                     Py_TYPE(first_item)->tp_name);
    }

    PythonArgKind first_arg_kind = *first_item_res;
    PyTypeObject* first_item_type = first_item->ob_type;

    Result<ArrayRepr> first_repr_res = get_array_repr(first_arg_kind, first_item,
                                                      list_ann.element.index_bitwidth,
                                                      helper.arena);
    if (!first_repr_res.is_ok()) return ErrorRaised;

    helper.array_ptr_arena_offsets.push_back(first_repr_res->repr);
    ArenaOffset item_offsets = arena_alloc_words(helper.arena, len);
    size_t item_size_words = 1 + 2 * first_repr_res->arrty.ndim;

    // Push a relative offset in place of the base pointer for now, since we don't know the actual
    // address yet. We will patch it later via `ListArg.base_ptr_cuarg`).
    ArenaOffset base_ptr_cuarg = push_single_word_cuarg(
            helper, {.size = helper.total_list_data_size_words});
    helper.total_list_data_size_words += len * item_size_words;
    push_single_word_cuarg(helper, {.i32 = static_cast<int32_t>(len)});

    helper.list_args.push_back({.base_ptr_cuarg = base_ptr_cuarg,
                                .length = len,
                                .item_offsets = item_offsets,
                                .item_size_words = item_size_words});
    helper.arena[item_offsets].arena_offset = first_repr_res->repr;

    ArrayTypeConstantBuilder builder;
    if (!builder.update(helper.arena, *first_repr_res, list_ann.element.static_shape_dims,
                        list_ann.element.static_stride_dims))
        return ErrorRaised;

    // Handle the rest of the list
    for (size_t i = 1; i < len; ++i) {
        PyObject* item = PyList_GET_ITEM(pyobj, i);
        PythonArgKind kind = first_arg_kind;

        // Avoid calling classify_list_item() if the object type is the same
        if (first_item_type != item->ob_type) {
             Result<PythonArgKind> res = classify_list_item(item, i);
             if (!res.is_ok()) return ErrorRaised;
             kind = *res;
        }

        Result<ArrayRepr> repr_res = get_array_repr(kind, item, list_ann.element.index_bitwidth,
                                                    helper.arena);
        if (!repr_res.is_ok()) return ErrorRaised;
        helper.array_ptr_arena_offsets.push_back(repr_res->repr);
        helper.arena[item_offsets + i].arena_offset = repr_res->repr;

        // TODO: nicer error messages
        if (dtype_as_uint(first_repr_res->arrty.dtype) != dtype_as_uint(repr_res->arrty.dtype))
            return raise(PyExc_TypeError, "Arrays in list vary in data type");
        if (first_repr_res->arrty.ndim != repr_res->arrty.ndim)
            return raise(PyExc_TypeError, "Arrays in list vary in rank");

        if (!builder.update(helper.arena, *repr_res, list_ann.element.static_shape_dims,
                            list_ann.element.static_stride_dims))
            return ErrorRaised;
    }

    // TODO: If we accept lists of things other than arrays, then to disambiguate,
    //       we need to push another constant here that specifies the type of the list element .
    if (!builder.finalize(driver, first_repr_res->arrty, list_ann.element.static_shape_dims,
                          list_ann.element.static_stride_dims, helper))
        return ErrorRaised;
    return OK;
}

static PyPtr parse_list_constraint(ConstantCursor& cursor, const Vec<int64_t>& static_shape_dims,
                                   const Vec<int64_t>& static_stride_dims) {
    PyPtr element = parse_array_constraint(cursor, static_shape_dims, static_stride_dims);
    if (!element) return {};

    PyObject* signature_module = get_signature_module();
    if (!signature_module) return {};

    PyPtr constraint_class = getattr(signature_module, "ListConstraint");
    if (!constraint_class) return {};

    PyPtr args = steal(PyTuple_New(0));
    if (!args) return {};

    PyPtr kwargs = steal(Py_BuildValue(
            "{sO s() sO}",
            "element", element.get(),
            "alias_groups",
            "elements_may_alias", Py_True
            ));
    if (!kwargs) return {};

    return steal(PyObject_Call(constraint_class.get(), args.get(), kwargs.get()));
}

struct AggregateCursor {
    Cursor<PyTypeObject*> pytype_cursor;
    Cursor<std::optional<AggregateArgType>> agg_cursor;

    bool at_aggregate_end() const {
        return pytype_cursor.peek() == nullptr;
    }

    const std::optional<AggregateArgType>& next() {
        pytype_cursor.next();
        return agg_cursor.next();
    }
};

struct ParameterAnnotationNode : SimpleRefcount<ParameterAnnotationNode> {
    enum Kind { Leaf, HomogeneousTuple, HeterogeneousTuple };
    virtual Kind kind() const = 0;
    virtual Status flatten_aggregate(AggregateCursor* cursor,
                                     const AggregateArgType& agg_type,
                                     Vec<RefPtr<LeafAnnotationNode>>* out) = 0;
    virtual ~ParameterAnnotationNode() {}
};


static Status flatten_parameter_annotation_node(ParameterAnnotationNode* node,
                                                AggregateCursor* cursor,
                                                Vec<RefPtr<LeafAnnotationNode>>* out);


struct LeafAnnotationNode : ParameterAnnotationNode {
    bool constant = false;
    ScalarAnnotation scalar;
    ArrayAnnotation array;
    ListAnnotation list;

    virtual Kind kind() const { return Leaf; }

    virtual Status flatten_aggregate(AggregateCursor* cursor,
                                     const AggregateArgType& agg_type,
                                     Vec<RefPtr<LeafAnnotationNode>>* out) override {
        for (size_t item_idx = 0; !cursor->at_aggregate_end(); ++item_idx) {
            if (!flatten_parameter_annotation_node(this, cursor, out))
                return ErrorRaised;
        }
        return OK;
    }
};


struct HomogeneousTupleNode : ParameterAnnotationNode {
    RefPtr<ParameterAnnotationNode> each;

    virtual Kind kind() const { return HomogeneousTuple; }

    virtual Status flatten_aggregate(AggregateCursor* cursor,
                                     const AggregateArgType& agg_type,
                                     Vec<RefPtr<LeafAnnotationNode>>* out) override {
        if (agg_type.kind != AggregateArgType::Tuple)
            return raise(PyExc_TypeError,
                         "Received a non-tuple argument for a parameter annotated as a tuple");
        for (size_t item_idx = 0; !cursor->at_aggregate_end(); ++item_idx) {
            if (!flatten_parameter_annotation_node(each.get(), cursor, out))
                return ErrorRaised;
        }
        return OK;
    }
};


struct HeterogeneousTupleNode : ParameterAnnotationNode {
    Vec<RefPtr<ParameterAnnotationNode>> items;

    virtual Kind kind() const { return HeterogeneousTuple; }

    virtual Status flatten_aggregate(AggregateCursor* cursor,
                                     const AggregateArgType& agg_type,
                                     Vec<RefPtr<LeafAnnotationNode>>* out) override {
        if (agg_type.kind != AggregateArgType::Tuple)
            return raise(PyExc_TypeError,
                         "Received a non-tuple argument for a parameter annotated as a tuple");

        LeafAnnotationNode default_node;

        size_t item_idx = 0;
        while (!cursor->at_aggregate_end()) {
            ParameterAnnotationNode* item_node;
            if (item_idx < items.size()) {
                item_node = items[item_idx].get();
            } else {
                // Use a dummy node to keep going so that at the end, we generate an error
                // message with the correct tuple length.
                item_node = &default_node;
            }
            ++item_idx;
            if (!flatten_parameter_annotation_node(item_node, cursor, out))
                return ErrorRaised;
        }
        if (item_idx != items.size())
            return raise(PyExc_TypeError,
                         "Received a tuple of length %zu"
                         " for a parameter annotated as a tuple of length %zu",
                         item_idx, items.size());
        return OK;
    }
};

static Status flatten_parameter_annotation_node(ParameterAnnotationNode* node,
                                                AggregateCursor* cursor,
                                                Vec<RefPtr<LeafAnnotationNode>>* out) {
    const std::optional<AggregateArgType>& agg_type = cursor->next();
    if (agg_type.has_value()) {
        if (!node->flatten_aggregate(cursor, *agg_type, out))
            return ErrorRaised;
        CHECK(cursor->at_aggregate_end());
        cursor->next();
    } else {
        if (node->kind() != ParameterAnnotationNode::Leaf)
            return raise(PyExc_TypeError,
                         "Received a non-tuple argument for a parameter annotated as a tuple");
        out->push_back(newref(static_cast<LeafAnnotationNode*>(node)));
    }
    return OK;
}

static Result<Vec<RefPtr<LeafAnnotationNode>>>
flatten_parameter_annotation_nodes(const Vec<RefPtr<ParameterAnnotationNode>>& nodes,
                                   const Vec<PyTypeObject*>& pyarg_types_depth_first,
                                   const Vec<std::optional<AggregateArgType>>& agg_types,
                                   size_t num_leaves) {
    Vec<RefPtr<LeafAnnotationNode>> ret;
    ret.reserve(num_leaves);
    AggregateCursor cursor{{pyarg_types_depth_first}, {agg_types}};
    for (const RefPtr<ParameterAnnotationNode>& node : nodes) {
        if (!flatten_parameter_annotation_node(node.get(), &cursor, &ret))
            return ErrorRaised;
    }
    CHECK(cursor.pytype_cursor.len == 1);
    CHECK(cursor.agg_cursor.len == 1);
    CHECK(cursor.at_aggregate_end());
    CHECK(ret.size() == num_leaves);
    return ret;
}

static Status extract_arg(const DriverApi* driver, PyObject* obj, PythonArgKind kind,
                          LeafAnnotationNode* annotation, LaunchHelper& helper) {
    switch (kind) {
    case PythonArgKind::ConstantBool:
        return extract_bool_constant(obj, &helper.constants);
    case PythonArgKind::ConstantInt:
        return extract_int_constant(obj, &helper.constants);
    case PythonArgKind::ConstantFloat:
        extract_float_constant(obj, &helper.constants);
        return OK;
    case PythonArgKind::ConstantNone:
        return OK;
    case PythonArgKind::ConstantString:
        return extract_string_constant(obj, &helper.constants, &helper.identity_constants,
                                       &helper.pyarg_refs);
    case PythonArgKind::IdentityConstant:
        extract_identity_constant(obj, &helper.constants, &helper.identity_constants);
        return OK;
    case PythonArgKind::ForeignDTypeConstant:
        return extract_foreign_dtype_constant(obj, &helper.constants, &helper.identity_constants);
    case PythonArgKind::TorchTensorDlpack:
        return extract_array<arrayrepr_torch_tensor_dlpack>(driver, obj, annotation->array, helper);
    case PythonArgKind::DlpackArray:
        return extract_array<arrayrepr_dlpack>(driver, obj, annotation->array, helper);
    case PythonArgKind::CudaArray:
        return extract_array<arrayrepr_cuda_array_iface>(driver, obj, annotation->array, helper);
    case PythonArgKind::PyBool:
        return extract_py_bool(obj, helper);
    case PythonArgKind::PyLong:
        return extract_py_long(obj, annotation->scalar.bitwidth, helper);
    case PythonArgKind::PyFloat:
        extract_py_float(obj, helper);
        return OK;
    case PythonArgKind::PyList:
        return extract_py_list(driver, obj, annotation->list, helper);
    }
    CHECK_UNREACHABLE;
}

static Status extract_cuda_args(const DriverApi* driver,
                                const Vec<PyObject*>& pyarg_objs,
                                const Vec<PythonArgKind>& arg_kinds,
                                const Vec<RefPtr<LeafAnnotationNode>>& flat_param_annotations,
                                LaunchHelper& helper) {
    CHECK(pyarg_objs.size() == arg_kinds.size());
    CHECK(flat_param_annotations.size() == arg_kinds.size());
    helper.arena.clear();
    helper.cuarg_offsets.clear();
    helper.array_ptr_arena_offsets.clear();
    helper.list_args.clear();
    helper.total_list_data_size_words = 0;
    helper.constants.clear();
    helper.identity_constants.clear();
    for (size_t i = 0; i < arg_kinds.size(); ++i) {
        PythonArgKind kind = arg_kinds[i];
        if (!extract_arg(driver, pyarg_objs[i], kind, flat_param_annotations[i].get(), helper))
            return ErrorRaised;
    }
    return OK;
}

static PyPtr parse_element_constraint(ConstantCursor& cursor, ParameterKind::Category category,
                                      const LeafAnnotationNode& annotation,
                                      const Vec<PyObject*>& identity_constants) {
    switch (category) {
    case ParameterKind::ConstantBool:
        return parse_bool_constant_constraint(cursor);
    case ParameterKind::ConstantInt:
        return parse_int_constant_constraint(cursor);
    case ParameterKind::ConstantFloat:
        return parse_float_constant_constraint(cursor);
    case ParameterKind::ConstantNone:
        return make_constant_constraint(Py_None);
    case ParameterKind::IdentityConstant:
        return parse_identity_constant_constraint(cursor, identity_constants);
    case ParameterKind::Array:
        return parse_array_constraint(cursor, annotation.array.static_shape_dims,
                                      annotation.array.static_stride_dims);
    case ParameterKind::Boolean:
        return make_scalar_constraint(DLDataType{kDLBool, 8, 1});
    case ParameterKind::Integer:
        return make_scalar_constraint(
                DLDataType{kDLInt, static_cast<uint8_t>(annotation.scalar.bitwidth), 1});
    case ParameterKind::Float:
        return make_scalar_constraint(DLDataType{kDLFloat, 32, 1});
    case ParameterKind::List:
        return parse_list_constraint(cursor, annotation.list.element.static_shape_dims,
                                     annotation.list.element.static_stride_dims);
    case ParameterKind::AggregateBegin:
    case ParameterKind::AggregateEnd:
        CHECK_UNREACHABLE;  // Should be handled before parse_element_constraint
    }
    CHECK_UNREACHABLE;
}

static PyPtr create_tuple_constraint(PyObject* items_list) {
    PyObject* signature_module = get_signature_module();
    if (!signature_module) return {};
    PyPtr constraint_class = getattr(signature_module, "TupleConstraint");
    if (!constraint_class) return {};
    return steal(PyObject_CallOneArg(constraint_class.get(), items_list));
}

#ifdef ENABLE_CCONV_V3
static PyPtr create_dataclass_constraint(PyObject* dataclass, PyObject* items_list) {
    PyObject* signature_module = get_signature_module();
    if (!signature_module) return {};
    PyPtr constraint_class = getattr(signature_module, "DataclassConstraint");
    if (!constraint_class) return {};
    return steal(PyObject_CallFunctionObjArgs(constraint_class.get(),
                                              dataclass, items_list, nullptr));
}
#endif

static PyPtr parse_param_constraint(ConstantCursor& cursor,
                                    Cursor<ParameterKind>* param_cursor,
                                    Cursor<RefPtr<LeafAnnotationNode>>* annotation_cursor,
                                    const Vec<PyObject*>& identity_constants) {
    const ParameterKind& pk = param_cursor->next();
    if (pk.category == ParameterKind::AggregateBegin) {
        PyPtr items_list = steal(PyList_New(0));
        if (!items_list) return {};
        while (param_cursor->peek().category != ParameterKind::AggregateEnd) {
            PyPtr item = parse_param_constraint(cursor, param_cursor, annotation_cursor,
                                                identity_constants);
            if (!item) return {};
            if (PyList_Append(items_list.get(), item.get()) < 0) return {};
        }
        const ParameterKind& aggregate_end = param_cursor->next();
        CHECK(aggregate_end.category == ParameterKind::AggregateEnd);

        switch (pk.agg_type.kind) {
        case AggregateArgType::Tuple:
            return create_tuple_constraint(items_list.get());
#ifdef ENABLE_CCONV_V3
        case AggregateArgType::Dataclass:
            return create_dataclass_constraint(pk.agg_type.dataclass_info->dataclass.get(),
                                               items_list.get());
#endif
        }
        CHECK(false);
    }
    LeafAnnotationNode* annotation = annotation_cursor->next().get();
    return parse_element_constraint(cursor, pk.category, *annotation, identity_constants);
}

static PyPtr parse_parameter_constraints(
        ConstantCursor cursor,
        const Vec<PyObject*>& identity_constants,
        const Vec<ParameterKind>& param_kinds,
        const Vec<RefPtr<LeafAnnotationNode>>& flat_param_annotations) {
    PyPtr param_constraints = steal(PyList_New(0));
    if (!param_constraints) return {};

    Cursor<ParameterKind> param_cursor(param_kinds);
    Cursor<RefPtr<LeafAnnotationNode>> annotation_cursor(flat_param_annotations);
    while (param_cursor.peek().category != ParameterKind::AggregateEnd) {
        PyPtr constraint = parse_param_constraint(cursor, &param_cursor, &annotation_cursor,
                                                  identity_constants);
        if (!constraint) return {};
        if (PyList_Append(param_constraints.get(), constraint.get()))
            return {};
    }
    CHECK(param_cursor.len == 1);  // only the final sentinel remaining
    CHECK(cursor.len == 0);
    CHECK(annotation_cursor.len == 0);
    return param_constraints;
}

static CallConvVersion minimum_calling_convention(
        const Vec<ParameterKind>& param_kinds,
        const Vec<RefPtr<LeafAnnotationNode>>& flat_param_annotations) {
    CallConvVersion version = CallConvVersion::CutilePython_V1;
    auto require = [&] (CallConvVersion u) { if (u > version) version = u; };

    for (const ParameterKind& pk : param_kinds) {
        if (pk.category == ParameterKind::AggregateBegin) {
            switch (pk.agg_type.kind) {
            case AggregateArgType::Tuple:
                require(CallConvVersion::CutilePython_V2);
                break;
#ifdef ENABLE_CCONV_V3
            case AggregateArgType::Dataclass:
                require(CallConvVersion::CutilePython_V3);
                break;
#endif
            }
        }
    }
    for (const RefPtr<LeafAnnotationNode>& f : flat_param_annotations) {
        if (!f->array.static_shape_dims.empty() || !f->list.element.static_shape_dims.empty())
            require(CallConvVersion::CutilePython_V2);
    }
    return version;
}

static PyPtr make_signature(ConstantCursor constants,
                            const Vec<PyObject*>& identity_constants,
                            const Vec<ParameterKind>& param_kinds,
                            const Vec<RefPtr<LeafAnnotationNode>>& flat_param_annotations,
                            const PyPtr& calling_convention) {
    PyPtr parameters = parse_parameter_constraints(constants, identity_constants, param_kinds,
                                                   flat_param_annotations);
    if (!parameters) return {};

    PyObject* signature_module = get_signature_module();
    if (!signature_module) return {};

    PyPtr signature_class = getattr(signature_module, "KernelSignature");
    if (!signature_class) return {};

    return steal(PyObject_CallFunctionObjArgs(
            signature_class.get(), parameters.get(), calling_convention.get(), nullptr));
}

namespace { struct TileContext {
    PyPtr config;
    PyPtr autotune_cache;
#ifdef Py_GIL_DISABLED
    PyMutex accessor_mutex = {0};
#endif

    static PyTypeObject pytype;
}; }


struct TileContextDispatcher {
    ProfileMap arg_profiles;
    Vec<RefPtr<KernelFamily>> kernel_families;
};


static void host_program_eval(const HostProgram& prog,
                              const Arena& arena,
                              const Vec<ArenaOffset>& cuarg_offsets,
                              int64_t stack[HostProgram::kMaxStackDepth]) {
    int64_t* top = stack;
    const int64_t* op_attrs = prog.op_attrs.data();
    for (SizeOpcode opcode : prog.opcodes) {
        switch (opcode) {
        case SizeOpcode::Const: *top++ = *op_attrs++; break;
        case SizeOpcode::KernelArgI32: *top++ = arena[cuarg_offsets[*op_attrs++]].i32; break;
        case SizeOpcode::KernelArgI64: *top++ = arena[cuarg_offsets[*op_attrs++]].i64; break;
        case SizeOpcode::Add: top[-2] += top[-1]; --top; break;  // TODO: overflow check?
        case SizeOpcode::Mul: top[-2] *= top[-1]; --top; break;  // TODO: overflow check?
        case SizeOpcode::RoundUpToPow2: {
            const int64_t alignment = *op_attrs++;
            const int64_t mask = alignment - 1;
            const int64_t value = top[-1];
            top[-1] = (value + mask) & ~mask;
            break;
        }
        }
    }
}

static Result<HostProgram> host_program_parse(PyObject* prog_pyobj, int expected_results) {
    if (prog_pyobj == Py_None)
        return HostProgram{{SizeOpcode::Const}, {0}};
    PyPtr opcodes_pylist = getattr(prog_pyobj, "opcodes");
    if (!opcodes_pylist) return ErrorRaised;
    PyPtr attrs_pylist = getattr(prog_pyobj, "op_attrs");
    if (!attrs_pylist) return ErrorRaised;

    Py_ssize_t num_opcodes = PyList_Size(opcodes_pylist.get());
    Py_ssize_t num_attrs = PyList_Size(attrs_pylist.get());
    if (PyErr_Occurred()) return ErrorRaised;

    HostProgram prog;
    Py_ssize_t remaining_attrs = num_attrs;
    int depth = 0;
    for (Py_ssize_t i = 0; i < num_opcodes; ++i) {
        PyObject* py_opcode = PyList_GetItem(opcodes_pylist.get(), i);
        if (!py_opcode) return ErrorRaised;

        int opcode_attrs, min_stack, stack_eff;
        Result<SizeOpcode> opcode_res = size_opcode_parse(
                py_opcode, &opcode_attrs, &min_stack, &stack_eff);
        if (!opcode_res.is_ok()) return ErrorRaised;

        if (remaining_attrs < opcode_attrs)
            return raise(PyExc_ValueError,
                         "Invalid host program (at op #%zd): not enough attributes"
                         " for opcode %u (need %d, have %zd)",
                         i, static_cast<unsigned>(*opcode_res), opcode_attrs, remaining_attrs);
        remaining_attrs -= opcode_attrs;

        if (depth < min_stack)
            return raise(PyExc_ValueError, "Invalid host program: not enough values on stack");
        depth += stack_eff;
        if (depth > HostProgram::kMaxStackDepth)
            return raise(PyExc_ValueError, "Invalid host program: stack overflow");

        prog.opcodes.push_back(*opcode_res);
    }

    if (remaining_attrs != 0)
        return raise(PyExc_ValueError, "Invalid host program: too many attributes");
    if (depth != expected_results)
        return raise(PyExc_ValueError,
                     "Invalid host program: expected exactly %zu result(s) on stack at the end,"
                     " got %d", expected_results, depth);

    for (Py_ssize_t i = 0; i < num_attrs; ++i) {
        PyObject* py_attr = PyList_GetItem(attrs_pylist.get(), i);
        if (!py_attr) return ErrorRaised;

        prog.op_attrs.push_back(pylong_as<int64_t>(py_attr));
        if (PyErr_Occurred()) return ErrorRaised;
    }

    return prog;
}

#define FOREACH_INTEGER_CONSTANT(X) \
    X(CU_TENSOR_MAP_DATA_TYPE_UINT8) \
    X(CU_TENSOR_MAP_DATA_TYPE_UINT16) \
    X(CU_TENSOR_MAP_DATA_TYPE_UINT32) \
    X(CU_TENSOR_MAP_DATA_TYPE_INT32) \
    X(CU_TENSOR_MAP_DATA_TYPE_UINT64) \
    X(CU_TENSOR_MAP_DATA_TYPE_INT64) \
    X(CU_TENSOR_MAP_DATA_TYPE_FLOAT16) \
    X(CU_TENSOR_MAP_DATA_TYPE_FLOAT32) \
    X(CU_TENSOR_MAP_DATA_TYPE_FLOAT64) \
    X(CU_TENSOR_MAP_DATA_TYPE_BFLOAT16) \
    X(CU_TENSOR_MAP_DATA_TYPE_FLOAT32_FTZ) \
    X(CU_TENSOR_MAP_DATA_TYPE_TFLOAT32) \
    X(CU_TENSOR_MAP_DATA_TYPE_TFLOAT32_FTZ) \
    X(CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B) \
    X(CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B) \
    X(CU_TENSOR_MAP_DATA_TYPE_16U6_ALIGN16B) \
    X(CU_TENSOR_MAP_SWIZZLE_NONE) \
    X(CU_TENSOR_MAP_SWIZZLE_32B) \
    X(CU_TENSOR_MAP_SWIZZLE_64B) \
    X(CU_TENSOR_MAP_SWIZZLE_128B) \
    X(CU_TENSOR_MAP_SWIZZLE_128B_ATOM_32B) \
    X(CU_TENSOR_MAP_SWIZZLE_128B_ATOM_32B_FLIP_8B) \
    X(CU_TENSOR_MAP_SWIZZLE_128B_ATOM_64B) \
    X(CU_TENSOR_MAP_L2_PROMOTION_NONE) \
    X(CU_TENSOR_MAP_L2_PROMOTION_L2_64B) \
    X(CU_TENSOR_MAP_L2_PROMOTION_L2_128B) \
    X(CU_TENSOR_MAP_L2_PROMOTION_L2_256B)

#define INTEGER_CONSTANT_ENTRY(name) {name, #name},

static Status define_integer_constants(PyObject* m) {
    static const struct {int value; const char* name;} entries[] = {
        FOREACH_INTEGER_CONSTANT(INTEGER_CONSTANT_ENTRY)
    };
    for (size_t i = 0; i < std::size(entries); ++i) {
        PyPtr val = steal(PyLong_FromLong(entries[i].value));
        if (!val) return ErrorRaised;
        if (PyModule_AddObjectRef(m, entries[i].name, val.get()) < 0)
            return ErrorRaised;
    }
    return OK;
}

static Result<uint32_t> tensor_map_item_size(CUtensorMapDataType dtype) {
    switch (dtype) {
    case CU_TENSOR_MAP_DATA_TYPE_UINT8:
        return 1;
    case CU_TENSOR_MAP_DATA_TYPE_UINT16:
    case CU_TENSOR_MAP_DATA_TYPE_FLOAT16:
    case CU_TENSOR_MAP_DATA_TYPE_BFLOAT16:
        return 2;
    case CU_TENSOR_MAP_DATA_TYPE_UINT32:
    case CU_TENSOR_MAP_DATA_TYPE_INT32:
    case CU_TENSOR_MAP_DATA_TYPE_FLOAT32:
    case CU_TENSOR_MAP_DATA_TYPE_FLOAT32_FTZ:
    case CU_TENSOR_MAP_DATA_TYPE_TFLOAT32:
    case CU_TENSOR_MAP_DATA_TYPE_TFLOAT32_FTZ:
        return 4;
    case CU_TENSOR_MAP_DATA_TYPE_UINT64:
    case CU_TENSOR_MAP_DATA_TYPE_INT64:
    case CU_TENSOR_MAP_DATA_TYPE_FLOAT64:
        return 8;
    default:
        return raise(PyExc_ValueError, "Can't create tensor map: unsupported data type %d",
                     static_cast<int>(dtype));
    }
}


static Result<HoistedTensorMap> hoisted_tensor_map_parse(PyObject* map_pyobj) {
    HoistedTensorMap ret;

    // Data type & item size
    PyPtr py_data_type = getattr(map_pyobj, "data_type");
    if (!py_data_type) return ErrorRaised;
    ret.dtype = static_cast<CUtensorMapDataType>(pylong_as<long>(py_data_type));
    if (PyErr_Occurred()) return ErrorRaised;
    Result<uint32_t> item_size_res = tensor_map_item_size(ret.dtype);
    if (!item_size_res.is_ok()) return ErrorRaised;
    ret.item_size = *item_size_res;

    // Base ptr
    PyPtr py_base_ptr_param_idx = getattr(map_pyobj, "base_ptr_param");
    ret.base_ptr_param_idx = pylong_as<uint32_t>(py_base_ptr_param_idx);
    if (PyErr_Occurred()) return ErrorRaised;

    // Rank
    PyPtr py_rank = getattr(map_pyobj, "rank");
    if (!py_rank) return ErrorRaised;
    long rank = pylong_as<long>(py_rank);
    if (PyErr_Occurred()) return ErrorRaised;
    if (rank < 1)
        return raise(PyExc_ValueError, "Rank of HoistedTensorMap is too small");
    if (rank > HoistedTensorMap::kMaxRank)
        return raise(PyExc_ValueError, "Rank of HoistedTensorMap is too large");
    ret.rank = static_cast<uint32_t>(rank);

    // Shape/stride program
    PyPtr py_shape_stride_program = getattr(map_pyobj, "shape_stride_program");
    if (!py_shape_stride_program) return ErrorRaised;
    Result<HostProgram> prog_res = host_program_parse(py_shape_stride_program.get(), rank * 2);
    if (!prog_res.is_ok()) return ErrorRaised;
    ret.shape_stride_program = *prog_res;

    // Box dim & traversal steps
    PyPtr py_tile_shape = getattr(map_pyobj, "tile_shape");
    if (!py_tile_shape) return ErrorRaised;
    if (!PyTuple_Check(py_tile_shape.get()))
        return raise(PyExc_TypeError, "HoistedTensorMap.tile_shape is not a tuple");
    if (PyTuple_GET_SIZE(py_tile_shape.get()) != rank)
        return raise(PyExc_TypeError, "Size of HoistedTensorMap.tile_shape doesn't match rank");
    for (long i = 0; i < rank; ++i) {
        PyObject* py_dim = PyTuple_GET_ITEM(py_tile_shape.get(), i);
        ret.traversal_steps[i] = 1;
        ret.box_dim[i] = pylong_as<uint32_t>(py_dim);
        if (PyErr_Occurred()) return ErrorRaised;
    }

    // Swizzle
    PyPtr py_swizzle = getattr(map_pyobj, "swizzle");
    if (!py_swizzle) return ErrorRaised;
    PyPtr py_swizzle_val = getattr(py_swizzle, "_value_");
    if (!py_swizzle_val) return ErrorRaised;
    ret.swizzle = static_cast<CUtensorMapSwizzle>(pylong_as<long>(py_swizzle_val));
    if (PyErr_Occurred()) return ErrorRaised;

    // L2 promotion
    PyPtr py_l2_promotion = getattr(map_pyobj, "l2_promotion");
    if (!py_l2_promotion) return ErrorRaised;
    PyPtr py_l2_promotion_val = getattr(py_l2_promotion, "_value_");
    if (!py_l2_promotion_val) return ErrorRaised;
    ret.l2_promotion = static_cast<CUtensorMapL2promotion>(
            pylong_as<long>(py_l2_promotion_val));
    if (PyErr_Occurred()) return ErrorRaised;

    ret.interleave = CU_TENSOR_MAP_INTERLEAVE_NONE;
    ret.oob_fill = CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE;
    return ret;
}

static Status hoisted_tensor_map_encode(const DriverApi& driver,
                                        const Vec<HoistedTensorMap>& maps,
                                        LaunchHelper& helper) {
    if (maps.empty()) return OK;

    for (const HoistedTensorMap& m : maps) {
        arena_pad_to_alignment<alignof(CUtensorMap)>(helper.arena);
        ArenaOffset tensor_map_offset = arena_alloc_words(
                helper.arena, sizeof(CUtensorMap) / sizeof(Word));
        void* storage = static_cast<void*>(helper.arena.data() + tensor_map_offset);
        CUtensorMap* dst = new (storage) CUtensorMap();

        int64_t stack[HostProgram::kMaxStackDepth];
        host_program_eval(m.shape_stride_program, helper.arena, helper.cuarg_offsets, stack);

        uint32_t rank = m.rank;
        uint64_t global_dim[HoistedTensorMap::kMaxRank];
        mem_copy(global_dim, stack, rank * sizeof(global_dim[0]));

        int64_t stride0 = stack[rank];
        if (stride0 != 1) {
            return raise(PyExc_ValueError,
                    "Can't create a tensor map: stride of last array dimension must be 1, got %lld",
                    static_cast<long long>(stride0));
        }

        uint64_t global_strides[HoistedTensorMap::kMaxRank - 1];
        for (uint32_t i = 1; i < rank; ++i) {
            int64_t s = stack[rank + i];
            if (s < 0)
                return raise(PyExc_ValueError,
                        "Can't create a tensor map: strides must be positive, got %lld",
                        static_cast<long long>(s));
            uint64_t u = s;
            uint64_t bytes = u * m.item_size;
            if (bytes / m.item_size != u)
                return raise(PyExc_OverflowError,
                        "Can't create a tensor map: stride %lld is too big",
                        static_cast<long long>(s));
            global_strides[i - 1] = static_cast<uint32_t>(bytes);
        }

        CUresult res = driver.cuTensorMapEncodeTiled(
            dst,
            m.dtype,
            rank,
            helper.arena[helper.cuarg_offsets[m.base_ptr_param_idx]].device_ptr,
            global_dim,
            global_strides,
            m.box_dim,
            m.traversal_steps,
            m.interleave,
            m.swizzle,
            m.l2_promotion,
            m.oob_fill
        );
        if (res != CUDA_SUCCESS)
            return raise(PyExc_RuntimeError, "Failed to encode tiled tensor map: %s",
                         get_cuda_error(&driver, res));

        helper.cuarg_offsets.push_back(tensor_map_offset);
    }
    return OK;
}


namespace { struct TileDispatcher {
    Vec<RefPtr<ParameterAnnotationNode>> param_annotations;
    TileContextDispatcher default_context_dispatcher;

    static PyTypeObject pytype;
}; }


static Result<TileKernel> compile(const DriverApi* driver,
                                  PyObject* dispatcher_pyobj,
                                  PyObject* signature,
                                  PyObject* py_tile_context,
                                  KernelImage* image,
                                  PyObject* py_compute_capability) {
    PyPtr compile_result = steal(PyObject_CallMethod(
            dispatcher_pyobj, "_compile", "(OOO)",
            signature, py_tile_context, py_compute_capability));
    if (!compile_result) return ErrorRaised;

    if (!PyTuple_Check(compile_result.get()))
        return raise(PyExc_TypeError, "Expected compile() to return a tuple, got %s",
                     Py_TYPE(compile_result.get())->tp_name);

    if (PyTuple_GET_SIZE(compile_result.get()) != 4)
        return raise(PyExc_TypeError, "Expected compile() to return a 4-tuple, got length %zd",
                     PyTuple_GET_SIZE(compile_result.get()));

    PyObject* py_cubin_bytes = PyTuple_GET_ITEM(compile_result.get(), 0);
    PyObject* py_cufunc_name = PyTuple_GET_ITEM(compile_result.get(), 1);
    PyObject* py_dyn_smem_size_prog = PyTuple_GET_ITEM(compile_result.get(), 2);
    PyObject* py_hoisted_tensor_maps = PyTuple_GET_ITEM(compile_result.get(), 3);

    if (!PyBytes_Check(py_cubin_bytes)
            || !PyUnicode_Check(py_cufunc_name)
            || (py_hoisted_tensor_maps != Py_None && !PyList_Check(py_hoisted_tensor_maps))) {
        return raise(PyExc_TypeError,
                     "Expected compile() to return (bytes, str, HostProgram|None, list|None),"
                     " got %s, %s",
                     Py_TYPE(py_cubin_bytes)->tp_name,
                     Py_TYPE(py_cufunc_name)->tp_name);
    }

    char* cubin_data;
    Py_ssize_t cubin_size;
    if (PyBytes_AsStringAndSize(py_cubin_bytes, &cubin_data, &cubin_size) < 0)
        return ErrorRaised;

    Py_ssize_t py_cufunc_name_size;
    const char* cufunc_name = PyUnicode_AsUTF8AndSize(py_cufunc_name, &py_cufunc_name_size);
    if (!cufunc_name) return ErrorRaised;

    Result<CudaKernel> cukernel = load_cuda_kernel(driver, cubin_data, cubin_size, cufunc_name);
    if (!cukernel.is_ok()) return ErrorRaised;

    if (py_dyn_smem_size_prog != Py_None) {
        Status status = enable_maximum_dynamic_shared_memory(
            driver, cukernel->kernel, cufunc_name);
        if (!status) return ErrorRaised;
    }

    Result<HostProgram> dyn_smem_size_prog = host_program_parse(py_dyn_smem_size_prog, 1);
    if (!dyn_smem_size_prog.is_ok()) return ErrorRaised;

    Py_ssize_t num_hoisted_tensor_maps = PyList_Size(py_hoisted_tensor_maps);
    Vec<HoistedTensorMap> hoisted_tensor_maps;
    hoisted_tensor_maps.reserve(num_hoisted_tensor_maps);
    for (Py_ssize_t i = 0; i < num_hoisted_tensor_maps; ++i) {
        PyObject* map_pyobj = PyList_GetItem(py_hoisted_tensor_maps, i);
        if (!map_pyobj) return ErrorRaised;
        Result<HoistedTensorMap> map_res = hoisted_tensor_map_parse(map_pyobj);
        if (!map_res.is_ok()) return ErrorRaised;
        hoisted_tensor_maps.push_back(*map_res);
    }

    if (image) {
        image->cubin = newref(py_cubin_bytes);
        image->symbol = newref(py_cufunc_name);
    }

    return TileKernel{std::move(*cukernel),
                      std::move(*dyn_smem_size_prog),
                      std::move(hoisted_tensor_maps)};
}

enum class StreamKind {
    Error = 0,
    Torch,
    Cupy,
    NumbaCuda,
    RawInt,
};

static StreamKind do_classify_stream_type(PyTypeObject* ty) {
    if (is_torch_cuda_stream_subtype(ty)) {
        return StreamKind::Torch;
    } else if (is_cupy_cuda_stream_subtype(ty)) {
        return StreamKind::Cupy;
    } else if (is_numba_cuda_driver_stream_subtype(ty)) {
        return StreamKind::NumbaCuda;
    } else if (PyType_IsSubtype(ty, &PyLong_Type)) {
        return StreamKind::RawInt;
    } else if (ty == Py_TYPE(Py_None)) {
        raise(PyExc_TypeError, "Stream is required, got None");
        return StreamKind::Error;
    } else {
        // TODO: support more stream types, for example, cuda.core.experimental._stream.Stream
        raise(PyExc_TypeError, "Unsupported stream type %s.", ty->tp_name);
        return StreamKind::Error;
    }
}

// Must be holding GIL or g_launch_mutex to call this
static StreamKind classify_stream(PyObject* py_stream) {
    // Cache the last stream type we were called with.
    // The hypothesis is that the user is probably using one host framework to make all the
    // launches, so we will nearly always get a cache hit here. And comparing a single pointer
    // is much faster than performing a hash map lookup.
    // The cache miss path is reasonably efficient as well, so it's not the end of the world
    // if our guess is wrong. If necessary, we could add another level of caching here.
    static PyTypeObject* last_ty = nullptr;
    static StreamKind last_kind = StreamKind::Error;

    PyTypeObject* ty = Py_TYPE(py_stream);
    if (ty == last_ty) return last_kind;

    StreamKind res = do_classify_stream_type(ty);
    if (res != StreamKind::Error) {
        Py_XDECREF(last_ty);
        last_ty = reinterpret_cast<PyTypeObject*>(Py_NewRef(ty));
        last_kind = res;
    }
    return res;
}


static Result<CUstream> parse_stream(PyObject* py_stream) {
    auto from_raw = [] (PyObject* raw) -> Result<CUstream> {
        if (!raw) return ErrorRaised;
        CUstream stream = static_cast<CUstream>(PyLong_AsVoidPtr(raw));
        if (PyErr_Occurred()) {
            if (!PyLong_Check(raw))
                raise(PyExc_TypeError, "Raw stream pointer must be a long, got %s",
                      Py_TYPE(raw)->tp_name);
            return ErrorRaised;
        }
        return stream;
    };

    StreamKind kind = classify_stream(py_stream);
    switch (kind) {
    case StreamKind::Error:
        return ErrorRaised;
    case StreamKind::Torch:
        return from_raw(getattr(py_stream, g_cuda_stream_pyunicode).get());
    case StreamKind::Cupy:
        return from_raw(getattr(py_stream, g_ptr_pyunicode).get());
    case StreamKind::NumbaCuda:
        {
            PyPtr py_stream_handle = getattr(py_stream, "handle");
            if (!py_stream_handle) return ErrorRaised;

            // numba-cuda >= 0.30: handle is cuda.bindings.driver.CUstream
            // numba-cuda < 0.30: handle is ctypes c_void_p
            if (is_cuda_bindings_driver_custream_subtype(Py_TYPE(py_stream_handle.get()))) {
                PyPtr pylong = steal(PyNumber_Long(py_stream_handle.get()));
                return from_raw(pylong.get());
            } else {
                PyPtr py_stream_handle_value = getattr(py_stream_handle, "value");
                if (!py_stream_handle_value) return ErrorRaised;

                // numba stream.handle.value is None for default stream
                if (py_stream_handle_value.get() == Py_None)
                    return static_cast<CUstream>(nullptr);

                return from_raw(py_stream_handle_value.get());
            }
        }
        break;
    case StreamKind::RawInt:
        return from_raw(py_stream);
    }
    CHECK_UNREACHABLE;
}


using StreamBufferPoolMap = HashMap<unsigned long long, StreamBufferPool*>;

// Protected by GIL or g_launch_mutex.
// We have no reliable way to detect when a context is destroyed, so we never clean these up.
static StreamBufferPoolMap* g_stream_buffer_pool_by_ctx_id;


static Result<StreamBufferPool*> get_stream_buffer_pool(const DriverApi* driver, CUcontext ctx) {
    if (!ctx) {
        CUresult res = driver->cuCtxGetCurrent(&ctx);
        if (res != CUDA_SUCCESS) {
            return raise(PyExc_RuntimeError, "Failed to get current CUDA context: %s",
                         get_cuda_error(driver, res));
        }
    }

    unsigned long long ctx_id = 0;
    CUresult res = driver->cuCtxGetId(ctx, &ctx_id);
    if (res != CUDA_SUCCESS)
        return raise(PyExc_RuntimeError,
                     "Failed to get CUDA context ID: %s", get_cuda_error(driver, res));

    StreamBufferPoolMap::Item* item = g_stream_buffer_pool_by_ctx_id->find(ctx_id);
    if (item) {
        return item->value;
    } else {
        StreamBufferPool* pool = stream_buffer_pool_new();
        g_stream_buffer_pool_by_ctx_id->insert(ctx_id, pool);
        return pool;
    }
}

struct Grid {
    enum { Len = 3 };
    unsigned dims[Len];
};

static bool validate_grid(const Grid& grid) {
    constexpr unsigned kMaxGridDim = (1 << 24) - 1;
    for (int i = 0; i < Grid::Len; ++i) {
        // Restrict grid dims to 2^24 due to an OCG bug.
        // Larger dimensions may result in incorrect tile block ID calculations.
        if (grid.dims[i] > kMaxGridDim) {
            raise(
                PyExc_ValueError,
                "Grid[%d] exceeds 24-bit limit: max=%d, got=%lu. "
                "Use multiple kernel launches for larger workloads.",
                i, kMaxGridDim, grid.dims[i]);
            return false;
        }
    }
    return true;
}

static bool try_clarify_invalid_value_error(const DriverApi* driver, const Grid& grid) {
    CUdevice dev;
    if (driver->cuCtxGetDevice(&dev) != CUDA_SUCCESS) return false;

    for (int i = 0; i < Grid::Len; ++i) {
        int v;
        CUdevice_attribute attr = static_cast<CUdevice_attribute>(
            CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X + i
        );
        if (driver->cuDeviceGetAttribute(&v, attr, dev) != CUDA_SUCCESS) return false;

        if (grid.dims[i] > static_cast<unsigned>(v)) {
            raise(PyExc_ValueError, "Grid[%d] is too big: max=%d, got=%lu",
                  i, v, grid.dims[i]);
            return true;
        }
    }
    return false;
}

struct PreparedLaunch {
    LaunchHelperPtr helper;
    CUkernel kernel;
    unsigned dynamic_smem_bytes;
    TileKernel* tile_kernel;
    std::optional<KernelImage> kernel_image;
};

static Result<CUcontext> get_stream_context(const DriverApi* driver, CUstream stream) {
    CUcontext ctx = nullptr;
    CUresult res = driver->cuStreamGetCtx(stream, &ctx);
    // INVALID_CONTEXT can happen when it is NULL stream and there is
    // no active context in current thread. We will still get the context
    // from the array arguments later during `extract_cuda_args`.
    if (res != CUDA_SUCCESS && res != CUDA_ERROR_INVALID_CONTEXT) {
        return raise(PyExc_RuntimeError, "Failed to get a CUDA context from a stream: %s",
                     get_cuda_error(driver, res));
    }
    return ctx;
}

static Status stage_list_args_on_stream(const DriverApi* driver,
                                        CUstream launch_stream,
                                        CUcontext cuda_context,
                                        Arena& arena,
                                        const Vec<ListArg>& list_args,
                                        size_t total_list_data_size_words,
                                        StreamBufferTransaction& tx) {
    if (list_args.empty())
        return OK;

    if (!tx) {
        CUstreamCaptureStatus status;
        CUresult res = driver->cuStreamIsCapturing(launch_stream, &status);
        if (res != CUDA_SUCCESS)
            return raise(PyExc_RuntimeError, "Failed to check stream capturing status: %s",
                    get_cuda_error(driver, res));
        if (status != CU_STREAM_CAPTURE_STATUS_NONE)
            return raise(PyExc_RuntimeError, "List argument in CUDAGraph isn't supported yet");

        Result<StreamBufferPool*> pool_res = get_stream_buffer_pool(driver, cuda_context);
        if (!pool_res.is_ok()) return ErrorRaised;

        tx = stream_buffer_transaction_open(driver, *pool_res, launch_stream);
        if (!tx) return raise(PyExc_RuntimeError, "Failed to open a stream buffer transaction");
    }

    size_t size = total_list_data_size_words * sizeof(Word);
    DualPointer ptr = tx.allocate(size);
    if (!ptr)
        return raise(PyExc_RuntimeError, "Failed to allocate memory in stream buffer");

    for (const ListArg& list_arg : list_args) {
        size_t data_offset_words = arena[list_arg.base_ptr_cuarg].size;
        arena[list_arg.base_ptr_cuarg].device_ptr = reinterpret_cast<void*>(
                ptr.device + data_offset_words * sizeof(Word));
        Word* dst = reinterpret_cast<Word*>(ptr.host) + data_offset_words;
        size_t item_size_words = list_arg.item_size_words;
        size_t item_size_bytes = item_size_words * sizeof(Word);
        for (size_t i = 0; i < list_arg.length; ++i) {
            ArenaOffset item_offset = arena[list_arg.item_offsets + i].arena_offset;
            mem_copy(dst, arena.data() + item_offset, item_size_bytes);
            dst += item_size_words;
        }
    }

    CUresult res = driver->cuMemcpyHtoDAsync(ptr.device, ptr.host, size, launch_stream);
    if (res != CUDA_SUCCESS) {
        return raise(PyExc_RuntimeError, "Failed to copy memory from host to device: %s",
                     get_cuda_error(driver, res));
    }

    return OK;
}

#ifdef ENABLE_CCONV_V3
static bool is_dataclass(PyTypeObject* ty) {
    return PyObject_HasAttr(reinterpret_cast<PyObject*>(ty),
                            g___dataclass_fields___pyunicode);
}
#endif

static Result<std::optional<AggregateArgType>> classify_aggregate_type(PyTypeObject* ty) {
    if (ty == nullptr) {
        return {std::nullopt};
    } else if (ty == &PyTuple_Type) {
        return {{{ AggregateArgType::Tuple, {} }}};
#ifdef ENABLE_CCONV_V3
    } else if (is_dataclass(ty)) {
        RefPtr<DataclassInfo> info = get_dataclass_info(ty);
        if (!info) return ErrorRaised;
        return {{{ AggregateArgType::Dataclass, std::move(info) }}};
#endif
    } else {
        return {std::nullopt};
    }
}

static ErrorRaised_t raise_invalid_kernel_arg_type_impl(
        const Vec<PyTypeObject*>& pyarg_types_depth_first,
        const Vec<std::optional<AggregateArgType>>& agg_types,
        size_t culprit_leaf_idx,
        PyObject* message) {
    CHECK(pyarg_types_depth_first.size() == agg_types.size());

    // Reconstruct the path to the leaf so that we can make a decent error message
    struct PathItem {
        size_t depth_first_idx;
        size_t item_idx;
    };
    Vec<PathItem> path = {{SIZE_MAX, SIZE_MAX}};
    size_t next_leaf_idx = 0;
    for (size_t depth_first_idx = 0; depth_first_idx < agg_types.size(); ++depth_first_idx) {
        ++path.back().item_idx;
        if (agg_types[depth_first_idx].has_value()) {
            path.push_back(PathItem{depth_first_idx, SIZE_MAX});
        } else if (pyarg_types_depth_first[depth_first_idx] == nullptr) {
            path.pop_back();
            CHECK(!path.empty());
        } else {
            if (next_leaf_idx == culprit_leaf_idx)
                break;
            ++next_leaf_idx;
        }
    }
    CHECK(next_leaf_idx == culprit_leaf_idx);
    CHECK(!path.empty());
    CHECK(path[0].depth_first_idx == SIZE_MAX);

    // Walk the path backwards and build the message
    PyPtr ret = steal(PyUnicode_FromString("Invalid "));
    if (!ret) return {};

    for (size_t i = path.size() - 1; i > 0; --i) {
        const std::optional<AggregateArgType>& agg_type = agg_types[path[i].depth_first_idx];
        CHECK(agg_type.has_value());

        PyPtr new_str;
        switch (agg_type->kind) {
        case AggregateArgType::Tuple:
            new_str = steal(PyUnicode_FromFormat("%Uitem #%zu of ", ret.get(), path[i].item_idx));
            break;
#ifdef ENABLE_CCONV_V3
        case AggregateArgType::Dataclass:
            CHECK(path[i].item_idx < agg_type->dataclass_info->field_names.size());
            new_str = steal(PyUnicode_FromFormat("%Ufield '%U' of ", ret.get(),
                    agg_type->dataclass_info->field_names[path[i].item_idx].get()));
            break;
#endif
        }
        if (!new_str) return {};
        ret = std::move(new_str);
    }
    return raise(PyExc_TypeError,
            "%Ukernel argument #%zu: %U", ret.get(), path[0].item_idx, message);
}

template <typename... Args>
ErrorRaised_t raise_invalid_kernel_arg_type(const Vec<PyTypeObject*>& pyarg_types_depth_first,
                                            const Vec<std::optional<AggregateArgType>>& agg_types,
                                            size_t culprit_leaf_idx,
                                            const char* fmt, Args&&... args) {
    PyPtr msg = steal(PyUnicode_FromFormat(fmt, std::forward<Args>(args)...));
    if (!msg) return ErrorRaised;

    return raise_invalid_kernel_arg_type_impl(
            pyarg_types_depth_first, agg_types, culprit_leaf_idx, msg.get());
}

static Result<Vec<PythonArgKind>>
get_pyarg_kinds(const Vec<PyTypeObject*>& pyarg_types_depth_first,
                const Vec<std::optional<AggregateArgType>>& agg_types,
                const Vec<PyObject*>& leaf_pyarg_objs,
                const Vec<RefPtr<LeafAnnotationNode>>& flat_param_annotations) {
    size_t n = leaf_pyarg_objs.size();
    CHECK(flat_param_annotations.size() == n);
    Vec<PythonArgKind> ret;
    ret.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        PyObject* obj = leaf_pyarg_objs[i];
        if (flat_param_annotations[i]->constant) {
            std::optional<ConstantKind> kind = classify_constant(obj, true);
            if (!kind.has_value()) {
                return raise_invalid_kernel_arg_type(
                        pyarg_types_depth_first, agg_types, i,
                        "Could not interpret object of type '%s' as a constant.",
                        Py_TYPE(obj)->tp_name);
            }
            ret.push_back(constant_kind_as_arg_kind(*kind));
        } else {
            std::optional<PythonArgKind> kind = classify_nonconstant_arg(obj);
            if (!kind.has_value()) {
                if (PyType_IsSubtype(Py_TYPE(obj), &PyTuple_Type)) {
                    return raise_invalid_kernel_arg_type(
                            pyarg_types_depth_first, agg_types, i,
                            "'%s' is a subclass of 'tuple'. Only plain tuples are accepted.",
                            Py_TYPE(obj)->tp_name);
                } else {
                    return raise_invalid_kernel_arg_type(
                            pyarg_types_depth_first, agg_types, i,
                            "Objects of type '%s' are not supported as non-constant arguments.",
                            Py_TYPE(obj)->tp_name);
                }
            }
            ret.push_back(*kind);
        }
    }
    return ret;
}

static Vec<ParameterKind>
get_parameter_kinds(const Vec<PyTypeObject*>& pyarg_types_depth_first,
                    const Vec<std::optional<AggregateArgType>>& agg_types,
                    const Vec<PythonArgKind>& pyarg_kinds) {
    Vec<ParameterKind> ret;
    ret.reserve(pyarg_types_depth_first.size());
    size_t leaf_idx = 0;
    for (size_t depth_first_idx = 0;
            depth_first_idx < pyarg_types_depth_first.size(); ++depth_first_idx) {
        if (pyarg_types_depth_first[depth_first_idx] == nullptr) {
            ret.push_back({ParameterKind::AggregateEnd, {}});
            continue;
        }
        const std::optional<AggregateArgType>& agg_ty = agg_types[depth_first_idx];
        if (agg_ty.has_value()) {
            ret.push_back({ParameterKind::AggregateBegin, *agg_ty});
        } else {
            PythonArgKind arg_kind = pyarg_kinds[leaf_idx++];
            ret.push_back({param_category_from_pyarg_kind(arg_kind), {}});
        }
    }
    CHECK(leaf_idx == pyarg_kinds.size());
    return ret;
}

static KernelFamily* get_or_create_kernel_family(Vec<RefPtr<KernelFamily>>* families,
                                                 Vec<ParameterKind>&& param_kinds) {
    for (const RefPtr<KernelFamily>& fam : *families) {
        if (fam->param_kinds == param_kinds)
            return fam.get();
    }
    families->push_back(steal(new KernelFamily(std::move(param_kinds))));
    return families->back().get();
}

static void get_pyarg_objects_and_types(PyObject* const* objects, Py_ssize_t num_objects,
                                        Vec<PyObject*>* pyarg_objs,
                                        Vec<PyTypeObject*>* pyarg_types) {
    for (Py_ssize_t i = 0; i < num_objects; ++i) {
        pyarg_objs->push_back(objects[i]);
        pyarg_types->push_back(Py_TYPE(objects[i]));
    }
    // Push a sentinel to separate items of different aggregates
    pyarg_objs->push_back(nullptr);
    pyarg_types->push_back(nullptr);
}

static const AggregateArgType* get_aggregate_arg_type(ExpandAggregates* ea,
                                                      size_t breadth_first_idx) {
    // Could in theory use binary search, but the array is most likely very small,
    // and this is the slow path anyway.
    for (const AggregateArgInfo& info : ea->aggregate_args) {
        if (info.breadth_first_index == breadth_first_idx)
            return &info.type;
    }
    return nullptr;
}

// As we walk the ProfileMap in `python_arg_profile_lookup_impl()`, we store the Python arguments
// and their types in the breadth-first order, in the arrays called `pyarg_objs_breadth_first`
// and `pyarg_types_breadth_first`, respectively. For example, suppose we have 3 kernel arguments:
//
//        (a, b), c, (d, (e, f))   [where a...f are, say, arrays;  t1,t2,t3 are tuples]
//        ^          ^   ^
//        `~~~t1     |   `~~~t3
//                   `~~~~~~~~t2
//
// Then the breadth-first order is
//
//     0   1  2   3     4  5  6     7  8   9     10 11 12
//     t1, c, t2, null, a, b, null, d, t3, null, e, f, null
//     `~~~~~~~~~~~~~'  `---------------------'  `--------'
//         depth=0              depth=1            depth=2
//
// `null`s are sentinels that are inserted at the end of each tuple, as well as the end
// of the top-level argument list.
//
// For further processing of the arguments, it is more convenient to rearrange them in
// the depth-first order, which matches the order of the parenthesised notation, e.g.
// (a, b), c, (d, (e, f)). This function does precisely that. For the example above,
// the depth-first order will be
//
//     (         )        (      (         )     )
//     t1, a, b, null, c, t2, d, t3, e, f, null, null, null
//               ^                         ^     ^      ^-----end of args
//               `end of t1                |     `end of t2
//                                       end of t3
//
// Tuples t1, t2, t3 take positions of left parentheses, and null sentinels take positions
// of right parentheses.
//
// In the process, we also build an array `leaf_pyarg_breadth_first_indices` which contains
// a partial permutation that enables us to quickly extract leaf arguments in the depth-first
// order from the `pyarg_objs_breadth_first` array in the hot path (see `gather_leaf_pyargs()`).
// For our example, it will be:
//
//     4  5  1  7  10  11
//    (a  b  c  d  e   f)
//
static void pyargs_breadth2depth(int depth,
                                 size_t leaf_size,
                                 ExpandAggregates* parent,
                                 const Vec<PyTypeObject*>& pyarg_types_breadth_first,
                                 Vec<PyTypeObject*>* pyarg_types_depth_first,
                                 Vec<size_t>* leaf_pyarg_breadth_first_indices,
                                 Vec<std::optional<AggregateArgType>>* aggregate_types) {
    // Reconstruct the path to the `profile` by walking the parent links
    Vec<ExpandAggregates*> path(depth);
    for (int i = depth - 1; i >= 0; --i) {
        CHECK(parent);
        path[i] = parent;
        parent = parent->parent;
    }

    // For each node in the path (including the leaf),
    // calculate the offset into `pyarg_types_breadth_first`.
    Vec<size_t> offsets;
    Vec<size_t> end_offsets;
    offsets.reserve(depth + 1);
    size_t cumul_offset = 0;
    size_t total_aggregates = 0;
    for (ExpandAggregates* ea : path) {
        offsets.push_back(cumul_offset);
        cumul_offset += ea->arg_types.size() + 1;  // +1 for the sentinel
        end_offsets.push_back(cumul_offset);
        total_aggregates += ea->aggregate_args.size();
    }
    offsets.push_back(cumul_offset);
    end_offsets.push_back(cumul_offset + leaf_size + 1);

    pyarg_types_depth_first->reserve(pyarg_types_breadth_first.size());

    // Calculate number of leaves in order to pre-allocate `leaf_pyarg_breadth_first_indices`.
    size_t total_aggregates_and_sentinels = total_aggregates * 2 + 1;
    CHECK(total_aggregates_and_sentinels <= pyarg_types_breadth_first.size());
    size_t total_leaf_args = pyarg_types_breadth_first.size() - total_aggregates_and_sentinels;
    leaf_pyarg_breadth_first_indices->reserve(total_leaf_args);

    aggregate_types->reserve(pyarg_types_breadth_first.size());

    // Do a depth-first traversal of the tree.
    size_t cur_depth = 0;
    while (true) {
        CHECK(cur_depth < offsets.size());
        size_t idx = offsets[cur_depth]++;
        CHECK(idx < end_offsets[cur_depth]);
        CHECK(idx < pyarg_types_breadth_first.size());
        pyarg_types_depth_first->push_back(pyarg_types_breadth_first[idx]);
        const AggregateArgType* agg_type;
        if (!pyarg_types_breadth_first[idx]) {
            // Reached a sentinel? Go back up a level.
            aggregate_types->push_back({});
            if (cur_depth-- == 0) break;
        } else if (cur_depth < path.size()
                && (agg_type = get_aggregate_arg_type(path[cur_depth], idx)) != nullptr) {
            // Found an aggregate? Go a level deeper.
            ++cur_depth;
            aggregate_types->push_back(*agg_type);
        } else {
            // Else this is a leaf (non-aggregate) argument: record its source index.
            leaf_pyarg_breadth_first_indices->push_back(idx);
            aggregate_types->push_back({});
        }
    }

    CHECK(pyarg_types_depth_first->size() == pyarg_types_breadth_first.size());
    CHECK(aggregate_types->size() == pyarg_types_breadth_first.size());
    CHECK(leaf_pyarg_breadth_first_indices->size() == total_leaf_args);
    CHECK(offsets == end_offsets);
}

// Apply the partial permutation `leaf_pyarg_breadth_first_indices` to get an array
// of leaf Python arguments from `pyarg_objs_breadth_first`.
static void gather_leaf_pyargs(const Vec<PyObject*>& pyarg_objs_breadth_first,
                               const Vec<size_t>& leaf_pyarg_breadth_first_indices,
                               Vec<PyObject*>* leaf_pyarg_objs) {
    leaf_pyarg_objs->clear();
    for (size_t i : leaf_pyarg_breadth_first_indices)
        leaf_pyarg_objs->push_back(pyarg_objs_breadth_first[i]);
}

#ifdef ENABLE_CCONV_V3
static Status expand_dataclass_instance(PyObject* arg, const DataclassInfo& info,
                                        Vec<PyObject*>* pyarg_objs_breadth_first,
                                        Vec<PyTypeObject*>* pyarg_types_breadth_first,
                                        Vec<PyPtr>* pyarg_refs) {
    for (const PyPtr& name : info.field_names) {
        PyPtr value = steal(PyObject_GetAttr(arg, name.get()));
        if (!value) return ErrorRaised;
        pyarg_objs_breadth_first->push_back(value.get());
        pyarg_types_breadth_first->push_back(Py_TYPE(value.get()));
        pyarg_refs->push_back(std::move(value));
    }
    pyarg_objs_breadth_first->push_back(nullptr);
    pyarg_types_breadth_first->push_back(nullptr);
    return OK;
}
#endif

static Status expand_aggregate_arg(
        PyObject* arg,
        const AggregateArgType& agg_type,
        Vec<PyObject*>* pyarg_objs_breadth_first,
        Vec<PyTypeObject*>* pyarg_types_breadth_first,
        Vec<PyPtr>* pyarg_refs) {
    switch (agg_type.kind) {
    case AggregateArgType::Tuple:
        CHECK(PyTuple_CheckExact(arg));
        get_pyarg_objects_and_types(reinterpret_cast<PyTupleObject*>(arg)->ob_item,
                                    PyTuple_GET_SIZE(arg),
                                    pyarg_objs_breadth_first,
                                    pyarg_types_breadth_first);
        return OK;
#ifdef ENABLE_CCONV_V3
    case AggregateArgType::Dataclass:
        return expand_dataclass_instance(arg, *agg_type.dataclass_info,
                pyarg_objs_breadth_first, pyarg_types_breadth_first, pyarg_refs);
#endif
    }
    CHECK(false);
}

static PythonArgProfile* python_arg_profile_lookup_impl(
        ProfileMap* map,
        PyObject* const* pyargs,
        Py_ssize_t num_pyargs,
        const Vec<RefPtr<ParameterAnnotationNode>>& param_annotations,
        Vec<RefPtr<KernelFamily>>* kernel_families,
        Vec<PyObject*>* pyarg_objs_breadth_first,
        Vec<PyTypeObject*>* pyarg_types_breadth_first,
        Vec<PyObject*>* leaf_pyarg_objs,
        Vec<PyPtr>* pyarg_refs) {
    ProfileMapQuery query = {pyarg_types_breadth_first, 0, 0};
    query.mark_start();
    get_pyarg_objects_and_types(pyargs, num_pyargs,
                                pyarg_objs_breadth_first, pyarg_types_breadth_first);
    query.mark_end();
    ExpandAggregates* parent = nullptr;

    constexpr int kMaxAggregateNestingDepth = 64;
    for (int depth = 0; depth <= kMaxAggregateNestingDepth; ++depth) {
        ProfileMap::Item* item = map->find(query);
        ExpandAggregates* next = nullptr;
        if (item) {
            ProfileMapNode* node = item->key.node.get();
            if (node->leaf) {
                // Fastest path possible: at a leaf node (no need to expand aggregates)
                PythonArgProfile* profile = static_cast<PythonArgProfile*>(node);
                gather_leaf_pyargs(*pyarg_objs_breadth_first,
                                   profile->leaf_pyarg_breadth_first_indices, leaf_pyarg_objs);
                return profile;
            }
            // Still fast: expand a known aggregate type such as a tuple and do another lookup.
            next = static_cast<ExpandAggregates*>(node);
        } else {
            // Slower path: allocate a new ProfileMapNode.

            if (depth == 0 && (size_t)num_pyargs != param_annotations.size()) {
                raise(PyExc_TypeError, "Kernel expects %zu %s but %zd %s given",
                        param_annotations.size(),
                        param_annotations.size() == 1 ? "argument" : "arguments",
                        num_pyargs,
                        num_pyargs == 1 ? "was" : "were");
                return nullptr;
            }

            // Determine which arguments are aggregate.
            Vec<AggregateArgInfo> aggregate_args;
            for (size_t i = 0; i < query.size(); ++i) {
                PyTypeObject* ty = query[i];
                Result<std::optional<AggregateArgType>> aggty = classify_aggregate_type(ty);
                if (!aggty.is_ok()) return nullptr;

                if (*aggty)
                    aggregate_args.push_back({query.offset + i, **aggty});
            }

            if (aggregate_args.empty()) {
                // Need to create a new leaf node (i.e., PythonArgProfile).

                // Transform the breadth-first order of args into depth-first.
                Vec<PyTypeObject*> pyarg_types_depth_first;
                Vec<size_t> leaf_pyarg_breadth_first_indices;
                Vec<std::optional<AggregateArgType>> aggregate_types;
                pyargs_breadth2depth(depth, query.size(), parent, *pyarg_types_breadth_first,
                                     &pyarg_types_depth_first, &leaf_pyarg_breadth_first_indices,
                                     &aggregate_types);
                gather_leaf_pyargs(*pyarg_objs_breadth_first, leaf_pyarg_breadth_first_indices,
                                   leaf_pyarg_objs);

                // Flatten the parameter annotations against this argument structure.
                Result<Vec<RefPtr<LeafAnnotationNode>>> flat_param_annotations
                       = flatten_parameter_annotation_nodes(
                           param_annotations, pyarg_types_depth_first, aggregate_types,
                           leaf_pyarg_objs->size());
                if (!flat_param_annotations.is_ok())
                    return nullptr;

                // Classify the arguments and get the matching KernelFamily.
                Result<Vec<PythonArgKind>> arg_kinds = get_pyarg_kinds(
                        pyarg_types_depth_first, aggregate_types,
                        *leaf_pyarg_objs, *flat_param_annotations);
                if (!arg_kinds.is_ok()) return nullptr;

                Vec<ParameterKind> param_kinds = get_parameter_kinds(
                        pyarg_types_depth_first, aggregate_types, *arg_kinds);

                KernelFamily* family = get_or_create_kernel_family(
                        kernel_families, std::move(param_kinds));

                RefPtr<PythonArgProfile> new_profile = steal(new PythonArgProfile(
                            query.to_owned(), parent, depth, family,
                            std::move(leaf_pyarg_breadth_first_indices),
                            std::move(*arg_kinds),
                            std::move(*flat_param_annotations)));
                map->insert(ProfileMapKey{new_profile}, 0);
                return new_profile.get();
            }

            RefPtr<ExpandAggregates> new_node = steal(new ExpandAggregates(
                    query.to_owned(), parent, depth, std::move(aggregate_args)));
            map->insert(ProfileMapKey{new_node}, 0);
            next = new_node.get();
        }

        query.mark_start();
        for (const AggregateArgInfo& agg_info : next->aggregate_args) {
            PyObject* arg = (*pyarg_objs_breadth_first)[agg_info.breadth_first_index];
            if (!expand_aggregate_arg(arg, agg_info.type,
                                      pyarg_objs_breadth_first, pyarg_types_breadth_first,
                                      pyarg_refs))
                return nullptr;
        }
        query.mark_end();

        map = &next->children;
        parent = next;
    }

    raise(PyExc_RecursionError,
          "Argument nesting exceeds maximum depth of %d", kMaxAggregateNestingDepth);
    return nullptr;
}

static PythonArgProfile* python_arg_profile_lookup(PyObject* const* pyargs,
                                                   Py_ssize_t num_pyargs,
                                                   TileDispatcher* dispatcher,
                                                   LaunchHelper* helper) {
    TileContextDispatcher* ctx_dispatcher = &dispatcher->default_context_dispatcher;
    return python_arg_profile_lookup_impl(
            &ctx_dispatcher->arg_profiles,
            pyargs,
            num_pyargs,
            dispatcher->param_annotations,
            &ctx_dispatcher->kernel_families,
            &helper->pyarg_objs_breadth_first,
            &helper->pyarg_types_breadth_first,
            &helper->leaf_pyarg_objs,
            &helper->pyarg_refs);
}

static Result<PreparedLaunch> prepare_launch(
        const DriverApi* driver,
        PyObject* dispatcher_pyobj,
        CUstream launch_stream,
        PyObject* const* pyargs,
        Py_ssize_t num_pyargs,
        bool capture_kernel_image,
        bool stage_list_args,
        StreamBufferTransaction& tx,
        CudaContextGuard& ctx_guard) {

    LaunchHelperPtr helper = launch_helper_get();

    Result<CUcontext> stream_context = get_stream_context(driver, launch_stream);
    if (!stream_context.is_ok()) return ErrorRaised;
    helper->cuda_context = *stream_context;

    TileDispatcher& dispatcher = py_unwrap<TileDispatcher>(dispatcher_pyobj);
    PythonArgProfile* profile = python_arg_profile_lookup(
            pyargs, num_pyargs, &dispatcher, helper.get());
    if (!profile) return ErrorRaised;

    if (!extract_cuda_args(driver, helper->leaf_pyarg_objs, profile->arg_kinds,
                           profile->flat_param_annotations, *helper)) {
        return ErrorRaised;
    }

    // Get the compute capability of the device this launch targets.
    // Devices with the same compute capability can share a compiled kernel.
    CUdevice dev;
    CUresult dev_res = driver->cuCtxGetDevice_v2(&dev, helper->cuda_context);
    if (dev_res != CUDA_SUCCESS) {
        return raise(PyExc_RuntimeError, "Failed to get current CUDA device: %s",
                     get_cuda_error(driver, dev_res));
    }

    Result<ComputeCapability> compute_capability = get_device_compute_capability(driver, dev);
    if (!compute_capability.is_ok()) return ErrorRaised;

    // Append the compute capability to the constants so that it takes part in the cache key.
    helper->constants.push_back(compute_capability->as_key());

    KernelMap& kernel_map = profile->family->kernels_by_constants;
    KernelMap::Item* kernel_item = kernel_map.find(helper->constants);
    std::optional<KernelImage> kernel_image;
    if (!kernel_item || capture_kernel_image) {
        PyPtr cconv = get_cconv(minimum_calling_convention(
                    profile->family->param_kinds,
                    profile->flat_param_annotations));
        if (!cconv) return ErrorRaised;

        // The last constant is the compute capability. Construct a cursor that skips it.
        ConstantCursor constants_cursor(helper->constants.data(), helper->constants.size() - 1);
        PyPtr signature = make_signature(
                constants_cursor,
                helper->identity_constants,
                profile->family->param_kinds,
                profile->flat_param_annotations,
                cconv);
        if (!signature) return ErrorRaised;

        PyPtr py_compute_capability = steal(Py_BuildValue(
                "(ii)", compute_capability->major, compute_capability->minor));
        if (!py_compute_capability) return ErrorRaised;

        KernelImage* image = capture_kernel_image ? &kernel_image.emplace() : nullptr;
        Result<TileKernel> res = compile(driver, dispatcher_pyobj, signature.get(),
                                         g_default_tile_context, image,
                                         py_compute_capability.get());
        if (!res.is_ok()) return ErrorRaised;

        for (PyObject* obj : helper->identity_constants)
            res->constant_refs.push_back(newref(obj));

        if (!kernel_item)
            kernel_item = kernel_map.insert(std::move(helper->constants), std::move(*res));
    }

    if (!ctx_guard.switch_to(helper->cuda_context))
        return ErrorRaised;

    if (stage_list_args
            && !stage_list_args_on_stream(driver, launch_stream, helper->cuda_context,
                                          helper->arena, helper->list_args,
                                          helper->total_list_data_size_words, tx)) {
        return ErrorRaised;
    }

    if (!hoisted_tensor_map_encode(*driver, kernel_item->value.hoisted_tensor_maps, *helper))
        return ErrorRaised;

    int64_t stack[HostProgram::kMaxStackDepth];
    host_program_eval(kernel_item->value.dyn_smem_size_prog,
        helper->arena, helper->cuarg_offsets, stack);
    int64_t dyn_smem_size = stack[0];
    if (dyn_smem_size < 0 || dyn_smem_size > UINT_MAX)
        return raise(PyExc_RuntimeError, "Invalid dynamic shared memory size");

    return PreparedLaunch{std::move(helper), kernel_item->value.cukernel.kernel,
                          static_cast<unsigned>(dyn_smem_size),
                          &kernel_item->value, std::move(kernel_image)};
}


static constexpr unsigned kMaxCUlaunchAttrs = /*CU_LAUNCH_ATTRIBUTE_MAX=*/17;

static Status launch(const DriverApi* driver,
                     PyObject* dispatcher_pyobj,
                     Grid grid,
                     Grid block,
                     CUstream launch_stream,
                     CUlaunchAttribute launch_attrs[kMaxCUlaunchAttrs],
                     unsigned num_attrs,
                     PyObject* const* pyargs,
                     Py_ssize_t num_pyargs
                     ) {
    CudaContextGuard ctx_guard(driver);
    StreamBufferTransaction tx;
    Result<PreparedLaunch> prep = prepare_launch(
            driver, dispatcher_pyobj, launch_stream, pyargs, num_pyargs,
            /*capture_kernel_image=*/false, /*stage_list_args=*/true, tx, ctx_guard);
    if (!prep.is_ok()) return ErrorRaised;

    CUlaunchConfig config = {
      .gridDimX = grid.dims[0],
      .gridDimY = grid.dims[1],
      .gridDimZ = grid.dims[2],
      .blockDimX = block.dims[0],
      .blockDimY = block.dims[1],
      .blockDimZ = block.dims[2],
      .sharedMemBytes = prep->dynamic_smem_bytes,
      .hStream = launch_stream,
      .attrs = launch_attrs,
      .numAttrs = num_attrs,
    };

    CUresult res = driver->cuLaunchKernelEx(
            &config,
            reinterpret_cast<CUfunction>(prep->kernel),
            make_launch_params(*prep->helper),
            nullptr);

    if (res != CUDA_SUCCESS) {
        if (res == CUDA_ERROR_INVALID_VALUE && try_clarify_invalid_value_error(driver, grid))
            return ErrorRaised;

        return raise(PyExc_RuntimeError, "Failed to launch cuTile kernel: %s",
                     get_cuda_error(driver, res));
    }

    return OK;
}

static Result<double> benchmark(const DriverApi* driver,
                                Grid grid,
                                CUstream launch_stream,
                                CUcontext ctx,
                                CUkernel kernel,
                                unsigned dynamic_smem_bytes,
                                LaunchHelper& helper) {
#define CU_CHECK(name, expr) \
    do { \
        CUresult res = (expr); \
        if (res != CUDA_SUCCESS) \
            return raise(PyExc_RuntimeError, name ": %s", get_cuda_error(driver, res)); \
    } while (0)

    CUdevice device;
    CU_CHECK("cuCtxGetDevice", driver->cuCtxGetDevice(&device));

    // Query L2 cache size for inter-kernel flush
    int l2_cache_size = 0;
    CU_CHECK("cuDeviceGetAttribute", driver->cuDeviceGetAttribute(
             &l2_cache_size, CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE, device));
    // In case the returned l2_cache_size is 0, we set it to a small number
    // so malloc/memset API below will still work.
    l2_cache_size = std::max(1024, l2_cache_size);

    CudaEvent ev_start(driver);
    CU_CHECK("cuEventCreate", ev_start.create());
    CudaEvent ev_end(driver);
    CU_CHECK("cuEventCreate", ev_end.create());

    CudaGraph graph(driver);
    CU_CHECK("cuGraphCreate", graph.create());

    // Build graph:
    //  1. Malloc the L2 flush buffer.
    //  2. Flush L2 cache using memset.
    //  3. Record start event
    //  4. Launch kernel
    //  5. Record end event
    //  6. Free the L2 flush buffer.

    CUgraphNode malloc_node = nullptr;

    CUDA_MEM_ALLOC_NODE_PARAMS malloc_params = {};
    malloc_params.bytesize = l2_cache_size;
    malloc_params.poolProps.allocType = CU_MEM_ALLOCATION_TYPE_PINNED;
    malloc_params.poolProps.handleTypes = CU_MEM_HANDLE_TYPE_NONE;
    malloc_params.poolProps.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    malloc_params.poolProps.location.id = device;

    CU_CHECK("cuGraphAddMemAllocNode",
             driver->cuGraphAddMemAllocNode(&malloc_node, graph.get(), nullptr, 0, &malloc_params));

    CUDA_MEMSET_NODE_PARAMS mparams = {};
    mparams.dst = malloc_params.dptr;
    mparams.value = 0x5a;
    mparams.elementSize = 1;
    mparams.width = l2_cache_size;
    mparams.height = 1;
    mparams.pitch = l2_cache_size;

    CUgraphNode flush_node;
    CU_CHECK("cuGraphAddMemsetNode",
             driver->cuGraphAddMemsetNode(
                     &flush_node, graph.get(),
                     &malloc_node, 1,
                     &mparams, ctx));

    CUgraphNode start_node;
    CU_CHECK("cuGraphAddEventRecordNode",
             driver->cuGraphAddEventRecordNode(
                     &start_node, graph.get(),
                     &flush_node, 1,
                     ev_start.get()));

    // Kernel
    CUDA_KERNEL_NODE_PARAMS kparams = {};
    kparams.func = nullptr;
    kparams.gridDimX = grid.dims[0];
    kparams.gridDimY = grid.dims[1];
    kparams.gridDimZ = grid.dims[2];
    kparams.blockDimX = 1;
    kparams.blockDimY = 1;
    kparams.blockDimZ = 1;
    kparams.sharedMemBytes = dynamic_smem_bytes;
    kparams.kernelParams = make_launch_params(helper);
    kparams.extra = nullptr;
    kparams.kern = kernel;
    kparams.ctx = ctx;

    CUgraphNode kernel_node;
    CU_CHECK("cuGraphAddKernelNode",
             driver->cuGraphAddKernelNode(&kernel_node, graph.get(), &start_node, 1, &kparams));

    // Event: end of kernel
    CUgraphNode end_node;
    CU_CHECK("cuGraphAddEventRecordNode",
             driver->cuGraphAddEventRecordNode(
                     &end_node, graph.get(), &kernel_node, 1, ev_end.get()));


    CUgraphNode free_node;
    CU_CHECK("cuGraphAddMemFreeNode",
             driver->cuGraphAddMemFreeNode(
                 &free_node, graph.get(), &end_node, 1, malloc_params.dptr));

    // Launch and synchronize
    CudaGraphExec graph_exec(driver);
    CU_CHECK("cuGraphInstantiateWithFlags", graph_exec.instantiate(graph));
    CU_CHECK("cuGraphLaunch", driver->cuGraphLaunch(graph_exec.get(), launch_stream));
    CU_CHECK("cuStreamSynchronize", driver->cuStreamSynchronize(launch_stream));

    double total_us = 0;
    float ms;
    CU_CHECK("cuEventElapsedTime",
             driver->cuEventElapsedTime(&ms, ev_start.get(), ev_end.get()));
    total_us = static_cast<double>(ms) * 1000.0;

#undef CU_CHECK

    return total_us;
}

static Result<unsigned> parse_int32_or_int64_dtype_as_bitwidth(PyObject* py_dtype) {
    PyPtr dtype_name = getattr(py_dtype, "name");
    if (!dtype_name) return ErrorRaised;

    if (!PyUnicode_Check(dtype_name.get()))
        return raise(PyExc_TypeError, "DType.name must be a string");

    if (!PyUnicode_CompareWithASCIIString(dtype_name.get(), "int32"))
        return 32;
    if (!PyUnicode_CompareWithASCIIString(dtype_name.get(), "int64"))
        return 64;
    return raise(PyExc_ValueError, "Expected int32 or int64 dtype");
}

// Parse a Python ScalarAnnotation into its C++ equivalent.
static Status parse_scalar_annotation(PyObject* py_scalar_annotation, ScalarAnnotation* dst) {
    if (py_scalar_annotation == Py_None)
        return OK;

    PyPtr dtype = getattr(py_scalar_annotation, "dtype");
    if (!dtype) return ErrorRaised;

    Result<unsigned> bitwidth_res = parse_int32_or_int64_dtype_as_bitwidth(dtype.get());
    if (!bitwidth_res.is_ok()) return ErrorRaised;

    dst->bitwidth = *bitwidth_res;
    return OK;
}


// Extract a tuple-of-ints attribute (e.g. `static_shape_dims`) into `dst`.
static Status extract_dim_tuple_attr(PyObject* py_array_annotation, const char* attr_name,
                                     Vec<int64_t>* dst) {
    PyPtr dims = getattr(py_array_annotation, attr_name);
    if (!dims) return ErrorRaised;

    if (!PyTuple_Check(dims.get()))
        return raise(PyExc_TypeError, "`ArrayAnnotation.%s` must be a tuple, got %s",
                     attr_name, Py_TYPE(dims.get())->tp_name);
    Py_ssize_t nd = PyTuple_GET_SIZE(dims.get());

    dst->reserve(nd);
    for (Py_ssize_t i = 0; i < nd; ++i) {
        dst->push_back(pylong_as<int64_t>(PyTuple_GET_ITEM(dims.get(), i)));
        if (PyErr_Occurred()) return ErrorRaised;
    }
    return OK;
}

// Parse a Python ArrayAnnotation into its C++ equivalent.
static Status parse_array_annotation(PyObject* py_array_annotation, ArrayAnnotation* dst) {
    if (py_array_annotation == Py_None)
        return OK;

    PyPtr index_dtype = getattr(py_array_annotation, "index_dtype");
    if (!index_dtype) return ErrorRaised;

    Result<unsigned> index_bitwidth_res = parse_int32_or_int64_dtype_as_bitwidth(index_dtype.get());
    if (!index_bitwidth_res.is_ok()) return ErrorRaised;

    dst->index_bitwidth = *index_bitwidth_res;

    if (!extract_dim_tuple_attr(py_array_annotation, "static_shape_dims", &dst->static_shape_dims))
        return ErrorRaised;
    if (!extract_dim_tuple_attr(py_array_annotation, "static_stride_dims",
                                &dst->static_stride_dims))
        return ErrorRaised;

    return OK;
}

// Parse a Python ListAnnotation into its C++ equivalent.
static Status parse_list_annotation(PyObject* py_list_annotation, ListAnnotation* dst) {
    if (py_list_annotation == Py_None)
        return OK;

    PyPtr element = getattr(py_list_annotation, "element");
    if (!element) return ErrorRaised;

    return parse_array_annotation(element.get(), &dst->element);
}


// Parse one Python ParameterAnnotationNode into its C++ equivalent.
static RefPtr<ParameterAnnotationNode> parse_parameter_annotation_node(PyObject* obj) {
    PyPtr kind = getattr(obj, "KIND");
    if (!kind) return {};
    if (!PyUnicode_Check(kind.get())) {
        raise(PyExc_TypeError, "KIND must be a string");
        return {};
    }

    if (!PyUnicode_CompareWithASCIIString(kind.get(), "leaf")) {
        PyPtr constant = getattr(obj, "constant");
        PyPtr scalar = getattr(obj, "scalar");
        PyPtr array = getattr(obj, "array");
        PyPtr list = getattr(obj, "list");
        if (!constant || !scalar || !array || !list) return {};

        RefPtr<LeafAnnotationNode> node = steal(new LeafAnnotationNode);
        node->constant = (constant.get() == Py_True);

        if (!parse_scalar_annotation(scalar.get(), &node->scalar))
            return {};
        if (!parse_array_annotation(array.get(), &node->array))
            return {};
        if (!parse_list_annotation(list.get(), &node->list))
            return {};

        return node;
    } else if (!PyUnicode_CompareWithASCIIString(kind.get(), "homogeneous_tuple")) {
        PyPtr py_each = getattr(obj, "each");
        if (!py_each) return {};

        RefPtr<ParameterAnnotationNode> each = parse_parameter_annotation_node(py_each.get());
        if (!each) return {};

        RefPtr<HomogeneousTupleNode> node = steal(new HomogeneousTupleNode);
        node->each = std::move(each);
        return node;
    } else if (!PyUnicode_CompareWithASCIIString(kind.get(), "heterogeneous_tuple")) {
        PyPtr items = getattr(obj, "items");
        if (!items) return {};
        if (!PyTuple_Check(items.get())) {
            raise(PyExc_TypeError, "heterogeneous_tuple `items` must be a tuple, got %s",
                  Py_TYPE(items.get())->tp_name);
            return {};
        }
        Py_ssize_t n = PyTuple_GET_SIZE(items.get());
        RefPtr<HeterogeneousTupleNode> node = steal(new HeterogeneousTupleNode);
        node->items.reserve(n);
        for (Py_ssize_t i = 0; i < n; ++i) {
            RefPtr<ParameterAnnotationNode> item = parse_parameter_annotation_node(
                    PyTuple_GET_ITEM(items.get(), i));
            if (!item) return {};
            node->items.push_back(std::move(item));
        }
        return node;
    } else {
        raise(PyExc_TypeError,
              "expected a ParameterAnnotationNode (leaf/homogeneous_tuple/heterogeneous_tuple),"
              " got KIND=%R", kind.get());
        return {};
    }
}

// Parse a Python sequence of per-parameter ParameterAnnotationNode trees.
static Result<Vec<RefPtr<ParameterAnnotationNode>>>
parse_parameter_annotation_nodes_seq(PyObject* nodes_seq) {
    if (!PyTuple_Check(nodes_seq))
        return raise(PyExc_TypeError, "expected a tuple of parameter annotation nodes");
    Py_ssize_t n = PyTuple_GET_SIZE(nodes_seq);
    Vec<RefPtr<ParameterAnnotationNode>> result;
    result.reserve(n);
    for (Py_ssize_t i = 0; i < n; ++i) {
        RefPtr<ParameterAnnotationNode> node = parse_parameter_annotation_node(
                PyTuple_GET_ITEM(nodes_seq, i));
        if (!node) return ErrorRaised;
        result.push_back(std::move(node));
    }
    return result;
}




static int TileContext_init(PyObject* self, PyObject* args, PyObject* kwargs) {
    const char* keywords[] = {"config", nullptr};
    PyObject* config = nullptr;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "$O", const_cast<char**>(keywords), &config))
        return -1;
    TileContext& context = py_unwrap<TileContext>(self);
    context.config = newref(config);

    // autotune cache starts with None.
    context.autotune_cache = newref(Py_None);

    return 0;
}


static PyObject * TileContext_get_config(PyObject* self, void *closure) {
    TileContext& context = py_unwrap<TileContext>(self);
#ifdef Py_GIL_DISABLED
    PyCriticalSectionGuard guard(&context.accessor_mutex);
#endif
    return Py_NewRef(context.config.get());
}


static PyObject * TileContext_get_autotune_cache(PyObject* self, void *closure) {
    TileContext& context = py_unwrap<TileContext>(self);
#ifdef Py_GIL_DISABLED
    PyCriticalSectionGuard guard(&context.accessor_mutex);
#endif
    return Py_NewRef(context.autotune_cache.get());
}

static int TileContext_set_autotune_cache(PyObject* self, PyObject* value, void* closure) {
    TileContext& context = py_unwrap<TileContext>(self);
#ifdef Py_GIL_DISABLED
    PyCriticalSectionGuard guard(&context.accessor_mutex);
#endif

    // `del ctx.autotune_cache` → set back to None
    if (value == nullptr) {
        context.autotune_cache = newref(Py_None);
        return 0;
    }
    context.autotune_cache = newref(value);
    return 0;
}

static PyGetSetDef TileContext_getsetters[] = {
    {"config", (getter)TileContext_get_config, nullptr},
    {"autotune_cache",
        (getter)TileContext_get_autotune_cache,
        (setter)TileContext_set_autotune_cache,
        nullptr},
    {}  /* Sentinel */
};


PyTypeObject TileContext::pytype = {
    .tp_name = "cuda.tile._cext.TileContext",
    .tp_basicsize = sizeof(PythonWrapper<TileContext>),
    .tp_dealloc = pywrapper_dealloc<TileContext>,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    .tp_getset = TileContext_getsetters,
    .tp_init = TileContext_init,
    .tp_new = pywrapper_new<TileContext>,
};


static int TileDispatcher_init(PyObject* self, PyObject* args, PyObject* kwargs) {
    const char* keywords[] = {"", nullptr};
    PyObject* py_parameter_annotations = nullptr;
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O", const_cast<char**>(keywords),
                                     &py_parameter_annotations))
        return -1;

    Result<Vec<RefPtr<ParameterAnnotationNode>>> param_annotations
            = parse_parameter_annotation_nodes_seq(py_parameter_annotations);
    if (!param_annotations.is_ok()) return -1;

    TileDispatcher& dispatcher = py_unwrap<TileDispatcher>(self);
    dispatcher.param_annotations = std::move(*param_annotations);
    return 0;
}

PyTypeObject TileDispatcher::pytype = {
    .tp_name = "cuda.tile._cext.TileDispatcher",
    .tp_basicsize = sizeof(PythonWrapper<TileDispatcher>),
    .tp_dealloc = pywrapper_dealloc<TileDispatcher>,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    .tp_init = TileDispatcher_init,
    .tp_new = pywrapper_new<TileDispatcher>,
};

static PyObject* get_parameter_constraints_from_pyargs(PyObject* self, PyObject* args) {
#ifdef Py_GIL_DISABLED
    PyCriticalSectionGuard guard(&g_launch_mutex);
#endif
    PyObject* dispatcher_pyobj = nullptr;
    PyObject* pyargs = nullptr;
    PyObject* cconv = nullptr;
    if (!PyArg_ParseTuple(args, "O!O!O!",
                          &TileDispatcher::pytype, &dispatcher_pyobj,
                          &PyTuple_Type, &pyargs,
                          &CallingConvention::pytype, &cconv)) {
        return nullptr;
    }

    TileDispatcher& dispatcher = py_unwrap<TileDispatcher>(dispatcher_pyobj);

    PyObject** kernel_args = reinterpret_cast<PyTupleObject*>(pyargs)->ob_item;
    Py_ssize_t num_kernel_args = PyTuple_GET_SIZE(pyargs);

    LaunchHelperPtr helper = launch_helper_get();

    PythonArgProfile* profile = python_arg_profile_lookup(
            kernel_args, num_kernel_args, &dispatcher, helper.get());
    if (!profile) return nullptr;

    Result<const DriverApi*> driver = get_driver_api();
    if (!driver.is_ok()) return nullptr;

    if (!extract_cuda_args(*driver, helper->leaf_pyarg_objs, profile->arg_kinds,
                           profile->flat_param_annotations, *helper)) {
        return nullptr;
    }

    PyPtr ret = parse_parameter_constraints(
            helper->constants, helper->identity_constants,
            profile->family->param_kinds, profile->flat_param_annotations);
    return ret.release();
}

static Result<Grid> parse_grid(PyObject* tuple) {
    if (!PyTuple_Check(tuple))
        return raise(PyExc_TypeError, "Grid must be a tuple");

    Py_ssize_t tuple_size = PyTuple_GET_SIZE(tuple);
    if (tuple_size > Grid::Len)
        return raise(PyExc_ValueError, "Grid dimensions must be at most %d, got length %zd",
                     Grid::Len, tuple_size);

    Grid grid;
    for (int i = 0; i < Grid::Len; ++i) {
        // Pad with 1s on the right if tuple size < Grid::Len
        unsigned long val = 1;
        if (i < tuple_size) {
            val = PyLong_AsUnsignedLong(PyTuple_GET_ITEM(tuple, i));
            if (PyErr_Occurred()) return ErrorRaised;
            if (val > UINT_MAX)
                return raise(PyExc_ValueError, "Grid[%d] value too big: got=%lu",
                             i, val);
        }
        grid.dims[i] = val;
    }
    if (!validate_grid(grid)) return ErrorRaised;
    return grid;
}

struct LaunchArgs {
    CUstream stream;
    Grid grid;
    Grid block;
    PyObject* dispatcher;
    PyObject** kernel_args;
    Py_ssize_t num_kernel_args;
};


static Result<unsigned> parse_tile_launch_kwargs(PyObject *const *args,
                                                 Py_ssize_t nargs, PyObject *kwargs,
                                                 CUlaunchAttribute launch_attrs[kMaxCUlaunchAttrs]
                                                ) {
    if (kwargs == nullptr)
        return 0;

    CHECK(PyTuple_Check(kwargs) &&
          "Keyword argument tuple is nonnull and not a tuple");

    const auto nkwargs = PyTuple_GET_SIZE(kwargs);
    size_t num_attrs = 0;

    for (Py_ssize_t i = 0; i < nkwargs; i++) {
        PyObject *keyword = PyTuple_GET_ITEM(kwargs, i);
        PyObject *kwarg = args[nargs + i];
        CHECK(keyword && kwarg);

        if (PyUnicode_Compare(keyword, g_programmatic_dependent_launch_pyunicode) == 0) {
            if (!PyBool_Check(kwarg))
                return raise(PyExc_TypeError,
                             "expected argument %U to have type bool", keyword);
            CUlaunchAttribute *attr = &launch_attrs[num_attrs++];
            attr->id = CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION;
            attr->value.programmaticStreamSerializationAllowed = Py_IsTrue(kwarg);
        } else {
            return raise(PyExc_RuntimeError, "Unexpected keyword argument %U",
                         keyword);
        }
    }

    return num_attrs;
}

// Parse extra keyword arguments accepted by the extended launch api into
// launch attributes.
static Result<unsigned> parse_lang_launch_kwargs(PyObject *const *args,
                                                 Py_ssize_t nargs, PyObject *kwargs,
                                                 CUlaunchAttribute launch_attrs[kMaxCUlaunchAttrs]
                                                ) {
    if (kwargs == nullptr)
        return 0;

    CHECK(PyTuple_Check(kwargs) &&
          "Keyword argument tuple is nonnull and not a tuple");

    const auto nkwargs = PyTuple_GET_SIZE(kwargs);
    bool has_block_in_cluster_count = false;
    bool has_preferred_block_in_cluster_count = false;
    size_t num_attrs = 0;

    for (Py_ssize_t i = 0; i < nkwargs; i++) {
        PyObject *keyword = PyTuple_GET_ITEM(kwargs, i);
        PyObject *kwarg = args[nargs + i];
        CHECK(keyword && kwarg);
        if (PyUnicode_Compare(keyword, g_cooperative_pyunicode) == 0) {
            if (!PyBool_Check(kwarg))
                return raise(PyExc_TypeError,
                             "expected argument %U to have type bool", keyword);
            CUlaunchAttribute *attr = &launch_attrs[num_attrs++];
            attr->id = CU_LAUNCH_ATTRIBUTE_COOPERATIVE;
            attr->value.cooperative = Py_IsTrue(kwarg);
        } else if (PyUnicode_Compare(keyword, g_programmatic_dependent_launch_pyunicode) == 0) {
            if (!PyBool_Check(kwarg))
                return raise(PyExc_TypeError,
                             "expected argument %U to have type bool", keyword);
            CUlaunchAttribute *attr = &launch_attrs[num_attrs++];
            attr->id = CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION;
            attr->value.programmaticStreamSerializationAllowed = Py_IsTrue(kwarg);
        } else if (PyUnicode_Compare(keyword, g_block_in_cluster_count_pyunicode) == 0) {
            if (Py_IsNone(kwarg))
                continue;
            const auto grid = parse_grid(kwarg);
            if (!grid.is_ok())
                return ErrorRaised;
            const auto &dims = grid->dims;
            CUlaunchAttribute *attr = &launch_attrs[num_attrs++];
            attr->id = CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION;
            attr->value.clusterDim = {.x = dims[0], .y = dims[1], .z = dims[2]};
            has_block_in_cluster_count = true;
        } else if (PyUnicode_Compare(keyword, g_preferred_block_in_cluster_count_pyunicode) ==
                   0) {
            if (Py_IsNone(kwarg))
                continue;
            const auto grid = parse_grid(kwarg);
            if (!grid.is_ok())
                return ErrorRaised;
            const auto &dims = grid->dims;
            CUlaunchAttribute *attr = &launch_attrs[num_attrs++];
            attr->id = CU_LAUNCH_ATTRIBUTE_PREFERRED_CLUSTER_DIMENSION;
            attr->value.preferredClusterDim = {
                .x = dims[0], .y = dims[1], .z = dims[2]};
            has_preferred_block_in_cluster_count = true;
        } else {
            return raise(PyExc_RuntimeError, "Unexpected keyword argument %U",
                         keyword);
        }
    }

    // ctk docs say: "This attribute will only take effect when a regular
    // cluster dimension has been specified." We could technically allow it, but
    // the user likely made a mistake if preferred dims were passed and
    // "regular" dims were not.
    if (has_preferred_block_in_cluster_count && !has_block_in_cluster_count)
        return raise(PyExc_ValueError,
                     "Keyword argument %U requires that %U is also passed",
                     g_preferred_block_in_cluster_count_pyunicode,
                     g_block_in_cluster_count_pyunicode);

    return num_attrs;
}

static Status parse_launch_args(PyObject* const* args, Py_ssize_t nargs, const char* signature,
                                bool with_block, LaunchArgs* out) {
    if (nargs != 4 + with_block)
        return raise(PyExc_TypeError, "Wrong number of arguments to %s", signature);

    PyObject* stream_pyobj = args[0];
    Result<CUstream> stream_res = parse_stream(stream_pyobj);
    if (!stream_res.is_ok()) return ErrorRaised;
    out->stream = *stream_res;

    PyObject* grid_pyobj = args[1];
    Result<Grid> grid_res = parse_grid(grid_pyobj);
    if (!grid_res.is_ok()) return ErrorRaised;
    out->grid = *grid_res;

    if (with_block) {
        PyObject* block_pyobj = args[2];
        Result<Grid> block_res = parse_grid(block_pyobj);
        if (!block_res.is_ok()) return ErrorRaised;
        out->block = *block_res;
    } else {
        out->block = Grid{1, 1, 1};
    }

    PyObject* dispatcher_pyobj = args[2 + with_block];
    if (!PyObject_TypeCheck(dispatcher_pyobj, &TileDispatcher::pytype)) {
        const char* which = with_block ? "fourth" : "third";
        return raise(PyExc_TypeError,
                "%s expects a tile kernel as the %s argument, got %s",
                signature, which, Py_TYPE(dispatcher_pyobj)->tp_name);
    }
    out->dispatcher = dispatcher_pyobj;

    PyObject* kernel_args_pyobj = args[3 + with_block];
    if (!PyTuple_Check(kernel_args_pyobj)) {
        const char* which = with_block ? "fifth" : "fourth";
        return raise(PyExc_TypeError,
                "%s expects a tuple as the %s argument, got %s",
                signature, which, Py_TYPE(kernel_args_pyobj)->tp_name);
    }

    out->kernel_args = reinterpret_cast<PyTupleObject*>(kernel_args_pyobj)->ob_item;
    out->num_kernel_args = PyTuple_GET_SIZE(kernel_args_pyobj);
    return OK;
}

static PyObject* launch_impl(PyObject* const* args, Py_ssize_t nargs,
                             PyObject* kwargs, const char* signature, bool with_block
                             ) {
#ifdef Py_GIL_DISABLED
    PyCriticalSectionGuard guard(&g_launch_mutex);
#endif
    LaunchArgs launch_args;
    if (!parse_launch_args(args, nargs, signature, with_block, &launch_args))
        return nullptr;

    CUlaunchAttribute launch_attrs[kMaxCUlaunchAttrs];

    const auto num_attrs = with_block
                           ? parse_lang_launch_kwargs(args, nargs, kwargs, launch_attrs)
                           : parse_tile_launch_kwargs(args, nargs, kwargs, launch_attrs);

    if (!num_attrs.is_ok())
        return nullptr;

    Result<const DriverApi*> driver = get_driver_api();
    if (!driver.is_ok()) return nullptr;

    if (!launch(*driver, launch_args.dispatcher, launch_args.grid,
                launch_args.block, launch_args.stream, launch_attrs, *num_attrs,
                launch_args.kernel_args, launch_args.num_kernel_args))
        return nullptr;

    return Py_NewRef(Py_None);
}

#define LAUNCH_SIGNATURE \
  "launch(stream, grid, kernel, kernel_args, /, *, " \
  "programmatic_dependent_launch=False)"

static PyObject* cuda_tile_launch(PyObject*, PyObject* const* args, Py_ssize_t nargs,
                                  PyObject* kwargs) {
  return launch_impl(args, nargs, kwargs, LAUNCH_SIGNATURE,
                     /*with_block=*/false);
}

#define LAUNCH_EXTENDED_SIGNATURE                                                           \
  "launch(stream, block_count, thread_count, kernel, kernel_args, /, *, "                   \
  "cooperative=False, block_in_cluster_count=None, preferred_block_in_cluster_count=None, " \
  "programmatic_dependent_launch=False)"

static PyObject *launch_extended(PyObject *, PyObject *const *args,
                                 Py_ssize_t nargs, PyObject *kwargs) {
  return launch_impl(args, nargs, kwargs, LAUNCH_EXTENDED_SIGNATURE,
                     /*with_block=*/true);
}

#define BENCHMARK_SIGNATURE "_benchmark(stream, grid, kernel, pyargs_tuples, /)"

static PyObject* cuda_tile_benchmark(PyObject* mod, PyObject* const* args, Py_ssize_t nargs) {
#ifdef Py_GIL_DISABLED
    PyCriticalSectionGuard guard(&g_launch_mutex);
#endif
    LaunchArgs launch_args;
    if (!parse_launch_args(args, nargs, BENCHMARK_SIGNATURE, false, &launch_args))
        return nullptr;

    Result<const DriverApi*> driver = get_driver_api();
    if (!driver.is_ok()) return nullptr;

    CudaContextGuard ctx_guard(*driver);
    StreamBufferTransaction tx;
    Result<PreparedLaunch> prep = prepare_launch(
            *driver, launch_args.dispatcher, launch_args.stream,
            launch_args.kernel_args, launch_args.num_kernel_args,
            /*capture_kernel_image=*/false, /*stage_list_args=*/true, tx, ctx_guard);
    if (!prep.is_ok()) return nullptr;

    Result<double> elapsed_us = benchmark(
            *driver, launch_args.grid, launch_args.stream,
            prep->helper->cuda_context, prep->kernel,
            prep->dynamic_smem_bytes, *prep->helper);
    if (!elapsed_us.is_ok()) return nullptr;

    return PyFloat_FromDouble(*elapsed_us);
}

#define EXPORT_IPC_BENCHMARK_PAYLOAD_SIGNATURE \
    "_export_ipc_benchmark_payload(stream, grid, kernel, pyargs_tuples, /)"

static PyObject* cuda_tile_export_ipc_benchmark_payload(PyObject*, PyObject* const* args,
                                                        Py_ssize_t nargs) {
#ifdef Py_GIL_DISABLED
    PyCriticalSectionGuard guard(&g_launch_mutex);
#endif
    LaunchArgs launch_args;
    if (!parse_launch_args(args, nargs, EXPORT_IPC_BENCHMARK_PAYLOAD_SIGNATURE, false,
                           &launch_args))
        return nullptr;

    Result<const DriverApi*> driver = get_driver_api();
    if (!driver.is_ok()) return nullptr;

    CudaContextGuard ctx_guard(*driver);
    StreamBufferTransaction tx;
    Result<PreparedLaunch> prep = prepare_launch(
            *driver, launch_args.dispatcher, launch_args.stream,
            launch_args.kernel_args, launch_args.num_kernel_args,
            /*capture_kernel_image=*/true, /*stage_list_args=*/false, tx, ctx_guard);
    if (!prep.is_ok()) return nullptr;

    LaunchHelper& helper = *prep->helper;
    TileKernel* tile_kernel = prep->tile_kernel;
    CHECK(prep->kernel_image.has_value());
    KernelImage& kernel_image = *prep->kernel_image;

    // IPC payload is not supported for hoisted tensor maps yet.
    // TODO: support hoisted tensor maps
    if (!tile_kernel->hoisted_tensor_maps.empty())
        Py_RETURN_NONE;

    char* cubin_data;
    Py_ssize_t cubin_size;
    if (PyBytes_AsStringAndSize(kernel_image.cubin.get(), &cubin_data, &cubin_size) < 0)
        return nullptr;

    Py_ssize_t symbol_size;
    const char* symbol = PyUnicode_AsUTF8AndSize(kernel_image.symbol.get(), &symbol_size);
    if (!symbol) return nullptr;


    CUdevice device;
    CUresult res = (*driver)->cuCtxGetDevice(&device);
    if (res != CUDA_SUCCESS) {
        raise(PyExc_RuntimeError, "Failed to get current CUDA device: %s",
              get_cuda_error(*driver, res));
        return nullptr;
    }
    int device_id = static_cast<int>(device);
    if (device_id < 0) {
        raise(PyExc_RuntimeError, "Invalid CUDA device id");
        return nullptr;
    }

    IpcHandleExporter ipc_pointer_exporter(*driver);
    // Return None to allow fallback to non-IPC when IPC is not supported.
    if (!helper.array_ptr_arena_offsets.empty()) {
        Result<bool> ipc_supported = ipc_pointer_exporter.check_ipc_supported(device);
        if (!ipc_supported.is_ok()) return nullptr;
        if (!*ipc_supported)
            Py_RETURN_NONE;
    }

    Vec<IpcArrayPtrPatch> arena_array_ptrs;
    arena_array_ptrs.reserve(helper.array_ptr_arena_offsets.size());
    for (ArenaOffset arena_offset : helper.array_ptr_arena_offsets) {
        CUdeviceptr device_ptr = reinterpret_cast<CUdeviceptr>(
                helper.arena[arena_offset].device_ptr);

        Result<bool> capable = ipc_pointer_exporter.is_legacy_ipc_capable(device_ptr);
        if (!capable.is_ok())
            return nullptr;

        // Return None to allow fallback to non-IPC when pointer is not legacy IPC capable.
        if (!*capable)
            Py_RETURN_NONE;

        Result<IpcDevicePtrRef> ipc_array_ptr =
                ipc_pointer_exporter.get_ipc_pointer(device_ptr);
        if (!ipc_array_ptr.is_ok())
            return nullptr;

        arena_array_ptrs.push_back(IpcArrayPtrPatch{arena_offset, *ipc_array_ptr});
    }

    uint32_t grid_dims[Grid::Len];
    for (size_t i = 0; i < Grid::Len; ++i)
        grid_dims[i] = static_cast<uint32_t>(launch_args.grid.dims[i]);

    PyPtr payload = serialize_ipc_benchmark_payload(
            grid_dims, device_id, prep->dynamic_smem_bytes, helper.arena,
            helper.cuarg_offsets, helper.list_args, helper.total_list_data_size_words,
            arena_array_ptrs, ipc_pointer_exporter.ipc_mem_handles,
            cubin_data, static_cast<size_t>(cubin_size),
            symbol, static_cast<size_t>(symbol_size) + 1);
    if (!payload) return nullptr;

    // cuda_tile_benchmark_with_ipc_payload runs in a different process and cannot share
    // the same stream. So synchronize the stream before returning the payload.
    res = (*driver)->cuStreamSynchronize(launch_args.stream);
    if (res != CUDA_SUCCESS) {
        raise(PyExc_RuntimeError, "Failed to synchronize stream for IPC benchmark payload: %s",
              get_cuda_error(*driver, res));
        return nullptr;
    }

    return payload.release();
}

#define BENCHMARK_WITH_IPC_PAYLOAD_SIGNATURE "_benchmark_with_ipc_payload(payload, /)"

static PyObject* cuda_tile_benchmark_with_ipc_payload(PyObject*, PyObject* const* args,
                                                      Py_ssize_t nargs) {
#ifdef Py_GIL_DISABLED
    PyCriticalSectionGuard guard(&g_launch_mutex);
#endif
    if (nargs != 1) {
        raise(PyExc_TypeError, "Wrong number of arguments to %s",
              BENCHMARK_WITH_IPC_PAYLOAD_SIGNATURE);
        return nullptr;
    }

    PyObject* py_payload = args[0];
    if (!PyBytes_Check(py_payload)) {
        raise(PyExc_TypeError, "IPC benchmark payload must be bytes");
        return nullptr;
    }

    char* payload_data;
    Py_ssize_t payload_nbytes;
    if (PyBytes_AsStringAndSize(py_payload, &payload_data, &payload_nbytes) < 0)
        return nullptr;

    LaunchHelperPtr helper = launch_helper_get();
    Result<IpcBenchmarkPayload> payload_res = deserialize_ipc_benchmark_payload(
            payload_data, static_cast<size_t>(payload_nbytes), *helper);
    if (!payload_res.is_ok()) return nullptr;
    IpcBenchmarkPayload& payload = *payload_res;

    Grid grid;
    for (size_t i = 0; i < Grid::Len; ++i)
        grid.dims[i] = payload.grid_dims[i];
    if (!validate_grid(grid)) return nullptr;

    Result<const DriverApi*> driver = get_driver_api();
    if (!driver.is_ok()) return nullptr;

    int device_id = payload.device_id;
    CUdevice device;
    CUresult res = (*driver)->cuDeviceGet(&device, device_id);
    if (res != CUDA_SUCCESS) {
        raise(PyExc_RuntimeError, "cuDeviceGet: %s", get_cuda_error(*driver, res));
        return nullptr;
    }

    CUcontext ctx;
    res = (*driver)->cuDevicePrimaryCtxRetain(&ctx, device);
    if (res != CUDA_SUCCESS) {
        raise(PyExc_RuntimeError, "cuDevicePrimaryCtxRetain: %s",
              get_cuda_error(*driver, res));
        return nullptr;
    }

    CudaContextGuard ctx_guard(*driver);
    if (!ctx_guard.switch_to(ctx))
        return nullptr;

    IpcHandleCreator ipc_mem_handle(*driver);
    if (!ipc_mem_handle.open_handles(payload.ipc_mem_handles)) return nullptr;

    for (const IpcArrayPtrPatch& arena_array_ptr : payload.arena_array_ptrs) {
        if (arena_array_ptr.arena_offset >= helper->arena.size()) {
            raise(PyExc_ValueError, "IPC arena array pointer offset is out of range");
            return nullptr;
        }
        const IpcDevicePtrRef& ipc_array_ptr = arena_array_ptr.array_ptr;
        if (ipc_array_ptr.ipc_mem_handle_index >= ipc_mem_handle.mapped_handles.size()) {
            raise(PyExc_ValueError,
                  "IPC arena array pointer ipc_mem_handle_index is out of range");
            return nullptr;
        }

        helper->arena[arena_array_ptr.arena_offset].device_ptr = reinterpret_cast<void*>(
                ipc_mem_handle.mapped_handles[ipc_array_ptr.ipc_mem_handle_index]
                + ipc_array_ptr.offset);
    }

    CUstream launch_stream = nullptr;
    StreamBufferTransaction tx;
    if (!stage_list_args_on_stream(*driver, launch_stream, ctx, helper->arena,
                                   helper->list_args, helper->total_list_data_size_words,
                                   tx)) {
        return nullptr;
    }

    Result<CudaKernel> kernel = load_cuda_kernel(
            *driver, payload.cubin.data(), payload.cubin.size(), payload.symbol.data());
    if (!kernel.is_ok()) return nullptr;

    Result<double> elapsed_us = benchmark(
            *driver, grid, launch_stream, ctx, kernel->kernel, payload.dynamic_smem_bytes,
            *helper);
    if (!elapsed_us.is_ok()) return nullptr;

    return PyFloat_FromDouble(*elapsed_us);
}

static Status init_default_tile_context() {
    PyPtr context_module = steal(PyImport_ImportModule("cuda.tile._context"));
    if (!context_module) return ErrorRaised;

    PyPtr default_context_config = steal(
        PyObject_CallMethod(context_module.get(), "init_context_config_from_env", "")
    );
    if (!default_context_config) return ErrorRaised;

    g_default_tile_context = pywrapper_new<TileContext>(&TileContext::pytype, nullptr, nullptr);
    if (!g_default_tile_context) return ErrorRaised;
    TileContext& tile_context = py_unwrap<TileContext>(g_default_tile_context);
    tile_context.config = default_context_config;

    tile_context.autotune_cache = newref(Py_None);

    return OK;
};


static Status get_standard_globals() {
    PyPtr enum_mod = steal(PyImport_ImportModule("enum"));
    if (!enum_mod) return ErrorRaised;

    PyPtr enum_type = getattr(enum_mod, "Enum");
    if (!enum_type) return ErrorRaised;

    g_enum_Enum_type = enum_type.release();
    return OK;
}


static PyObject* dev_features_enabled(PyObject*, PyObject*) {
#ifdef CUDA_TILE_ENABLE_DEV_FEATURES
    Py_RETURN_TRUE;
#else
    Py_RETURN_FALSE;
#endif
}

static PyObject* cconv_v3_enabled(PyObject*, PyObject*) {
#ifdef ENABLE_CCONV_V3
    Py_RETURN_TRUE;
#else
    Py_RETURN_FALSE;
#endif
}

static PyMethodDef functions[] = {
    {"dev_features_enabled", dev_features_enabled, METH_NOARGS, nullptr},
    {"cconv_v3_enabled", cconv_v3_enabled, METH_NOARGS, nullptr},
    {"launch", reinterpret_cast<PyCFunction>(cuda_tile_launch),
        METH_FASTCALL | METH_KEYWORDS, LAUNCH_SIGNATURE "\n"
        "--\n\n"
        "Launch a cuTile kernel.\n\n"
        "Args:\n"
        "   stream: The CUDA stream to execute the |kernel| on.\n"
        "   grid: Tuple of up to 3 grid dimensions to execute the |kernel| over.\n"
        "   kernel: The |kernel| to execute.\n"
        "   kernel_args: Positional arguments to pass to the kernel.\n"
    },
    {"get_parameter_constraints_from_pyargs", get_parameter_constraints_from_pyargs,
      METH_VARARGS, ""},
    {"classify_constant", py_classify_constant, METH_VARARGS,
      "Classify a constant Python value into a ConstantKind"},
    {"foreign_dtype_object_register", foreign_dtype_object_register, METH_VARARGS,
     "Register a foreign dtype object"},
    {"foreign_dtype_object_to_native", foreign_dtype_object_to_native, METH_O,
     "Get a native dtype object for a foreign dtype object"},
    {"_benchmark", reinterpret_cast<PyCFunction>(cuda_tile_benchmark), METH_FASTCALL,
        BENCHMARK_SIGNATURE "\n"
        "--\n\n"
        "Benchmark a cuTile kernel using CUDA graphs.\n\n"
        "Returns total elapsed time in microseconds (L2 flush between invocations).\n"
    },
    {"_export_ipc_benchmark_payload",
        reinterpret_cast<PyCFunction>(cuda_tile_export_ipc_benchmark_payload), METH_FASTCALL,
        EXPORT_IPC_BENCHMARK_PAYLOAD_SIGNATURE "\n"
        "--\n\n"
        "Build a CUDA IPC benchmark payload for the _benchmark_with_ipc_payload call.\n"
        "Returns None when the payload is not supported by CUDA IPC.\n"
    },
    {"_benchmark_with_ipc_payload",
        reinterpret_cast<PyCFunction>(cuda_tile_benchmark_with_ipc_payload), METH_FASTCALL,
        BENCHMARK_WITH_IPC_PAYLOAD_SIGNATURE "\n"
        "--\n\n"
        "Benchmark a cuTile kernel with a CUDA IPC payload using CUDA graphs.\n"
        "The IPC payload must be generated by _export_ipc_benchmark_payload().\n"
        "Returns total elapsed time in microseconds (L2 flush between invocations).\n"
    },
    {}
};

// Add the launch_extended() function separately because we want its name
// to be just "launch"
static Status add_launch_extended_func(PyObject* m) {
    static PyMethodDef launch_extended_def = {
        "launch", reinterpret_cast<PyCFunction>(launch_extended),
        METH_FASTCALL | METH_KEYWORDS,
        LAUNCH_EXTENDED_SIGNATURE"\n"
        "--\n\n"
    };
    PyPtr func = steal(PyCFunction_New(&launch_extended_def, m));
    if (PyModule_AddObjectRef(m, "launch_extended", func.get()) < 0)
        return ErrorRaised;
    return OK;
}

#define INIT_STRING_CONSTANT(name, value) \
    if (!(name = PyUnicode_InternFromString(value))) return ErrorRaised

#define INIT_STRING_IDENT(ident) INIT_STRING_CONSTANT(g_##ident##_pyunicode, #ident)


Status tile_kernel_init(PyObject* m) {
    INIT_STRING_IDENT(__cuda_array_interface__);
    INIT_STRING_IDENT(typestr);
    INIT_STRING_IDENT(shape);
    INIT_STRING_IDENT(data);
    INIT_STRING_IDENT(strides);
    INIT_STRING_IDENT(__dlpack__);
    INIT_STRING_IDENT(compile);
    INIT_STRING_IDENT(dynamic_shared_memory_bytes);
    INIT_STRING_IDENT(cooperative);
    INIT_STRING_IDENT(block_in_cluster_count);
    INIT_STRING_IDENT(preferred_block_in_cluster_count);
    INIT_STRING_IDENT(programmatic_dependent_launch);
    INIT_STRING_IDENT(__dataclass_fields__);
    INIT_STRING_IDENT(torch);
    INIT_STRING_IDENT(cupy);
    INIT_STRING_IDENT(cuda_stream);
    INIT_STRING_IDENT(ptr);
    INIT_STRING_CONSTANT(g_numba_cuda_pyunicode, "numba.cuda");
    INIT_STRING_CONSTANT(g_cuda_bindings_driver_pyunicode, "cuda.bindings.driver");

    if (!get_standard_globals()) return ErrorRaised;

    g_constant_kind_enum = define_constant_kind_enum().release();
    if (!g_constant_kind_enum) return ErrorRaised;

    if (PyModule_AddObjectRef(m, "ConstantKind", g_constant_kind_enum) < 0)
        return ErrorRaised;

    g_stream_buffer_pool_by_ctx_id = new StreamBufferPoolMap();

    if (PyType_Ready(&CallingConvention::pytype) < 0)
        return ErrorRaised;

    if (PyType_Ready(&TileContext::pytype) < 0)
        return ErrorRaised;

    if (PyType_Ready(&TileDispatcher::pytype) < 0)
        return ErrorRaised;

    if (PyModule_AddObjectRef(m, "CallingConvention",
                reinterpret_cast<PyObject*>(&CallingConvention::pytype)) < 0)
        return ErrorRaised;

    if (PyModule_AddObjectRef(m, "TileContext",
                reinterpret_cast<PyObject*>(&TileContext::pytype)) < 0)
        return ErrorRaised;

    if (PyModule_AddObjectRef(m, "TileDispatcher",
                reinterpret_cast<PyObject*>(&TileDispatcher::pytype)) < 0)
        return ErrorRaised;

    if (PyModule_AddFunctions(m, functions) < 0)
        return ErrorRaised;

    if (!add_launch_extended_func(m))
        return ErrorRaised;

    if (!init_default_tile_context()) return ErrorRaised;

    if (PyModule_AddObjectRef(m, "default_tile_context", g_default_tile_context) < 0)
        return ErrorRaised;

    if (!define_integer_constants(m))
        return ErrorRaised;

    return OK;
}
