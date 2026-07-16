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

static PyTypeObject* g_torch_Tensor_type;
static PyTypeObject* g_torch_cuda_Stream_type;
static PyObject* g_torch_to_dlpack_func;

static PyObject* g_default_tile_context;


static PyObject* get_datatype_module() {
    static PyObject* m;
    if (!m) m = PyImport_ImportModule("cuda.tile._datatype");
    return m;
}


static PyObject* get_signature_module() {
    static PyObject* m;
    if (!m) m = PyImport_ImportModule("cuda.tile.compilation._signature");
    return m;
}


#define FOREACH_TORCH_DTYPE(X) \
    X(bool, 8, 1, kDLBool) \
    X(uint8, 8, 1, kDLUInt) \
    X(uint16, 16, 1, kDLUInt) \
    X(uint32, 32, 1, kDLUInt) \
    X(uint64, 64, 1, kDLUInt) \
    X(int8, 8, 1, kDLInt) \
    X(int16, 16, 1, kDLInt) \
    X(int32, 32, 1, kDLInt) \
    X(int64, 64, 1, kDLInt) \
    X(float16, 16, 1, kDLFloat) \
    X(float32, 32, 1, kDLFloat) \
    X(float64, 64, 1, kDLFloat) \
    X(bfloat16, 16, 1, kDLBfloat) \
    X(float8_e4m3fn, 8, 1, kDLFloat8_e4m3fn) \
    X(float8_e5m2, 8, 1, kDLFloat8_e5m2) \
    X(float8_e8m0fnu, 8, 1, kDLFloat8_e8m0fnu)


#define DECLARE_TORCH_DTYPE_GLOBAL(name, bitwidth, lanes, typecode) \
    static PyObject* g_torch_dtype_##name;


FOREACH_TORCH_DTYPE(DECLARE_TORCH_DTYPE_GLOBAL)


static PyTypeObject* g_cupy_cuda_Stream_type;

static PyTypeObject* g_numba_cuda_Stream_type;
static PyTypeObject* g_cuda_bindings_CUstream_type;

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

enum class CallConvVersion {
    CutilePython_V1 = 1,
    CutilePython_V2 = 2,
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

static PyObject* CallingConvention_cutile_python_v1(PyObject*, PyObject*) {
    static PyObject* c;
    if (!c) c = get_cconv(CallConvVersion::CutilePython_V1).release();
    return Py_NewRef(c);
}

static PyObject* CallingConvention_cutile_python_v2(PyObject*, PyObject*) {
    static PyObject* c;
    if (!c) c = get_cconv(CallConvVersion::CutilePython_V2).release();
    return Py_NewRef(c);
}

static PyPtr parse_cutile_python_calling_convention(const char* s) {
    if (s[0] == '1' && !s[1])
        return get_cconv(CallConvVersion::CutilePython_V1);
    if (s[0] == '2' && !s[1])
        return get_cconv(CallConvVersion::CutilePython_V2);
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
        ret->pyarg_types.clear();
        ret->pyarg_objs.clear();
        return LaunchHelperPtr(ret);
    } else {
        return LaunchHelperPtr(new LaunchHelper());
    }
}

enum class ParameterKind {
    Array,
    Boolean,
    Integer,
    Float,
    List,
    TupleBegin,
    TupleEnd,
};

struct KernelFamily : SimpleRefcount<KernelFamily> {
    Vec<ParameterKind> param_kinds;
    KernelMap kernels_by_constants;

    explicit KernelFamily(Vec<ParameterKind>&& param_kinds) : param_kinds(std::move(param_kinds)) {}
};


enum class PythonArgKind {
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

static ParameterKind param_kind_from_pyarg_kind(PythonArgKind k) {
    switch (k) {
    case PythonArgKind::TorchTensorDlpack: return ParameterKind::Array;
    case PythonArgKind::DlpackArray: return ParameterKind::Array;
    case PythonArgKind::CudaArray: return ParameterKind::Array;
    case PythonArgKind::PyBool: return ParameterKind::Boolean;
    case PythonArgKind::PyLong: return ParameterKind::Integer;
    case PythonArgKind::PyFloat: return ParameterKind::Float;
    case PythonArgKind::PyList: return ParameterKind::List;
    }
    CHECK(false);
}

static constexpr PyTypeObject* kTupleEndType = nullptr;

static Result<PythonArgKind> classify_arg(PyObject* arg) {
    if (PyType_IsSubtype(Py_TYPE(arg), &PyTuple_Type))
        return raise(PyExc_TypeError,
            "Unsupported argument type '%s': only plain tuple is accepted, not subclasses",
            Py_TYPE(arg)->tp_name);

    if (PyBool_Check(arg))
        return PythonArgKind::PyBool;

    if (PyLong_Check(arg))
        return PythonArgKind::PyLong;

    if (PyFloat_Check(arg))
        return PythonArgKind::PyFloat;

    if (PyList_Check(arg))
        return PythonArgKind::PyList;

    if (g_torch_Tensor_type && PyObject_TypeCheck(arg, g_torch_Tensor_type)) {
        // Calling torch._C._to_dlpack(arg) is much faster than calling arg.__dlpack__()
        // because it goes straight into C++ code, with no Python in between.
        // So we always prefer that.
        if (g_torch_to_dlpack_func)
            return PythonArgKind::TorchTensorDlpack;
    }

    if (PyObject_HasAttr(arg, g___dlpack___pyunicode))
        return PythonArgKind::DlpackArray;

    if (PyObject_HasAttr(arg, g___cuda_array_interface___pyunicode))
        return PythonArgKind::CudaArray;

    return raise(PyExc_TypeError, "Unsupported argument type %s", Py_TYPE(arg)->tp_name);
}

struct LeafAnnotationNode;

struct PythonArgProfile {
    RefPtr<KernelFamily> family;
    Vec<PythonArgKind> arg_kinds;
    Vec<RefPtr<LeafAnnotationNode>> flat_param_annotations;
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

static constexpr int u8_pair(uint8_t x, uint8_t y) {
    return x | (y << 8);
}

static Result<const char*> dtype_name(DLDataType dtype) {
    if (dtype.lanes != 1)
        return raise(PyExc_TypeError, "Array dtypes with multiple lanes are not supported");

    switch (u8_pair(dtype.code, dtype.bits)) {
    case u8_pair(kDLBool, 8): return "bool_";

    case u8_pair(kDLInt, 8): return "int8";
    case u8_pair(kDLInt, 16): return "int16";
    case u8_pair(kDLInt, 32): return "int32";
    case u8_pair(kDLInt, 64): return "int64";

    case u8_pair(kDLUInt, 8): return "uint8";
    case u8_pair(kDLUInt, 16): return "uint16";
    case u8_pair(kDLUInt, 32): return "uint32";
    case u8_pair(kDLUInt, 64): return "uint64";

    case u8_pair(kDLFloat, 16): return "float16";
    case u8_pair(kDLFloat, 32): return "float32";
    case u8_pair(kDLFloat, 64): return "float64";

    case u8_pair(kDLBfloat, 16): return "bfloat16";

    case u8_pair(kDLFloat8_e4m3fn, 8): return "float8_e4m3fn";
    case u8_pair(kDLFloat8_e5m2, 8): return "float8_e5m2";
    case u8_pair(kDLFloat8_e8m0fnu, 8): return "float8_e8m0fnu";

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
        const Word* array_repr, size_t ndim, unsigned dtype_bitwidth, unsigned index_bitwidth) {

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
                    is_singleton_stride_one)
                ret.stride_16byte_divisible |= 1u << i;

            if (stride == 1 && !is_singleton_stride_one)
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

    explicit Cursor(const Vec<T>& vec)
        : data(vec.data()), len(vec.size())
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

static Result<size_t> normalize_dim(int64_t signed_dim, size_t ndim) {
    int64_t ndim_i64 = static_cast<int64_t>(ndim);
    if (signed_dim < 0)
        signed_dim += ndim_i64;
    if (signed_dim < 0 || signed_dim >= ndim_i64)
        return raise(PyExc_ValueError,
                     "static_shape_dims contains axis %lld, but array rank is %zu",
                     static_cast<long long>(signed_dim), ndim);
    return static_cast<size_t>(signed_dim);
}


struct ArrayTypeConstantBuilder {
    void* device_ptr = nullptr;
    uint64_t bits = -1;
    const Word* first_array_shape = nullptr;

    Status update(const Arena& arena, const ArrayRepr& ar, const Vec<int64_t>& static_shape_dims) {
        const Word* repr = arena.data() + ar.repr;
        device_ptr = repr[0].device_ptr;
        bits &= compute_array_specialization_bits(
                    repr, ar.arrty.ndim, ar.arrty.dtype.bits * ar.arrty.dtype.lanes,
                    ar.arrty.index_bitwidth).u64;


        const Word* shape_words = repr + 1;
        if (!first_array_shape) {
            first_array_shape = shape_words;
        } else {
            // When several arrays share one constraint (e.g. the items of a
            // list), the kernel is specialized to a single set of compile-time
            // shape values, so every array must agree on those dims.
            for (int64_t dim : static_shape_dims) {
                Result<size_t> normalized_dim = normalize_dim(dim, ar.arrty.ndim);
                if (!normalized_dim.is_ok()) return ErrorRaised;

                int64_t expected = first_array_shape[*normalized_dim].i64;
                int64_t actual = shape_words[*normalized_dim].i64;
                if (expected != actual)
                    return raise(PyExc_ValueError,
                                 "Arrays in list vary in static shape at axis %zu (%lld vs %lld)",
                                 *normalized_dim,
                                 static_cast<long long>(expected),
                                 static_cast<long long>(actual));
            }
        }
        return OK;
    }

    void finalize(const DriverApi* driver, const ArrayType& arrty,
                  const Vec<int64_t>& static_shape_dims,
                  LaunchHelper& helper) {
        CHECK(first_array_shape);
        // Written as [packed_array_type, specialization_bits, static_shape_value...];
        // parse_array_constraint() must read in the same order.
        helper.constants.push_back(pack_array_type(arrty));
        helper.constants.push_back(bits);
        for (int64_t dim : static_shape_dims) {
            Result<size_t> normalized_dim = normalize_dim(dim, arrty.ndim);
            CHECK(normalized_dim.is_ok());
            helper.constants.push_back(first_array_shape[*normalized_dim].i64);
        }
        if (!helper.cuda_context) {
            driver->cuPointerGetAttribute(&helper.cuda_context, CU_POINTER_ATTRIBUTE_CONTEXT,
                    reinterpret_cast<CUdeviceptr>(device_ptr));
        }
    }
};


// Parse the constants generated by ArrayTypeConstantBuilder.finalize()
// into an ArrayConstraint object.
static PyPtr parse_array_constraint(ConstantCursor& cursor, const Vec<int64_t>& static_shape_dims) {
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

    PyPtr constant_strides = steal(PyTuple_New(arrty.ndim));
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
        PyTuple_SET_ITEM(constant_strides.get(), i, Py_NewRef(obj));

        obj = special_bits.is_stride_16byte_divisible(i) ? stride_divisor.get() : one.get();
        PyTuple_SET_ITEM(stride_divisible_by.get(), i, Py_NewRef(obj));

        obj = special_bits.is_shape_divisible_by_16(i) ? sixteen.get() : one.get();
        PyTuple_SET_ITEM(shape_divisible_by.get(), i, Py_NewRef(obj));

        PyList_SET_ITEM(constant_shape.get(), i, Py_NewRef(Py_None));
    }

    for (int64_t dim : static_shape_dims) {
        Result<size_t> normalized_dim = normalize_dim(dim, arrty.ndim);
        CHECK(normalized_dim.is_ok());  // already checked this in extract_array()

        if (PyList_GET_ITEM(constant_shape.get(), *normalized_dim) != Py_None) {
            raise(PyExc_ValueError,
                  "static_shape_dims contains duplicate axis %lld", static_cast<long long>(dim));
            return {};
        }

        PyPtr size_pylong = steal(PyLong_FromLongLong(cursor.next()));
        if (!size_pylong) return {};

        if (PySequence_SetItem(constant_shape.get(), *normalized_dim, size_pylong.get()) < 0)
            return {};
    }

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


#define TORCH_DTYPE_TO_DL_DTYPE(name, bitwidth, lanes, typecode) \
if (torch_dtype == g_torch_dtype_##name) \
    return DLDataType{typecode, bitwidth, lanes};


static Result<DLDataType> dtype_from_torch_dtype(PyObject* torch_dtype) {
    FOREACH_TORCH_DTYPE(TORCH_DTYPE_TO_DL_DTYPE)
    return raise(PyExc_TypeError, "dtype is not supported");
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
    PyPtr dlpack_capsule = steal(PyObject_CallFunctionObjArgs(
                g_torch_to_dlpack_func, pyobj, nullptr));

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
    if (!builder.update(helper.arena, *ar, array_ann.static_shape_dims)) return ErrorRaised;
    builder.finalize(driver, ar->arrty, array_ann.static_shape_dims, helper);
    return OK;
}

enum class PylongConstantEncoding : int64_t {
    I64,
    U64
};

static inline Status extract_py_bool(PyObject* pyobj, bool is_constant, LaunchHelper& helper) {
    int val = PyObject_IsTrue(pyobj);
    if (val < 0) return ErrorRaised;

    if (is_constant)
        helper.constants.push_back(val);
    else
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

static PyPtr parse_pybool_constraint(ConstantCursor& cursor, bool is_constant) {
    if (is_constant) {
        int64_t val = cursor.next();
        return make_constant_constraint(val ? Py_True : Py_False);
    } else {
        return make_scalar_constraint(DLDataType{kDLBool, 8, 1});
    }
}

static inline Status extract_py_long(PyObject* pyobj, bool is_constant, unsigned bitwidth,
                                     LaunchHelper& helper) {
    if (is_constant) {
        int overflow;
        int64_t value = pylong_as_overflow_and<int64_t>(pyobj, &overflow);
        if (PyErr_Occurred()) return ErrorRaised;
        if (overflow) {
            // TODO: support big values by extracting all digits
            helper.constants.push_back(static_cast<int64_t>(PylongConstantEncoding::U64));
            uint64_t uval = pylong_as<uint64_t>(pyobj);
            if (PyErr_Occurred()) return ErrorRaised;
            helper.constants.push_back(uval);
        } else {
            helper.constants.push_back(static_cast<int64_t>(PylongConstantEncoding::I64));
            helper.constants.push_back(value);
        }
    } else {
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
    }
    return OK;
}

static PyPtr parse_pylong_constraint(ConstantCursor& cursor, bool is_constant, unsigned bitwidth) {
    if (is_constant) {
        int64_t format = cursor.next();
        PyPtr value;
        if (format == static_cast<int64_t>(PylongConstantEncoding::I64)) {
            value = steal(PyLong_FromLongLong(cursor.next()));
        } else if (format == static_cast<int64_t>(PylongConstantEncoding::U64)) {
            value = steal(PyLong_FromUnsignedLongLong(cursor.next()));
        } else {
            CHECK(false);
        }
        if (!value) return {};
        return make_constant_constraint(value.get());
    } else {
        return make_scalar_constraint(DLDataType{kDLInt, static_cast<uint8_t>(bitwidth), 1});
    }
}

static void extract_py_float(PyObject* pyobj, bool is_constant, LaunchHelper& helper) {
    double value = PyFloat_AS_DOUBLE(pyobj);
    if (is_constant) {
        int64_t i64_val = 0;
        static_assert(sizeof(i64_val) == sizeof(value));
        mem_copy(&i64_val, &value, sizeof(i64_val));
        helper.constants.push_back(i64_val);
    } else {
        push_single_word_cuarg(helper, {.f32 = static_cast<float>(value)});
    }
}

static PyPtr parse_pyfloat_constraint(ConstantCursor& cursor, bool is_constant) {
    if (is_constant) {
        union { int64_t i64; double f64; } u;
        u.i64 = cursor.next();
        PyPtr value = steal(PyFloat_FromDouble(u.f64));
        return make_constant_constraint(value.get());
    } else {
        return make_scalar_constraint(DLDataType{kDLFloat, 32, 1});
    }
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
    Result<PythonArgKind> first_item_res = classify_arg(first_item);
    if (!first_item_res.is_ok()) return ErrorRaised;

    if (param_kind_from_pyarg_kind(*first_item_res) != ParameterKind::Array) {
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
    if (!builder.update(helper.arena, *first_repr_res, list_ann.element.static_shape_dims))
        return ErrorRaised;

    // Handle the rest of the list
    for (size_t i = 1; i < len; ++i) {
        PyObject* item = PyList_GET_ITEM(pyobj, i);
        PythonArgKind kind = first_arg_kind;

        // Avoid calling classify_arg() if the object type is the same
        if (first_item_type != item->ob_type) {
             Result<PythonArgKind> res = classify_arg(item);
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

        if (!builder.update(helper.arena, *repr_res, list_ann.element.static_shape_dims))
            return ErrorRaised;
    }

    // TODO: If we accept lists of things other than arrays, then to disambiguate,
    //       we need to push another constant here that specifies the type of the list element .
    builder.finalize(driver, first_repr_res->arrty, list_ann.element.static_shape_dims, helper);
    return OK;
}

static PyPtr parse_list_constraint(ConstantCursor& cursor, const Vec<int64_t>& static_shape_dims) {
    PyPtr element = parse_array_constraint(cursor, static_shape_dims);
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

struct ParameterAnnotationNode : SimpleRefcount<ParameterAnnotationNode> {
    enum Kind { Leaf, HomogeneousTuple, HeterogeneousTuple };
    virtual Kind kind() const = 0;
    virtual Status flatten_tuple(Cursor<ParameterKind>* cursor,
                                 Vec<RefPtr<LeafAnnotationNode>>* out) = 0;
    virtual ~ParameterAnnotationNode() {}
};


static Status flatten_parameter_annotation_node(ParameterAnnotationNode* node,
                                                Cursor<ParameterKind>* cursor,
                                                Vec<RefPtr<LeafAnnotationNode>>* out);


struct LeafAnnotationNode : ParameterAnnotationNode {
    bool constant = false;
    ScalarAnnotation scalar;
    ArrayAnnotation array;
    ListAnnotation list;

    virtual Kind kind() const { return Leaf; }

    virtual Status flatten_tuple(Cursor<ParameterKind>* cursor,
                                 Vec<RefPtr<LeafAnnotationNode>>* out) override {
        for (size_t item_idx = 0; cursor->peek() != ParameterKind::TupleEnd; ++item_idx) {
            if (!flatten_parameter_annotation_node(this, cursor, out))
                return ErrorRaised;
        }
        return OK;
    }
};


struct HomogeneousTupleNode : ParameterAnnotationNode {
    RefPtr<ParameterAnnotationNode> each;

    virtual Kind kind() const { return HomogeneousTuple; }

    virtual Status flatten_tuple(Cursor<ParameterKind>* cursor,
                                 Vec<RefPtr<LeafAnnotationNode>>* out) override {
        for (size_t item_idx = 0; cursor->peek() != ParameterKind::TupleEnd; ++item_idx) {
            if (!flatten_parameter_annotation_node(each.get(), cursor, out))
                return ErrorRaised;
        }
        return OK;
    }
};


struct HeterogeneousTupleNode : ParameterAnnotationNode {
    Vec<RefPtr<ParameterAnnotationNode>> items;

    virtual Kind kind() const { return HeterogeneousTuple; }

    virtual Status flatten_tuple(Cursor<ParameterKind>* cursor,
                                 Vec<RefPtr<LeafAnnotationNode>>* out) override {
        LeafAnnotationNode default_node;

        size_t item_idx = 0;
        while (cursor->peek() != ParameterKind::TupleEnd) {
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
                                                Cursor<ParameterKind>* cursor,
                                                Vec<RefPtr<LeafAnnotationNode>>* out) {
    ParameterKind kind = cursor->next();
    if (kind == ParameterKind::TupleBegin) {
        if (!node->flatten_tuple(cursor, out))
            return ErrorRaised;
        ParameterKind tuple_end = cursor->next();
        CHECK(tuple_end == ParameterKind::TupleEnd);
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
                                   const Vec<ParameterKind>& param_kinds) {
    Vec<RefPtr<LeafAnnotationNode>> ret;
    ret.reserve(param_kinds.size());
    Cursor<ParameterKind> cursor(param_kinds);
    for (const RefPtr<ParameterAnnotationNode>& node : nodes) {
        if (!flatten_parameter_annotation_node(node.get(), &cursor, &ret))
            return ErrorRaised;
    }
    CHECK(!cursor.len);
    return ret;
}

static Status extract_arg(const DriverApi* driver, PyObject* obj, PythonArgKind kind,
                          LeafAnnotationNode* annotation, LaunchHelper& helper) {
    switch (kind) {
    case PythonArgKind::TorchTensorDlpack:
        return extract_array<arrayrepr_torch_tensor_dlpack>(driver, obj, annotation->array, helper);
    case PythonArgKind::DlpackArray:
        return extract_array<arrayrepr_dlpack>(driver, obj, annotation->array, helper);
    case PythonArgKind::CudaArray:
        return extract_array<arrayrepr_cuda_array_iface>(driver, obj, annotation->array, helper);
    case PythonArgKind::PyBool:
        return extract_py_bool(obj, annotation->constant, helper);
    case PythonArgKind::PyLong:
        return extract_py_long(obj, annotation->constant, annotation->scalar.bitwidth, helper);
    case PythonArgKind::PyFloat:
        extract_py_float(obj, annotation->constant, helper);
        return OK;
    case PythonArgKind::PyList:
        return extract_py_list(driver, obj, annotation->list, helper);
    }
    // Unsupported types are already rejected by classify_arg, so an unhandled kind here
    // is an internal logic error (a new PythonArgKind, or a tuple kind that leaked through).
    CHECK(false);
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
    for (size_t i = 0; i < arg_kinds.size(); ++i) {
        PythonArgKind kind = arg_kinds[i];
        if (!extract_arg(driver, pyarg_objs[i], kind, flat_param_annotations[i].get(), helper))
            return ErrorRaised;
    }
    return OK;
}

static PyPtr parse_element_constraint(ConstantCursor& cursor, ParameterKind kind,
                                      const LeafAnnotationNode& annotation) {
    switch (kind) {
    case ParameterKind::Array:
        return parse_array_constraint(cursor, annotation.array.static_shape_dims);
    case ParameterKind::Boolean:
        return parse_pybool_constraint(cursor, annotation.constant);
    case ParameterKind::Integer:
        return parse_pylong_constraint(cursor, annotation.constant, annotation.scalar.bitwidth);
    case ParameterKind::Float:
        return parse_pyfloat_constraint(cursor, annotation.constant);
    case ParameterKind::List:
        return parse_list_constraint(cursor, annotation.list.element.static_shape_dims);
    case ParameterKind::TupleBegin:
    case ParameterKind::TupleEnd:
        CHECK(false);  // Should be handled before parse_element_constraint
        return {};
    }
    CHECK(false);
    return {};
}

static PyPtr parse_param_constraint(ConstantCursor& cursor,
                                    Cursor<ParameterKind>* param_cursor,
                                    Cursor<RefPtr<LeafAnnotationNode>>* annotation_cursor) {
    ParameterKind pk = param_cursor->next();
    if (pk == ParameterKind::TupleBegin) {
        PyPtr items_list = steal(PyList_New(0));
        if (!items_list) return {};
        while (param_cursor->peek() != ParameterKind::TupleEnd) {
            PyPtr item = parse_param_constraint(cursor, param_cursor, annotation_cursor);
            if (!item) return {};
            if (PyList_Append(items_list.get(), item.get()) < 0) return {};
        }
        ParameterKind tuple_end = param_cursor->next();
        CHECK(tuple_end == ParameterKind::TupleEnd);

        PyObject* signature_module = get_signature_module();
        if (!signature_module) return {};
        PyPtr constraint_class = getattr(signature_module, "TupleConstraint");
        if (!constraint_class) return {};
        return steal(PyObject_CallOneArg(constraint_class.get(), items_list.get()));
    }
    LeafAnnotationNode* annotation = annotation_cursor->next().get();
    return parse_element_constraint(cursor, pk, *annotation);
}

static PyPtr parse_parameter_constraints(
        const Vec<int64_t>& constants,
        const Vec<ParameterKind>& param_kinds,
        const Vec<RefPtr<LeafAnnotationNode>>& flat_param_annotations) {
    ConstantCursor cursor(constants);
    PyPtr param_constraints = steal(PyList_New(0));
    if (!param_constraints) return {};

    Cursor<ParameterKind> param_cursor(param_kinds);
    Cursor<RefPtr<LeafAnnotationNode>> annotation_cursor(flat_param_annotations);
    while (param_cursor.len) {
        PyPtr constraint = parse_param_constraint(cursor, &param_cursor, &annotation_cursor);
        if (!constraint) return {};
        if (PyList_Append(param_constraints.get(), constraint.get()))
            return {};
    }
    CHECK(cursor.len == 0);
    CHECK(annotation_cursor.len == 0);
    return param_constraints;
}

static CallConvVersion minimum_calling_convention(
        const Vec<ParameterKind>& param_kinds,
        const Vec<RefPtr<LeafAnnotationNode>>& flat_param_annotations) {
    for (ParameterKind pk : param_kinds) {
        if (pk == ParameterKind::TupleBegin)
            return CallConvVersion::CutilePython_V2;
    }
    for (const RefPtr<LeafAnnotationNode>& f : flat_param_annotations) {
        if (!f->array.static_shape_dims.empty() || !f->list.element.static_shape_dims.empty())
            return CallConvVersion::CutilePython_V2;
    }
    return CallConvVersion::CutilePython_V1;
}

static PyPtr make_signature(const Vec<int64_t>& constants,
                            const Vec<ParameterKind>& param_kinds,
                            const Vec<RefPtr<LeafAnnotationNode>>& flat_param_annotations,
                            const PyPtr& calling_convention) {
    PyPtr parameters = parse_parameter_constraints(constants, param_kinds, flat_param_annotations);
    if (!parameters) return {};

    PyObject* signature_module = get_signature_module();
    if (!signature_module) return {};

    PyPtr signature_class = getattr(signature_module, "KernelSignature");
    if (!signature_class) return {};

    return steal(PyObject_CallFunctionObjArgs(
            signature_class.get(), parameters.get(), calling_convention.get(), nullptr));
}

using ProfileMap = HashMap<Vec<PyPtr>, PythonArgProfile>;

// Allow heterogeneous lookup for ProfileMap
template <>
struct CompareKey <Vec<PyTypeObject*>, Vec<PyPtr>> {
    static bool equals(const Vec<PyTypeObject*>& a, const Vec<PyPtr>& b) {
        size_t n = a.size();
        if (n != b.size()) return false;
        for (size_t i = 0; i < n; ++i) {
            if (reinterpret_cast<PyObject*>(a[i]) != b[i].get())
                return false;
        }
        return true;
    }
};

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
    X(CU_TENSOR_MAP_SWIZZLE_128B_ATOM_64B)

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

    ret.interleave = CU_TENSOR_MAP_INTERLEAVE_NONE;
    ret.l2_promotion = CU_TENSOR_MAP_L2_PROMOTION_NONE;
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


static Status flatten_pyargs(PyObject* const* pyargs, Py_ssize_t num_pyargs, int depth,
                             Vec<PyTypeObject*>& pyarg_types, Vec<PyObject*>& pyarg_objs) {
    for (Py_ssize_t i = 0; i < num_pyargs; ++i) {
        PyTypeObject* ty = Py_TYPE(pyargs[i]);
        pyarg_types.push_back(ty);
        if (ty == &PyTuple_Type) {
            // Tuple subclasses (e.g. namedtuple) are not exact tuples, so they fall through as
            // ordinary leaf types and are rejected later by classify_arg(). That keeps the more
            // expensive PyTuple_Check() off this per-argument hot path (run on every dispatch) and
            // on the cache-miss path instead.
            constexpr int kMaxTupleNestingDepth = 64;
            if (depth >= kMaxTupleNestingDepth)
                return raise(PyExc_RecursionError,
                             "Tuple argument nesting exceeds maximum depth of %d",
                             kMaxTupleNestingDepth);
            if (!flatten_pyargs(reinterpret_cast<PyTupleObject*>(pyargs[i])->ob_item,
                                PyTuple_GET_SIZE(pyargs[i]),
                                depth + 1,
                                pyarg_types, pyarg_objs))
                return ErrorRaised;
            pyarg_types.push_back(kTupleEndType);
        } else {
            pyarg_objs.push_back(pyargs[i]);
        }
    }
    return OK;
}

static Result<TileKernel> compile(const DriverApi* driver,
                                  PyObject* dispatcher_pyobj,
                                  PyObject* signature,
                                  PyObject* py_tile_context,
                                  KernelImage* image) {
    PyPtr compile_result = steal(PyObject_CallMethod(
            dispatcher_pyobj, "_compile", "(OO)",
            signature, py_tile_context));
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

static Result<CUstream> parse_stream(PyObject* py_stream) {
    PyPtr py_raw_stream;
    if (g_torch_cuda_Stream_type && PyObject_TypeCheck(py_stream, g_torch_cuda_Stream_type)) {
        py_raw_stream = getattr(py_stream, "cuda_stream");
        if (!py_raw_stream) return ErrorRaised;

    } else if (g_cupy_cuda_Stream_type && PyObject_TypeCheck(py_stream, g_cupy_cuda_Stream_type)) {
        py_raw_stream = getattr(py_stream, "ptr");
        if (!py_raw_stream) return ErrorRaised;

    } else if (g_numba_cuda_Stream_type
            && PyObject_TypeCheck(py_stream, g_numba_cuda_Stream_type)) {
        PyPtr py_stream_handle = getattr(py_stream, "handle");
        if (!py_stream_handle) return ErrorRaised;

        // numba-cuda >= 0.30: handle is cuda.bindings.driver.CUstream
        // numba-cuda < 0.30: handle is ctypes c_void_p
        if (g_cuda_bindings_CUstream_type
                && PyObject_TypeCheck(py_stream_handle.get(),
                                      g_cuda_bindings_CUstream_type)) {
            py_raw_stream = steal(PyNumber_Long(py_stream_handle.get()));
            if (!py_raw_stream) return ErrorRaised;
        } else {
            PyPtr py_stream_handle_value = getattr(py_stream_handle, "value");
            if (!py_stream_handle_value) return ErrorRaised;

            // numba stream.handle.value is None for default stream
            if (py_stream_handle_value.get() == Py_None)
                return static_cast<CUstream>(nullptr);

            if (PyLong_Check(py_stream_handle_value.get()))
                py_raw_stream = py_stream_handle_value;
        }

    } else if (PyLong_Check(py_stream)) {
        py_raw_stream = newref(py_stream);
    } else if (py_stream == Py_None) {
        return raise(PyExc_TypeError, "Stream is required, got None");
    } else {
        return raise(PyExc_TypeError, "Unsupported stream type %s.",
                     Py_TYPE(py_stream)->tp_name);
    }

    // TODO: support more stream types, for example, cuda.core.experimental._stream.Stream

    if (!PyLong_Check(py_raw_stream.get())) {
        return raise(PyExc_TypeError, "Raw stream pointer must be a long, got %s",
                     Py_TYPE(py_raw_stream.get())->tp_name);
    }

    CUstream stream = static_cast<CUstream>(PyLong_AsVoidPtr(py_raw_stream.get()));
    if (PyErr_Occurred()) return ErrorRaised;

    return stream;
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

static Status get_parameter_and_pyarg_kinds(const Vec<PyTypeObject*>& pyarg_types,
                                            const Vec<PyObject*>& pyarg_objs,
                                            Vec<ParameterKind>* param_kinds,
                                            Vec<PythonArgKind>* arg_kinds) {
    arg_kinds->clear();
    arg_kinds->reserve(pyarg_types.size());
    param_kinds->clear();
    param_kinds->reserve(pyarg_types.size());
    size_t obj_idx = 0;
    for (PyTypeObject* pyarg_type : pyarg_types) {
        if (pyarg_type == &PyTuple_Type) {
            param_kinds->push_back(ParameterKind::TupleBegin);
        } else if (pyarg_type == kTupleEndType) {
            param_kinds->push_back(ParameterKind::TupleEnd);
        } else {
            Result<PythonArgKind> kind = classify_arg(pyarg_objs[obj_idx++]);
            if (!kind.is_ok()) return ErrorRaised;
            arg_kinds->push_back(*kind);
            param_kinds->push_back(param_kind_from_pyarg_kind(*kind));
        }
    }
    return OK;
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

    if (!flatten_pyargs(pyargs, num_pyargs, 0, helper->pyarg_types, helper->pyarg_objs))
        return ErrorRaised;
    TileDispatcher& dispatcher = py_unwrap<TileDispatcher>(dispatcher_pyobj);
    TileContextDispatcher& ctx_dispatcher = dispatcher.default_context_dispatcher;
    ProfileMap::Item* profile_item = ctx_dispatcher.arg_profiles.find(helper->pyarg_types);
    if (!profile_item) {
        // Slower path
        if (static_cast<size_t>(num_pyargs) != dispatcher.param_annotations.size()) {
            return raise(PyExc_TypeError, "Kernel expects %zu arguments but %zd %s given",
                    dispatcher.param_annotations.size(), num_pyargs,
                    num_pyargs == 1 ? "was" : "were");
        }

        Vec<PythonArgKind> arg_kinds;
        Vec<ParameterKind> param_kinds;
        if (!get_parameter_and_pyarg_kinds(helper->pyarg_types, helper->pyarg_objs,
                                           &param_kinds, &arg_kinds))
            return ErrorRaised;

        KernelFamily* family = get_or_create_kernel_family(
                &ctx_dispatcher.kernel_families, std::move(param_kinds));

        // Flatten the parameter annotations against this argument structure.
        Result<Vec<RefPtr<LeafAnnotationNode>>> flat_param_annotations
               = flatten_parameter_annotation_nodes(dispatcher.param_annotations,
                                                    family->param_kinds);
        if (!flat_param_annotations.is_ok())
            return ErrorRaised;

        Vec<PyPtr> typeobj_refs;
        typeobj_refs.reserve(helper->pyarg_types.size());
        for (PyTypeObject* typeobj : helper->pyarg_types) {
            // kTupleEnd is a null sentinel, not a real type object: store an empty PyPtr
            // (which hashes/compares as null, matching the kTupleEnd entries in the lookup
            // key) rather than calling newref(nullptr), which would abort.
            PyObject* obj = reinterpret_cast<PyObject*>(typeobj);
            typeobj_refs.push_back(obj ? newref(obj) : PyPtr{});
        }

        profile_item = ctx_dispatcher.arg_profiles.insert(
                    std::move(typeobj_refs),
                    PythonArgProfile{newref(family), std::move(arg_kinds),
                                     std::move(*flat_param_annotations)});
    }

    if (!extract_cuda_args(driver, helper->pyarg_objs, profile_item->value.arg_kinds,
                           profile_item->value.flat_param_annotations, *helper)) {
        return ErrorRaised;
    }

    KernelMap& kernel_map = profile_item->value.family->kernels_by_constants;
    KernelMap::Item* kernel_item = kernel_map.find(helper->constants);
    std::optional<KernelImage> kernel_image;
    if (!kernel_item || capture_kernel_image) {
        PyPtr cconv = get_cconv(minimum_calling_convention(
                    profile_item->value.family->param_kinds,
                    profile_item->value.flat_param_annotations));
        if (!cconv) return ErrorRaised;

        PyPtr signature = make_signature(helper->constants,
                                         profile_item->value.family->param_kinds,
                                         profile_item->value.flat_param_annotations,
                                         cconv);
        if (!signature) return ErrorRaised;

        KernelImage* image = capture_kernel_image ? &kernel_image.emplace() : nullptr;
        Result<TileKernel> res = compile(driver, dispatcher_pyobj, signature.get(),
                                         g_default_tile_context, image);
        if (!res.is_ok()) return ErrorRaised;

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


// Parse a Python ArrayAnnotation into its C++ equivalent.
static Status parse_array_annotation(PyObject* py_array_annotation, ArrayAnnotation* dst) {
    if (py_array_annotation == Py_None)
        return OK;

    PyPtr index_dtype = getattr(py_array_annotation, "index_dtype");
    if (!index_dtype) return ErrorRaised;

    Result<unsigned> index_bitwidth_res = parse_int32_or_int64_dtype_as_bitwidth(index_dtype.get());
    if (!index_bitwidth_res.is_ok()) return ErrorRaised;

    dst->index_bitwidth = *index_bitwidth_res;

    PyPtr static_shape_dims = getattr(py_array_annotation, "static_shape_dims");
    if (!static_shape_dims) return ErrorRaised;

    if (!PyTuple_Check(static_shape_dims.get()))
        return raise(PyExc_TypeError, "`ArrayAnnotation.static_shape_dims` must be a tuple, got %s",
                     Py_TYPE(static_shape_dims.get())->tp_name);
    Py_ssize_t nd = PyTuple_GET_SIZE(static_shape_dims.get());

    dst->static_shape_dims.reserve(nd);
    for (Py_ssize_t i = 0; i < nd; ++i) {
        dst->static_shape_dims.push_back(
                pylong_as<int64_t>(PyTuple_GET_ITEM(static_shape_dims.get(), i)));
        if (PyErr_Occurred()) return ErrorRaised;
    }

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

    if (!flatten_pyargs(kernel_args, num_kernel_args, 0, helper->pyarg_types, helper->pyarg_objs))
        return nullptr;

    Vec<PythonArgKind> arg_kinds;
    Vec<ParameterKind> param_kinds;
    if (!get_parameter_and_pyarg_kinds(helper->pyarg_types, helper->pyarg_objs,
                                       &param_kinds, &arg_kinds))
        return nullptr;

    Result<const DriverApi*> driver = get_driver_api();
    if (!driver.is_ok()) return nullptr;

    Result<Vec<RefPtr<LeafAnnotationNode>>> flat_param_annotations
        = flatten_parameter_annotation_nodes(dispatcher.param_annotations, param_kinds);
    if (!flat_param_annotations.is_ok())
        return nullptr;

    if (!extract_cuda_args(*driver, helper->pyarg_objs, arg_kinds, *flat_param_annotations,
                           *helper)) {
        return nullptr;
    }

    return parse_parameter_constraints(
            helper->constants, param_kinds, *flat_param_annotations).release();
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

// Parse extra keyword arguments accepted by the extended launch api into
// launch attributes.
static Result<unsigned> parse_launch_kwargs(PyObject *const *args,
                                            Py_ssize_t nargs, PyObject *kwargs,
                                            CUlaunchAttribute launch_attrs[kMaxCUlaunchAttrs]) {
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
    const auto num_attrs = parse_launch_kwargs(args, nargs, kwargs, launch_attrs);
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

#define LAUNCH_SIGNATURE "launch(stream, grid, kernel, kernel_args, /)"

static PyObject* cuda_tile_launch(PyObject*, PyObject* const* args, Py_ssize_t nargs) {
  return launch_impl(args, nargs, nullptr, LAUNCH_SIGNATURE,
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


static Result<bool> py_environ_has(const char* name) {
    PyPtr os = steal(PyImport_ImportModule("os"));
    if (!os) return ErrorRaised;

    PyPtr py_environ = steal(PyObject_GetAttrString(os.get(), "environ"));
    if (!py_environ) return ErrorRaised;

    PyPtr value = steal(PyMapping_GetItemString(py_environ.get(), name));
    if (value) {
        return true;
    } else {
        if (PyErr_ExceptionMatches(PyExc_KeyError)) {
            PyErr_Clear();
            return false;
        }
        return ErrorRaised;
    }
}


static void try_get_torch_globals() {
    PyPtr torch = try_import("torch");
    if (!torch) return;

    // Save a reference to torch.Tensor
    if (PyPtr torch_Tensor = try_getattr(torch, "Tensor")) {
        if (PyType_Check(torch_Tensor.get()))
            g_torch_Tensor_type = reinterpret_cast<PyTypeObject*>(torch_Tensor.release());
    }

    // Save references to torch.cuda.Stream
    if (PyPtr torch_cuda = try_getattr(torch, "cuda")) {
        if (PyPtr torch_cuda_Stream = try_getattr(torch_cuda, "Stream")) {
            if (PyType_Check(torch_cuda_Stream.get())) {
                g_torch_cuda_Stream_type = reinterpret_cast<PyTypeObject*>(
                        torch_cuda_Stream.release());
            }
        }
    }

    // Save references to torch._C._to_dlpack
    if (PyPtr torch_C = try_getattr(torch, "_C")) {
        g_torch_to_dlpack_func = try_getattr(torch_C, "_to_dlpack").release();
    }


#define LOAD_TORCH_DTYPE_GLOBAL(name, bitwidth, lanes, typecode) \
    if (PyPtr dtype_ptr = try_getattr(torch, #name)) { \
        g_torch_dtype_##name = dtype_ptr.release(); \
    }

    FOREACH_TORCH_DTYPE(LOAD_TORCH_DTYPE_GLOBAL)
}

static void try_get_cupy_globals() {
    PyPtr cupy = try_import("cupy");
    if (!cupy) return;

    // Save references to cupy.cuda.Stream
    if (PyPtr cupy_cuda = try_getattr(cupy, "cuda")) {
        if (PyPtr cupy_cuda_Stream = try_getattr(cupy_cuda, "Stream")) {
            if (PyType_Check(cupy_cuda_Stream.get())) {
                g_cupy_cuda_Stream_type = reinterpret_cast<PyTypeObject*>(
                        cupy_cuda_Stream.release());
            }
        }
    }
}

static void try_get_numba_globals() {
    PyPtr numba_cuda = try_import("numba.cuda");
    if (!numba_cuda) return;

    // Save a reference to numba.cuda.driver.Stream
    if (PyPtr numba_cuda_driver = try_getattr(numba_cuda, "driver")) {
        if (PyPtr numba_cuda_Stream = try_getattr(numba_cuda_driver, "Stream")) {
            if (PyType_Check(numba_cuda_Stream.get())) {
                g_numba_cuda_Stream_type = reinterpret_cast<PyTypeObject*>(
                        numba_cuda_Stream.release());
            }
        }
    }
}

static void try_get_cuda_bindings_globals() {
    PyPtr cuda_bindings_driver = try_import("cuda.bindings.driver");
    if (!cuda_bindings_driver) return;

    if (PyPtr CUstream = try_getattr(cuda_bindings_driver, "CUstream")) {
        if (PyType_Check(CUstream.get())) {
            g_cuda_bindings_CUstream_type = reinterpret_cast<PyTypeObject*>(
                    CUstream.release());
        }
    }
}

static PyMethodDef functions[] = {
    {"launch", reinterpret_cast<PyCFunction>(cuda_tile_launch), METH_FASTCALL,
        LAUNCH_SIGNATURE "\n"
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

#define INIT_STRING_CONSTANT(ident) \
    if (!(g_##ident##_pyunicode = PyUnicode_InternFromString(#ident))) return ErrorRaised;

Status tile_kernel_init(PyObject* m) {
    INIT_STRING_CONSTANT(__cuda_array_interface__);
    INIT_STRING_CONSTANT(typestr);
    INIT_STRING_CONSTANT(shape);
    INIT_STRING_CONSTANT(data);
    INIT_STRING_CONSTANT(strides);
    INIT_STRING_CONSTANT(__dlpack__);
    INIT_STRING_CONSTANT(compile);
    INIT_STRING_CONSTANT(dynamic_shared_memory_bytes);
    INIT_STRING_CONSTANT(cooperative);
    INIT_STRING_CONSTANT(block_in_cluster_count);
    INIT_STRING_CONSTANT(preferred_block_in_cluster_count);
    INIT_STRING_CONSTANT(programmatic_dependent_launch);

    g_stream_buffer_pool_by_ctx_id = new StreamBufferPoolMap();

    Result<bool> is_ipc_benchmark_worker = py_environ_has(
            "CUDA_TILE_IPC_BENCHMARK_WORKER");
    if (!is_ipc_benchmark_worker.is_ok()) return ErrorRaised;

    // The IPC benchmark worker receives an already serialized launch and does
    // not inspect Python framework objects. Avoid importing several large,
    // optional framework packages in that short-lived helper process.
    if (!*is_ipc_benchmark_worker) {
        try_get_torch_globals();

        try_get_cupy_globals();

        try_get_numba_globals();

        try_get_cuda_bindings_globals();
    }

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
