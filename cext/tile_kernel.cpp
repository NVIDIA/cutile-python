// SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0

#include "tile_kernel.h"

#include "check.h"
#include "cuda_loader.h"
#include "cuda_helper.h"
#include "hash_map.h"
#include "ref_ptr.h"
#include "stream_buffer.h"
#include "vec.h"

#include <cuda.h>
#include <dlpack.h>

#include <memory>
#include <algorithm>
#include <utility>

static PyObject* g___cuda_array_interface___pyunicode;
static PyObject* g_typestr_pyunicode;
static PyObject* g_shape_pyunicode;
static PyObject* g_data_pyunicode;
static PyObject* g_strides_pyunicode;
static PyObject* g___dlpack___pyunicode;
static PyObject* g_compile_pyunicode;

static PyTypeObject* g_torch_Tensor_type;
static PyTypeObject* g_torch_cuda_Stream_type;
static PyObject* g_torch_to_dlpack_func;

static PyObject* g_default_tile_context;

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


static PyTypeObject* g_cupy_ndarray_type;
static PyTypeObject* g_cupy_cuda_Stream_type;

static PyTypeObject* g_numba_cuda_Stream_type;

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


// RAII wrapper around CUlibrary
namespace { class CudaLibrary {
public:
    explicit CudaLibrary(const DriverApi* driver, CUlibrary lib) : driver_(driver), lib_(lib) {}

    CudaLibrary(CudaLibrary&& other) : driver_(other.driver_), lib_(other.lib_) {
        other.lib_ = nullptr;
    }

    CudaLibrary(const CudaLibrary&) = delete;
    void operator=(const CudaLibrary&) = delete;

    ~CudaLibrary() {
        if (lib_) {
            CUresult res = driver_->cuLibraryUnload(lib_);
            CHECK(res == CUDA_SUCCESS);
        }
    }

    const CUlibrary& get() const {
        return lib_;
    }

private:
    const DriverApi* driver_;
    CUlibrary lib_;
}; }

static Result<CudaLibrary> load_cuda_library(const DriverApi* driver, const char* filename) {
    CUlibrary lib;
    CUresult res = driver->cuLibraryLoadFromFile(&lib, filename, nullptr, nullptr, 0,
                                           nullptr, nullptr, 0);
    if (res == CUDA_SUCCESS)
        return CudaLibrary(driver, lib);

    return raise(PyExc_RuntimeError, "Failed to load CUDA library from %s: %s",
                 filename, get_cuda_error(driver, res));
}

struct CudaKernel {
    CudaLibrary lib;
    CUkernel kernel;
};

static Result<CudaKernel> load_cuda_kernel(const DriverApi* driver,
                                           const char* filename,
                                           const char* func_name) {
    Result<CudaLibrary> lib = load_cuda_library(driver, filename);
    if (!lib.is_ok()) return ErrorRaised;

    CUkernel kernel;
    CUresult res = driver->cuLibraryGetKernel(&kernel, lib->get(), func_name);
    if (res == CUDA_SUCCESS)
        return CudaKernel{std::move(*lib), kernel};

    return raise(PyExc_RuntimeError, "Failed to get kernel %s from library %s: %s",
                 func_name, filename, get_cuda_error(driver, res));
}


static inline void hash_combine(size_t& h, size_t other) {
    h ^= other + 0x9e3779b9 + (h << 6) + (h >> 2);
}

template <typename T>
struct HashVector {
    size_t operator() (const Vec<T>& v) const {
        size_t ret = 0;
        const std::hash<T> elem_hash;
        for (const T& x : v)
            hash_combine(ret, elem_hash(x));
        return ret;
    }
};

struct TileKernel {
    CudaKernel cukernel;
};

struct KernelFamily : SimpleRefcount<KernelFamily> {
    using KernelMap = HashMap<Vec<int64_t>, TileKernel>;

    KernelMap kernels_by_constants;
};

union Word {
    void* device_ptr;
    int32_t i32;
    int64_t i64;
    float f32;
};

static_assert(sizeof(Word) == 8);

struct ListArg {
    size_t cuarg_idx;  // offset into LaunchHelper.cuargs
    size_t offset;  // offset into LaunchHelper.cuargs.nested_arrays
};

struct LaunchHelper {
    Vec<PyTypeObject*> pyarg_types;
    Vec<Word> cuargs;
    Vec<Word> nested_arrays;
    Vec<ListArg> list_args;
    Vec<void*> cuarg_pointers;
    Vec<int64_t> constants;
    CUcontext cuda_context;
    LaunchHelper* next_free;
};

static LaunchHelper* g_helper_freelist;  // protected by the GIL

namespace { struct LaunchHelperDeleter {
    void operator() (LaunchHelper* helper) const {
        helper->next_free = g_helper_freelist;
        g_helper_freelist = helper;
    }
}; }

using LaunchHelperPtr = std::unique_ptr<LaunchHelper, LaunchHelperDeleter>;


static LaunchHelperPtr launch_helper_get() {
    if (g_helper_freelist) {
        LaunchHelper* ret = g_helper_freelist;
        g_helper_freelist = ret->next_free;
        return LaunchHelperPtr(ret);
    } else {
        return LaunchHelperPtr(new LaunchHelper());
    }
}

enum class PythonArgKind {
    // A torch.Tensor that we can access via torch._C._to_dlpack
    TorchTensorDlpack,
    // An object with __dlpack__ method
    DlpackArray,
    // An object with __cuda_array_interface__
    CudaArray,
    // Python `int`,
    PyLong,
    // Python `float`
    PyFloat,
    // Python `list`
    PyList
};

enum class ParameterKind {
    Array,
    Integer,
    Float,
    List,
};

static Result<std::pair<PythonArgKind, ParameterKind>> classify_arg(PyObject* arg) {
    if (PyLong_Check(arg))
        return {{PythonArgKind::PyLong, ParameterKind::Integer}};

    if (PyFloat_Check(arg))
        return {{PythonArgKind::PyFloat, ParameterKind::Float}};

    if (PyList_Check(arg))
        return {{PythonArgKind::PyList, ParameterKind::List}};

    if (g_torch_Tensor_type && PyObject_TypeCheck(arg, g_torch_Tensor_type)) {
        // Calling torch._C._to_dlpack(arg) is much faster than calling arg.__dlpack__()
        // because it goes straight into C++ code, with no Python in between.
        // So we always prefer that.
        if (g_torch_to_dlpack_func)
            return {{PythonArgKind::TorchTensorDlpack, ParameterKind::Array}};
    }

    if (PyObject_HasAttr(arg, g___dlpack___pyunicode))
        return {{PythonArgKind::DlpackArray, ParameterKind::Array}};

    if (PyObject_HasAttr(arg, g___cuda_array_interface___pyunicode))
        return {{PythonArgKind::CudaArray, ParameterKind::Array}};

    PyErr_Format(PyExc_TypeError, "Unsupported argument type %s", Py_TYPE(arg)->tp_name);
    return ErrorRaised;
}


struct PythonArgProfile {
    RefPtr<KernelFamily> family;
    Vec<PythonArgKind> arg_kinds;
};

// Concatenate values of two chars in a single unsigned integer
#define CHAR_PAIR(x, y) \
    (\
        (static_cast<unsigned>(static_cast<unsigned char>((x))) << 16) \
        | (static_cast<unsigned>(static_cast<unsigned char>((y)))) \
    )


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
    switch (CHAR_PAIR(str[2], str[3])) {
    case CHAR_PAIR('1', '\0'): ret.bits = 8; break;
    case CHAR_PAIR('2', '\0'): ret.bits = 16; break;
    case CHAR_PAIR('4', '\0'): ret.bits = 32; break;
    case CHAR_PAIR('8', '\0'): ret.bits = 64; break;
    case CHAR_PAIR('1', '6'):
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
};

// This should compile to a no-op
static inline uint32_t dtype_as_uint(DLDataType dtype) {
    return static_cast<uint32_t>(dtype.code)
        | (static_cast<uint32_t>(dtype.bits) << 8)
        | (static_cast<uint32_t>(dtype.lanes) << 16);
}

// Pack data type and array rank in a single int64_t so it could be used
// as a single constant for looking up the kernel in a family
static int64_t pack_array_type(ArrayType a) {
    uint64_t dtype_u = static_cast<uint64_t>(dtype_as_uint(a.dtype));
    return static_cast<int64_t>(dtype_u | (static_cast<uint64_t>(a.ndim) << 32));
}

static Status extract_compact_row_major_strides(size_t ndim, size_t shape_offset, Vec<Word>& dst) {
    if (ndim == 0) return OK;

    dst.resize(dst.size() + ndim);
    size_t stride_idx = dst.size();
    size_t shape_idx = shape_offset + ndim;
    uint64_t prev_stride = 1;
    dst[--stride_idx].i32 = 1;

    for (size_t i = 0; i < ndim - 1; ++i) {
        uint64_t new_stride = prev_stride * static_cast<uint64_t>(dst[--shape_idx].i32);
        if (new_stride > INT32_MAX)
            return raise(PyExc_OverflowError, "stride is too big");
        dst[--stride_idx].i32 = new_stride;
        prev_stride = new_stride;
    }
    return OK;
}

static ArraySpecializationBits compute_array_specialization_bits(
    void* data_ptr, size_t ndim, int32_t dtype_bitwidth, const Word* shape_stride) {

    ArraySpecializationBits ret = {};

    const Word* strides = shape_stride + ndim;

    // Only specialize stride divisibility, stride 1 and shape divisibility for ndim <= TMA_MAX_NDIM
    if (ndim <= TMA_MAX_NDIM) {
        for (size_t i = 0; i < ndim; ++i) {
            int32_t stride = strides[i].i32;
            int32_t shape = shape_stride[i].i32;
            int64_t stride_bitwidth = stride * dtype_bitwidth;
            int64_t shape_bitwidth = shape * dtype_bitwidth;
            bool is_stride_byte_aligned = stride_bitwidth % BYTE_BITWIDTH == 0;
            bool is_stride_16_byte_divisible =
                    (stride_bitwidth / BYTE_BITWIDTH) % DIVISOR_16 == 0;
            bool is_shape_byte_aligned = shape_bitwidth % BYTE_BITWIDTH == 0;
            bool is_shape_divisible_by_16 = shape % DIVISOR_16 == 0;

            if (is_stride_byte_aligned && is_stride_16_byte_divisible)
                ret.stride_16byte_divisible |= 1u << i;

            if (stride == 1)
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
    Vec<std::pair<int32_t, size_t>> strides_and_shape(ndim);
    for (size_t i = 0; i < ndim; ++i) {
        strides_and_shape[i] = {strides[i].i32, shape_stride[i].i32};
    }
    std::sort(strides_and_shape.begin(), strides_and_shape.end());

    // disjointness check:
    // - 0 dimension array elements are always disjoint.
    // - >0 dimension array elements are disjoint if every stride is positive
    //    and greater than or equal to the product of the previous stride and
    //    the previous shape.
    bool elems_disjoint = (ndim == 0) || (strides_and_shape[0].first > 0);
    for (size_t i = 0; i + 1 < ndim; ++i) {
        int32_t prev_stride = strides_and_shape[i].first;
        int32_t prev_shape = strides_and_shape[i].second;
        int32_t cur_stride = strides_and_shape[i + 1].first;
        elems_disjoint &= (
            cur_stride > 0 && cur_stride >= prev_stride * prev_shape);
    }
    ret.disjoint_elements = elems_disjoint;

    return ret;
}

static void extract_array_specialization_constants(const DriverApi* driver,
                                                   ArrayType arrtype,
                                                   const Word* array_repr,
                                                   size_t num_arrays,
                                                   LaunchHelper& helper) {
    CHECK(num_arrays >= 1);
    helper.constants.push_back(pack_array_type(arrtype));

    if (!helper.cuda_context) {
        void* first_data_ptr = array_repr[0].device_ptr;
        driver->cuPointerGetAttribute(&helper.cuda_context, CU_POINTER_ATTRIBUTE_CONTEXT,
                reinterpret_cast<CUdeviceptr>(first_data_ptr));
    }

    uint64_t special_bits = ~static_cast<uint64_t>(0);
    size_t repr_len = 1 + 2 * arrtype.ndim;
    for (size_t i = 0; i < num_arrays; ++i) {
        void* data_ptr = array_repr[0].device_ptr;
        special_bits &= compute_array_specialization_bits(
                data_ptr, arrtype.ndim, arrtype.dtype.bits * arrtype.dtype.lanes,
                array_repr + 1).u64;
        array_repr += repr_len;
    }

    helper.constants.push_back(special_bits);
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
    if (static_cast<uintmax_t>(ndim) > UINT32_MAX) \
        return raise(PyExc_TypeError, "Input array exceeds max supported dimensions: %ld > %u", \
                     ndim, UINT32_MAX);


static Status arrayrepr_cuda_array_iface(PyObject* pyobj, Vec<Word>& dst, ArrayType& arrtype) {
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
    dst.push_back({.device_ptr = reinterpret_cast<void*>(data_ptr_int)});

    Py_ssize_t ndim = PyTuple_GET_SIZE(shape);
    ASSERT_NDIM(ndim);

    // Parse the shape
    if (!PyTuple_Check(shape))
        return raise(PyExc_TypeError, "__cuda_array_interface['shape'] is not a tuple");

    size_t shape_offset = dst.size();
    for (Py_ssize_t i = 0; i < ndim; ++i) {
        int32_t size = pylong_as<int32_t>(PyTuple_GET_ITEM(shape, i));
        if (PyErr_Occurred()) return ErrorRaised;
        dst.push_back({.i32 = size});
    }

    // Parse the strides
    PyObject* strides = PyDict_GetItem(dict.get(), g_strides_pyunicode);
    if (PyErr_Occurred()) return ErrorRaised;
    if (!strides || strides == Py_None) {
        if (!extract_compact_row_major_strides(ndim, shape_offset, dst))
            return ErrorRaised;
    } else if (PyTuple_Check(strides)) {
        // Only byte-aligned types should be supported by __cuda_array_interface__
        uint8_t dtype_bytewidth = dtype->bits / BYTE_BITWIDTH;
        for (Py_ssize_t i = 0; i < ndim; ++i) {
            int32_t stride = pylong_as<int32_t>(PyTuple_GET_ITEM(strides, i));
            if (PyErr_Occurred()) return ErrorRaised;
            dst.push_back(
                    {.i32 = static_cast<int32_t>(stride / dtype_bytewidth)});
        }
    } else {
        return raise(PyExc_TypeError, "__cuda_array_interface['strides'] can only be"
                                      " absent, None, or a tuple");
    }

    arrtype.dtype = *dtype;
    arrtype.ndim = ndim;
    return OK;
}

static Status arrayrepr_dlpack_common(PyObject* dlpack_capsule, Vec<Word>& dst,
                                      ArrayType& arrtype) {
    void* ptr = PyCapsule_GetPointer(dlpack_capsule, "dltensor");
    if (!ptr) return ErrorRaised;
    DLManagedTensor* tensor = static_cast<DLManagedTensor*>(ptr);

    if (tensor->dl_tensor.device.device_type != kDLCUDA)
        return raise(PyExc_ValueError, "Input array is not on a CUDA device");

    // TODO: check device ID

    void* data_ptr = static_cast<char*>(tensor->dl_tensor.data) + tensor->dl_tensor.byte_offset;
    dst.push_back({.device_ptr = data_ptr});

    int32_t ndim = tensor->dl_tensor.ndim;
    ASSERT_NDIM(ndim);

    size_t shape_offset = dst.size();
    for (int32_t i = 0; i < ndim; ++i) {
        if (tensor->dl_tensor.shape[i] < INT32_MIN || tensor->dl_tensor.shape[i] > INT32_MAX)
            return raise(PyExc_OverflowError, "shape is too big");
        dst.push_back({.i32 = static_cast<int32_t>(tensor->dl_tensor.shape[i])});
    }

    if (!tensor->dl_tensor.strides) {
        if (!extract_compact_row_major_strides(ndim, shape_offset, dst))
            return ErrorRaised;
    } else {
        for (int32_t i = 0; i < ndim; ++i) {
            if(tensor->dl_tensor.strides[i] < INT32_MIN || tensor->dl_tensor.strides[i] > INT32_MAX)
                return raise(PyExc_OverflowError, "stride is too big");
            dst.push_back(
                    {.i32 = static_cast<int32_t>(tensor->dl_tensor.strides[i])});
        }
    }

    arrtype.dtype = tensor->dl_tensor.dtype;
    arrtype.ndim = ndim;

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
    return OK;
}


#define TORCH_DTYPE_TO_DL_DTYPE(name, bitwidth, lanes, typecode) \
if (torch_dtype == g_torch_dtype_##name) { \
    out = DLDataType{typecode, bitwidth, lanes}; \
    return OK; \
}

static Status dtype_from_torch_dtype(PyObject* torch_dtype, DLDataType& out) {
    FOREACH_TORCH_DTYPE(TORCH_DTYPE_TO_DL_DTYPE)
    return raise(PyExc_TypeError, "dtype is not supported");
}

static Status arrayrepr_torch_tensor_pymethod(PyObject* tensor,
        Vec<Word>& dst, ArrayType& arrtype) {
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
    dst.push_back({.device_ptr=(void*)addr});

    // Extract shape
    if (!PyTuple_Check(shape_ptr.get()))
        return raise(PyExc_TypeError, "expect shape to be an tuple");
    Py_ssize_t len = PyTuple_GET_SIZE(shape_ptr.get());
    if (len == -1) return ErrorRaised;
    if (len < INT32_MIN || len > INT32_MAX)
        return raise(PyExc_OverflowError, "rank is too big");
    int32_t ndim = (int32_t)(len);
    ASSERT_NDIM(ndim);

    for (int32_t i = 0; i < ndim; ++i) {
        PyObject* item_ptr = PyTuple_GetItem(shape_ptr.get(), i);
        if (!item_ptr) return ErrorRaised;
        if (!PyLong_Check(item_ptr))
            return raise(PyExc_TypeError, "unexpected type from .shape");
        long long si = PyLong_AsLongLong(item_ptr);
        if (PyErr_Occurred()) return ErrorRaised;
        if (si < INT32_MIN || si > INT32_MAX)
            return raise(PyExc_OverflowError, "shape is too big");
        dst.push_back({.i32 = static_cast<int32_t>(si)});
    }

    // Extract stride
    if (!PyTuple_Check(stride_ptr.get()))
        return raise(PyExc_TypeError, "expect stride to be an tuple");
    Py_ssize_t stride_len = PyTuple_GET_SIZE(stride_ptr.get());
    if (stride_len == -1) return ErrorRaised;
    if (stride_len != ndim)
        return raise(PyExc_ValueError, "shape and stride have different length");

    for (int32_t i = 0; i < ndim; ++i) {
        PyObject* item_ptr = PyTuple_GetItem(stride_ptr.get(), i);
        if (!item_ptr) return ErrorRaised;
        if (!PyLong_Check(item_ptr))
            return raise(PyExc_TypeError, "unexpected type of .stride");
        long long si = PyLong_AsLongLong(item_ptr);
        if (PyErr_Occurred()) return ErrorRaised;
        if (si < INT32_MIN || si > INT32_MAX)
            return raise(PyExc_OverflowError, "stride is too big");
        dst.push_back({.i32 = static_cast<int32_t>(si)});
    }

    arrtype.ndim = ndim;
    return dtype_from_torch_dtype(dtype_ptr.get(), arrtype.dtype);
}

static Status arrayrepr_torch_tensor_dlpack(PyObject* pyobj, Vec<Word>& dst, ArrayType& arrtype) {
    PyPtr dlpack_capsule = steal(PyObject_CallFunctionObjArgs(
                g_torch_to_dlpack_func, pyobj, nullptr));

    if (!dlpack_capsule) {
        SavedException exc = save_raised_exception();
        LOG_PYTHON_ERROR("debug", exc, "Fail to convert to dlpack, use fallback path");
        return arrayrepr_torch_tensor_pymethod(pyobj, dst, arrtype);
    }

    return arrayrepr_dlpack_common(dlpack_capsule.get(), dst, arrtype);
}

static Status arrayrepr_dlpack(PyObject* pyobj, Vec<Word>& dst, ArrayType& arrtype) {
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

    return arrayrepr_dlpack_common(dlpack_capsule.get(), dst, arrtype);
}


typedef Status(*ArrayReprFunc)(PyObject*, Vec<Word>&, ArrayType&);


template <ArrayReprFunc F>
static Status extract_array(const DriverApi* driver, PyObject* pyobj, LaunchHelper& helper) {
    size_t offset = helper.cuargs.size();

    ArrayType arrtype;
    if (!F(pyobj, helper.cuargs, arrtype))
        return ErrorRaised;

    CHECK(helper.cuargs.size() == offset + 1 + 2 * arrtype.ndim);
    extract_array_specialization_constants(driver, arrtype, &helper.cuargs[offset], 1, helper);
    return OK;
}


static inline Status extract_py_long(PyObject* pyobj, bool is_constant, LaunchHelper& helper) {
    if (is_constant) {
        int overflow;
        long long value = PyLong_AsLongLongAndOverflow(pyobj, &overflow);
        if (PyErr_Occurred()) return ErrorRaised;
        if (overflow) {
            // TODO: support big values by extracting all digits
            helper.constants.push_back(1);
            unsigned long long uval = PyLong_AsUnsignedLongLong(pyobj);
            if (PyErr_Occurred()) return ErrorRaised;
            helper.constants.push_back(uval);
        } else {
            helper.constants.push_back(0);
            helper.constants.push_back(value);
        }
    } else {
        int32_t value = pylong_as<int32_t>(pyobj);
        if (PyErr_Occurred()) return ErrorRaised;
        helper.cuargs.push_back({.i32 = value});
    }
    return OK;
}

static void extract_py_float(PyObject* pyobj, bool is_constant, LaunchHelper& helper) {
    double value = PyFloat_AS_DOUBLE(pyobj);
    if (is_constant) {
        int64_t i64_val = 0;
        static_assert(sizeof(i64_val) == sizeof(value));
        mem_copy(&i64_val, &value, sizeof(i64_val));
        helper.constants.push_back(i64_val);
    } else {
        helper.cuargs.push_back({.f32 = static_cast<float>(value)});
    }
}

static Status get_array_repr(PythonArgKind kind, PyObject* pyobj, Vec<Word>& dst,
                             ArrayType& arrtype) {
    switch (kind) {
        case PythonArgKind::TorchTensorDlpack:
            return arrayrepr_torch_tensor_dlpack(pyobj, dst, arrtype);
        case PythonArgKind::DlpackArray:
            return arrayrepr_dlpack(pyobj, dst, arrtype);
        case PythonArgKind::CudaArray:
            return arrayrepr_cuda_array_iface(pyobj, dst, arrtype);
        default:
            return raise(PyExc_AssertionError, "Unexpected argument kind for array: %d",
                         static_cast<int>(kind));
    }
}

static Status extract_py_list(const DriverApi* driver, PyObject* pyobj, LaunchHelper& helper) {
    size_t len = PyList_GET_SIZE(pyobj);
    if (len > INT32_MAX)
        return raise(PyExc_TypeError, "List is too long");

    // TODO: support empty list as its own type?
    if (!len)
        return raise(PyExc_TypeError, "Empty lists are not supported as kernel arguments");

    helper.list_args.push_back({.cuarg_idx = helper.cuargs.size(),
                                .offset = helper.nested_arrays.size()});
    helper.cuargs.push_back({.device_ptr = nullptr});  // to be filled later based on list_args
    helper.cuargs.push_back({.i32 = static_cast<int32_t>(len)});

    // Handle the first item separately in order to determine the item type

    PyObject* first_item = PyList_GET_ITEM(pyobj, 0);
    Result<std::pair<PythonArgKind, ParameterKind>> first_item_res = classify_arg(first_item);
    if (!first_item_res.is_ok()) return ErrorRaised;

    if (first_item_res->second != ParameterKind::Array) {
        return raise(PyExc_TypeError, "Expected list items to be arrays, got %s",
                     Py_TYPE(first_item)->tp_name);
    }

    PythonArgKind first_arg_kind = first_item_res->first;
    PyTypeObject* first_item_type = first_item->ob_type;

    size_t offset = helper.nested_arrays.size();
    ArrayType first_arrtype;
    if (!get_array_repr(first_arg_kind, first_item, helper.nested_arrays, first_arrtype))
        return ErrorRaised;

    // Handle the rest of the list
    for (size_t i = 1; i < len; ++i) {
        PyObject* item = PyList_GET_ITEM(pyobj, i);
        PythonArgKind kind = first_arg_kind;

        // Avoid calling classify_arg() if the object type is the same
        if (first_item_type != item->ob_type) {
             Result<std::pair<PythonArgKind, ParameterKind>> res = classify_arg(item);
             if (!res.is_ok()) return ErrorRaised;
             kind = res->first;
        }

        ArrayType arrtype;
        if (!get_array_repr(kind, item, helper.nested_arrays, arrtype))
            return ErrorRaised;

        // TODO: nicer error messages
        if (dtype_as_uint(first_arrtype.dtype) != dtype_as_uint(arrtype.dtype))
            return raise(PyExc_TypeError, "Arrays in list vary in data type");
        if (first_arrtype.ndim != arrtype.ndim)
            return raise(PyExc_TypeError, "Arrays in list vary in rank");
    }

    // TODO: If we accept lists of things other than arrays, then to disambiguate,
    //       we need to push another constant here that specifies the type of the list element .
    CHECK(helper.nested_arrays.size() == offset + len * (1 + 2 * first_arrtype.ndim));
    extract_array_specialization_constants(driver, first_arrtype,
                                           &helper.nested_arrays[offset], len,
                                           helper);
    return OK;
}

static Status extract_cuda_args(const DriverApi* driver,
                                PyObject* const* pyargs, size_t num_pyargs,
                                const Vec<PythonArgKind>& arg_kinds,
                                const Vec<bool>& constant_arg_flags,
                                LaunchHelper& helper) {
    CHECK(num_pyargs == arg_kinds.size());
    helper.cuargs.clear();
    helper.nested_arrays.clear();
    helper.list_args.clear();
    helper.constants.clear();
    for (size_t i = 0; i < num_pyargs; ++i) {
        PyObject* pyobj = pyargs[i];
        bool is_constant = constant_arg_flags[i];
        // TODO: apply is_constant to array args?

        switch (arg_kinds[i]) {
        case PythonArgKind::TorchTensorDlpack:
            if (!extract_array<arrayrepr_torch_tensor_dlpack>(driver, pyobj, helper))
                return ErrorRaised;
            break;
        case PythonArgKind::DlpackArray:
            if (!extract_array<arrayrepr_dlpack>(driver, pyobj, helper))
                return ErrorRaised;
            break;
        case PythonArgKind::CudaArray:
            if (!extract_array<arrayrepr_cuda_array_iface>(driver, pyobj, helper))
                return ErrorRaised;
            break;
        case PythonArgKind::PyLong:
            if (!extract_py_long(pyobj, is_constant, helper)) return ErrorRaised;
            break;
        case PythonArgKind::PyFloat:
            extract_py_float(pyobj, is_constant, helper);
            break;
        case PythonArgKind::PyList:
            if (!extract_py_list(driver, pyobj, helper)) return ErrorRaised;
            break;
        }
    }
    return OK;
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

struct TileContext {
    PyPtr config;
    PyPtr autotune_cache;
};


struct TileContextDispatcher {
    using FamilyMap = HashMap<Vec<ParameterKind>, RefPtr<KernelFamily>>;
    ProfileMap arg_profiles;
    FamilyMap kernel_families;
};


struct TileDispatcher {
    Vec<bool> constant_arg_flags;
    PyPtr compile_func;
    TileContextDispatcher default_context_dispatcher;
};


static void get_pyarg_types(PyObject* const* pyargs, Py_ssize_t num_pyargs,
                            Vec<PyTypeObject*>& pyarg_types) {
    pyarg_types.clear();
    for (Py_ssize_t i = 0; i < num_pyargs; ++i)
        pyarg_types.push_back(Py_TYPE(pyargs[i]));
}

static Result<TileKernel> compile(const DriverApi* driver,
                                  TileDispatcher& dispatcher,
                                  PyObject* const* pyargs,
                                  Py_ssize_t num_pyargs,
                                  PyObject* py_tile_context) {
    PyPtr pyargs_tuple = steal(PyTuple_New(num_pyargs));
    if (!pyargs_tuple) return ErrorRaised;

    for (Py_ssize_t i = 0; i < num_pyargs; ++i)
        PyTuple_SET_ITEM(pyargs_tuple.get(), i, Py_NewRef(pyargs[i]));

    PyPtr compile_result = steal(PyObject_CallFunctionObjArgs(
            dispatcher.compile_func.get(), pyargs_tuple.get(), py_tile_context, nullptr));
    if (!compile_result) return ErrorRaised;

    if (!PyTuple_Check(compile_result.get()))
        return raise(PyExc_TypeError, "Expected compile() to return a tuple, got %s",
                     Py_TYPE(compile_result.get())->tp_name);

    if (PyTuple_GET_SIZE(compile_result.get()) != 2)
        return raise(PyExc_TypeError, "Expected compile() to return a 2-tuple, got length %zd",
                     PyTuple_GET_SIZE(compile_result.get()));

    PyObject* py_cubin_filename = PyTuple_GET_ITEM(compile_result.get(), 0);
    PyObject* py_cufunc_name = PyTuple_GET_ITEM(compile_result.get(), 1);

    if (!PyUnicode_Check(py_cubin_filename) || !PyUnicode_Check(py_cufunc_name))
        return raise(PyExc_TypeError,
                     "Expected compile() to return two strings (cubin_filename, func_name),"
                     " got %s, %s",
                     Py_TYPE(py_cubin_filename)->tp_name,
                     Py_TYPE(py_cufunc_name)->tp_name);

    const char* cubin_filename = PyUnicode_AsUTF8(py_cubin_filename);
    if (!cubin_filename) return ErrorRaised;

    const char* cufunc_name = PyUnicode_AsUTF8(py_cufunc_name);
    if (!cufunc_name) return ErrorRaised;

    Result<CudaKernel> cukernel = load_cuda_kernel(driver, cubin_filename, cufunc_name);
    if (!cukernel.is_ok()) return ErrorRaised;

    return TileKernel{std::move(*cukernel)};
}

static inline bool has_torch_tensor_input(const Vec<PyTypeObject*>& pyarg_types) {
    return std::any_of(pyarg_types.begin(), pyarg_types.end(), [](PyTypeObject* pytype) {
        return PyType_IsSubtype(pytype, g_torch_Tensor_type);
    });
}

static inline bool has_cupy_array_input(const Vec<PyTypeObject*>& pyarg_types) {
    return std::any_of(pyarg_types.begin(), pyarg_types.end(), [](PyTypeObject* pytype) {
        return PyType_IsSubtype(pytype, g_cupy_ndarray_type);
    });
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
        PyPtr py_stream_handle_value = getattr(py_stream_handle, "value");
        if (!py_stream_handle_value) return ErrorRaised;

        // numba stream.handle.value is None for default stream
        if (py_stream_handle_value.get() == Py_None)
            return static_cast<CUstream>(nullptr);

        if (PyLong_Check(py_stream_handle_value.get()))
            py_raw_stream = py_stream_handle_value;

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

// Protected by GIL.
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

    StreamBufferPool** pp = g_stream_buffer_pool_by_ctx_id->find(ctx_id);
    if (pp) {
        return *pp;
    } else {
        StreamBufferPool* pool = stream_buffer_pool_new();
        g_stream_buffer_pool_by_ctx_id->insert(ctx_id, pool);
        return pool;
    }
}

namespace { struct ContextGuard {
    bool need_to_pop;
    const DriverApi* driver_;

    ContextGuard(const DriverApi* driver) : need_to_pop(false), driver_(driver) {}

    ContextGuard() = delete;
    ContextGuard(const ContextGuard&) = delete;
    void operator=(const ContextGuard&) = delete;

    ~ContextGuard() {
        if (need_to_pop) {
            CUcontext old;
            CUresult res = driver_->cuCtxPopCurrent(&old);
            CHECK(res == CUDA_SUCCESS);
        }
    }
}; }

static Status maybe_switch_context(const DriverApi* driver, CUcontext target, ContextGuard& guard) {
    if (!target) return OK;

    CUcontext current;
    CUresult res = driver->cuCtxGetCurrent(&current);
    if (res != CUDA_SUCCESS) {
        return raise(PyExc_RuntimeError, "Failed to get current CUDA context: %s",
                     get_cuda_error(driver, res));
    }

    if (current == target) return OK;

    res = driver->cuCtxPushCurrent(target);
    if (res != CUDA_SUCCESS) {
        return raise(PyExc_RuntimeError, "Failed to switch CUDA context: %s",
                     get_cuda_error(driver, res));
    }

    guard.need_to_pop = true;
    return OK;
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

static Status launch(const DriverApi* driver,
                     PyObject* dispatcher_pyobj,
                     Grid grid,
                     CUstream launch_stream,
                     PyObject* const* pyargs,
                     Py_ssize_t num_pyargs) {
    if (!validate_grid(grid)) return ErrorRaised;

    LaunchHelperPtr helper = launch_helper_get();
    get_pyarg_types(pyargs, num_pyargs, helper->pyarg_types);
    {
        CUresult res = driver->cuStreamGetCtx(launch_stream, &helper->cuda_context);
        // INVALID_CONTEXT can happen when it is NULL stream and there is
        // no active context in current thread. We will still get the context
        // from the array arguments later during `extract_cuda_args`.
        if (res != CUDA_SUCCESS && res != CUDA_ERROR_INVALID_CONTEXT) {
            return raise(PyExc_RuntimeError, "Failed to get a CUDA context from a stream: %s",
                         get_cuda_error(driver, res));
        }
    }

    TileDispatcher& dispatcher = py_unwrap<TileDispatcher>(dispatcher_pyobj);
    TileContextDispatcher& ctx_dispatcher = dispatcher.default_context_dispatcher;
    PythonArgProfile* profile = ctx_dispatcher.arg_profiles.find(helper->pyarg_types);
    if (!profile) {
        // Slower path

        Vec<PythonArgKind> arg_kinds;
        arg_kinds.reserve(num_pyargs);
        Vec<ParameterKind> param_kinds;
        param_kinds.reserve(num_pyargs);
        for (Py_ssize_t i = 0; i < num_pyargs; ++i) {
            Result<std::pair<PythonArgKind, ParameterKind>> c = classify_arg(pyargs[i]);
            if (!c.is_ok()) return ErrorRaised;
            arg_kinds.push_back(c->first);
            param_kinds.push_back(c->second);
        }

        RefPtr<KernelFamily>* family_pp = ctx_dispatcher.kernel_families.find(param_kinds);
        if (!family_pp) {
            RefPtr<KernelFamily> new_family = steal(new KernelFamily());
            family_pp = ctx_dispatcher.kernel_families.insert(
                    std::move(param_kinds), std::move(new_family));
        }

        Vec<PyPtr> typeobj_refs;
        typeobj_refs.reserve(helper->pyarg_types.size());
        for (PyTypeObject* typeobj : helper->pyarg_types)
            typeobj_refs.push_back(newref(reinterpret_cast<PyObject*>(typeobj)));

        profile = ctx_dispatcher.arg_profiles.insert(
                    std::move(typeobj_refs),
                    PythonArgProfile{*family_pp, std::move(arg_kinds)});
    }

    if (!extract_cuda_args(driver, pyargs, num_pyargs, profile->arg_kinds,
                           dispatcher.constant_arg_flags, *helper)) {
        return ErrorRaised;
    }

    ContextGuard ctx_guard(driver);
    if (!maybe_switch_context(driver, helper->cuda_context, ctx_guard))
        return ErrorRaised;

    KernelFamily::KernelMap& kernel_map = profile->family->kernels_by_constants;
    TileKernel* kernel = kernel_map.find(helper->constants);
    if (!kernel) {
        // Slowest path: need to compile a new kernel
        Result<TileKernel> res = compile(driver, dispatcher, pyargs, num_pyargs,
                                         g_default_tile_context);
        if (!res.is_ok()) return ErrorRaised;

        kernel = kernel_map.insert(std::move(helper->constants), std::move(*res));
    }

    StreamBufferTransaction tx;
    if (!helper->list_args.empty()) {
        {
            // check stream is not in capturing mode
            CUstreamCaptureStatus status;
            CUresult res = driver->cuStreamIsCapturing(launch_stream, &status);
            if (res != CUDA_SUCCESS)
                return raise(PyExc_RuntimeError, "Failed to check stream capturing status: %s",
                        get_cuda_error(driver, res));
            if (status != CU_STREAM_CAPTURE_STATUS_NONE)
                return raise(PyExc_RuntimeError, "List argument in CUDAGraph isn't supported yet");
        }
        Result<StreamBufferPool*> pool_res = get_stream_buffer_pool(driver, helper->cuda_context);
        if (!pool_res.is_ok()) return ErrorRaised;

        tx = stream_buffer_transaction_open(driver, *pool_res, launch_stream);
        if (!tx) return raise(PyExc_RuntimeError, "Failed to open a stream buffer transaction");

        size_t size = helper->nested_arrays.size() * sizeof(helper->nested_arrays[0]);
        DualPointer ptr = tx.allocate(size);
        if (!ptr)
            return raise(PyExc_RuntimeError, "Failed to allocate memory in stream buffer");

        // This is a bit of a hack. We use the same in-memory representation of `nested_arrays`
        // and the GPU buffer.
        mem_copy(ptr.host, helper->nested_arrays.data(), size);

        CUresult res = driver->cuMemcpyHtoDAsync(ptr.device, ptr.host, size, launch_stream);
        if (res != CUDA_SUCCESS) {
            return raise(PyExc_RuntimeError, "Failed to copy memory from host to device: %s",
                         get_cuda_error(driver, res));
        }

        for (ListArg la : helper->list_args) {
            helper->cuargs[la.cuarg_idx].device_ptr = reinterpret_cast<void*>(
                    ptr.device + la.offset * sizeof(Word));
        }
    }

    helper->cuarg_pointers.clear();
    for (Word& arg : helper->cuargs)
        helper->cuarg_pointers.push_back(&arg);

    CUresult res = driver->cuLaunchKernel(
            reinterpret_cast<CUfunction>(kernel->cukernel.kernel),
            grid.dims[0], grid.dims[1], grid.dims[2],
            1, 1, 1, // block size set by driver
            0, // shared memory size set by driver
            launch_stream,
            helper->cuarg_pointers.data(),
            nullptr);

    if (res != CUDA_SUCCESS) {
        if (res == CUDA_ERROR_INVALID_VALUE && try_clarify_invalid_value_error(driver, grid))
            return ErrorRaised;

        return raise(PyExc_RuntimeError, "Failed to launch cuTile kernel: %s",
                     get_cuda_error(driver, res));
    }

    return OK;
}

static Result<Vec<bool>> parse_constant_arg_flags(PyObject* tuple) {
    if (!PyTuple_Check(tuple))
        return raise(PyExc_TypeError, "constant_arg_flags must be a tuple");

    Vec<bool> constant_arg_flags;
    Py_ssize_t tuple_size = PyTuple_GET_SIZE(tuple);
    constant_arg_flags.reserve(tuple_size);
    for (Py_ssize_t i = 0; i < tuple_size; ++i) {
        PyObject* item = PyTuple_GET_ITEM(tuple, i);
        if (!PyBool_Check(item))
            return raise(PyExc_TypeError, "constant_arg_flags must be a tuple of booleans");

        int is_constant = PyObject_IsTrue(item);
        if (is_constant < 0) return ErrorRaised;
        constant_arg_flags.push_back(static_cast<bool>(is_constant));
    }
    return constant_arg_flags;
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
    return Py_NewRef(py_unwrap<TileContext>(self).config.get());
}


static PyObject * TileContext_get_autotune_cache(PyObject* self, void *closure) {
    return Py_NewRef(py_unwrap<TileContext>(self).autotune_cache.get());
}

static int TileContext_set_autotune_cache(PyObject* self, PyObject* value, void* closure) {
    TileContext& context = py_unwrap<TileContext>(self);

    // `del ctx.autotune_cache` → set back to None
    if (value == NULL) {
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
    {nullptr}  /* Sentinel */
};


static PyTypeObject TileContext_type = {
    .tp_name = "cuda.tile._cext.TileContext",
    .tp_basicsize = sizeof(PythonWrapper<TileContext>),
    .tp_dealloc = pywrapper_dealloc<TileContext>,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    .tp_getset = TileContext_getsetters,
    .tp_init = TileContext_init,
    .tp_new = pywrapper_new<TileContext>,
};


static int TileDispatcher_init(PyObject* self, PyObject* args, PyObject* kwargs) {
    const char* keywords[] = {"", "", nullptr};
    PyObject* py_constant_arg_flags = nullptr;
    PyObject* compile_func = nullptr;
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO", const_cast<char**>(keywords),
                                     &py_constant_arg_flags, &compile_func))
        return -1;

    Result<Vec<bool>> constant_arg_flags = parse_constant_arg_flags(py_constant_arg_flags);
    if (!constant_arg_flags.is_ok()) return -1;

    TileDispatcher& dispatcher = py_unwrap<TileDispatcher>(self);
    dispatcher.constant_arg_flags = std::move(*constant_arg_flags);
    dispatcher.compile_func = newref(compile_func);

    return 0;
}

static PyTypeObject TileDispatcher_type = {
    .tp_name = "cuda.tile._cext.TileDispatcher",
    .tp_basicsize = sizeof(PythonWrapper<TileDispatcher>),
    .tp_dealloc = pywrapper_dealloc<TileDispatcher>,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    .tp_init = TileDispatcher_init,
    .tp_new = pywrapper_new<TileDispatcher>,
};

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

    return grid;
}

#define LAUNCH_SIGNATURE "launch(stream, grid, kernel, kernel_args, /)"

static PyObject* cuda_tile_launch(PyObject* mod, PyObject* const* args, Py_ssize_t nargs) {
    Result<const DriverApi*> driver = get_driver_api();
    if (!driver.is_ok()) return nullptr;

    if (nargs != 4)
        return PyErr_Format(PyExc_TypeError, "Wrong number of arguments to " LAUNCH_SIGNATURE);

    PyObject* stream_pyobj = args[0];
    Result<CUstream> stream_res = parse_stream(stream_pyobj);
    if (!stream_res.is_ok()) return nullptr;

    PyObject* grid_pyobj = args[1];
    Result<Grid> grid_res = parse_grid(grid_pyobj);
    if (!grid_res.is_ok()) return nullptr;

    PyObject* dispatcher_pyobj = args[2];
    if (!PyObject_TypeCheck(dispatcher_pyobj, &TileDispatcher_type)) {
        return PyErr_Format(PyExc_TypeError,
                LAUNCH_SIGNATURE " expects a tile kernel as the third argument, got %s",
                Py_TYPE(dispatcher_pyobj)->tp_name);
    }

    PyObject* kernel_args_pyobj = args[3];
    if (!PyTuple_Check(kernel_args_pyobj)) {
        return PyErr_Format(PyExc_TypeError,
                LAUNCH_SIGNATURE " expects a tuple as the fourth argument, got %s",
                Py_TYPE(kernel_args_pyobj)->tp_name);
    }

    PyObject** kernel_args = reinterpret_cast<PyTupleObject*>(kernel_args_pyobj)->ob_item;
    Py_ssize_t num_kernel_args = PyTuple_GET_SIZE(kernel_args_pyobj);

    if (!launch(*driver, dispatcher_pyobj, *grid_res, *stream_res,
                kernel_args, num_kernel_args))
        return nullptr;

    return Py_NewRef(Py_None);
}


struct ArraySpecialization {
    size_t ndim;
    int32_t dtype_bitwidth;
    ArraySpecializationBits bits;
};


static int ArraySpecialization_init(PyObject* self, PyObject* args, PyObject* kwargs) {
    const char* keywords[] = {"base_ptr", "dtype_bitwidth", "shape", "strides", nullptr};
    PyObject* base_ptr = nullptr;
    PyObject* dtype_bitwidth = nullptr;
    PyObject* shape = nullptr;
    PyObject* strides = nullptr;
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OOOO", const_cast<char**>(keywords),
                                    &base_ptr, &dtype_bitwidth, &shape, &strides))
        return -1;

    // Parse base_ptr
    if (!PyLong_Check(base_ptr)) {
        PyErr_SetString(PyExc_TypeError, "base_ptr must be an integer");
        return -1;
    }

    intptr_t base_ptr_int = pylong_as<intptr_t>(base_ptr);
    if (base_ptr_int < 0) {
        PyErr_SetString(PyExc_ValueError, "base_ptr must be non-negative");
        return -1;
    }

    // Parse dtype_bitwidth
    if (!PyLong_Check(dtype_bitwidth)) {
        PyErr_SetString(PyExc_TypeError, "dtype_bitwidth must be an integer");
        return -1;
    }

    int32_t dtype_bitwidth_int = pylong_as<int32_t>(dtype_bitwidth);
    if (dtype_bitwidth_int < 0) {
        PyErr_SetString(PyExc_ValueError, "dtype_bitwidth must be non-negative");
        return -1;
    }

    // Parse shape and strides
    if (!PyTuple_Check(shape)) {
        PyErr_SetString(PyExc_TypeError, "shape must be a tuple");
        return -1;
    }

    if (!PyTuple_Check(strides)) {
        PyErr_SetString(PyExc_TypeError, "strides must be a tuple");
        return -1;
    }

    Py_ssize_t shape_len = PyTuple_Size(shape);
    if (shape_len == -1) return -1;

    Py_ssize_t strides_len = PyTuple_Size(strides);
    if (strides_len == -1) return -1;

    if (shape_len != strides_len) {
        PyErr_SetString(PyExc_ValueError, "shape and strides must have the same length");
        return -1;
    }

    Vec<Word> shape_stride;
    for (Py_ssize_t i = 0; i < shape_len; ++i) {
        // Parse shape at dim i
        PyObject* shape_item = PyTuple_GetItem(shape, i);
        if (!PyLong_Check(shape_item)) {
            PyErr_SetString(PyExc_TypeError, "shape must be a tuple of integers");
            return -1;
        }
        int32_t shape_int = pylong_as<int32_t>(shape_item);
        if (shape_int < 0) {
            PyErr_SetString(PyExc_ValueError, "shape must be non-negative");
            return -1;
        }
        shape_stride.push_back({.i32 = shape_int});
    }

    for (Py_ssize_t i = 0; i < shape_len; ++i) {
        // Parse stride at dim i
        PyObject* stride = PyTuple_GetItem(strides, i);
        if (!PyLong_Check(stride)) {
            PyErr_SetString(PyExc_TypeError, "strides must be a tuple of integers");
            return -1;
        }
        int32_t stride_int = pylong_as<int32_t>(stride);
        if (stride_int < 0) {
            PyErr_SetString(PyExc_ValueError, "strides must be non-negative");
            return -1;
        }
        shape_stride.push_back({.i32 = stride_int});
    }

    // Compute array specialization bits
    ArraySpecializationBits arr_special_bits = compute_array_specialization_bits(
        reinterpret_cast<void*>(base_ptr_int), shape_len,
        dtype_bitwidth_int, shape_stride.data());

    ArraySpecialization& arr_special = py_unwrap<ArraySpecialization>(self);
    arr_special.bits = arr_special_bits;
    arr_special.ndim = shape_len;
    arr_special.dtype_bitwidth = dtype_bitwidth_int;
    return 0;
}

static PyObject* ArraySpecialization_get_elements_disjoint(PyObject* self, void*) {
    ArraySpecialization& arr_special = py_unwrap<ArraySpecialization>(self);
    return Py_NewRef(arr_special.bits.disjoint_elements ? Py_True : Py_False);
}

PyObject* ArraySpecialization_get_base_ptr_div_by(PyObject* self, void* Py_UNUSED(ignored)) {
    ArraySpecialization& arr_special = py_unwrap<ArraySpecialization>(self);
    if (!arr_special.bits.baseptr_16byte_aligned)
        return Py_NewRef(Py_None);

    return PyLong_FromLong(DIVISOR_16);
}

PyObject* ArraySpecialization_get_stride_div_by(PyObject* self, void* Py_UNUSED(ignored)) {
    ArraySpecialization& arr_special = py_unwrap<ArraySpecialization>(self);
    PyPtr tuple = steal(PyTuple_New(arr_special.ndim));
    if (!tuple) return nullptr;
    for (size_t i = 0; i < arr_special.ndim; ++i) {
        if (!arr_special.bits.is_stride_16byte_divisible(i)) {
            PyTuple_SET_ITEM(tuple.get(), i, Py_NewRef(Py_None));
        } else {
            int stride_div_by_int = DIVISOR_16 * BYTE_BITWIDTH / arr_special.dtype_bitwidth;
            PyPtr stride_div_by = steal(PyLong_FromLong(stride_div_by_int));
            if (!stride_div_by) return nullptr;
            PyTuple_SET_ITEM(tuple.get(), i, Py_NewRef(stride_div_by.get()));
        }
    }
    return tuple.release();
}

static PyObject* ArraySpecialization_get_shape_div_by(PyObject* self, void* Py_UNUSED(ignored)) {
    ArraySpecialization& arr_special = py_unwrap<ArraySpecialization>(self);
    PyPtr tuple = steal(PyTuple_New(arr_special.ndim));
    if (!tuple) return nullptr;
    for (size_t i = 0; i < arr_special.ndim; ++i) {
        if (!arr_special.bits.is_shape_divisible_by_16(i)) {
            PyTuple_SET_ITEM(tuple.get(), i, Py_NewRef(Py_None));
        } else {
            PyPtr shape_div_by = steal(PyLong_FromLong(DIVISOR_16));
            if (!shape_div_by) return nullptr;
            PyTuple_SET_ITEM(tuple.get(), i, Py_NewRef(shape_div_by.get()));
        }
    }
    return tuple.release();
}

static PyObject* ArraySpecialization_get_stride_is_static(PyObject* self, void*) {
    ArraySpecialization& arr_special = py_unwrap<ArraySpecialization>(self);
    PyPtr tuple = steal(PyTuple_New(arr_special.ndim));
    if (!tuple) return nullptr;
    for (size_t i = 0; i < arr_special.ndim; ++i) {
        PyTuple_SET_ITEM(tuple.get(), i,
            Py_NewRef(arr_special.bits.is_stride_one(i) ? Py_True : Py_False));
    }
    return tuple.release();
}

static PyGetSetDef ArraySpecialization_getset[] = {
    {"elements_disjoint", ArraySpecialization_get_elements_disjoint, nullptr,
     "check if array elements are disjoint", nullptr},
    {"base_ptr_div_by", ArraySpecialization_get_base_ptr_div_by, nullptr,
     "get base_ptr_div_by", nullptr},
    {"stride_div_by", ArraySpecialization_get_stride_div_by, nullptr,
     "get stride_div_by", nullptr},
    {"shape_div_by", ArraySpecialization_get_shape_div_by, nullptr,
     "get shape_div_by", nullptr},
    {"stride_is_static", ArraySpecialization_get_stride_is_static, nullptr,
     "get stride_is_static", nullptr},
    {nullptr}
};

static PyTypeObject ArraySpecialization_type = {
    .tp_name = "cuda.tile._cext.ArraySpecialization",
    .tp_basicsize = sizeof(PythonWrapper<ArraySpecialization>),
    .tp_dealloc = pywrapper_dealloc<ArraySpecialization>,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_getset = ArraySpecialization_getset,
    .tp_init = ArraySpecialization_init,
    .tp_new = pywrapper_new<ArraySpecialization>,
};


static Status init_default_tile_context() {
    PyPtr context_module = steal(PyImport_ImportModule("cuda.tile._context"));
    if (!context_module) return ErrorRaised;

    PyPtr default_context_config = steal(
        PyObject_CallMethod(context_module.get(), "init_context_config_from_env", "")
    );
    if (!default_context_config) return ErrorRaised;

    g_default_tile_context = pywrapper_new<TileContext>(&TileContext_type, nullptr, nullptr);
    if (!g_default_tile_context) return ErrorRaised;
    TileContext& tile_context = py_unwrap<TileContext>(g_default_tile_context);
    tile_context.config = default_context_config;

    tile_context.autotune_cache = newref(Py_None);

    return OK;
};


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

    // Save a reference to cupy.ndarray
    if (PyPtr cupy_ndarray = try_getattr(cupy, "ndarray")) {
        if (PyType_Check(cupy_ndarray.get()))
            g_cupy_ndarray_type = reinterpret_cast<PyTypeObject*>(cupy_ndarray.release());
    }

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
    NULL
};


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

    g_stream_buffer_pool_by_ctx_id = new StreamBufferPoolMap();

    try_get_torch_globals();

    try_get_cupy_globals();

    try_get_numba_globals();

    if (PyType_Ready(&TileContext_type) < 0)
        return ErrorRaised;

    if (PyType_Ready(&TileDispatcher_type) < 0)
        return ErrorRaised;

    if (PyModule_AddObjectRef(m, "TileContext",
                reinterpret_cast<PyObject*>(&TileContext_type)) < 0)
        return ErrorRaised;

    if (PyModule_AddObjectRef(m, "TileDispatcher",
                reinterpret_cast<PyObject*>(&TileDispatcher_type)) < 0)
        return ErrorRaised;

    if (PyType_Ready(&ArraySpecialization_type) < 0)
        return ErrorRaised;

    if (PyModule_AddObjectRef(m, "ArraySpecialization",
                reinterpret_cast<PyObject*>(&ArraySpecialization_type)) < 0)
        return ErrorRaised;

    if (PyModule_AddFunctions(m, functions) < 0)
        return ErrorRaised;

    if (!init_default_tile_context()) return ErrorRaised;

    if (PyModule_AddObjectRef(m, "default_tile_context", g_default_tile_context) < 0)
        return ErrorRaised;

    return OK;
}


