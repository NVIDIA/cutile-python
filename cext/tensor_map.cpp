// SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0

#include "tensor_map.h"
#include "cuda_helper.h"
#include "cuda_loader.h"


static PyObject* g_tensor_map_data_type;
static PyObject* g_tensor_map_interleave;
static PyObject* g_swizzle_mode;
static PyObject* g_tensor_map_l2_promotion;
static PyObject* g_tensor_map_float_oob_fill;
static PyObject* g_tma_load_mode;
static PyObject* g_tma_store_mode;

enum class TMALoadMode : int {
    TILE = 0,
    IM2COL = 1,
    IM2COL_W = 2,
    IM2COL_W_128 = 3,
    TILE_GATHER4 = 4,
};

enum class TMAStoreMode : int {
    TILE = 0,
    IM2COL = 1,
    TILE_SCATTER4 = 2,
};


Result<uint32_t> tensor_map_data_type_bitwidth(CUtensorMapDataType dtype) {
    switch (dtype) {
    case CU_TENSOR_MAP_DATA_TYPE_UINT8:
        return 8;
    case CU_TENSOR_MAP_DATA_TYPE_UINT16:
    case CU_TENSOR_MAP_DATA_TYPE_FLOAT16:
    case CU_TENSOR_MAP_DATA_TYPE_BFLOAT16:
        return 16;
    case CU_TENSOR_MAP_DATA_TYPE_UINT32:
    case CU_TENSOR_MAP_DATA_TYPE_INT32:
    case CU_TENSOR_MAP_DATA_TYPE_FLOAT32:
    case CU_TENSOR_MAP_DATA_TYPE_FLOAT32_FTZ:
    case CU_TENSOR_MAP_DATA_TYPE_TFLOAT32:
    case CU_TENSOR_MAP_DATA_TYPE_TFLOAT32_FTZ:
        return 32;
    case CU_TENSOR_MAP_DATA_TYPE_UINT64:
    case CU_TENSOR_MAP_DATA_TYPE_INT64:
    case CU_TENSOR_MAP_DATA_TYPE_FLOAT64:
        return 64;
    case CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B:
    case CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B:
    case CU_TENSOR_MAP_DATA_TYPE_16U6_ALIGN16B:
        // TODO: Support packed tensor-map data types
        return raise(
                PyExc_ValueError,
                "Can't create tensor map: unsupported data type ", dtype);
    default:
        return raise(PyExc_ValueError, "Can't create tensor map: unsupported data type ", dtype);
    }
}


Status tensor_map_validate_global_address(
        CUtensorMapDataType data_type,
        CUtensorMapInterleave interleave,
        const void* global_address) {
    if (!global_address)
        return raise(
                PyExc_ValueError,
                "Can't create a tensor map: global address must not be null");

    uintptr_t alignment = 16;
    if (interleave == CU_TENSOR_MAP_INTERLEAVE_32B
            || data_type == CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B
            || data_type == CU_TENSOR_MAP_DATA_TYPE_16U6_ALIGN16B) {
        alignment = 32;
    }
    uintptr_t address = reinterpret_cast<uintptr_t>(global_address);
    if (address & (alignment - 1)) {
        return raise(
                PyExc_ValueError,
                "Can't create a tensor map: global address must be aligned to ",
                alignment, " bytes");
    }
    return OK;
}


Status tensor_map_encode_global_dimensions(
        CUtensorMapDataType data_type,
        int32_t rank,
        const int64_t* global_dimensions,
        uint64_t* encoded_dimensions) {
    // TODO: Validate packed-data-type dimension constraints here once packed
    // tensor maps are supported.
    (void)data_type;
    constexpr uint64_t max_global_dimension = uint64_t{1} << 31;
    for (int32_t i = 0; i < rank; ++i) {
        int64_t dimension = global_dimensions[i];
        if (dimension <= 0
                || static_cast<uint64_t>(dimension) > max_global_dimension) {
            return raise(PyExc_ValueError,
                         "Can't create a tensor map: dimensions must be between 1 and 2^31, "
                         "got ", dimension);
        }
        encoded_dimensions[i] = static_cast<uint64_t>(dimension);
    }
    return OK;
}


Status tensor_map_encode_global_strides(
        CUtensorMapDataType data_type,
        CUtensorMapInterleave interleave,
        int32_t rank,
        const int64_t* global_element_strides,
        uint64_t* encoded_byte_strides) {
    Result<uint32_t> element_bitwidth = tensor_map_data_type_bitwidth(data_type);
    if (!element_bitwidth.is_ok()) return ErrorRaised;

    if (global_element_strides[0] != 1) {
        return raise(PyExc_ValueError,
                     "Can't create a tensor map: stride of descriptor axis zero "
                     "must be 1, got ", global_element_strides[0]);
    }

    constexpr uint64_t max_global_stride_bytes = uint64_t{1} << 40;
    constexpr uint64_t max_global_stride_bits =
        (max_global_stride_bytes - 1) * 8;
    for (int32_t i = 1; i < rank; ++i) {
        int64_t stride = global_element_strides[i];
        if (stride <= 0) {
            return raise(PyExc_ValueError,
                         "Can't create a tensor map: strides must be positive, got ", stride);
        }
        uint64_t unsigned_stride = static_cast<uint64_t>(stride);
        if (unsigned_stride > max_global_stride_bits / *element_bitwidth) {
            return raise(PyExc_ValueError,
                         "Can't create a tensor map: byte strides must be less than 2^40, "
                         "got ", stride, " elements of ", *element_bitwidth, " bits");
        }
        uint64_t stride_bits = unsigned_stride * *element_bitwidth;
        if (stride_bits % 8 != 0) {
            return raise(PyExc_ValueError,
                         "Can't create a tensor map: element stride must describe a whole "
                         "number of bytes, got ", stride, " elements of ",
                         *element_bitwidth, " bits");
        }
        uint64_t byte_stride = stride_bits / 8;
        if (byte_stride % 16 != 0) {
            return raise(PyExc_ValueError,
                         "Can't create a tensor map: byte strides must be a multiple of 16, "
                         "got ", byte_stride);
        }
        if (interleave == CU_TENSOR_MAP_INTERLEAVE_32B && byte_stride % 32 != 0) {
            return raise(PyExc_ValueError,
                         "Can't create a tensor map: 32-byte interleave requires byte "
                         "strides to be a multiple of 32, got ", byte_stride);
        }
        encoded_byte_strides[i - 1] = byte_stride;
    }
    return OK;
}


static constexpr uint32_t dlpack_dtype_key(
        uint8_t code,
        uint8_t bits,
        uint16_t lanes) {
    return static_cast<uint32_t>(code)
        | (static_cast<uint32_t>(bits) << 8)
        | (static_cast<uint32_t>(lanes) << 16);
}


static Result<CUtensorMapDataType> from_dldatatype(DLDataType dtype) {
    switch (dlpack_dtype_key(dtype.code, dtype.bits, dtype.lanes)) {
    case dlpack_dtype_key(kDLUInt, 8, 1):
    case dlpack_dtype_key(kDLInt, 8, 1):
    case dlpack_dtype_key(kDLFloat8_e4m3fn, 8, 1):
    case dlpack_dtype_key(kDLFloat8_e5m2, 8, 1):
    case dlpack_dtype_key(kDLFloat8_e8m0fnu, 8, 1):
        return CU_TENSOR_MAP_DATA_TYPE_UINT8;
    case dlpack_dtype_key(kDLUInt, 16, 1):
        return CU_TENSOR_MAP_DATA_TYPE_UINT16;
    case dlpack_dtype_key(kDLUInt, 32, 1):
        return CU_TENSOR_MAP_DATA_TYPE_UINT32;
    case dlpack_dtype_key(kDLInt, 32, 1):
        return CU_TENSOR_MAP_DATA_TYPE_INT32;
    case dlpack_dtype_key(kDLUInt, 64, 1):
        return CU_TENSOR_MAP_DATA_TYPE_UINT64;
    case dlpack_dtype_key(kDLInt, 64, 1):
        return CU_TENSOR_MAP_DATA_TYPE_INT64;
    case dlpack_dtype_key(kDLFloat, 16, 1):
        return CU_TENSOR_MAP_DATA_TYPE_FLOAT16;
    case dlpack_dtype_key(kDLFloat, 32, 1):
        return CU_TENSOR_MAP_DATA_TYPE_FLOAT32;
    case dlpack_dtype_key(kDLFloat, 64, 1):
        return CU_TENSOR_MAP_DATA_TYPE_FLOAT64;
    case dlpack_dtype_key(kDLBfloat, 16, 1):
        return CU_TENSOR_MAP_DATA_TYPE_BFLOAT16;
    default:
        return raise(PyExc_TypeError,
                     "Data type with DLPack code ", dtype.code,
                     ", bit width ", dtype.bits,
                     ", and lane count ", dtype.lanes,
                     " is not supported by tensor map");
    }
}


Status tensor_map_validate_tile(
        CUtensorMapDataType data_type,
        size_t rank,
        const uint32_t* shape,
        CUtensorMapInterleave interleave,
        CUtensorMapSwizzle swizzle) {
    if (rank < 1 || rank > kTensorMapMaxRank) {
        return raise(PyExc_ValueError, "Tensor-map array rank must be between one and five");
    }
    if (interleave != CU_TENSOR_MAP_INTERLEAVE_NONE && rank < 3)
        return raise(PyExc_ValueError,
                     "Can't create a tensor map: interleaved tensor maps require rank >= 3");
    if (interleave == CU_TENSOR_MAP_INTERLEAVE_32B && swizzle != CU_TENSOR_MAP_SWIZZLE_32B)
        return raise(PyExc_ValueError,
                     "Can't create a tensor map: 32-byte interleave requires SWIZZLE_32B");
    for (size_t i = 0; i < rank; ++i) {
        uint32_t dim = shape[i];
        if (dim < 1 || dim > 256) {
            return raise(PyExc_ValueError,
                         "Can't create a tensor map: tile dimensions must be between 1 and "
                         "256, got ", dim);
        }
    }
    Result<uint32_t> element_bitwidth = tensor_map_data_type_bitwidth(data_type);
    if (!element_bitwidth.is_ok()) return ErrorRaised;
    uint64_t first_tile_bits = uint64_t{shape[0]} * *element_bitwidth;
    if (interleave == CU_TENSOR_MAP_INTERLEAVE_NONE && first_tile_bits % 128 != 0) {
        return raise(PyExc_ValueError,
                     "Can't create a tensor map: the first tile dimension times the element "
                     "bit width must be a multiple of 128 bits, got ", first_tile_bits);
    }
    uint32_t swizzle_bytes = 0;
    switch (swizzle) {
        case CU_TENSOR_MAP_SWIZZLE_NONE:
            break;
        case CU_TENSOR_MAP_SWIZZLE_32B:
            swizzle_bytes = 32;
            break;
        case CU_TENSOR_MAP_SWIZZLE_64B:
            swizzle_bytes = 64;
            break;
        case CU_TENSOR_MAP_SWIZZLE_128B:
        case CU_TENSOR_MAP_SWIZZLE_128B_ATOM_32B:
        case CU_TENSOR_MAP_SWIZZLE_128B_ATOM_32B_FLIP_8B:
        case CU_TENSOR_MAP_SWIZZLE_128B_ATOM_64B:
            swizzle_bytes = 128;
            break;
        default:
            return raise(PyExc_ValueError, "Can't create a tensor map: invalid swizzle mode");
    }
    if (interleave == CU_TENSOR_MAP_INTERLEAVE_NONE
            && swizzle_bytes && first_tile_bits > uint64_t{swizzle_bytes} * 8) {
        return raise(PyExc_ValueError,
                     "Can't create a tensor map: the first tile dimension spans ",
                     first_tile_bits / 8, " bytes, exceeding the ", swizzle_bytes,
                     "-byte swizzle span");
    }
    return OK;
}


static Status encode_tensor_map_tiled(
        const DriverApi* driver,
        CUtensorMap* descriptor,
        CUtensorMapDataType data_type,
        int32_t rank,
        void* global_address,
        const int64_t* global_dimensions,
        const int64_t* global_element_strides,
        const uint32_t* tile_dimensions,
        CUtensorMapInterleave interleave,
        CUtensorMapSwizzle swizzle,
        CUtensorMapL2promotion l2_promotion,
        CUtensorMapFloatOOBfill oob_fill) {
    if (!descriptor || !global_dimensions || !global_element_strides || !tile_dimensions)
        return raise(PyExc_RuntimeError, "Invalid tensor-map encoding arguments");
    if (rank < 1 || rank > static_cast<int32_t>(kTensorMapMaxRank))
        return raise(PyExc_ValueError, "Tensor-map rank must be between one and five");

    if (!tensor_map_validate_tile(data_type, rank, tile_dimensions, interleave, swizzle))
        return ErrorRaised;

    uint64_t encoded_dimensions[kTensorMapMaxRank];
    uint64_t encoded_byte_strides[kTensorMapMaxRank - 1];
    uint32_t traversal_steps[kTensorMapMaxRank] = {1, 1, 1, 1, 1};
    if (!tensor_map_validate_global_address(
                data_type, interleave, global_address))
        return ErrorRaised;
    if (!tensor_map_encode_global_dimensions(
                data_type, rank, global_dimensions, encoded_dimensions))
        return ErrorRaised;
    if (!tensor_map_encode_global_strides(
                data_type, interleave, rank, global_element_strides,
                encoded_byte_strides))
        return ErrorRaised;

    CUresult result = driver->cuTensorMapEncodeTiled(
            descriptor,
            data_type,
            static_cast<cuuint32_t>(rank),
            global_address,
            encoded_dimensions,
            encoded_byte_strides,
            tile_dimensions,
            traversal_steps,
            interleave,
            swizzle,
            l2_promotion,
            oob_fill);
    if (result != CUDA_SUCCESS) {
        return raise(
                PyExc_RuntimeError,
                "Failed to encode tiled tensor map: ",
                get_cuda_error(driver, result));
    }
    return OK;
}


int32_t native_tensor_map_encode_tiled(
        GlobalLock& lock,
        void* descriptor,
        int32_t data_type,
        int32_t rank,
        void* global_address,
        const int64_t* global_dimensions,
        const int64_t* global_element_strides,
        const int64_t* tile_dimensions,
        int32_t interleave,
        int32_t swizzle,
        int32_t l2_promotion,
        int32_t oob_fill) {
    if (!tile_dimensions) {
        raise(PyExc_RuntimeError, "Invalid tensor-map encoding arguments");
        return -1;
    }
    if (rank < 1 || rank > static_cast<int32_t>(kTensorMapMaxRank)) {
        raise(PyExc_ValueError, "Tensor-map rank must be between one and five");
        return -1;
    }
    uint32_t checked_tile_dimensions[kTensorMapMaxRank];
    for (int32_t i = 0; i < rank; ++i) {
        int64_t dim = tile_dimensions[i];
        if (dim < 1 || dim > 256) {
            raise(PyExc_ValueError,
                  "Can't create a tensor map: tile dimensions must be between 1 and "
                  "256, got ", dim);
            return -1;
        }
        checked_tile_dimensions[i] = static_cast<uint32_t>(dim);
    }
    Result<const DriverApi*> driver = get_driver_api(lock);
    if (!driver.is_ok()) return -1;
    if (!encode_tensor_map_tiled(
                *driver,
                static_cast<CUtensorMap*>(descriptor),
                static_cast<CUtensorMapDataType>(data_type),
                rank,
                global_address,
                global_dimensions,
                global_element_strides,
                checked_tile_dimensions,
                static_cast<CUtensorMapInterleave>(interleave),
                static_cast<CUtensorMapSwizzle>(swizzle),
                static_cast<CUtensorMapL2promotion>(l2_promotion),
                static_cast<CUtensorMapFloatOOBfill>(oob_fill))) {
        return -1;
    }
    return CUDA_SUCCESS;
}


static PyObject* py_tensor_map_data_type_bitwidth(PyObject*, PyObject* object) {
    long data_type = pylong_as<long>(object);
    if (PyErr_Occurred()) return nullptr;
    Result<uint32_t> bitwidth = tensor_map_data_type_bitwidth(
            static_cast<CUtensorMapDataType>(data_type));
    if (!bitwidth.is_ok()) return nullptr;
    return PyLong_FromUnsignedLong(*bitwidth);
}

static PyMethodDef tensor_map_functions[] = {
    {"_tensor_map_data_type_bitwidth", py_tensor_map_data_type_bitwidth, METH_O, nullptr},
    {}  // sentinel
};


struct PythonEnumEntry {
    const char name[32];
    int value;
};


static PyPtr define_python_enum(
        PyObject* enum_type,
        const char* name,
        const PythonEnumEntry* entries,
        size_t entry_count) {
    PyPtr members = steal(PyDict_New());
    if (!members) return {};
    for (size_t i = 0; i < entry_count; ++i) {
        PyPtr value = steal(PyLong_FromLong(entries[i].value));
        if (!value) return {};
        if (PyDict_SetItemString(members.get(), entries[i].name, value.get()) < 0)
            return {};
    }
    return steal(PyObject_CallFunction(enum_type, "sO", name, members.get()));
}


template <size_t N>
static Status add_python_enum(
        PyObject* module,
        PyObject* enum_type,
        const char* name,
        const PythonEnumEntry (&entries)[N],
        PyObject** result_out = nullptr) {
    PyPtr result = define_python_enum(enum_type, name, entries, N);
    if (!result || PyModule_AddObjectRef(module, name, result.get()) < 0)
        return ErrorRaised;
    if (result_out)
        *result_out = result.release();
    return OK;
}


#define TENSOR_MAP_DATA_TYPE_ENUMS(X) \
    X(UINT8, CU_TENSOR_MAP_DATA_TYPE_UINT8) \
    X(UINT16, CU_TENSOR_MAP_DATA_TYPE_UINT16) \
    X(UINT32, CU_TENSOR_MAP_DATA_TYPE_UINT32) \
    X(INT32, CU_TENSOR_MAP_DATA_TYPE_INT32) \
    X(UINT64, CU_TENSOR_MAP_DATA_TYPE_UINT64) \
    X(INT64, CU_TENSOR_MAP_DATA_TYPE_INT64) \
    X(FLOAT16, CU_TENSOR_MAP_DATA_TYPE_FLOAT16) \
    X(FLOAT32, CU_TENSOR_MAP_DATA_TYPE_FLOAT32) \
    X(FLOAT64, CU_TENSOR_MAP_DATA_TYPE_FLOAT64) \
    X(BFLOAT16, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16) \
    X(FLOAT32_FTZ, CU_TENSOR_MAP_DATA_TYPE_FLOAT32_FTZ) \
    X(TFLOAT32, CU_TENSOR_MAP_DATA_TYPE_TFLOAT32) \
    X(TFLOAT32_FTZ, CU_TENSOR_MAP_DATA_TYPE_TFLOAT32_FTZ) \
    X(_16U4_ALIGN8B, CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B) \
    X(_16U4_ALIGN16B, CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B) \
    X(_16U6_ALIGN16B, CU_TENSOR_MAP_DATA_TYPE_16U6_ALIGN16B)

#define TENSOR_MAP_SWIZZLE_ENUMS(X) \
    X(SWIZZLE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE) \
    X(SWIZZLE_32B, CU_TENSOR_MAP_SWIZZLE_32B) \
    X(SWIZZLE_64B, CU_TENSOR_MAP_SWIZZLE_64B) \
    X(SWIZZLE_128B, CU_TENSOR_MAP_SWIZZLE_128B) \
    X(SWIZZLE_128B_ATOM_32B, CU_TENSOR_MAP_SWIZZLE_128B_ATOM_32B) \
    X(SWIZZLE_128B_ATOM_32B_FLIP_8B, \
      CU_TENSOR_MAP_SWIZZLE_128B_ATOM_32B_FLIP_8B) \
    X(SWIZZLE_128B_ATOM_64B, CU_TENSOR_MAP_SWIZZLE_128B_ATOM_64B)

#define TENSOR_MAP_INTERLEAVE_ENUMS(X) \
    X(NONE, CU_TENSOR_MAP_INTERLEAVE_NONE) \
    X(INTERLEAVE_16B, CU_TENSOR_MAP_INTERLEAVE_16B) \
    X(INTERLEAVE_32B, CU_TENSOR_MAP_INTERLEAVE_32B)

#define TENSOR_MAP_L2_PROMOTION_ENUMS(X) \
    X(NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE) \
    X(L2_64B, CU_TENSOR_MAP_L2_PROMOTION_L2_64B) \
    X(L2_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B) \
    X(L2_256B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B)

#define TENSOR_MAP_FLOAT_OOB_FILL_ENUMS(X) \
    X(NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) \
    X(NAN_REQUEST_ZERO_FMA, CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA)

#define TMA_LOAD_MODE_ENUMS(X) \
    X(TILE, static_cast<int>(TMALoadMode::TILE)) \
    X(IM2COL, static_cast<int>(TMALoadMode::IM2COL)) \
    X(IM2COL_W, static_cast<int>(TMALoadMode::IM2COL_W)) \
    X(IM2COL_W_128, static_cast<int>(TMALoadMode::IM2COL_W_128)) \
    X(TILE_GATHER4, static_cast<int>(TMALoadMode::TILE_GATHER4))

#define TMA_STORE_MODE_ENUMS(X) \
    X(TILE, static_cast<int>(TMAStoreMode::TILE)) \
    X(IM2COL, static_cast<int>(TMAStoreMode::IM2COL)) \
    X(TILE_SCATTER4, static_cast<int>(TMAStoreMode::TILE_SCATTER4))

#define PYTHON_ENUM_ENTRY(name, value) {#name, value},

static Status add_tensor_map_enums(PyObject* module) {
    static const PythonEnumEntry data_type_entries[] = {
        TENSOR_MAP_DATA_TYPE_ENUMS(PYTHON_ENUM_ENTRY)
    };
    static const PythonEnumEntry interleave_entries[] = {
        TENSOR_MAP_INTERLEAVE_ENUMS(PYTHON_ENUM_ENTRY)
    };
    static const PythonEnumEntry swizzle_entries[] = {
        TENSOR_MAP_SWIZZLE_ENUMS(PYTHON_ENUM_ENTRY)
    };
    static const PythonEnumEntry l2_promotion_entries[] = {
        TENSOR_MAP_L2_PROMOTION_ENUMS(PYTHON_ENUM_ENTRY)
    };
    static const PythonEnumEntry float_oob_fill_entries[] = {
        TENSOR_MAP_FLOAT_OOB_FILL_ENUMS(PYTHON_ENUM_ENTRY)
    };
    static const PythonEnumEntry load_mode_entries[] = {
        TMA_LOAD_MODE_ENUMS(PYTHON_ENUM_ENTRY)
    };
    static const PythonEnumEntry store_mode_entries[] = {
        TMA_STORE_MODE_ENUMS(PYTHON_ENUM_ENTRY)
    };

    PyPtr enum_mod = steal(PyImport_ImportModule("enum"));
    if (!enum_mod) return ErrorRaised;

    PyPtr enum_type = getattr(enum_mod, "Enum");
    if (!enum_type) return ErrorRaised;

    if (!add_python_enum(
                module, enum_type.get(), "TensorMapDataType", data_type_entries,
                &g_tensor_map_data_type))
        return ErrorRaised;
    if (!add_python_enum(
                module, enum_type.get(), "TensorMapInterleave", interleave_entries,
                &g_tensor_map_interleave))
        return ErrorRaised;
    if (!add_python_enum(
                module, enum_type.get(), "SwizzleMode", swizzle_entries,
                &g_swizzle_mode))
        return ErrorRaised;
    if (!add_python_enum(
                module, enum_type.get(), "TensorMapL2Promotion", l2_promotion_entries,
                &g_tensor_map_l2_promotion))
        return ErrorRaised;
    if (!add_python_enum(
                module, enum_type.get(), "TensorMapFloatOOBFill", float_oob_fill_entries,
                &g_tensor_map_float_oob_fill))
        return ErrorRaised;
    if (!add_python_enum(
                module, enum_type.get(), "TMALoadMode", load_mode_entries,
                &g_tma_load_mode))
        return ErrorRaised;
    if (!add_python_enum(
                module, enum_type.get(), "TMAStoreMode", store_mode_entries,
                &g_tma_store_mode))
        return ErrorRaised;
    return OK;
}

#undef PYTHON_ENUM_ENTRY
#undef TMA_STORE_MODE_ENUMS
#undef TMA_LOAD_MODE_ENUMS
#undef TENSOR_MAP_FLOAT_OOB_FILL_ENUMS
#undef TENSOR_MAP_L2_PROMOTION_ENUMS
#undef TENSOR_MAP_SWIZZLE_ENUMS
#undef TENSOR_MAP_INTERLEAVE_ENUMS
#undef TENSOR_MAP_DATA_TYPE_ENUMS




PyPtr tensor_map_tiled(
        DLDataType array_dtype,
        void* array_address,
        size_t rank,
        const int64_t* array_dimensions,
        const int64_t* array_strides,
        const uint32_t* shape,
        const uint32_t* order,
        int interleave,
        int swizzle,
        int l2_promotion,
        int oob_fill,
        GlobalLock& lock) {

    Result<CUtensorMapDataType> data_type = from_dldatatype(array_dtype);
    if (!data_type.is_ok()) return {};

    if (rank < 1 || rank > kTensorMapMaxRank) {
        raise(PyExc_ValueError, "Tensor-map array rank must be between one and five");
        return {};
    }
    CUtensorMap descriptor{};

    int64_t global_dimensions[kTensorMapMaxRank];
    int64_t global_element_strides[kTensorMapMaxRank];

    for (size_t i = 0; i < rank; ++i) {
        uint32_t axis = order[i];
        global_dimensions[i] = array_dimensions[axis];
        global_element_strides[i] = array_strides[axis];
    }
    Result<const DriverApi*> driver = get_driver_api(lock);
    if (!driver.is_ok()) return {};
    if (!encode_tensor_map_tiled(
            *driver,
            &descriptor,
            *data_type,
            static_cast<int32_t>(rank),
            array_address,
            global_dimensions,
            global_element_strides,
            shape,
            static_cast<CUtensorMapInterleave>(interleave),
            static_cast<CUtensorMapSwizzle>(swizzle),
            static_cast<CUtensorMapL2promotion>(l2_promotion),
            static_cast<CUtensorMapFloatOOBfill>(oob_fill)))
        return {};
    return steal(PyBytes_FromStringAndSize(
            reinterpret_cast<const char*>(&descriptor), sizeof(descriptor)));
}


Status tensor_map_init(PyObject* m) {
    if (!add_tensor_map_enums(m))
        return ErrorRaised;
    if (PyModule_AddIntConstant(m, "_TENSOR_MAP_DESCRIPTOR_BYTES", sizeof(CUtensorMap)) < 0)
        return ErrorRaised;
    if (PyModule_AddIntConstant(m, "_TENSOR_MAP_DESCRIPTOR_ALIGNMENT", alignof(CUtensorMap)) < 0)
        return ErrorRaised;
    if (PyModule_AddIntConstant(m, "_TENSOR_MAP_MAX_RANK", kTensorMapMaxRank) < 0)
        return ErrorRaised;
    if (PyModule_AddFunctions(m, tensor_map_functions) < 0)
        return ErrorRaised;
    return OK;
}
