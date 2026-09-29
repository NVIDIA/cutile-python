// SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0

#ifdef _WIN32

#include "xla/ffi/api/c_api.h"
#include "xla_ffi.h"

XLA_FFI_Error* cutile_call_handler(XLA_FFI_CallFrame*) { return nullptr; }

XLA_FFI_TypeId* cutile_call_state_type_id() {
    static XLA_FFI_TypeId id = {};
    return &id;
}

const XLA_FFI_TypeInfo* cutile_call_state_type_info() {
    return nullptr;
}

#else  // !_WIN32

#include "xla/ffi/api/c_api.h"
#include "xla_ffi.h"
#include "py.h"
#include "cuda_loader.h"
#include "hash_map.h"
#include "memory.h"
#include "vec.h"


struct CubinId {
    unsigned char bytes[32];

    bool operator==(const CubinId& o) const {
        for (size_t i = 0; i < sizeof(bytes); ++i)
            if (bytes[i] != o.bytes[i]) return false;
        return true;
    }
};


template <>
struct Hash<CubinId> {
    static void hash(const CubinId& k, Hasher& h) {
        // SHA-256 output is already uniform; one 64-bit word is enough entropy.
        uint64_t v;
        mem_copy(&v, k.bytes, sizeof(v));
        h.hash(v);
    }
};


namespace {

// =============== Error handling ===================
XLA_FFI_Error* xla_error(const XLA_FFI_Api* api,
                         XLA_FFI_Error_Code code,
                         const char* fmt, va_list ap) {
  char buf[512];
  PyOS_vsnprintf(buf, sizeof(buf), fmt, ap);
  XLA_FFI_Error_Create_Args args = {
      .struct_size = XLA_FFI_Error_Create_Args_STRUCT_SIZE,
      .extension_start = nullptr,
      .message = buf,
      .errc = code
  };
  return api->XLA_FFI_Error_Create(&args);
}

XLA_FFI_Error* xla_internal_error(const XLA_FFI_Api* api, const char* fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  XLA_FFI_Error* e = xla_error(api, XLA_FFI_Error_Code_INTERNAL, fmt, ap);
  va_end(ap);
  return e;
}


// =============== Kernel registry ===================
// `KernelEntry` owns the cuLibrary and cuKernel, with a
// reference count of how many `CutileCallState`s currently reference it.
// Every populated entry starts at refcount=1 (the constructing call
// state's reference); the registry owns these via heap allocation.
// `~KernelEntry` unloads the library best-effort.
struct KernelEntry {
    CUlibrary lib = nullptr;
    CUkernel kernel = nullptr;
    size_t refcount = 0;
    const DriverApi* driver_api;

    KernelEntry(CUlibrary l, CUkernel k, const DriverApi* driver_api)
        : lib(l), kernel(k), refcount(1), driver_api(driver_api) {}
    KernelEntry(const KernelEntry&) = delete;
    KernelEntry& operator=(const KernelEntry&) = delete;
    ~KernelEntry() {
        if (!lib) return;
        driver_api->cuLibraryUnload(lib);
    }
};

struct KernelRegistry {
#ifdef Py_GIL_DISABLED
    PyMutex m = {0};
#endif
    // Slot value is nullptr after the last referent is destroyed. Future
    // registrations of the same cubin_id repopulate the same slot.
    HashMap<CubinId, KernelEntry*> map;
};

static KernelRegistry* g_kernel_registry = new KernelRegistry();


// =============== Call State ===================
// Owns a +1 refcount on `entry`; ~CutileCallState releases it and (when
// the count hits zero) unloads the cubin and nulls the registry slot.
struct CutileCallState {
  static XLA_FFI_TypeId id;

  const DriverApi* d = nullptr;
  KernelEntry* entry = nullptr;
  CubinId cubin_id = {};
  int32_t   num_inputs   = 0;
  int32_t   num_outputs  = 0;
  int32_t   grid_x       = 0;
  int32_t   grid_y       = 0;
  int32_t   grid_z       = 0;
  // Per kernel-arg position, the index into the FFI-side buffer list
  // (inputs first, then outputs, then scalar_packed).
  Vec<int32_t> buffer_ids;
  // Index bitwidth (32 or 64) the kernel was compiled with for the
  // corresponding buffer. Parallel to `buffer_ids`.
  Vec<int32_t> index_bitwidths;
  // Packed scalar argument: each entry is the raw 64-bit bit pattern of a
  // runtime scalar (zero-padded high bits when the dtype is narrower).
  // Cast into a Word at execute time.
  Vec<uint64_t> scalar_packed;

  ~CutileCallState() {
      if (!entry) return;
      GILGuard g;
#ifdef Py_GIL_DISABLED
      PyCriticalSectionGuard guard(&g_kernel_registry->m);
#endif
      if (--entry->refcount == 0) {
          delete entry;
          if (auto* item = g_kernel_registry->map.find(cubin_id))
              item->value = nullptr;
      }
  }
};

XLA_FFI_TypeId CutileCallState::id = {};


const XLA_FFI_TypeInfo kCutileCallStateTypeInfo = {
    XLA_FFI_TypeInfo_STRUCT_SIZE,
    /*extension_start=*/nullptr,
    /*deleter=*/[](void* object) {
      delete static_cast<CutileCallState*>(object);
    },
};


template<typename StateType>
XLA_FFI_Error* set_state(const XLA_FFI_Api* api,
                         XLA_FFI_ExecutionContext* ctx,
                         XLA_FFI_ExecutionStage stage,
                         StateType* state) {
  XLA_FFI_State_Set_Args args = {
      .struct_size = XLA_FFI_State_Set_Args_STRUCT_SIZE,
      .extension_start = nullptr,
      .ctx = ctx,
      .stage = stage,
      .type_id = &StateType::id,
      .state = state
  };
  return api->XLA_FFI_State_Set(&args);
}


template<typename StateType>
XLA_FFI_Error* get_state(const XLA_FFI_Api* api,
                         XLA_FFI_ExecutionContext* ctx,
                         XLA_FFI_ExecutionStage stage,
                         StateType** out) {
  XLA_FFI_State_Get_Args args = {
      .struct_size = XLA_FFI_State_Get_Args_STRUCT_SIZE,
      .extension_start = nullptr,
      .ctx = ctx,
      .stage = stage,
      .type_id = &StateType::id,
      .state = nullptr
  };
  if (XLA_FFI_Error* err = api->XLA_FFI_State_Get(&args)) return err;
  if (args.state == nullptr) {
    return xla_internal_error(api, "State is null.");
  }
  *out = static_cast<StateType*>(args.state);
  return nullptr;
}


XLA_FFI_Error* get_stream(const XLA_FFI_Api* api, XLA_FFI_ExecutionContext* ctx, void** out) {
  XLA_FFI_Stream_Get_Args args = {
      .struct_size = XLA_FFI_Stream_Get_Args_STRUCT_SIZE,
      .extension_start = nullptr,
      .ctx = ctx,
      .stream = nullptr,
  };
  if (XLA_FFI_Error* err = api->XLA_FFI_Stream_Get(&args)) return err;
  *out = args.stream;
  return nullptr;
}

// =============== Call Handler ===================

bool is_metadata_query(const XLA_FFI_CallFrame* cf) {
    return cf->extension_start != nullptr &&
           cf->extension_start->type == XLA_FFI_Extension_Metadata;
}


XLA_FFI_Error* populate_metadata(XLA_FFI_CallFrame* cf,
                                 XLA_FFI_Handler_Traits traits,
                                 XLA_FFI_TypeId state_type_id) {
  auto* ext = reinterpret_cast<XLA_FFI_Metadata_Extension*>(cf->extension_start);
  ext->metadata->api_version = XLA_FFI_Api_Version{
      XLA_FFI_Api_Version_STRUCT_SIZE,
      /*extension_start=*/nullptr,
      XLA_FFI_API_MAJOR,
      XLA_FFI_API_MINOR,
  };
  ext->metadata->traits = traits;
  ext->metadata->state_type_id = state_type_id;
  return nullptr;
}


// ============== Attr Decode Helper ===============

static const XLA_FFI_ByteSpan* attr_string(const XLA_FFI_Attrs* attrs, size_t idx) {
    if (idx >= static_cast<size_t>(attrs->size)) return nullptr;
    if (attrs->types[idx] != XLA_FFI_AttrType_STRING) return nullptr;
    return static_cast<const XLA_FFI_ByteSpan*>(attrs->attrs[idx]);
}

static const XLA_FFI_Scalar* attr_scalar(const XLA_FFI_Attrs* attrs, size_t idx) {
    if (idx >= static_cast<size_t>(attrs->size)) return nullptr;
    if (attrs->types[idx] != XLA_FFI_AttrType_SCALAR) return nullptr;
    return static_cast<const XLA_FFI_Scalar*>(attrs->attrs[idx]);
}

static const XLA_FFI_Array* attr_array(const XLA_FFI_Attrs* attrs, size_t idx) {
    if (idx >= static_cast<size_t>(attrs->size)) return nullptr;
    if (attrs->types[idx] != XLA_FFI_AttrType_ARRAY) return nullptr;
    return static_cast<const XLA_FFI_Array*>(attrs->attrs[idx]);
}

template <typename T>
static bool read_scalar(const XLA_FFI_Attrs* attrs, size_t idx,
                        XLA_FFI_DataType expected, T* out) {
    const XLA_FFI_Scalar* s = attr_scalar(attrs, idx);
    if (!s || s->dtype != expected) return false;
    *out = *static_cast<const T*>(s->value);
    return true;
}


// Loads `code` into context independent CUlibrary + CUKernel
// and returns a heap-allocated KernelEntry (refcount=1).
static void format_sha256_hex(const CubinId& k, char out[65]) {
    static const char kHex[] = "0123456789abcdef";
    for (size_t i = 0; i < sizeof(k.bytes); ++i) {
        out[2*i]     = kHex[(k.bytes[i] >> 4) & 0xF];
        out[2*i + 1] = kHex[k.bytes[i] & 0xF];
    }
    out[64] = '\0';
}

KernelEntry* load_kernel(const XLA_FFI_Api* api, const DriverApi* d,
                         const CubinId& cubin_id,
                         const XLA_FFI_ByteSpan* code,
                         const XLA_FFI_ByteSpan* function,
                         XLA_FFI_Error** err_out) {
    CUlibrary lib = nullptr;
    CUresult rc = d->cuLibraryLoadData(&lib, code->ptr, nullptr, nullptr, 0, nullptr, nullptr, 0);
    if (rc != CUDA_SUCCESS) {
        char hex[65];
        format_sha256_hex(cubin_id, hex);
        *err_out = xla_internal_error(api,
            "LoadData failed for cubin_id=%s (rc=%d)",
            hex, static_cast<int>(rc));
        return nullptr;
    }

    // ByteSpan is not NULL-terminated; copy into a null-terminated scratch.
    Vec<char> name_buf(function->len + 1);
    mem_copy(name_buf.data(), function->ptr, function->len);
    name_buf[function->len] = '\0';

    CUkernel cukernel = nullptr;
    rc = d->cuLibraryGetKernel(&cukernel, lib, name_buf.data());
    if (rc != CUDA_SUCCESS) {
        d->cuLibraryUnload(lib);
        *err_out = xla_internal_error(api,
            "GetKernel failed for %s (rc=%d)",
            name_buf.data(), static_cast<int>(rc));
        return nullptr;
    }
    return new KernelEntry(lib, cukernel, d);
}


// ============= cuTile Call Handler =================================
// Attrs (alphabetical):
//   buffer_ids, cubin_code, cubin_id, function_name, grid_x, grid_y,
//   grid_z, index_bitwidths, num_inputs, num_outputs, scalar_packed.
enum : size_t {
    kCall_buffer_ids = 0,
    kCall_cubin_code,
    kCall_cubin_id,
    kCall_function_name,
    kCall_grid_x,
    kCall_grid_y,
    kCall_grid_z,
    kCall_index_bitwidths,
    kCall_num_inputs,
    kCall_num_outputs,
    kCall_scalar_packed,
};

XLA_FFI_Error* cutile_call_instantiate(XLA_FFI_CallFrame* cf) {
    GILGuard g;
    GlobalLock lock(g);
#ifdef Py_GIL_DISABLED
    PyCriticalSectionGuard guard(&g_kernel_registry->m);
#endif
    Result<const DriverApi*> driver_result = get_driver_api(lock);
    if (!driver_result.is_ok())
        return xla_internal_error(cf->api, "CUDA driver not available.");
    const DriverApi* d = *driver_result;

    CubinId cubin_id = {};
    const XLA_FFI_Array* cubin_id_arr = attr_array(&cf->attrs, kCall_cubin_id);
    if (!cubin_id_arr || cubin_id_arr->dtype != XLA_FFI_DataType_U8
        || cubin_id_arr->size != sizeof(cubin_id.bytes))
        return xla_internal_error(cf->api,
            "cutile_call: missing/invalid cubin_id");
    mem_copy(cubin_id.bytes, cubin_id_arr->data, sizeof(cubin_id.bytes));

    const XLA_FFI_ByteSpan* code = attr_string(&cf->attrs, kCall_cubin_code);
    if (!code) return xla_internal_error(cf->api,
            "cutile_call: missing cubin_code");

    const XLA_FFI_ByteSpan* fn = attr_string(&cf->attrs, kCall_function_name);
    if (!fn) return xla_internal_error(cf->api,
            "cutile_call: missing function_name");

    int32_t num_inputs = 0, num_outputs = 0;
    int32_t grid_x = 1, grid_y = 1, grid_z = 1;
    read_scalar(&cf->attrs, kCall_num_inputs,  XLA_FFI_DataType_S32, &num_inputs);
    read_scalar(&cf->attrs, kCall_num_outputs, XLA_FFI_DataType_S32, &num_outputs);
    read_scalar(&cf->attrs, kCall_grid_x,      XLA_FFI_DataType_S32, &grid_x);
    read_scalar(&cf->attrs, kCall_grid_y,      XLA_FFI_DataType_S32, &grid_y);
    read_scalar(&cf->attrs, kCall_grid_z,      XLA_FFI_DataType_S32, &grid_z);

    const XLA_FFI_Array* bids = attr_array(&cf->attrs, kCall_buffer_ids);
    if (!bids || bids->dtype != XLA_FFI_DataType_S32)
        return xla_internal_error(cf->api, "cutile_call: missing/invalid buffer_ids");

    const XLA_FFI_Array* ibws = attr_array(&cf->attrs, kCall_index_bitwidths);
    if (!ibws || ibws->dtype != XLA_FFI_DataType_S32 || ibws->size != bids->size)
        return xla_internal_error(cf->api,
            "cutile_call: missing/invalid index_bitwidths");

    const XLA_FFI_Array* spk = attr_array(&cf->attrs, kCall_scalar_packed);
    if (!spk || spk->dtype != XLA_FFI_DataType_U64)
        return xla_internal_error(cf->api,
            "cutile_call: missing/invalid scalar_packed");

    // Find-or-load the kernel entry and bump its refcount. The state below
    // owns this +1 ref and releases it on destruction.
    KernelEntry* entry = nullptr;
    {
        auto* item = g_kernel_registry->map.find(cubin_id);
        if (item && item->value) {
            entry = item->value;
            ++entry->refcount;
        } else {
            XLA_FFI_Error* error = nullptr;
            entry = load_kernel(cf->api, d, cubin_id, code, fn, &error);
            if (!entry) return error;
            if (item) {
                item->value = entry;  // resurrect dead slot
            } else {
                g_kernel_registry->map.insert(cubin_id, entry);
            }
        }
    }

    auto* state = new CutileCallState();
    state->d = d;
    state->entry = entry;
    state->cubin_id = cubin_id;
    state->num_inputs = num_inputs;
    state->num_outputs = num_outputs;
    state->grid_x = grid_x;
    state->grid_y = grid_y;
    state->grid_z = grid_z;
    state->buffer_ids.resize(bids->size);
    mem_copy(state->buffer_ids.data(), bids->data, bids->size * sizeof(int32_t));
    state->index_bitwidths.resize(ibws->size);
    mem_copy(state->index_bitwidths.data(), ibws->data,
             ibws->size * sizeof(int32_t));
    state->scalar_packed.resize(spk->size);
    mem_copy(state->scalar_packed.data(), spk->data,
             spk->size * sizeof(uint64_t));

    if (XLA_FFI_Error* err = set_state(cf->api, cf->ctx, cf->stage, state)) {
        // ~CutileCallState releases our refcount on the registry entry.
        delete state;
        return err;
    }
    return nullptr;
}


XLA_FFI_Error* cutile_call_initialize(XLA_FFI_CallFrame* cf) {
    CutileCallState* state = nullptr;
    if (XLA_FFI_Error* err = get_state(cf->api, cf->ctx, cf->stage, &state))
        return err;

    const DriverApi* d = state->d;
    CUfunction f;
    // Ensure a context independent kernel is loaded into cuda context before execution stage.
    CUresult rc = d->cuKernelGetFunction(&f, state->entry->kernel);
    if (rc != CUDA_SUCCESS)
        return xla_internal_error(cf->api,
            "cuKernelGetFunction failed (rc=%d)", static_cast<int>(rc));
    return nullptr;
}



union Word {
    void*   device_ptr;
    int32_t i32;
    int64_t i64;
    float   f32;
};
static_assert(sizeof(Word) == 8);

// Pack one buffer into the cuTile-python-v1 calling convention:
//   1 Word for the device pointer, then ndim Words for shape, then ndim Words
//   for compact row-major strides (in elements, strides[ndim-1] = 1).
// Shape/stride are written to the full i64 lane: an i32-indexed kernel reads
// the low 4 bytes (correct for values < 2^31), an i64-indexed kernel reads
// all 8. When `index_bitwidth == 32` we refuse oversize values up front so
// the kernel can't silently read a truncated index.
static XLA_FFI_Error* pack_buffer(const XLA_FFI_Api* api,
                                  const XLA_FFI_Buffer* buf,
                                  int32_t index_bitwidth,
                                  Vec<Word>* cuargs) {
    constexpr int64_t kI32Max = (int64_t{1} << 31) - 1;
    const int64_t ndim = buf->rank;
    if (index_bitwidth == 32) {
        for (int64_t i = 0; i < ndim; ++i) {
            if (buf->dims[i] > kI32Max)
                return xla_internal_error(api,
                    "cutile_call: shape[%lld]=%lld exceeds the int32 index "
                    "range; annotate the kernel parameter with "
                    "ct.IndexedWithInt64",
                    (long long)i, (long long)buf->dims[i]);
        }
    }

    cuargs->push_back({.device_ptr = buf->data});

    for (int64_t i = 0; i < ndim; ++i) {
        Word w{};
        w.i64 = buf->dims[i];
        cuargs->push_back(w);
    }

    // Row-major strides, computed back-to-front.
    size_t stride_base = cuargs->size();
    for (int64_t i = 0; i < ndim; ++i) {
        cuargs->push_back(Word{});
    }
    int64_t stride = 1;
    for (int64_t i = ndim - 1; i >= 0; --i) {
        if (index_bitwidth == 32 && stride > kI32Max)
            return xla_internal_error(api,
                "cutile_call: stride at dim %lld = %lld exceeds "
                "the int32 index range; annotate the kernel parameter with "
                "ct.IndexedWithInt64",
                (long long)i, (long long)stride);
        (*cuargs)[stride_base + i].i64 = stride;
        stride *= buf->dims[i];
    }
    return nullptr;
}

XLA_FFI_Error* cutile_call_execute(XLA_FFI_CallFrame* cf) {
    CutileCallState* state = nullptr;
    if (XLA_FFI_Error* err = get_state(cf->api, cf->ctx, cf->stage, &state))
        return err;

    void* stream_v = nullptr;
    if (XLA_FFI_Error* err = get_stream(cf->api, cf->ctx, &stream_v)) return err;
    CUstream stream = static_cast<CUstream>(stream_v);

    const DriverApi* d = state->d;

    // Pack each kernel-arg slot from its corresponding XLA buffer.
    Vec<Word> cuargs;
    for (size_t k = 0; k < state->buffer_ids.size(); ++k) {
        int32_t bid = state->buffer_ids[k];
        const XLA_FFI_Buffer* buf = nullptr;
        if (bid < state->num_inputs) {
            if (bid >= cf->args.size)
                return xla_internal_error(cf->api,
                    "buffer_ids[%zu]=%d out of range (args.size=%lld)",
                    k, bid, static_cast<long long>(cf->args.size));
            buf = static_cast<const XLA_FFI_Buffer*>(cf->args.args[bid]);
        } else if (bid < state->num_inputs + state->num_outputs) {
            int32_t ridx = bid - state->num_inputs;
            if (ridx >= cf->rets.size)
                return xla_internal_error(cf->api,
                    "buffer_ids[%zu]=%d out of range (rets.size=%lld)",
                    k, bid, static_cast<long long>(cf->rets.size));
            buf = static_cast<const XLA_FFI_Buffer*>(cf->rets.rets[ridx]);
        }
        if (buf) {
            int32_t ibw = state->index_bitwidths[k];
            if (XLA_FFI_Error* err = pack_buffer(cf->api, buf, ibw, &cuargs))
                return err;
        } else {
            // scalar argument
            int32_t ridx = bid - state->num_inputs - state->num_outputs;
            if (ridx < 0 || static_cast<size_t>(ridx) >= state->scalar_packed.size())
                return xla_internal_error(cf->api,
                    "buffer_ids[%zu]=%d out of range (scalar_packed.size=%lld)",
                    k, bid, static_cast<long long>(state->scalar_packed.size()));
            Word w{};
            w.i64 = static_cast<int64_t>(state->scalar_packed[ridx]);
            cuargs.push_back(w);
        }
    }

    // cuLaunchKernel expects an array of pointers, one per kernel parameter.
    Vec<void*> cuarg_pointers;
    cuarg_pointers.reserve(cuargs.size());
    for (size_t i = 0; i < cuargs.size(); ++i) {
        cuarg_pointers.push_back(&cuargs[i]);
    }

    CUresult rc = d->cuLaunchKernel(
        reinterpret_cast<CUfunction>(state->entry->kernel),
        state->grid_x, state->grid_y, state->grid_z,
        /*block=*/1, 1, 1,
        /*dynamic_shared_mem=*/0,
        stream,
        cuarg_pointers.data(),
        nullptr);
    if (rc != CUDA_SUCCESS)
        return xla_internal_error(cf->api,
            "cuLaunchKernel failed (rc=%d)", static_cast<int>(rc));
    return nullptr;
}

}  // namespace


// =================== Exported handlers / accessors ========================

XLA_FFI_Error* cutile_call_handler(XLA_FFI_CallFrame* cf) {
    if (is_metadata_query(cf)) {
        return populate_metadata(cf,
                XLA_FFI_HANDLER_TRAITS_COMMAND_BUFFER_COMPATIBLE,
                CutileCallState::id);
    }
    if (cf->stage == XLA_FFI_ExecutionStage_INSTANTIATE) {
        return cutile_call_instantiate(cf);
    } else if (cf->stage == XLA_FFI_ExecutionStage_INITIALIZE) {
        return cutile_call_initialize(cf);
    }
    if (cf->stage == XLA_FFI_ExecutionStage_EXECUTE) {
        return cutile_call_execute(cf);
    }
    return xla_internal_error(cf->api, "stage %d not implemented", cf->stage);
}


XLA_FFI_TypeId* cutile_call_state_type_id() {
    return &CutileCallState::id;
}

const XLA_FFI_TypeInfo* cutile_call_state_type_info() {
    return &kCutileCallStateTypeInfo;
}

#endif  // _WIN32
