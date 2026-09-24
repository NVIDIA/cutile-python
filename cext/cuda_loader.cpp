// SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0

#include "cuda_loader.h"
#include "cuda_helper.h"


namespace {



void* do_get_proc_address(cuGetProcAddress_v2_t getter,
                          const char* name, int cuda_version) {
    void* ret = nullptr;
    CUresult res = getter(name, &ret, cuda_version, CU_GET_PROC_ADDRESS_DEFAULT, nullptr);
    if (res != CUDA_SUCCESS) {
        raise(PyExc_RuntimeError,
              "Failed to load '", name, "' from the CUDA library: cuGetProcAddress_v2 returned ",
              res);
        return nullptr;
    }

    if (!ret) {
        raise(PyExc_RuntimeError,
              "Function '", name,"' is not available in the CUDA library");
        return nullptr;
    }

    return ret;
}

template <typename F>
F get_proc_address(cuGetProcAddress_v2_t getter,
                   const char* name, int cuda_version) {
    return reinterpret_cast<F>(do_get_proc_address(getter, name, cuda_version));
}

} // anonymous namespace


#define DEFINE_CUDA_FUNCTION_GLOBAL(name, _key, _cuda_version) \
    decltype(name)* g_##name;

FOREACH_CUDA_FUNCTION_TO_LOAD(DEFINE_CUDA_FUNCTION_GLOBAL)

#define GET_PROC_ADDRESS(name, key, cuda_ver) \
        if (!(driver_api->name = \
                    get_proc_address<decltype(name)*>(_cuGetProcAddress, key, cuda_ver))) \
            return ErrorRaised;


Status driver_api_init(DriverApi* driver_api, cuGetProcAddress_v2_t _cuGetProcAddress) {
    FOREACH_CUDA_FUNCTION_TO_LOAD(GET_PROC_ADDRESS)
    driver_api->cuda_version = 0;
    return OK;
}

std::optional<CUlaunchAttribute> DriverApi::get_shared_memory_mode_attribute() const {
#if CUDA_VERSION >= 13040
    if (cuda_version >= 13040) {
        CUlaunchAttribute attribute = {};
        attribute.id = CU_LAUNCH_ATTRIBUTE_SHARED_MEMORY_MODE;
        attribute.value.sharedMemoryMode = CU_SHARED_MEMORY_MODE_ALLOW_OVERSIZED_SHARED_MEMORY;
        return attribute;
    }
#endif
    return std::nullopt;
}

static Result<cuGetProcAddress_v2_t> get_cuGetProcAddress_from_python() {
    PyPtr load_libcuda_mod = steal(PyImport_ImportModule("cuda.tile._load_libcuda"));
    if (!load_libcuda_mod) return ErrorRaised;

    PyPtr cuGetProcAddr_pyobj = steal(PyObject_GetAttrString(
            load_libcuda_mod.get(), "cuGetProcAddress_v2_ptrptr"));
    if (!cuGetProcAddr_pyobj) return ErrorRaised;

    cuGetProcAddress_v2_t* cuGetProcAddr_pp = reinterpret_cast<cuGetProcAddress_v2_t*>(
            PyLong_AsSize_t(cuGetProcAddr_pyobj.get()));
    if (PyErr_Occurred()) return ErrorRaised;

    return *cuGetProcAddr_pp;
}


static constexpr int MIN_DRIVER_VERSION = 13000;

Result<const DriverApi*> get_driver_api(GlobalLock& lock) {
    static bool initialized;
    static DriverApi instance;
    if (!initialized) {
        Result<cuGetProcAddress_v2_t> proc_addr_res = get_cuGetProcAddress_from_python();
        if (!proc_addr_res.is_ok())
            return ErrorRaised;
        if (!driver_api_init(&instance, *proc_addr_res))
            return ErrorRaised;
        CUresult res = instance.cuInit(0);
        if (res != CUDA_SUCCESS)
            return raise(PyExc_RuntimeError, "cuInit: ", get_cuda_error(&instance, res));
        res = instance.cuDriverGetVersion(&instance.cuda_version);
        if (res != CUDA_SUCCESS)
            return raise(PyExc_RuntimeError, "cuDriverGetVersion: ",
                         get_cuda_error(&instance, res));
        if (!check_driver_version(&instance, MIN_DRIVER_VERSION))
            return ErrorRaised;
        initialized = true;
    }
    return &instance;
}

static Status call_library_lifecycle(PyObject* module_obj, const char* function_name,
                                     CUlibrary library) {
    PyPtr function = getattr(module_obj, function_name);
    if (!function) return ErrorRaised;
    PyPtr handle = steal(PyLong_FromVoidPtr(reinterpret_cast<void*>(library)));
    if (!handle) return ErrorRaised;
    PyPtr result = steal(PyObject_CallOneArg(function.get(), handle.get()));
    if (!result) return ErrorRaised;
    return OK;
}

CudaLibrary::CudaLibrary(const DriverApi* driver, CUlibrary lib)
    : driver_(driver), lib_(lib) {}


CudaLibrary::CudaLibrary(CudaLibrary&& other)
    : driver_(other.driver_), lib_(other.lib_),
      lifecycle_modules_(std::move(other.lifecycle_modules_)) {
    other.lib_ = nullptr;
}


CudaLibrary::~CudaLibrary() {
    if (lib_) {
        if (!lifecycle_modules_.empty()) {
            ErrorGuard guard;
            for (size_t i = lifecycle_modules_.size(); i != 0; i--) {
                PyObject* module_obj = lifecycle_modules_[i - 1].get();
                if (!call_library_lifecycle(module_obj, "library_finalize", lib_))
                    PyErr_WriteUnraisable(module_obj);
            }
        }
        CUresult res = driver_->cuLibraryUnload(lib_);
        CHECK(res == CUDA_SUCCESS);
    }
}


Status CudaLibrary::initialize_lifecycle_providers(PyObject* providers) {
    CHECK(lib_);
    CHECK(lifecycle_modules_.empty());
    CHECK(PyTuple_Check(providers));

    Py_ssize_t num_providers = PyTuple_GET_SIZE(providers);
    for (Py_ssize_t i = 0; i < num_providers; ++i) {
        if (!PyModule_Check(PyTuple_GET_ITEM(providers, i)))
            return raise(PyExc_TypeError,
                         "library lifecycle providers must be modules");
    }

    lifecycle_modules_.reserve(num_providers);
    for (Py_ssize_t i = 0; i < num_providers; ++i) {
        PyObject* module_obj = PyTuple_GET_ITEM(providers, i);
        if (!call_library_lifecycle(module_obj, "library_init", lib_))
            return ErrorRaised;
        lifecycle_modules_.push_back(newref(module_obj));
    }
    return OK;
}


bool CudaLibrary::has_lifecycle_providers() const {
    return !lifecycle_modules_.empty();
}


const CUlibrary& CudaLibrary::get() const {
    return lib_;
}


static Result<CudaLibrary> load_cuda_library(const DriverApi* driver, const void* code) {
    CUlibrary lib;
    CUresult res = driver->cuLibraryLoadData(&lib, code, nullptr, nullptr, 0,
                                             nullptr, nullptr, 0);
    if (res == CUDA_SUCCESS)
        return CudaLibrary(driver, lib);

    return raise(PyExc_RuntimeError, "Failed to load CUDA library: ",
                 get_cuda_error(driver, res));
}


Result<CudaKernel> load_cuda_kernel(
        const DriverApi* driver,
        const char* cubin_data,
        size_t cubin_size,
        const char* func_name) {
    (void) cubin_size;

    Result<CudaLibrary> lib = load_cuda_library(driver, cubin_data);
    if (!lib.is_ok()) return ErrorRaised;

    CUkernel kernel;
    CUresult res = driver->cuLibraryGetKernel(&kernel, lib->get(), func_name);
    if (res == CUDA_SUCCESS)
        return CudaKernel{std::move(*lib), kernel};

    return raise(PyExc_RuntimeError, "Failed to get kernel ", func_name, " from library: ",
                 get_cuda_error(driver, res));
}


Status CudaContextGuard::switch_to(CUcontext target) {
    if (!target) return OK;

    CUcontext current;
    CUresult res = driver->cuCtxGetCurrent(&current);
    if (res != CUDA_SUCCESS) {
        return raise(PyExc_RuntimeError, "Failed to get current CUDA context: ",
                     get_cuda_error(driver, res));
    }
    if (current == target) return OK;

    res = driver->cuCtxPushCurrent(target);
    if (res != CUDA_SUCCESS) {
        return raise(PyExc_RuntimeError, "Failed to switch CUDA context: ",
                     get_cuda_error(driver, res));
    }
    need_to_pop = true;
    return OK;
}


CudaContextGuard::~CudaContextGuard() {
    if (need_to_pop) {
        CUcontext old;
        CUresult res = driver->cuCtxPopCurrent(&old);
        CHECK(res == CUDA_SUCCESS);
    }
}
