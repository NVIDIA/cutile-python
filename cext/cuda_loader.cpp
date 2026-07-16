// SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0

#include "cuda_loader.h"
#include "cuda_helper.h"


namespace {


typedef CUresult (*cuGetProcAddress_v2_t)
    (const char *symbol, void **funcPtr, int cudaVersion,
     cuuint64_t flags, CUdriverProcAddressQueryResult *symbolStatus);


void* do_get_proc_address(cuGetProcAddress_v2_t getter,
                          const char* name, int cuda_version) {
    void* ret = nullptr;
    CUresult res = getter(name, &ret, cuda_version, CU_GET_PROC_ADDRESS_DEFAULT, nullptr);
    if (res != CUDA_SUCCESS) {
        raise(PyExc_RuntimeError,
              "Failed to load '%s' from the CUDA library: cuGetProcAddress_v2 returned %d",
              static_cast<int>(res));
        return nullptr;
    }

    if (!ret) {
        raise(PyExc_RuntimeError,
              "Function '%s' is not available in the CUDA library",
              static_cast<int>(res));
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


#define DEFINE_CUDA_FUNCTION_GLOBAL(name, _cuda_version) \
    decltype(name)* g_##name;

FOREACH_CUDA_FUNCTION_TO_LOAD(DEFINE_CUDA_FUNCTION_GLOBAL)

#define GET_PROC_ADDRESS(name, cuda_ver) \
        if (!(driver_api.name = \
                    get_proc_address<decltype(name)*>(_cuGetProcAddress, #name, cuda_ver))) \
            return ErrorRaised;


static Status cuda_loader_init(DriverApi& driver_api) {
    PyPtr load_libcuda_mod = steal(PyImport_ImportModule("cuda.tile._load_libcuda"));
    if (!load_libcuda_mod) return ErrorRaised;

    PyPtr cuGetProcAddr_pyobj = steal(PyObject_GetAttrString(
            load_libcuda_mod.get(), "cuGetProcAddress_v2_ptrptr"));
    if (!cuGetProcAddr_pyobj) return ErrorRaised;

    cuGetProcAddress_v2_t* cuGetProcAddr_pp = reinterpret_cast<cuGetProcAddress_v2_t*>(
            PyLong_AsSize_t(cuGetProcAddr_pyobj.get()));
    if (PyErr_Occurred()) return ErrorRaised;

    cuGetProcAddress_v2_t _cuGetProcAddress = *cuGetProcAddr_pp;

    FOREACH_CUDA_FUNCTION_TO_LOAD(GET_PROC_ADDRESS)

    return OK;
}


static constexpr int MIN_DRIVER_VERSION = 13000;

Result<const DriverApi*> get_driver_api() {
    static bool initialized;
    static DriverApi instance;
    if (!initialized) {
        if (!cuda_loader_init(instance))
            return ErrorRaised;
        CUresult res = instance.cuInit(0);
        if (res != CUDA_SUCCESS)
            return raise(PyExc_RuntimeError, "cuInit: %s", get_cuda_error(&instance, res));
        if (!check_driver_version(&instance, MIN_DRIVER_VERSION))
            return ErrorRaised;
        initialized = true;
    }
    return &instance;
}


CudaLibrary::CudaLibrary(const DriverApi* driver, CUlibrary lib)
    : driver_(driver), lib_(lib) {}


CudaLibrary::CudaLibrary(CudaLibrary&& other)
    : driver_(other.driver_), lib_(other.lib_) {
    other.lib_ = nullptr;
}


CudaLibrary::~CudaLibrary() {
    if (lib_) {
        CUresult res = driver_->cuLibraryUnload(lib_);
        CHECK(res == CUDA_SUCCESS);
    }
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

    return raise(PyExc_RuntimeError, "Failed to load CUDA library: %s",
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

    return raise(PyExc_RuntimeError, "Failed to get kernel %s from library: %s",
                 func_name, get_cuda_error(driver, res));
}


Status CudaContextGuard::switch_to(CUcontext target) {
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
