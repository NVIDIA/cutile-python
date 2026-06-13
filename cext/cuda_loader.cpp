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
              name, static_cast<int>(res));
        return nullptr;
    }

    if (!ret) {
        raise(PyExc_RuntimeError,
              "Function '%s' is not available in the CUDA library",
              name);
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
