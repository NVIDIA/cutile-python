// SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0

#include "cuda_helper.h"
#include "cuda_loader.h"
#include "vec.h"

#include <utility>


const char* get_cuda_error(const DriverApi* driver, CUresult res) {
    const char* str = nullptr;
    driver->cuGetErrorString(res, &str);
    return str ? str : "Unknown error";
}

Status check_driver_version(const DriverApi* driver, int minimum_version) {
    int version;
    CUresult res = driver->cuDriverGetVersion(&version);
    if (res != CUDA_SUCCESS) {
        PyErr_Format(PyExc_RuntimeError, "cuDriverGetVersion: %s", get_cuda_error(driver, res));
        return ErrorRaised;
    }
    if (version < minimum_version) {
        int major = version / 1000;
        int minor = (version % 1000) / 10;
        int required_major = minimum_version / 1000;
        PyErr_Format(PyExc_RuntimeError,
                     "Minimum driver version required is %d.0, got %d.%d",
                     required_major, major, minor);
        return ErrorRaised;
    }
    return OK;
}

PyObject* get_max_grid_size(PyObject *self, PyObject *args) {
    int device_id;
    if (!PyArg_ParseTuple(args, "i", &device_id))
        return nullptr;

    Result<const DriverApi*> driver = get_driver_api();
    if (!driver.is_ok()) return nullptr;

    CUdevice dev;
    CUresult res = (*driver)->cuDeviceGet(&dev, device_id);
    if (res != CUDA_SUCCESS)
        return PyErr_Format(PyExc_RuntimeError, "cuDeviceGet: %s", get_cuda_error(*driver, res));

    int max_grid_size[3];
    for (int i = 0; i < 3; ++i) {
        res = (*driver)->cuDeviceGetAttribute(&max_grid_size[i],
            static_cast<CUdevice_attribute>(CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X + i),
            dev);
        if (res != CUDA_SUCCESS) {
            return PyErr_Format(PyExc_RuntimeError,
                                "cuDeviceGetAttribute: %s", get_cuda_error(*driver, res));
        }
    }
    return Py_BuildValue("(iii)", max_grid_size[0], max_grid_size[1], max_grid_size[2]);
}

static constexpr int32_t kDeviceCapabilityUnavailable = -1;

static const Vec<ComputeCapability>* get_compute_capability_by_device(const DriverApi* driver) {
    // Protected by the GIL or g_compute_capability_mutex
    static Vec<ComputeCapability>* cached;
#ifdef Py_GIL_DISABLED
    static PyMutex g_compute_capability_mutex = {0};
    PyCriticalSectionGuard guard(&g_compute_capability_mutex);
#endif
    if (cached) return cached;

    int device_count = 0;
    CUresult res = driver->cuDeviceGetCount(&device_count);
    if (res != CUDA_SUCCESS) {
        PyErr_Format(PyExc_RuntimeError, "cuDeviceGetCount: %s", get_cuda_error(driver, res));
        return nullptr;
    }

    Vec<ComputeCapability> table(device_count);
    for (int i = 0; i < device_count; ++i) {
        table[i].major = kDeviceCapabilityUnavailable;

        CUdevice dev;
        if (driver->cuDeviceGet(&dev, i) != CUDA_SUCCESS) continue;

        int major, minor;
        if (driver->cuDeviceGetAttribute(
                &major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev) != CUDA_SUCCESS)
            continue;
        if (driver->cuDeviceGetAttribute(
                &minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev) != CUDA_SUCCESS)
            continue;
        table[i].major = major;
        table[i].minor = minor;
    }

    cached = new Vec<ComputeCapability>(std::move(table));
    return cached;
}

Result<ComputeCapability> get_device_compute_capability(const DriverApi* driver, int device_id) {
    const Vec<ComputeCapability>* compute_capability_by_device =
            get_compute_capability_by_device(driver);
    if (!compute_capability_by_device) return ErrorRaised;

    size_t device_count = compute_capability_by_device->size();
    if (device_id < 0 || static_cast<size_t>(device_id) >= device_count) {
        return raise(PyExc_RuntimeError, "invalid device ordinal %d (%zu device(s) present)",
                     device_id, device_count);
    }

    const ComputeCapability& entry = (*compute_capability_by_device)[device_id];
    if (entry.major == kDeviceCapabilityUnavailable) {
        return raise(PyExc_RuntimeError,
                     "Failed to query the compute capability of device %d", device_id);
    }
    return entry;
}

PyObject* get_compute_capability(PyObject *self, PyObject *args) {
    int device_id = 0;
    if (!PyArg_ParseTuple(args, "|i", &device_id)) return nullptr;

    Result<const DriverApi*> driver_result = get_driver_api();
    if (!driver_result.is_ok()) return nullptr;

    Result<ComputeCapability> computeCapability =
            get_device_compute_capability(*driver_result, device_id);
    if (!computeCapability.is_ok()) return nullptr;
    return Py_BuildValue("(ii)", computeCapability->major, computeCapability->minor);
}

PyObject* get_driver_version(PyObject *self, PyObject *Py_UNUSED(ignored)) {
    int major, minor;

    Result<const DriverApi*> driver_result = get_driver_api();
    if (!driver_result.is_ok()) return nullptr;
    const DriverApi* d = *driver_result;

    CUresult res = d->cuDriverGetVersion(&major);
    if (res != CUDA_SUCCESS) {
        return PyErr_Format(PyExc_RuntimeError, "cuDriverGetVersion: %s", get_cuda_error(d, res));
    }
    minor = (major % 1000) / 10;
    major = major / 1000;
    return Py_BuildValue("(ii)", major, minor);
}

// ========== Context helpers ==========

PyObject* synchronize_context(PyObject* self, PyObject* Py_UNUSED(ignored)) {
    Result<const DriverApi*> driver_result = get_driver_api();
    if (!driver_result.is_ok()) return nullptr;
    const DriverApi* d = *driver_result;

    CUresult res = d->cuCtxSynchronize();
    if (res != CUDA_SUCCESS) {
        return PyErr_Format(PyExc_RuntimeError,
                            "cuCtxSynchronize: %s", get_cuda_error(d, res));
    }
    Py_RETURN_NONE;
}

// ========== Stream helpers ==========

PyObject* create_stream(PyObject* self, PyObject* Py_UNUSED(ignored)) {
    Result<const DriverApi*> driver_result = get_driver_api();
    if (!driver_result.is_ok()) return nullptr;
    const DriverApi* d = *driver_result;

    CUstream stream;
    CUresult res = d->cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING);
    if (res != CUDA_SUCCESS) {
        return PyErr_Format(PyExc_RuntimeError,
                            "cuStreamCreate: %s", get_cuda_error(d, res));
    }
    return PyLong_FromVoidPtr(stream);
}

PyObject* destroy_stream(PyObject* self, PyObject* arg) {
    CUstream stream = static_cast<CUstream>(PyLong_AsVoidPtr(arg));
    if (PyErr_Occurred()) return nullptr;

    Result<const DriverApi*> driver_result = get_driver_api();
    if (!driver_result.is_ok()) return nullptr;
    const DriverApi* d = *driver_result;

    CUresult res = d->cuStreamDestroy(stream);
    if (res != CUDA_SUCCESS) {
        return PyErr_Format(PyExc_RuntimeError,
                            "cuStreamDestroy: %s", get_cuda_error(d, res));
    }
    Py_RETURN_NONE;
}

static decltype(cuLaunchKernelEx)* g_real_cuLaunchKernelEx; // Protected by the GIL or g_spy_mutex
static PyObject* g_cuLaunchKernelEx_spy_callback; // Protected by the GIL or g_spy_mutex

#ifdef Py_GIL_DISABLED
static PyMutex g_spy_mutex = {0};
#endif

static CUresult shim_cuLaunchKernelEx(
        const CUlaunchConfig *config,
        CUfunction f,
        void** kernelParams,
        void** extra) {
#ifdef Py_GIL_DISABLED
    PyCriticalSectionGuard guard(&g_spy_mutex);
#endif
    PyPtr res = steal(PyObject_CallFunction(
            g_cuLaunchKernelEx_spy_callback,
            "(K III III I K)",
            reinterpret_cast<unsigned long long>(f),
            config->gridDimX, config->gridDimY, config->gridDimZ,
            config->blockDimX, config->blockDimY, config->blockDimZ,
            config->sharedMemBytes,
            reinterpret_cast<unsigned long long>(config->hStream)
    ));
    if (!res) return CUDA_ERROR_LAUNCH_FAILED;

    return g_real_cuLaunchKernelEx(config, f, kernelParams, extra);
}

static PyObject* spy_on_cuLaunchKernel_begin(PyObject* self, PyObject* arg) {
#ifdef Py_GIL_DISABLED
    PyCriticalSectionGuard guard(&g_spy_mutex);
#endif
    if (g_real_cuLaunchKernelEx)
        return PyErr_Format(PyExc_RuntimeError, "Already spying");

    Result<const DriverApi*> driver_result = get_driver_api();
    if (!driver_result.is_ok()) return nullptr;

    DriverApi* api = const_cast<DriverApi*>(*driver_result);
    g_real_cuLaunchKernelEx = api->cuLaunchKernelEx;
    g_cuLaunchKernelEx_spy_callback = Py_NewRef(arg);
    api->cuLaunchKernelEx = shim_cuLaunchKernelEx;
    return Py_NewRef(Py_None);
}

static PyObject* spy_on_cuLaunchKernel_end(PyObject* self, PyObject* arg) {
#ifdef Py_GIL_DISABLED
    PyCriticalSectionGuard guard(&g_spy_mutex);
#endif
    if (!g_real_cuLaunchKernelEx)
        return PyErr_Format(PyExc_RuntimeError, "Not spying");

    Result<const DriverApi*> driver_result = get_driver_api();
    if (!driver_result.is_ok()) return nullptr;

    DriverApi* api = const_cast<DriverApi*>(*driver_result);
    api->cuLaunchKernelEx = g_real_cuLaunchKernelEx;
    g_real_cuLaunchKernelEx = nullptr;
    Py_CLEAR(g_cuLaunchKernelEx_spy_callback);
    return Py_NewRef(Py_None);
}

static PyMethodDef functions[] = {
    {"get_compute_capability", get_compute_capability, METH_VARARGS,
        "Get compute capability of a CUDA device, given device id (default 0)"},
    {"get_driver_version", get_driver_version, METH_NOARGS,
        "Get the cuda driver version"},
    {"_get_max_grid_size", get_max_grid_size, METH_VARARGS,
        "Get max grid size of a CUDA device, given device id"},
    {"_synchronize_context", synchronize_context, METH_NOARGS,
        "Synchronize the current CUDA context (drain all streams)."},
    {"_create_stream", create_stream, METH_NOARGS,
        "Create a non-blocking CUDA stream. Returns int handle."},
    {"_destroy_stream", destroy_stream, METH_O,
        "Destroy a CUDA stream given its int handle."},
    {"_spy_on_cuLaunchKernel_begin", spy_on_cuLaunchKernel_begin, METH_O, nullptr},
    {"_spy_on_cuLaunchKernel_end", spy_on_cuLaunchKernel_end, METH_NOARGS, nullptr},
    {}
};

Status cuda_helper_init(PyObject* m) {
    if (PyModule_AddFunctions(m, functions) < 0)
        return ErrorRaised;

    return OK;
}
