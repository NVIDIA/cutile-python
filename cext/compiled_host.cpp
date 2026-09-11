// SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0

#include "compiled_host.h"

#include "cuda_helper.h"
#include "cuda_loader.h"
#include "tile_kernel.h"
#include "tensor_map.h"
#include "vec.h"

#include <cuda.h>

#include <utility>


namespace {

using CompiledHostEntryFn = int32_t (*)(void** arguments, void* runtime,
                                       GlobalLock* lock);


struct HostRuntimeSymbol {
    const char* symbol;
    void* address;
};

/** Per-program runtime state passed to the compiled host entry. */
struct CompiledHostRuntime {
    Vec<NativeLaunchSite*> launch_sites;

    ~CompiledHostRuntime() {
        for (NativeLaunchSite* site : launch_sites)
            native_launch_site_destroy(site);
    }
};


int32_t host_runtime_launch_kernel(
        CompiledHostRuntime* runtime,
        uint32_t launch_site_index,
        void* stream,
        int64_t grid_x,
        int64_t grid_y,
        int64_t grid_z,
        int64_t block_x,
        int64_t block_y,
        int64_t block_z,
        int64_t cluster_x,
        int64_t cluster_y,
        int64_t cluster_z,
        int64_t preferred_cluster_x,
        int64_t preferred_cluster_y,
        int64_t preferred_cluster_z,
        int32_t has_cluster,
        int32_t has_preferred_cluster,
        void** arguments,
        int32_t cooperative,
        int32_t programmatic_dependent_launch) {
    if (!runtime) {
        PyErr_SetString(PyExc_RuntimeError, "compiled host runtime is null");
        return -1;
    }
    if (launch_site_index >= runtime->launch_sites.size()) {
        PyErr_SetString(PyExc_RuntimeError, "compiled host launch-site index is invalid");
        return -1;
    }
    NativeLaunchConfig config = {
        .stream = stream,
        .grid = {grid_x, grid_y, grid_z},
        .block = {block_x, block_y, block_z},
        .cluster = {cluster_x, cluster_y, cluster_z},
        .preferred_cluster = {
            preferred_cluster_x, preferred_cluster_y, preferred_cluster_z},
        .has_cluster = has_cluster,
        .has_preferred_cluster = has_preferred_cluster,
        .cooperative = cooperative,
        .programmatic_dependent_launch = programmatic_dependent_launch,
    };
    return native_launch_site_launch(
            runtime->launch_sites[launch_site_index], config, arguments);
}


/** Runtime functions available to compiled host code through JITLink. */
const HostRuntimeSymbol host_runtime_symbols[] = {
    {
        "cuda_lang_runtime_launch_kernel",
        reinterpret_cast<void*>(&host_runtime_launch_kernel),
    },
    {
        "cuda_lang_runtime_encode_tensor_map_tiled",
        reinterpret_cast<void*>(&native_tensor_map_encode_tiled),
    },
};


PyObject* get_compiled_host_runtime_symbols(PyObject*, PyObject*) {
    constexpr size_t num_symbols =
            sizeof(host_runtime_symbols) / sizeof(host_runtime_symbols[0]);
    PyPtr result = steal(PyTuple_New(num_symbols));
    if (!result) return nullptr;
    for (size_t i = 0; i < num_symbols; ++i) {
        const HostRuntimeSymbol& symbol = host_runtime_symbols[i];
        PyPtr address = steal(PyLong_FromVoidPtr(symbol.address));
        if (!address) return nullptr;
        PyPtr item = steal(Py_BuildValue(
                "(sO)", symbol.symbol, address.get()));
        if (!item) return nullptr;
        PyTuple_SET_ITEM(result.get(), i, item.release());
    }
    return result.release();
}


struct CompiledHostExecutable {
    CompiledHostEntryFn entry = nullptr;
    PyPtr loaded_host_code;
};


struct CompiledHostProgram {
    CompiledHostExecutable executable;
    CompiledHostRuntime runtime;
    PyPtr compilation;

    static PyTypeObject pytype;
};


int CompiledHostProgram_init(PyObject* self, PyObject* args, PyObject* kwargs) {
    if (kwargs && PyDict_Size(kwargs)) {
        PyErr_SetString(
                PyExc_TypeError, "_CompiledHostProgram accepts positional arguments only");
        return -1;
    }
    PyObject* py_launch_descriptions;
    PyObject* py_loaded_host_code;
    PyObject* py_compilation;
    if (!PyArg_ParseTuple(
                args, "O!OO",
                &PyTuple_Type, &py_launch_descriptions,
                &py_loaded_host_code, &py_compilation))
        return -1;

    PyPtr py_entry = steal(PyObject_GetAttrString(
            py_loaded_host_code, "entry_address"));
    if (!py_entry) return -1;
    CompiledHostEntryFn entry = reinterpret_cast<CompiledHostEntryFn>(
            PyLong_AsVoidPtr(py_entry.get()));
    if (PyErr_Occurred()) return -1;
    if (!entry) {
        PyErr_SetString(
                PyExc_ValueError, "compiled host entry address must not be null");
        return -1;
    }

    CompiledHostProgram& program = py_unwrap<CompiledHostProgram>(self);
    Py_ssize_t num_launch_sites = PyTuple_GET_SIZE(py_launch_descriptions);
    program.runtime.launch_sites.reserve(num_launch_sites);
    for (Py_ssize_t i = 0; i < num_launch_sites; ++i) {
        PyObject* launch_description =
                PyTuple_GET_ITEM(py_launch_descriptions, i);
        PyObject* dispatcher;
        PyObject* arguments;
        PyObject* host_constant_args;
        if (!PyArg_ParseTuple(
                    launch_description, "OOO:native launch description",
                    &dispatcher, &arguments, &host_constant_args))
            return -1;
        if (!PyTuple_Check(arguments)) {
            PyErr_SetString(
                    PyExc_TypeError,
                    "native launch description arguments must be a tuple");
            return -1;
        }
        Result<NativeLaunchSite*> launch_site =
                native_launch_site_create(
                        dispatcher,
                        reinterpret_cast<PyTupleObject*>(arguments)->ob_item,
                        PyTuple_GET_SIZE(arguments),
                        host_constant_args);
        if (!launch_site.is_ok()) return -1;
        program.runtime.launch_sites.push_back(*launch_site);
    }

    program.executable.entry = entry;
    program.executable.loaded_host_code = newref(py_loaded_host_code);
    program.compilation = newref(py_compilation);
    return 0;
}


PyObject* invoke_host_entry(
        CompiledHostProgram& program,
        void** arguments,
        GlobalLock& lock) {
    int32_t result = program.executable.entry(arguments, &program.runtime, &lock);
    if (result < 0) {
        if (PyErr_Occurred()) return nullptr;
        raise(PyExc_RuntimeError, "compiled host code failed with status ", result);
        return nullptr;
    }
    if (result != CUDA_SUCCESS) {
        Result<const DriverApi*> driver_result = get_driver_api(lock);
        if (!driver_result.is_ok()) return nullptr;
        const DriverApi* driver = *driver_result;
        raise(PyExc_RuntimeError, "cuda error occurred: ",
                get_cuda_error(driver, static_cast<CUresult>(result)));
        return nullptr;
    }
    Py_RETURN_NONE;
}


PyObject* CompiledHostProgram_invoke(PyObject* self, PyObject* argument_addresses) {
    if (!PyTuple_Check(argument_addresses)) {
        raise(PyExc_TypeError, "compiled host argument addresses must be a tuple");
        return nullptr;
    }
    GlobalLock lock;
    Py_ssize_t count = PyTuple_GET_SIZE(argument_addresses);
    Vec<void*> arguments;
    arguments.reserve(count);
    for (Py_ssize_t i = 0; i < count; ++i) {
        void* address = PyLong_AsVoidPtr(
                PyTuple_GET_ITEM(argument_addresses, i));
        if (PyErr_Occurred()) return nullptr;
        arguments.push_back(address);
    }
    return compiled_host_program_invoke(self, arguments.data(), lock);
}


PyObject* CompiledHostProgram_get_compilation(PyObject* self, void*) {
    CompiledHostProgram& program = py_unwrap<CompiledHostProgram>(self);
    return Py_NewRef(program.compilation.get());
}


PyObject* CompiledHostProgram_get_entry_address(PyObject* self, void*) {
    CompiledHostProgram& program = py_unwrap<CompiledHostProgram>(self);
    return PyLong_FromVoidPtr(
            reinterpret_cast<void*>(program.executable.entry));
}


PyMethodDef CompiledHostProgram_methods[] = {
    {
        "_invoke",
        CompiledHostProgram_invoke,
        METH_O,
        "Invoke compiled host code with a tuple of native argument addresses.",
    },
    {},
};


PyGetSetDef CompiledHostProgram_getsetters[] = {
    {"_compilation", CompiledHostProgram_get_compilation, nullptr, nullptr},
    {"entry_address", CompiledHostProgram_get_entry_address, nullptr, nullptr},
    {},
};


PyTypeObject CompiledHostProgram::pytype = {
    .tp_name = "cuda.tile._cext._CompiledHostProgram",
    .tp_basicsize = sizeof(PythonWrapper<CompiledHostProgram>),
    .tp_dealloc = pywrapper_dealloc<CompiledHostProgram>,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_methods = CompiledHostProgram_methods,
    .tp_getset = CompiledHostProgram_getsetters,
    .tp_init = CompiledHostProgram_init,
    .tp_new = pywrapper_new<CompiledHostProgram>,
};


PyMethodDef compiled_host_methods[] = {
    {
        "_get_compiled_host_runtime_symbols",
        get_compiled_host_runtime_symbols,
        METH_NOARGS,
        "Return native runtime symbols supported by compiled host code.",
    },
    {},
};

}  // namespace


bool compiled_host_program_check(PyObject* object) {
    return PyObject_TypeCheck(object, &CompiledHostProgram::pytype);
}


PyObject* compiled_host_program_invoke(PyObject* program_object, void** arguments,
                                       GlobalLock& lock) {
    if (!compiled_host_program_check(program_object)) {
        raise(
                PyExc_TypeError,
                "expected a compiled host program, got ",
                Py_TYPE(program_object)->tp_name);
        return nullptr;
    }
    CompiledHostProgram& program = py_unwrap<CompiledHostProgram>(program_object);
    return invoke_host_entry(program, arguments, lock);
}


Status compiled_host_init(PyObject* module) {
    if (PyType_Ready(&CompiledHostProgram::pytype) < 0)
        return ErrorRaised;
    if (PyModule_AddObjectRef(
                module, "_CompiledHostProgram",
                reinterpret_cast<PyObject*>(&CompiledHostProgram::pytype)) < 0)
        return ErrorRaised;
    if (PyModule_AddFunctions(module, compiled_host_methods) < 0)
        return ErrorRaised;
    return OK;
}
