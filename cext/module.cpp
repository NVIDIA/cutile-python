// SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0

#include "py.h"

#include "compiled_host.h"
#include "dtype.h"
#include "enums.h"
#include "error.h"
#include "tile_kernel.h"
#include "cuda_helper.h"
#include "coroutine_util.h"
#include "xla_ffi_py.h"
#include "bitstream.h"
#include "pynvvm.h"
#include "tensor_map.h"
#include "value.h"

#ifdef _WIN32
extern "C" int _fltused = 0;
extern "C" int __cdecl _purecall() { CHECK(false); }
#endif


static int module_exec(PyObject* m) {
    if (enums_module_init() < 0)
        return -1;

    if (!dtype_init(m))
        return -1;

    if (!error_module_init(m))
        return -1;

    if (!value_module_init(m))
        return -1;

    if (!tile_kernel_init(m))
        return -1;

    if (!compiled_host_init(m))
        return -1;

    if (!cuda_helper_init(m))
        return -1;

    if (!coroutine_util_init(m))
        return -1;

    if (!tensor_map_init(m))
        return -1;

    if (!xla_ffi_init(m))
        return -1;

    if (!bitstream_init(m))
        return -1;

    if (!pynvvm_init(m))
        return -1;

    return 0;
}

static PyModuleDef_Slot module_slots[] = {
    {Py_mod_exec, reinterpret_cast<void*>(module_exec)},
#if PY_VERSION_HEX >= 0x030C0000
    // The module state is global
    {Py_mod_multiple_interpreters, Py_MOD_MULTIPLE_INTERPRETERS_NOT_SUPPORTED},
#endif
#ifdef Py_GIL_DISABLED
    {Py_mod_gil, Py_MOD_GIL_NOT_USED},
#endif
    {}
};

static PyModuleDef module_def = {
    .m_base = PyModuleDef_HEAD_INIT,
    .m_name = "cuda.tile._cext",
    .m_size = 0,
    .m_slots = module_slots,
};


PyMODINIT_FUNC PyInit__cext() {
    PyPtr m = steal(PyModuleDef_Init(&module_def));
    if (!m) return nullptr;

    return m.release();
}

