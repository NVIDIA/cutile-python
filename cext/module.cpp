// SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0

#include "py.h"

#include "compiled_host.h"
#include "dtype.h"
#include "tile_kernel.h"
#include "cuda_helper.h"
#include "coroutine_util.h"
#include "xla_ffi_py.h"
#include "bitstream.h"
#include "pynvvm.h"
#include "tensor_map.h"

#ifdef _WIN32
extern "C" int _fltused = 0;
extern "C" int __cdecl _purecall() { CHECK(false); }
#endif


static PyModuleDef module_def = {
    .m_base = PyModuleDef_HEAD_INIT,
    .m_name = "cuda.tile._cext",
    .m_size = 0,
};

PyMODINIT_FUNC PyInit__cext() {
    PyPtr m = steal(PyModule_Create(&module_def));
    if (!m) return nullptr;

#ifdef Py_GIL_DISABLED
    if (PyUnstable_Module_SetGIL(m.get(), Py_MOD_GIL_NOT_USED) != 0 )
        return nullptr;
#endif

    if (!dtype_init(m.get()))
        return nullptr;

    if (!tile_kernel_init(m.get()))
        return nullptr;

    if (!compiled_host_init(m.get()))
        return nullptr;

    if (!cuda_helper_init(m.get()))
        return nullptr;

    if (!coroutine_util_init(m.get()))
        return nullptr;

    if (!tensor_map_init(m.get()))
        return nullptr;

    if (!xla_ffi_init(m.get()))
        return nullptr;

    if (!bitstream_init(m.get()))
        return nullptr;

    if (!pynvvm_init(m.get()))
        return nullptr;

    return m.release();
}

