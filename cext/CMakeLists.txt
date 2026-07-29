# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

if (MSVC)
    # /GR-: -fno-rtti
    # /GS-: disable buffer security check to avoid exception
    # /EHs-: disable synchronous exception handling
    # Remove default /EHs to avoid overriding warning
    # Default visibility is already hidden
    string(REPLACE "/EHsc" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
    set(cext_compile_flags /GR- /GS- /EHs- -DNDEBUG)
    # /NODEFAULTLIB and /NOENTRY to disable dependency on runtime lib msvcrt.dll
    set(cext_link_flags /NODEFAULTLIB /NOENTRY)
    # set Py_GIL_DISABLED macro, at it is not defined automatically on Windows
    execute_process(
        COMMAND ${Python_EXECUTABLE} -c
            "import sysconfig; print(sysconfig.get_config_var('Py_GIL_DISABLED') or 0)"
        OUTPUT_VARIABLE PY_GIL_DISABLED
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    if (PY_GIL_DISABLED EQUAL 1)
        list(APPEND cext_compile_flags -DPy_GIL_DISABLED=1)
    endif()
else()
    set(cext_compile_flags -fno-exceptions -fno-rtti -fPIC -fvisibility=hidden)
    # For arm64 add flag to disable calls to out-of-line helpers for atomic operations 
    # to avoid undefined reference error during compilation
    if (CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64)$")
        list(APPEND cext_compile_flags -mno-outline-atomics)
    endif()
    set(cext_link_flags ${cext_compile_flags})
    set(nostdlib_flags -nostdlib -DNDEBUG -fno-builtin)
    # Strip the absolute path from __FILE__
    list(APPEND cext_compile_flags "-fmacro-prefix-map=${CMAKE_SOURCE_DIR}/=")
endif()


set(cext_include_dirs
    ${dlpack_INCLUDE_DIR}
    ${xla_INCLUDE_DIR}
    ${Python_INCLUDE_DIRS}
    ${CUDAToolkit_INCLUDE_DIRS}
)


# Build a static library first, so that we could reuse it for several build targets

add_library(_cext_static STATIC
    coroutine_util.cpp
    cuda_loader.cpp
    cuda_helper.cpp
    ipc_util.cpp
    memory.cpp
    py.cpp
    stream_buffer.cpp
    tile_kernel.cpp
    xla_ffi.cpp
    xla_ffi_py.cpp
    )

target_compile_options(_cext_static PUBLIC ${cext_compile_flags} ${nostdlib_flags})
target_include_directories(_cext_static PRIVATE ${cext_include_dirs})
target_link_options(_cext_static PUBLIC ${cext_link_flags} ${nostdlib_flags})


# Build a Python extension

add_library(_cext MODULE module.cpp)

target_compile_options(_cext PUBLIC ${cext_compile_flags} ${nostdlib_flags})
target_include_directories(_cext PRIVATE ${cext_include_dirs})
if (MSVC)
    # on windows linker resolve symbols eagerly so we must explicitly provide libpython.so
    target_link_libraries(_cext PUBLIC _cext_static ${Python_LIBRARIES})
else()
    target_link_libraries(_cext PUBLIC _cext_static)
endif()
target_link_options(_cext PUBLIC ${cext_link_flags} ${nostdlib_flags})

set_target_properties(_cext PROPERTIES SKIP_BUILD_RPATH TRUE)

if (ENABLE_DEV_FEATURES)
    target_compile_definitions(_cext PRIVATE CUDA_TILE_ENABLE_DEV_FEATURES)
endif()

if (APPLE)
    target_link_options(_cext PRIVATE -undefined dynamic_lookup)
endif()


# Link another shared library with --no-undefined to make sure we don't have any unresolved symbols.

if (ENABLE_ASAN)
    set(asan_library asan)
endif()

add_library(_cext_shared SHARED module.cpp)
target_link_libraries(_cext_shared _cext_static ${Python_LIBRARIES} ${asan_library})
target_compile_options(_cext_shared PUBLIC ${cext_compile_flags} ${nostdlib_flags})
target_include_directories(_cext_shared PRIVATE ${cext_include_dirs})
target_link_options(_cext_shared PUBLIC ${cext_link_flags} ${nostdlib_flags} -Wl,--no-undefined)

# Tests

add_executable(test_stream_buffer
    test/test_stream_buffer.cpp
    cuda_loader.cpp
    cuda_helper.cpp
    memory.cpp
    )
target_compile_options(test_stream_buffer PUBLIC ${cext_compile_flags} ${test_coverage_options})
target_include_directories(test_stream_buffer PRIVATE ${cext_include_dirs})
target_link_libraries(test_stream_buffer PRIVATE ${Python_LIBRARIES})
target_link_options(test_stream_buffer PRIVATE ${test_coverage_options})


add_executable(test_hash_map test/test_hash_map.cpp memory.cpp)
target_include_directories(test_hash_map PRIVATE ${cext_include_dirs})
target_compile_options(test_hash_map PUBLIC ${cext_compile_flags} ${test_coverage_options})
target_link_options(test_hash_map PRIVATE ${test_coverage_options})
target_link_libraries(test_hash_map PRIVATE ${Python_LIBRARIES})


add_executable(test_vec test/test_vec.cpp memory.cpp)
target_include_directories(test_vec PRIVATE ${cext_include_dirs})
target_compile_options(test_vec PUBLIC ${cext_compile_flags} ${test_coverage_options})
target_link_options(test_vec PRIVATE ${test_coverage_options})
target_link_libraries(test_vec PRIVATE ${Python_LIBRARIES})
