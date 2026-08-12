# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.18)
project(cuda-tile-python)
enable_testing()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")

if (MSVC)
    set(CMAKE_CXX_STANDARD 20)
else()
    set(CMAKE_CXX_STANDARD 17)
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(ALLOW_WARNINGS "Don't treat warnings as errors" OFF)
if (NOT ALLOW_WARNINGS)
    if (MSVC)
        add_compile_options(/WX)
    else()
        add_compile_options(-Wall -Werror)
    endif()
else()
    if (MSVC)
        add_compile_options(/WX-)
    else()
        add_compile_options(-Wall)
    endif()
endif()

option(ENABLE_DEV_FEATURES "Enable development-only features" OFF)

option(ENABLE_COVERAGE_FOR_TESTS "Enable code coverage for tests" OFF)
set(test_coverage_options)

if (ENABLE_COVERAGE_FOR_TESTS)
    if ((CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
            OR (CMAKE_CXX_COMPILER_ID STREQUAL "Clang"))
        set(test_coverage_options "--coverage")
    else()
        message(FATAL_ERROR "Not sure how to enable coverage "
                            "for the current compiler ${CMAKE_CXX_COMPILER_ID}.")
    endif()
    message(STATUS "Enabling code coverage for tests")
endif()


# You may need something like
#    ASAN_OPTIONS=protect_shadow_gap=0 \
#       LD_PRELOAD="/lib/x86_64-linux-gnu/libasan.so.8 \
#                   /usr/lib/x86_64-linux-gnu/libstdc++.so.6" python ...
# to run the address sanitizer.
option(ENABLE_ASAN "Enable address sanitizer" OFF)
if (ENABLE_ASAN)
    if ((CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
            OR (CMAKE_CXX_COMPILER_ID STREQUAL "Clang"))
        add_compile_options(-fsanitize=address)
        add_link_options(-fsanitize=address)
    else()
        message(FATAL_ERROR "Not sure how to enable address sanitizer "
                            "for the current compiler ${CMAKE_CXX_COMPILER_ID}.")
    endif()
    message(STATUS "Enabling address sanitizer")
endif()



find_package(Python REQUIRED COMPONENTS Interpreter Development)
find_package(CUDAToolkit REQUIRED)

add_custom_target(devinstall
    COMMAND ln -fs "${CMAKE_BINARY_DIR}/cext/lib_cext.so"
            ${CMAKE_SOURCE_DIR}/cuda/tile/_cext.so)


if (DLPACK_PATH)
    set(dlpack_INCLUDE_DIR "${DLPACK_PATH}/include/dlpack")
    message(STATUS "Using local dlpack, include path: ${dlpack_INCLUDE_DIR}")
else()
    include(FetchContent)
    FetchContent_Declare(
        dlpack
        GIT_REPOSITORY https://github.com/dmlc/dlpack.git
        GIT_TAG v1.1
    )
    FetchContent_MakeAvailable(dlpack)
    set(dlpack_INCLUDE_DIR "${dlpack_SOURCE_DIR}/include/dlpack")
    message(STATUS "Fetching dlpack")
endif()

include(cmake/FetchXLAHeaders.cmake)
fetch_xla_headers()

add_subdirectory(cext)


option(DISABLE_INTERNAL "disable internal" OFF)
if (NOT DISABLE_INTERNAL)
    message(STATUS "Try build internal extension")
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/internal/CMakeLists.txt")
        message(STATUS "Including internal/ overlay")
        add_subdirectory(internal)
    else()
        message(STATUS "Building without internal/")
    endif()
endif()
