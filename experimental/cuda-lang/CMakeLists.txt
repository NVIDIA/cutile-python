# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

if (NOT CUDA_LANG_COMPILER_LIBS)
    return()
endif()

set(nvvm_static_library "${CUDA_LANG_COMPILER_LIBS}/nvvm/lib64/libnvvm_solid_internal_static.a")
set(libdevice "${CUDA_LANG_COMPILER_LIBS}/nvvm/libdevice/libdevice.10.bc")
set(ptxas "${CUDA_LANG_COMPILER_LIBS}/bin/ptxas${CMAKE_EXECUTABLE_SUFFIX}")

foreach(input nvvm_static_library libdevice ptxas)
    if (NOT EXISTS "${${input}}" OR IS_DIRECTORY "${${input}}")
        message(FATAL_ERROR "CUDA_LANG_COMPILER_LIBS is missing ${input}: ${${input}}")
    endif()
endforeach()

add_library(nvvm SHARED nvvm_dummy.cpp)
target_link_options(nvvm PRIVATE
    "-Wl,--whole-archive"
    "${nvvm_static_library}"
    "-Wl,--no-whole-archive")

set(libdevice_dst "${CMAKE_CURRENT_BINARY_DIR}/libdevice.10.bc")
add_custom_command(
    OUTPUT "${libdevice_dst}"
    COMMAND ${CMAKE_COMMAND} -E copy "${libdevice}" "${libdevice_dst}"
    DEPENDS "${libdevice}"
)
add_custom_target(copy_libdevice ALL DEPENDS "${libdevice_dst}")

set(ptxas_dst "${CMAKE_CURRENT_BINARY_DIR}/ptxas${CMAKE_EXECUTABLE_SUFFIX}")
add_custom_command(
    OUTPUT "${ptxas_dst}"
    COMMAND ${CMAKE_COMMAND} -E copy "${ptxas}" "${ptxas_dst}"
    DEPENDS "${ptxas}"
)
add_custom_target(copy_ptxas ALL DEPENDS "${ptxas_dst}")
