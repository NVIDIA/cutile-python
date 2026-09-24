# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0


FOREIGN_CALL_SYMBOL_PREFIXES = ("nvshmem_", "nvshmemx_")


def _get_nvshmem_core():
    try:
        import nvshmem.core
    except ImportError as error:
        raise ImportError(
            "NVSHMEM foreign calls require the nvshmem Python package"
        ) from error
    return nvshmem.core


def library_init(culibrary_handle: int) -> None:
    """Initialize NVSHMEM state for a loaded CUDA library."""
    nvshmem_core = _get_nvshmem_core()
    kernel_object = nvshmem_core.NvshmemKernelObject.from_handle(culibrary_handle)
    nvshmem_core.library_init(kernel_object)


def library_finalize(culibrary_handle: int) -> None:
    """Finalize NVSHMEM state for a loaded CUDA library."""
    nvshmem_core = _get_nvshmem_core()
    kernel_object = nvshmem_core.NvshmemKernelObject.from_handle(culibrary_handle)
    nvshmem_core.library_finalize(kernel_object)
