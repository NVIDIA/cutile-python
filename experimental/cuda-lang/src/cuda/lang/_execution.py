# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations
from typing import TypeAlias, Any
from types import FunctionType
from typing import TYPE_CHECKING

from cuda.lang._ir import ir
from ._compiler_options import CompilerOptions
from cuda.tile import _cext
from cuda.tile._cext import launch_extended
from cuda.tile._execution import function, stub

if TYPE_CHECKING:
    from cuda.lang.compilation import KernelSignature


__all__ = (
    "function",
    "kernel",
    "launch",
    "stub",
)


Dim3: TypeAlias = tuple[int] | tuple[int, int] | tuple[int, int, int]


def launch(
    stream,
    block_count: Dim3,
    thread_count: Dim3,
    kernel,
    kernel_args: tuple[Any, ...],
    /,
    *,
    cooperative: bool = False,
    block_in_cluster_count: Dim3 | None = None,
    preferred_block_in_cluster_count: Dim3 | None = None,
    programmatic_dependent_launch: bool = False,
):
    """Launch a cuda.lang kernel.

    Args:
        stream: CUDA stream the kernel launch should be enqueued on. Accepts
            stream objects from PyTorch, CuPy, Numba and cuda.bindings, as well
            as integer-valued raw ``CUStream`` handles.
        block_count (Dim3): Grid dimensions in thread blocks, specified as
            ``(x,)``, ``(x, y)``, or ``(x, y, z)``. Omitted dimensions default
            to 1.
        thread_count (Dim3): Thread-block dimensions in threads, specified as
            ``(x,)``, ``(x, y)``, or ``(x, y, z)``. Omitted dimensions default
            to 1.
        kernel: Kernel to launch. Must be a function decorated with
            :func:`cuda.lang.kernel`.
        kernel_args (tuple[Any, ...]): Positional arguments passed to
            ``kernel``. Their number and order must match the kernel parameters,
            and each value must be a supported kernel argument type.
        cooperative (bool): Whether to use a cooperative grid launch.
        block_in_cluster_count (Dim3 | None): Thread-block cluster dimensions,
            measured in blocks.Each grid dimension must be divisible by
            the corresponding cluster dimension. If ``None``, no explicit
            cluster dimensions are requested.
        preferred_block_in_cluster_count (Dim3 | None): Preferred substitute
            cluster dimensions, measured in blocks. These dimensions will be
            preferred, but when resources are insufficient,
            ``block_in_cluster_count`` may be used instead. Requires
            ``block_in_cluster_count``. Each dimension must be a positive
            integer multiple of the corresponding regular cluster dimension.
        programmatic_dependent_launch (bool): Whether this kernel may resolve
            its dependency on the preceding kernel in the same stream,
            allowing their execution to potentially overlap. The dependent
            kernel must call :func:`cuda.lang.grid_dependency_control_wait`
            to wait for the previous kernel to call
            :func:`cuda.lang.grid_dependency_control_launch_dependents`.

    The launch is asynchronous with respect to the host and is ordered on
    ``stream``.
    """
    launch_extended(
        stream,
        block_count,
        thread_count,
        kernel,
        kernel_args,
        cooperative=cooperative,
        block_in_cluster_count=block_in_cluster_count,
        preferred_block_in_cluster_count=preferred_block_in_cluster_count,
        programmatic_dependent_launch=programmatic_dependent_launch,
    )


class kernel(_cext.TileDispatcher):
    """A |kernel| is a function executed by each |thread| in each |block| in a |grid|.
    See :func:`cuda.lang.launch` for how to execute a kernel on the GPU.

    Examples:

        .. testcode::
            :template: setup_only.py

            @cl.kernel
            def kernel():
                print("Hello!")

            cl.launch(stream, (1,), (3,), kernel, ())

        .. testoutput::

            Hello!
            Hello!
            Hello!

    """

    def __new__(cls, function=None, /, **kwargs):
        if function is None:

            def decorate(func):
                return kernel(func, **kwargs)

            return decorate

        return super().__new__(cls, function, **kwargs)

    def __init__(
        self,
        function=None,
        /,
        *,
        opt_level: int | None = 3,
        arch: str | None = None,
        gpu_name: str | None = None,
        max_threads_per_block: Dim3 | None = None,
        max_blocks_per_cluster: int | None = None,
        max_registers_per_thread: int | None = None,
        min_blocks_per_sm: int | None = None,
    ):
        """
        Args:
            function: Python function to be compiled.
            opt_level (int | None): Optimization level applied to the kernel.
            arch (str): GPU architecture this kernel should be compiled for.
                ``None`` selects an appropriate value for the current device.
            gpu_name (str): GPU name this kernel should be compiled for.
                ``None`` selects an appropriate value for the current device.
            max_threads_per_block (tuple[int, int, int] | None):
            max_blocks_per_cluster (int | None):
            max_registers_per_thread (int | None):
            min_blocks_per_sm (int | None):
        """
        if not isinstance(function, FunctionType):
            raise TypeError("`kernel` decorator must be applied to a Python function")

        from cuda.tile._annotated_function import get_annotated_function
        if isinstance(max_threads_per_block, tuple) and len(max_threads_per_block) < 3:
            max_threads_per_block = (*max_threads_per_block, *(1, 1, 1))[:3]

        ann_func = get_annotated_function(function)
        super().__init__(ann_func.parameter_annotations)
        self._annotated_function = ann_func
        self._compiler_options = CompilerOptions(
            opt_level=opt_level,
            max_threads_per_block=max_threads_per_block,
            max_blocks_per_cluster=max_blocks_per_cluster,
            max_registers_per_thread=max_registers_per_thread,
            min_blocks_per_sm=min_blocks_per_sm,
        )
        self._arch = arch
        self._gpu_name = gpu_name

    def _compile(self, signature: KernelSignature, ctx: ir.IRContext):
        from cuda.lang._compile import compile_simt

        result = compile_simt(
            self._annotated_function,
            (signature,),
            arch=self._arch,
            gpu_name=self._gpu_name,
            compiler_options=self._compiler_options,
            ctx=None,  # the launcher currently provides a cutile context
        )
        [kernel_sig] = result.kernel_signatures
        return (
            result.cubin,
            kernel_sig.symbol,
            result.dyn_smem_size_program,
            result.hoisted_tensor_maps,
        )

    @property
    def _pyfunc(self):
        return self._annotated_function.pyfunc

    def __call__(self, *args, **kwargs):
        raise TypeError(
            "kernels cannot be called directly. Use cuda.lang.launch() instead."
        )
