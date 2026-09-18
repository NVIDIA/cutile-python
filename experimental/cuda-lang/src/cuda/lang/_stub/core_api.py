# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from typing import TypeVar, Generic

import cuda.lang as cl
from cuda.lang._execution import stub, function
from cuda.lang._exception import TypeCheckingError
from cuda.tile._stub import (
    Array as TileArray,
    cdiv as tile_cdiv,
    static_assert,
    static_eval,
)
from cuda.lang._enums import AtomicOp, MemoryOrder, ShuffleKind
from cuda.tile._memory_model import MemoryScope, MemorySpace
from cuda.lang._datatype import DType, uint32, uint64
from .types import Pointer, Scalar, Vector

T = TypeVar("T")


class LocalArrayContextManager(Generic[T]):
    @stub
    def __enter__(self) -> "Array[T]": ...

    @stub
    def __exit__(self, exc_type, exc_val, exc_tb): ...


class Array(TileArray, Generic[T]):
    """
    N-dimensional array type.
    """

    @staticmethod
    @stub(compiled_host=True)
    def from_parts(
        pointer: Pointer[T],
        shape: int | tuple[int, ...],
        strides: int | tuple[int, ...] | None = None,
    ) -> "Array[T]":
        """Create an array from a pointer, shape, and optional strides.

        The pointer sets the array data type, so the pointer must be typed.
        If ``strides`` is ``None``, the array uses contiguous row-major strides.

        Args:
            pointer: Typed pointer to the first array element.
            shape: Number of elements in each array dimension.
            strides: Step in elements for each dimension.

        Returns:
            Array that refers to the memory at ``pointer``.

        Examples:

        .. testcode::
            :template: kernel_wrapper.py

            smem_array = cl.shared_array(1, cl.int32)
            smem_array[0] = 5

            smem_ptr = smem_array.pointer()
            smem_array_2 = cl.Array.from_parts(smem_ptr, 1)

            # Assignment through the new array changes memory referenced by the
            # original array.
            smem_array_2[0] = 7
            print(smem_array[0])

        .. testoutput::

            7
        """

    @property
    @stub(compiled_host=True)
    def dtype(self): ...

    @property
    @stub(compiled_host=True)
    def ndim(self): ...

    @property
    @stub(compiled_host=True)
    def shape(self): ...

    @property
    @stub(compiled_host=True)
    def strides(self): ...

    @stub(compiled_host=True)
    def pointer(
        self, index_or_indices: int | tuple[int, ...] | None = None
    ) -> "Pointer[T]":
        """Return a pointer to an array element.

        With no index, return a pointer to the first element. With a scalar
        index or tuple of scalar indices, return a pointer to the specified
        element. This is equivalent to ``&array[index]`` in CUDA C++.
        """
        ...

    @stub
    def __setitem__(self, indices: int | tuple[int, ...], value: T):
        """Assign ``value`` to index given by ``indices``.
        Equivalent to ``self.pointer(indices).store(value)``.
        """
        ...

    @stub
    def __getitem__(self, indices: int | tuple[int, ...]) -> T:
        """Retrieve the value given by ``indices``.
        Equivalent to ``self.pointer(indices).load()``.
        """
        ...


@stub(host=True)
def dtype_of(value, /) -> DType:
    """
    Returns the data type of a scalar, pointer, or vector value.
    """
    if isinstance(value, Scalar | Pointer | Vector):
        return value._var.get_type().tensor_dtype()
    elif isinstance(value, bool | int | float):
        from cuda.tile._ir.typing_support import dtype_of_constant_scalar
        return dtype_of_constant_scalar(value)
    else:
        raise TypeCheckingError(
            f"dtype_of() expects a scalar, pointer, or vector as the argument,"
            f" got {type(value)}"
        )


FULL_MASK = 0xFFFFFFFF


@function(host=True)
def full_mask() -> int:
    """Return a warp mask with all lanes selected."""
    return FULL_MASK


@stub
def thread_index(axis: int, /) -> int:
    """Gets the index of the current thread in a block.

    `axis` must be an integer constant equal to 0, 1 or 2.

    For each `axis`, the returned value is an ``int32`` guaranteed to satisfy

        0 <= thread_index(axis) < thread_count(axis).
    """


@stub
def thread_count(axis: int, /) -> int:
    """Gets the number of threads in a block.

    `axis` must be an integer constant equal to 0, 1 or 2.
    Returns an ``int32``.
    """


@stub
def block_index(axis: int, /) -> int:
    """Gets the index of the current block in the grid.

    `axis` must be an integer constant equal to 0, 1 or 2.

    For each `axis`, the returned value is an ``int32`` guaranteed to satisfy

        0 <= block_index(axis) < block_count(axis).
    """


@stub
def block_count(axis: int, /) -> int:
    """Gets the total number of blocks in the grid.

    `axis` must be an integer constant equal to 0, 1 or 2.
    Returns an ``int32``.
    """


@function()
def lane_index() -> int:
    """
    Gets the index of the current thread within its warp.

    The returned value is an ``int32`` guaranteed to satisfy

        0 <= lane_index() < lane_count().
    """
    return nvvm.read_ptx_sreg_laneid()


@function()
def lane_count() -> int:
    """Gets the number of threads in a warp (also known as warp size).

    Returns a loosely typed constant.
    """
    return 32


@function()
def warp_index() -> int:
    """Gets the current virtual warp index within its thread block."""
    tx, ty, tz = thread_index(0), thread_index(1), thread_index(2)
    bdx, bdy = thread_count(0), thread_count(1)
    tid = tx + ty * bdx + tz * bdx * bdy
    return tid // lane_count()


@function()
def warp_count() -> int:
    bdx, bdy, bdz = thread_count(0), thread_count(1), thread_count(2)
    return (bdx * bdy * bdz - 1) // lane_count() + 1


@stub
def cluster_index(axis: int, /) -> int:
    """Gets the index of the current cluster in the grid.

    `axis` must be an integer constant equal to 0, 1 or 2.

    For each `axis`, the returned value is an ``int32`` guaranteed to satisfy

        0 <= cluster_index(axis) < cluster_count(axis).
    """


@stub
def cluster_count(axis: int, /) -> int:
    """Gets the total number of clusters in the grid.

    `axis` must be an integer constant equal to 0, 1 or 2.
    Returns an ``int32``.
    """


@stub
def block_in_cluster_index(axis: int, /) -> int:
    """Gets the index of the current block within its cluster.

    `axis` must be an integer constant equal to 0, 1 or 2.

    For each `axis`, the returned value is an ``int32`` guaranteed to satisfy

        0 <= block_in_cluster_index(axis) < block_in_cluster_count(axis).
    """


@stub
def block_in_cluster_count(axis: int, /) -> int:
    """Get the number of blocks in a cluster.

    `axis` must be an integer constant equal to 0, 1 or 2.
    Returns an ``int32``.
    """


@stub
def shared_array(
    shape: int | tuple[int, ...],
    dtype: DType,
    dynamic: bool = False,
    alignment: int | None = None,
) -> Array[T]:
    """Create an on-device array in shared memory.

    Args:
        shape (int | tuple[int, ...]):
        dtype (DType):
        dynamic (bool):
        alignment (int | None):

    Shared arrays must be declared at the beginning of the kernel.
    The optional alignment is specified in bytes and must be a positive power of
    two.

    If `dynamic` is `False` (default), the array will be placed in the statically allocated
    shared memory. In this case, `shape` must be a compile-time constant.

    If `dynamic` is `True`, the array will be placed in the dynamically allocated shared memory
    (regardless of whether the provided shape is actually constant).
    In this case, `shape` is allowed to be dynamic. However, only a restricted set of expressions
    is allowed to be used for the dynamic shape: currently, only referencing an integer
    kernel parameter directly is permitted.

    Static shared memory example:

    .. testcode::
        :template: setup_only.py

        @cl.kernel
        def kernel():
            shmem = cl.shared_array(32, cl.int32)
            tx = cl.thread_index(0)
            if tx == 0:
                shmem[0] = 42

            cl.barrier_sync_block_aligned()

            if tx == 1:
                print(f"thread id {tx} sees shmem[0] = {shmem[0]}")

        cl.launch(stream, (1,), (2,), kernel, ())

    .. testoutput::

        thread id 1 sees shmem[0] = 42

    Dynamic shared memory example:

    .. testcode::
        :template: setup_only.py

        @cl.kernel
        def kernel(n):
            shmem = cl.shared_array(n, cl.int32, dynamic=True)
            tx = cl.thread_index(0)
            if tx == 0:
                shmem[0] = 42

            cl.barrier_sync_block_aligned()

            if tx == 1:
                print(f"thread id {tx} sees shmem[0] = {shmem[0]}")

        cl.launch(stream, (1,), (2,), kernel, (32,))

    .. testoutput::

        thread id 1 sees shmem[0] = 42
    """


@stub
def local_array(
    shape: int | tuple[int, ...],
    dtype: DType,
    alignment: int | None = None,
) -> LocalArrayContextManager:
    """Create an on-device array in local memory.

    Args:
        shape (int | tuple[int, ...]):
        dtype (DType):
        alignment (int | None):

    Local arrays must be declared in a `with` statement and have static shape.
    The local memory is only valid inside the with block.
    The optional alignment is specified in bytes and must be a positive power
    of two.

    Examples:

        .. testcode::
            :template: setup_only.py

            @cl.kernel
            def kernel(out):
                tx = cl.thread_index(0)
                with cl.local_array(shape=(2,), dtype=cl.int32) as tmp:
                    tmp[0] = tx
                    tmp[1] = tx + 10
                    out[tx] = tmp[0] + tmp[1]

            out = torch.empty(2, dtype=torch.int32, device="cuda:0")
            cl.launch(stream, (1,), (2,), kernel, (out,))
            torch.cuda.synchronize()
            print(out.cpu().tolist())

        .. testoutput::

            [10, 12]
    """


@stub
def setmaxregister_increase(number_of_registers):
    """Hint to change the number of registers owned by the warp.

    Args:
        number_of_registers: Number of registers requested for this warp.
            Must be in the range [24, 256] and a multiple of 8.
    """
    pass


@stub
def setmaxregister_decrease(number_of_registers):
    """Hint to change the number of registers owned by the warp.

    Args:
        number_of_registers: Number of registers requested for this warp.
            Must be in the range [24, 256] and a multiple of 8.
    """
    pass


@stub
def elect_sync(membermask: int = FULL_MASK, /) -> bool:
    """Return whether the caller is the elected thread in ``membermask``."""
    pass


@stub
def vote_all_sync(predicate: bool, mask: int = FULL_MASK) -> bool:
    """Return whether ``predicate`` is true for all lanes in ``mask``.

    Args:
        predicate: Per-lane boolean value.
        mask: Mask indicating membership where bit ``i`` selects lane ``i``.
    """


@stub
def vote_any_sync(predicate: bool, mask: int = FULL_MASK) -> bool:
    """Return whether ``predicate`` is true for one or more lanes in ``mask``.

    Args:
        predicate: Per-lane boolean value.
        mask: Mask indicating membership where bit ``i`` selects lane ``i``.
    """


@stub
def vote_uniform_sync(predicate: bool, mask: int = FULL_MASK) -> bool:
    """Return whether ``predicate`` has the same value in all lanes in ``mask``.

    Args:
        predicate: Per-lane boolean value.
        mask: Mask indicating membership where bit ``i`` selects lane ``i``.
    """


@stub
def vote_ballot_sync(predicate: bool, mask: int = FULL_MASK) -> int:
    """Return the selected lanes for which ``predicate`` is true.

    Args:
        predicate: Per-lane boolean value.
        mask: Mask indicating membership where bit ``i`` selects lane ``i``.
    """


@stub
def _inline_ptx(ptx_code: str, /, *args: Scalar | Pointer | DType) -> tuple:
    """Execute inline PTX.

    Args:
        ptx_code (str):
            The PTX source string. May include placeholders of the form "%N" (e.g., %0, %1, ...)
            to refer to the registers specified by `args`.
        *args:
            For each used placeholder in `ptx_code`, either a scalar/pointer that represents
            an input argument, or a DType that specifies an output.

    Returns:

        The returned tuple depends on the number of write-only input arguments:
        - no write-only outputs: `()`
        - one write-only output: `(value,)`
        - multiple write-only outputs: `(value0, value1, ...)`

    Examples:

        .. testcode::
            :template: kernel_wrapper.py

            i = 12
            j = 30

            # CUDA C++ would use:
            # asm("add.s32 %0, %1, %2;" : "=r"(result) : "r"(i), "r"(j));

            (result,) = cl._inline_ptx("add.s32 %0, %1, %2;", cl.int32, i, j)
            print(f"result: {result}")

        .. testoutput::

            result: 42

    Notes:
        - See CUDA C++ documentation for more details on the `asm` statement.
            https://docs.nvidia.com/cuda/inline-ptx-assembly/index.html
        - Constraint type strings map to data types as follows:
            - ``h``: ``cl.int16``
            - ``r``: ``cl.int32``
            - ``l``: ``cl.int64``
            - ``f``: ``cl.float32``
            - ``d``: ``cl.float64``
            - ``p``: pointer value, or a pointer dtype for an output
        - ``p`` is not in CUDA C++'s inline ptx. The compiler selects the
          correct register size for the pointer based on its address space.
        - Use ``%0``, ``%1``, and so on for operands. Escape literal percent
          signs with a second percent sign, as in ``%%clock``.

    """


@function()
def ptx_comment(comment: str):
    _inline_ptx(cl.static_eval("// " + comment))


@function()
def clock(dtype=uint32):
    static_assert(dtype in (uint32, uint64))
    fn = static_eval(
        nvvm.read_ptx_sreg_clock if dtype is uint32 else nvvm.read_ptx_sreg_clock64
    )
    return fn()


@stub
def atomic_rmw(
    op: AtomicOp,
    ptr: Pointer[T],
    operand: T,
    operand2: T | None = None,
    /,
    *,
    memory_order: MemoryOrder = MemoryOrder.ACQ_REL,
    memory_scope: MemoryScope = MemoryScope.DEVICE,
    alignment: int | None = None,
) -> T:
    """Atomically modify one value and return its old value.

    ``AtomicOp.CAS`` uses ``operand`` as the expected old value and
    ``operand2`` as the new value. It requires both operands. Other operations
    use only ``operand`` and require ``operand2`` to be ``None``.

    Args:
        op: Operation to perform.
        ptr: Pointer to the value to modify.
        operand: Operand for the operation, or the expected value for ``CAS``.
        operand2: New value. Only valid for compare-and-swap.
        memory_order: Memory order for the operation.
        memory_scope: Scope of threads that participate in memory ordering.
        alignment: Minimum byte alignment that the compiler can assume. The
            value must be a positive power of two. If the value is ``None``,
            the natural alignment of the pointee data type is used.

    Returns:
        Original value at ``ptr`` before the operation.

    Examples:

        .. testcode::
            :template: kernel_2d_array_wrapper.py

            array[2, 3] = 7
            ptr = array.pointer((2, 3))
            prev_val = cl.atomic_rmw(cl.AtomicOp.ADD, ptr, 3)
            print(f"after: {array[2, 3]}, prev_val: {prev_val}")

        .. testoutput::

            after: 10, prev_val: 7
    """


@stub
def shuffle_sync(
    kind: ShuffleKind,
    value: T,
    offset: int,
    width: int = 32,
    mask: int = FULL_MASK,
) -> tuple[T, bool]:
    """Exchange register data between threads of a warp.

    Args:
        kind: Compile-time constant `ShuffleKind` indicating the shuffle operation.
        value: This lane's input. Must be an ``int32``, ``uint32``, or ``float32``
            scalar. The returned value has the same type.
        offset: specifies a source lane or source lane offset
            (depending on kind).
        width: Sub-warp width: 1, 2, 4, 8, 16, or 32.
        mask: Integer participation mask. Defaults to all 32 lanes.

    Returns:
        ``(shuffled_value, in_range)``. ``shuffled_value`` is the result of
        the shuffle operation and ``in_range`` is a predicate indicating if
        the computed source lane index is valid.

    .. testcode::
        :template: kernel_wrapper.py

        lane = cl.lane_index()
        value, in_range = cl.shuffle_sync(cl.ShuffleKind.INDEX, lane, 0)
        cl.assert_(value == 0)
        cl.assert_(in_range)
    """


@stub
def address_space_cast(value: Pointer[T], memory_space: MemorySpace) -> Pointer[T]:
    """Cast a pointer to the given memory space, preserving dtype.

    Args:
        value (pointer): Pointer value to be casted.
        memory_space (MemorySpace): Address space of the resulting pointer.

    .. testcode::
        :template: kernel_wrapper.py

        smem = cl.shared_array(1, cl.int32)
        smem_ptr = smem.pointer()
        generic_ptr = cl.address_space_cast(smem_ptr, cl.MemorySpace.GENERIC)

    """


@stub
def map_shared_to_cluster(pointer: Pointer[T], rank: int) -> Pointer[T]:
    """
    Map a pointer in shared memory from another CTA within the same cluster
    with rank ``rank`` to this CTA.
    The pointer is expected to have memory space
    ``MemorySpace.SHARED`` and a pointer with memory space
    ``MemorySpace.SHARED_CLUSTER`` is returned.
    Corresponds to the ptx instruction ``mapa.shared::cluster``.

    Args:
        pointer: Address to be mapped.
        rank (int): Rank of the destination CTA within the cluster.
    """


@function(host=True)
def shared_cluster_leader_bit_mask() -> int:
    """
    Masks off the 24th bit of an address in shared memory.
    The shared state space is accessible to all blocks in a thread-block
    cluster and masking off the 24th bit yields an address corresponding to
    the block with an even numbered index within the cluster.
    This mask is the same as CuTe's Sm100MmaPeerBitMask.
    """
    return 0xFEFFFFFF


@stub
def map_shared_to_leader_block(pointer: Pointer[T]) -> Pointer[T]:
    """
    Map a shared-memory pointer to the same offset in the leader block of its
    two-block group while preserving the pointer type.

    Args:
        pointer: Address in shared memory to be mapped into the leader-block's
            shared memory.
    """


def nanosleep(nanoseconds: int):
    """
    Sleep for ``nanoseconds`` nanoseconds.
    """
    nvvm.nanosleep(nanoseconds)


@stub
def memory_barrier(scope: MemoryScope) -> None:
    """Issue a memory fence with the given scope."""


@stub
def grid_dependency_control_wait() -> None:
    """Wait for prerequisite grids in a programmatic dependent launch."""


@stub
def grid_dependency_control_launch_dependents() -> None:
    """Launch dependent grids in a programmatic dependent launch."""


@stub
def bitcast(x, /, dtype):
    """Reinterpret a value as being of specified data type.
    """


def assert_(condition, /, message=None) -> None:
    """Assert that ``condition`` evaluates to True.

    Args:
        condition (bool): Boolean scalar.
        message: Optional message to be shown when the assertion fails.

    Notes:
        This operation has significant overhead, and should only be used
        for debugging purpose.
    """
    if not condition:
        tid = thread_index(0), thread_index(1), thread_index(2)
        bid = block_index(0), block_index(1), block_index(2)
        print(
            "Assertion failed on thread index",
            tid,
            "block index",
            bid,
            message if message is not None else "",
        )
        nanosleep(10)
        _inline_ptx("trap;")


def cdiv(x, y, /):
    """Computes ceil(x / y). Can be used on the host.

    Args:
        x: integer scalar.
        y: integer scalar.

    Returns:
        scalar
    """
    return tile_cdiv(x, y)


# Need these imports at the end in order to overcome the circular import problem
from . import nvvm  # noqa: E402
