# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from typing import TypeVar, Generic, Literal

from cuda.lang._execution import stub, function
from cuda.tile._stub import (
    Array as TileArray,
    static_eval,
)
from cuda.tile._memory_model import MemoryOrder, MemoryScope, MemorySpace
from cuda.lang._datatype import DType, int32
from .types import Pointer, Scalar, Vector
from cuda.tile._exception import TileTypeError

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

    @stub
    def get_base_pointer(self) -> "Pointer[T]": ...

    @stub
    def get_element_pointer(self, indices: int | tuple[int, ...]) -> "Pointer[T]": ...

    @stub
    def __setitem__(self, indices: int | tuple[int, ...], value: T): ...

    @stub
    def __getitem__(self, indices: int | tuple[int, ...]) -> T: ...

    @function
    def load_element(
        self,
        indices: int | tuple[int, ...],
        *,
        count: int | None = None,
        alignment: int | None = None,
        volatile: bool = False,
        memory_order: MemoryOrder | None = None,
    ) -> "T | Vector[T]":
        """Load the element at ``indices``.

        Shorthand for ``self.get_element_pointer(indices).load(...)``: the
        element pointer is derived from ``indices`` and the keyword arguments
        (``count``, ``alignment``, ``volatile``, ``memory_order``) are forwarded
        unchanged to :meth:`Pointer.load`, which documents their semantics.

        Args:
            indices: Scalar or tuple index of the element to load.

        Returns:
            The loaded scalar, or a vector of ``count`` elements when ``count``
            is given.
        """
        return self.get_element_pointer(indices).load(
            count=count,
            alignment=alignment,
            volatile=volatile,
            memory_order=memory_order,
        )

    @function
    def store_element(
        self,
        indices: int | tuple[int, ...],
        value: "T | Vector[T]",
        *,
        alignment: int | None = None,
        volatile: bool = False,
        memory_order: Literal[MemoryOrder.RELAXED,
                              MemoryOrder.RELEASE, MemoryOrder.WEAK] | None = None,
    ) -> None:
        """Store ``value`` to the element at ``indices``.

        Shorthand for ``self.get_element_pointer(indices).store(value, ...)``:
        the element pointer is derived from ``indices`` and the keyword
        arguments (``alignment``, ``volatile``, ``memory_order``) are forwarded
        unchanged to :meth:`Pointer.store`, which documents their semantics.

        Args:
            indices: Scalar or tuple index of the element to store to.
            value: Scalar or vector value to store.
        """
        self.get_element_pointer(indices).store(
            value,
            alignment=alignment,
            volatile=volatile,
            memory_order=memory_order,
        )


@stub(host=True)
def dtype_of(value, /) -> DType:
    """
    Returns the data type of a scalar or pointer value.
    """
    if isinstance(value, Scalar):
        # We expect SymbolicScalar here
        return value._var.get_type().dtype
    elif isinstance(value, Pointer):
        # Similarly, we expect SymbolicPointer here
        return value._var.get_type().pointer_dtype
    elif isinstance(value, bool | int | float):
        from cuda.tile._ir.typing_support import dtype_of_constant_scalar
        return dtype_of_constant_scalar(value)
    else:
        raise TileTypeError(f"dtype_of() expects a scalar or a pointer as the argument,"
                            f" got {type(value)}")


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


@function
def lane_index() -> int:
    """
    Gets the index of the current thread within its warp.

    The returned value is an ``int32`` guaranteed to satisfy

        0 <= lane_index() < lane_count().
    """
    return nvvm.read_ptx_sreg_laneid()


@function
def lane_count() -> int:
    """Gets the number of threads in a warp (also known as warp size).

    Returns a loosely typed constant.
    """
    return 32


@function
def warp_index() -> int:
    """Gets the current virtual warp index within its thread block."""
    tx, ty, tz = thread_index(0), thread_index(1), thread_index(2)
    bdx, bdy = thread_count(0), thread_count(1)
    tid = tx + ty * bdx + tz * bdx * bdy
    return tid // lane_count()


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
            shmem = cl.shared_array(shape=(32,), dtype=cl.int32)
            tx = cl.thread_index(0)
            if tx == 0:
                shmem[0] = 42

            cl.barrier_sync_block()

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
            shmem = cl.shared_array(shape=(n,), dtype=cl.int32, dynamic=True)
            tx = cl.thread_index(0)
            if tx == 0:
                shmem[0] = 42

            cl.barrier_sync_block()

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

            out = torch.empty(2, dtype=torch.int32, device="cuda")
            cl.launch(stream, (1,), (2,), kernel, (out,))
            torch.cuda.synchronize()
            print(out.cpu().tolist())

        .. testoutput::

            [10, 12]
    """


@function
def setmaxregister_increase(value: int32):
    nvvm.setmaxnreg_inc_sync_aligned_u32(int32(value))


@function
def setmaxregister_decrease(value: int32):
    nvvm.setmaxnreg_dec_sync_aligned_u32(int32(value))


@stub
def elect_sync(membermask: int = FULL_MASK, /) -> bool:
    """Return whether the caller is the elected thread in ``membermask``."""
    pass


@stub
def _inline_ptx(ptx_code: str, *constraint_pairs: tuple) -> tuple:
    """Execute inline PTX.

    The API mirrors CUDA C++'s device-side `asm` statement:
    `cl._inline_ptx(ptx_code, (constraint1, value1), (constraint2, value2), ...)`.

    Args:

        ptx_code (str):
            The PTX source string.

        *constraint_pairs:
            Constraint/value pairs.
            Constraints must be compile-time constant strings.

            Read-only operands use constraints ``"h"``, ``"r"``, ``"l"``,
            ``"f"``, ``"d"``, or ``"C"`` and are paired with runtime values.

            Write-only operands use constraints ``"=h"``, ``"=r"``, ``"=l"``,
            ``"=f"``, or ``"=d"`` and are paired with dtype specs.
            This determines the type of the output.

            Read-write operands use constraints ``"+h"``, ``"+r"``, ``"+l"``,
            ``"+f"``, ``"+d"``, or ``"+C"`` and are paired with runtime values.

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

            (result,) = cl._inline_ptx(
                "add.s32 %0, %1, %2;",
                ("=r", cl.int32),
                ("r", i),
                ("r", j),
            )
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
            - ``C``: pointer value from ``array.get_base_pointer()``

    """


@function
def ptx_comment(comment: str):
    _inline_ptx(static_eval("// " + comment))


@stub
def atomic_add(
    ptr: Pointer[T],
    val: T,
    /,
    *,
    memory_order: MemoryOrder = MemoryOrder.ACQ_REL,
    memory_scope: MemoryScope = MemoryScope.DEVICE,
) -> T:
    """
    Perform atomic ``ptr.store(ptr.load() + val)``.

    Args:
        ptr: Pointer to the value to update atomically.
        val: Operand for the atomic operation.
        memory_order: Memory ordering for the atomic operation. Defaults to
            ``MemoryOrder.ACQ_REL``.
        memory_scope: Memory scope for the atomic operation. Defaults to
            ``MemoryScope.DEVICE``.

    Returns:
        Original value at ``ptr`` before the operation.

    Supported ``T``: ``int32``, ``uint32``, ``int64``, ``uint64``,
    ``float16``, ``bfloat16``, ``float32``, and ``float64``.

    Examples:

        .. testcode::
            :template: kernel_2d_array_wrapper.py

            print(f"before: {array[2, 3]}")
            ptr = array.get_element_pointer((2, 3))
            prev_val = cl.atomic_add(ptr, 1)
            print(f"after: {array[2, 3]}, prev_val: {prev_val}")

        .. testoutput::

            before: 0
            after: 1, prev_val: 0

    """


@stub
def atomic_sub(
    ptr: Pointer[T],
    val: T,
    /,
    *,
    memory_order: MemoryOrder = MemoryOrder.ACQ_REL,
    memory_scope: MemoryScope = MemoryScope.DEVICE,
) -> T:
    """
    Perform atomic ``ptr.store(ptr.load() - val)``.

    Args:
        ptr: Pointer to the value to update atomically.
        val: Operand for the atomic operation.
        memory_order: Memory ordering for the atomic operation. Defaults to
            ``MemoryOrder.ACQ_REL``.
        memory_scope: Memory scope for the atomic operation. Defaults to
            ``MemoryScope.DEVICE``.

    Returns:
        Original value at ``ptr`` before the operation.

    Supported ``T``: ``int32``, ``uint32``, ``int64``, ``uint64``,
    ``float32``, and ``float64``.

    Examples:

        .. testcode::
            :template: kernel_2d_array_wrapper.py

            print(f"before: {array[2, 3]}")
            ptr = array.get_element_pointer((2, 3))
            prev_val = cl.atomic_sub(ptr, 1)
            print(f"after: {array[2, 3]}, prev_val: {prev_val}")

        .. testoutput::

            before: 0
            after: -1, prev_val: 0

    """


@stub
def atomic_and(
    ptr: Pointer[T],
    val: T,
    /,
    *,
    memory_order: MemoryOrder = MemoryOrder.ACQ_REL,
    memory_scope: MemoryScope = MemoryScope.DEVICE,
) -> T:
    """
    Perform atomic ``ptr.store(ptr.load() & val)``.

    Args:
        ptr: Pointer to the value to update atomically.
        val: Operand for the atomic operation.
        memory_order: Memory ordering for the atomic operation. Defaults to
            ``MemoryOrder.ACQ_REL``.
        memory_scope: Memory scope for the atomic operation. Defaults to
            ``MemoryScope.DEVICE``.

    Returns:
        Original value at ``ptr`` before the operation.

    Supported ``T``: ``int32``, ``uint32``, ``int64``, and ``uint64``.

    Examples:

        .. testcode::
            :template: kernel_2d_array_wrapper.py

            array[2, 3] = 14  # 0b1110
            print(f"before: {array[2, 3]}")
            ptr = array.get_element_pointer((2, 3))
            prev_val = cl.atomic_and(ptr, 11)  # 0b1011
            print(f"after: {array[2, 3]}, prev_val: {prev_val}")

        .. testoutput::

            before: 14
            after: 10, prev_val: 14

    """


@stub
def atomic_or(
    ptr: Pointer[T],
    val: T,
    /,
    *,
    memory_order: MemoryOrder = MemoryOrder.ACQ_REL,
    memory_scope: MemoryScope = MemoryScope.DEVICE,
) -> T:
    """
    Perform atomic ``ptr.store(ptr.load() | val)``.

    Args:
        ptr: Pointer to the value to update atomically.
        val: Operand for the atomic operation.
        memory_order: Memory ordering for the atomic operation. Defaults to
            ``MemoryOrder.ACQ_REL``.
        memory_scope: Memory scope for the atomic operation. Defaults to
            ``MemoryScope.DEVICE``.

    Returns:
        Original value at ``ptr`` before the operation.

    Supported ``T``: ``int32``, ``uint32``, ``int64``, and ``uint64``.

    Examples:

        .. testcode::
            :template: kernel_2d_array_wrapper.py

            array[2, 3] = 12  # 0b1100
            print(f"before: {array[2, 3]}")
            ptr = array.get_element_pointer((2, 3))
            prev_val = cl.atomic_or(ptr, 3)  # 0b0011
            print(f"after: {array[2, 3]}, prev_val: {prev_val}")

        .. testoutput::

            before: 12
            after: 15, prev_val: 12

    """


@stub
def atomic_xor(
    ptr: Pointer[T],
    val: T,
    /,
    *,
    memory_order: MemoryOrder = MemoryOrder.ACQ_REL,
    memory_scope: MemoryScope = MemoryScope.DEVICE,
) -> T:
    """
    Perform atomic ``ptr.store(ptr.load() ^ val)``.

    Args:
        ptr: Pointer to the value to update atomically.
        val: Operand for the atomic operation.
        memory_order: Memory ordering for the atomic operation. Defaults to
            ``MemoryOrder.ACQ_REL``.
        memory_scope: Memory scope for the atomic operation. Defaults to
            ``MemoryScope.DEVICE``.

    Returns:
        Original value at ``ptr`` before the operation.

    Supported ``T``: ``int32``, ``uint32``, ``int64``, and ``uint64``.

    Examples:

        .. testcode::
            :template: kernel_2d_array_wrapper.py

            array[2, 3] = 12  # 0b1100
            print(f"before: {array[2, 3]}")
            ptr = array.get_element_pointer((2, 3))
            prev_val = cl.atomic_xor(ptr, 10)  # 0b1010
            print(f"after: {array[2, 3]}, prev_val: {prev_val}")

        .. testoutput::

            before: 12
            after: 6, prev_val: 12

    """


@stub
def atomic_min(
    ptr: Pointer[T],
    val: T,
    /,
    *,
    memory_order: MemoryOrder = MemoryOrder.ACQ_REL,
    memory_scope: MemoryScope = MemoryScope.DEVICE,
) -> T:
    """
    Perform atomic ``ptr.store(min(ptr.load(), val))``.

    Args:
        ptr: Pointer to the value to update atomically.
        val: Operand for the atomic operation.
        memory_order: Memory ordering for the atomic operation. Defaults to
            ``MemoryOrder.ACQ_REL``.
        memory_scope: Memory scope for the atomic operation. Defaults to
            ``MemoryScope.DEVICE``.

    Returns:
        Original value at ``ptr`` before the operation.

    Supported ``T``: ``int32``, ``uint32``, ``int64``, ``uint64``,
    ``float32``, and ``float64``.

    Examples:

        .. testcode::
            :template: kernel_2d_array_wrapper.py

            array[2, 3] = 7
            print(f"before: {array[2, 3]}")
            ptr = array.get_element_pointer((2, 3))
            prev_val = cl.atomic_min(ptr, 3)
            print(f"after: {array[2, 3]}, prev_val: {prev_val}")

        .. testoutput::

            before: 7
            after: 3, prev_val: 7

    """


@stub
def atomic_max(
    ptr: Pointer[T],
    val: T,
    /,
    *,
    memory_order: MemoryOrder = MemoryOrder.ACQ_REL,
    memory_scope: MemoryScope = MemoryScope.DEVICE,
) -> T:
    """
    Perform atomic ``ptr.store(max(ptr.load(), val))``.

    Args:
        ptr: Pointer to the value to update atomically.
        val: Operand for the atomic operation.
        memory_order: Memory ordering for the atomic operation. Defaults to
            ``MemoryOrder.ACQ_REL``.
        memory_scope: Memory scope for the atomic operation. Defaults to
            ``MemoryScope.DEVICE``.

    Returns:
        Original value at ``ptr`` before the operation.

    Supported ``T``: ``int32``, ``uint32``, ``int64``, ``uint64``,
    ``float32``, and ``float64``.

    Examples:

        .. testcode::
            :template: kernel_2d_array_wrapper.py

            array[2, 3] = 7
            print(f"before: {array[2, 3]}")
            ptr = array.get_element_pointer((2, 3))
            prev_val = cl.atomic_max(ptr, 11)
            print(f"after: {array[2, 3]}, prev_val: {prev_val}")

        .. testoutput::

            before: 7
            after: 11, prev_val: 7

    """


@stub
def atomic_inc(
    ptr: Pointer[T],
    val: T,
    /,
    *,
    memory_order: MemoryOrder = MemoryOrder.ACQ_REL,
    memory_scope: MemoryScope = MemoryScope.DEVICE,
) -> T:
    """
    Perform atomic increment at ``ptr`` with wrap at ``val``.

    This behaves as ``ptr.store(0 if ptr.load() >= val else ptr.load() + 1)``.
    Supported ``T``: ``uint32`` only.

    Args:
        ptr: Pointer to the value to update atomically.
        val: Wrap threshold for the atomic increment.
        memory_order: Memory ordering for the atomic operation. Defaults to
            ``MemoryOrder.ACQ_REL``.
        memory_scope: Memory scope for the atomic operation. Defaults to
            ``MemoryScope.DEVICE``.

    Returns:
        Original value at ``ptr`` before the operation.

    Examples:

        .. testcode::
            :template: kernel_2d_uint32_array_wrapper.py

            array[2, 3] = 7
            print(f"before: {array[2, 3]}")
            ptr = array.get_element_pointer((2, 3))
            prev_val = cl.atomic_inc(ptr, 7)
            print(f"after: {array[2, 3]}, prev_val: {prev_val}")

        .. testoutput::

            before: 7
            after: 0, prev_val: 7

    """


@stub
def atomic_dec(
    ptr: Pointer[T],
    val: T,
    /,
    *,
    memory_order: MemoryOrder = MemoryOrder.ACQ_REL,
    memory_scope: MemoryScope = MemoryScope.DEVICE,
) -> T:
    """
    Perform atomic decrement at ``ptr`` with wrap at ``val``.

    This stores ``val`` when the current value is ``0`` or greater than
    ``val``; otherwise, it decrements the current value by one.
    Supported ``T``: ``uint32`` only.

    Args:
        ptr: Pointer to the value to update atomically.
        val: Wrap threshold for the atomic decrement.
        memory_order: Memory ordering for the atomic operation. Defaults to
            ``MemoryOrder.ACQ_REL``.
        memory_scope: Memory scope for the atomic operation. Defaults to
            ``MemoryScope.DEVICE``.

    Returns:
        Original value at ``ptr`` before the operation.

    Examples:

        .. testcode::
            :template: kernel_2d_uint32_array_wrapper.py

            array[2, 3] = 0
            print(f"before: {array[2, 3]}")
            ptr = array.get_element_pointer((2, 3))
            prev_val = cl.atomic_dec(ptr, 7)
            print(f"after: {array[2, 3]}, prev_val: {prev_val}")

        .. testoutput::

            before: 0
            after: 7, prev_val: 0

    """


@stub
def atomic_xchg(
    ptr: Pointer[T],
    val: T,
    /,
    *,
    memory_order: MemoryOrder = MemoryOrder.ACQ_REL,
    memory_scope: MemoryScope = MemoryScope.DEVICE,
) -> T:
    """
    Perform atomic exchange ``ptr.store(val)``.

    Args:
        ptr: Pointer to the value to update atomically.
        val: Value to store atomically.
        memory_order: Memory ordering for the atomic operation. Defaults to
            ``MemoryOrder.ACQ_REL``.
        memory_scope: Memory scope for the atomic operation. Defaults to
            ``MemoryScope.DEVICE``.

    Returns:
        Original value at ``ptr`` before the operation.

    Supported ``T``: ``int32``, ``uint32``, ``float32``, ``int64``,
    ``uint64``, and ``float64``.

    Examples:

        .. testcode::
            :template: kernel_2d_array_wrapper.py

            array[2, 3] = 7
            print(f"before: {array[2, 3]}")
            ptr = array.get_element_pointer((2, 3))
            prev_val = cl.atomic_xchg(ptr, 4)
            print(f"after: {array[2, 3]}, prev_val: {prev_val}")

        .. testoutput::

            before: 7
            after: 4, prev_val: 7

    """


@stub
def atomic_cas(
    ptr: Pointer[T],
    old: T,
    val: T,
    /,
    *,
    memory_order: MemoryOrder = MemoryOrder.ACQ_REL,
    memory_scope: MemoryScope = MemoryScope.DEVICE,
) -> T:
    """
    Perform atomic compare-and-swap at ``ptr``.

    If the current value equals ``old``, store ``val``.

    Args:
        ptr: Pointer to the value to update atomically.
        old: Expected value for the compare-and-swap.
        val: Value to store when the current value equals ``old``.
        memory_order: Memory ordering for the atomic operation. Defaults to
            ``MemoryOrder.ACQ_REL``.
        memory_scope: Memory scope for the atomic operation. Defaults to
            ``MemoryScope.DEVICE``.

    Returns:
        Original value at ``ptr`` before the operation.

    Supported ``T``: ``int16``, ``uint16``, ``int32``, ``uint32``,
    ``int64``, and ``uint64``.

    Examples:

        .. testcode::
            :template: kernel_2d_array_wrapper.py

            array[2, 3] = 7
            print(f"before: {array[2, 3]}")
            ptr = array.get_element_pointer((2, 3))
            prev_val = cl.atomic_cas(ptr, 7, 4)
            print(f"after: {array[2, 3]}, prev_val: {prev_val}")

        .. testoutput::

            before: 7
            after: 4, prev_val: 7

    """


@stub
def shfl_sync(value: int, src_lane: int, width: int = 32, mask: int = FULL_MASK) -> int:
    """
    Return ``value`` from lane ``src_lane`` within the logical warp subdivision.
    """


@stub
def shfl_up_sync(value: int, delta: int, width: int = 32, mask: int = FULL_MASK) -> int:
    """
    Return ``value`` from the lane ``delta`` positions lower in the logical warp
    subdivision.
    """


@stub
def shfl_down_sync(value: int, delta: int, width: int = 32, mask: int = FULL_MASK) -> int:
    """
    Return ``value`` from the lane ``delta`` positions higher in the logical
    warp subdivision.
    """


@stub
def shfl_xor_sync(value: int, lane_mask: int, width: int = 32, mask: int = FULL_MASK) -> int:
    """
    Return ``value`` from the lane addressed by XORing the caller lane with
    ``lane_mask`` within the logical warp subdivision.
    """


@stub
def address_space_cast(value: Pointer[T], memory_space: MemorySpace) -> Pointer[T]:
    """
    Cast a pointer to the given memory space.

    .. testcode::
        :template: kernel_wrapper.py

        smem = cl.shared_array(1, cl.int32)
        smem_ptr = smem.get_base_pointer()
        generic_ptr = cl.address_space_cast(smem_ptr, cl.MemorySpace.GENERIC)

    """


@stub
def map_shared_to_cluster(ptr: Pointer[T], rank: int) -> Pointer[T]:
    """
    Map a pointer in shared memory from another CTA within the same cluster
    with rank ``rank`` to this CTA.
    The pointer is expected to have memory space
    ``MemorySpace.SHARED`` and a pointer with memory space
    ``MemorySpace.SHARED_CLUSTER`` is returned.
    Corresponds to the ptx instruction ``mapa.shared::cluster``.
    """


@stub
def reinterpret_pointer_as_array(
    pointer: Pointer[T],
    dtype: DType,
    shape: tuple[int, ...],
    strides: tuple[int, ...] | None = None,
) -> Array[T]:
    """
    Args:
        pointer: Pointer[T]
        dtype: DType
        shape: tuple[int, ...]
        strides: tuple[int, ...] | None = None

    Returns:
        Array with the specified base pointer, dtype, shape, and strides.

    Examples:

    .. testcode::
        :template: kernel_wrapper.py

        smem_array = cl.shared_array(1, cl.int32)
        smem_array[0] = 5

        smem_ptr = smem_array.get_base_pointer()
        smem_array_2 = cl.reinterpret_pointer_as_array(smem_ptr, shape=1, dtype=cl.int32)

        # Assignment through the reconstructed array is equivalent
        # to assignment through the original.
        smem_array_2[0] = 7
        print(smem_array[0])

    .. testoutput::

        7

    """


def nanosleep(nanoseconds: int):
    """
    Sleep for ``nanoseconds`` nanoseconds.
    """
    nvvm.nanosleep(nanoseconds)


def memory_barrier(scope: MemoryScope) -> None:
    """Issue a memory fence with the given scope."""
    return nvvm_mlir_interfaces.memory_barrier(scope=scope)


def griddepcontrol_wait() -> None:
    """Wait for prerequisite grids in a programmatic dependent launch."""
    nvvm_mlir_interfaces.griddepcontrol(
        kind=nvvm_mlir_interfaces.GridDepActionKind.wait
    )


def griddepcontrol_launch_dependents() -> None:
    """Launch dependent grids in a programmatic dependent launch."""
    nvvm_mlir_interfaces.griddepcontrol(
        kind=nvvm_mlir_interfaces.GridDepActionKind.launch_dependents
    )


@stub
def bitcast(x, dtype):
    """
    Cast a value to another type of the same bitwidth.
    """


# Need these imports at the end in order to overcome the circular import problem
from . import nvvm  # noqa: E402
from . import nvvm_mlir_interfaces  # noqa: E402
