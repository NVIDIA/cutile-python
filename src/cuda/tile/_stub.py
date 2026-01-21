# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations
import functools
import inspect
import textwrap
from typing import Annotated, TypeVar, Union, Literal, Optional, Protocol

from cuda.tile._memory_model import MemoryOrder, MemoryScope
from cuda.tile._execution import function
from cuda.tile._datatype import DType
from cuda.tile._numeric_semantics import RoundingMode, PaddingMode


###############################################################################
# Types


class ScalarProtocol(Protocol):
    @function
    def __index__(self) -> int:
        """Scalar can be used as index in range"""

    @function
    def __add__(self, other) -> "TileOrScalar":
        ...

    @function
    def __sub__(self, other) -> "TileOrScalar":
        ...

    @function
    def __mul__(self, other) -> "TileOrScalar":
        ...

    @function
    def __truediv__(self, other) -> "TileOrScalar":
        ...

    @function
    def __floordiv__(self, other) -> "TileOrScalar":
        ...

    @function
    def __mod__(self, other) -> "TileOrScalar":
        ...

    @function
    def __pow__(self, other) -> "TileOrScalar":
        ...

    @function
    def __and__(self, other) -> "TileOrScalar":
        ...

    @function
    def __or__(self, other) -> "TileOrScalar":
        ...

    @function
    def __xor__(self, other) -> "TileOrScalar":
        ...

    @function
    def __radd__(self, other) -> "TileOrScalar":
        ...

    @function
    def __rsub__(self, other) -> "TileOrScalar":
        ...

    @function
    def __rmul__(self, other) -> "TileOrScalar":
        ...

    @function
    def __rtruediv__(self, other) -> "TileOrScalar":
        ...

    @function
    def __rfloordiv__(self, other) -> "TileOrScalar":
        ...

    @function
    def __rmod__(self, other) -> "TileOrScalar":
        ...

    @function
    def __rpow__(self, other) -> "TileOrScalar":
        ...

    @function
    def __rand__(self, other) -> "TileOrScalar":
        ...

    @function
    def __ror__(self, other) -> "TileOrScalar":
        ...

    @function
    def __rxor__(self, other) -> "TileOrScalar":
        ...

    @function
    def __ge__(self, other) -> "TileOrScalar":
        ...

    @function
    def __gt__(self, other) -> "TileOrScalar":
        ...

    @function
    def __le__(self, other) -> "TileOrScalar":
        ...

    @function
    def __lt__(self, other) -> "TileOrScalar":
        ...

    @function
    def __eq__(self, other) -> "TileOrScalar":
        ...

    @function
    def __ne__(self, other) -> "TileOrScalar":
        ...


Scalar = int | float | ScalarProtocol


class Array:
    """A *global array* (or *array*) is a container of objects stored in a logical
    multidimensional space.

    |Global arrays| are always stored in memory.
    Copying an |array| does not copy the underlying data.

    |Global arrays| can be used in |host code| and |tile code|.
    They can be |kernel| parameters.

    Any object that implements the |DLPack| format or the |CUDA Array Interface| can be used
    as a |global array|. Example: |CuPy| arrays and |PyTorch| tensors.
    """

    @property
    @function
    def dtype(self) -> "DType":
        """The |data type| of the |array|'s elements.

        Returns:
            DType (constant):
        """

    @property
    @function
    def shape(self) -> tuple[int, ...]:
        """The number of elements in each of the |array|'s dimensions.

        Returns:
            tuple[int32,...]:
        """

    @property
    @function
    def strides(self) -> tuple[int, ...]:
        """The number of elements to step in each dimension while traversing the |array|.

        Returns:
            tuple[int32,...]:
        """

    @property
    @function
    def ndim(self) -> int:
        """The number of dimensions in the |array|.

        Returns:
            int (constant):
        """

    @function
    def slice(self, axis, start, stop) -> "Array":
        """Creates a view of the |array| sliced along a single `axis`.

        The returned array references the same underlying memory as |array|,
        but with a restricted range from index `start` (inclusive) to `stop` (exclusive)
        along the specified axis. No data is copied.

        `axis` must be a constant integer. Negative values are supported and count
        from the last dimension (e.g., ``axis=-1`` refers to the last axis).

        `start` and `stop` must be integers (scalars or 0D tiles).
        They must satisfy ``0 <= start < N`` and ``start <= stop <= N``, where ``N``
        is the size of `array` along the sliced axis.

        For example, consider a 2-dimensional array A of shape ``(M, N)``.
        Slicing along axis 0 from `start` to `stop`:

            >>> sub = A.slice(axis=0, start=start, stop=stop)

        The result `sub` will be an array of shape ``(stop - start, N)``.
        Using NumPy slice notation for illustration, this is equivalent to::

            sub = A[start:stop, :]  # NumPy notation for reference only

        The slice bounds can be dynamic (runtime values):

            >>> # Process variable-length segments
            >>> segment = A.slice(axis=1, start=offset, stop=offset + length)
            >>> tile = ct.load(segment, (0, 0), shape=(TILE_M, TILE_N))
        """


class Tile:
    """A *tile array* (or *tile*) is an immutable multidimensional collection of values that is
    local to a |block|.

    The contents of a |tile| do not necessarily have a representation in memory.
    |Tiles| can be created by loading from |global arrays| or with |factory| functions.
    |Tiles| can also be stored into |global arrays|.

    |Tiles| shall not be used in |host code|; they can only be used in |tile code|.
    |Tiles| shall not be |kernel| parameters.

    Each dimension of a |tile| shall be a power of 2.
    """

    @property
    @function
    def dtype(self) -> "DType":
        """The |data type| of the |tile|'s elements.

        Returns:
            DType (constant):
        """

    @property
    @function
    def shape(self) -> tuple[int, ...]:
        """The number of elements in each of the |tile|'s dimensions.

        Returns:
            tuple[const int,...]:
        """

    @property
    @function
    def ndim(self) -> int:
        """The number of dimensions in the |tile|.

        Returns:
            int (constant):
        """

    @function
    def item(self) -> "Tile":
        """Extract scalar (0D Tile) from a single element tile.
        Tile must contain only 1 element.

        Returns:
            Tile: A 0D Tile usable as a scalar.

        Examples:

            >>> tx = ct.full((1,), 0, dtype=ct.int32)
            >>> x = tx.item()
            >>> ty = ct.load(array, (0, x), shape=(4, 4))
        """

    @function
    def extract(self, index, shape):
        """See :py:func:`extract`."""

    @function
    def reshape(self, shape) -> "Tile":
        """See :py:func:`reshape`."""

    @function
    def permute(self, axes) -> "Tile":
        """See :py:func:`permute`."""

    @function
    def transpose(self, axis0=None, axis1=None) -> "Tile":
        """See :py:func:`transpose`."""

    @function
    def astype(self, dtype) -> "Tile":
        """See :py:func:`astype`."""

    @function
    def __index__(self) -> int:
        """0D Tile can be used as index in range"""

    @function
    def __getitem__(self, index) -> "Tile":
        """Syntax sugar for expand_dim"""

    @function
    def __add__(self, other) -> "Tile":
        ...

    @function
    def __sub__(self, other) -> "Tile":
        ...

    @function
    def __mul__(self, other) -> "Tile":
        ...

    @function
    def __truediv__(self, other) -> "Tile":
        ...

    @function
    def __floordiv__(self, other) -> "Tile":
        ...

    @function
    def __mod__(self, other) -> "Tile":
        ...

    @function
    def __pow__(self, other) -> "Tile":
        ...

    @function
    def __and__(self, other) -> "Tile":
        ...

    @function
    def __or__(self, other) -> "Tile":
        ...

    @function
    def __xor__(self, other) -> "Tile":
        ...

    @function
    def __radd__(self, other) -> "Tile":
        ...

    @function
    def __rsub__(self, other) -> "Tile":
        ...

    @function
    def __rmul__(self, other) -> "Tile":
        ...

    @function
    def __rtruediv__(self, other) -> "Tile":
        ...

    @function
    def __rfloordiv__(self, other) -> "Tile":
        ...

    @function
    def __rmod__(self, other) -> "Tile":
        ...

    @function
    def __rpow__(self, other) -> "Tile":
        ...

    @function
    def __rand__(self, other) -> "Tile":
        ...

    @function
    def __ror__(self, other) -> "Tile":
        ...

    @function
    def __rxor__(self, other) -> "Tile":
        ...

    @function
    def __ge__(self, other) -> "Tile":
        ...

    @function
    def __gt__(self, other) -> "Tile":
        ...

    @function
    def __le__(self, other) -> "Tile":
        ...

    @function
    def __lt__(self, other) -> "Tile":
        ...

    @function
    def __eq__(self, other) -> "Tile":
        ...

    @function
    def __ne__(self, other) -> "Tile":
        ...


Shape = Union[int, tuple[int, ...]]
Order = Union[tuple[int, ...], Literal['C'], Literal['F']]

TileOrScalar = Union[Tile, Scalar]


###############################################################################
# Constantness Hints


class ConstantAnnotation:
    """
    A ``typing.Annotated`` metadata class indicating that an object shall be |constant embedded|.

    If an object of this class is passed as a metadata argument to a ``typing.Annotated`` type hint
    on a parameter, then the parameter shall be a constant embedded.
    """

    def __repr__(self):
        return "ConstantAnnotation()"


T = TypeVar("T")
Constant = Annotated[T, ConstantAnnotation()]
Constant.__doc__ = """A type hint indicating that a value shall be |constant embedded|.
It can be used either with (``Constant[int]``) or without (``Constant``, meaning a constant of any
type) an underlying type hint.
"""


###############################################################################
# Operations


@function
def bid(axis) -> int:
    """Gets the index of current block.

    Args:
        axis (const int): The axis of the block index space. Possible values are 0, 1, 2.

    Returns:
        int32:

    Examples:

        >>> bid_x = ct.bid(0)
        >>> bid_y = ct.bid(1)
        >>> bid_z = ct.bid(2)
    """


@function
def num_blocks(axis) -> int:
    """Gets the number of blocks along the axis.

    Args:
        axis (const int): The axis of the block index space. Possible values are 0, 1, 2.

    Returns:
        int32:

    Examples:

        >>> num_blocks_x = ct.num_blocks(0)
        >>> num_blocks_y = ct.num_blocks(1)
        >>> num_blocks_z = ct.num_blocks(2)
    """


@function
def num_tiles(array: Array, /,
              axis: int,
              shape: Constant[Shape],
              order: Constant[Order] = "C") -> int:
    """Gets the number of tiles in the |tile space| of the array along the `axis`.

    Args:
        array (Array): An array object on a cuda device.
        axis (const int): The axis of the tile partition space to get the dim size.
        shape (const int...): A sequence of const integers definining the shape of the tile.
        order ("C" or "F", or tuple[const int,...]): Order of axis mapping. See :py:func:`load`.

    Returns:
        int32

    Examples:

        Suppose array size is (32, 16), tile shape (4, 8),
        the partition space will be (cdiv(32, 4), cdiv(16, 8)) == (8, 2)

        >>> ct.num_tiles(array, 0, shape=(4, 8))
        8
        >>> ct.num_tiles(array, 1, shape=(4, 8))
        2
    """


@function
def load(array: Array, /,
         index: Shape,
         shape: Constant[Shape], *,
         order: Constant[Order] = "C",
         padding_mode: PaddingMode = PaddingMode.UNDETERMINED,
         latency: Optional[int] = None,
         allow_tma: Optional[bool] = None) -> Tile:
    """Loads a tile from the `array` which is partitioned into a |tile space|.

    The |tile space| is the result of partitioning the `array` into a grid of equally
    sized tiles specified by `shape`.

    For example, partitoning a 2D `array` of shape ``(M, N)`` using tile shape
    ``(tm, tn)`` results in a 2D tile space of size ``(cdiv(M, tm), cdiv(N, tn))``.
    An index into this tile space using index ``(i, j)`` produces a tile of size ``(tm, tn)``:

        >>> t = ct.load(array, (i, j), (tm, tn))  # `t` has shape (tm, tn)

    The result tile `t` will be computed according to ::

        t[x, y] = array[i * tm + x, j * tn + y]  (for all 0<=x<tm, 0<=y<tn)

    For access that is out of bound, the value will be determined by `padding_mode`.

    `order` is used to map the tile axis to the array axis. The transposed example of the above call
    to `load` would be:

        >>> ct.load(array, (j, i), shape=(tn, tm), order=(1, 0))

    The result tile `t` will be computed according to ::

        t[y, x] = array[i * tm + x, j * tn + y]



    Args:
        array (Array): The |array| to load from.
        index (tuple[int,...]): An index in the |tile space| of ``shape`` from ``array``.
        shape (tuple[const int,...]): A tuple of const integers definining the shape of the tile.
        order ("C" or "F", or tuple[const int,...]): Permutation applied to array axes before the
            logical |tile space| is constructed. Can be specified either as a tuple of constants,
            or as one of the two special string literal values:

            * "C" is an alias for ``(0, 1, 2, ...)``, i.e. no permutation applied;
            * "F" is an alias for ``(..., 2, 1, 0)``, i.e. axis order is reversed.

        padding_mode (PaddingMode): The value used to pad the tile when it extends beyond the array
            boundaries. By default, the padding value is undetermined.
        latency (const int): A hint indicating how heavy DRAM traffic will be. It shall be an
            integer between 1 (low) and 10 (high). By default, the compiler will infer the latency.
        allow_tma (const bool): If False, the load will not use TMA. By default, TMA is allowed.

    Returns:
        Tile:

    Examples:

        >>> # Regular load.
        >>> tile = ct.load(array2d, (0, 0), shape=(2, 4))
        >>> # Load with a transpose.
        >>> tile = ct.load(array2d, (0, 0), shape=(4, 2), order='F')
        >>> # Load transposing the last two axes.
        >>> tile = ct.load(array3d, (0, 0, 0), shape=(8, 4, 2), order=(0, 2, 1))
        >>> # Load a single element as 0d tile
        >>> tile = ct.load(array3d, (0, 0, 0), shape=())

    .. seealso::
        - :py:func:`store`
        - :py:func:`gather`
        - |Tile space|
    """


@function
def store(array: Array, /,
          index: Shape,
          tile: TileOrScalar, *,
          order: Constant[Order] = "C",
          latency: Optional[int] = None,
          allow_tma: Optional[bool] = None) -> None:
    """Stores a `tile` value into the `array` at the `index` of its |tile space|.

    The |tile space| is the result of partitioning the `array` into a grid of tiles
    with equal size defined by the shape of the `tile`.

    For example, given a tile `t` of shape ``(tm, tn)`` and array of shape ``(M, N)``:

        >>> # tile `t` has shape (tm, tn)
        >>> ct.store(array, (i, j), t)

    The above call to `store` will store elements according to::

        array[i * tm + x, i * tn + y] = t[x, y]  (for 0<=x<tm, 0<=y<tn)

    Access which falls out of the boundary of the `array` will be ignored.

    Args:
        array (Array): The |array| to store to.
        index (tuple[int,...]): An index in the |tile space| of ``array``.
            ``shape`` is inferred from the ``tile`` argument.
        tile (Tile): The |tile| to store. The rank of the tile must match rank of the array,
            unless it is a scalar or 0d tile.
        order ("C" or "F", or tuple[const int,...]): Order of axis mapping. See :py:func:`load`.
        latency (int, optional): A hint indicating how heavy DRAM traffic will be. It shall be an
            integer between 1 (low) and 10 (high). By default, the compiler will infer the latency.
        allow_tma (bool, optional): If False, the load will not use TMA. By default, TMA is allowed.

    Examples:

        >>> tile = ct.load(array_in, bid_x, shape=4)
        >>> tile = tile * 2
        >>> ct.store(array_out, (bid_x,), tile=tile)
        # store a scalar
        >>> ct.store(array_out, (0,), tile=0)

    .. seealso::
        - :py:func:`load`
        - :py:func:`scatter`
        - |Tile space|
    """


@function
def gather(array, indices, /, *, padding_value=0, check_bounds=True, latency=None) -> Tile:
    """
    Loads a tile from the `array` elements specified by `indices`.

    `indices` must be a tuple whose length equals the `array` rank.
    All elements of this tuple must be integer tiles or scalars of the same shape,
    or different shapes that are broadcastable to a common shape.

    The result shape will be the same as the broadcasted shape of indices.

    For example, consider a 2-dimensional array. In this case, indices must be a tuple
    of length 2. Suppose that ``ind0`` and ``ind1`` are integer tiles
    of shapes ``(M, N, 1)`` and ``(M, 1, K)``.
    Then the result tile will have the broadcasted shape ``(M, N, K)``:

        >>> t = ct.gather(array, (ind0, ind1))   # `t` has shape (M, N, K)

    The result tile `t` will be computed according to ::

        t[i, j, k] = array[ind0[i, j, 0], ind1[i, 0, k]]   (for all 0<=i<M, 0<=j<N, 0<=k<K)

    If the array is 1-dimensional, `indices` can be passed as a tile rather than a tuple.
    This is a convenience notation that is strictly equivalent to passing a tuple of length 1:

        >>> ct.gather(array, ind0)   # equivalent to ct.gather(array, (ind0,))

    `gather()` checks that indices are within the bounds of the array. For indices
    that are out of bounds, `padding_value` will be returned (zero by default).
    It must be a scalar or a tile whose shape is broadcastable to the common shape of indices.

    To disable bounds checking, set `check_bounds` to ``False``.
    In this mode, the caller is responsible for ensuring that all indices are within the bounds
    of the array, and any out-of-bounds access will result in undefined behavior.

    Negative indices are interpreted as out of bounds, i.e. they don't follow the Python's
    negative index convention.
    """


@function
def scatter(array, indices, value, /, *, check_bounds=True, latency=None):
    """
    Stores a tile `value` into the `array` elements specified by `indices`.

    `indices` must be a tuple whose length equals the `array` rank.
    All elements of this tuple must be integer tiles or scalars of the same shape,
    or different shapes that are broadcastable to a common shape.

    `value` must be a scalar or a tile whose shape is broadcastable to the
    common shape of `indices`.

    For example, consider a 2-dimensional array. In this case, indices must be a tuple
    of length 2. Suppose that ``ind0`` and ``ind1`` are integer tiles
    of shapes ``(M, N, 1)`` and ``(M, 1, K)``, and ``value`` is a tile of shape of ``(N, K)``:

        >>> # ind0: (M, N, 1),  ind1: (M, 1, K),  value: (N, K)
        >>> ct.scatter(array, (ind0, ind1), value)

    The above call to `scatter` will store elements according to ::

        array[ind0[i, j, 0], ind1[i, 0, k]] = value[j, k]

    If the array is 1-dimensional, `indices` can be passed as a tile rather than a tuple.
    This is a convenience notation that is strictly equivalent to passing a tuple of length 1:

        >>> ct.scatter(array, ind0, value)   # equivalent to ct.scatter(array, (ind0,), value)

    `scatter()` checks that indices are within the bounds of the array. For indices
    that are out of bounds, nothing is stored. To disable bounds checking,
    set `check_bounds` to ``False``. In this mode, the caller is responsible for ensuring that
    all indices are within the bounds of the array, and any out-of-bounds access
    will result in undefined behavior.
    """


# =========== Atomic ============


@function
def atomic_cas(array, indices, expected, desired, /, *,
               check_bounds=True,
               memory_order=MemoryOrder.ACQ_REL,
               memory_scope=MemoryScope.DEVICE) -> Tile:
    """Bulk atomic compare-and-swap on array elements with given indices.

    For each specified index, `atomic_cas()` compares the corresponding array element
    to the `expected` value. If it matches, it is then overwritten with the `desired` value;
    otherwise, no update is performed. In either case, the old value of the element is returned.
    For each individual element, the described compare-and-swap operation is performed atomically,
    but the operation as a whole is not atomic, and the order of individual updates is unspecified.

    `atomic_cas()` follows the same convention as :py:func:`gather()` and :py:func:`scatter()`:
    `indices` must be a tuple whose length equals the `array` rank.
    All elements of this tuple must be integer tiles or scalars of the same shape,
    or different shapes that are broadcastable to a common shape.
    If the array is 1-dimensional, `indices` can be passed as a single tile
    rather than a tuple of length 1.

    `expected` and `desired` must be scalars or tiles whose shapes are broadcastable
    to the common shape of `indices`.

    By default, `atomic_cas()` checks that indices are within the bounds of the array.
    For indices that are out of bounds, no operation is performed, and a corresponding `expected`
    value is returned. To disable bounds checking, set `check_bounds` to ``False``.
    In this mode, the caller is responsible for ensuring that all indices are within
    the bounds of the array, and any out-of-bounds access will result in undefined behavior.

    As an example, consider a 2-dimensional array. In this case, indices must be a tuple
    of length 2. Suppose that ``ind0`` and ``ind1`` are integer tiles
    of shapes ``(M, N, 1)`` and ``(M, 1, K)``, and both ``expected`` and ``desrired``
    are tiles of shape of ``(N, K)``:

        >>> # ind0: (M, N, 1),  ind1: (M, 1, K),  expected: (N, K),  desired: (N, K)
        >>> ct.atomic_cas(array, (ind0, ind1), expected, desired)

    The above call to `atomic_cas` will behave similarly to the following pseudocode::

        in parallel, for all (i, j, k) such that 0<=i<M, 0<=j<N, i<=k<K:
            if not check_bounds or (0 <= ind0[i, j, 0] < array.shape[0]
                                    and 0 <= ind1[i, 0, k] < array.shape[1]):
                do atomically:
                    actual = array[ind0[i, j, 0], ind1[i, 0, k]]
                    if actual == expected[j, k]:
                        array[ind0[i, j, 0], ind1[i, 0, k]] = desired[j, k]
                result[i, j, k] = actual
            else:
                result[i, j, k] = expected[j, k]

    Examples:

        >>> indices = ct.arange(32, dtype=ct.int32)
        >>> expected = ct.full((32,), 1, dtype=ct.int32)
        >>> desired = ct.arange(32, dtype=ct.int32)
        >>> old_value = ct.atomic_cas(array, indices, expected, desired)
    """


def _doc_atomic_rmw_op(f):
    op_name = f.__name__
    f.__doc__ += f"""\

    For each individual element, the operation is performed atomically,
    but the operation as a whole is not atomic, and the order of individual writes is unspecified.

    `{op_name}()` follows the same convention as :py:func:`gather()` and :py:func:`scatter()`:
    `indices` must be a tuple whose length equals the `array` rank.
    All elements of this tuple must be integer tiles or scalars of the same shape,
    or different shapes that are broadcastable to a common shape.
    If the array is 1-dimensional, `indices` can be passed as a single tile
    rather than a tuple of length 1.

    `update` must be a scalar or a tile whose shape is broadcastable to the
    common shape of `indices`.

    By default, `{op_name}()` checks that indices are within the bounds of the array.
    For indices that are out of bounds, no operation is performed, and an implementation-defined
    value is returned. To disable bounds checking, set `check_bounds` to ``False``.
    In this mode, the caller is responsible for ensuring that all indices are within
    the bounds of the array, and any out-of-bounds access will result in undefined behavior.

    Examples:

        >>> indices = ct.arange(32, dtype=ct.int32)
        >>> update = ct.arange(32, dtype=ct.int32)
        >>> old_value = ct.{op_name}(array, indices, update)
    """

    return f


@function
@_doc_atomic_rmw_op
def atomic_xchg(array, indices, update, /, *,
                check_bounds=True,
                memory_order=MemoryOrder.ACQ_REL,
                memory_scope=MemoryScope.DEVICE) -> Tile:
    """Bulk atomic exchange of array elements at given indices.

    For each specified index, `atomic_xchg()` stores the corresponding `update`
    to the array element at that location, and returns the original value of the element
    before the update.
    """


@function
@_doc_atomic_rmw_op
def atomic_add(array, indices, update, /, *,
               check_bounds=True,
               memory_order=MemoryOrder.ACQ_REL,
               memory_scope=MemoryScope.DEVICE) -> Tile:
    """Bulk atomic post-increment of array elements at given indices.

    For each specified index, `atomic_add()` reads the corresponding array element,
    adds `update` to it, and writes the modified value back to the same location.
    The original value of the element before the update is returned.
    """


@function
@_doc_atomic_rmw_op
def atomic_max(array, indices, update, /, *,
               check_bounds=True,
               memory_order=MemoryOrder.ACQ_REL,
               memory_scope=MemoryScope.DEVICE) -> TileOrScalar:
    """Bulk atomic maximum value assignment on array elements at given indices.

    For each specified index, `atomic_max()` reads the corresponding array element,
    computes the maximum between its value and the corresponding value of `update`,
    and writes the modified value back to the same location.
    The original value of the element before the update is returned.
    """


@function
@_doc_atomic_rmw_op
def atomic_min(array, indices, update, /, *,
               check_bounds=True,
               memory_order=MemoryOrder.ACQ_REL,
               memory_scope=MemoryScope.DEVICE) -> TileOrScalar:
    """Bulk atomic minimum value assignment on array elements at given indices.

    For each specified index, `atomic_min()` reads the corresponding array element,
    computes the minimum between its value and the corresponding value of `update`,
    and writes the modified value back to the same location.
    The original value of the element before the update is returned.
    """


@function
@_doc_atomic_rmw_op
def atomic_and(array, indices, update, /, *,
               check_bounds=True,
               memory_order=MemoryOrder.ACQ_REL,
               memory_scope=MemoryScope.DEVICE) -> TileOrScalar:
    """Bulk atomic AND operation on array elements at given indices.

    For each specified index, `atomic_and()` reads the corresponding array element,
    computes the bitwise AND between its value and the corresponding value of `update`,
    and writes the modified value back to the same location.
    The original value of the element before the update is returned.
    """


@function
@_doc_atomic_rmw_op
def atomic_or(array, indices, update, /, *,
              check_bounds=True,
              memory_order=MemoryOrder.ACQ_REL,
              memory_scope=MemoryScope.DEVICE) -> Tile:
    """Bulk atomic OR operation on array elements at given indices.

    For each specified index, `atomic_or()` reads the corresponding array element,
    computes the bitwise OR between its value and the corresponding value of `update`,
    and writes the modified value back to the same location.
    The original value of the element before the update is returned.
    """


@function
@_doc_atomic_rmw_op
def atomic_xor(array, indices, update, /, *,
               check_bounds=True,
               memory_order=MemoryOrder.ACQ_REL,
               memory_scope=MemoryScope.DEVICE) -> Tile:
    """Bulk atomic XOR operation on array elements at given indices.

    For each specified index, `atomic_xor()` reads the corresponding array element,
    computes the bitwise XOR between its value and the corresponding value of `update`,
    and writes the modified value back to the same location.
    The original value of the element before the update is returned.
    """


# ======== Factory ==============


@function
def arange(size, /, *, dtype) -> Tile:
    """Creates a tile with value starting from 0 to `size - 1`.

    Args:
        size (const int): Size of the tile.
        dtype (DType): Datatype of the tile.

    Returns:
        Tile:

    Examples:

        >>> tile = ct.arange(16, dtype=ct.int32)
    """


@function
def full(shape: Shape, fill_value: Scalar, dtype: DType) -> Tile:
    """Creates a tile filled with given value.

    Args:
        shape (tuple[const int,...]):  The shape of the tile.
        fill_value (int | float | bool]): Value for the tile.
        dtype (DType): The |Data type| of the tile.

    Returns:
        Tile:

    Examples:

        >>> tile = ct.full((4, 4), 3.14, dtype=ct.float32)
    """


@function
def ones(shape, dtype) -> Tile:
    """Creates a tile filled with ones.

    Args:
        shape (tuple[const int,...]):  The shape of the tile.
        dtype (DType): The |Data type| of the tile.

    Returns:
        Tile:

    Examples:

        >>> tile = ct.ones((4, 4), dtype=ct.float32)
    """


@function
def zeros(shape, dtype) -> Tile:
    """Creates a tile filled with zeros.

    Args:
        shape (tuple[const int,...]):  The shape of the tile.
        dtype (DType): The |Data type| of the tile.

    Returns:
        Tile:

    Examples:

        >>> tile = ct.zeros((4, 4), dtype=ct.float32)
    """

# =========== Matmul ============


@function
def mma(x, y, /, acc) -> Tile:
    """Matrix multiply-accumulate.

    Computes ``(x @ y) + acc`` as a single operation
    (where ``@`` denotes matrix multiplication).
    Preserves the dtype of `acc`.

    Args:
        x (Tile): LHS of the mma, 2D or 3D.
        y (Tile): RHS of the mma, 2D or 3D.
        acc (Tile): Accumulator of mma.

    Supported datatypes:

    +----------+---------------+
    | Input    |  Acc/Ouput    |
    +==========+===============+
    | f16      |  f16 or f32   |
    +----------+---------------+
    | bf16     |  f32          |
    +----------+---------------+
    | f32      |  f32          |
    +----------+---------------+
    | f64      |  f64          |
    +----------+---------------+
    | tf32     |  f32          |
    +----------+---------------+
    | f8e4m3fn |  f16 or f32   |
    +----------+---------------+
    | f8e5m2   |  f16 or f32   |
    +----------+---------------+
    | [u|i]8   |  i32          |
    +----------+---------------+

    If `x` and `y` have different dtype, they will NOT be promoted to common dtype.
    Shape of `x` and `y` will be broadcasted to up until the last two axes.

    Returns:
        Tile:

    Example:

        >>> tx = ct.full((2, 4), 3, dtype=ct.float32)
        >>> ty = ct.full((4, 8), 4, dtype=ct.float32)
        >>> acc = ct.full((2, 8), 0, dtype=ct.float32)
        # default
        >>> tz = ct.mma(tx, ty, acc)
    """


@function
def matmul(x, y, /) -> Tile:
    """Performs matrix multiply on the given tiles.

    Args:
        x (Tile): LHS of the matmul, 1D, 2D, or 3D.
        y (Tile): RHS of the matmul, 1D, 2D, or 3D.

    Supported input datatypes: [f16, bf16, f32, f64, tf32, f8e4m3fn, f8e5m2, i8, u8]

    If `x` and `y` have different dtype, they will first be promoted to common
    dtype. The result dtype is the same as the promoted input types.
    Shape of `x` and `y` will be broadcasted to up until the last two axes.

    Returns:
        Tile:

    Example:

        >>> tx = ct.full((2, 4), 3, dtype=ct.float32)
        >>> ty = ct.full((4, 8), 4, dtype=ct.float32)
        # default
        >>> tz = ct.matmul(tx, ty)
        # use builtin `@`
        >>> tz = tx @ ty
    """


# ======== Shape and Dtype ==============
@function
def expand_dims(x, /, axis) -> Tile:
    """Reshapes the tile by inserting a new axis of size 1 at given position.

    This can also be done via the NumPy-style syntax: `x[:, None]` or `x[np.newaxis, :]`

    Args:
        x (Tile): input tile.
        axis (const int): axis to expand the tile dimension.

    Returns:
        Tile:

    Examples:

        >>> tx = ct.arange(16, dtype=ct.float32)
        >>> tx.shape
        (16,)
        >>> ty = ct.expand_dims(x, 1)
        >>> ty.shape
        (16,1)
        >>> ty = x[None, ..., None, None]
        >>> ty.shape
        (1, 16, 1, 1)
    """


@function
def cat(tiles, /, axis) -> Tile:
    """Concatenates two tiles along the `axis`.

    Args:
        tiles (tuple): a pair of tiles to concatenate.
        axis (const int): axis to concatenate the tiles.

    Returns:
        Tile:

    Notes:
        Due to power-of-two assumption on all tile shapes,
        the two input tiles must have the same shape.

    Examples:

        >>> tx = ct.full((2, 4), 3., dtype=ct.float32)
        >>> ty = ct.full((2, 4), 4., dtype=ct.float32)
        >>> tz = ct.cat((tx, ty), 0)
        >>> tz.shape
        (4,4)
        >>> tz = ct.cat((tx, ty), 1)
        >>> tz.shape
        (2,8)
    """


@function
def broadcast_to(x, /, shape) -> Tile:
    """Broadcasts a tile to the specified shape
    following |Numpy broadcasting rule|.

    Args:
        x (Tile): input tile.
        shape (tuple[const int,...]): target shape.

    Returns:
        Tile:

    Examples:

        >>> tx = ct.arange(4, dtype=ct.float32)
        >>> tx.shape
        (4,)
        >>> ty = ct.broadcast_to(tx, (2, 4))
        >>> ty.shape
        (2, 4)
    """


@function
def reshape(x, /, shape) -> Tile:
    """Reshapes a tile to the specified shape.

    One of the shape elements may be specified as -1 to indicate that the
    corresponding dimension is to be inferred automatically.

    For example, reshaping a ``(16, 2)`` tile to ``(8, -1)`` will
    produce a tile of shape ``(8, 4)``: as there are 32 elements in total,
    the second dimension will be computed as 32 divided by 8.

    Args:
        x (Tile): input tile.
        shape (tuple[const int,...]): target shape.

    Returns:
        Tile:

    Examples:

        >>> tx = ct.arange(8, dtype=ct.float32)
        >>> tx.shape
        (8,)
        >>> ty = ct.reshape(tx, (2, 4))
        >>> ty.shape
        (2, 4)
        >>> tz = ct.reshape(tx, (2, -1))
        >>> tz.shape
        (2, 4)
    """


@function
def permute(x, /, axes) -> Tile:
    """Permutes the axes of the input tile.

    Args:
        x (Tile): input tile.
        axes (tuple[const int,...]): the desired axes order.

    Returns:
        Tile:

    Examples:

        >>> tx = ct.full((2, 4, 8), 0., dtype=ct.float32)
        >>> ty = ct.permute(tx, (0, 2, 1))
        >>> ty.shape
        (2, 8, 4)
    """


@function
def transpose(x, /, axis0=None, axis1=None) -> Tile:
    """Transposes two axes of the input tile with at least 2 dimensions.

    For a 2-dimensional tile, the two axes are transposed if `axis0` and `axis1` are not specified.
    For tiles with more than 2 dimensions, `axis0` and `axis1` must be explicitly specified.

    Args:
        x (Tile): input tile.
        axis0 (const int): the first axis to transpose.
        axis1 (const int): the second axis to transpose.

    Returns:
        Tile:

    Examples:

        >>> tx = ct.full((2, 4, 8), 0., dtype=ct.float32)
        >>> ty = ct.transpose(tx, axis0=0, axis1=1)
        >>> ty.shape
        (4, 2, 8)
        >>> tx = ct.full((2, 4), 0., dtype=ct.float32)
        >>> ty = ct.transpose(tx)
        >>> ty.shape
        (4, 2)
    """


@function
def astype(x, dtype, /) -> Tile:
    """Converts a tile to the specified data type.

    Args:
        x (Tile): input tile.
        dtype (DType): target data type.

    Returns:
        Tile:

    Examples:

        >>> tx = ct.arange(8, dtype=ct.float32)
        >>> ty = ct.astype(tx, ct.float16)
        >>> ty.dtype
        float16
    """


@function
def bitcast(x, /, dtype) -> Tile:
    """Reinterpets tile as being of specified data type.

    Args:
        x (Tile): input tile.
        dtype (DType): target data type.

    Returns:
        Tile:

    Examples:

        >>> tx = ct.arange(8, dtype=ct.float32)
        >>> ty = ct.bitcast(tx, ct.int32)
        >>> ty.dtype
        int32
    """


def _math_op_extra_block(f, indent):
    base = inspect.unwrap(f)
    sig = inspect.signature(base)
    extra = []
    for name in sig.parameters:
        if name == "rounding_mode":
            extra.append(
                f"{name} (RoundingMode): The rounding mode for the operation, only supported "
                "for float types, default is RoundingMode.RN when applicable."
            )
        elif name == "flush_to_zero":
            extra.append(
                f"{name} (const bool): If True, flushes subnormal inputs and results to "
                "sign-preserving zero, default is False."
            )
    return ("\n" + textwrap.indent("\n".join(extra), indent)) if extra else ""


# ======== Reduction ==============
def _doc_reduce_op(f):

    @functools.wraps(f)
    def wrapped(*args, **kwargs):
        return f(*args, **kwargs)

    op_name = f.__name__
    extra_block = _math_op_extra_block(f, indent="        ")

    wrapped.__doc__ = f"""Performs {op_name} reduction on tile along the `axis`.

    Args:
        x (Tile): input tile.
        axis (None | const int | tuple[const int,...]): the axis for reduction.
            The default, `axis=None`, will reduce all of the elements.
        keep_dims (const bool): If true, preserves the number of dimension
            from the input tile.{extra_block}

    Returns:
        Tile:

    Examples:

        >>> tx = ct.full((2, 4), 3, dtype=ct.float32)
        >>> ty = ct.{op_name}(tx, 1)
        >>> ty.shape
        (2,)
        >>> ty = ct.{op_name}(tx, 1, keepdims=True)
        >>> ty.shape
        (2, 1)
    """

    return wrapped


@_doc_reduce_op
@function
def sum(x, /, axis=None, *, keepdims=False, rounding_mode: Optional[RoundingMode] = None,
        flush_to_zero: bool = False) -> Tile:
    pass


@_doc_reduce_op
@function
def max(x, /, axis=None, *, keepdims=False, flush_to_zero: bool = False) -> Tile:
    pass


@_doc_reduce_op
@function
def min(x, /, axis=None, *, keepdims=False, flush_to_zero: bool = False) -> Tile:
    pass


@_doc_reduce_op
@function
def prod(x, /, axis=None, *, keepdims=False, rounding_mode: Optional[RoundingMode] = None,
         flush_to_zero: bool = False) -> Tile:
    pass


@_doc_reduce_op
@function
def argmax(x, /, axis=None, *, keepdims=False) -> Tile:
    pass


@_doc_reduce_op
@function
def argmin(x, /, axis=None, *, keepdims=False) -> Tile:
    pass


@function
def reduce(x, /, axis, func, identity, *, keepdims=False):
    """
    Apply custom reduction function along axis.

    Args:
        x: input tile or a tuple of tiles to be reduced. If a tuple is provided, shapes
            of the tiles in the tuple must be broadcastable to a common shape.
        axis (int): an integer constant that specifies the axis to reduce along.
        func: function for combining two values. If `x` is a single tile, then the function
            must take two 0d tile arguments and return the combined 0d tile. For example,
            `lambda a, b: a + b` or `operator.add` can be used to implement the sum reduction.
            If `x` is a tuple of N tiles, then the function takes 2N tiles and returns a tuple
            of N combined tiles. The first N arguments correspond to one of the groups of values
            being combined, while the rest correspond to the other.
        identity: a constant scalar or a tuple of constant scalars that specifies the identity
            element of the `func`.
        keepdims (bool): True to remove the reduced axis, False to keep the axis of size 1.
            Default: False.

    Returns:
        Reduced tile, or tuple of reduced tiles, depending on the type of `x`.
    """


# ======== Scan ==============
def _doc_scan_op(f):

    @functools.wraps(f)
    def wrapped(*args, **kwargs):
        return f(*args, **kwargs)

    op_name = f.__name__
    extra_block = _math_op_extra_block(f, indent="        ")

    wrapped.__doc__ = f"""Performs {op_name} on tile along the `axis`.

    Args:
        x (Tile): input tile
        axis (const int): the axis for scan, default 0.
        reverse (const bool): if True, the scan is performed in the reverse direction.{extra_block}

    Returns:
        Tile:

    Examples:

        >>> tx = ct.full((2, 4), 3, dtype=ct.float32)
        >>> ty = ct.{op_name}(tx, 1)
        >>> ty.shape
        (2, 4)
        >>> ty = ct.{op_name}(tx, 1, reverse=True)
        >>> ty.shape
        (2, 4)
    """

    return wrapped


@_doc_scan_op
@function
def cumsum(x, /, axis=0, *, reverse=False, rounding_mode: Optional[RoundingMode] = None,
           flush_to_zero: bool = False) -> Tile:
    pass


@_doc_scan_op
@function
def cumprod(x, /, axis=0, *, reverse=False, rounding_mode: Optional[RoundingMode] = None,
            flush_to_zero: bool = False) -> Tile:
    pass


# ======== Math binary ==============
def _doc_binary_op(builtin_op):
    def decorator(f):
        @functools.wraps(f)
        def wrapped(*args, **kwargs):
            return f(*args, **kwargs)

        op_name = f.__name__
        extra_block = _math_op_extra_block(f, indent="            ")

        if builtin_op in ("min", "max"):
            builtin_example = f"{builtin_op}({{}}, {{}})"
        else:
            builtin_example = f"{{}} {builtin_op} {{}}"

        wrapped.__doc__ = f"""Elementwise {op_name} on two tiles.

        Can also use builtin operation `{builtin_example.format('x', 'y')}`.

        Args:
            x (Tile): LHS tile.
            y (Tile): RHS tile.{extra_block}

        The `shape` of `x` and `y` will be broadcasted and
        `dtype` promoted to common dtype.

        Returns:
            Tile:

        Examples:

            >>> # tile and tile
            >>> tx = ct.full((2, 4), 7, dtype=ct.int32)
            >>> ty = ct.full((2, 4), 3, dtype=ct.int32)
            >>> tz = ct.{op_name}(tx, ty)

            >>> # Can also use the builtin op
            >>> tz = {builtin_example.format('tx', 'ty')}

            >>> # shape broadcast
            >>> tx = ct.full((2, 4), 7, dtype=ct.int32)
            >>> ty = ct.full((2,), 3, dtype=ct.int32)
            >>> tz = {builtin_example.format('tx', 'ty')}

            >>> # dtype cast
            >>> tx = ct.full((2, 4), 7, dtype=ct.int32)
            >>> ty = ct.full((2, 4), 3, dtype=ct.int64)
            >>> tz = {builtin_example.format('tx', 'ty')}

            >>> # tile and scalar
            >>> tx = ct.full((2, 4), 7, dtype=ct.int32)
            >>> y = 2
            >>> tz = {builtin_example.format('tx', 'y')}

            >>> # scalar and scalar
            >>> z = {builtin_example.format(7, 2)}
        """
        return wrapped
    return decorator


@_doc_binary_op('+')
@function
def add(x, y, /, *, rounding_mode: Optional[RoundingMode] = None,
        flush_to_zero: bool = False) -> TileOrScalar:
    pass


@_doc_binary_op('-')
@function
def sub(x, y, /, *, rounding_mode: Optional[RoundingMode] = None,
        flush_to_zero: bool = False) -> TileOrScalar:
    pass


@_doc_binary_op('*')
@function
def mul(x, y, /, *, rounding_mode: Optional[RoundingMode] = None,
        flush_to_zero: bool = False) -> TileOrScalar:
    pass


@_doc_binary_op('/')
@function
def truediv(x, y, /, *, rounding_mode: Optional[RoundingMode] = None,
            flush_to_zero: bool = False) -> TileOrScalar:
    pass


@_doc_binary_op('//')
@function
def floordiv(x, y, /) -> TileOrScalar:
    pass


@_doc_binary_op('**')
@function
def pow(x, y, /) -> TileOrScalar:
    pass


@_doc_binary_op('%')
@function
def mod(x, y, /) -> TileOrScalar:
    pass


@_doc_binary_op('&')
@function
def bitwise_and(x, y, /) -> TileOrScalar:
    pass


@_doc_binary_op('|')
@function
def bitwise_or(x, y, /) -> TileOrScalar:
    pass


@_doc_binary_op('^')
@function
def bitwise_xor(x, y, /) -> TileOrScalar:
    pass


@_doc_binary_op('<<')
@function
def bitwise_lshift(x, y, /) -> TileOrScalar:
    pass


@_doc_binary_op('>>')
@function
def bitwise_rshift(x, y, /) -> TileOrScalar:
    pass


@function
def bitwise_not(x, /) -> TileOrScalar:
    """Elementwise bitwise not on a tile.

    Can also use builtin operator `~x`.

    Args:
        x (Tile): input tile.

    Returns:
        Tile:

    Examples:

        >>> tx = ct.full((4, 4), 0, dtype=ct.int32)
        >>> ty = ct.bitwise_not(x)
        >>> ty = ~tx
    """

# TODO:  Do we support logical and, or, not?


@_doc_binary_op('min')
@function
def minimum(x, y, /, *, flush_to_zero: bool = False) -> TileOrScalar:
    pass


@_doc_binary_op('max')
@function
def maximum(x, y, /, *, flush_to_zero: bool = False) -> TileOrScalar:
    pass


@function(host=True)
def cdiv(x, y, /) -> TileOrScalar:
    """Computes ceil(x / y). Can be used on the host.

    Args:
        x (Tile): int tile.
        y (Tile): int tile.

    Returns:
        Tile:

    Examples:

        >>> tile = ct.full((2, 2), 7, dtype=ct.int32)
        >>> ct.cdiv(tile, 4)
        Tile((2,2), dtype=int32)

        >>> ct.cdiv(7, 4)
        2
    """
    return (x - 1) // y + 1


# ======== Comparison ==============

def _doc_cmp_op(builtin_op):
    def decorator(f):
        @functools.wraps(f)
        def wrapped(*args, **kwargs):
            return f(*args, **kwargs)

        op_name = f.__name__

        wrapped.__doc__ = f"""Compare two tiles elementwise with `{builtin_op}`.

        Can also use builtin operation `x {builtin_op} y`.

        Args:
            x (Tile): LHS tile.
            y (Tile): RHS tile.

        The `shape` of `x` and `y` will be broadcasted and
        `dtype` promoted to common dtype.

        Returns:
            Tile:

        Examples:

            >>> # tile and tile
            >>> tx = ct.arange(8, dtype=ct.int32) - 4
            >>> ty = ct.arange(8, dtype=ct.int32)
            >>> tz = ct.{op_name}(tx, ty)

            >>> # Can also use the builtin op
            >>> tz = tx {builtin_op} ty

            >>> # shape broadcast
            >>> tx = ct.arange(8, dtype=ct.int32)
            >>> ty = ct.full((1,), 0, dtype=ct.int32)
            >>> tz = tx {builtin_op} ty

            >>> # dtype broadcast
            >>> tx = ct.arange(8, dtype=ct.int32) - 4
            >>> ty = ct.arange(8, dtype=ct.int64)
            >>> tz = tx {builtin_op} ty

            >>> # tile and scalar
            >>> tx = ct.arange(8, dtype=ct.int32) - 4
            >>> tz = tx {builtin_op} 0

            >>> # scalar and scala
            >>> z = 5 {builtin_op} 3
        """
        return wrapped
    return decorator


@_doc_cmp_op('>')
@function
def greater(x, y, /) -> TileOrScalar:
    pass


@_doc_cmp_op('>=')
@function
def greater_equal(x, y, /) -> TileOrScalar:
    pass


@_doc_cmp_op('<')
@function
def less(x, y, /) -> TileOrScalar:
    pass


@_doc_cmp_op('<=')
@function
def less_equal(x, y, /) -> TileOrScalar:
    pass


@_doc_cmp_op('==')
@function
def equal(x, y, /) -> TileOrScalar:
    pass


@_doc_cmp_op('!=')
@function
def not_equal(x, y, /) -> TileOrScalar:
    pass


# ======== Math unary ==============
def _doc_unary_op(f):

    @functools.wraps(f)
    def wrapped(*args, **kwargs):
        return f(*args, **kwargs)

    op_name = f.__name__
    extra_block = _math_op_extra_block(f, indent="        ")

    wrapped.__doc__ = f"""
    Perform `{op_name}` on a tile.

    Args:
        x (Tile):{extra_block}

    Returns:
        Tile:

    Examples:

        >>> tx = ct.full((32, 32), 3.0, dtype=ct.float32)
        >>> tx = ct.{op_name}(tx)
    """
    return wrapped


@_doc_unary_op
@function
def exp(x, /) -> TileOrScalar:
    pass


@_doc_unary_op
@function
def exp2(x, /, *, flush_to_zero: bool = False) -> TileOrScalar:
    pass


@_doc_unary_op
@function
def log(x, /) -> TileOrScalar:
    pass


@_doc_unary_op
@function
def log2(x, /) -> TileOrScalar:
    pass


@_doc_unary_op
@function
def sqrt(x, /, *, rounding_mode: Optional[RoundingMode] = None,
         flush_to_zero: bool = False) -> TileOrScalar:
    pass


@_doc_unary_op
@function
def rsqrt(x, /, *, flush_to_zero: bool = False) -> TileOrScalar:
    pass


@_doc_unary_op
@function
def sin(x, /) -> TileOrScalar:
    pass


@_doc_unary_op
@function
def cos(x, /) -> TileOrScalar:
    pass


@_doc_unary_op
@function
def tan(x, /) -> TileOrScalar:
    pass


@_doc_unary_op
@function
def sinh(x, /) -> TileOrScalar:
    pass


@_doc_unary_op
@function
def cosh(x, /) -> TileOrScalar:
    pass


@_doc_unary_op
@function
def tanh(x, /) -> TileOrScalar:
    pass


@_doc_unary_op
@function
def floor(x, /) -> TileOrScalar:
    pass


@_doc_unary_op
@function
def ceil(x, /) -> TileOrScalar:
    pass


@function
def negative(x, /) -> TileOrScalar:
    """Same as `-x`.

    Args:
        x (Tile): input tile.

    Returns:
        Tile:

    Examples:

        >>> Negate a tile
        >>> tx = ct.arange(8, dtype=ct.int32)
        >>> ty = ct.negative(tx)
        >>> ty = -tx

        >>> Negate a scalar
        >>> x = 3
        >>> y = -x
    """


# ======== Select ==============

@function
def where(cond, x, y, /) -> Tile:
    """Returns elements chosen from x or y depending on condition.

    Args:
        cond (Tile): Boolean tile of shape `S`.
        x (Tile): Tile of shape `S` and dtype `T`, selected if `cond` is True.
        y (Tile): Tile of shape `S` and dtype `T`, selected if `cond` is False.

    Returns:
        Tile:

    Examples:

        >>> cond = ct.arange(4, dtype=ct.int32)
        >>> cond = cond > 2
        >>> x_true = ct.full((4,), 1.0, dtype=ct.float32)
        >>> x_false = ct.full((4,), -1.0, dtype=ct.float32)
        >>> y = ct.where(cond, x_true, x_false)
        >>> y
        [1., 1., -1., -1.]
        >>> z = ct.where(cond, 1.0, -1.0)
        >>> z
        [1., 1., -1., -1.]
    """


@function
def extract(x, /, index, shape) -> Tile:
    """Extracts a smaller tile from input tile.

    Partition the input tile into a grid with subtile shape
    and return a tile given the index into the grid. Similar
    to :py:func:`load` but performed on a tile.

    Args:
        x (Tile): input tile.
        index (Shape): An index in the sub |tile space|.
        shape (Shape): The shape of the extracted tile.

    Returns:
        Tile:

    Examples:

        >>> tile = ct.full((8, 8), 3.14, dtype=ct.float32)
        >>> sub_tile = ct.extract(x, (0, 0), shape=(4, 4))
        >>> sub_tile.shape
        (4, 4)
    """


# ============ Utility =================

@function
def printf(format, *args) -> None:
    """Print the values at runtime from the device

    Args:
        format (str): a c-printf style format string
            in the form of ``%[flags][width][.precision][length]specifier``,
            where specifier is limited to integer and float for now, i.e.
            ``[diuoxXeEfFgGaA]``

        *args (tuple[Tile, ...]):
            Only tile input is supported.

    Examples:

        >>> tile = ct.arange(4, dtype=ct.int32)
        >>> ct.printf("one tile: %d", tile)
        >>> ct.printf("two tiles: %d, %f", tile, tile * 2.0)

    Notes:
        This operation has significant overhead, and should only be used
        for debugging purpose.

        When printing from multiple tile blocks, outputs will be interleaved.
        One workaround is to set optimization level to 0:

        .. code-block:: python

            @ct.kernel(opt_level=0)
            def my_print_kernel():
                ct.printf("%d", 123)
    """


@function
def assert_(cond, /, message=None) -> None:
    """Assert that all elements of the given tile are True.

    Args:
        cond (Tile): Boolean tile.
        message (str): Message to print if condition is false.

    Notes:
        This operation has significant overhead, and should only be used
        for debugging purpose.


    Examples:

        >>> tile = ct.arange(4, dtype=ct.int32)
        >>> ct.assert_(tile > 2)
        >>> ct.assert_(tile > 2, "Not all elements in tile are greater than 2")
    """


# ==== Private stubs ====


def _m_array_slice(array, axis, start, stop): ...  # Array.slice(axis, start, stop)


def _m_tile_item(tile): ...  # Tile.item()
