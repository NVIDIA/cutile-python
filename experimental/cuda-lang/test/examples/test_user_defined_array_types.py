# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

__doc__ = """
These tests demonstrate the sorts of abstractions that users are able to build
at the library level. They create data classes that handle custom shapes and
strides, negative indexing like native Python arrays, and bounds checking.
Users could swap these arrays into their code for better safety, and then go
back to the builtin, low-overhead, less safe array types for the final kernel
or when running the kernel for performance measurements.

The bounds-checked array lets the user pass their own crash callback similar
to the way users can define their own address sanitizer callbacks when
compiling C or C++ code with address sanitization enabled.
"""

from dataclasses import dataclass
from enum import Enum
import subprocess
import sys
from typing import Callable

import cuda.lang as cl
from cuda.tile import TileStaticAssertionError
import pytest
import torch

if __name__ != "__main__":
    from test.util import compile_kernel


@cl.static_def
def contiguous_strides(shape):
    stride = 1
    strides = []
    for extent in reversed(shape):
        strides.append(stride)
        stride *= extent
    return tuple(reversed(strides))


class AccessKind(Enum):
    load = "load"
    store = "store"


def assert_invalid_access(access_kind, axis, index, extent):
    access_name = cl.static_eval(access_kind.value)
    cl.assert_(index >= 0, f"{access_name} index is less than zero")
    cl.assert_(
        index < extent,
        f"{access_name} index is not less than the axis extent",
    )


def custom_invalid_access(access_kind, axis, index, extent):
    print("custom invalid access", access_kind, axis, index, extent)
    message = cl.static_eval(f"custom invalid {access_kind.value}")
    cl.assert_(False, message)


def returning_invalid_access(access_kind, axis, index, extent):
    print("returning invalid access", access_kind, axis, index, extent)


def handle_invalid_access(
    invalid_access_callback,
    access_kind,
    axis,
    index,
    extent,
):
    invalid_access_callback(access_kind, axis, index, extent)
    assert_invalid_access(access_kind, axis, index, extent)


@cl.static_def
def initialize_array(
    array,
    base_pointer,
    shape,
    strides,
    invalid_access_callback,
    allow_negative_strides,
):
    if isinstance(shape, int):
        shape = (shape,)
    else:
        shape = tuple(shape)

    if strides is None:
        strides = contiguous_strides(shape)
    elif isinstance(strides, int):
        strides = (strides,)
    else:
        strides = tuple(strides)

    if len(shape) != len(strides):
        raise ValueError("shape and strides must have the same length")
    if any(extent < 0 for extent in shape):
        raise ValueError("shape extents must be nonnegative")
    if not allow_negative_strides and any(stride < 0 for stride in strides):
        raise ValueError("strides must be nonnegative")

    object.__setattr__(array, "base_pointer", base_pointer)
    object.__setattr__(array, "shape", shape)
    object.__setattr__(array, "strides", strides)
    object.__setattr__(
        array,
        "invalid_access_callback",
        invalid_access_callback,
    )


def _checked_offset(shape, strides, key, invalid_access_callback, access_kind):
    offset = cl.int64(0)
    for axis in cl.static_iter(range(len(key))):
        index = cl.int64(key[axis])
        if index < 0:
            handle_invalid_access(
                invalid_access_callback,
                access_kind,
                axis,
                index,
                shape[axis],
            )
        if index >= shape[axis]:
            handle_invalid_access(
                invalid_access_callback,
                access_kind,
                axis,
                index,
                shape[axis],
            )
        offset += index * strides[axis]
    return offset


def _checked_python_offset(
    shape,
    strides,
    key,
    invalid_access_callback,
    access_kind,
):
    offset = cl.int64(0)
    for axis in cl.static_iter(range(len(key))):
        index = cl.int64(key[axis])
        original_index = index
        if index < 0:
            index += shape[axis]
        if index < 0:
            handle_invalid_access(
                invalid_access_callback,
                access_kind,
                axis,
                original_index,
                shape[axis],
            )
        if index >= shape[axis]:
            handle_invalid_access(
                invalid_access_callback,
                access_kind,
                axis,
                original_index,
                shape[axis],
            )
        offset += index * strides[axis]
    return offset


@cl.static_def
def _python_slice_plan(key, shape, strides):
    has_slices = False
    slice_axes = []
    slice_offsets = []
    sliced_shape = []
    sliced_strides = []

    for axis, item in enumerate(key):
        if isinstance(item, slice):
            has_slices = True
            start, stop, step = item.indices(shape[axis])
            extent = len(range(start, stop, step))

            slice_axes.append(True)
            slice_offsets.append(start if extent > 0 else 0)
            sliced_shape.append(extent)
            sliced_strides.append(strides[axis] * step)
        else:
            slice_axes.append(False)
            slice_offsets.append(0)

    return (
        has_slices,
        tuple(slice_axes),
        tuple(slice_offsets),
        tuple(sliced_shape),
        tuple(sliced_strides),
    )


@cl.static_def
def _key_has_slice(key):
    return any(isinstance(item, slice) for item in key)


@cl.static_def
def _key_has_zero_slice_step(key):
    return any(isinstance(item, slice) and item.step == 0 for item in key)


def _checked_python_sliced_offset(
    shape,
    strides,
    key,
    slice_axes,
    slice_offsets,
    invalid_access_callback,
    access_kind,
):
    offset = cl.int64(0)
    for axis in cl.static_iter(range(len(key))):
        if slice_axes[axis]:
            offset += slice_offsets[axis] * strides[axis]
        else:
            index = cl.int64(key[axis])
            original_index = index
            if index < 0:
                index += shape[axis]
            if index < 0:
                handle_invalid_access(
                    invalid_access_callback,
                    access_kind,
                    axis,
                    original_index,
                    shape[axis],
                )
            if index >= shape[axis]:
                handle_invalid_access(
                    invalid_access_callback,
                    access_kind,
                    axis,
                    original_index,
                    shape[axis],
                )
            offset += index * strides[axis]
    return offset


@dataclass(frozen=True)
class BoundsCheckedArray:
    """Array that reports an index outside its shape.

    ``invalid_access_callback`` is called as
    ``callback(access_kind, axis, index, extent)``. If it returns, a default
    assertion stops the kernel.
    """

    base_pointer: cl.Pointer
    shape: tuple[int, ...]
    strides: tuple[int, ...]
    invalid_access_callback: Callable[[AccessKind, int, int, int], None]

    @cl.static_def
    def __init__(
        self,
        base_pointer,
        shape,
        strides=None,
        invalid_access_callback=assert_invalid_access,
    ):
        initialize_array(
            self,
            base_pointer,
            shape,
            strides,
            invalid_access_callback,
            False,
        )

    def __getitem__(self, key):
        cl.static_assert(isinstance(key, tuple), "indices must be a tuple")
        cl.static_assert(len(key) == len(self.shape), "index count must match its rank")
        offset = _checked_offset(
            self.shape,
            self.strides,
            key,
            self.invalid_access_callback,
            AccessKind.load,
        )
        return (self.base_pointer + offset).load()

    def __setitem__(self, key, value):
        cl.static_assert(isinstance(key, tuple), "indices must be a tuple")
        cl.static_assert(len(key) == len(self.shape), "index count must match its rank")
        offset = _checked_offset(
            self.shape,
            self.strides,
            key,
            self.invalid_access_callback,
            AccessKind.store,
        )
        (self.base_pointer + offset).store(value)


@dataclass(frozen=True)
class NegativeIndexedBoundsCheckedArray:
    """Bounds-checked array that supports Python indices and slices.

    Slice bounds and steps must be compile-time constants. An integer removes
    an axis, and a slice keeps an axis in the returned view.

    ``invalid_access_callback`` is called as
    ``callback(access_kind, axis, index, extent)``. If it returns, a default
    assertion stops the kernel.
    """

    base_pointer: cl.Pointer
    shape: tuple[int, ...]
    strides: tuple[int, ...]
    invalid_access_callback: Callable[[AccessKind, int, int, int], None]

    @cl.static_def
    def __init__(
        self,
        base_pointer,
        shape,
        strides=None,
        invalid_access_callback=assert_invalid_access,
    ):
        initialize_array(
            self,
            base_pointer,
            shape,
            strides,
            invalid_access_callback,
            True,
        )

    def __getitem__(self, key):
        cl.static_assert(isinstance(key, tuple), "indices must be a tuple")
        cl.static_assert(len(key) == len(self.shape), "index count must match its rank")
        cl.static_assert(
            not _key_has_zero_slice_step(key),
            "slice step must not be zero",
        )
        (
            has_slices,
            slice_axes,
            slice_offsets,
            sliced_shape,
            sliced_strides,
        ) = _python_slice_plan(key, self.shape, self.strides)
        offset = _checked_python_sliced_offset(
            self.shape,
            self.strides,
            key,
            slice_axes,
            slice_offsets,
            self.invalid_access_callback,
            AccessKind.load,
        )
        if has_slices:
            return NegativeIndexedBoundsCheckedArray(
                self.base_pointer + offset,
                sliced_shape,
                sliced_strides,
                self.invalid_access_callback,
            )
        return (self.base_pointer + offset).load()

    def __setitem__(self, key, value):
        cl.static_assert(isinstance(key, tuple), "indices must be a tuple")
        cl.static_assert(len(key) == len(self.shape), "index count must match its rank")
        cl.static_assert(
            not _key_has_slice(key),
            "slice assignment is not supported",
        )
        offset = _checked_python_offset(
            self.shape,
            self.strides,
            key,
            self.invalid_access_callback,
            AccessKind.store,
        )
        (self.base_pointer + offset).store(value)


def test_boundschecked_array():
    @cl.kernel
    def kernel(tensor, output):
        array = BoundsCheckedArray(tensor.get_base_pointer(), (2, 3, 4))

        output[0] = array[1, 2, 3]
        output[1] = array[1, 0, 1]
        array[0, 1, 2] = 101
        array[1, 1, 0] = 102

    tensor = torch.arange(24, device="cuda")
    output = torch.zeros(2, dtype=torch.int64, device="cuda")
    cl.launch(torch.cuda.current_stream(), (1,), (1,), kernel, (tensor, output))

    assert output.cpu().tolist() == [23, 13]
    assert tensor.cpu().tolist() == [
        0,
        1,
        2,
        3,
        4,
        5,
        101,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        102,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
    ]


def test_python_boundschecked_array():
    @cl.kernel
    def kernel(tensor, output):
        array = NegativeIndexedBoundsCheckedArray(
            tensor.get_base_pointer(),
            (2, 3, 4),
        )

        output[0] = array[-1, -1, -1]
        output[1] = array[-2, -3, -4]
        array[-1, -2, -3] = 101
        array[-2, -1, -1] = 102

    tensor = torch.arange(24, device="cuda")
    output = torch.zeros(2, dtype=torch.int64, device="cuda")
    cl.launch(torch.cuda.current_stream(), (1,), (1,), kernel, (tensor, output))

    assert output.cpu().tolist() == [23, 0]
    assert tensor.cpu().tolist() == [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        102,
        12,
        13,
        14,
        15,
        16,
        101,
        18,
        19,
        20,
        21,
        22,
        23,
    ]


def test_python_boundschecked_array_slice():
    @cl.kernel
    def kernel(tensor, output):
        array = NegativeIndexedBoundsCheckedArray(
            tensor.get_base_pointer(),
            (3, 6, 12),
        )
        view = array[:, 1:5, 10]
        nested_view = view[::-1, 1:]

        cl.static_assert(view.shape == (3, 4))
        cl.static_assert(view.strides == (72, 12))
        cl.static_assert(nested_view.shape == (3, 3))
        cl.static_assert(nested_view.strides == (-72, 12))

        output[0] = view[0, 0]
        output[1] = view[-1, -1]
        output[2] = nested_view[0, 0]
        output[3] = nested_view[-1, -1]
        view[-2, -3] = 999

    tensor = torch.arange(216, device="cuda")
    output = torch.zeros(4, dtype=torch.int64, device="cuda")
    cl.launch(torch.cuda.current_stream(), (1,), (1,), kernel, (tensor, output))

    assert output.cpu().tolist() == [22, 202, 178, 58]
    assert tensor[106].item() == 999


def test_python_boundschecked_array_negative_step_slice():
    @cl.kernel
    def kernel(tensor, output):
        array = NegativeIndexedBoundsCheckedArray(
            tensor.get_base_pointer(),
            (3, 6, 12),
        )
        view = array[-1::-1, 4:0:-2, -1]
        empty = array[-20:-20:-1, :, 0]

        cl.static_assert(view.shape == (3, 2))
        cl.static_assert(view.strides == (-72, -24))
        cl.static_assert(empty.shape == (0, 6))
        cl.static_assert(empty.strides == (-72, 12))

        output[0] = view[0, 0]
        output[1] = view[-1, -1]
        view[-2, -1] = 999

    tensor = torch.arange(216, device="cuda")
    output = torch.zeros(2, dtype=torch.int64, device="cuda")
    cl.launch(torch.cuda.current_stream(), (1,), (1,), kernel, (tensor, output))

    assert output.cpu().tolist() == [203, 35]
    assert tensor[107].item() == 999


def test_custom_strides():
    @cl.kernel
    def kernel(tensor, output):
        array = BoundsCheckedArray(tensor.get_base_pointer(), (2, 3), (1, 2))

        output[0] = array[1, 2]
        array[0, 1] = 101

    tensor = torch.arange(6, device="cuda")
    output = torch.zeros(1, dtype=torch.int64, device="cuda")
    cl.launch(torch.cuda.current_stream(), (1,), (1,), kernel, (tensor, output))

    assert output.item() == 5
    assert tensor.cpu().tolist() == [0, 1, 101, 3, 4, 5]


def test_custom_callback_is_not_called_for_valid_indices():
    @cl.kernel
    def kernel(tensor, output):
        array = BoundsCheckedArray(
            tensor.get_base_pointer(),
            (2, 3),
            None,
            custom_invalid_access,
        )
        python_array = NegativeIndexedBoundsCheckedArray(
            tensor.get_base_pointer(),
            (2, 3),
            None,
            custom_invalid_access,
        )

        output[0] = array[1, 2]
        output[1] = python_array[-2, -3]
        array[0, 1] = 101
        python_array[-1, -1] = 102

    tensor = torch.arange(6, device="cuda")
    output = torch.zeros(2, dtype=torch.int64, device="cuda")
    cl.launch(torch.cuda.current_stream(), (1,), (1,), kernel, (tensor, output))

    assert output.cpu().tolist() == [5, 0]
    assert tensor.cpu().tolist() == [0, 101, 2, 3, 4, 102]


def test_indices_must_be_a_tuple():
    def kernel():
        pointer = cl.shared_array(1, cl.int8).get_base_pointer()
        array = BoundsCheckedArray(pointer, (4,))
        array[0]

    compile_kernel(
        kernel,
        raises=pytest.raises(
            TileStaticAssertionError,
            match="indices must be a tuple",
        ),
    )


def test_index_tuple_must_match_rank():
    def kernel():
        pointer = cl.shared_array(1, cl.int8).get_base_pointer()
        array = BoundsCheckedArray(pointer, (2, 2))
        array[0,]

    compile_kernel(
        kernel,
        raises=pytest.raises(
            TileStaticAssertionError,
            match="index count must match its rank",
        ),
    )


def test_slice_assignment_is_not_supported():
    def kernel():
        pointer = cl.shared_array(1, cl.int8).get_base_pointer()
        array = NegativeIndexedBoundsCheckedArray(pointer, (2, 2))
        array[:, 1:] = 0

    compile_kernel(
        kernel,
        raises=pytest.raises(
            TileStaticAssertionError,
            match="slice assignment is not supported",
        ),
    )


def test_zero_slice_step_is_not_supported():
    def kernel():
        pointer = cl.shared_array(1, cl.int8).get_base_pointer()
        array = NegativeIndexedBoundsCheckedArray(pointer, (2, 2))
        array[::0, 0]

    compile_kernel(
        kernel,
        raises=pytest.raises(
            TileStaticAssertionError,
            match="slice step must not be zero",
        ),
    )


def crashes_with(*expected_messages):
    def decorate(kernel):
        kernel._expected_messages = expected_messages
        return kernel

    return decorate


@crashes_with("store index is less than zero")
@cl.kernel
def negative_indexed_array_store_below_extent(tensor):
    array = NegativeIndexedBoundsCheckedArray(tensor.get_base_pointer(), (2,))
    array[-3,] = 0


@crashes_with(
    "custom invalid access AccessKind.load 0 2 2",
    "custom invalid load",
)
@cl.kernel
def bounds_checked_array_custom_callback_load_above_extent(tensor):
    array = BoundsCheckedArray(
        tensor.get_base_pointer(),
        (2,),
        None,
        custom_invalid_access,
    )
    tensor[0] = array[2,]


@crashes_with(
    "returning invalid access AccessKind.load 0 2 2",
    "load index is not less than the axis extent",
)
@cl.kernel
def bounds_checked_array_returning_callback_load_above_extent(tensor):
    array = BoundsCheckedArray(
        tensor.get_base_pointer(),
        (2,),
        None,
        returning_invalid_access,
    )
    tensor[0] = array[2,]


ALL_CRASHING_KERNELS = (
    negative_indexed_array_store_below_extent,
    bounds_checked_array_custom_callback_load_above_extent,
    bounds_checked_array_returning_callback_load_above_extent,
)
CRASHING_KERNELS = {
    kernel._pyfunc.__name__: kernel
    for kernel in ALL_CRASHING_KERNELS
}


def run_crashing_kernel(kernel_name):
    kernel = CRASHING_KERNELS[kernel_name]
    tensor = torch.arange(6, device="cuda")
    cl.launch(torch.cuda.current_stream(), (1,), (1,), kernel, (tensor,))
    torch.cuda.synchronize()


@pytest.mark.parametrize("kernel_name", CRASHING_KERNELS)
def test_invalid_access_traps(kernel_name):
    kernel = CRASHING_KERNELS[kernel_name]
    proc = subprocess.run(
        [sys.executable, __file__, kernel_name],
        capture_output=True,
    )
    assert proc.returncode != 0
    stdout = proc.stdout.decode("UTF-8")
    for expected_message in kernel._expected_messages:
        assert expected_message in stdout


if __name__ == "__main__":
    run_crashing_kernel(sys.argv[1])
