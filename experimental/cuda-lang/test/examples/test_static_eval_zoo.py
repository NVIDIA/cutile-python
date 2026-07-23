# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any
from typing import Callable
from cuda.lang.compilation import KernelSignature
from cuda.lang._exception import CompilerExecutionError
from test.util import compile_kernel
import cuda.lang as cl
from cuda.lang._stub.foreign_function import _call_foreign_function as ffi
from cuda.lang._ir.type import (
    SymbolicArray,
    SymbolicVector,
    SymbolicScalar,
)

from dataclasses import dataclass, replace
import functools
import pytest
import sys
import torch


def static_define(function):
    """Decorator that wraps every call to ``function`` in ``static_eval``."""

    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        return cl.static_eval(function(*args, **kwargs))

    return wrapper


static_print = static_define(print)


@static_define
def static_breakpoint(*args, **kwarsg):
    """
    Function allowing users to inspect values inside a cuda.lang kernel at
    compile time.
    """
    breakpoint()


def test_static_breakpoint(monkeypatch):
    """
    Set a breakpoint in kernel code to inspect symbolic values for runtime
    values or the real values of compile-time constants.
    """
    call_count = 0

    def spy_breakpointhook(*args, **kwargs):
        nonlocal call_count
        call_count += 1

    monkeypatch.setattr(sys, "breakpointhook", spy_breakpointhook)

    def kernel():
        with cl.local_array(1, cl.int32) as array:
            # examine the symbol representing the runtime value ``array``,
            # and the literal value ``5`` in the Python debugger at compile
            # time.
            static_breakpoint(array=array, five=5)

    compile_kernel(kernel)
    assert call_count == 1


@pytest.mark.parametrize("problem_size", [1, 32, 40])
@pytest.mark.parametrize("unroll_factor", [1, 4, 8])
def test_loop_unroller(unroll_factor, problem_size):
    """
    With higher order programming and static_eval/static_iter, we are able
    to run some iterations of the loop at compile time and some at runtime,
    expressing a naive loop unroller in <10 lines of code.
    """

    def unroll(function, /, *, count: int, unroll_factor: int):
        count_per_unrolled = count // unroll_factor

        for i in range(count_per_unrolled):
            for c in cl.static_iter(range(unroll_factor)):
                iv = i * unroll_factor + c
                if iv < count:
                    function(iv)

    @cl.kernel
    def kernel(out):

        def loop_body(induction_variable: int):
            print(induction_variable)
            out[induction_variable] = induction_variable

        unroll(
            loop_body,
            count=problem_size,
            unroll_factor=unroll_factor,
        )

    out = torch.zeros(problem_size, dtype=torch.int32).cuda()
    sig = KernelSignature.from_kernel_args(kernel, [out])
    compiled = compile_kernel(
        kernel,
        signature=sig,
        keep_final_ir=True,
    )
    assert compiled and compiled.final_ir
    assert str(compiled.final_ir).count("tile_printf") == unroll_factor, (
        "Expected number of calls to print to be the same as the unrolling factor"
    )

    cl.launch(torch.cuda.current_stream(), (1,), (1,), kernel, (out,))
    assert out.cpu().tolist() == list(range(problem_size))


def static_overload(function):
    """Similar to numba's high-level extension api, this function decorates
    a function called at compile time which returns a function to be called
    with the same arguments at runtime."""

    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        resolved = cl.static_eval(function(*args, **kwargs))
        return resolved(*args, **kwargs)

    return wrapper


@static_overload
def ffi_overload(arg):
    match arg:
        case SymbolicScalar():
            dtype = arg.dtype
            entrypoint = f"fn_{dtype}"

            def fn(x):
                return ffi(entrypoint, dtype, (x,))

            return fn

        case int():

            def fn(x):
                return ffi("fn_int32", cl.int32, (cl.int32(x),))

            return fn
        case float():

            def fn(x):
                return ffi("fn_float32", cl.float32, (cl.float32(x),))

            return fn
        case _:
            assert False


def test_static_overload_ffi_example_symbolic():

    def kernel():
        smem1 = cl.shared_array(1, cl.int32)
        ffi_overload(smem1[0])

    compile_kernel(
        kernel,
        raises=pytest.raises(
            CompilerExecutionError, match="Unresolved extern function 'fn_int32'"
        ),
    )


def test_static_overload_ffi_example_constant():

    def kernel():
        ffi_overload(5.0)

    compile_kernel(
        kernel,
        raises=pytest.raises(
            CompilerExecutionError, match="Unresolved extern function 'fn_float32'"
        ),
    )


def test_static_overload_selection():

    @static_overload
    def overloaded_function(symbolic_value):
        """This runs at compile-time, so we could invoke nvrtc to compile an
        object file and link it to the final kernel if we supported ltoir."""

        match symbolic_value:
            case SymbolicArray():

                def func(array):
                    return array[0] + 1

                return func
            case SymbolicScalar() | int() | float():

                def func(scalar):
                    return cl.int32(scalar) + 1

                return func
            case SymbolicVector() if symbolic_value.element_count == 2:

                def func(vector):
                    return vector[0] + vector[1]

                return func
            case _:
                raise TypeError(f"Unexpected type {symbolic_value}")

    @cl.kernel
    def kernel(out):
        vector = out.get_element_pointer(0).load(count=2)
        out[0] = overloaded_function(out)  # array
        out[1] = overloaded_function(5)  # constant scalar
        out[2] = overloaded_function(out[0])  # symbolic scalar
        out[3] = overloaded_function(vector)  # vector

    out = torch.tensor(list(range(5)), dtype=torch.int32).cuda()
    cl.launch(torch.cuda.current_stream(), (1,), (1,), kernel, (out,))
    assert out.cpu().tolist() == [1, 6, 2, 1, 4]


def bitonic_schedule(width):
    """Build compare/exchange graph in plain python."""
    assert width > 0 and width & (width - 1) == 0
    size = 2
    while size <= width:
        stride = size // 2
        while stride:
            for lhs in range(width):
                rhs = lhs ^ stride
                if rhs > lhs:
                    yield lhs, rhs, (lhs & size) == 0
            stride //= 2
        size *= 2


@cl.kernel
def bitonic_sort_kernel(inp, out, SORT_WIDTH: cl.Constant[int]):
    with cl.local_array(SORT_WIDTH, cl.int32) as values:
        for index in cl.static_iter(range(SORT_WIDTH)):
            values[index] = inp[index]

        for lhs, rhs, ascending in cl.static_iter(bitonic_schedule(SORT_WIDTH)):
            lhs_value = values[lhs]
            rhs_value = values[rhs]
            low = cl.minimum(lhs_value, rhs_value)
            high = cl.maximum(lhs_value, rhs_value)
            if ascending:
                values[lhs] = low
                values[rhs] = high
            else:
                values[lhs] = high
                values[rhs] = low

        for index in cl.static_iter(range(SORT_WIDTH)):
            out[index] = values[index]


def test_static_bitonic_sorting_schedule():
    inp = torch.tensor([7, -2, 5, 5, 0, 9, 1, -4], dtype=torch.int32, device="cuda")
    out = torch.empty_like(inp)
    cl.launch(
        torch.cuda.current_stream(),
        (1,),
        (1,),
        bitonic_sort_kernel,
        (inp, out, inp.shape[0]),
    )
    assert out.cpu().tolist() == sorted(inp.cpu().tolist())


def test_continuation_passing_style_control_flow():
    def ite(i, t, e):
        def thunk(*args, **kwargs):
            if i(*args, **kwargs):
                return t(*args, **kwargs)
            else:
                return e(*args, **kwargs)

        return thunk

    clamp_max = 5

    @static_overload
    def build_and_call_dynamic_control_flow(array, i):
        """At compile time, build dynamic control flow to clamp the input value"""

        def condition(array, i):
            return array[i] > clamp_max

        def then(array, i):
            return clamp_max

        def else_(array, i):
            return array[i]

        return ite(condition, then, else_)

    @cl.kernel
    def kernel(out):
        out[0] = build_and_call_dynamic_control_flow(out, 0)
        out[1] = build_and_call_dynamic_control_flow(out, 1)

    out = torch.tensor([3, 7], dtype=torch.int32).cuda()
    cl.launch(torch.cuda.current_stream(), (1,), (1,), kernel, (out,))
    assert out.cpu().tolist() == [3, 5]


def test_epilogue_operation_tuple():
    @dataclass(frozen=True)
    class ElementWiseApply:
        fn: Callable[[Any], Any]

        def run(self, tensor: cl.Array):
            cl.static_assert(tensor.ndim == 1)
            for i in range(tensor.shape[0]):
                print(i, tensor[i])
                tensor[i] = self.fn(tensor[i])
                print(i, tensor[i])

    @cl.kernel
    def kernel(out):
        def plus(how_much):
            def thunk(value):
                return value + how_much

            return thunk

        def clamp(minval, maxval):
            def thunk(value):
                return cl.minimum(
                    cl.maximum(
                        minval,
                        value,
                    ),
                    maxval,
                )

            return thunk

        operations = (
            ElementWiseApply(clamp(minval=-2, maxval=8)),
            ElementWiseApply(plus(3)),
        )
        for operation in cl.static_iter(operations):
            operation.run(out)

    out = torch.tensor(range(16), dtype=torch.int32).cuda()
    cl.launch(torch.cuda.current_stream(), (1,), (1,), kernel, (out,))
    assert out.cpu().tolist() == [
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        11,
        11,
        11,
        11,
        11,
        11,
        11,
    ]


def test_dataclass_state_management():
    """
    Only frozen dataclasses are supported so that dataclasses can be plain
    aggregates and don't need backing storage allocated somewhere.
    This means we can't assign to fields (`self.foo = bar`), but we can use
    `dataclasses.replace(self, foo=bar)` and static methods to perform setup
    or validation on construction.
    """

    @dataclass(frozen=True)
    class C:
        state_: int = 0

        @property
        def state(self):
            return self.state_

        @staticmethod
        def new(initial_state):
            # perform some validation on construction
            cl.assert_(initial_state >= 0)
            return C(state_=initial_state)

        def advance(self):
            # "modify" the state by returning a new object with the updated value
            return replace(self, state_=self.state + 1)

    @cl.kernel
    def kernel(trip_count, out):
        c = C.new(3)
        for i in range(trip_count):
            c = c.advance()
        out[0] = c.state

    out = torch.zeros(1, dtype=torch.int32, device="cuda")
    cl.launch(torch.cuda.current_stream(), (1,), (1,), kernel, (24, out))
    assert out.cpu().tolist() == [27]
