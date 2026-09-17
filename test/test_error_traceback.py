# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import inspect
import re
import sys
from itertools import zip_longest

import pytest

import cuda.tile as ct
import torch

from cuda.tile._execution import static_def


def raising_helper():
    raise ValueError("你好，世界")


sourceless_helper = eval("lambda: raising_helper()")


def expr_helper():
    print(sourceless_helper())


def indented_helper(use_static_eval: bool):
    if ct.ensure_constant(use_static_eval):
        ct.static_eval(expr_helper())
    else:
        ct.static_assert(False, "Boom")


@ct.kernel
def kernel_1(use_static_eval: ct.Constant[bool]):
    indented_helper(use_static_eval)


def test_traceback_formatting():
    kernel_1_line = _source_line_no(kernel_1._annotated_function.pyfunc, "indented_helper")
    indented_helper_line = _source_line_no(indented_helper, "ct.static_assert(False,")

    expected = f"""Static assertion failed: Boom
    "WHATEVERtest_error_traceback.py", line {kernel_1_line}, col 5-36, in kernel_1:
        indented_helper(use_static_eval)
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    "WHATEVERtest_error_traceback.py", line {indented_helper_line}, col 9-39, in indented_helper:
        ct.static_assert(False, "Boom")
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^"""

    with pytest.raises(ct.TileStaticAssertionError) as e:
        ct.launch(torch.cuda.current_stream(), (1,), kernel_1, (False,))
    _check_message(str(e.value), expected)


def test_traceback_formatting_static_eval():
    kernel_1_line = _source_line_no(kernel_1._annotated_function.pyfunc, "indented_helper")
    indented_helper_line = _source_line_no(indented_helper, "ct.static_eval(expr_helper())")
    expr_helper_line = _source_line_no(expr_helper, "print")
    raising_helper_line = _source_line_no(raising_helper, "raise")

    expected = f"""Exception was raised inside static_eval() (ValueError: 你好，世界)
    "WHATEVERtest_error_traceback.py", line {kernel_1_line}, col 5-36, in kernel_1:
        indented_helper(use_static_eval)
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    "WHATEVERtest_error_traceback.py", line {indented_helper_line}, col 9-37, in indented_helper:
        ct.static_eval(expr_helper())
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    [Entering compile-time evaluation inside static_eval()]
    "WHATEVERtest_error_traceback.py", line {indented_helper_line}, col 24-36, in indented_helper [inside static_eval()]:
        ct.static_eval(expr_helper())
                       ^^^^^^^^^^^^^
    "WHATEVERtest_error_traceback.py", line {expr_helper_line}, col 11-29, in expr_helper [inside static_eval()]:
        print(sourceless_helper())
              ^^^^^^^^^^^^^^^^^^^
    "<string>", line 1, in <lambda> [inside static_eval()]
    "WHATEVERtest_error_traceback.py", line {raising_helper_line}, col 5-34, in raising_helper [inside static_eval()]:
        raise ValueError("你好，世界")
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^"""  # noqa

    with pytest.raises(ct.StaticException) as e:
        ct.launch(torch.cuda.current_stream(), (1,), kernel_1, (True,))
    _check_message(str(e.value), expected)


def test_traceback_formatting_static_exception():
    @ct.kernel
    def kernel():
        raise ct.static_exception(TypeError("12345"))

    kernel_line = _source_line_no(kernel._annotated_function.pyfunc, "raise")

    expected = f"""Exception was raised at compile time (TypeError: 12345)
    "WHATEVERtest_error_traceback.py", line {kernel_line}, col 9-53, in kernel:
        raise ct.static_exception(TypeError("12345"))
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^"""

    with pytest.raises(ct.StaticException) as e:
        ct.launch(torch.cuda.current_stream(), (1,), kernel, ())
    _check_message(str(e.value), expected)


def test_traceback_formatting_static_def():
    @static_def
    def static_def_helper(x, y):
        return x // y

    @ct.kernel
    def kernel():
        static_def_helper(1, 0)

    with pytest.raises(ct.StaticException) as e:
        ct.launch(torch.cuda.current_stream(), (1,), kernel, ())

    kernel_line = _source_line_no(kernel._annotated_function.pyfunc, "static_def_helper(1, 0)")
    helper_line = _source_line_no(static_def_helper, "return x // y")

    expected = f"""Exception was raised inside @static_def function (ZeroDivisionError: WHATEVER)
    "WHATEVERtest_error_traceback.py", line {kernel_line}, col 9-31, in kernel:
        static_def_helper(1, 0)
        ^^^^^^^^^^^^^^^^^^^^^^^
    [Entering compile-time evaluation inside @static_def]
    "WHATEVERtest_error_traceback.py", line {helper_line}, col 16-21, in static_def_helper [inside @static_def]:
        return x // y
               ^^^^^^"""  # noqa
    _check_message(str(e.value), expected)


def test_traceback_formatting_enter_static_eval_inside_lambda():
    def foo():
        raise TypeError("hi")

    @ct.kernel
    def kernel():
        lamb = lambda: ct.static_eval(foo())  # noqa
        lamb()

    kernel_lambda_line = _source_line_no(kernel._annotated_function.pyfunc, "lambda:")
    kernel_call_line = _source_line_no(kernel._annotated_function.pyfunc, "lamb()")
    foo_helper_line = _source_line_no(foo, "raise TypeError")

    expected = f"""Exception was raised inside static_eval() (TypeError: hi)
    "WHATEVERtest_error_traceback.py", line {kernel_call_line}, col 9-14, in kernel:
        lamb()
        ^^^^^^
    "WHATEVERtest_error_traceback.py", line {kernel_lambda_line}, col 24-44, in <lambda>:
        lamb = lambda: ct.static_eval(foo())  # noqa
                       ^^^^^^^^^^^^^^^^^^^^^
    [Entering compile-time evaluation inside static_eval()]
    "WHATEVERtest_error_traceback.py", line {kernel_lambda_line}, col 39-43, in <lambda> [inside static_eval()]:
        lamb = lambda: ct.static_eval(foo())  # noqa
                                      ^^^^^
    "WHATEVERtest_error_traceback.py", line {foo_helper_line}, col 9-29, in foo [inside static_eval()]:
        raise TypeError("hi")
        ^^^^^^^^^^^^^^^^^^^^^"""  # noqa

    with pytest.raises(ct.StaticException) as e:
        ct.launch(torch.cuda.current_stream(), (1,), kernel, ())
    _check_message(str(e.value), expected)


def test_traceback_formatting_error_inside_static_assert_expr():
    @ct.kernel
    def kernel():
        ct.static_assert(1 // 0)

    kernel_line = _source_line_no(kernel._annotated_function.pyfunc, "ct.static_assert(1 // 0)")

    expected = f"""Exception was raised inside static_assert() condition (ZeroDivisionError: WHATEVER)
    "WHATEVERtest_error_traceback.py", line {kernel_line}, col 9-32, in kernel:
        ct.static_assert(1 // 0)
        ^^^^^^^^^^^^^^^^^^^^^^^^
    [Entering compile-time evaluation inside static_assert() condition]
    "WHATEVERtest_error_traceback.py", line {kernel_line}, col 26-31, in kernel [inside static_assert() condition]:
        ct.static_assert(1 // 0)
                         ^^^^^^"""  # noqa

    with pytest.raises(ct.StaticException) as e:
        ct.launch(torch.cuda.current_stream(), (1,), kernel, ())
    _check_message(str(e.value), expected)


def test_traceback_formatting_error_inside_static_assert_message():
    @ct.kernel
    def kernel():
        ct.static_assert(False, 1 // 0)

    kernel_line = _source_line_no(kernel._annotated_function.pyfunc, "ct.static_assert(")

    expected = f"""Exception was raised inside static_assert() message (ZeroDivisionError: WHATEVER)
    "WHATEVERtest_error_traceback.py", line {kernel_line}, col 9-39, in kernel:
        ct.static_assert(False, 1 // 0)
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    [Entering compile-time evaluation inside static_assert() message]
    "WHATEVERtest_error_traceback.py", line {kernel_line}, col 33-38, in kernel [inside static_assert() message]:
        ct.static_assert(False, 1 // 0)
                                ^^^^^^"""  # noqa

    with pytest.raises(ct.StaticException) as e:
        ct.launch(torch.cuda.current_stream(), (1,), kernel, ())
    _check_message(str(e.value), expected)


def _source_line_no(func, substring: str) -> int:
    lines, first_line_no = inspect.getsourcelines(func)
    for i, s in enumerate(lines):
        if substring in s:
            return i + first_line_no
    assert False, f"Line containing '{substring}' not found in function's source"


def _check_message(actual: str, expected_template: str):
    expected_lines = expected_template.splitlines()

    # Prior to Python 3.11, code objects have no column-level info.
    # Need to strip column information from the expected message.
    if (sys.version_info.major, sys.version_info.minor) < (3, 11):
        filtered_lines = []
        static_eval_frame = False
        for line in expected_lines:
            if static_eval_frame:
                # Skip caret underlines (^^^^^^^^^^^^^^^)
                if re.fullmatch(" *\\^+", line):
                    continue
                filtered_lines.append(re.sub("col [^,]*, ", "", line))
            else:
                filtered_lines.append(line)
                if re.search("Entering compile-time", line):
                    static_eval_frame = True
        expected_lines = filtered_lines
        print("\n".join(expected_lines))

    actual_lines = actual.splitlines()
    for expected, actual in zip_longest(expected_lines, actual_lines):
        assert expected is not None
        assert actual is not None
        pat = re.escape(expected).replace("WHATEVER", ".*")
        assert re.fullmatch(pat, actual), f"Mismatch!\nActual:   {actual}\nExpected: {expected}\n"
