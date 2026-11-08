# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
import dataclasses
import linecache
import os.path
import traceback
from dataclasses import dataclass
from typing import Optional
from unicodedata import east_asian_width


@dataclass(eq=False, frozen=True)
class FunctionDesc:
    name: str | None  # None for lambdas
    filename: str
    line: int  # 1-based
    column: int  # 1-based
    # If this FunctionDesc represents a concrete specialization of a source
    # function other than the kernel entry point, this value will hold a
    # unique identifier, which is used to distinguish distinct specialized
    # functions in debug info.
    specialization_id: str | None = None
    # True for the FunctionDesc that represents the kernel entry point.
    is_entry: bool = False

    def __str__(self):
        return f"'{self.name}' @{self.filename}:{self.line}:{self.column}"

    def short_str(self):
        if self.name is None:
            base_name = os.path.basename(self.filename)
            return f"<lambda at {base_name}:{self.line}:{self.column}>"
        else:
            return f"<function {self.name}>"


@dataclass(slots=True, frozen=True)
class Loc:
    line: int
    col: int
    filename: Optional[str] = None
    last_line: Optional[int] = None
    end_col: Optional[int] = None
    function: Optional[FunctionDesc] = None
    call_site: Optional["Loc"] = None

    def with_call_site(self, call_site) -> "Loc":
        return dataclasses.replace(self, call_site=call_site)

    def __str__(self) -> str:
        if self.filename:
            return f"{self.filename}:{self.line}:{self.col}"
        return f"<unknown>:{self.line}:{self.col}"

    @classmethod
    def unknown(cls) -> "Loc":
        return _unknown_loc

    def is_unknown(self) -> bool:
        return self is _unknown_loc

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None and issubclass(exc_type, TileError):
            if exc_val.loc.is_unknown():
                exc_val.loc = self


_unknown_loc = Loc(line=0, col=0, filename=None)


# Returns the visual column width of a string, accounting for double-wide characters
def _wcwidth(s: str) -> int:
    return sum(2 if east_asian_width(c) in ("W", "F") else 1 for c in s)


def format_location(loc: Loc, exc: Exception | None = None):
    pieces = []
    while loc is not None:
        if loc.is_unknown():
            pieces.append("Unknown location")
        else:
            func_name = "<lambda>" if loc.function.name is None else loc.function.name
            pieces.append(_format_location_frame(
                loc.filename, func_name, loc.line, loc.last_line, loc.col, loc.end_col))
        loc = loc.call_site
    pieces.reverse()

    if isinstance(exc, StaticEvalError) and exc.__cause__ is not None:
        from cuda.tile._ir.static_eval_ops import do_static_eval_impl
        from cuda.tile._passes.hir2ir import _call_static_def_function
        static_eval_traceback = exc.__cause__.__traceback__
        while static_eval_traceback is not None:
            frame = static_eval_traceback.tb_frame
            static_eval_traceback = static_eval_traceback.tb_next
            if frame.f_code is do_static_eval_impl.__code__:
                break
            if frame.f_code is _call_static_def_function.__code__:
                break

        if static_eval_traceback is not None:
            where = getattr(exc, "_where", None)
            if where is None:
                where_suffix = " [static]"
                where_str = ""
            else:
                where_suffix = f" [{where}]"
                where_str = f" {where}"
            pieces.append(f"    [Entering compile-time evaluation{where_str}]\n")

            for frame in traceback.extract_tb(static_eval_traceback):
                pieces.extend(_format_location_frame(
                        frame.filename, frame.name + where_suffix, frame.lineno,
                        getattr(frame, "end_lineno", None),
                        getattr(frame, "colno", None),
                        getattr(frame, "end_colno", None)))

    return "".join(pieces)


def _format_location_frame(filename: str | None,
                           function_name: str | None,
                           line: int | None,
                           last_line: int | None,
                           col: int | None,
                           end_col: int | None) -> str:
    if line is None:
        lines_str = ""
    elif last_line is None or last_line == line:
        lines_str = f", line {line}"
    else:
        lines_str = f", lines {line}--{last_line}"

    line_text = "" if filename is None or line is None else linecache.getline(filename, line)
    cols_str = ""
    text_str = ""

    if line_text:
        line_text = line_text.rstrip()
        line_bytes = line_text.encode()

        stripped_text = line_text.lstrip()
        stripped_count = len(line_text) - len(stripped_text)
        stripped_width = _wcwidth(line_text[:stripped_count])

        text_str = f":\n        {stripped_text}"

        if col is not None:
            if last_line is None or last_line < line:
                end_col = None
            elif last_line > line:
                end_col = len(line_bytes)

            orig_visual_col = _wcwidth(line_bytes[:col].decode())
            visual_col = max(orig_visual_col - stripped_width, 0)
            if end_col is None or end_col == col + 1:
                end_visual_col = visual_col + 1
                cols_str = f", col {orig_visual_col + 1}"
            else:
                orig_end_visual_col = _wcwidth(line_bytes[:end_col].decode())
                end_visual_col = max(orig_end_visual_col - stripped_width, visual_col + 1)
                cols_str = f", col {orig_visual_col + 1}-{orig_end_visual_col}"

            spaces = " " * visual_col
            carets = "^" * (end_visual_col - visual_col)
            text_str += f"\n        {spaces}{carets}"

    func_str = "" if function_name is None else f", in {function_name}"
    return f'    "{filename}"{lines_str}{cols_str}{func_str}{text_str}\n'


class TileError(Exception):
    def __init__(self, message: str, loc: Loc = Loc.unknown()):
        self.loc = loc
        self.message = message

    def __str__(self):
        return f"{self.message}\n{format_location(self.loc, self)}"


class UnsupportedSyntaxError(TileError):
    """Exception when a python syntax not supported by cuTile is encountered."""
    pass


TileSyntaxError = UnsupportedSyntaxError


class TypeCheckingError(TileError):
    """Exception when an unexpected type or |data type| is encountered."""
    pass


TileTypeError = TypeCheckingError


class RecursionLimitError(TileError):
    """Thrown at compile time to indicate that the recursion limit has been reached
    when inlining a function call.
    """


TileRecursionError = RecursionLimitError


class InvalidValueError(TileError):
    """Exception when an unexpected python value is encountered."""
    pass


TileValueError = InvalidValueError


class UnsupportedFeatureError(TileError):
    """Exception when a feature is not supported by the underlying compiler or
      the GPU architecture."""
    pass


TileUnsupportedFeatureError = UnsupportedFeatureError


class InternalError(TileError):
    pass


TileInternalError = InternalError


class StaticEvalError(TileError):
    """Thrown at compile time when the expression inside static_eval() violates the compile-time
    evaluation constraints."""


TileStaticEvalError = StaticEvalError


class StaticException(StaticEvalError):
    """Raised at compile time when a compile-time exception is raised, either via
    `raise static_exception(...)` or by a `raise` statement inside a `static_eval()` expression."""


def make_static_exception(orig_exception: BaseException) -> StaticException:
    if not isinstance(orig_exception, BaseException):
        raise TypeError(f"'{type(orig_exception).__name__}' is not derived from 'BaseException'")
    return StaticException(
            f"Exception was raised at compile time ({exception_type_and_str(orig_exception)})")


class StaticAssertionError(TileError):
    """Thrown at compile time when the condition of static_assert() evaluates to False."""

    def __init__(self, message: str, loc: Loc = Loc.unknown()):
        full_message = "Static assertion failed"
        if len(message) > 0:
            full_message += ": " + message
        super().__init__(full_message, loc)


def exception_type_and_str(e: BaseException) -> str:
    name = type(e).__name__
    e_str = str(e)
    return name + ": " + e_str if len(e_str) > 0 else name


TileStaticAssertionError = StaticAssertionError


class UnsupportedCallError(TileError):
    """Raised when a function or type is unsupported in the current execution space."""


class ConstantNotFoundError(Exception):
    pass


class InternalCompilerError(InternalError):
    def __init__(self,
                 message: str,
                 loc: Loc,
                 compiler_flags: str,
                 compiler_version: Optional[str]):
        super().__init__(message, loc)
        self.compiler_flags = compiler_flags
        self.compiler_version = compiler_version


TileCompilerError = InternalCompilerError


class CompilerExecutionError(InternalCompilerError):
    """Exception when compiler throws an error."""
    def __init__(self,
                 return_code: int,
                 message: str,
                 loc: Loc,
                 compiler_flags: str,
                 compiler_version: Optional[str]):
        self.return_code = return_code
        super().__init__(f"Return code {return_code}\n{message}", loc,
                         compiler_flags, compiler_version)


TileCompilerExecutionError = CompilerExecutionError


class CompilerTimeoutError(InternalCompilerError):
    """Exception when the compiler timeout limit is exceeded."""
    def __init__(self,
                 message: str,
                 compiler_flags: str,
                 compiler_version: Optional[str]):
        super().__init__(message, _unknown_loc, compiler_flags, compiler_version)


TileCompilerTimeoutError = CompilerTimeoutError
