# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import linecache
import re
from dataclasses import dataclass
from typing import Optional, Callable
from unicodedata import east_asian_width


@dataclass(slots=True)
class Loc:
    line: int
    col: int
    filename: Optional[str] = None
    last_line: Optional[int] = None
    end_col: Optional[int] = None
    function: Optional[Callable] = None
    call_site: Optional["Loc"] = None

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


def format_location(loc: Loc):
    if loc.is_unknown():
        return "Unknown location"

    if loc.last_line is None or loc.last_line == loc.line:
        lines_str = f"line {loc.line}"
    else:
        lines_str = f"lines {loc.line}--{loc.last_line}"

    line_text = linecache.getline(loc.filename, loc.line)
    if line_text.endswith("\n"):
        line_text = line_text[:-1]

    line_bytes = line_text.encode()

    if loc.end_col is None or loc.last_line is None or loc.last_line != loc.line:
        end_col = len(line_bytes)
    else:
        end_col = loc.end_col

    visual_col = _wcwidth(line_bytes[:loc.col].decode())
    if end_col == loc.col + 1:
        end_visual_col = visual_col
        cols_str = f"col {visual_col + 1}"
    else:
        end_visual_col = _wcwidth(line_bytes[:end_col].decode())
        cols_str = f"col {visual_col + 1}--{end_visual_col}"

    spaces = " " * visual_col
    carets = "^" * (end_visual_col - visual_col)

    return (f'  In file "{loc.filename}", {lines_str}, {cols_str}:\n'
            f"    {line_text}\n"
            f"    {spaces}{carets}\n")


class TileError(Exception):
    def __init__(self, message: str, loc: Loc = Loc.unknown()):
        self.loc = loc
        self.message = message

    def __str__(self):
        return f"{self.message}\n{format_location(self.loc)}"


class TileSyntaxError(TileError):
    pass


class TileTypeError(TileError):
    pass


class TileValueError(TileError):
    pass


class TileInternalError(TileError):
    pass


class ConstFoldNotImplementedError(Exception):
    pass


class ConstantNotFoundError(Exception):
    pass


# Simple: loc("file":line:col): error: ...
LOC_RE_SIMPLE = re.compile(
    r'loc\("([^"]+)"(?::(\d+):(\d+))?\):\s*error:\s*(.*)',
    re.I,
)

# Fused/debug wrapper: loc(fused<...>["file":line:col]): error: ...
LOC_RE_FUSED = re.compile(
    r'loc\((?:[^)]*?)\["([^"]+)":(\d+):(\d+)\]\):\s*error:\s*(.*)',
    re.I,
)

# error: ...
ERROR_RE = re.compile(r'^\s*error:\s*(.*)', re.I)


def _parse_tileir_stderr(stderr: str) -> tuple[str, Optional[Loc]]:
    msgs = []
    loc = None
    for line in stderr.splitlines():
        msg = None
        for loc_re in (LOC_RE_SIMPLE, LOC_RE_FUSED):
            if m := loc_re.search(line):
                file, line, col, msg = m.groups()
                if loc is None:
                    # Only capture the first location
                    loc = Loc(int(line) if line else None, int(col) if col else None, file)
                msg = msg.strip()
                break
        if msg is None and (m := ERROR_RE.search(line)):
            msg = m.group(1).strip()
        if msg is None:
            # fallback to the original line
            msg = line
        msgs.append(msg)
    return "\n".join(msgs), loc


class TileCompilerError(TileInternalError):
    def __init__(self,
                 message: str,
                 loc: Loc,
                 compiler_flags: str,
                 compiler_version: Optional[str]):
        super().__init__(message, loc)
        self.compiler_flags = compiler_flags
        self.compiler_version = compiler_version


class TileCompilerExecutionError(TileCompilerError):
    def __init__(self,
                 return_code: int,
                 stderr: str,
                 compiler_flags: str,
                 compiler_version: Optional[str]):
        message, loc = _parse_tileir_stderr(stderr)
        if loc is None:
            loc = _unknown_loc
        super().__init__(f"Return code {return_code}\n{message}", loc,
                         compiler_flags, compiler_version)


class TileCompilerTimeoutError(TileCompilerError):
    def __init__(self,
                 message: str,
                 compiler_flags: str,
                 compiler_version: Optional[str]):
        super().__init__(message, _unknown_loc, compiler_flags, compiler_version)
