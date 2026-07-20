# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import argparse
import contextlib
import datetime
import itertools
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import textwrap
from collections.abc import Iterable, Sequence
from typing import TextIO

from cutile_cache._cache import CacheEntry, cache_entries
from cutile_cache._env import get_cache_dir_from_env

_COMMIT_COLOR = "\033[33m"
_COLOR_RESET = "\033[m"
_REMARK_TAG_COLORS = {
    "Passed": "\033[32m",
    "Failure": "\033[31m",
    "Analysis": "\033[36m",
}


def _format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "unknown"
    if seconds < 1:
        return f"{seconds * 1000:.1f} ms"
    return f"{seconds:.3f} s"


def _format_size(size: int) -> str:
    if size < 1024:
        return f"{size:,} bytes"
    units = ("KiB", "MiB", "GiB", "TiB")
    value = float(size)
    for unit in units:
        value /= 1024
        if value < 1024 or unit == units[-1]:
            return f"{size:,} bytes ({value:.1f} {unit})"
    raise AssertionError("unreachable")


def _output_width() -> int:
    return max(60, min(shutil.get_terminal_size((100, 24)).columns, 140))


def format_date(timestamp: float | None) -> str:
    if timestamp is None:
        return "unknown"
    date = datetime.datetime.fromtimestamp(timestamp).astimezone()
    return date.strftime("%a %b %d %H:%M:%S %Y %z")


def _format_compiler_version(compiler_ver: str | None) -> str:
    if compiler_ver is not None:
        match = re.search(
            r"Cuda compilation tools,\s*release\s*([^,\n]+),\s*V([^\s\n]+)",
            compiler_ver,
            re.IGNORECASE,
        )
        if match is not None:
            _, version = match.groups()
            return f"tileiras {version}"
    return "unknown"


def _format_field(label: str, value: str, wrap: bool = False) -> str:
    prefix = f"{label}:".ljust(12)
    padding = " " * len(prefix)
    lines = value.splitlines() or [value]
    if wrap:
        wrapped = []
        for line in lines:
            wrapped.extend(textwrap.wrap(
                line,
                width=max(20, _output_width() - len(prefix)),
                break_long_words=True,
                break_on_hyphens=False,
            ) or [""])
        lines = wrapped
    return prefix + lines[0] + "".join(f"\n{padding}{line}" for line in lines[1:])


def _highlight_remark_tags(value: str) -> str:
    def highlight(match: re.Match) -> str:
        tag = match.group(1)
        color = _REMARK_TAG_COLORS.get(tag)
        if color is None:
            return match.group(0)
        return f"--- {color}!{tag}{_COLOR_RESET}"

    return re.sub(r"(?m)^--- !([^ \t\r\n]+)$", highlight, value)


def _format_remarks(value: str | None, color: bool = False) -> list[str]:
    if not value:
        return []
    if color:
        value = _highlight_remark_tags(value)
    heading = "Remarks:"
    return [heading, textwrap.indent(value.rstrip(), "    ")]


def _format_entry(entry: CacheEntry, color: bool = False) -> str:
    commit = f"commit {entry.key}"
    if color:
        commit = f"{_COMMIT_COLOR}{commit}{_COLOR_RESET}"

    meta = entry.metadata
    names = meta.kernel_names
    kernels = ", ".join(names) if names else "unknown"
    fields = [
        commit,
        _format_field("Kernel", kernels, wrap=True),
        _format_field("Compiler", _format_compiler_version(meta.compiler_version)),
        _format_field("Date", format_date(meta.compilation_timestamp)),
        _format_field("Last used", format_date(entry.atime)),
    ]
    fields.extend([
        _format_field("Duration", _format_duration(meta.compilation_time_seconds)),
        _format_field("CUBIN", _format_size(entry.size)),
    ])
    remarks = _format_remarks(meta.remarks, color=color)
    if remarks:
        fields.extend(["", *remarks])
    return "\n".join(fields)


def _write_entries(entries: Iterable[CacheEntry], stream: TextIO,
                   color: bool = False) -> None:
    first = True
    for entry in entries:
        if not first:
            stream.write("\n")
        stream.write(_format_entry(entry, color=color))
        stream.write("\n")
        stream.flush()
        first = False


def _page_entries(entries: Iterable[CacheEntry]) -> None:
    if not sys.stdout.isatty():
        _write_entries(entries, sys.stdout)
        return

    color = os.environ.get("TERM") != "dumb"
    pager = os.environ.get("MANPAGER") or os.environ.get("PAGER")
    shell = pager is not None
    if pager is None:
        if less := shutil.which("less"):
            pager = [less, "-FRX"]
        elif more := shutil.which("more"):
            pager = [more]
        else:
            _write_entries(entries, sys.stdout, color=color)
            return

    pager_env = os.environ.copy()
    pager_env.setdefault("LESS", "FRX")
    process = subprocess.Popen(
        pager, shell=shell, stdin=subprocess.PIPE, text=True, env=pager_env
    )
    assert process.stdin is not None
    try:
        _write_entries(entries, process.stdin, color=color)
    except BrokenPipeError:
        pass
    finally:
        try:
            process.stdin.close()
        except BrokenPipeError:
            pass
        process.wait()


def _cache_log(_args: argparse.Namespace) -> int:
    cache_dir = get_cache_dir_from_env()
    if cache_dir is None:
        print("cutile-cache: compilation cache is disabled", file=sys.stderr)
        return 1

    entries = cache_entries(cache_dir)
    try:
        with contextlib.closing(entries):
            try:
                first = next(entries)
            except StopIteration:
                return 0
            _page_entries(itertools.chain((first,), entries))
    except (OSError, sqlite3.Error) as error:
        print(f"cutile-cache: cannot read compilation cache: {error}", file=sys.stderr)
        return 1
    return 0


def _create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cutile-cache", description="inspect the cuTile compilation cache"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    log_parser = commands.add_parser(
        "log", help="show compilation cache entries in most-recently-accessed order"
    )
    log_parser.set_defaults(handler=_cache_log)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _create_parser().parse_args(argv)
    try:
        return args.handler(args)
    except BrokenPipeError:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
