#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import os
import sys

file_extensions = [
    ".h",
    ".hpp",
    ".hh",
    ".c",
    ".C",
    ".cpp",
    ".cxx",
    ".cc",
    ".pyx",
    ".pxd",
]
max_line_len = 100


def should_lint(filename: str):
    return any(filename.endswith(x) for x in file_extensions)


def lint(paths):
    num_errors = 0
    num_files = 0

    def report_error(message: str):
        nonlocal num_errors
        print(f"{full_name[len(path) + 1:]}:{i + 1}: {message}", file=sys.stderr)
        num_errors += 1

    for path in paths:
        for root, dirs, files in os.walk(path):
            for filename in files:
                if not should_lint(filename):
                    continue
                full_name = os.path.join(root, filename)
                with open(full_name, "r") as f:
                    for i, line in enumerate(f):
                        if "noqa" in line:
                            continue
                        if "SPDX" in line:
                            continue

                        length = len(line)
                        if line.endswith("\n"):
                            length -= 1
                        if length > max_line_len:
                            report_error(
                                f"Line is longer than {max_line_len} characters"
                            )
                        if length > 0 and line[length - 1].isspace():
                            report_error("Trailing whitespace at the end of the line")
                num_files += 1

    if num_errors > 0:
        print(f"Found {num_errors} errors", file=sys.stderr)
        sys.exit(1)
    elif num_files == 0:
        print("No input files found!", file=sys.stderr)
        sys.exit(2)
    else:
        print(f"Checked {num_files} files, all OK")


if __name__ == "__main__":
    script_dir = os.path.dirname(__file__)
    project_root = os.getcwd()
    dirs = ["cext", "src"]
    paths = [os.path.join(project_root, d) for d in dirs]
    lint(paths)
