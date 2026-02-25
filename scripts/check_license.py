# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import subprocess
import sys

skip_patterns = [
    "VERSION",
    ".svg",
    "uv.lock",
    "LICENSES/",
]

max_lines_to_check = 10


def check_license():
    result = subprocess.run(
        ["git", "ls-files"],
        capture_output=True,
        text=True,
        check=True,
    )
    bad_files = 0
    num_files = 0

    for filepath in result.stdout.splitlines():
        if any(s in filepath for s in skip_patterns):
            continue
        try:
            with open(filepath, "r") as f:
                head = ""
                for i, line in enumerate(f):
                    if i >= max_lines_to_check:
                        break
                    head += line
        except (UnicodeDecodeError, OSError):
            continue

        if not head:
            continue
        num_files += 1
        has_error = False
        if "SPDX-FileCopyrightText" not in head:
            print(f"{filepath}: no copyright notice", file=sys.stderr)
            has_error = True
        if "SPDX-License-Identifier" not in head:
            print(f"{filepath}: no license identifier", file=sys.stderr)
            has_error = True
        if has_error:
            bad_files += 1

    if bad_files > 0:
        print(f"Found {bad_files} files with missing or incomplete license headers",
              file=sys.stderr)
        sys.exit(1)
    elif num_files == 0:
        print("No input files found!", file=sys.stderr)
        sys.exit(2)
    else:
        print(f"Checked {num_files} files, all OK")


if __name__ == "__main__":
    check_license()
