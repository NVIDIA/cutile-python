# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Postprocessing workardound for compiler remark output."""

import re


def cleanup(value: str):
    """WAR for tileiars remarks yaml output have duplicated sections and reasons"""
    value, _ = _deduplicate_yaml_documents(value)
    value = _fold_repeated_reasons(value)
    return value


def _deduplicate_yaml_documents(value: str) -> tuple[str, int]:
    starts = [
        match.start()
        for match in re.finditer(r"(?m)^---(?:[ \t].*)?\r?$", value)
    ]
    if len(starts) < 2:
        return value, 0

    prefix = value[:starts[0]]
    ends = starts[1:] + [len(value)]
    documents = [value[start:end] for start, end in zip(starts, ends)]
    unique = []
    seen = set()
    duplicates = 0
    for document in documents:
        if document in seen:
            duplicates += 1
            continue
        seen.add(document)
        unique.append(document)
    return prefix + "".join(unique), duplicates


def _fold_repeated_reasons(value: str) -> str:
    lines = value.splitlines(keepends=True)
    folded = []
    index = 0
    while index < len(lines):
        line = lines[index]
        folded.append(line)
        if line.startswith("  - Reason:"):
            end = index + 1
            while end < len(lines) and lines[end] == line:
                end += 1
            index = end
        else:
            index += 1
    return "".join(folded)
