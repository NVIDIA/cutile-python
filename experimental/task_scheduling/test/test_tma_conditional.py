# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Trace marker tests for the conditional TMA copy tutorial.

``03_copy_tma_conditional`` writes a six-column trace side channel: column 0
holds ``BEGIN_MARKER`` at ``trace[0, 0]``, column 1 holds
``HEARTBEAT_MARKER`` every fourth row starting at row 0, column 2 holds
``HIGHLIGHT_MARKER`` at ``trace[highlight_row, 2]``, and column 3 holds
``END_MARKER`` at ``trace[rows - 1, 3]``. Column 4 marks the highlight-or-last
boolean expression and column 5 marks first-and-highlight. The helpers assume
``rows > 0`` and an integer ``highlight_row`` with
``0 <= highlight_row < rows``.

The parametrized rejection cases mutate one marker at a time to check the
begin, heartbeat, highlight, end, edge, and first-highlight columns.
"""

import importlib

import pytest
import torch

from task_scheduling_test_requirements import require_task_scheduling_dependencies


def _conditional_copy_module():
    require_task_scheduling_dependencies()
    return importlib.import_module(
        "experimental.task_scheduling.tutorial.01_copy_basics."
        "03_copy_tma_conditional"
    )


def _valid_trace(rows: int, highlight_row: int) -> torch.Tensor:
    mod = _conditional_copy_module()
    trace = torch.zeros(rows, 6, dtype=torch.float16)
    trace[0, 0] = mod.BEGIN_MARKER
    trace[list(range(0, rows, 4)), 1] = mod.HEARTBEAT_MARKER
    trace[highlight_row, 2] = mod.HIGHLIGHT_MARKER
    trace[rows - 1, 3] = mod.END_MARKER
    trace[sorted({highlight_row, rows - 1}), 4] = mod.EDGE_MARKER
    if highlight_row == 0:
        trace[0, 5] = mod.FIRST_HIGHLIGHT_MARKER
    return trace


def test_conditional_tma_trace_verifier_accepts_exact_markers():
    mod = _conditional_copy_module()
    rows = 9
    highlight_row = 5

    mod._verify_trace_markers(
        _valid_trace(rows, highlight_row),
        num_rows=rows,
        highlight_row=highlight_row,
    )


def test_conditional_tma_trace_verifier_accepts_first_highlight():
    mod = _conditional_copy_module()
    rows = 9
    mod._verify_trace_markers(
        _valid_trace(rows, 0),
        num_rows=rows,
        highlight_row=0,
    )


def test_conditional_copy_program_infers_resource_roles():
    program = _conditional_copy_module().make_tma_copy_program()
    load_task, store_task, padding_task = program.manager.user_tasks

    assert [resource.name for resource in load_task.src_resources] == [
        "inputGmemResource"
    ]
    assert [resource.name for resource in load_task.dst_resources] == [
        "smemResource",
        "traceGmemResource",
    ]
    assert [resource.name for resource in store_task.src_resources] == [
        "smemResource"
    ]
    assert [resource.name for resource in store_task.dst_resources] == [
        "outputGmemResource",
        "traceGmemResource",
    ]
    assert padding_task.resources == []

    assert program.manager.resource_dependency_graph == {}


@pytest.mark.parametrize(
    "mutate, message",
    [
        pytest.param(
            lambda trace, mod: trace.__setitem__((1, 0), mod.BEGIN_MARKER),
            "begin trace column 0",
            id="extra-begin",
        ),
        pytest.param(
            lambda trace, mod: trace.__setitem__((0, 0), 0),
            "begin trace column 0",
            id="missing-begin",
        ),
        pytest.param(
            lambda trace, mod: trace.__setitem__((1, 1), mod.HEARTBEAT_MARKER),
            "heartbeat trace column 1",
            id="extra-heartbeat",
        ),
        pytest.param(
            lambda trace, mod: trace.__setitem__((4, 1), 0),
            "heartbeat trace column 1",
            id="missing-heartbeat",
        ),
        pytest.param(
            lambda trace, mod: trace.__setitem__((6, 2), mod.HIGHLIGHT_MARKER),
            "highlight trace column 2",
            id="extra-highlight",
        ),
        pytest.param(
            lambda trace, mod: trace.__setitem__((5, 2), 0),
            "highlight trace column 2",
            id="missing-highlight",
        ),
        pytest.param(
            lambda trace, mod: trace.__setitem__((0, 3), mod.END_MARKER),
            "end trace column 3",
            id="extra-end",
        ),
        pytest.param(
            lambda trace, mod: trace.__setitem__((8, 3), 0),
            "end trace column 3",
            id="missing-end",
        ),
        pytest.param(
            lambda trace, mod: trace.__setitem__((5, 4), 0),
            "edge trace column 4",
            id="missing-edge",
        ),
        pytest.param(
            lambda trace, mod: trace.__setitem__((0, 5), mod.FIRST_HIGHLIGHT_MARKER),
            "first-highlight trace column 5",
            id="extra-first-highlight",
        ),
    ],
)
def test_conditional_tma_trace_verifier_rejects_extra_or_missing_markers(
    mutate,
    message,
):
    mod = _conditional_copy_module()
    rows = 9
    highlight_row = 5
    trace = _valid_trace(rows, highlight_row)
    mutate(trace, mod)

    with pytest.raises(RuntimeError, match=message):
        mod._verify_trace_markers(
            trace,
            num_rows=rows,
            highlight_row=highlight_row,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("highlight_row", [0, 128, 255])
def test_copy_tma_conditional_on_device(highlight_row):
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("requires an SM100-class GPU")
    _conditional_copy_module().run_tma_copy_conditional_kernel_prim(
        (256, 512), highlight_row, verbose=False
    )
