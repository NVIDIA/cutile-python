# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the while-exit TMA copy tutorial.

Covers trace-marker checking, the schedule-only sequence printer
(``--verify-num-iters`` / ``--inject-nested-break`` /
``--inject-unconditional-break``), host validation of the full program, and the
end-to-end copy on an SM100-class GPU.
"""

import importlib
import io
from contextlib import redirect_stdout

import pytest
import torch

from task_scheduling_test_requirements import require_task_scheduling_dependencies


def _while_exit_module():
    require_task_scheduling_dependencies()
    return importlib.import_module(
        "experimental.task_scheduling.tutorial.01_copy_basics.04_copy_tma_while_exit"
    )


def _valid_trace(rows: int) -> torch.Tensor:
    mod = _while_exit_module()
    trace = torch.zeros(rows, 3, dtype=torch.float16)
    trace[:, 0] = mod.ITERATION_MARKER
    trace[rows - 1, 1] = mod.EXIT_MARKER
    return trace


def _printed(**kwargs) -> str:
    buf = io.StringIO()
    with redirect_stdout(buf):
        _while_exit_module().print_while_assignment_sequences(**kwargs)
    return buf.getvalue()


def test_while_exit_trace_verifier_accepts_exact_markers():
    mod = _while_exit_module()
    rows = 9
    mod._verify_trace_markers(_valid_trace(rows), num_rows=rows)


@pytest.mark.parametrize(
    "mutate, message",
    [
        pytest.param(
            lambda trace, mod: trace.__setitem__((0, 0), 0),
            "iteration trace column 0",
            id="missing-iteration",
        ),
        pytest.param(
            lambda trace, mod: trace.__setitem__((8, 1), 0),
            "exit trace column 1",
            id="missing-exit",
        ),
        pytest.param(
            lambda trace, mod: trace.__setitem__((0, 1), mod.EXIT_MARKER),
            "exit trace column 1",
            id="extra-exit",
        ),
        pytest.param(
            lambda trace, mod: trace.__setitem__((8, 2), mod.AFTER_BREAK_MARKER),
            "after-break trace column 2",
            id="break-fallthrough",
        ),
    ],
)
def test_while_exit_trace_verifier_rejects_mutated_markers(mutate, message):
    mod = _while_exit_module()
    rows = 9
    trace = _valid_trace(rows)
    mutate(trace, mod)
    with pytest.raises(RuntimeError, match=message):
        mod._verify_trace_markers(trace, num_rows=rows)


def test_while_exit_sequence_printer_retry_then_break():
    text = _printed(verify_num_iters=2)
    assert "WhileTrueLoop(" in text
    assert "BreakLoop(" in text
    assert "False" in text and "True" in text
    assert " -> " in text


def test_while_exit_fire_count_filters_unreachable_pipeline_rounds():
    assert "3 modeled assignment sequence(s)" in _printed(verify_num_iters=(2, 3, 4))
    text = _printed(verify_num_iters=(2, 3, 4), verify_num_fires=1)
    assert "1 modeled assignment sequence(s)" in text
    assert "verify_num_fires=1" in text
    assert "=False -> " in text and "=True" in text


def test_while_exit_unreachable_fire_count_is_rejected():
    with pytest.raises(ValueError, match="verify_num_fires=3"):
        _printed(verify_num_iters=(2,), verify_num_fires=3)


def test_while_exit_nested_break_is_conjunction():
    assert " and " in _printed(verify_num_iters=2, inject_nested_break=True)


def test_while_exit_unconditional_break_with_n2_errors():
    text = _printed(verify_num_iters=2, inject_unconditional_break=True)
    assert "ScheduleError:" in text
    assert "unconditional" in text


def test_while_exit_program_validates_one_retry_plus_break():
    program = _while_exit_module().make_tma_copy_while_exit_program(verbose=False)
    load_task, store_task, padding_task = program.manager.user_tasks
    assert [r.name for r in store_task.src_resources] == ["smemResource"]
    assert [r.name for r in store_task.dst_resources] == [
        "outputGmemResource",
        "traceGmemResource",
    ]
    assert padding_task.resources == []
    # verify_num_fires=1 keeps only the history matching one producer round.
    ((scenario, result, _),) = program.manager._interleaving_results
    assert [list(round_.values()) for round_ in scenario.for_task(store_task)] == [
        [False],
        [True],
    ]
    assert result.is_safe


def test_copy_tma_while_exit_rejects_partial_column_tile():
    mod = _while_exit_module()
    with pytest.raises(ValueError, match="multiple of tile_size=128"):
        mod.run_tma_copy_while_exit_kernel_prim((256, 129))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("rows_cols", [(256, 512), (512, 256), (1, 128), (9, 384)])
def test_copy_tma_while_exit_on_device(rows_cols):
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("requires an SM100-class GPU")
    _while_exit_module().run_tma_copy_while_exit_kernel_prim(rows_cols, verbose=False)
