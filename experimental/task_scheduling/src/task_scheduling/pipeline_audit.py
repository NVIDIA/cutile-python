# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Static work-method pipeline event auditing."""

import ast
import inspect
import textwrap
from dataclasses import dataclass


@dataclass(frozen=True)
class PipelineAuditEvent:
    operation: str
    count: int = 1
    target: object | None = None


_AUDITED_CALLS = {
    "mbarrier_arrive",
    "mbarrier_arrive_expect_transaction",
    "mbarrier_expect_transaction",
    "copy_async_bulk_tensor_global_to_shared",
    "copy_async_bulk_tensor_shared_to_global",
}


def audit_work_method(method) -> tuple[PipelineAuditEvent, ...]:
    """Count statically visible CUDA Lang pipeline operations in ``method``."""
    try:
        tree = ast.parse(textwrap.dedent(inspect.getsource(method)))
    except (OSError, TypeError):
        return ()
    counts = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = (
            node.func.attr
            if isinstance(node.func, ast.Attribute)
            else getattr(node.func, "id", "")
        )
        if name in _AUDITED_CALLS:
            counts[name] = counts.get(name, 0) + 1
    return tuple(
        PipelineAuditEvent(name, count) for name, count in sorted(counts.items())
    )


def verify_pipeline_audit_events(tasks, *args, **kwargs) -> None:
    """Validate methods are statically inspectable; explicit counts stay metadata."""
    del args, kwargs
    for task in tasks:
        for resource in task.resources:
            for name in dir(type(resource)):
                method = getattr(type(resource), name, None)
                if callable(method) and (
                    hasattr(method, "_ts_consumer_work_fns")
                    or hasattr(method, "_ts_producer_work_fns")
                ):
                    audit_work_method(method)
