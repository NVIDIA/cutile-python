# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from time import perf_counter_ns
from types import MappingProxyType


@dataclass(frozen=True)
class CompilationTimings:
    phases_ns: Mapping[str, int]

    def __post_init__(self) -> None:
        object.__setattr__(self, "phases_ns", MappingProxyType(dict(self.phases_ns)))

    def seconds(self, phase: str) -> float:
        return self.phases_ns[phase] / 1_000_000_000

    def format_summary(self) -> str:
        lines = []
        for phase, elapsed_ns in self.phases_ns.items():
            elapsed_ms = elapsed_ns / 1_000_000
            lines.append(f"{phase:40} {elapsed_ms:10.3f} ms")
        return "\n".join(lines)


class CompilationTimer:
    def __init__(self, enabled: bool = True) -> None:
        self._enabled = enabled
        self._start_ns = perf_counter_ns() if enabled else 0
        self._phases_ns: dict[str, int] = {}

    @contextmanager
    def phase(self, name: str) -> Iterator[None]:
        if not self._enabled:
            yield
            return

        start_ns = perf_counter_ns()
        try:
            yield
        finally:
            self._record(name, perf_counter_ns() - start_ns)

    def finish(self) -> CompilationTimings:
        if not self._enabled:
            return CompilationTimings({})
        self._record("total", perf_counter_ns() - self._start_ns)
        return CompilationTimings(self._phases_ns)

    def add_phases(self, prefix: str, phases_ns: Mapping[str, int]) -> None:
        if not self._enabled:
            return
        for name, elapsed_ns in phases_ns.items():
            self._record(f"{prefix}.{name}", elapsed_ns)

    def _record(self, name: str, elapsed_ns: int) -> None:
        if name in self._phases_ns:
            raise RuntimeError(f"Compilation phase {name!r} was measured more than once")
        self._phases_ns[name] = elapsed_ns


__all__ = ("CompilationTimer", "CompilationTimings")
