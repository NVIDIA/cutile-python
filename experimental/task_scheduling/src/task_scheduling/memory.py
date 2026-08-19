# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pure-Python shared-, tensor-, and barrier-memory accounting."""

from dataclasses import dataclass, field

from .resources import CooperativeGroup, MbarrierLayout


def _align_up(value: int, alignment: int) -> int:
    if alignment <= 0 or alignment & (alignment - 1):
        raise ValueError(f"alignment must be a positive power of two, got {alignment}")
    return (value + alignment - 1) & ~(alignment - 1)


@dataclass
class SmemAllocation:
    name: str
    size_bytes: int = 0
    alignment: int = 128
    dtype: object | None = None
    count: int = 1
    offset: int = field(init=False, default=0)
    space: str = field(init=False, default="smem")

    def __post_init__(self) -> None:
        _align_up(0, self.alignment)
        if self.dtype is not None and self.size_bytes == 0:
            width = getattr(self.dtype, "width", None)
            if width is None:
                raise TypeError("dtype must expose a bit width")
            self.size_bytes = self.count * width // 8
        if self.size_bytes < 0 or self.count < 1:
            raise ValueError("allocation size must be nonnegative and count positive")

    @property
    def size(self) -> int:
        return self.size_bytes


@dataclass
class TmemAllocation:
    name: str
    num_columns: int
    offset: int = field(init=False, default=0)
    space: str = field(init=False, default="tmem")

    def __post_init__(self) -> None:
        if self.num_columns <= 0:
            raise ValueError("TMEM columns must be positive")

    @property
    def size(self) -> int:
        return self.num_columns


@dataclass(frozen=True)
class ResourceContext:
    smem_base: object | None = None
    tmem_ptr_i32: object | None = None


class _LayoutAllocator:
    def __init__(self, capacity: int | None = None) -> None:
        self._allocations: list[object] = []
        self._alias_groups: list[list[list[object]]] = []
        self._aliased_ids: set[int] = set()
        self._layout_computed = False
        self._total = 0
        self.capacity = capacity

    def _size(self, allocation: object) -> int:
        raise NotImplementedError

    def _alignment(self, allocation: object) -> int:
        return 1

    def add(self, allocation: object) -> object:
        if self._layout_computed:
            raise RuntimeError("cannot add allocations after compute_layout()")
        self._allocations.append(allocation)
        return allocation

    def add_resource(self, resource: object) -> None:
        raise NotImplementedError

    def add_alias_group(self, phases: list[list[object]]) -> None:
        if len(phases) < 2 or any(not phase for phase in phases):
            raise ValueError("alias group must contain at least two nonempty phases")
        seen = set()
        for phase in phases:
            for allocation in phase:
                if id(allocation) in seen or id(allocation) in self._aliased_ids:
                    raise ValueError(
                        "an allocation may occur only once in alias groups"
                    )
                seen.add(id(allocation))
        self._aliased_ids.update(seen)
        self._alias_groups.append([list(phase) for phase in phases])

    def _phase_size(self, phase: list[object]) -> int:
        cursor = 0
        for allocation in phase:
            cursor = _align_up(cursor, self._alignment(allocation))
            cursor += self._size(allocation)
        return cursor

    def compute_layout(self) -> None:
        if self._layout_computed:
            raise RuntimeError("compute_layout() already called")
        blocks = []
        for allocation in self._allocations:
            if id(allocation) not in self._aliased_ids:
                blocks.append(
                    (
                        self._alignment(allocation),
                        self._size(allocation),
                        [[allocation]],
                    )
                )
        for phases in self._alias_groups:
            blocks.append(
                (
                    max(self._alignment(a) for phase in phases for a in phase),
                    max(self._phase_size(phase) for phase in phases),
                    phases,
                )
            )
        blocks.sort(key=lambda item: item[0], reverse=True)
        cursor = 0
        for alignment, size, phases in blocks:
            base = _align_up(cursor, alignment)
            for phase in phases:
                phase_cursor = base
                for allocation in phase:
                    phase_cursor = _align_up(phase_cursor, self._alignment(allocation))
                    allocation.offset = phase_cursor
                    phase_cursor += self._size(allocation)
            cursor = base + size
        if self.capacity is not None and cursor > self.capacity:
            raise ValueError(
                f"memory budget exceeded: required {cursor}, capacity {self.capacity}"
            )
        self._total = cursor
        self._layout_computed = True

    @property
    def layout_computed(self) -> bool:
        return self._layout_computed

    @property
    def total(self) -> int:
        if not self._layout_computed:
            raise RuntimeError("call compute_layout() first")
        return self._total

    def _unit_label(self) -> str:
        return "units"

    def _report_tag(self) -> str:
        return "layout"

    def _report_title(self) -> str:
        return "Memory Usage Report"

    def _report_allocations(self) -> list[object]:
        allocations = list(self._allocations)
        seen = {id(allocation) for allocation in allocations}
        for group in self._alias_groups:
            for phase in group:
                for allocation in phase:
                    if id(allocation) not in seen:
                        seen.add(id(allocation))
                        allocations.append(allocation)
        return allocations

    def _report_total(self) -> int:
        return self._total

    def _report_extra_lines(self) -> list[str]:
        return []

    def print_usage_report(self) -> None:
        """Print the computed allocation layout and aliasing savings."""
        if not self._layout_computed:
            raise RuntimeError("call compute_layout() before print_usage_report()")

        allocations = sorted(
            self._report_allocations(), key=lambda allocation: allocation.offset
        )
        has_alignment = any(self._alignment(allocation) > 1 for allocation in allocations)
        print(f"\n[{self._report_tag()}] {self._report_title()}")
        if has_alignment:
            header = (
                f"  {'Name':<28} {'Size':>8}  {'Align':>5}  "
                f"{'Offset':>8}  {'End':>8}"
            )
        else:
            header = f"  {'Name':<28} {'Size':>8}  {'Offset':>8}  {'End':>8}"
        print(header)
        print(f"  {'─' * (len(header) - 2)}")
        for allocation in allocations:
            size = self._size(allocation)
            end = allocation.offset + size
            alias = " *" if id(allocation) in self._aliased_ids else ""
            if has_alignment:
                print(
                    f"  {allocation.name + alias:<28} {size:>8}  "
                    f"{self._alignment(allocation):>5}  "
                    f"{allocation.offset:>8}  {end:>8}"
                )
            else:
                print(
                    f"  {allocation.name + alias:<28} {size:>8}  "
                    f"{allocation.offset:>8}  {end:>8}"
                )

        if self._alias_groups:
            print(f"  {'─' * (len(header) - 2)}")
            for group_index, group in enumerate(self._alias_groups, 1):
                print(f"  Alias group {group_index}:")
                for phase_index, phase in enumerate(group, 1):
                    entries = ", ".join(
                        f"{allocation.name} ({self._size(allocation)} "
                        f"{self._unit_label()})"
                        for allocation in phase
                    )
                    print(f"    Phase {phase_index}: {entries}")

        report_total = self._report_total()
        savings = sum(self._size(allocation) for allocation in allocations)
        savings -= report_total
        print(f"  {'─' * (len(header) - 2)}")
        for line in self._report_extra_lines():
            print(f"  {line}")
        print(f"  Total:          {report_total:>8} {self._unit_label()}")
        if savings > 0:
            print(f"  Alias savings:  {savings:>8} {self._unit_label()}")
        print()


class SmemAllocator(_LayoutAllocator):
    def __init__(self, default_add_barriers: bool = True, capacity: int | None = None):
        super().__init__(capacity)
        self.default_add_barriers = default_add_barriers
        self._barrier_resources: list[object] = []
        self._barrier_allocations: list[SmemAllocation] = []

    def _size(self, allocation: SmemAllocation) -> int:
        return allocation.size_bytes

    def _alignment(self, allocation: SmemAllocation) -> int:
        return allocation.alignment

    def add_resource(self, resource: object, add_barriers: bool | None = None) -> None:
        for allocation in resource.get_smem_requirements():
            self.add(allocation)
        use_barriers = (
            self.default_add_barriers if add_barriers is None else add_barriers
        )
        config = getattr(resource, "pipeline_config", None)
        if use_barriers and config is not None and config.barrier_ptr is None:
            self._barrier_resources.append(resource)
            allocation = SmemAllocation(
                f"{resource.name}.barriers",
                size_bytes=16 * config.num_stages,
                alignment=8,
            )
            self._barrier_allocations.append(allocation)
            self.add(allocation)

    @property
    def barrier_resources(self) -> list[object]:
        return list(self._barrier_resources)

    @property
    def total_smem_bytes(self) -> int:
        return self.total

    @property
    def barrier_smem_bytes(self) -> int:
        return sum(allocation.size_bytes for allocation in self._barrier_allocations)

    @property
    def data_smem_bytes(self) -> int:
        return self.total - self.barrier_smem_bytes

    @property
    def data_alignment(self) -> int:
        """Maximum alignment required by the data-only SMEM arena."""
        if not self.layout_computed:
            raise RuntimeError("call compute_layout() first")
        allocations = self._report_allocations()
        return max(
            (self._alignment(allocation) for allocation in allocations),
            default=128,
        )

    @property
    def allocation_offsets(self) -> tuple[tuple[str, int], ...]:
        """Return device-visible data-allocation names and byte offsets."""
        if not self.layout_computed:
            raise RuntimeError("call compute_layout() first")
        allocations = self._report_allocations()
        names = [allocation.name for allocation in allocations]
        if len(names) != len(set(names)):
            raise ValueError("SMEM allocation names must be unique")
        return tuple(
            (allocation.name, allocation.offset)
            for allocation in sorted(allocations, key=lambda item: item.offset)
        )

    def _unit_label(self) -> str:
        return "B"

    def _report_tag(self) -> str:
        return "smem-layout"

    def _report_title(self) -> str:
        return "SMEM Usage Report"

    def _report_allocations(self) -> list[object]:
        barrier_ids = {id(allocation) for allocation in self._barrier_allocations}
        return [
            allocation
            for allocation in super()._report_allocations()
            if id(allocation) not in barrier_ids
        ]

    def _report_total(self) -> int:
        return self.data_smem_bytes

    def _report_extra_lines(self) -> list[str]:
        lines = [f"Data SMEM:        {self.data_smem_bytes:>8} B"]
        if self.barrier_smem_bytes:
            lines.append(f"Barrier SMEM:     {self.barrier_smem_bytes:>8} B")
            lines.append(
                f"Total SMEM:       {self.total_smem_bytes:>8} B  "
                "(data + barriers)"
            )
        return lines


class TmemAllocator(_LayoutAllocator):
    def _size(self, allocation: TmemAllocation) -> int:
        return allocation.num_columns

    def add_resource(self, resource: object) -> None:
        for allocation in resource.get_tmem_requirements():
            self.add(allocation)

    @property
    def total_tmem_columns(self) -> int:
        return self.total

    def _unit_label(self) -> str:
        return "cols"

    def _report_tag(self) -> str:
        return "tmem-layout"

    def _report_title(self) -> str:
        return "TMEM Usage Report"


@dataclass
class BarrierAllocation:
    name: str
    num_barriers: int
    arrive_count: int
    mbarrier_layout: MbarrierLayout = MbarrierLayout.V0
    offset: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        if self.num_barriers <= 0 or self.arrive_count <= 0:
            raise ValueError("barrier count and arrival count must be positive")

    @classmethod
    def from_group(
        cls,
        name: str,
        num_barriers: int,
        group: CooperativeGroup,
        mbarrier_layout: MbarrierLayout = MbarrierLayout.V0,
    ) -> "BarrierAllocation":
        return cls(name, num_barriers, group.size, mbarrier_layout)


class BarrierAllocator:
    """Coalesce mbarrier runs, separating hardware layout regions."""

    def __init__(self) -> None:
        self._allocations: dict[str, BarrierAllocation] = {}
        self._resources: list[object] = []
        self._computed = False
        self._padded = 0

    def add(self, allocation: BarrierAllocation) -> BarrierAllocation:
        if self._computed:
            raise RuntimeError("cannot add after compute_layout()")
        if allocation.name in self._allocations:
            raise ValueError(f"duplicate barrier name {allocation.name!r}")
        self._allocations[allocation.name] = allocation
        return allocation

    def add_producer_consumer(
        self,
        name: str,
        num_stages: int,
        producer_group: CooperativeGroup,
        consumer_group: CooperativeGroup,
    ) -> tuple[BarrierAllocation, BarrierAllocation]:
        return (
            self.add(
                BarrierAllocation.from_group(f"{name}.full", num_stages, producer_group)
            ),
            self.add(
                BarrierAllocation.from_group(
                    f"{name}.empty", num_stages, consumer_group
                )
            ),
        )

    def add_resource(self, resource: object) -> None:
        if any(item is resource for item in self._resources):
            return
        config = getattr(resource, "pipeline_config", None)
        if config is None:
            raise ValueError(f"resource {resource.name!r} has no pipeline_config")
        self.add_producer_consumer(
            resource.name,
            config.num_stages,
            config.producer_group,
            config.consumer_group,
        )
        self._resources.append(resource)

    def compute_layout(self) -> None:
        if self._computed:
            raise RuntimeError("compute_layout() already called")
        cursor = 0
        for layout in MbarrierLayout:
            allocations = [
                a for a in self._allocations.values() if a.mbarrier_layout is layout
            ]
            if not allocations:
                continue
            cursor = _align_up(cursor, 32)
            # Barriers with the same arrival count share one vectorized
            # initialization run, independent of resource registration order.
            by_arrive_count: dict[int, list[BarrierAllocation]] = {}
            for allocation in allocations:
                by_arrive_count.setdefault(allocation.arrive_count, []).append(
                    allocation
                )
            for arrive_count in sorted(by_arrive_count):
                for allocation in by_arrive_count[arrive_count]:
                    allocation.offset = cursor
                    cursor += allocation.num_barriers
            cursor = _align_up(cursor, 32)
        self._padded = cursor
        self._computed = True

    @property
    def layout_computed(self) -> bool:
        return self._computed

    @property
    def padded_size(self) -> int:
        """Padded arena size in mbarrier elements."""
        if not self._computed:
            raise RuntimeError("call compute_layout() first")
        return self._padded

    @property
    def padded_size_bytes(self) -> int:
        return self.padded_size * 8

    @property
    def used_size(self) -> int:
        """Highest occupied mbarrier offset, excluding layout padding."""
        if not self._computed:
            raise RuntimeError("call compute_layout() first")
        return max(
            (
                allocation.offset + allocation.num_barriers
                for allocation in self._allocations.values()
            ),
            default=0,
        )

    @property
    def initialization_runs(self) -> tuple[tuple[int, int, int], ...]:
        """Return contiguous ``(begin, end, arrive_count)`` initialization runs."""
        if not self._computed:
            raise RuntimeError("call compute_layout() first")
        runs: list[tuple[int, int, int, MbarrierLayout]] = []
        for allocation in sorted(
            self._allocations.values(), key=lambda item: item.offset
        ):
            begin = allocation.offset
            end = begin + allocation.num_barriers
            if (
                runs
                and runs[-1][1] == begin
                and runs[-1][2] == allocation.arrive_count
                and runs[-1][3] is allocation.mbarrier_layout
            ):
                previous = runs[-1]
                runs[-1] = (previous[0], end, previous[2], previous[3])
            else:
                runs.append(
                    (begin, end, allocation.arrive_count, allocation.mbarrier_layout)
                )
        return tuple((begin, end, count) for begin, end, count, _ in runs)

    @property
    def allocation_offsets(self) -> tuple[tuple[str, int], ...]:
        """Return device-visible barrier names and mbarrier offsets."""
        if not self._computed:
            raise RuntimeError("call compute_layout() first")
        return tuple(
            (allocation.name, allocation.offset)
            for allocation in sorted(
                self._allocations.values(), key=lambda item: item.offset
            )
        )

    def offset_of(self, name: str) -> int:
        if not self._computed:
            raise RuntimeError("call compute_layout() first")
        return self._allocations[name].offset

    def print_usage_report(self) -> None:
        """Print barrier runs and the padded allocation size."""
        if not self._computed:
            raise RuntimeError("call compute_layout() before print_usage_report()")

        print("\n[barrier-layout] Barrier Usage Report")
        header = (
            f"  {'Name':<28} {'Layout':>7} {'Count':>6}  "
            f"{'Arrive':>7}  {'Offset':>7}  {'End':>4}"
        )
        print(header)
        print(f"  {'─' * (len(header) - 2)}")
        for allocation in sorted(
            self._allocations.values(), key=lambda item: item.offset
        ):
            end = allocation.offset + allocation.num_barriers
            print(
                f"  {allocation.name:<28} "
                f"{allocation.mbarrier_layout.name:>7} "
                f"{allocation.num_barriers:>6}  "
                f"{allocation.arrive_count:>7}  "
                f"{allocation.offset:>7}  {end:>4}"
            )
        print(f"  {'─' * (len(header) - 2)}")
        for layout in MbarrierLayout:
            allocations = sorted(
                (
                    allocation
                    for allocation in self._allocations.values()
                    if allocation.mbarrier_layout is layout
                ),
                key=lambda item: item.offset,
            )
            if not allocations:
                continue

            region_start = allocations[0].offset
            region_used_end = max(
                allocation.offset + allocation.num_barriers
                for allocation in allocations
            )
            region_end = _align_up(region_used_end, 32)
            print(
                f"  Region {layout.name}: offsets {region_start}..{region_end} "
                f"({region_end - region_start} padded slots)"
            )

            bucket_start = region_start
            bucket_arrivals = allocations[0].arrive_count
            bucket_end = bucket_start
            for allocation in allocations:
                if allocation.arrive_count != bucket_arrivals:
                    print(
                        f"    [{bucket_start:>3},{bucket_end:>3}): "
                        f"arrive={bucket_arrivals:<6}"
                    )
                    bucket_start = allocation.offset
                    bucket_arrivals = allocation.arrive_count
                bucket_end = allocation.offset + allocation.num_barriers
            print(
                f"    [{bucket_start:>3},{bucket_end:>3}): "
                f"arrive={bucket_arrivals:<6}"
            )
        used = sum(
            allocation.num_barriers for allocation in self._allocations.values()
        )
        print(f"  Used barriers: {used:>3}")
        print(f"  Padded total:  {self._padded:>3}")
        print()
