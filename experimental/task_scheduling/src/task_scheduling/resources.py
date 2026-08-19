# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pure-Python resource, pipeline, and work-method metadata."""

import functools
import inspect
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable

from .enums import PipelineType, ScheduleStage, SignalingThreads, WorkAttr


_CONSUMER_WORK_FNS_ATTR = "_ts_consumer_work_fns"
_PRODUCER_WORK_FNS_ATTR = "_ts_producer_work_fns"


class PipelineOp(Enum):
    AsyncThread = "AsyncThread"
    AsyncLoad = "AsyncLoad"
    TmaLoad = "TmaLoad"
    TmaStore = "TmaStore"
    Umma = "Umma"


class MbarrierLayout(Enum):
    V0 = "V0"
    V1 = "V1"


@dataclass(frozen=True)
class CooperativeGroup:
    """Host substitute for a pipeline cooperative group."""

    size: int

    def __post_init__(self) -> None:
        if type(self.size) is not int or self.size <= 0:
            raise ValueError("cooperative-group size must be a positive integer")


@dataclass(frozen=True)
class PipelineConfig:
    """Immutable metadata describing a staged producer/consumer pipeline."""

    num_stages: int
    num_bytes: int
    producer_group: CooperativeGroup
    consumer_group: CooperativeGroup
    pipeline_type: PipelineType
    barrier_ptr: object | None = None
    cta_layout_vmnk: object | None = None
    producer_signaling_threads: SignalingThreads = SignalingThreads.All
    consumer_signaling_threads: SignalingThreads = SignalingThreads.All
    consumer_wait_signaling_threads: SignalingThreads | None = None
    advance_on_wait: bool = False
    advance_on_acquire: bool = False
    num_bytes_per_warp_per_cta: int | None = None
    mcast_mode_mn: tuple[int, int] = (1, 1)
    interleave_stride: int | tuple[int, int, int, int] = 1
    async_producer_op: PipelineOp = PipelineOp.AsyncThread
    umma_consumer_producer_op: PipelineOp = PipelineOp.AsyncThread
    defer_init: bool = False
    storage_offset_full: int | None = None
    storage_offset_empty: int | None = None
    full_mbarrier_layout: MbarrierLayout = MbarrierLayout.V0
    empty_mbarrier_layout: MbarrierLayout = MbarrierLayout.V0

    def __post_init__(self) -> None:
        if type(self.num_stages) is not int or self.num_stages <= 0:
            raise ValueError("num_stages must be a positive integer")
        if type(self.num_bytes) is not int or self.num_bytes < 0:
            raise ValueError("num_bytes must be a nonnegative integer")
        SignalingThreads.validate(
            self.producer_signaling_threads, "producer_signaling_threads"
        )
        SignalingThreads.validate(
            self.consumer_signaling_threads, "consumer_signaling_threads"
        )
        if self.consumer_wait_signaling_threads is not None:
            SignalingThreads.validate(
                self.consumer_wait_signaling_threads,
                "consumer_wait_signaling_threads",
            )
        strides = self.interleave_strides
        for stride in strides:
            if self.num_stages % stride:
                raise ValueError(
                    f"interleave_stride={stride} must evenly divide "
                    f"num_stages={self.num_stages}"
                )
        if strides[0] != strides[1] and not self.advance_on_acquire:
            raise ValueError("split producer strides require advance_on_acquire=True")
        if strides[2] != strides[3] and not self.advance_on_wait:
            raise ValueError("split consumer strides require advance_on_wait=True")
        if (self.storage_offset_full is None) != (self.storage_offset_empty is None):
            raise ValueError("storage_offset_{full,empty} must both be set or neither")
        if self.defer_init and self.barrier_ptr is None:
            raise ValueError("defer_init=True requires a pre-supplied barrier_ptr")

    @property
    def interleave_strides(self) -> tuple[int, int, int, int]:
        value = self.interleave_stride
        if isinstance(value, tuple):
            if len(value) != 4:
                raise ValueError("interleave_stride tuple must have four entries")
            strides = value
        else:
            strides = (value, value, value, value)
        if any(type(item) is not int or item < 1 for item in strides):
            raise ValueError("interleave strides must be positive integers")
        return strides

    @property
    def producer_acquire_interleave_stride(self) -> int:
        return self.interleave_strides[0]

    @property
    def producer_commit_interleave_stride(self) -> int:
        return self.interleave_strides[1]

    @property
    def consumer_wait_interleave_stride(self) -> int:
        return self.interleave_strides[2]

    @property
    def consumer_release_interleave_stride(self) -> int:
        return self.interleave_strides[3]

    @property
    def max_interleave_stride(self) -> int:
        return max(self.interleave_strides)

    @property
    def has_interleaved_stride(self) -> bool:
        return self.max_interleave_stride > 1

    @property
    def resolved_storage_offset_full(self) -> int:
        return self.storage_offset_full if self.storage_offset_full is not None else 0

    @property
    def resolved_storage_offset_empty(self) -> int:
        return (
            self.storage_offset_empty
            if self.storage_offset_empty is not None
            else self.num_stages
        )

    @property
    def supports_try_probe_ops(self) -> bool:
        return self.pipeline_type != PipelineType.ClcFetchAsync

    @classmethod
    def _create(
        cls,
        kind: PipelineType,
        num_stages: int,
        producer_group: CooperativeGroup,
        consumer_group: CooperativeGroup,
        *,
        num_bytes: int = 0,
        **kwargs: object,
    ) -> "PipelineConfig":
        return cls(
            num_stages=num_stages,
            num_bytes=num_bytes,
            producer_group=producer_group,
            consumer_group=consumer_group,
            pipeline_type=kind,
            **kwargs,
        )

    @classmethod
    def create_async_async_pipeline_cfg(
        cls, num_stages, producer_group, consumer_group, cta_layout_vmnk=None, **kwargs
    ):
        return cls._create(
            PipelineType.AsyncAsync,
            num_stages,
            producer_group,
            consumer_group,
            cta_layout_vmnk=cta_layout_vmnk,
            **kwargs,
        )

    @classmethod
    def create_tma_async_pipeline_cfg(
        cls, num_stages, num_bytes, producer_group, consumer_group, **kwargs
    ):
        return cls._create(
            PipelineType.TmaAsync,
            num_stages,
            producer_group,
            consumer_group,
            num_bytes=num_bytes,
            **kwargs,
        )

    @classmethod
    def create_tma_umma_pipeline_cfg(
        cls, num_stages, num_bytes, producer_group, consumer_group, **kwargs
    ):
        return cls._create(
            PipelineType.TmaUmma,
            num_stages,
            producer_group,
            consumer_group,
            num_bytes=num_bytes,
            **kwargs,
        )

    @classmethod
    def create_umma_async_pipeline_cfg(
        cls, num_stages, producer_group, consumer_group, **kwargs
    ):
        return cls._create(
            PipelineType.UmmaAsync, num_stages, producer_group, consumer_group, **kwargs
        )

    @classmethod
    def create_async_umma_pipeline_cfg(
        cls, num_stages, producer_group, consumer_group, **kwargs
    ):
        return cls._create(
            PipelineType.AsyncUmma, num_stages, producer_group, consumer_group, **kwargs
        )

    @classmethod
    def create_umma_umma_pipeline_cfg(
        cls, num_stages, producer_group, consumer_group, **kwargs
    ):
        return cls._create(
            PipelineType.UmmaUmma, num_stages, producer_group, consumer_group, **kwargs
        )

    @classmethod
    def create_clc_fetch_async_pipeline_cfg(
        cls, num_stages, producer_group, consumer_group, **kwargs
    ):
        return cls._create(
            PipelineType.ClcFetchAsync,
            num_stages,
            producer_group,
            consumer_group,
            **kwargs,
        )


@dataclass(frozen=True)
class StageInfo:
    """Current loop and pipeline stage passed to device work methods."""

    index: object = 0
    stage_idx: object | None = None
    phase: object = 0
    barrier: object | None = None
    count: object = 0
    loop_offset: object | None = None
    loop_start: object = 0
    loop_end: object = 0
    loop_step: object = 1
    label: object | None = None
    work_tile: object | None = None
    context: object | None = None


def _work_schema(
    fn: Callable[..., object], *, static: bool = False
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    params = list(inspect.signature(fn).parameters.values())
    required_prefix = 1 if static else 2
    if len(params) < required_prefix:
        signature = "stage_info" if static else "self and stage_info"
        raise TypeError(f"work method {fn.__name__} must accept {signature}")
    routed = []
    required = []
    for param in params[required_prefix:]:
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            raise TypeError("work methods cannot use *args or **kwargs")
        routed.append(param.name)
        if param.default is param.empty:
            required.append(param.name)
    return tuple(routed), tuple(required)


def _work_decorator(
    method: Callable[..., object] | None,
    *,
    producer: bool,
    work_attrs: WorkAttr,
    outputs: int = 0,
    static_args: tuple[str, ...] = (),
) -> Callable[..., object]:
    WorkAttr.validate(work_attrs, "work_attrs")
    if type(outputs) is not int or outputs < 0:
        raise TypeError("outputs must be a nonnegative int")

    def decorate(fn: Callable[..., object]) -> Callable[..., object]:
        if isinstance(fn, classmethod):
            raise TypeError("work methods cannot be class methods")
        is_static = isinstance(fn, staticmethod)
        work_fn = fn.__func__ if is_static else fn
        parameter_names, required = _work_schema(work_fn, static=is_static)
        if not isinstance(static_args, tuple) or not all(
            isinstance(name, str) for name in static_args
        ):
            raise TypeError("static_args must be a tuple of parameter names")
        if len(set(static_args)) != len(static_args):
            raise ValueError("static_args cannot contain duplicate names")
        unknown_static = set(static_args) - set(parameter_names)
        if unknown_static:
            raise ValueError(
                f"static_args contains unknown parameters {unknown_static}"
            )
        routed = tuple(name for name in parameter_names if name not in static_args)
        required = tuple(name for name in required if name not in static_args)

        if is_static:
            @functools.wraps(work_fn)
            def wrapper(stage_info, *args, **kwargs):
                return work_fn(stage_info, *args, **kwargs)

        else:
            @functools.wraps(work_fn)
            def wrapper(self, stage_info, *args, **kwargs):
                return work_fn(self, stage_info, *args, **kwargs)

        prefix = "producer" if producer else "consumer"
        registry_attr = _PRODUCER_WORK_FNS_ATTR if producer else _CONSUMER_WORK_FNS_ATTR
        setattr(wrapper, registry_attr, {work_fn.__name__: work_fn.__name__})
        setattr(wrapper, f"_{prefix}_routed_names", routed)
        setattr(wrapper, f"_{prefix}_routed_positional_names", routed)
        setattr(wrapper, f"_{prefix}_required_routed_names", required)
        setattr(wrapper, f"_{prefix}_static_names", static_args)
        setattr(wrapper, f"_{prefix}_parameter_names", parameter_names)
        setattr(wrapper, f"_{prefix}_work_attrs", work_attrs)
        setattr(
            wrapper,
            f"_{prefix}_work_stage",
            ScheduleStage.ProducerAuxWork
            if producer and work_attrs.is_auxiliary()
            else ScheduleStage.ConsumerAuxWork
            if work_attrs.is_auxiliary()
            else ScheduleStage.ProducerWork
            if producer
            else ScheduleStage.ConsumerWork,
        )
        setattr(wrapper, "_ts_route_output_count", outputs)
        return staticmethod(wrapper) if is_static else wrapper

    if method is not None:
        return decorate(method)
    return decorate


def consumer_work(
    method=None,
    *,
    work_attrs=WorkAttr.NONE,
    outputs=0,
    static_args=(),
):
    return _work_decorator(
        method,
        producer=False,
        work_attrs=work_attrs,
        outputs=outputs,
        static_args=static_args,
    )


def producer_work(
    method=None,
    *,
    work_attrs=WorkAttr.NONE,
    outputs=0,
    static_args=(),
):
    return _work_decorator(
        method,
        producer=True,
        work_attrs=work_attrs,
        outputs=outputs,
        static_args=static_args,
    )


def _collect_named_work_fns(cls: type, attr: str) -> dict[str, str]:
    result = {}
    for base in reversed(cls.__mro__):
        for value in vars(base).values():
            if isinstance(value, (staticmethod, classmethod)):
                value = value.__func__
            result.update(getattr(value, attr, {}))
    return result


def _get_static_work_fn(resource: object, label: str) -> object | None:
    """Return the directly decorated static work function for ``label``."""
    cls = type(resource)
    method_names = {
        **_collect_named_work_fns(cls, _CONSUMER_WORK_FNS_ATTR),
        **_collect_named_work_fns(cls, _PRODUCER_WORK_FNS_ATTR),
    }
    method_name = method_names.get(label)
    if method_name is None:
        return None
    descriptor = inspect.getattr_static(cls, method_name)
    if not isinstance(descriptor, staticmethod):
        return None
    return descriptor.__func__.__wrapped__


@dataclass(kw_only=True, eq=False)
class MemoryResource:
    """Identity-based host metadata for a schedulable memory resource."""

    name: str = ""
    is_barrier: bool = False
    pipeline_config: PipelineConfig | None = None
    smem_requirements: list[object] = field(default_factory=list)
    tmem_requirements: list[object] = field(default_factory=list)
    pipeline_group: object | None = field(default=None, init=False, repr=False)

    def __hash__(self) -> int:
        return id(self)

    @property
    def state_src(self) -> "MemoryResource":
        return self

    def get_smem_requirements(self) -> list[object]:
        return list(self.smem_requirements)

    def get_tmem_requirements(self) -> list[object]:
        return list(self.tmem_requirements)

    def physical_ranges(self) -> list[tuple[str, int, int]]:
        ranges = []
        for allocation in (*self.smem_requirements, *self.tmem_requirements):
            if getattr(allocation, "offset", None) is not None:
                ranges.append(
                    (
                        getattr(allocation, "space", "smem"),
                        allocation.offset,
                        allocation.offset + allocation.size,
                    )
                )
        return ranges

    def create_pipeline(self) -> object:
        if self.pipeline_config is None:
            return None
        # Keep the resource model independent of the device implementation at
        # import time; pipeline.py imports PipelineConfig from this module.
        from .pipeline import require_device_support

        require_device_support(self.pipeline_config)
        return self.pipeline_config


@dataclass(kw_only=True, eq=False)
class WorkQueue(MemoryResource):
    """Resource marker used by persistent work-tile loops."""

    @consumer_work(outputs=1)
    def get_and_advance_work_tile(self, stage_info):
        raise RuntimeError("WorkQueue methods are executed only in a device context")

    @producer_work
    def fetch_work_tile(self, stage_info):
        raise RuntimeError("WorkQueue methods are executed only in a device context")
