# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Frozen CUDA Lang pipeline metadata and device initialization."""

from dataclasses import dataclass, replace

import cuda.lang as cl

from .enums import PipelineType, SignalingThreads
from .resources import MbarrierLayout, PipelineConfig


@dataclass(frozen=True)
class PipelineState:
    index: object = 0
    phase: object = 0
    status: object = False

    def advance(self, stages: int) -> "PipelineState":
        next_index = self.index + 1
        wrapped = cl.int32(next_index == stages)
        return PipelineState(
            next_index - wrapped * stages,
            self.phase ^ wrapped,
            False,
        )

    def with_status(self, status: object) -> "PipelineState":
        return replace(self, status=status)


@dataclass(frozen=True)
class DevicePipelineBinding:
    """Compile-time pipeline metadata bound to manager-owned barrier storage.

    Barrier offsets are assigned internally by :class:`TaskManager` when the
    complete resource set is frozen. Kernel authors never address the
    manager-owned barrier arena directly.
    """

    ASYNC_ASYNC = 0
    TMA_ASYNC = 1
    TMA_UMMA = 2
    UMMA_ASYNC = 3
    ASYNC_UMMA = 4

    kind: int
    num_stages: int
    num_bytes: int
    producer_arrivals: int
    consumer_arrivals: int
    producer_elected: bool
    consumer_elected: bool
    cta_group_size: int = 1
    producer_cta_leader: bool = False
    consumer_cta_leader: bool = False
    full_barrier_offset: int = -1
    empty_barrier_offset: int = -1

    @classmethod
    def from_config(
        cls,
        config: PipelineConfig,
    ) -> "DevicePipelineBinding":
        require_device_support(config)
        kinds = {
            PipelineType.AsyncAsync: cls.ASYNC_ASYNC,
            PipelineType.TmaAsync: cls.TMA_ASYNC,
            PipelineType.TmaUmma: cls.TMA_UMMA,
            PipelineType.UmmaAsync: cls.UMMA_ASYNC,
            PipelineType.AsyncUmma: cls.ASYNC_UMMA,
        }
        return cls(
            kind=kinds[config.pipeline_type],
            num_stages=config.num_stages,
            num_bytes=config.num_bytes,
            producer_arrivals=config.producer_group.size,
            consumer_arrivals=config.consumer_group.size,
            producer_elected=config.producer_group.size == 1,
            consumer_elected=config.consumer_group.size == 1,
            cta_group_size=(
                1
                if config.cta_layout_vmnk in (None, (1, 1, 1, 1))
                else config.cta_layout_vmnk[0]
            ),
            producer_cta_leader=config.producer_signaling_threads.has_cta_leader(),
            consumer_cta_leader=config.consumer_signaling_threads.has_cta_leader(),
        )

    @property
    def has_barrier_offsets(self) -> bool:
        return self.full_barrier_offset >= 0 and self.empty_barrier_offset >= 0

    def at_offsets(
        self, full_offset: int, empty_offset: int
    ) -> "DevicePipelineBinding":
        """Return this immutable binding with offsets into one barrier arena."""
        if type(full_offset) is not int or full_offset < 0:
            raise ValueError("full barrier offset must be a nonnegative integer")
        if type(empty_offset) is not int or empty_offset < 0:
            raise ValueError("empty barrier offset must be a nonnegative integer")
        return replace(
            self,
            full_barrier_offset=full_offset,
            empty_barrier_offset=empty_offset,
        )

    def full_barrier(self, barrier_arena, index):
        """Return the full barrier for one stage in the shared arena."""
        return barrier_arena.get_element_pointer(
            self.full_barrier_offset + index
        )

    def empty_barrier(self, barrier_arena, index):
        """Return the empty barrier for one stage in the shared arena."""
        return barrier_arena.get_element_pointer(
            self.empty_barrier_offset + index
        )

    @property
    def uses_cluster(self) -> bool:
        return self.cta_group_size > 1

    def is_leader_cta(self):
        """Select the leading CTA of the current two-CTA tcgen05 group."""
        if not self.uses_cluster:
            return True
        return cl.block_in_cluster_index(0) % self.cta_group_size == 0

    def multicast_mask(self):
        """Return the mask covering the current tcgen05 CTA group."""
        return (1 << self.cta_group_size) - 1

    def work_barrier(self, barrier_arena, index):
        """Return the full barrier consumed by device work for this stage.

        CTA_2 TMA instructions clear the peer bit of the completion-barrier
        address. Give every producer CTA the corresponding leader address so
        all transactions retire into the one barrier armed by the leader.
        """
        barrier = self.full_barrier(barrier_arena, index)
        if self.kind == self.TMA_UMMA and self.uses_cluster:
            return cl.map_shared_to_leader_block(barrier)
        return barrier

    def consumer_release_barrier(self, barrier_arena, index):
        """Route async consumer arrivals to the CTA-group leader barrier."""
        barrier = self.empty_barrier(barrier_arena, index)
        if self.kind == self.UMMA_ASYNC and self.uses_cluster:
            leader_rank = (
                cl.block_in_cluster_index(0) // self.cta_group_size
            ) * self.cta_group_size
            return cl.map_shared_to_cluster(barrier, leader_rank)
        return barrier

    def initialize(self, barrier_arena) -> None:
        """Initialize this binding's barriers in a manager-owned arena."""
        for stage in cl.static_iter(range(self.num_stages)):
            cl.mbarrier_initialize(
                self.full_barrier(barrier_arena, stage),
                self.producer_arrivals,
            )
            cl.mbarrier_initialize(
                self.empty_barrier(barrier_arena, stage),
                self.consumer_arrivals,
            )


def require_device_support(config: PipelineConfig) -> None:
    """Fail at the narrow lowering boundary for metadata-only pipelines."""
    if config.pipeline_type not in (
        PipelineType.AsyncAsync,
        PipelineType.TmaAsync,
        PipelineType.TmaUmma,
        PipelineType.UmmaAsync,
        PipelineType.AsyncUmma,
    ):
        raise NotImplementedError(
            f"cuda.lang lowering for {config.pipeline_type.value} is not implemented; "
            "its host metadata remains analyzable"
        )
    unsupported = []
    if config.has_interleaved_stride:
        unsupported.append("interleave strides")
    if config.advance_on_wait or config.advance_on_acquire:
        unsupported.append("split state advancement")
    cluster_layout = config.cta_layout_vmnk not in (None, (1, 1, 1, 1))
    if cluster_layout and config.cta_layout_vmnk != (2, 1, 1, 1):
        unsupported.append("cluster layouts other than (2, 1, 1, 1)")
    if cluster_layout and config.pipeline_type not in (
        PipelineType.TmaUmma,
        PipelineType.UmmaAsync,
        PipelineType.AsyncUmma,
    ):
        unsupported.append(
            "cluster lowering for pipeline types other than "
            "TmaUmma/UmmaAsync/AsyncUmma"
        )
    if config.mcast_mode_mn != (1, 1):
        unsupported.append("non-default multicast modes")
    if config.defer_init or config.barrier_ptr is not None:
        unsupported.append("deferred/external barrier storage")
    if (
        config.storage_offset_full is not None
        or config.storage_offset_empty is not None
    ):
        unsupported.append("barrier storage offsets")
    if config.num_bytes_per_warp_per_cta is not None and not (
        cluster_layout and config.pipeline_type is PipelineType.TmaUmma
    ):
        unsupported.append("per-warp transaction bytes")
    supported_signaling = (
        config.consumer_wait_signaling_threads is None
        and (
            not cluster_layout
            and config.producer_signaling_threads is SignalingThreads.All
            and config.consumer_signaling_threads is SignalingThreads.All
            or cluster_layout
            and config.pipeline_type is PipelineType.TmaUmma
            and config.producer_signaling_threads is SignalingThreads.All
            and config.consumer_signaling_threads
            in (SignalingThreads.All, SignalingThreads.CtaLeader)
            or cluster_layout
            and config.pipeline_type is PipelineType.UmmaAsync
            and config.producer_signaling_threads
            in (SignalingThreads.All, SignalingThreads.CtaLeader)
            and config.consumer_signaling_threads is SignalingThreads.All
            or cluster_layout
            and config.pipeline_type is PipelineType.AsyncUmma
            and config.producer_signaling_threads
            in (SignalingThreads.All, SignalingThreads.CtaLeader)
            and config.consumer_signaling_threads
            in (SignalingThreads.All, SignalingThreads.CtaLeader)
        )
    )
    if not supported_signaling:
        unsupported.append("custom signaling threads")
    if (
        config.full_mbarrier_layout is not MbarrierLayout.V0
        or config.empty_mbarrier_layout is not MbarrierLayout.V0
    ):
        unsupported.append("non-V0 mbarrier layouts")
    if unsupported:
        raise NotImplementedError(
            "cuda.lang generic pipeline lowering does not implement "
            + ", ".join(unsupported)
        )
