# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Frozen CUDA Lang pipeline metadata and device initialization."""

from dataclasses import dataclass, replace

import cuda.lang as cl

from ._device import block_in_cluster_rank
from .enums import PipelineType
from .resources import MbarrierLayout, PipelineConfig


def _normalize_cta_layout(layout: object | None) -> tuple[int, int, int, int]:
    """Return a static compact VMNK cluster layout."""
    if layout is None:
        return (1, 1, 1, 1)
    if not isinstance(layout, (tuple, list)) or len(layout) != 4:
        raise NotImplementedError(
            "cuda.lang pipeline lowering requires cta_layout_vmnk to be a "
            "static four-element tuple or list"
        )
    result = tuple(layout)
    if not all(type(size) is int and size > 0 for size in result):
        raise ValueError("cta_layout_vmnk entries must be positive integers")
    if result[0] * result[1] * result[2] * result[3] > 16:
        raise ValueError("cta_layout_vmnk may contain at most 16 CTAs")
    return result


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
    CLC_FETCH_ASYNC = 5

    kind: int
    num_stages: int
    num_bytes: int
    producer_arrivals: int
    consumer_arrivals: int
    producer_elected: bool
    consumer_elected: bool
    producer_task_warp_leader: bool = False
    consumer_elected_per_warp: bool = False
    consumer_all_threads: bool = False
    cta_group_size: int = 1
    cluster_size: int = 1
    cta_layout_vmnk: tuple[int, int, int, int] = (1, 1, 1, 1)
    mcast_mode_mn: tuple[int, int] = (1, 1)
    producer_cta_leader: bool = False
    consumer_cta_leader: bool = False
    consumer_wait_cta_leader: bool = False
    advance_on_wait: bool = False
    tcgen05_fence_after_wait: bool = True
    full_barrier_offset: int = -1
    empty_barrier_offset: int = -1

    @classmethod
    def from_config(
        cls,
        config: PipelineConfig,
        *,
        consumer_elected_per_warp: bool = False,
        consumer_all_threads: bool = False,
    ) -> "DevicePipelineBinding":
        require_device_support(config)
        cta_layout_vmnk = _normalize_cta_layout(config.cta_layout_vmnk)
        consumer_wait_signaling = (
            config.consumer_wait_signaling_threads
            if config.consumer_wait_signaling_threads is not None
            else config.consumer_signaling_threads
        )
        kinds = {
            PipelineType.AsyncAsync: cls.ASYNC_ASYNC,
            PipelineType.TmaAsync: cls.TMA_ASYNC,
            PipelineType.TmaUmma: cls.TMA_UMMA,
            PipelineType.UmmaAsync: cls.UMMA_ASYNC,
            PipelineType.AsyncUmma: cls.ASYNC_UMMA,
            PipelineType.ClcFetchAsync: cls.CLC_FETCH_ASYNC,
        }
        return cls(
            kind=kinds[config.pipeline_type],
            num_stages=config.num_stages,
            num_bytes=config.num_bytes,
            producer_arrivals=config.producer_group.size,
            consumer_arrivals=config.consumer_group.size,
            producer_elected=config.producer_group.size == 1,
            consumer_elected=config.consumer_group.size == 1,
            producer_task_warp_leader=(
                config.producer_signaling_threads.has_task_warp_leader()
            ),
            consumer_elected_per_warp=consumer_elected_per_warp,
            consumer_all_threads=consumer_all_threads,
            cta_group_size=cta_layout_vmnk[0],
            cluster_size=(
                cta_layout_vmnk[0]
                * cta_layout_vmnk[1]
                * cta_layout_vmnk[2]
                * cta_layout_vmnk[3]
            ),
            cta_layout_vmnk=cta_layout_vmnk,
            mcast_mode_mn=config.mcast_mode_mn,
            producer_cta_leader=config.producer_signaling_threads.has_cta_leader(),
            consumer_cta_leader=config.consumer_signaling_threads.has_cta_leader(),
            consumer_wait_cta_leader=consumer_wait_signaling.has_cta_leader(),
            advance_on_wait=config.advance_on_wait,
            tcgen05_fence_after_wait=config.tcgen05_fence_after_wait,
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
        return barrier_arena.pointer(
            self.full_barrier_offset + index
        )

    def empty_barrier(self, barrier_arena, index):
        """Return the empty barrier for one stage in the shared arena."""
        return barrier_arena.pointer(
            self.empty_barrier_offset + index
        )

    @property
    def uses_cluster(self) -> bool:
        return self.cluster_size > 1

    @property
    def uses_two_cta_group(self) -> bool:
        return self.cta_group_size == 2

    def is_leader_cta(self):
        """Select the leading CTA of the current tcgen05 V-group."""
        if self.cta_group_size == 1:
            return True
        return block_in_cluster_rank() % self.cta_group_size == 0

    def is_cluster_leader_cta(self):
        """Select the one CTA that owns a cluster-wide operation."""
        if not self.uses_cluster:
            return True
        return block_in_cluster_rank() == 0

    def v_group_leader_rank(self):
        """Return the leading cluster rank of the current VMNK V-group."""
        return (
            block_in_cluster_rank() // self.cta_group_size
        ) * self.cta_group_size

    def _rank_coordinates(self, rank):
        """Map a compact linear cluster rank to its VMNK coordinates."""
        v_size, m_size, n_size, _ = self.cta_layout_vmnk
        return (
            rank % v_size,
            (rank // v_size) % m_size,
            (rank // (v_size * m_size)) % n_size,
            rank // (v_size * m_size * n_size),
        )

    def cta_group_multicast_mask(self):
        """Return the mask for the current one- or two-CTA tcgen05 V-group."""
        if self.cluster_size == self.cta_group_size:
            return cl.int16((1 << self.cta_group_size) - 1)
        if not self.uses_two_cta_group:
            return cl.int16(1 << block_in_cluster_rank())
        leader_rank = self.v_group_leader_rank()
        return cl.int16(((1 << self.cta_group_size) - 1) << leader_rank)

    def tma_umma_multicast_mask(self):
        """Return VMNK multicast destinations for an UMMA release."""
        if self.cluster_size == self.cta_group_size:
            return self.cta_group_multicast_mask()
        current_rank = block_in_cluster_rank()
        _, current_m, current_n, current_k = self._rank_coordinates(current_rank)
        multicast_m, multicast_n = self.mcast_mode_mn
        mask = cl.int16(0)
        for destination_rank in cl.static_iter(range(self.cluster_size)):
            _, destination_m, destination_n, destination_k = (
                self._rank_coordinates(destination_rank)
            )
            selected = destination_k == current_k and (
                (multicast_m and destination_m == current_m)
                or (multicast_n and destination_n == current_n)
            )
            mask = mask | cl.int16(selected) * cl.int16(1 << destination_rank)
        return mask

    def tma_async_consumer_destination_rank(self):
        """Return the destination selected by this lane's multicast release."""
        return cl.lane_index() % self.cluster_size

    def tma_async_consumer_selected(self):
        """Implement PipelineTmaAsync's VMNK empty-arrive lane predicate."""
        lane = cl.lane_index()
        destination_rank = self.tma_async_consumer_destination_rank()
        current = self._rank_coordinates(block_in_cluster_rank())
        destination = self._rank_coordinates(destination_rank)
        multicast_m, multicast_n = self.mcast_mode_mn
        along_m = (
            destination[0] == current[0]
            and destination[1] == current[1]
            and destination[3] == current[3]
        )
        along_n = (
            destination[0] == current[0]
            and destination[2] == current[2]
            and destination[3] == current[3]
        )
        return lane < self.cluster_size and (
            (multicast_m and along_m) or (multicast_n and along_n)
        )

    def work_barrier(self, barrier_arena, index):
        """Return the full barrier consumed by device work for this stage.

        CTA_2 TMA instructions clear the peer bit of the completion-barrier
        address. Give every producer CTA the corresponding leader address so
        all transactions retire into the one barrier armed by the leader.
        """
        barrier = self.full_barrier(barrier_arena, index)
        if self.kind == self.TMA_UMMA and self.uses_two_cta_group:
            return cl.map_shared_to_block_pair_leader(barrier)
        return barrier

    def consumer_release_barrier(self, barrier_arena, index):
        """Route async consumer arrivals to the CTA-group leader barrier."""
        barrier = self.empty_barrier(barrier_arena, index)
        if self.kind == self.CLC_FETCH_ASYNC and self.uses_cluster:
            return cl.map_shared_to_cluster(barrier, 0)
        if self.kind == self.UMMA_ASYNC and self.uses_two_cta_group:
            return cl.map_shared_to_cluster(barrier, self.v_group_leader_rank())
        if (
            self.kind == self.TMA_ASYNC
            and self.uses_cluster
            and not self.consumer_all_threads
        ):
            return cl.map_shared_to_cluster(
                barrier, self.tma_async_consumer_destination_rank()
            )
        return barrier

    def producer_commit_barrier(self, barrier_arena, index):
        """Route AsyncUmma producer arrivals to the current V-group leader."""
        barrier = self.full_barrier(barrier_arena, index)
        if self.kind == self.ASYNC_UMMA and self.uses_two_cta_group:
            return cl.map_shared_to_cluster(barrier, self.v_group_leader_rank())
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
        PipelineType.ClcFetchAsync,
    ):
        raise NotImplementedError(
            f"cuda.lang lowering for {config.pipeline_type.value} is not implemented; "
            "its host metadata remains analyzable"
        )
    unsupported = []
    if config.has_interleaved_stride:
        unsupported.append("interleave strides")
    if config.advance_on_acquire:
        unsupported.append("producer split state advancement")
    cta_layout_vmnk = _normalize_cta_layout(config.cta_layout_vmnk)
    if (
        config.pipeline_type
        in (PipelineType.TmaUmma, PipelineType.UmmaAsync, PipelineType.AsyncUmma)
        and cta_layout_vmnk[0] not in (1, 2)
    ):
        unsupported.append("tcgen05 V-group sizes other than one or two CTAs")
    if config.mcast_mode_mn not in ((1, 0), (0, 1), (1, 1)):
        raise ValueError(
            "mcast_mode_mn must be (1, 0), (0, 1), or (1, 1)"
        )
    if config.defer_init or config.barrier_ptr is not None:
        unsupported.append("deferred/external barrier storage")
    if config.num_bytes_per_warp_per_cta is not None and config.pipeline_type not in (
        PipelineType.TmaAsync,
        PipelineType.TmaUmma,
    ):
        unsupported.append("per-warp transaction bytes")
    if (
        config.producer_signaling_threads.has_task_warp_leader()
        and config.pipeline_type
        not in (PipelineType.TmaAsync, PipelineType.TmaUmma)
    ):
        unsupported.append("producer TaskWarpLeader on a non-TMA pipeline")
    if config.consumer_signaling_threads.has_task_warp_leader():
        unsupported.append("consumer TaskWarpLeader")
    if (
        config.consumer_wait_signaling_threads is not None
        and config.consumer_wait_signaling_threads.has_task_warp_leader()
    ):
        unsupported.append("consumer-wait TaskWarpLeader")
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
