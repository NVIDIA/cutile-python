# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Persistent work-tile schedulers shared by task-scheduled kernels."""

from dataclasses import dataclass, field, replace

import cuda.lang as cl

from ._device import block_in_cluster_rank
from .enums import TileSchedulerType
from .resources import WorkTileInfo


def _fast_divmod_aux(divisor: int) -> tuple[int, int, int]:
    """Return the multiplier and shifts for branch-free unsigned divmod."""
    shift = (divisor - 1).bit_length()
    multiplier = (
        (1 << 32) * ((1 << shift) - divisor) // divisor + 1
    ) & 0xFFFFFFFF
    return multiplier, min(shift, 1), max(shift - 1, 0)


def _fast_divmod(dividend, divisor, multiplier, shift1, shift2):
    """Decode one unsigned quotient/remainder pair without division."""
    dividend = cl.uint32(dividend)
    high_product = cl.uint32(
        cl._libdevice.umulhi(dividend, cl.uint32(multiplier))
    )
    quotient = (
        high_product
        + ((dividend - high_product) >> cl.uint32(shift1))
    ) >> cl.uint32(shift2)
    remainder = dividend - quotient * cl.uint32(divisor)
    return cl.int32(quotient), cl.int32(remainder)


def _cluster_rank_and_coord(cluster_shape_mnk):
    """Read one cluster rank and decode its compile-time CTA coordinates."""
    rank = block_in_cluster_rank()
    cluster_m, cluster_n, _ = cluster_shape_mnk
    cta_m = rank % cluster_m if cluster_m > 1 else 0
    cta_n = (rank // cluster_m) % cluster_n if cluster_n > 1 else 0
    return rank, (cta_m, cta_n, 0)


@dataclass(frozen=True)
class PersistentTileSchedulerParams:
    """Static-persistent scheduler configuration in CTA-tile units."""

    problem_shape_ntile_mnl: tuple[int, int, int]
    cluster_shape_mnk: tuple[int, int, int]
    swizzle_size: int = 1
    raster_along_m: bool = True
    _cluster_major_fastdiv: tuple[int, int, int, int] | None = field(
        default=None, repr=False
    )
    _cluster_minor_fastdiv: tuple[int, int, int, int] | None = field(
        default=None, repr=False
    )

    def __post_init__(self) -> None:
        if (
            self._cluster_major_fastdiv is not None
            and self._cluster_minor_fastdiv is not None
        ):
            return
        clusters_m, clusters_n, _ = self.problem_shape_ncluster_mnl
        major = clusters_m if self.raster_along_m else clusters_n
        minor = clusters_n if self.raster_along_m else clusters_m
        object.__setattr__(
            self, "_cluster_major_fastdiv", (major, *_fast_divmod_aux(major))
        )
        object.__setattr__(
            self, "_cluster_minor_fastdiv", (minor, *_fast_divmod_aux(minor))
        )

    def validate(self) -> None:
        if len(self.problem_shape_ntile_mnl) != 3:
            raise ValueError("problem_shape_ntile_mnl must have rank three")
        if len(self.cluster_shape_mnk) != 3:
            raise ValueError("cluster_shape_mnk must have rank three")
        if min(*self.problem_shape_ntile_mnl, *self.cluster_shape_mnk) <= 0:
            raise ValueError("problem and cluster shapes must be positive")
        if self.cluster_shape_mnk[2] != 1:
            raise ValueError("cluster_shape_k must be one")
        if self.swizzle_size != 1:
            raise NotImplementedError("persistent scheduler swizzling is not supported")

    @property
    def problem_shape_ncluster_mnl(self) -> tuple[int, int, int]:
        tiles_m, tiles_n, batches = self.problem_shape_ntile_mnl
        cluster_m, cluster_n, _ = self.cluster_shape_mnk
        return (
            (tiles_m + cluster_m - 1) // cluster_m,
            (tiles_n + cluster_n - 1) // cluster_n,
            batches,
        )

    @property
    def problem_cluster_count(self) -> int:
        m, n, batch = self.problem_shape_ncluster_mnl
        return m * n * batch

    def get_grid_shape(self, max_active_clusters: int) -> tuple[int, int, int]:
        """Return an occupancy-capped persistent cluster grid."""
        if type(max_active_clusters) is not int or max_active_clusters <= 0:
            raise ValueError("max_active_clusters must be a positive integer")
        cluster_m, cluster_n, _ = self.cluster_shape_mnk
        return (
            cluster_m,
            cluster_n,
            min(self.problem_cluster_count, max_active_clusters),
        )


@dataclass(frozen=True)
class StaticPersistentTileScheduler:
    """Immutable device state for a grid-stride persistent tile scheduler."""

    params: PersistentTileSchedulerParams
    num_persistent_clusters: object
    current_work_linear_idx: object
    cta_rank_in_cluster: object
    cta_id_in_cluster: tuple[object, object, object]
    cluster_major_divisor: int
    cluster_major_multiplier: object
    cluster_major_shift1: int
    cluster_major_shift2: int
    cluster_minor_divisor: int
    cluster_minor_multiplier: object
    cluster_minor_shift1: int
    cluster_minor_shift2: int
    num_tiles_executed: object = 0

    @staticmethod
    def create(params: PersistentTileSchedulerParams) -> "StaticPersistentTileScheduler":
        cluster_m, cluster_n, _ = params.cluster_shape_mnk
        grid_size = (
            cl.block_count(0) * cl.block_count(1) * cl.block_count(2)
        )
        num_persistent_clusters = grid_size // (cluster_m * cluster_n)
        cta_rank = block_in_cluster_rank()
        cta_coord = (
            cl.block_index(0) % cluster_m,
            cl.block_index(1) % cluster_n,
            0,
        )
        (
            cluster_major_divisor,
            major_multiplier,
            major_shift1,
            major_shift2,
        ) = params._cluster_major_fastdiv
        (
            cluster_minor_divisor,
            minor_multiplier,
            minor_shift1,
            minor_shift2,
        ) = params._cluster_minor_fastdiv
        return StaticPersistentTileScheduler(
            params=params,
            num_persistent_clusters=num_persistent_clusters,
            current_work_linear_idx=cl.block_index(2),
            cta_rank_in_cluster=cta_rank,
            cta_id_in_cluster=cta_coord,
            cluster_major_divisor=cluster_major_divisor,
            cluster_major_multiplier=major_multiplier,
            cluster_major_shift1=major_shift1,
            cluster_major_shift2=major_shift2,
            cluster_minor_divisor=cluster_minor_divisor,
            cluster_minor_multiplier=minor_multiplier,
            cluster_minor_shift1=minor_shift1,
            cluster_minor_shift2=minor_shift2,
        )

    @staticmethod
    def get_grid_shape(
        params: PersistentTileSchedulerParams, max_active_clusters: int
    ) -> tuple[int, int, int]:
        return params.get_grid_shape(max_active_clusters)

    def get_current_work(self) -> WorkTileInfo:
        work_idx = self.current_work_linear_idx
        remainder, cluster_major = _fast_divmod(
            work_idx,
            self.cluster_major_divisor,
            self.cluster_major_multiplier,
            self.cluster_major_shift1,
            self.cluster_major_shift2,
        )
        batch, cluster_minor = _fast_divmod(
            remainder,
            self.cluster_minor_divisor,
            self.cluster_minor_multiplier,
            self.cluster_minor_shift1,
            self.cluster_minor_shift2,
        )
        if self.params.raster_along_m:
            cluster_m, cluster_n = cluster_major, cluster_minor
        else:
            cluster_m, cluster_n = cluster_minor, cluster_major
        shape_m, shape_n, _ = self.params.cluster_shape_mnk
        cta_m, cta_n, _ = self.cta_id_in_cluster
        is_valid, _ = cl.shuffle_sync(
            cl.ShuffleKind.INDEX,
            cl.int32(work_idx < self.params.problem_cluster_count),
            0,
        )
        return WorkTileInfo(
            (
                cluster_m * shape_m + cta_m,
                cluster_n * shape_n + cta_n,
                batch,
            ),
            cl.bool_(is_valid),
        )

    def initial_work_tile_info(self) -> WorkTileInfo:
        return self.get_current_work()

    def advance_to_next_work(self) -> "StaticPersistentTileScheduler":
        return replace(
            self,
            current_work_linear_idx=(
                self.current_work_linear_idx + self.num_persistent_clusters
            ),
            num_tiles_executed=self.num_tiles_executed + 1,
        )


@dataclass(frozen=True)
class ClcDynamicPersistentTileSchedulerParams(PersistentTileSchedulerParams):
    """CLC dynamic-persistent scheduler configuration in CTA-tile units."""

    def get_grid_shape(self) -> tuple[int, int, int]:
        """Return the cluster-aligned launch grid consumed by CLC."""
        self.validate()
        cluster_m, cluster_n, _ = self.cluster_shape_mnk
        clusters_m, clusters_n, batches = self.problem_shape_ncluster_mnl
        if self.raster_along_m:
            return (
                clusters_m * cluster_m,
                clusters_n * cluster_n,
                batches,
            )
        return (cluster_m, cluster_n, self.problem_cluster_count)


@dataclass(frozen=True)
class ClcDynamicPersistentTileScheduler:
    """Immutable device state for a CLC dynamic-persistent scheduler."""

    params: ClcDynamicPersistentTileSchedulerParams
    response_tokens: object
    cta_rank_in_cluster: object
    cta_id_in_cluster: tuple[object, object, object]
    block_idx: tuple[object, object, object]
    num_tiles_executed: object = 0

    @staticmethod
    def create(
        params: ClcDynamicPersistentTileSchedulerParams,
        response_tokens,
    ) -> "ClcDynamicPersistentTileScheduler":
        cta_rank, cta_coord = _cluster_rank_and_coord(params.cluster_shape_mnk)
        return ClcDynamicPersistentTileScheduler(
            params=params,
            response_tokens=response_tokens,
            cta_rank_in_cluster=cta_rank,
            cta_id_in_cluster=cta_coord,
            block_idx=tuple(
                cl.block_index(axis) for axis in cl.static_iter(range(3))
            ),
        )

    @staticmethod
    def get_grid_shape(
        params: ClcDynamicPersistentTileSchedulerParams,
    ) -> tuple[int, int, int]:
        return params.get_grid_shape()

    def _rasterize(self, x_idx, y_idx, z_idx):
        if self.params.raster_along_m:
            return x_idx, y_idx, z_idx
        clusters_m, clusters_n, _ = self.params.problem_shape_ncluster_mnl
        cluster_n = z_idx % clusters_n
        minor_batch = z_idx // clusters_n
        cluster_m = minor_batch % clusters_m
        batch = minor_batch // clusters_m
        shape_m, shape_n, _ = self.params.cluster_shape_mnk
        return cluster_m * shape_m, cluster_n * shape_n, batch

    def initial_work_tile_info(self) -> WorkTileInfo:
        block_m, block_n, block_l = self.block_idx
        cta_m, cta_n, _ = self.cta_id_in_cluster
        leader_m, leader_n, batch = self._rasterize(
            block_m - cta_m,
            block_n - cta_n,
            block_l,
        )
        return WorkTileInfo((leader_m + cta_m, leader_n + cta_n, batch), True)

    def get_current_work(self, response_stage=0) -> WorkTileInfo:
        token = self.response_tokens.pointer(response_stage).load()
        has_work = cl.cluster_launch_control_is_canceled(token)
        leader_m = cl.cluster_launch_control_get_first_block_index(token, axis=0)
        leader_n = cl.cluster_launch_control_get_first_block_index(token, axis=1)
        batch = cl.cluster_launch_control_get_first_block_index(token, axis=2)
        cl.fence_proxy_bidirectional(
            cl.FenceProxy.ASYNC,
            restriction=cl.FenceRestriction.shared_block(),
        )
        leader_m, leader_n, batch = self._rasterize(
            leader_m, leader_n, batch
        )
        cta_m, cta_n, _ = self.cta_id_in_cluster
        return WorkTileInfo(
            (leader_m + cta_m, leader_n + cta_n, batch),
            has_work,
        )

    def fetch_next_work(self, barrier, response_stage):
        """Issue one multicast cluster-cancel query from the cluster leader."""
        is_cluster_leader = self.cta_rank_in_cluster == 0
        if is_cluster_leader and cl.elect_sync():
            cl.cluster_launch_control_try_cancel(
                self.response_tokens.pointer(response_stage),
                barrier,
                multicast=True,
            )
        return replace(self, num_tiles_executed=self.num_tiles_executed + 1)


@dataclass(frozen=True)
class DeviceStaticPersistentTileSchedulerConfig:
    """Compiler-visible static-persistent scheduler metadata."""

    tile_scheduler_params: object

    def create(self, smem_base=None):
        return StaticPersistentTileScheduler.create(self.tile_scheduler_params)


@dataclass(frozen=True)
class DeviceClcDynamicPersistentTileSchedulerConfig:
    """Compiler-visible CLC scheduler metadata with resolved SMEM storage."""

    tile_scheduler_params: object
    response_offset: int
    response_count: int

    def create(self, smem_base):
        pointer = smem_base.pointer() + self.response_offset
        typed_pointer = cl.bitcast(
            pointer,
            cl.pointer_dtype(cl.cluster_launch_control_token, pointer.memory_space),
        )
        response_tokens = cl.Array.from_parts(typed_pointer, self.response_count)
        return ClcDynamicPersistentTileScheduler.create(
            self.tile_scheduler_params,
            response_tokens,
        )


@dataclass(frozen=True)
class TileSchedulerConfig:
    """Host scheduler selection before response storage is laid out."""

    tile_scheduler_type: TileSchedulerType
    tile_scheduler_params: object
    response_allocation: object | None = None

    @staticmethod
    def create_static_persistent_tile_scheduler_params(
        tile_scheduler_params: PersistentTileSchedulerParams,
    ) -> "TileSchedulerConfig":
        tile_scheduler_params.validate()
        return TileSchedulerConfig(
            TileSchedulerType.StaticPersistent,
            tile_scheduler_params,
        )

    @staticmethod
    def create_clc_dynamic_persistent_tile_scheduler_params(
        tile_scheduler_params: ClcDynamicPersistentTileSchedulerParams,
        response_allocation,
    ) -> "TileSchedulerConfig":
        tile_scheduler_params.validate()
        if response_allocation is None:
            raise ValueError("CLC scheduling requires response storage")
        return TileSchedulerConfig(
            TileSchedulerType.ClcDynamicPersistent,
            tile_scheduler_params,
            response_allocation,
        )

    def to_device(self):
        if self.tile_scheduler_type is TileSchedulerType.StaticPersistent:
            return DeviceStaticPersistentTileSchedulerConfig(
                self.tile_scheduler_params
            )
        allocation = self.response_allocation
        return DeviceClcDynamicPersistentTileSchedulerConfig(
            self.tile_scheduler_params,
            allocation.offset,
            allocation.count,
        )
