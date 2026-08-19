# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Task-scheduled single-CTA FP16/BF16 Blackwell GEMM.

Each CTA computes one 128x256 output tile. Warps 0-3 drain FP32 TMEM
accumulators, warp 4 issues two TMA loads per K stage, warp 5 issues tcgen05
MMA, and warps 6-7 provide the second complete register-allocation warp group.
The immutable host graph is:

    GmemAB -> SmemAB (TMA/UMMA) -> TmemC (UMMA/async) -> GmemD

CUDA Lang does not yet generically materialize TmaUmma and UmmaAsync pipeline
objects. This tutorial keeps those configurations for host analysis and lowers
their validated steps through static work methods shared by host capture and
device lowering. The K-tile domain is resolved from the runtime K kernel
argument.
"""

import argparse
from dataclasses import dataclass

import cuda.lang as cl
import torch

import task_scheduling as ts


WARP_SIZE = 32
BLOCK_THREADS = 8 * WARP_SIZE
BLOCK_M = 128
BLOCK_N = 256
BLOCK_K = 64
MMA_K = 16
AB_STAGES = 3
ACC_STAGES = 2
STORE_WARPS = 4
LOAD_WARPS = 1
MMA_WARPS = 1
PADDING_WARPS = 2
STORE_TASK_WARP_IDX = 0
LOAD_TASK_WARP_IDX = 4
MMA_TASK_WARP_IDX = 5
PADDING_TASK_WARP_IDX = 6
TMEM_COLS = BLOCK_N * ACC_STAGES
TMEM_SYNC_BARRIER = 2
DEALLOC_BARRIER = 3
VEC_BYTES = 32
IO_ELEMENT_BYTES = 2
A_STAGE_ELEMS = BLOCK_M * BLOCK_K
B_STAGE_ELEMS = BLOCK_N * BLOCK_K
TMA_BYTES = (A_STAGE_ELEMS + B_STAGE_ELEMS) * IO_ELEMENT_BYTES
# This program owns the complete SMEM arena: A starts at its base and B follows A.
A_SMEM_OFFSET_BYTES = 0
B_SMEM_OFFSET_BYTES = AB_STAGES * A_STAGE_ELEMS * IO_ELEMENT_BYTES
_DEFAULT_MNK = (512, 512, 512)
_DEFAULT_TOLERANCE = 1.0e-1
_DTYPE_MAP = {"fp16": torch.float16, "bf16": torch.bfloat16}


# -----------------------------------------------------------------------------
# Host resource model and schedule tree
# -----------------------------------------------------------------------------


@dataclass(kw_only=True, eq=False)
class GmemAbResource(ts.MemoryResource):
    """Produce the independently routed coordinates for one TMA stage."""

    @ts.consumer_work(outputs=3)
    @staticmethod
    def compute_coords(stage_info):
        return (
            stage_info.loop_offset * BLOCK_K,
            cl.block_index(0) * BLOCK_M,
            cl.block_index(1) * BLOCK_N,
        )


@dataclass(kw_only=True, eq=False)
class SmemAbResource(ts.MemoryResource):
    """Own the routed A/B SMEM arrays and their per-stage MMA descriptors."""

    @staticmethod
    def _init_smem_state(stage_info):
        """Create the real A/B views into the manager-owned SMEM arena."""
        smem_base = stage_info.context.smem_base.get_base_pointer()
        a_smem = cl.reinterpret_pointer_as_array(
            smem_base + A_SMEM_OFFSET_BYTES,
            cl.uint16,
            (AB_STAGES, A_STAGE_ELEMS),
        )
        b_smem = cl.reinterpret_pointer_as_array(
            smem_base + B_SMEM_OFFSET_BYTES,
            cl.uint16,
            (AB_STAGES, B_STAGE_ELEMS),
        )
        return a_smem, b_smem

    @ts.producer_work(
        work_attrs=ts.WorkAttr.AUXILIARY,
        outputs=2,
    )
    @staticmethod
    def init_load_state(stage_info):
        smem_values = SmemAbResource._init_smem_state(stage_info)
        values = stage_info.context.tasks_inputs
        # Tensor-map prefetch belongs to load-task initialization, while the
        # returned arrays are routed into every loop iteration.
        if cl.lane_index() == 0:
            cl.prefetch_tensor_map(values.a_map)
            cl.prefetch_tensor_map(values.b_map)
        return smem_values

    @ts.consumer_work(
        work_attrs=ts.WorkAttr.AUXILIARY,
        outputs=2,
    )
    @staticmethod
    def init_descriptors(stage_info):
        return SmemAbResource._init_smem_state(stage_info)

    @ts.producer_work
    @staticmethod
    def tma_load(stage_info, a_smem, b_smem, coord_k, coord_m, coord_n):
        values = stage_info.context.tasks_inputs
        stage = stage_info.stage_idx
        if cl.lane_index() == 0:
            cl.copy_async_bulk_tensor_global_to_shared(
                values.a_map,
                (coord_k, coord_m),
                a_smem.get_element_pointer((stage, 0)),
                stage_info.barrier,
            )
            cl.copy_async_bulk_tensor_global_to_shared(
                values.b_map,
                (coord_k, coord_n),
                b_smem.get_element_pointer((stage, 0)),
                stage_info.barrier,
            )

    @ts.consumer_work(outputs=2)
    @staticmethod
    def build_descriptors(stage_info, a_smem, b_smem):
        # Route the A and B descriptors independently to the MMA work method.
        stage = stage_info.stage_idx
        a_desc = cl.Tcgen05SharedMemoryDescriptor(
            matrix_start_address=a_smem.get_element_pointer((stage, 0)),
            leading_dimension_byte_offset=16,
            stride_dimension_byte_offset=8 * 128,
            swizzle_mode=cl.SwizzleMode.SWIZZLE_128B,
        ).encode()
        b_desc = cl.Tcgen05SharedMemoryDescriptor(
            matrix_start_address=b_smem.get_element_pointer((stage, 0)),
            leading_dimension_byte_offset=16,
            stride_dimension_byte_offset=8 * 128,
            swizzle_mode=cl.SwizzleMode.SWIZZLE_128B,
        ).encode()
        return cl.int64(a_desc), cl.int64(b_desc)


@dataclass(kw_only=True, eq=False)
class TmemCResource(ts.MemoryResource):
    @staticmethod
    def _init_tmem_state(stage_info):
        """Read the TMEM base and initialize the MMA instruction descriptor."""
        values = stage_info.context.tasks_inputs
        # Tcgen05 F16Type uses F16=0 and BF16=1. Normalize the aggregate field
        # to uint32 so descriptor bit insertion has one type on both paths.
        input_type = cl.uint32(values.is_bf16)
        idesc = cl.Tcgen05InstructionDescriptor(
            d_type=cl.Tcgen05InstructionDescriptor.DType.F32,
            a_type=input_type,
            b_type=input_type,
            n=BLOCK_N,
            m=BLOCK_M,
        ).encode()
        return idesc

    @ts.producer_work
    @staticmethod
    def mma(stage_info, desc_a_base, desc_b_base, idesc):
        values = stage_info.context.tasks_inputs
        tmem = cl.tcgen05_tmem_offset(
            values.tmem, column_offset=stage_info.stage_idx * BLOCK_N
        )
        for kk in cl.static_iter(range(BLOCK_K // MMA_K)):
            if cl.lane_index() == 0:
                cl.tcgen05_mma(
                    cl.Tcgen05MMAKind.F16,
                    tmem,
                    cl.int64(desc_a_base + 2 * kk),
                    cl.int64(desc_b_base + 2 * kk),
                    cl.int32(idesc),
                    accumulate=stage_info.count != 0 or kk != 0,
                    cta_group=cl.CTAGroup.CTA_1,
                )

    @ts.consumer_work(outputs=1)
    @staticmethod
    def load_subtile(stage_info, subtile_idx):
        values = stage_info.context.tasks_inputs
        tmem_subtile = cl.tcgen05_tmem_offset(
            values.tmem,
            lane_offset=stage_info.context.warp_index * WARP_SIZE,
            column_offset=stage_info.stage_idx * BLOCK_N + subtile_idx * 32,
        )
        t2r_rmem = cl.tcgen05_load(
            cl.Tcgen05LoadStoreShape.SHAPE_32X32B,
            tmem_subtile,
            element_count=32,
            dtype=cl.float32,
        )
        cl.tcgen05_wait_load()
        return t2r_rmem

    @ts.producer_work(
        work_attrs=ts.WorkAttr.AUXILIARY,
        outputs=1,
    )
    @staticmethod
    def init_accumulator_state(stage_info):
        return TmemCResource._init_tmem_state(stage_info)

    @ts.consumer_work(
        work_attrs=ts.WorkAttr.AUXILIARY,
    )
    @staticmethod
    def init_store_state(stage_info):
        return TmemCResource._init_tmem_state(stage_info)

    @ts.producer_work(
        work_attrs=ts.WorkAttr.AUXILIARY,
    )
    @staticmethod
    def init_work_tile_state(stage_info):
        # The first-MMA overwrite is derived from stage_info.count, so no
        # mutable scale reset is needed here.
        return TmemCResource._init_tmem_state(stage_info)


@dataclass(kw_only=True, eq=False)
class GmemDResource(ts.MemoryResource):
    @ts.producer_work
    @staticmethod
    def store(stage_info, t2r_rmem, subtile_idx):
        resources = stage_info.context.tasks_inputs
        row = cl.block_index(0) * BLOCK_M + cl.thread_index(0)
        column = cl.block_index(1) * BLOCK_N + subtile_idx * 32
        vsize = VEC_BYTES // 2
        for vector_idx in cl.static_iter(range(32 // vsize)):
            vector_column = column + vector_idx * vsize
            # The launch grid and host validation admit only complete M/N tiles,
            # so this vector is always in bounds for the specialization.
            fragment = t2r_rmem[
                vector_idx * vsize:vector_idx * vsize + vsize
            ]
            packed = fragment.astype(resources.output.dtype)
            resources.output.get_element_pointer((row, vector_column)).store(
                packed, alignment=VEC_BYTES
            )


@ts.schedule
def load_schedule(stage_info, gmem_ab, smem_ab):
    # Initialize A/B views once, then route them with each coordinate value.
    a_smem, b_smem = smem_ab.init_load_state()
    with ts.domain_loop(stage_info.context.tasks_inputs.num_k_tiles):
        coord_k, coord_m, coord_n = gmem_ab.compute_coords()
        smem_ab.try_acquire()
        smem_ab.acquire()
        smem_ab.tma_load(a_smem, b_smem, coord_k, coord_m, coord_n)
        smem_ab.commit()


@ts.schedule
def mma_schedule(stage_info, smem_ab, tmem_c):
    # The MMA task has its own context and therefore initializes its own A/B route.
    a_smem, b_smem = smem_ab.init_descriptors()
    idesc = tmem_c.init_accumulator_state()
    tmem_c.init_work_tile_state()
    tmem_c.try_acquire()
    tmem_c.acquire()
    with ts.domain_loop(stage_info.context.tasks_inputs.num_k_tiles):
        smem_ab.try_wait()
        smem_ab.wait()
        desc_a_base, desc_b_base = smem_ab.build_descriptors(a_smem, b_smem)
        tmem_c.mma(desc_a_base, desc_b_base, idesc)
        smem_ab.release()
    tmem_c.commit()


@ts.schedule
def store_schedule(stage_info, tmem_c, gmem_d):
    tmem_c.init_store_state()
    with ts.domain_loop(stage_info.context.tasks_inputs.num_k_tiles):
        pass
    tmem_c.try_wait()
    tmem_c.wait()
    for subtile_idx in range(BLOCK_N // 32):
        t2r_rmem = tmem_c.load_subtile(subtile_idx=subtile_idx)
        gmem_d.store(t2r_rmem, subtile_idx=subtile_idx)
    tmem_c.release()


@ts.schedule
def padding_schedule(stage_info):
    with ts.domain_loop(stage_info.context.tasks_inputs.num_k_tiles):
        pass


@dataclass(frozen=True)
class TasksInputs:
    a_map: object
    b_map: object
    tmem: object
    output: object
    num_k_tiles: object
    is_bf16: object


@dataclass(frozen=True)
class GemmProgram:
    """Host model and frozen device manager for one GEMM specialization."""

    manager: object
    device_manager: object
    smem_ab: SmemAbResource
    tmem_c: TmemCResource
    smem_allocator: object
    tmem_allocator: object
    barrier_allocator: object
    a_allocation: object
    b_allocation: object


def make_gemm_program():
    """Construct and validate the complete task-scheduling program."""
    smem_config = ts.PipelineConfig.create_tma_umma_pipeline_cfg(
        num_stages=AB_STAGES,
        num_bytes=TMA_BYTES,
        producer_group=ts.CooperativeGroup(1),
        consumer_group=ts.CooperativeGroup(1),
        cta_layout_vmnk=(1, 1, 1, 1),
    )
    tmem_config = ts.PipelineConfig.create_umma_async_pipeline_cfg(
        num_stages=ACC_STAGES,
        producer_group=ts.CooperativeGroup(1),
        consumer_group=ts.CooperativeGroup(STORE_WARPS * WARP_SIZE),
        cta_layout_vmnk=(1, 1, 1, 1),
    )

    a_allocation = ts.SmemAllocation(
        "smem_a",
        AB_STAGES * A_STAGE_ELEMS * IO_ELEMENT_BYTES,
        alignment=128,
    )
    b_allocation = ts.SmemAllocation(
        "smem_b",
        AB_STAGES * B_STAGE_ELEMS * IO_ELEMENT_BYTES,
        alignment=128,
    )
    acc_allocation = ts.TmemAllocation("tmem_acc", TMEM_COLS)

    gmem_ab = GmemAbResource(name="GmemAb")
    smem_ab = SmemAbResource(
        name="SmemAb",
        pipeline_config=smem_config,
        smem_requirements=[a_allocation, b_allocation],
    )
    tmem_c = TmemCResource(
        name="TmemC",
        pipeline_config=tmem_config,
        tmem_requirements=[acc_allocation],
    )
    gmem_d = GmemDResource(name="GmemD")

    load_task = ts.Task(
        LOAD_TASK_WARP_IDX,
        LOAD_WARPS,
        schedule=load_schedule(gmem_ab, smem_ab),
        num_registers=40,
        name="LoadTask",
    )
    mma_task = ts.Task(
        MMA_TASK_WARP_IDX,
        MMA_WARPS,
        schedule=mma_schedule(smem_ab, tmem_c),
        num_registers=40,
        name="MmaTask",
    )
    store_task = ts.Task(
        STORE_TASK_WARP_IDX,
        STORE_WARPS,
        schedule=store_schedule(tmem_c, gmem_d),
        num_registers=160,
        name="StoreTask",
    )
    padding_task = ts.Task(
        PADDING_TASK_WARP_IDX,
        PADDING_WARPS,
        schedule=padding_schedule(),
        num_registers=40,
        name="PaddingTask",
    )

    smem_allocator = ts.SmemAllocator()
    smem_allocator.add_resource(smem_ab)
    smem_allocator.compute_layout()
    if a_allocation.offset != A_SMEM_OFFSET_BYTES:
        raise ValueError("A SMEM offset does not match its allocator layout")
    if b_allocation.offset != B_SMEM_OFFSET_BYTES:
        raise ValueError("B SMEM offset does not match its allocator layout")
    if a_allocation.offset % IO_ELEMENT_BYTES:
        raise ValueError("A SMEM allocation offset must be element-aligned")
    if b_allocation.offset % IO_ELEMENT_BYTES:
        raise ValueError("B SMEM allocation offset must be element-aligned")

    tmem_allocator = ts.TmemAllocator()
    tmem_allocator.add_resource(tmem_c)
    tmem_allocator.compute_layout()
    barrier_allocator = ts.BarrierAllocator()
    barrier_allocator.add_producer_consumer(
        "SmemAb", AB_STAGES, ts.CooperativeGroup(1), ts.CooperativeGroup(1)
    )
    barrier_allocator.add_producer_consumer(
        "TmemC",
        ACC_STAGES,
        ts.CooperativeGroup(1),
        ts.CooperativeGroup(STORE_WARPS * WARP_SIZE),
    )
    barrier_allocator.compute_layout()

    manager = ts.TaskManager(
        tasks=[load_task, mma_task, store_task, padding_task],
        smem_allocator=smem_allocator,
        tmem_allocator=tmem_allocator,
        barrier_allocator=barrier_allocator,
        verbose=True,
        exhaustive_representative_domain=True,
    )
    device_manager = manager.to_device()
    return GemmProgram(
        manager,
        device_manager,
        smem_ab,
        tmem_c,
        smem_allocator,
        tmem_allocator,
        barrier_allocator,
        a_allocation,
        b_allocation,
    )


# -----------------------------------------------------------------------------
# Kernel launch and validation
# -----------------------------------------------------------------------------


def make_gemm_kernel(device_manager):
    """Specialize the kernel for one frozen task manager."""

    @cl.kernel
    def gemm_kernel(a, b, c, k, is_bf16: cl.Constant[bool]):
        a_map = cl.tensor_map_tiled(
            a,
            (BLOCK_K, BLOCK_M),
            order="F",
            swizzle=cl.SwizzleMode.SWIZZLE_128B,
        )
        b_map = cl.tensor_map_tiled(
            b,
            (BLOCK_K, BLOCK_N),
            order="F",
            swizzle=cl.SwizzleMode.SWIZZLE_128B,
        )
        device_allocators = device_manager.setup_resources_and_tasks()
        tmem_storage = cl.shared_array(
            1,
            cl.pointer_dtype(cl.float32, cl.MemorySpace.TENSOR),
            alignment=4,
        )

        warp_index = device_allocators.warp_index
        if warp_index == 0:
            cl.tcgen05_allocate(
                tmem_storage.get_base_pointer(),
                TMEM_COLS,
                cta_group=cl.CTAGroup.CTA_1,
            )
            cl.tcgen05_relinquish_allocation_permit(cta_group=cl.CTAGroup.CTA_1)
        if warp_index < STORE_WARPS or warp_index == MMA_TASK_WARP_IDX:
            cl.barrier_sync_block(
                number_of_threads=(STORE_WARPS + 1) * WARP_SIZE,
                barrier_id=TMEM_SYNC_BARRIER,
            )
        tmem = tmem_storage[0]

        device_manager.run(
            TasksInputs(
                a_map,
                b_map,
                tmem,
                c,
                k // BLOCK_K,
                is_bf16,
            ),
            device_allocators,
        )

        if warp_index < STORE_WARPS:
            cl.barrier_sync_block(
                number_of_threads=STORE_WARPS * WARP_SIZE,
                barrier_id=DEALLOC_BARRIER,
            )
            if warp_index == 0:
                cl.tcgen05_deallocate(
                    tmem, TMEM_COLS, cta_group=cl.CTAGroup.CTA_1
                )

    return gemm_kernel


def _validate_mnk(mnk: tuple[int, int, int]) -> None:
    if len(mnk) != 3:
        raise ValueError("MNK must contain exactly three values")
    m, n, k = mnk
    if m <= 0 or m % BLOCK_M:
        raise ValueError(f"M must be a positive multiple of {BLOCK_M}")
    if n <= 0 or n % BLOCK_N:
        raise ValueError(f"N must be a positive multiple of {BLOCK_N}")
    if k <= 0 or k % BLOCK_K:
        raise ValueError(f"K must be a positive multiple of {BLOCK_K}")


def prepare_tensors(
    m: int, n: int, k: int, dtype: str = "fp16"
) -> dict[str, torch.Tensor]:
    _validate_mnk((m, n, k))
    if dtype not in _DTYPE_MAP:
        raise ValueError(f"dtype must be one of {tuple(_DTYPE_MAP)}")
    torch.manual_seed(1111)
    torch_dtype = _DTYPE_MAP[dtype]

    def make(rows, columns):
        return (
            torch.empty((rows, columns), dtype=torch.int32)
            .random_(-2, 2)
            .to(device="cuda", dtype=torch_dtype)
        )

    return {
        "a": make(m, k),
        "b": make(n, k),
        "c": torch.empty((m, n), device="cuda", dtype=torch_dtype),
    }


def run(tensors: dict[str, torch.Tensor], stream=None, *, verbose=False) -> None:
    a, b, c = tensors["a"], tensors["b"], tensors["c"]
    m, k = a.shape
    if b.ndim != 2 or b.shape[1] != k:
        raise ValueError("B must have shape (N, K) with the same K as A")
    n = b.shape[0]
    _validate_mnk((m, n, k))
    if c.shape != (m, n):
        raise ValueError("C must have shape (M, N)")
    if a.dtype != b.dtype or a.dtype != c.dtype:
        raise ValueError("A, B, and C must have the same dtype")
    if a.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("A, B, and C must be FP16 or BF16")

    program = make_gemm_program()
    if verbose:
        program.manager.print_verbose_report()
    gemm_kernel = make_gemm_kernel(program.device_manager)
    cl.launch(
        torch.cuda.current_stream() if stream is None else stream,
        (m // BLOCK_M, n // BLOCK_N, 1),
        (BLOCK_THREADS, 1, 1),
        gemm_kernel,
        (a, b, c, k, a.dtype == torch.bfloat16),
    )


def verify_output(tensors: dict[str, torch.Tensor], tolerance=_DEFAULT_TOLERANCE):
    a, b, c = tensors["a"], tensors["b"], tensors["c"]
    expected = torch.einsum("mk,nk->mn", a.float(), b.float()).to(c.dtype)
    torch.testing.assert_close(c, expected, atol=tolerance, rtol=1.0e-5)


def verify(
    mnk: tuple[int, int, int] = _DEFAULT_MNK,
    dtype: str = "fp16",
    tolerance: float = _DEFAULT_TOLERANCE,
):
    print("===================================================================")
    print("Running Blackwell 16-bit GEMM TS tutorial kernel with:")
    print(f"  mnk:       {mnk}")
    print(f"  dtype:     {dtype}")
    print("  scheduler: direct CTA tile mapping")
    print(f"  tolerance: {tolerance}")
    print("===================================================================")
    print()
    tensors = prepare_tensors(*mnk, dtype=dtype)
    run(tensors, verbose=True)
    torch.cuda.synchronize()
    verify_output(tensors, tolerance)
    print("PASS")


def _parse_mnk(value):
    try:
        result = tuple(int(item.strip()) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from error
    if len(result) != 3:
        raise argparse.ArgumentTypeError("expected exactly three MNK values")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mnk", type=_parse_mnk, default=_DEFAULT_MNK)
    parser.add_argument("--dtype", choices=tuple(_DTYPE_MAP), default="fp16")
    parser.add_argument("--tolerance", type=float, default=_DEFAULT_TOLERANCE)
    arguments = parser.parse_args()
    verify(arguments.mnk, arguments.dtype, arguments.tolerance)
