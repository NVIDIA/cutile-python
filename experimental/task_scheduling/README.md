<!--- SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!--- SPDX-License-Identifier: Apache-2.0 -->

# CUDA Lang task scheduling

This package separates a pure-Python scheduling model from CUDA Lang device lowering. Import `task_scheduling` as `ts`, define resources and named `@ts.consumer_work` or `@ts.producer_work` methods, and capture a schedule with `@ts.schedule`. The captured `Schedule` is immutable; `TaskManager` consumes it for host validation, and `Task.to_device()` lowers it for device execution.

`Task.to_device()` freezes the validated schedule into CUDA Lang-compatible nodes. Directly decorated static work methods are used for device lowering. Explicit callback mappings remain available as low-level overrides and are keyed by readable captured labels (optionally `"resource.label"` or `(resource, label)` when a task needs disambiguation). Pipeline steps are generated from their `PipelineConfig` and `DevicePipelineBinding`; they do not require kernel-specific acquire/wait/commit/release callbacks. At runtime, an immutable `ExecutionContext` carries internal routed values and pipeline index/phase state while kernel resources remain available through a named `tasks_inputs` object. Domain loops, first/last/every guards, opaque guards, warp ranges, CTA pinning, and `setmaxnreg` are lowered from this frozen program.

A work decorator placed above `@staticmethod` lets host task execution and device lowering share one implementation. Instance work methods remain usable for host schedule construction; low-level users must supply an explicit callback mapping to lower them because resource instances are not passed into CUDA Lang callbacks.

`TaskManager.to_device()` infers static work callbacks across all host tasks, derives device pipeline bindings from each resource's validated `PipelineConfig`, preserves validated task order, assigns private pipeline-state slots and allocator-derived barrier offsets, and returns a frozen `DeviceTaskManager`. Explicit pipeline bindings remain available only through the low-level `Task.to_device()` interface. Tasks containing only generated scheduling or pipeline operations require no empty callback entry. Device code may use `run(inputs)` for a managed one-call lifecycle or manage resource setup explicitly:

```python
device_allocators = device_manager.setup_resources_and_tasks()
device_manager.run(inputs, device_allocators)
```

`Task` also supports a schedule-inferred form, `Task(warp_idx, num_warps, schedule=...)`. Consumer and producer resource roles are derived from captured stages. Dependency graphs are optional explicit validation metadata; omitting one leaves the graph empty. Passing `cta_warps` additionally fills uncovered warp ranges with internal padding tasks. SMEM, TMEM, and multi-pipeline barrier allocators are constructed from resource requirements when no explicit allocator is provided; explicit allocators remain authoritative for alias-sensitive layouts.

`TaskManager.freeze()` returns a canonical immutable `ProgramIR` snapshot with stable resource, task, step, value, control-flow, and dependency records. Host tools can inspect this snapshot without depending on mutable resource identity, while existing device lowering continues to use the manager's proven routing and pipeline representation.

`setup_resources_and_tasks()` materializes the dynamic SMEM and coalesced barrier arenas, initializes every pipeline's full/empty runs, and returns immutable `DeviceAllocators`. Its `smem_allocator.get(name, dtype, shape)` and `barrier_allocator.get_ptr(name)` methods expose host-laid-out named storage to kernel infrastructure such as explicit TMEM allocation and clustered deallocation. `run()` then creates each task's private routing context and executes the complete task set. Kernel authors neither construct per-task contexts nor place barrier arrays in a positional values tuple. The SMEM arena base remains available as `stage_info.context.smem_base`.

`PdlWaitBarrier` and `PdlLaunchBarrier` let schedules express programmatic launch ordering directly. Device lowering preserves each operation at its captured task position, so a leading `pdl_wait.wait_griddep()` gates that task's work without introducing a manager-wide synchronization point.

With `verbose=True`, `TaskManager.print_verbose_report()` prints task register budgets, memory usage, captured schedules with routed values, exhaustive host safety summaries, flattened checker inputs, and a representative complete/deadlock/race timeline. Allocator reports include allocation names, sizes, alignments, offsets, alias groups, barrier bytes, and physical totals. When a schedule contains runtime domain bounds, `exhaustive_representative_domain=True` makes the host checker use a bounded domain derived from the loop starts, while device lowering resolves the actual trip count from kernel inputs.

The manager also computes a warp-group-rounded initial register budget and threads it into device task lowering, so `setmaxnreg` direction and verbose register contributions use the same baseline.

Automatic value routing hides storage positions from callback authors. Each callback receives its routed inputs after the `StageInfo` argument and returns the values declared by its captured work method. The frozen `DeviceStep` performs the corresponding value-stack operations, while `DeviceTask.make_context(tasks_inputs)` creates the immutable execution context. The tuple representation remains an internal compiler detail; no placeholder or named routing slot is exposed to resource authors.

`outputs=N` creates `N` independent lexical `ScheduleValue` instances. A work call returns one value directly or a tuple that can be unpacked normally, and its device callback returns the same number of runtime values. A value produced inside a conditional or domain loop cannot escape that scope accidentally. A domain loop that needs state across iterations uses `domain_loop(start, end, step, body, *initial_values)`, with one callback parameter per initial value. The schedule builder invokes `body(*iter_values)` once while capturing the loop body, so every resource call remains a schedule step. The callback must return one work-call output per input; those outputs become both the next-iteration values and the post-loop results. The device adapter preserves zero-trip pass-through and materializes the backedge without exposing routing positions to callbacks. Work methods read the current device offset from `stage_info.loop_offset`; it is not a callback parameter or a routed value. Loop bodies may use `first_iter()`, `last_iter()`, and `every(period, start=...)` to capture iteration-guarded regions.

`when_true()` and `when_false()` regions may nest, including inside iteration
guards. An inner region executes only when every enclosing guard is active.
Predicates and other work outputs produced inside a branch may be used by its
nested regions, but cannot escape to a parent or sibling scope. Reusing the same
routed predicate preserves true/false correlation in host analysis.

The runnable `tutorial/01_copy_basics/04_copy_tma_nested_conditional.py` example
extends the conditional TMA copy with all four two-level true/false combinations
and a third-level condition. It checks the copied tensor and exact per-row branch
markers, including the absence of writes from inactive branches:

```bash
python experimental/task_scheduling/tutorial/01_copy_basics/04_copy_tma_nested_conditional.py --rows-cols 9,512
```

`break_loop()` exits the enclosing domain loop immediately, including from a
nested conditional. It skips the rest of that iteration and resumes after the
loop. The context-manager handle provides the same operation:

```python
with ts.domain_loop(num_tiles) as loop:
    done = resource.is_done()
    with ts.when_true(done):
        loop.break_loop()
    resource.process()
```

Use `ts.break_loop()` in a functional loop body. A bare break skips the normal
backedge: returned loop-carried results retain their values from the last
completed iteration (or the initial values if the first iteration breaks).
Pass explicit results to return values computed during the interrupted iteration:

```python
initial = resource.init_state()

def body(state):
    updated = resource.advance_state(state)
    done = resource.is_done(updated)
    with ts.when_true(done):
        ts.break_loop(updated)
    return updated

result = ts.domain_loop(0, num_tiles, 1, body, initial)
resource.consume_state(result)
```

For multiple carried inputs, use `ts.break_loop(next_count, next_total)` in the
same order as the functional loop's inputs. Supply all carried results or none.
Each explicit argument must be a visible routed `ScheduleValue` with the same
pipeline-stage provenance and a compatible CUDA type as its carried input;
ordinary constants must first be returned by a work method. Values created
inside a nested branch may be returned directly by a break in that branch.
The exit values are captured before branch-local routes are discarded. Wrong
arity, foreign-schedule values, and values escaping a sibling/child scope are
rejected during capture; CUDA types are checked during device compilation.
An untaken break uses the normal body return values, and a zero-trip loop still
returns its initial values.

Side effects and pipeline-state updates performed before the break are retained.
A break exits only the domain loop, so an enclosing `work_tile_loop()` continues.
`last_iter()` still means the last iteration of the declared range; it is not an
exit hook. Breaks outside a domain loop and expired loop handles are rejected.
Unbounded while loops remain unsupported.

The nested-copy tutorial also accepts `--stop-row 5`. Both producer and consumer
break before acquiring/waiting for row 5, and the example verifies that the
uncopied output and trace remain zero. A pipeline schedule must ensure that its
producer and consumer exit consistently and finish any acquired work before
exiting; `break_loop()` does not implicitly release or drain an interrupted step.

Host expansion honors breaks for its selected opaque-condition assignment and
representative loop bounds, including zero-trip loops. Opaque assignments remain
fixed during each explored execution; these checks do not prove safety for all
possible iteration-varying runtime predicate sequences.

`StageInfo` contains the current `stage_idx`, `phase`, selected full `barrier`, zero-based iteration count, loop offset/bounds, work label, and owning `ExecutionContext`; loop offset/bounds are `None` for peeled work outside a domain loop. This lets pipeline payload work use `stage_info.barrier` without knowing how barrier arrays are stored by the device manager.

Work-call arguments are classified during schedule capture: `ScheduleValue` arguments are routed, while ordinary Python values are captured as compile-time operands. The frozen device step preserves the method's original parameter order when it combines both kinds. Work methods may therefore declare any number of static arguments and route any statically known number of inputs or results without annotating the argument kinds. Routed values may have different CUDA Lang types. Work methods declare their result count with `outputs=N`, and each call creates independent lexical SSA values.

The public host API also includes identity-based `MemoryResource` metadata, pipeline configurations and groups, aligned SMEM/TMEM/barrier allocators, bounded exhaustive interleaving checks, and static pipeline auditing. These objects can be constructed and validated on a machine without a CUDA device.

`TmemAllocator` validates and reports the host-side TMEM layout, while kernels own the device lifecycle explicitly. A kernel allocates TMEM, synchronizes its participating warps, passes the pointer through its named task inputs, and performs CTA or clustered peer-CTA deallocation after task execution.

## Device support boundary

`AsyncAsync`, `TmaAsync`, `TmaUmma`, and `UmmaAsync` metadata have a generic CUDA Lang lowering path. It owns immutable producer/consumer index and phase state, initializes offset-addressed full/empty runs in one manager-owned mbarrier arena, lowers try/acquire/wait, and emits the pipeline-specific mbarrier or `tcgen05.commit` signals. The remaining UMMA, CLC, DPC, and DLC pipeline kinds remain available to host analysis but raise `NotImplementedError` when device pipeline creation is requested.

Persistent `WorkTileLoop` execution requires a scheduler-specific work-queue adapter and is currently host-analysis-only. Domain-loop starts and ends may be runtime values declared with `dynamic_domain_bound(resolver)`. A schedule may also declare `stage_info` as its first parameter and use, for example, `stage_info.context.tasks_inputs.num_rows` directly as a dynamic loop bound. The resolver receives the manager-owned `tasks_inputs` object inside the kernel; host analysis continues to use an explicit representative-domain fallback. Device domain loops currently require a positive static step. Literal opaque guards need an explicit device-condition adapter, while routed `ScheduleValue` opaque guards are supported.

The exhaustive checker is bounded by `max_states`. Its result reports `hit_state_limit=True` when the bound prevents a proof; callers must not treat that result as a successful exhaustive proof.

## Current limitations

The current checker proves acquire/wait availability and physical-range alias safety across expanded domain/work-tile loops and opaque assignments. It does not yet model PDL launch ordering or scheduler-specific CLC queue state. PDL and CLC metadata can be retained on host objects, but they are not currently part of the interleaving state, so those configurations require an external check.

Pipeline groups, interleave strides, multicast signaling, and deferred barrier storage are validated as host metadata. Generic device materialization for those combinations is not implemented. The copy and GEMM tutorials exercise the shared TMA-to-async, TMA-to-UMMA, and UMMA-to-async lowering while their callbacks contain only pipeline payload work. Static pipeline auditing recognizes the corresponding CUDA Lang calls but does not infer effects hidden behind arbitrary user wrappers.
