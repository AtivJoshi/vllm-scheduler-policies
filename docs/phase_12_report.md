# Phase 12 Report: Primal LP Relaxation Helper and Dry-Run Bridge

## Purpose

Phase 12 moved the project from documentation-only LP scheduler planning toward
the first implementation pieces for a primal LP relaxation scheduler.

Phase 12 started after the Phase 11 bridge documentation commit:

```text
9805e98 Add scheduler v1 implementation bridge docs
```

Phase 11 translated the LaTeX section
`Primal Heuristic 1: Approximation via LP Relaxation` into implementation-facing
notes. Phase 12 intentionally diverged from the original Phase 1-11 plan, which
had emphasized simpler scheduler policies first. The current direction is to
work toward the full primal LP relaxation scheduler, while staging the work to
avoid unsafe edits to native vLLM scheduler internals.

The core safety principle remains:

```text
InstrumentedSchedulerMixin.schedule() is the outer timing/logging wrapper.
Instrumented policies override _schedule_impl(), not schedule().
Behavior-preserving policies delegate to native scheduling unchanged.
```

## Phase 12.1: Read-Only vLLM Scheduler Mapping

Phase 12.1 was a read-only inspection phase against the pinned vLLM scheduler.
Its goal was to identify concrete vLLM state, fields, methods, and invariants
needed before implementing LP integration.

No scheduler implementation happened in Phase 12.1.

### Verified scheduler state

Phase 12.1 verified these scheduler containers:

- `Scheduler.waiting`
- `Scheduler.skipped_waiting`
- `Scheduler.running`

These are the live request containers that a future LP scheduler must inspect or
eventually manipulate. In Phase 12.4 dry-run mode they are read only.

### Verified request fields

Phase 12.1 verified these request fields as relevant to LP snapshotting:

- `Request.request_id`
- `Request.arrival_time`
- `Request.num_computed_tokens`

`Request.num_computed_tokens` is the key field for partial prefill. It indicates
how much of the request has already been computed and lets the LP layer reason
about remaining prefill work.

Decode eligibility is not a single stored vLLM flag. It must be derived from
request status and token counters, and it becomes more delicate for advanced
request modes.

### Verified capacity mappings

Phase 12.1 verified these scheduler capacity mappings:

- Token budget maps to `self.max_num_scheduled_tokens`.
- Sequence budget maps to `self.max_num_running_reqs` / `max_num_seqs`.
- Free KV blocks are readable through
  `self.kv_cache_manager.block_pool.get_num_free_blocks()`.

The LP memory unit for vLLM integration is KV blocks, not bytes and not raw GPU
utilization.

### Verified native scheduler APIs

Phase 12.1 verified these native scheduler APIs and invariants:

- `allocate_slots()` is the authoritative vLLM allocation safety check.
- `_preempt_request()` is the native recomputation-preemption path.
- `Scheduler.schedule()` returns a valid native `SchedulerOutput`.

These APIs are important for future action translation. They are not called by
the Phase 12.4 dry-run snapshot path.

## Phase 12.2: Documentation and Design Update

Phase 12.2 updated the LP scheduler design after the Phase 12.1 mapping.

The main design correction was that LP memory planning should use KV blocks.
`gpu_memory_utilization` is not `W_t`; it is a startup memory-pool sizing
parameter and should not be treated as the dynamic per-step LP memory capacity.

The dynamic LP memory capacity should be based on free KV blocks:

```text
self.kv_cache_manager.block_pool.get_num_free_blocks()
```

Any scheduler-local reserve such as `lp_memory_reserve_blocks` is a future
tunable policy parameter. Suggested values, including `8` blocks, were proposals
only and were not implemented defaults.

Phase 12.2 also clarified:

- `allocate_slots()` remains authoritative for real vLLM feasibility.
- Decode eligibility is derived, not read from a stored flag.
- Advanced request states should initially fall back to default scheduling.
- `LPActionPlan` is an internal planning object, not a valid `SchedulerOutput`.

## Phase 12.3: Pure-Python LP Helper Layer

Phase 12.3 implemented the pure-Python helper layer for the primal LP relaxation
under:

```text
vllm_scheduler_policies/primal_lp/
```

This layer is separate from live vLLM scheduler state. It is intended to be
unit-testable with synthetic request snapshots.

### Added types

Phase 12.3 introduced dataclasses including:

- `LPRequestSnapshot`
- `LPCapacities`
- `RelaxedLPSolution`
- `LPActionPlan`

`LPRequestSnapshot` represents the scheduler state needed by the LP layer.
`LPCapacities` represents the per-step token, sequence, and KV-block budgets.
`RelaxedLPSolution` represents the relaxed solver result. `LPActionPlan`
represents extracted synthetic actions and remains internal to the LP layer.

`LPActionPlan` is not a vLLM `SchedulerOutput` and must not be returned from a
vLLM scheduler.

### Solver formulation

The helper layer uses SciPy:

```text
linprog(method="highs")
```

The formulation includes these variables:

- `x_i`: prefill token amount
- `y_i`: decode action
- `z_i`: preemption action
- `I_i^P`: prefill indicator

The LP helper builds a relaxed problem, solves it, and exposes the relaxed
values for extraction.

### Extraction

Phase 12.3 implemented integral/fractional extraction from the relaxed solution.
The extraction layer partitions variables into integral and fractional decisions
and produces an `LPActionPlan`.

It also records:

```text
fractional_rule_violation
```

This diagnostic identifies cases where the solution violates the expected
almost-integral structure.

### Tests

Phase 12.3 added synthetic tests for the pure helper layer. These tests do not
require a server, live vLLM scheduling, CUDA execution, or benchmarks.

### Phase 12.3 boundaries

Phase 12.3 intentionally did not:

- edit native vLLM files;
- add a real scheduler class;
- implement `_translate_lp_actions()`;
- construct or return `SchedulerOutput`;
- call `allocate_slots()`;
- call `_preempt_request()`;
- run server experiments;
- run benchmarks.

## Phase 12.4: Dry-Run Scheduler Bridge

Phase 12.4 connected the pure LP helper layer to live vLLM scheduler state in
diagnostic dry-run mode.

The bridge is implemented in:

```text
vllm_scheduler_policies/primal_lp_dry_run.py
```

The scheduler class is:

```python
class PrimalLPDryRunScheduler(InstrumentedSchedulerMixin, Scheduler):
    ...
```

It overrides `_schedule_impl()`, not `schedule()`.

### Delegation invariant

`InstrumentedSchedulerMixin.schedule()` remains the outer wrapper. It times the
entire scheduler call and writes the normal `scheduler_call` JSONL record.

`PrimalLPDryRunScheduler._schedule_impl()` runs dry-run LP diagnostics and then
delegates to native scheduling through:

```python
Scheduler.schedule(self)
```

or through a helper that does exactly that.

Subclass `_schedule_impl()` must never call:

```python
super().schedule()
```

In a class with MRO
`PrimalLPDryRunScheduler -> InstrumentedSchedulerMixin -> Scheduler`, calling
`super().schedule()` from the subclass `_schedule_impl()` re-enters
`InstrumentedSchedulerMixin.schedule()`, which calls `self._schedule_impl()`
again and causes recursion.

### Snapshot behavior

The dry-run scheduler snapshots:

- `waiting`
- `skipped_waiting`
- `running`

It reads request fields and scheduler capacity fields, constructs LP request
snapshots, runs the LP planner, logs diagnostics, and then discards the plan.

It returns native scheduler output unchanged. Therefore Phase 12.4 does not
change scheduling behavior.

### JSONL logging

Phase 12.4 uses the existing instrumentation path and the existing scheduler
JSONL environment variable:

```text
SCHEDULER_POLICIES_ITER_LOG
```

It adds `lp_dry_run` records alongside the existing `scheduler_call` records.
The dry-run records are diagnostic only. They summarize snapshot, solver,
extraction, fallback, and timing information for the LP planner.

### Unsupported states

The dry-run bridge is conservative. It falls back for unsupported or advanced
states including:

- speculative decoding;
- async output placeholders;
- pooling;
- KV transfer;
- encoder or multimodal inputs;
- waiting for remote KVs;
- structured-output grammar waits;
- unknown request statuses.

Fallback means the LP dry-run record can note why the LP path was skipped, while
native vLLM scheduling still proceeds unchanged.

### Phase 12.4 boundaries

Phase 12.4 intentionally did not:

- edit native vLLM files;
- change scheduling behavior;
- mutate queues during dry-run snapshot collection;
- mutate request state during dry-run snapshot collection;
- mutate KV state during dry-run snapshot collection;
- call `allocate_slots()`;
- call `_preempt_request()`;
- construct `SchedulerOutput`;
- run server experiments;
- run benchmarks.

## Validation Evidence

Repo docs report that targeted tests passed for Phase 12.3 and Phase 12.4b.

The read-only Unity-local artifact audit found no independent `logs`, `results`,
or `tmp` artifacts corroborating Phase 12.3/12.4b validation outside the repo
docs.

External validation evidence for Phase 12.3/12.4b therefore remains unknown.

Do not cite Phase 12.3/12.4b validation as independently reproduced from Unity
run artifacts unless such artifacts are produced in a later phase.

## Current Outcome

Phase 12 established:

- a pure-Python primal LP helper layer;
- synthetic LP dataclasses and tests;
- LP relaxation solving with SciPy HiGHS;
- integral/fractional extraction;
- a dry-run scheduler class;
- live scheduler state snapshotting for simple states;
- JSONL dry-run diagnostics;
- behavior-preserving delegation back to native vLLM scheduling.

The implemented dry-run bridge is diagnostic. It can measure and inspect the LP
planner path, but it does not execute LP actions.

## Remaining Risks

The main remaining risks are:

1. Real vLLM snapshot collection beyond simple states.
   Advanced request modes may require more precise classification and fallback
   rules.

2. Exact KV-block cost estimation.
   The dry-run bridge uses conservative planning estimates. Exact incremental
   KV allocation depends on native block-boundary behavior, prefix caching,
   hybrid KV groups, and `allocate_slots()`.

3. Decode eligibility for advanced modes.
   Decode readiness is derived, not stored as a single flag, and advanced modes
   such as speculative decoding or async placeholders complicate the derivation.

4. Solver overhead in the scheduler loop.
   The dry-run bridge can log LP diagnostics, but server-side LP overhead has
   not yet been independently measured with Unity-local artifacts.

5. Real action translation remains the highest-risk future phase.
   Translating `LPActionPlan` into native scheduling behavior requires correct
   queue updates, request-state updates, KV allocation, preemption handling, and
   valid `SchedulerOutput` construction.

6. Preemption remains high risk.
   Any real preemption policy must preserve vLLM recomputation-preemption
   invariants and should use native mechanisms such as `_preempt_request()` only
   in a carefully validated implementation phase.

## Recommended Next Steps

1. Keep `docs/project_status.md` as the concise current-state source of truth.
2. Use `docs/scheduler_experiment_reference.md` for operational experiment
   commands.
3. Run a Phase 12.4c/12.5 server experiment for `PrimalLPDryRunScheduler` to
   measure dry-run LP overhead before attempting real action translation.
4. Only after dry-run overhead and fallback behavior are understood, design a
   separate action-translation phase.
