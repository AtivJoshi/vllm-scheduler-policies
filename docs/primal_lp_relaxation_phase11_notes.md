# Primal LP Relaxation Phase 11 Notes

## Purpose

This document records Phase 11 assumptions, resolved ambiguities, and follow-up
questions for the future implementation of the complete LP-relaxation scheduler
from the LaTeX section:

```text
Primal Heuristic 1: Approximation via LP Relaxation
\label{subsec:lp_relaxation}
```

This file is documentation only. It does not implement or modify scheduler
behavior. Phase 12.1 extended these notes with read-only vLLM scheduler
inspection results.

## Scope decision

The original experiment plan suggested starting from a deliberately simple
scheduler bridge. That suggestion is superseded for this phase.

The future implementation target is now the complete algorithm in
`\label{subsec:lp_relaxation}`, including:

- `SolveLPRelaxation`,

- construction of the relaxed LP,

- SciPy-based LP solving,

- integral/fractional request partitioning,

- `ExtractFractionals`,

- preemption-first memory recovery,

- forced safety preemption,

- admission and fluid chunking,

- final action translation into vLLM scheduling actions.


Phase 11 remains documentation/specification only.

## Resolved assumptions

### Request universe

Use:

```text
U_t = all requests currently present in the system, except requests that have
      already been fully processed.
```

This includes waiting, partially-prefilled, running/decode-ready, and
preemptible requests.

Phase 12.1 verified the concrete containers:

- `Scheduler.waiting` and `Scheduler.skipped_waiting` are `RequestQueue`
  instances,

- `Scheduler.running` is `list[Request]`,

- partial prefill is represented by `Request` progress, especially
  `num_computed_tokens`,

- `Request.request_id` is the stable key,

- `Request.arrival_time` is available.

### Utility weights

The utility weights are inputs to the relaxed LP.

They are computed by the scheduler before constructing the LP:

```text
alpha_i(t) = decode utility
beta_i(t)  = prefill utility
gamma_i(t) = preemption penalty
```

These weights are the main scheduler policy knobs.

A native-vLLM-like behavior can in principle be synthesized by:

```text
alpha_i(t) -> infinity
```

for absolute decode dominance,

```text
beta_i(t) = 1 + c * (t - t_i_arrive)
```

for FCFS tie-breaking among new/prefill requests, and

```text
gamma_i(t) = 1e9 * (t - t_i_arrive)
```

for a massive LIFO-like preemption penalty.

Implementation note: actual numerical solvers cannot use infinity. A future
implementation must use a sufficiently large finite value, normalized weights,
or a lexicographic objective.

### Memory model

Phase 12.1 resolved the vLLM integration memory unit:

- use KV-cache blocks inside the LP, not bytes and not raw per-token memory,

- map `M_t_free` to
  `self.kv_cache_manager.block_pool.get_num_free_blocks()`,

- map `W_t` to a scheduler-local conservative reserve named
  `lp_memory_reserve_blocks`,

- do not treat `gpu_memory_utilization` as `W_t`.

`gpu_memory_utilization` helps determine total KV-cache capacity before
scheduling. `lp_memory_reserve_blocks` is an additional scheduler-local reserve
inside the LP planning problem.

For the LP:

- `c_i^P` should be a conservative estimate of incremental KV blocks needed
  for the candidate prefill chunk,

- `c_i^D` should be a conservative estimate of incremental KV blocks needed
  for one decode step,

- `c_i^Z` should be the currently allocated KV blocks recoverable by
  recomputation preemption.

The LP memory constraint remains a planning approximation. Final vLLM action
translation must still call `allocate_slots()` or the native allocation path
and handle allocation failure safely. The LP solution is not proof of actual
KV feasibility.

### Memory capacity and safety watermark

Use:

```text
M_t_free = current free KV-cache blocks
W_t      = lp_memory_reserve_blocks
```

No generic scheduler memory watermark was found beyond
`scheduler_reserve_full_isl`, which is an admission safety check. Introduce
`lp_memory_reserve_blocks` as a policy-level conservative reserve for LP
planning.

### Sequence budget

Use:

```text
S_max = max_num_seqs
```

Phase 12.1 verified that this maps to `self.max_num_running_reqs` /
`scheduler_config.max_num_seqs` semantics.

Do not add a separate `max_num_partial_prefills` LP constraint in the first
implementation. The inspected `schedule()` path visibly uses token budget and
`long_prefill_token_threshold`; `max_num_partial_prefills` was not clearly
enforced there. Revisit it later after the first LP path is stable.

### Token budget

Use:

```text
B_max = vLLM per-step token budget
```

Phase 12.1 verified this maps to `self.max_num_scheduled_tokens`.

### Decode readiness

There is no explicit decode-ready field in the inspected scheduler path.

Decode eligibility must be derived by a future helper:

```text
_classify_request_action_space()
```

That helper should use request status, `num_computed_tokens`,
`num_prompt_tokens`, the native `request.num_tokens` /
`request.num_tokens_with_spec` scheduled-work formula, and simple generative
request assumptions.

A naive `P_i_rem == 0` rule is insufficient for vLLM. For the first
implementation, support only simple generative requests. Speculative decoding,
async placeholders, pooling, KV transfer edge cases, multimodal encoder
complications, and other advanced states should be unsupported on the LP path
initially and should trigger fallback/delegation.

### Preemption model

Assume preempted sequences are deleted and later recomputed.

Do not implement swapping in the first LP-relaxation scheduler.

### LP solver

Assume SciPy for the first implementation attempt.

If SciPy is unavailable, difficult to package, or too slow inside the scheduler
loop, revisit solver choices later.

## Major implementation risks

### 1. Action translation is the main engineering risk

The LP produces abstract decisions:

```text
hat_x_i
hat_y_i
hat_z_i
hat_I_i^P
```

The future implementation must translate these into the exact data structures
returned by vLLM's scheduler path.

This requires verifying:

- how vLLM represents prefill scheduling,

- how vLLM represents chunked prefill length,

- how vLLM represents decode scheduling,

- how vLLM performs recomputation preemption,

- how vLLM updates request queues and KV-cache state,

- what `Scheduler.schedule()` returns in vLLM v0.20.2.


Phase 12.1 resolved that `_translate_lp_actions()` remains the highest-risk
integration step. LP action plans cannot be returned directly as scheduler
outputs.

A valid `SchedulerOutput` must match native vLLM construction, including
`scheduled_new_reqs`, `scheduled_cached_reqs`, `num_scheduled_tokens`, block
IDs, connector metadata, zeroing IDs, finished/preempted request IDs, and
post-schedule state updates.

Future implementation should first produce an internal `LPActionPlan` and
synthetic tests, then a dry-run scheduler that logs the LP plan while
delegating to default behavior. Real action translation should come after that.
Real preemption should come last.

### 2. Preemption is high-risk

The LP includes explicit preemption through `z_i`.

Future implementation must ensure that preemption preserves:

- request correctness,

- queue consistency,

- KV-cache consistency,

- recomputation behavior,

- metrics/logging consistency,

- compatibility with vLLM's expected scheduler output.


The first LP scheduler should not implement swap-based preemption. Phase 12.1
verified the recomputation path as `_preempt_request(request, timestamp)`, with
the native caller responsible for removing the request from `running` before
calling it.

### 3. Memory accounting is approximate

The first implementation target uses block-count planning and still simplifies
the true vLLM allocator behavior.

This simplification may be mathematically useful but can disagree with vLLM's
actual KV allocation behavior. The future implementation must retain vLLM's
real memory-safety checks or fail safely if the block model over-admits.

### 4. Solver latency may be too high

The LP is solved every scheduler iteration.

The future implementation must measure:

- LP build time,

- LP solve time,

- extraction time,

- action translation time,

- total scheduler time.


If the LP solve dominates iteration latency, later phases may need warm starts,
specialized solvers, batching of solver calls, or a derived greedy equivalent.

### 5. Numerical scaling may matter

The native-vLLM-like weights include very large decode and preemption weights.

Large coefficients can cause numerical conditioning problems for SciPy.

The future implementation should consider:

- coefficient normalization,

- lexicographic optimization,

- bounded finite constants,

- diagnostic logging of solver status and objective value.


### 6. Almost-integrality depends on preserving the LP structure

The LaTeX argument expects at most three fractional requests because there are
three global coupling constraints.

The implementation should log the number of fractional requests. If more than
three appear, possible explanations include:

- numerical tolerance issues,

- extra global constraints were added,

- variable coupling differs from the LaTeX formulation,

- solver returned a non-basic optimal solution,

- implementation bug.


### 7. Arrival time may need policy-maintained metadata

The default-like utility formulas use `t_i_arrive`.

Phase 12.1 verified that vLLM exposes `Request.arrival_time`. Policy-maintained
first-seen metadata may still be useful for scheduler-iteration age, but it is
not required just to obtain an arrival timestamp.

That metadata must not alter request behavior.

### 8. Safe fallback is required

If the LP path sees unsupported request states, solver failure, allocation
failure, or uncertain action translation, it should fail safely.

The first integrated LP scheduler should delegate to default scheduling rather
than corrupt scheduler state.

## Items to inspect before implementation

Phase 12.1 verified:

1. waiting request container: `Scheduler.waiting` and
   `Scheduler.skipped_waiting`,

2. running request container: `Scheduler.running`,

3. partially-prefilled request representation: `Request.num_computed_tokens`
   progress,

4. request ID / stable key: `Request.request_id`,

5. arrival timestamp: `Request.arrival_time`,

6. token-budget object: `self.max_num_scheduled_tokens`,

7. `max_num_seqs` path: `self.max_num_running_reqs` /
   `scheduler_config.max_num_seqs`,

8. KV-cache free-capacity API:
   `self.kv_cache_manager.block_pool.get_num_free_blocks()`,

9. per-request KV allocation state: KV manager block mappings/helpers,

10. recomputation preemption path: `_preempt_request(request, timestamp)`,

11. chunked-prefill length controls: `num_new_tokens`, token budget,
    `long_prefill_token_threshold`, model length, encoder constraints, and
    alignment,

12. scheduler output type and invariants: `SchedulerOutput`.

Remaining design items:

- conservative decode classification for advanced states,

- block-cost estimation without mutating scheduler state,

- action translation that preserves every native `SchedulerOutput` invariant,

- safe allocation-failure behavior.


## Future implementation decomposition

A later implementation phase should split the work into small helpers.

Possible decomposition:

```text
_collect_lp_state()
_classify_request_action_space()
_compute_utility_weights()
_estimate_kv_block_costs()
_build_lp()
_solve_lp()
_partition_integral_fractional()
_extract_fractionals()
_build_lp_action_plan()
_translate_lp_actions()
_log_lp_metrics()
```

The top-level policy method should remain:

```text
_schedule_impl()
```

The outer timing/logging wrapper should remain:

```text
InstrumentedSchedulerMixin.schedule()
```

## Suggested later fallback behavior

Before implementation, define a safe fallback for LP failure.

Resolved fallback for the first integrated LP scheduler:

1. delegate to default vLLM scheduler behavior through the template-method
   fallback when the LP path is unsupported or unsafe,

2. log the fallback reason in scheduler-policy instrumentation,

3. do not return partial hand-built outputs unless action translation is known
   to be complete.

## Validation notes for later phases

When implementation begins, validate in this order:

1. import scheduler package,

2. `python -m compileall vllm_scheduler_policies`,

3. synthetic LP construction test,

4. synthetic LP solve test,

5. synthetic `ExtractFractionals` test,

6. no-vLLM `LPActionPlan` test,

7. dry-run scheduler that logs the LP plan while delegating to default
   behavior,

8. real action translation without preemption,

9. real recomputation preemption,

10. server startup with `--scheduler-cls`,

11. one `/v1/chat/completions` request,

12. tiny benchmark,

13. JSONL inspection,

14. comparison against default, passthrough, and `SimplePolicy1Scheduler`.


Do not run long benchmarks until basic correctness and safety are established.

## Phase 11 stopping point

Stop after creating and committing these documentation files.

Do not proceed to scheduler implementation in Phase 11.
