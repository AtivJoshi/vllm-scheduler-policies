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
behavior.

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

For the first complete-algorithm implementation target:

- use per-token memory,

- ignore block/page allocations,

- ignore KV block rounding effects.


This is a deliberate modeling simplification.

### Memory capacity and safety watermark

Use:

```text
M_t_free = current available KV-cache capacity
W_t      = vLLM memory safety margin / watermark
```

If vLLM exposes an internal safety margin, use it. Otherwise, introduce a
policy-level conservative watermark.

### Sequence budget

Use:

```text
S_max = max_num_seqs
```

Exact config access path must be verified before implementation.

### Token budget

Use:

```text
B_max = vLLM per-step token budget
```

Exact scheduler-budget object/field must be verified before implementation.

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


No method names should be invented before code inspection.

### 2. Preemption is high-risk

The LP includes explicit preemption through `z_i`.

Future implementation must ensure that preemption preserves:

- request correctness,

- queue consistency,

- KV-cache consistency,

- recomputation behavior,

- metrics/logging consistency,

- compatibility with vLLM's expected scheduler output.


The first LP scheduler should not implement swap-based preemption.

### 3. Memory accounting is approximate

The first implementation target uses per-token memory and ignores block/page
allocation.

This simplification may be mathematically useful but can disagree with vLLM's
actual KV allocation behavior. The future implementation must retain vLLM's
real memory-safety checks or fail safely if the scalar model over-admits.

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

If vLLM does not expose a suitable arrival timestamp or arrival order, the
future scheduler may need to maintain first-seen metadata keyed by request ID.

That metadata must not alter request behavior.

## Items to inspect before implementation

Before coding the LP scheduler, inspect the vLLM v0.20.2 scheduler path for:

1. waiting request container,

2. running request container,

3. partially-prefilled request representation,

4. finished request removal path,

5. request ID / stable key,

6. arrival timestamp or arrival order,

7. prompt length and prompt-progress fields,

8. decode-ready condition,

9. token-budget object and exact units,

10. `max_num_seqs` config access,

11. KV-cache free-capacity API,

12. memory safety margin / watermark,

13. per-request KV allocation state,

14. recomputation preemption path,

15. chunked-prefill length selection,

16. scheduler output type and invariants.


## Future implementation decomposition

A later implementation phase should split the work into small helpers.

Possible decomposition:

```text
_collect_lp_state()
_compute_utility_weights()
_build_lp()
_solve_lp()
_partition_integral_fractional()
_extract_fractionals()
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

Potential fallback choices:

1. delegate to default vLLM scheduler behavior,

2. delegate to `super().schedule()` through the template-method fallback,

3. return no new admissions but preserve running decode behavior,

4. fail closed with a clear diagnostic in development mode.


The fallback choice should be decided before writing scheduler code.

## Validation notes for later phases

When implementation begins, validate in this order:

1. import scheduler package,

2. `python -m compileall vllm_scheduler_policies`,

3. synthetic LP construction test,

4. synthetic LP solve test,

5. synthetic `ExtractFractionals` test,

6. no-vLLM dry-run action-plan test,

7. server startup with `--scheduler-cls`,

8. one `/v1/chat/completions` request,

9. tiny benchmark,

10. JSONL inspection,

11. comparison against default, passthrough, and `SimplePolicy1Scheduler`.


Do not run long benchmarks until basic correctness and safety are established.

## Phase 11 stopping point

Stop after creating and committing these documentation files.

Do not proceed to scheduler implementation in Phase 11.