# Primal LP Relaxation Scheduler Bridge

## Purpose

This document translates the candidate algorithm from the LaTeX section
`Primal Heuristic 1: Approximation via LP Relaxation`
(`\label{subsec:lp_relaxation}`) into an implementation-oriented plan for a
future vLLM scheduler policy.

The future implementation target is the complete LP-relaxation algorithm in
that section, including:

- `SolveLPRelaxation`
- integral/fractional variable partitioning
- the almost-integral extraction logic
- `ExtractFractionals`
- preemption-first memory recovery
- admission and fluid chunking

This document is not an implementation. Phase 11 is documentation and
specification only.

## Intended future scheduler class

The future scheduler should likely be a new policy class, for example:

```text
PrimalLPRelaxationScheduler
```

It should use the Phase 10.5 template-method workflow:

```text
InstrumentedSchedulerMixin.schedule()
  -> timing/logging wrapper
  -> calls self._schedule_impl()
```

Therefore the future policy should override:

```text
_schedule_impl()
```

It should not override:

```text
schedule()
```

`SimplePolicy1Scheduler` remains the current behavior-preserving instrumented
passthrough baseline. The LP-relaxation scheduler should be a new scheduler
policy class rather than a mutation of `SimplePolicy1Scheduler`.

## High-level scheduler goal

At each scheduler step `t`, choose a feasible one-step action for every
non-finished request in the system.

For each request `i`, the possible actions are:

1. prefill some number of prompt tokens,

2. decode one token,

3. preempt the request,

4. do nothing.


The future scheduler should formulate this as the LP relaxation of the myopic
ILP in the LaTeX section, solve the relaxed LP, then convert the relaxed
solution into an integer vLLM scheduling decision using the deterministic
fractional extraction algorithm.

## Request universe

For this project, use the following interpretation:

```text
U_t = all requests currently present in the system, except requests that have
      already been fully processed.
```

Therefore `U_t` includes, where applicable:

- waiting requests,

- partially-prefilled requests,

- running/decode-ready requests,

- requests currently resident in GPU KV cache,

- requests that are eligible for preemption.


The future implementation must verify the exact vLLM scheduler data structures
that contain these request states.

## Mathematical decision variables

For each request `i in U_t`, the LP uses:

|Variable|Meaning|Final integer interpretation|
|---|---|---|
|`x_i(t)`|number of prefill tokens assigned to request `i` in this step|schedule `x_i` prompt tokens|
|`y_i(t)`|decode indicator|schedule one decode token|
|`z_i(t)`|preemption indicator|preempt/delete request state so it must be recomputed later|
|`I_i^P(t)`|prefill admission indicator|request receives a prefill chunk this step|

The relaxed LP allows:

```text
y_i(t), z_i(t), I_i^P(t) in [0, 1]
```

The prefill amount `x_i(t)` is continuous and bounded by the remaining prompt
tokens and chunk size.

## Objective

The LP maximizes:

```text
sum_i alpha_i(t) * y_i(t)
    + beta_i(t)  * x_i(t)
    - gamma_i(t) * z_i(t)
```

Implementation interpretation:

|Quantity|Meaning|
|---|---|
|`alpha_i(t)`|marginal utility of decoding request `i`|
|`beta_i(t)`|marginal utility of prefilling one token for request `i`|
|`gamma_i(t)`|penalty for preempting request `i`|

The weights `alpha_i(t)`, `beta_i(t)`, and `gamma_i(t)` are scheduler-computed
inputs to the LP. They are among the most important design choices because they
determine scheduler behavior.

A native-vLLM-like policy can be approximated in this framework by:

```text
alpha_i(t) -> very large value
```

to represent decode dominance over prefill,

```text
beta_i(t) = 1 + c * (t - t_i_arrive)
```

to represent FCFS tie-breaking for new/prefill requests, and

```text
gamma_i(t) = 1e9 * (t - t_i_arrive)
```

to represent a massive LIFO-like preemption penalty.

Because numerical LP solvers cannot use actual infinity, `alpha_i(t) -> infinity`
must be implemented either as a sufficiently large finite value or as a
lexicographic objective in a later implementation.

## Constraints

The future scheduler should construct the LP with the same constraints as the
LaTeX formulation.

### Token budget

```text
sum_i (x_i(t) + y_i(t)) <= B_max
```

Implementation interpretation:

```text
B_max maps to vLLM's per-step token budget.
```

This should correspond to the budget controlling how many total prefill and
decode tokens can be scheduled in one scheduler step.

### Sequence/concurrency budget

```text
sum_i (I_i^P(t) + y_i(t)) <= S_max
```

Implementation interpretation:

```text
S_max maps to max_num_seqs.
```

This limits how many request/sequence actions can be scheduled in the step.

### Memory budget

```text
sum_i (c_i^P x_i(t) + c_i^D y_i(t) - c_i^Z z_i(t))
    <= M_t_free - W_t
```

Implementation interpretation:

```text
M_t_free = current available KV-cache capacity
W_t      = vLLM memory safety margin / watermark, if available
```

For the first complete-algorithm implementation target, use a per-token memory
model and ignore block/page rounding.

### Chunked prefill

```text
x_i(t) <= min(P_i_rem(t), C_max) * I_i^P(t)
```

Implementation interpretation:

- `P_i_rem(t)` is the number of un-prefilled prompt tokens remaining.

- `C_max` is the maximum prefill chunk size for one forward pass.

- If `I_i^P(t) = 0`, then `x_i(t) = 0`.

- If `I_i^P(t) = 1`, then `x_i(t)` may schedule up to the smaller of remaining
    prompt length and maximum chunk size.


### Decode causality

```text
y_i(t) <= 1{P_i_rem(t) = 0}
```

Implementation interpretation:

A request may decode only after its prompt has been fully prefilled.

### Mutual exclusion

```text
I_i^P(t) + y_i(t) + z_i(t) <= 1
```

Implementation interpretation:

Each request receives at most one action per scheduler step:

- prefill,

- decode,

- preempt,

- or do nothing.


## LP solver plan

For the first implementation attempt, assume SciPy is available and use SciPy's
linear programming support to solve the relaxed LP.

The future implementation should isolate LP construction and solving behind a
small internal helper so that the solver can be replaced later if SciPy is
unavailable, too slow, or too difficult to package on Unity.

The implementation should measure LP overhead explicitly, including:

- LP input construction time,

- solver time,

- solver status,

- number of LP variables,

- number of LP constraints,

- number of requests in `U_t`,

- number of fractional requests.


## SolveLPRelaxation implementation plan

The future `_schedule_impl()` should conceptually perform the following steps.

### 1. Snapshot scheduler state

Collect all non-finished requests in `U_t`.

For each request, derive or approximate:

|Quantity|Meaning|
|---|---|
|`P_i_rem(t)`|remaining un-prefilled prompt tokens|
|`c_i^P`|memory consumed per prefill token|
|`c_i^D`|memory consumed by one decode token|
|`c_i^Z`|memory recovered if the request is preempted|
|`alpha_i(t)`|decode utility|
|`beta_i(t)`|prefill utility|
|`gamma_i(t)`|preemption penalty|
|arrival time / age|used by default-like utility weights|
|prefill-complete flag|used for decode causality|
|preemptible flag|used to decide whether `z_i` may be nonzero|

Also collect global capacities:

|Quantity|Implementation meaning|
|---|---|
|`B_max`|vLLM per-step token budget|
|`S_max`|`max_num_seqs`|
|`M_t_free`|current available KV-cache capacity|
|`W_t`|memory safety watermark|
|`C_max`|maximum prefill chunk size|

### 2. Build the relaxed LP

Create continuous variables for each request:

```text
x_i >= 0
0 <= y_i <= 1
0 <= z_i <= 1
0 <= I_i^P <= 1
```

Add the objective and constraints described above.

### 3. Solve the LP

Call the SciPy solver and obtain:

```text
tilde_x_i
tilde_y_i
tilde_z_i
tilde_I_i^P
```

The solver output is the continuous relaxation, not yet a valid vLLM action
plan.

### 4. Partition integral and fractional requests

For every request `i`:

```text
if tilde_y_i, tilde_z_i, and tilde_I_i^P are all integral:
    i in U_int
else:
    i in U_frac
```

Use a numerical tolerance rather than exact floating-point equality. For
example, a future implementation may treat values within `1e-6` of `0` or `1`
as integral.

The LaTeX algorithm relies on the almost-integral property: because the LP has
three global coupling constraints and all other constraints are local to each
request, at most three requests should be fractional in the ideal mathematical
formulation.

The implementation should log the actual size of `U_frac`. If more than three
fractional requests appear, that should be logged as a warning or diagnostic,
because it may indicate numerical tolerance issues or a mismatch between the
implemented LP and the theoretical formulation.

### 5. Lock integral assignments

For every `i in U_int`:

```text
hat_y_i   = tilde_y_i
hat_z_i   = tilde_z_i
hat_I_i^P = tilde_I_i^P
hat_x_i   = floor(tilde_x_i)
```

Then compute residual capacities:

```text
B_rem = B_max - sum_{i in U_int} (hat_x_i + hat_y_i)

S_rem = S_max - sum_{i in U_int} (hat_I_i^P + hat_y_i)

M_rem = (M_t_free - W_t)
        - sum_{i in U_int}
          (c_i^P * hat_x_i + c_i^D * hat_y_i - c_i^Z * hat_z_i)
```

These residual capacities are passed into fractional extraction.

## ExtractFractionals implementation plan

`ExtractFractionals` converts the small fractional set into a feasible integer
action plan.

### Phase 1: preemption-first memory resolution

For each `i in U_frac`, compute:

```text
DominantAction = argmax(tilde_y_i, tilde_I_i^P, tilde_z_i)
```

If the dominant action is preemption:

```text
hat_z_i = 1
M_curr += c_i^Z
```

This secures memory before evaluating additional admissions.

### Phase 2: forced safety preemption if needed

After dominant fractional preemptions have been selected, check:

```text
if M_curr < 0:
    choose k with largest tilde_z_k among fractional requests not already preempted
    hat_z_k = 1
    M_curr += c_k^Z
```

This is a conservative safety fallback for cases where the relaxed LP relied on
some fractional preemption to satisfy memory feasibility.

### Phase 3: admission and fluid chunking

For each remaining fractional request that was not preempted:

```text
DominantAction = argmax(tilde_y_i, tilde_I_i^P)
```

If decode is dominant and capacity remains:

```text
if B_curr >= 1 and M_curr >= c_i^D:
    hat_y_i = 1
    B_curr -= 1
    M_curr -= c_i^D
```

If prefill is dominant:

```text
hat_x_i = min(
    P_i_rem,
    C_max,
    B_curr,
    floor(M_curr / c_i^P)
)

if hat_x_i > 0:
    hat_I_i^P = 1
    B_curr -= hat_x_i
    M_curr -= c_i^P * hat_x_i
```

This is the fluid chunking rule. It permits the final prefill chunk to be
truncated to fit the remaining token and memory budgets.

## Translation from LP actions to vLLM scheduling actions

After extraction, the future scheduler has integer decisions:

```text
hat_x_i
hat_y_i
hat_z_i
hat_I_i^P
```

The implementation must translate them to vLLM actions:

|LP output|Intended vLLM action|
|---|---|
|`hat_x_i > 0` and `hat_I_i^P = 1`|schedule request `i` for prefill/chunked prefill with `hat_x_i` tokens|
|`hat_y_i = 1`|schedule request `i` for one decode token|
|`hat_z_i = 1`|preempt/delete request `i`; assume recomputation later, no swapping|
|all zero|do not schedule request `i` in this step|

The exact vLLM methods and fields for action translation must be verified
before coding. Do not invent method names in the implementation.

## Preemption model

For now, preempted sequences are assumed to be deleted and recomputed.

Do not implement swapping in the first LP-relaxation scheduler.

This means `z_i = 1` should correspond to a recomputation-style preemption,
where the request may later be admitted again and its KV state must be rebuilt.

## Memory model

For the first complete-algorithm implementation target:

- use per-token memory,

- ignore block/page allocation effects,

- treat `c_i^P`, `c_i^D`, and `c_i^Z` as scalar memory coefficients,

- map `M_t_free` to available KV-cache capacity,

- map `W_t` to vLLM's memory safety margin if one is available.


This is a deliberate approximation. Later implementations may replace this
with block-aware accounting.

## Default vLLM behavior to preserve where possible

Even though this scheduler targets the complete LP-relaxation algorithm, it
must preserve vLLM correctness and safety invariants.

The future implementation must preserve:

- valid request state transitions,

- decode only after prefill completion,

- KV-cache memory safety,

- token-budget safety,

- max sequence/concurrency safety,

- output correctness,

- existing scheduler instrumentation wrapper behavior,

- server compatibility with `--scheduler-cls`,

- no modification of native vLLM scheduler files unless explicitly approved.


The scheduler package's current instrumentation environment variable remains:

```text
SCHEDULER_POLICIES_ITER_LOG=/path/to/scheduler_iter.jsonl
```

Do not use the old environment variable:

```text
VLLM_SCHEDULER_ITER_LOG
```

## Instrumentation to use or add later

The existing `InstrumentedSchedulerMixin` should remain the outer timing and
logging wrapper.

Future LP-specific instrumentation should add fields such as:

|Field|Meaning|
|---|---|
|`lp_enabled`|whether LP path ran|
|`lp_num_requests`|size of `U_t`|
|`lp_num_variables`|number of LP variables|
|`lp_num_constraints`|number of LP constraints|
|`lp_build_time_ms`|time to build LP inputs|
|`lp_solve_time_ms`|SciPy solver time|
|`lp_extract_time_ms`|integral/fractional extraction time|
|`lp_translate_time_ms`|time to translate LP output into vLLM actions|
|`lp_solver_status`|solver success/failure/status code|
|`lp_objective_value`|relaxed LP objective value|
|`lp_num_integral_requests`|size of `U_int`|
|`lp_num_fractional_requests`|size of `U_frac`|
|`lp_fractional_rule_violation`|whether more than three fractional requests appeared|
|`lp_num_preemptions`|number of chosen `hat_z_i = 1` actions|
|`lp_num_forced_preemptions`|number of safety-forced preemptions|
|`lp_num_decode_actions`|number of chosen `hat_y_i = 1` actions|
|`lp_num_prefill_actions`|number of chosen `hat_I_i^P = 1` actions|
|`lp_num_prefill_tokens`|total chosen prefill tokens|
|`lp_memory_remaining`|final memory residual after extraction|
|`lp_token_budget_remaining`|final token budget residual|
|`lp_sequence_budget_remaining`|final sequence budget residual|

Scheduler overhead should continue to be measured with lightweight JSONL
records around each scheduler call.

## Out of scope for Phase 11

Phase 11 does not include:

- implementing the scheduler,

- adding a new scheduler class,

- adding SciPy to dependencies,

- modifying native vLLM scheduler files,

- modifying CUDA/C++/Triton code,

- running a server,

- running benchmarks,

- changing scheduler behavior,

- asking Codex to implement the algorithm.


## Out of scope for the first full-algorithm implementation attempt

The first implementation attempt should still avoid:

- block/page-aware memory accounting,

- swapping-based preemption,

- multi-step lookahead optimization,

- asynchronous/background LP solving,

- changes to native vLLM scheduler source files,

- non-LP primal heuristics from later LaTeX sections,

- algorithms after `Primal Heuristic 1`.


## Future validation plan

When implementation begins in a later phase, validate progressively:

1. import test for the scheduler package,

2. compile test,

3. unit-style LP construction test on synthetic requests,

4. unit-style fractional extraction test,

5. action translation dry-run test if feasible,

6. server startup with `--scheduler-cls`,

7. one curl request,

8. tiny benchmark,

9. scheduler JSONL inspection,

10. comparison against:

    - default vLLM scheduler,

    - `BaselinePassthroughScheduler`,

    - `SimplePolicy1Scheduler`.


Do not use profiling or long benchmarks until the tiny correctness path works.