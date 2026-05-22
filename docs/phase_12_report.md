# Phase 12 Report: Primal LP Relaxation Scheduler Implementation Planning

## Context

Phase 12 begins after the Phase 11 documentation commit:

```text
9805e98 Add scheduler v1 implementation bridge docs
```

Phase 11 created the initial bridge documentation for the complete LP-relaxation  
algorithm from the LaTeX section:

```text
Primal Heuristic 1: Approximation via LP Relaxation
\label{subsec:lp_relaxation}
```

The project has now intentionally diverged from the original  
`vllm_unity_experiment_plan_fresh.md` after Phase 11. The original plan  
recommended implementing simple schedulers first, but the current target is to  
work toward the complete LP-relaxation scheduler. The implementation will still  
be staged carefully to avoid unsafe edits to vLLM scheduler internals.

The guiding rule remains:

```text
Future scheduler policies should override _schedule_impl(), not schedule().
InstrumentedSchedulerMixin.schedule() remains the outer timing/logging wrapper.
```

## Phase 12.1: Read-only vLLM scheduler mapping

### Goal

The goal of Phase 12.1 was to use Codex in read-only mode to inspect the pinned  
vLLM v0.20.2 scheduler implementation and verify the concrete vLLM objects,  
fields, methods, and invariants needed for the future LP-relaxation scheduler.

No code was edited in this phase.

### Files and areas inspected

Codex inspected the external scheduler package docs and the native vLLM scheduler  
path, especially the vLLM v1 scheduler implementation under:

```text
~/vllm-sched/vllm/vllm/v1/core/sched/
```

The inspection focused on:

- request queues,
    
- running-request state,
    
- partial prefill representation,
    
- request identity and arrival metadata,
    
- prompt-progress accounting,
    
- decode eligibility,
    
- token and sequence budgets,
    
- KV-cache capacity accounting,
    
- preemption behavior,
    
- chunked prefill behavior,
    
- `SchedulerOutput` construction and invariants.
    

### Verified vLLM mappings

Codex verified the following important mappings.

|Need|Verified vLLM mapping|
|---|---|
|Waiting request container|`Scheduler.waiting` and `Scheduler.skipped_waiting`, both `RequestQueue` instances|
|Running request container|`Scheduler.running`, a `list[Request]`|
|Partially-prefilled request|No separate object; represented by request progress, especially `request.num_computed_tokens`|
|Stable request key|`Request.request_id`; scheduler global request map is `Scheduler.requests`|
|Arrival time/order|`Request.arrival_time`; priority ordering uses priority, arrival time, and request id|
|Pure prompt remaining tokens|`max(request.num_prompt_tokens - request.num_computed_tokens, 0)`|
|Native resumed/waiting scheduling progress|Native scheduler may use `request.num_tokens - request.num_computed_tokens`, because resumed requests may include output tokens|
|Decode-ready state|No explicit stored flag; decode eligibility is derived from request status/progress and native scheduling formulas|
|Token budget|`token_budget = self.max_num_scheduled_tokens`; units are scheduled tokens per scheduler iteration|
|`max_num_seqs` mapping|`scheduler_config.max_num_seqs`, copied into `self.max_num_running_reqs`|
|Free KV capacity|Block-based, via `self.kv_cache_manager.block_pool.get_num_free_blocks()`|
|Allocation safety|Native allocation safety is enforced by `allocate_slots()` returning `None` on failure|
|Generic memory watermark|No verified generic scheduler watermark beyond `scheduler_reserve_full_isl`|
|Per-request KV allocation|KV manager tracks request-to-block mappings; helper methods expose block IDs/blocks|
|Recomputation preemption|`_preempt_request(request, timestamp)` frees KV/encoder cache, sets status to `PREEMPTED`, resets computed tokens, clears spec tokens, increments preemption count, and prepends the request to waiting|
|Chunked prefill length|Controlled by `num_new_tokens`, token budget, `long_prefill_token_threshold`, model length, encoder constraints, and possibly alignment|
|Scheduler return type|`Scheduler.schedule()` returns `SchedulerOutput`; output construction has strict invariants|

### Phase 12.1 conclusions

The inspection showed that the LP algorithm and the vLLM action-translation  
problem must be treated separately.

The LP can produce an abstract action plan:

```text
hat_x_i, hat_y_i, hat_z_i, hat_I_i^P
```

but vLLM requires a valid `SchedulerOutput` with correct request lists, block  
IDs, connector metadata, zeroing IDs, KV-cache updates, and post-schedule state  
updates. Therefore, action translation is the highest-risk part of the future  
implementation.

### Phase 12.1 unknowns and risks

The main risks identified were:

1. **Memory units do not match the original scalar LP model.**  
    vLLM allocates KV cache in blocks, not scalar per-token memory. A future LP  
    scheduler should use block-based estimates and still call native allocation  
    checks.
    
2. **Hybrid KV-cache behavior may make memory coefficients request-dependent.**  
    A naive per-token memory coefficient is not sufficient for exact vLLM  
    feasibility.
    
3. **Decode readiness is derived, not stored.**  
    A simple `P_rem == 0` rule is insufficient because vLLM scheduling may involve  
    output tokens, placeholders, speculative decoding, resumed requests, and other  
    states.
    
4. **Advanced request modes complicate the first implementation.**  
    Speculative decoding, async placeholders, pooling, KV transfer states,  
    multimodal encoder constraints, and Mamba alignment should be excluded from  
    the first LP path or cause fallback.
    
5. **Preemption is high-risk.**  
    Native recomputation preemption does more than remove a request; it updates  
    KV state, encoder cache, request status, computed-token progress, spec-token  
    state, metrics, and queue position.
    
6. **Chunk sizing can be reduced to zero by native constraints.**  
    Encoder constraints, model length, token budget, and alignment can shrink a  
    requested LP prefill action.
    
7. **`SchedulerOutput` construction is nontrivial.**  
    A valid output requires matching native scheduler invariants and post-schedule  
    bookkeeping.
    

## Phase 12.2: Documentation update plan after verified mapping

### Goal

The goal of Phase 12.2 is to update the Phase 11 bridge documentation using the  
verified vLLM mappings from Phase 12.1 and the resolved risk decisions below.

This phase should remain documentation-only.

No scheduler implementation should happen in Phase 12.2.

### Files to update

Update the existing bridge docs:

```text
docs/primal_lp_relaxation_scheduler.md
docs/primal_lp_relaxation_latex_to_code_mapping.md
docs/primal_lp_relaxation_phase11_notes.md
```

The docs should be updated to distinguish:

1. the mathematical LP formulation,
    
2. the block-based vLLM feasibility model,
    
3. the pure LP helper layer,
    
4. the later dry-run scheduler integration,
    
5. the final high-risk action-translation layer.
    

### Resolved design decisions for Phase 12.2

#### 1. Memory units

The future vLLM integration should use KV blocks as the LP memory unit.

Use:

```text
M_t_free = self.kv_cache_manager.block_pool.get_num_free_blocks()
W_t      = lp_memory_reserve_blocks
```

The LP coefficients should be interpreted as conservative block estimates:

```text
c_i^P = estimated incremental KV blocks for prefill
c_i^D = estimated incremental KV blocks for one decode step
c_i^Z = currently allocated KV blocks recoverable by recomputation preemption
```

The original per-token model remains useful for the mathematical report, but the  
implementation bridge should state that vLLM integration is block-aware.

#### 2. `gpu_memory_utilization` and watermark

`gpu_memory_utilization` should not be treated directly as `W_t`.

Instead:

```text
gpu_memory_utilization determines the total KV-cache pool available to vLLM.
W_t is an additional scheduler-local reserve used by the LP scheduler.
```

Use a future scheduler parameter such as:

```text
lp_memory_reserve_blocks
```

A reasonable initial default is either:

```text
lp_memory_reserve_blocks = 8
```

or, if total KV-block count is easily available:

```text
lp_memory_reserve_blocks = max(4, ceil(0.01 * total_kv_blocks))
```

The docs should mark this as a tunable safety parameter.

#### 3. Allocation safety

The LP memory constraint is only a planning approximation.

The final implementation must still use native vLLM allocation checks. If  
`allocate_slots()` returns `None`, the scheduler must shrink the candidate  
action, drop the action, or fall back safely.

Therefore, the scheduler needs two layers of memory safety:

```text
Layer 1: LP uses conservative KV-block estimates.
Layer 2: vLLM allocate_slots() remains the final feasibility authority.
```

#### 4. Decode eligibility

There is no explicit decode-ready flag.

The future implementation should define a helper:

```text
_classify_request_action_space(request)
```

For the first LP implementation, support only simple generative requests.

The helper should derive whether a request is eligible for:

- prefill,
    
- decode,
    
- preemption,
    
- no action.
    

Requests involving speculative decoding, async placeholders, pooling, KV  
transfer edge cases, multimodal encoder complications, or other advanced states  
should initially be treated as unsupported for the LP path and should trigger  
fallback/delegation.

#### 5. `max_num_partial_prefills`

Do not add a separate LP constraint for `max_num_partial_prefills` in the first  
implementation.

The inspected scheduler path visibly uses token budget and  
`long_prefill_token_threshold`; `max_num_partial_prefills` should be documented  
as a revisit-later item unless a direct enforcement point is verified.

#### 6. Action translation

Action translation remains the highest-risk implementation step.

The docs should explicitly state that an LP action plan cannot be returned  
directly. A valid vLLM result must match the native `SchedulerOutput`  
construction, including:

- `scheduled_new_reqs`,
    
- `scheduled_cached_reqs`,
    
- block IDs,
    
- connector metadata,
    
- zeroing IDs,
    
- request status updates,
    
- KV-cache updates,
    
- running/waiting queue updates,
    
- post-schedule invariants.
    

Therefore, `_translate_lp_actions()` should not be implemented until the pure LP  
helper layer and a dry-run LP scheduler have been validated.

### Proposed decomposition after Phase 12.2

The future implementation should be decomposed as:

```text
_collect_lp_state()
_classify_request_action_space()
_estimate_kv_costs()
_compute_utility_weights()
_build_lp()
_solve_lp()
_partition_integral_fractional()
_extract_fractionals()
_translate_lp_actions()
_validate_scheduler_output()
_log_lp_metrics()
```

However, the next coding phase should implement only the pure LP/helper parts,  
not `_translate_lp_actions()`.

### Phase 12.2 validation

After the documentation edits, run:

```bash
git diff --check
git diff --stat -- docs
grep -R -n \
  -e 'KV blocks' \
  -e 'lp_memory_reserve_blocks' \
  -e 'gpu_memory_utilization' \
  -e '_classify_request_action_space' \
  -e 'allocate_slots' \
  -e 'SchedulerOutput' \
  -e 'max_num_partial_prefills' \
  docs/primal_lp_relaxation_*.md
```

Then commit the documentation update:

```bash
git add docs/primal_lp_relaxation_*.md
git commit -m "Document verified vLLM LP scheduler mapping"
```
## Phase 12.3: Pure-Python LP helper layer

### Goal

The goal of Phase 12.3 was to implement only the pure-Python helper layer for the
primal LP-relaxation scheduler, using synthetic inputs and tests.

This phase intentionally did not integrate with the real vLLM scheduler. It did
not create a new scheduler class, did not implement `_translate_lp_actions()`,
did not construct `SchedulerOutput`, did not call `allocate_slots()`, and did
not call `_preempt_request()`.

The output of this phase is an abstract `LPActionPlan`, not a vLLM scheduling
result.

### Scope implemented

Phase 12.3 added a new helper package:

```text
vllm_scheduler_policies/primal_lp/
````

with the following files:

```text
vllm_scheduler_policies/primal_lp/__init__.py
vllm_scheduler_policies/primal_lp/types.py
vllm_scheduler_policies/primal_lp/weights.py
vllm_scheduler_policies/primal_lp/solver.py
vllm_scheduler_policies/primal_lp/extraction.py
```

It also added targeted synthetic tests:

```text
tests/test_primal_lp_solver.py
tests/test_primal_lp_extraction.py
```

The package dependency list was updated to include:

```text
scipy
```

because Phase 12.3 uses SciPy’s `linprog` implementation for the relaxed LP.

A small import-hygiene change was also made in:

```text
vllm_scheduler_policies/__init__.py
```

`BaselinePassthroughScheduler` is now exported lazily via `__getattr__`. This  
preserves the public root-package export while avoiding unnecessary vLLM/CUDA  
imports when importing the pure `primal_lp` helper package.

### Data model

Phase 12.3 introduced synthetic dataclasses for the LP layer.

The main request input type is:

```text
LPRequestSnapshot
```

It contains only scheduler-independent fields such as:

```text
request_id
remaining_prefill_tokens
max_prefill_chunk_tokens
decode_eligible
preemptible
prefill_eligible
prefill_memory_blocks_per_token
decode_memory_blocks
preempt_recoverable_blocks
arrival_time
```

The main global-capacity type is:

```text
LPCapacities
```

with fields for:

```text
token_budget
sequence_budget
free_memory_blocks
lp_memory_reserve_blocks
```

The helper layer treats memory as synthetic KV-block planning units. This keeps  
the helper aligned with the Phase 12.1 decision that vLLM memory integration  
should be block-based, while still acknowledging that final vLLM integration must  
use native allocation checks.

The relaxed LP solution is represented by:

```text
RelaxedLPSolution
```

The extracted integer plan is represented by:

```text
LPActionPlan
```

`LPActionPlan` explicitly remains an internal action plan and is not a valid  
vLLM `SchedulerOutput`.

### Utility weights

Phase 12.3 implemented utility-weight helpers for the LP objective:

```text
alpha_i(t) = decode utility
beta_i(t)  = prefill utility
gamma_i(t) = preemption penalty
```

The helper layer includes default-like weight construction for decode dominance,  
age-aware prefill priority, and preemption penalty behavior. These weights remain  
policy knobs rather than fixed scheduler semantics.

### LP formulation

The solver constructs the relaxed LP with four variables per request:

```text
x_i         = prefill tokens
y_i         = decode indicator
z_i         = preemption indicator
I_i^P       = prefill admission indicator
```

The variable ordering is:

```text
x         = 4 * offset
y         = 4 * offset + 1
z         = 4 * offset + 2
admission = 4 * offset + 3
```

Because SciPy solves minimization problems, the maximization objective

```text
sum_i alpha_i y_i + beta_i x_i - gamma_i z_i
```

is encoded with signs:

```text
c[x]         = -beta
c[y]         = -alpha
c[z]         =  gamma
c[admission] = 0
```

The implemented global constraints are:

```text
sum_i (x_i + y_i) <= token_budget

sum_i (I_i^P + y_i) <= sequence_budget

sum_i (
    prefill_memory_blocks_per_token_i * x_i
  + decode_memory_blocks_i * y_i
  - preempt_recoverable_blocks_i * z_i
) <= usable_memory_blocks
```

The implemented local constraints are:

```text
I_i^P + y_i + z_i <= 1

x_i <= max_prefill_tokens_this_step_i * I_i^P
```

Bounds enforce decode, prefill, and preemption eligibility:

```text
x_i       in [0, max_prefill_tokens_this_step_i]
y_i       in [0, 1] if decode-eligible, otherwise [0, 0]
z_i       in [0, 1] if preemptible, otherwise [0, 0]
I_i^P     in [0, 1] if prefill is possible, otherwise [0, 0]
```

The solver uses:

```text
scipy.optimize.linprog(..., method="highs")
```

Solver failure is handled by returning an empty action plan with solver status  
and message, rather than pretending a valid plan exists.

### Integral and fractional extraction

Phase 12.3 implemented tolerance-based partitioning into:

```text
U_int
U_frac
```

A request is treated as integral when its relaxed binary variables

```text
y_i, z_i, I_i^P
```

are all within tolerance of either 0 or 1.

Integral requests are locked first:

```text
hat_x_i = floor(tilde_x_i)
hat_y_i = rounded tilde_y_i
hat_z_i = rounded tilde_z_i
hat_I_i^P = rounded tilde_I_i^P
```

The implementation normalizes zero-token prefill admissions away in the final  
`LPActionPlan`, because the plan represents executable synthetic actions. Thus,  
if `hat_x_i = 0`, the extracted `hat_I_i^P` is set to 0 even if the relaxed  
solution had an integral admission value of 1.

The helper then computes residual token, sequence, and memory budgets before  
extracting fractional requests.

### `ExtractFractionals`

The fractional extraction logic follows the documented three-stage structure.

First, it performs preemption-first memory recovery. For each fractional request,  
it compares the relaxed action values and chooses the dominant action. The  
dominant-action tie-breaking is explicit and deterministic:

```text
preemption phase: preempt > decode > prefill
admission phase:  decode > prefill
```

If preemption is dominant and the request is marked preemptible, the plan sets:

```text
hat_z_i = 1
```

and adds the estimated recoverable KV blocks back to the residual memory budget.

Second, if residual memory is still negative, the helper performs forced safety  
preemption. It chooses a remaining preemptible fractional request with the  
largest relaxed preemption value.

Third, it performs decode or prefill admission for the remaining fractional  
requests. Decode is chosen when it dominates and token, sequence, and memory  
budgets allow it. Prefill is chosen otherwise when prefill is feasible, with the  
prefill chunk truncated by:

```text
remaining request prefill tokens
max prefill chunk size
remaining token budget
remaining synthetic memory budget
```

This implements the intended fluid-chunking behavior.

### Fractional diagnostic

Phase 12.3 added a diagnostic field:

```text
fractional_rule_violation
```

This is set when more than three requests remain fractional after solving the  
relaxed LP.

This does not by itself make the plan invalid, but it records a potential  
mismatch with the almost-integrality structure expected from the mathematical LP.  
Possible causes include numerical tolerance issues, solver behavior, or a future  
implementation accidentally adding extra coupling constraints.

### Tests added

Phase 12.3 added synthetic pytest coverage for the solver and extraction layer.

The tests cover:

```text
prefill-only scheduling
decode-dominance behavior
memory-limited prefill chunking
memory reserve reducing usable capacity
solver failure / infeasible fallback behavior
default-like utility weight helpers
binary-integrality tolerance
dominant preemption extraction
forced safety preemption
fractional prefill chunking
deterministic tie behavior
zero-token admission normalization
fractional_rule_violation when more than three requests are fractional
```

### Validation

The final validation commands were:

```bash
~/vllm-sched/.venv/bin/python -m compileall vllm_scheduler_policies tests
~/vllm-sched/.venv/bin/python -m pytest tests/test_primal_lp_solver.py tests/test_primal_lp_extraction.py -q
git diff --check
```

The targeted pytest suite passed:

```text
13 passed
```

Additional smoke checks verified that:

```text
import vllm_scheduler_policies.primal_lp
```

works without importing the full vLLM scheduler path, and that the lazy root  
export still resolves:

```text
from vllm_scheduler_policies import BaselinePassthroughScheduler
```

### Phase 12.3 boundaries respected

Phase 12.3 did not:

```text
modify native vLLM files
modify CUDA/C++/Triton code
create a real LP scheduler class
override schedule()
implement _translate_lp_actions()
construct SchedulerOutput
call allocate_slots()
call _preempt_request()
run a vLLM server
run curl checks
run benchmarks
change real scheduling behavior
```

This was important because the LP helper can produce only an abstract action  
plan. It still cannot safely mutate vLLM queues, KV-cache state, request state,  
or scheduler outputs.

### Remaining risks after Phase 12.3

The main remaining risks are unchanged:

1. **Real vLLM snapshot collection is still unimplemented.**  
    The helper currently uses synthetic `LPRequestSnapshot` objects. It has not  
    yet been connected to `Scheduler.waiting`, `Scheduler.skipped_waiting`, or  
    `Scheduler.running`.
    
2. **Decode eligibility remains conservative future work.**  
    The helper accepts a `decode_eligible` boolean but does not yet derive it from  
    real vLLM request status and token-progress fields.
    
3. **KV-block cost estimation remains approximate.**  
    Phase 12.3 represents memory costs synthetically. Future integration must  
    estimate prefill/decode/preemption block costs from real vLLM KV-cache state  
    without mutating it.
    
4. **The LP plan is not executable by vLLM.**  
    Final integration still requires safe translation into native scheduling  
    actions and a valid `SchedulerOutput`.
    
5. **Preemption remains high-risk.**  
    Phase 12.3 can plan preemption abstractly, but it does not perform real  
    recomputation preemption or queue updates.
    
6. **Solver overhead has not been measured in the scheduler loop.**  
    The helper uses SciPy `linprog`, but no per-iteration runtime impact has been  
    measured under vLLM scheduling.
    

### Phase 12.3 outcome

Phase 12.3 successfully established the standalone algorithmic core needed for  
the future scheduler:

```text
synthetic request/capacity snapshots
utility weights
SciPy relaxed LP solving
integral/fractional partitioning
ExtractFractionals
LPActionPlan diagnostics
targeted synthetic tests
```

This provides a safe foundation for the next phase, where the LP helper can be  
run in dry-run mode against real vLLM scheduler state without changing actual  
scheduling behavior.

## Phase 12.4: Dry-run LP scheduler bridge

### Goal

The goal of Phase 12.4 was to connect the pure-Python LP helper layer from
Phase 12.3 to real vLLM scheduler state in a safe dry-run mode.

The dry-run bridge snapshots live scheduler state, builds synthetic
`LPRequestSnapshot` inputs, runs the LP helper, logs diagnostics, and then
delegates to native vLLM scheduling unchanged.

This phase intentionally did not implement real LP action translation. The LP
plan is diagnostic only.

### Phase 12.4a: Read-only implementation mapping

Phase 12.4a was a read-only mapping phase.

It produced the implementation map:

```text
docs/phase12_4b_implementation_map.md
````

This map specified how the dry-run scheduler should be integrated without  
changing scheduling behavior.

The planned scheduler class was:

```text
PrimalLPDryRunScheduler
```

with module path:

```text
vllm_scheduler_policies/primal_lp_dry_run.py
```

The required inheritance pattern was:

```python
class PrimalLPDryRunScheduler(InstrumentedSchedulerMixin, Scheduler):
    ...
```

The class must override:

```text
_schedule_impl()
```

and must not override:

```text
schedule()
```

This preserves `InstrumentedSchedulerMixin.schedule()` as the outer  
timing/logging wrapper.

### Critical delegation decision

Phase 12.4a identified an important method-resolution-order hazard.

Inside a subclass override of `_schedule_impl()`, this is forbidden:

```python
super().schedule()
```

For a class with MRO:

```text
PrimalLPDryRunScheduler
  -> InstrumentedSchedulerMixin
  -> Scheduler
```

calling `super().schedule()` inside `PrimalLPDryRunScheduler._schedule_impl()`  
would re-enter `InstrumentedSchedulerMixin.schedule()`, which would call  
`self._schedule_impl()` again, causing recursion.

The correct native/default delegation is the explicit unbound call:

```python
Scheduler.schedule(self)
```

or a helper that does exactly that.

This is different from the default `InstrumentedSchedulerMixin._schedule_impl()`,  
where `super().schedule()` resolves directly to the native vLLM scheduler.

### Snapshot design

Phase 12.4a specified that the dry-run scheduler must collect live scheduler  
state without mutation.

The allowed scheduler containers were:

```text
list(self.waiting)
list(self.skipped_waiting)
list(self.running)
```

The snapshot logic treats these containers and all request fields as read-only.

The request universe for the dry-run LP is:

```text
waiting requests
skipped_waiting requests
running requests
```

For each request, the dry-run bridge derives an `LPRequestSnapshot` containing:

```text
request_id
arrival_time
remaining_prefill_tokens
max_prefill_chunk_tokens
prefill_eligible
decode_eligible
preemptible
prefill_memory_blocks_per_token
decode_memory_blocks
preempt_recoverable_blocks
```

### Conservative request classification

Phase 12.4a required the dry-run LP to support only simple generative request  
states.

If any request is in an unsupported state, the scheduler logs an LP fallback  
record and skips the LP solve for that scheduler step.

Unsupported conditions include:

```text
non-empty spec_token_ids
num_output_placeholders > 0
pooling_params is not None
kv_transfer_params is not None
has_encoder_inputs is true
status == WAITING_FOR_REMOTE_KVS
status == WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR
status outside WAITING, RUNNING, PREEMPTED
missing fields needed for classification
```

This conservative rule avoids guessing for speculative decoding, async output  
placeholders, pooling, KV transfer, multimodal/encoder inputs, structured-output  
grammar waits, streaming waits, and unknown states.

### KV-block planning units

Phase 12.4 continued the Phase 12.1/12.2 decision that LP memory planning should  
use KV blocks, not bytes or raw GPU memory fractions.

Free memory blocks are read from:

```python
self.kv_cache_manager.block_pool.get_num_free_blocks()
```

Per-request allocated blocks are read, when possible, from:

```python
self.kv_cache_manager.get_blocks(request_id)
```

The dry-run bridge does not call any allocation or free API.

For Phase 12.4b, prefill and decode memory coefficients remain conservative  
planning approximations:

```text
prefill_memory_blocks_per_token = 1.0 / self.block_size
decode_memory_blocks            = 1.0 / self.block_size
```

The currently allocated KV blocks for a request are used as the estimated  
recoverable blocks for abstract preemption planning.

### Phase 12.4b: Dry-run implementation

Phase 12.4b implemented the dry-run scheduler bridge.

It added:

```text
vllm_scheduler_policies/primal_lp_dry_run.py
tests/test_primal_lp_dry_run.py
```

No native vLLM files were modified.

No CUDA, C++, or Triton code was modified.

The existing `InstrumentedSchedulerMixin` was not modified.

The existing `SimplePolicy1Scheduler` behavior was not modified.

The existing Phase 12.3 LP helper layer was not modified.

The root package import remained vLLM-lazy; the new dry-run scheduler was not  
eagerly imported from `vllm_scheduler_policies/__init__.py`.

### Implemented dry-run control flow

`PrimalLPDryRunScheduler._schedule_impl()` performs the following best-effort  
diagnostic path:

1. collect a non-mutating scheduler snapshot;
    
2. classify requests conservatively;
    
3. if an unsupported state is found, log an LP fallback record;
    
4. otherwise, compute default-like LP weights;
    
5. call `solve_lp_relaxation(...)`;
    
6. summarize the resulting `LPActionPlan`;
    
7. log a second JSONL event named `lp_dry_run`;
    
8. always delegate to native vLLM scheduling with `Scheduler.schedule(self)`.
    

Unexpected exceptions inside the LP dry-run path are caught and logged. Native  
scheduler exceptions from `Scheduler.schedule(self)` are not swallowed.

Therefore Phase 12.4b does not change scheduling behavior. The return value is  
always the native vLLM scheduler output, unless the native scheduler itself  
raises.

### LP dry-run logging

The dry-run bridge uses the existing scheduler instrumentation path controlled  
by:

```text
SCHEDULER_POLICIES_ITER_LOG
```

It does not use the old forbidden environment variable:

```text
VLLM_SCHEDULER_ITER_LOG
```

For each scheduler call, `InstrumentedSchedulerMixin.schedule()` still emits the  
normal:

```text
event = "scheduler_call"
```

record.

The dry-run bridge can additionally emit:

```text
event = "lp_dry_run"
```

through the existing `_instrumentation_write()` helper.

The LP dry-run record includes diagnostics such as:

```text
call_index
scheduler_class
lp_num_requests
lp_fallback
lp_unsupported_reason
lp_dry_run_error_type
lp_dry_run_error_message
lp_solver_success
lp_solver_status
lp_solver_message
lp_objective_value
lp_num_integral_requests
lp_num_fractional_requests
lp_fractional_rule_violation
lp_num_decode_actions
lp_num_prefill_actions
lp_num_prefill_tokens
lp_num_preemptions
lp_num_forced_preemptions
lp_token_budget_remaining
lp_sequence_budget_remaining
lp_memory_blocks_remaining
kv_block_read_error
```

A small follow-up revision fixed unsupported-state fallback logging so that  
`lp_num_requests` records the number of observed requests in  
`waiting + skipped_waiting + running` rather than incorrectly logging zero.

### Tests added

Phase 12.4b added targeted unit tests for the dry-run scheduler bridge.

The tests cover:

```text
root package import remains vLLM-lazy
PrimalLPDryRunScheduler overrides _schedule_impl, not schedule
forbidden strings are absent from primal_lp_dry_run.py
native delegation uses Scheduler.schedule(self)
waiting requests are classified as prefill-eligible
running decode-ready requests are classified as decode-eligible
unsupported states fallback conservatively
snapshot collection uses non-mutating iteration and KV block reads
unsupported fallback logs the observed request count and delegates
LPActionPlan diagnostic summarization
LP dry-run exceptions are caught/logged and default scheduling still runs
native Scheduler.schedule(self) exceptions are not swallowed
```

The forbidden-string test checks that the dry-run scheduler module does not  
contain:

```text
super().schedule()
allocate_slots(
_preempt_request(
SchedulerOutput(
pop_request(
prepend_request(
VLLM_SCHEDULER_ITER_LOG
```

### Validation

The final Phase 12.4b validation commands were:

```bash
~/vllm-sched/.venv/bin/python -c "import vllm_scheduler_policies; print('root import ok')"

~/vllm-sched/.venv/bin/python -m compileall vllm_scheduler_policies

~/vllm-sched/.venv/bin/python -m pytest \
  tests/test_primal_lp_solver.py \
  tests/test_primal_lp_extraction.py \
  tests/test_primal_lp_dry_run.py -v

git diff --check
git diff --stat
git status --short
```

The targeted pytest suite passed:

```text
33 passed
```

The observed warnings were dependency deprecation warnings from SWIG/Torch and  
were not caused by the scheduler patch.

### Phase 12.4b boundaries respected

Phase 12.4b did not:

```text
modify native vLLM files
modify CUDA/C++/Triton code
modify InstrumentedSchedulerMixin
modify SimplePolicy1Scheduler
modify the Phase 12.3 primal_lp helper layer
override schedule()
call super().schedule() from the dry-run scheduler
implement _translate_lp_actions()
construct SchedulerOutput
call allocate_slots()
call _preempt_request()
call pop_request()
call prepend_request()
mutate waiting, skipped_waiting, running, KV cache, or Request state during snapshot collection
run a vLLM server
run curl checks
run benchmarks
change real scheduling behavior
```

### Phase 12.4 outcome

Phase 12.4 successfully established the first real scheduler bridge for the LP  
relaxation project.

The project now has:

```text
a pure LP helper layer
a dry-run scheduler class
real vLLM state snapshot collection
conservative request classification
KV-block planning diagnostics
LPActionPlan summarization
JSONL dry-run logging
targeted dry-run tests
```

The implementation is still diagnostic only. It safely runs the LP planner  
against live scheduler state while returning native vLLM scheduling output  
unchanged.

### Remaining risks after Phase 12.4

The main remaining risks are:

1. **The LP plan is still not executable by vLLM.**  
    `LPActionPlan` is logged and discarded. It is not translated into native queue  
    mutations, KV allocation, request-state updates, or `SchedulerOutput`.
    
2. **Action translation remains the highest-risk future phase.**  
    A future `_translate_lp_actions()` must preserve vLLM scheduler invariants,  
    use native allocation checks, update request/queue/KV state consistently, and  
    construct a valid native output.
    
3. **KV-block cost estimates remain approximate.**  
    The dry-run bridge uses `1.0 / block_size` planning estimates for prefill and  
    decode memory. Exact incremental KV allocation may depend on block boundaries,  
    prefix cache hits, hybrid KV-cache groups, and native allocation logic.
    
4. **Advanced request states remain unsupported.**  
    Speculative decoding, async placeholders, pooling, KV transfer, encoder or  
    multimodal inputs, structured-output grammar waits, and other non-simple  
    states currently trigger fallback.
    
5. **Solver overhead is now measurable but not yet benchmarked.**  
    The dry-run bridge can log LP diagnostics during scheduler calls, but Phase  
    12.4b did not run server experiments or benchmarks.
    
6. **No scheduling-quality comparison has been performed yet.**  
    Since Phase 12.4b does not change scheduling decisions, it validates  
    integration safety and logging only, not LP policy performance.
    

# Factual/wording issues in the report:

1. In Phase 12.2, the section is written as a plan rather than a completed phase: “The goal … is to update,” “Files to update,” and “should remain documentation-only.” That is not necessarily false, but for a report it should probably be converted to past tense if Phase 12.2 has already been completed.

2. In Phase 12.2, the suggested initial reserve:

```text
lp_memory_reserve_blocks = 8
````

is a design proposal, not what Phase 12.4b actually implemented. Phase 12.4b uses a scheduler-local `lp_memory_reserve_blocks` attribute with default `0.0`. To avoid confusion, mark the Phase 12.2 value explicitly as a proposal/tunable, not an implemented default.

4. Phase 12.3 says the root lazy export still resolves:
    
    ```text
    from vllm_scheduler_policies import BaselinePassthroughScheduler
    ```
    
    This is true for the current root lazy pattern, but it also imports the vLLM scheduler path when that attribute is resolved. The important smoke check for pure helper import is instead that `import vllm_scheduler_policies` and `import vllm_scheduler_policies.primal_lp` remain vLLM/CUDA-lazy until scheduler classes are explicitly imported.
    
5. The title says “Implementation Planning,” but after Phase 12.3 and Phase 12.4 the document also reports real implementation work. Consider changing the title to:
    
    ```md
    # Phase 12 Report: Primal LP Relaxation Scheduler Planning and Dry-Run Bridge
    ```