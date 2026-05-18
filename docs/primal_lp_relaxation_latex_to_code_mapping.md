# Primal LP Relaxation LaTeX-to-Code Mapping

## Purpose

This document maps the LaTeX formulation from
`Primal Heuristic 1: Approximation via LP Relaxation`
(`\label{subsec:lp_relaxation}`) to candidate vLLM scheduler implementation
concepts.

The future implementation target is the complete LP-relaxation algorithm from
that section, including `SolveLPRelaxation` and `ExtractFractionals`.

This is a bridge document only. It does not implement the scheduler.

## Scheduler package context

The current behavior-preserving instrumented scheduler is:

```text
vllm_scheduler_policies.simple_policy_1.SimplePolicy1Scheduler
```

`SimplePolicy1Scheduler` remains a passthrough baseline.

A future LP-relaxation policy should likely be a new scheduler class using the
Phase 10.5 template-method workflow:

```text
InstrumentedSchedulerMixin.schedule()
  -> calls self._schedule_impl()
```

The future LP scheduler should override `_schedule_impl()`, not `schedule()`.

## Availability status definitions

|Status|Meaning|
|---|---|
|Directly available|Verified in Phase 12.1 as an existing vLLM scheduler state, field, method, or configuration path.|
|Available with approximation|The concept can be implemented using the project assumptions, but not exactly as written in the mathematical model.|
|Not available yet|The value or operation is not known to be available from verified scheduler-package state and needs code inspection or helper implementation.|
|Out of scope for first LP scheduler|Not planned for the first complete-algorithm implementation attempt.|

## Core objects and sets

|LaTeX variable / expression|Meaning in candidate LP algorithm|Candidate vLLM object / field / method|Availability status|Chosen approximation for first LP scheduler|Risk or caveat|
|---|---|---|---|---|---|
|`t`|Current scheduler iteration / decision epoch|Scheduler call count or logical iteration index from instrumentation|Available with approximation|Use scheduler-call index or monotonic counter maintained by policy/instrumentation|Wall-clock time and scheduler iteration count are not identical|
|`U_t`|All requests considered by the LP at scheduler step `t`|Union of `Scheduler.waiting`, `Scheduler.skipped_waiting`, and `Scheduler.running`, excluding finished requests|Directly available|Use scheduler-visible non-finished requests; initially fallback on unsupported states|Must avoid mutating queues during snapshot|
|`i in U_t`|One request in the current optimization set|`vllm.v1.request.Request`|Directly available|Use existing `Request` objects|Request object API may differ across vLLM versions|
|stable key for `i`|Stable request identifier|`Request.request_id` and `Scheduler.requests: dict[str, Request]`|Directly available|Use `request.request_id`|Must stay consistent with vLLM output dictionaries|
|`t_i^{arrive}`|Arrival time of request `i`|`Request.arrival_time`|Directly available|Use existing arrival timestamp for age-based weights|Arrival timestamp is wall-clock-like, not scheduler iteration count|
|`P_i^{rem}(t)`|Remaining un-prefilled prompt tokens for request `i`|`max(request.num_prompt_tokens - request.num_computed_tokens, 0)` for pure prompt progress|Available with approximation|Use only for simple generative requests on the first LP path|Native waiting/resumed scheduling uses `request.num_tokens - num_computed_tokens`, so prompt-only remaining tokens are not the whole scheduled-work formula|
|`1{P_i^{rem}(t)=0}`|Decode eligibility indicator|Derived by future `_classify_request_action_space()` from request status/progress and native scheduled-work formula|Available with approximation|Support only simple generative requests initially; fallback on advanced states|There is no explicit vLLM decode-ready field; naive `P_i^{rem} == 0` is insufficient|

## Phase 12.1 verified scheduler objects

|Needed concept|Verified vLLM mapping|Implementation note|
|---|---|---|
|waiting request container|`Scheduler.waiting` and `Scheduler.skipped_waiting`, both `RequestQueue` instances|Include both queues in `U_t`; preserve queue order and do not mutate during snapshot|
|running request container|`Scheduler.running: list[Request]`|Running requests include decode-ready requests and partial-prefill progress|
|partially-prefilled request representation|`Request` progress, especially `num_computed_tokens`; `is_prefill_chunk` is post-schedule state|No separate partial-prefill container was found|
|request ID / stable key|`Request.request_id`|Use as LP map key and scheduler-output key|
|arrival time / order|`Request.arrival_time`|Available for default-like utility weights|
|token budget|`self.max_num_scheduled_tokens`|Initialized from `scheduler_config.max_num_scheduled_tokens` or `max_num_batched_tokens`|
|sequence budget|`self.max_num_running_reqs` / `scheduler_config.max_num_seqs` semantics|Counts running request capacity, not a standalone LP object|
|free KV capacity|`self.kv_cache_manager.block_pool.get_num_free_blocks()`|Capacity unit is KV blocks|
|memory reserve / watermark|No generic scheduler watermark found beyond `scheduler_reserve_full_isl` admission safety|Use scheduler-local `lp_memory_reserve_blocks` for LP planning|
|per-request KV allocation|KV manager block mappings/helpers, including `get_blocks(req_id)` and underlying `req_to_blocks`|Use for block-count estimates only; native allocation remains authoritative|
|recomputation preemption|`_preempt_request(request, timestamp)`|Caller must remove request from `running` before invoking, following native semantics|
|chunk length control|`num_new_tokens`, token budget, `long_prefill_token_threshold`, model length, encoder constraints, alignment|LP chunk size may be reduced or rejected by translation|
|return type|`SchedulerOutput`|LP plans are not valid scheduler outputs by themselves|

## Utility weights

|LaTeX variable / expression|Meaning in candidate LP algorithm|Candidate vLLM object / field / method|Availability status|Chosen approximation for first LP scheduler|Risk or caveat|
|---|---|---|---|---|---|
|`alpha_i(t)`|Utility of scheduling one decode token for request `i`|Scheduler-computed LP input|Available with approximation|Compute inside LP scheduler before LP construction|Most important policy knob; determines decode-vs-prefill behavior|
|`alpha_i(t) -> infinity`|Native-vLLM-like absolute decode dominance|Scheduler-computed large decode weight or lexicographic objective|Available with approximation|Use a very large finite value initially; consider lexicographic solve later|Too-small value may fail to enforce decode dominance; too-large value may harm numerical conditioning|
|`beta_i(t)`|Utility of scheduling one prefill token for request `i`|Scheduler-computed LP input|Available with approximation|Compute inside LP scheduler before LP construction|Determines FCFS/fairness/throughput behavior for prefill|
|`beta_i(t)=1+c(t-t_i^{arrive})`|FCFS-like prefill tie-breaking by request age|Scheduler-computed age-based prefill weight|Available with approximation|Use logical scheduler age or arrival-order age|Requires reliable arrival time or first-seen time|
|`gamma_i(t)`|Penalty for preempting request `i`|Scheduler-computed LP input|Available with approximation|Compute inside LP scheduler before LP construction|Most important knob for preemption behavior|
|`gamma_i(t)=1e9(t-t_i^{arrive})`|Massive LIFO-like preemption penalty|Scheduler-computed age-based preemption penalty|Available with approximation|Use large finite age-scaled penalty|Numerical conditioning risk in SciPy LP; may need normalization|
|`c` in `beta_i(t)`|Small age coefficient for FCFS prefill priority|Scheduler policy hyperparameter|Not available yet|Add as future scheduler config/default constant|Needs experiment tuning|

## LP decision variables

|LaTeX variable / expression|Meaning in candidate LP algorithm|Candidate vLLM object / field / method|Availability status|Chosen approximation for first LP scheduler|Risk or caveat|
|---|---|---|---|---|---|
|`x_i(t)`|Number of prefill tokens assigned to request `i` this step|LP variable; later translated into vLLM prefill/chunk size|Not available yet|Create as SciPy LP variable|vLLM must support enforcing the chosen chunk size|
|`y_i(t)`|Decode indicator for request `i`|LP variable; later translated into one decode-token scheduling action|Not available yet|Create as SciPy LP variable in `[0,1]`|Must map exactly to vLLM decode scheduling semantics|
|`z_i(t)`|Preemption indicator for request `i`|LP variable; later translated into vLLM recomputation preemption/delete action|Not available yet|Create as SciPy LP variable in `[0,1]`|Preemption state transitions are high-risk|
|`I_i^P(t)`|Prefill admission indicator for request `i`|LP variable; later translated into prefill scheduling action|Not available yet|Create as SciPy LP variable in `[0,1]`|Must stay consistent with `x_i(t)`|
|`tilde{x}_i`|Relaxed LP prefill-token solution|SciPy solver output|Not available yet|Read from SciPy result vector|Floating-point tolerance required|
|`tilde{y}_i`|Relaxed LP decode solution|SciPy solver output|Not available yet|Read from SciPy result vector|Floating-point tolerance required|
|`tilde{z}_i`|Relaxed LP preemption solution|SciPy solver output|Not available yet|Read from SciPy result vector|Floating-point tolerance required|
|`tilde{I}_i^P`|Relaxed LP prefill-admission solution|SciPy solver output|Not available yet|Read from SciPy result vector|Floating-point tolerance required|
|`hat{x}_i`|Final integer/rounded prefill-token decision|Extracted action plan|Not available yet|Produced by integral locking and `ExtractFractionals`|Must be feasible under vLLM budgets|
|`hat{y}_i`|Final decode decision|Extracted action plan|Not available yet|Produced by integral locking and `ExtractFractionals`|Must not decode before prefill completion|
|`hat{z}_i`|Final preemption decision|Extracted action plan|Not available yet|Produced by integral locking and `ExtractFractionals`|Must safely delete/recompute request state|
|`hat{I}_i^P`|Final prefill admission decision|Extracted action plan|Not available yet|Produced by integral locking and `ExtractFractionals`|Must agree with `hat{x}_i > 0`|

## Global capacities and constants

|LaTeX variable / expression|Meaning in candidate LP algorithm|Candidate vLLM object / field / method|Availability status|Chosen approximation for first LP scheduler|Risk or caveat|
|---|---|---|---|---|---|
|`B_max`|Per-step token budget|`self.max_num_scheduled_tokens`|Directly available|Use native per-step scheduled-token budget|Must include both prefill and decode scheduled tokens|
|`S_max`|Max number of active scheduled request/sequence actions|`self.max_num_running_reqs` from `scheduler_config.max_num_seqs`|Directly available|Use max-running-request semantics|Not the same as `max_num_partial_prefills`; avoid adding that separate LP constraint initially|
|`M_t^{free}`|Current free KV-cache capacity|`self.kv_cache_manager.block_pool.get_num_free_blocks()`|Directly available|Use KV blocks as LP memory unit|Block availability is still only a planning input; allocation path remains authoritative|
|`W_t`|Memory safety reserve inside LP|Scheduler-local `lp_memory_reserve_blocks`|Available with approximation|Introduce as policy constant/config later|`gpu_memory_utilization` is not `W_t`; no generic scheduler watermark was found|
|`C_max`|Max prefill chunk size|Native chunk cap from `long_prefill_token_threshold`, residual token budget, model length, encoder constraints, and alignment|Available with approximation|Use `_classify_request_action_space()` / translation helper to compute candidate chunk caps|No single standalone chunk-cap field fully captures native behavior|
|`c_i^P`|KV memory consumed by prefill action|Conservative estimate of incremental KV blocks needed for the candidate prefill chunk|Available with approximation|Estimate with KV manager block accounting where possible; final call still uses `allocate_slots()`|Do not model as bytes or raw per-token memory|
|`c_i^D`|KV memory consumed by one decode action|Conservative estimate of incremental KV blocks needed for one decode step|Available with approximation|Estimate in KV blocks|Decode may require zero new blocks except at block boundaries|
|`c_i^Z`|KV memory recovered by preempting request `i`|Currently allocated KV blocks recoverable through recomputation preemption|Available with approximation|Count allocated KV blocks via KV manager mappings/helpers|Must match `_preempt_request()` state transition; shared/cached blocks complicate exact recovery|
|`M_t^{free} - W_t`|Usable KV block capacity after reserve|`get_num_free_blocks() - lp_memory_reserve_blocks`|Available with approximation|Use as planning capacity only|LP result is not proof of actual KV feasibility|

## LP objective and constraints

|LaTeX variable / expression|Meaning in candidate LP algorithm|Candidate vLLM object / field / method|Availability status|Chosen approximation for first LP scheduler|Risk or caveat|
|---|---|---|---|---|---|
|`max sum_i alpha_i y_i + beta_i x_i - gamma_i z_i`|One-step utility objective|SciPy LP objective vector|Not available yet|Build objective coefficients from scheduler-computed weights|SciPy minimizes by default, so signs must be handled carefully|
|`sum_i (x_i + y_i) <= B_max`|Token budget constraint|`self.max_num_scheduled_tokens`|Directly available|Add as LP inequality|Must include both prefill and decode tokens|
|`sum_i (I_i^P + y_i) <= S_max`|Sequence/concurrency budget constraint|`self.max_num_running_reqs` / `scheduler_config.max_num_seqs` semantics|Directly available|Add as LP inequality|Do not add separate `max_num_partial_prefills` constraint initially|
|`sum_i(c_i^P x_i + c_i^D y_i - c_i^Z z_i) <= M_t^{free} - W_t`|KV memory planning constraint|KV-cache block accounting|Available with approximation|Add scalar block-count inequality using `lp_memory_reserve_blocks`|Final translation must still call native allocation and handle failure safely|
|`x_i <= min(P_i^{rem}, C_max) I_i^P`|Chunked prefill coupling|Prompt progress and chunk cap|Available with approximation|Add local LP inequality for each request|Requires reliable prompt progress|
|`y_i <= 1{P_i^{rem}=0}`|Decode causality in the LaTeX model|Derived decode eligibility from `_classify_request_action_space()`|Available with approximation|Set upper bound of `y_i` to 0 unless simple generative decode eligibility is verified|Naive `P_i^{rem} == 0` is insufficient in vLLM|
|`I_i^P + y_i + z_i <= 1`|Mutual exclusion between prefill, decode, and preempt|Local LP inequality per request|Not available yet|Add local LP inequality|Must also enforce through action translation|
|`x_i >= 0`|Nonnegative prefill amount|SciPy variable bound|Not available yet|LP lower bound|None|
|`0 <= y_i,z_i,I_i^P <= 1`|Relaxed binary variables|SciPy variable bounds|Not available yet|LP bounds|Later extraction must restore integrality|

## SolveLPRelaxation mapping

|LaTeX variable / expression|Meaning in candidate LP algorithm|Candidate vLLM object / field / method|Availability status|Chosen approximation for first LP scheduler|Risk or caveat|
|---|---|---|---|---|---|
|`SolveLPRelaxation`|Main LP relaxation and rounding procedure|Future helper called inside `_schedule_impl()`|Not available yet|Implement later as internal helper, not in Phase 11|Must be fast enough for scheduler loop|
|Build relaxed LP|Convert scheduler snapshot to LP matrices/vectors|SciPy `linprog` input construction|Not available yet|Use SciPy first|Dependency and latency risk|
|Solve relaxed LP|Obtain fractional solution|SciPy LP solver result|Not available yet|Use SciPy/HiGHS if available|Solver failure path must fall back safely|
|`U_int`|Requests with integral relaxed action variables|Derived from SciPy solution|Not available yet|Values within tolerance of 0/1 are integral|Numerical tolerance choice matters|
|`U_frac`|Requests with at least one fractional relaxed action variable|Derived from SciPy solution|Not available yet|Values not near 0/1 are fractional|The theory predicts at most three only if LP structure matches formulation|
|`epsilon` / tolerance|Floating point integrality threshold|Scheduler constant|Not available yet|Start with `1e-6`|Too strict creates spurious fractionals; too loose may misround|
|`floor(tilde{x}_i)`|Integral prefill token count for integral requests|Python integer conversion|Not available yet|Floor continuous prefill amount|May waste one token of capacity due to numerical underflow|
|`B_rem`|Remaining token budget after integral decisions|Derived scalar|Not available yet|Compute from locked integral actions|Must stay nonnegative|
|`S_rem`|Remaining sequence budget after integral decisions|Derived scalar|Not available yet|Compute from locked integral actions|Must stay nonnegative|
|`M_rem`|Remaining usable memory after integral decisions|Derived scalar in KV blocks|Not available yet|Compute using block-count planning model|Must stay nonnegative for the plan, but native allocation can still fail|

## ExtractFractionals mapping

|LaTeX variable / expression|Meaning in candidate LP algorithm|Candidate vLLM object / field / method|Availability status|Chosen approximation for first LP scheduler|Risk or caveat|
|---|---|---|---|---|---|
|`ExtractFractionals`|Deterministic conversion of fractional LP solution to feasible integer decisions|Future helper called by `SolveLPRelaxation`|Not available yet|Implement later as internal helper|Correctness depends on capacity accounting|
|`argmax(tilde{y}_i, tilde{I}_i^P, tilde{z}_i)`|Dominant relaxed action for fractional request|Derived from SciPy solution|Not available yet|Use largest relaxed action component|Tie-breaking must be deterministic|
|Dominant preemption|Fractional request mostly wants preemption|Future extracted action|Not available yet|Set `hat_z_i = 1`|Must call safe vLLM preemption/delete path|
|`M_curr += c_i^Z`|Memory recovered by preemption|Block-count planning accounting|Available with approximation|Add recovered block estimate|Must match actual `_preempt_request()` effects|
|`M_curr < 0`|Memory still infeasible after dominant preemptions|Block-count planning check|Available with approximation|Trigger forced safety preemption|If block estimate is wrong, safety may be insufficient|
|Largest `tilde{z}_i` forced preemption|Conservative memory feasibility fallback|Future extracted action|Not available yet|Preempt fractional request with largest relaxed preemption variable|May reduce objective but protects memory|
|`argmax(tilde{y}_i, tilde{I}_i^P)`|Dominant admission action after preemption phase|Derived from SciPy solution|Not available yet|Choose decode or prefill among non-preempted fractionals|Must be deterministic on ties|
|Decode extraction|Schedule one decode token if capacity permits|Future action translation|Not available yet|Set `hat_y_i=1` if token and memory capacity remain|Must also satisfy decode causality|
|Prefill extraction|Schedule a feasible prefill chunk|Future action translation|Not available yet|Set `hat_x_i` using min of remaining prompt, chunk cap, token budget, and memory budget|Requires vLLM to honor chosen prefill length|
|`floor(M_curr / c_i^P)`|Memory-limited prefill chunk size in LaTeX-style extraction|Block-count planning accounting|Available with approximation|Use only if `c_i^P` is expressed as blocks for the candidate chunk granularity|Not linear in tokens at block boundaries; translation may need retry/shrink logic|
|Fluid chunking|Truncate final prefill chunk to fit residual capacity|vLLM chunked prefill scheduling|Available with approximation|Plan smaller chunk if needed|Native allocation path may still reject the chunk|

## Translation to vLLM actions

|LaTeX variable / expression|Meaning in candidate LP algorithm|Candidate vLLM object / field / method|Availability status|Chosen approximation for first LP scheduler|Risk or caveat|
|---|---|---|---|---|---|
|`hat_x_i > 0` and `hat_I_i^P = 1`|Final decision to prefill request `i`|Native scheduling path using `num_new_tokens` and `allocate_slots()`|Available with approximation|Translate to vLLM scheduling state only after dry-run validation|LP plan cannot bypass native allocation|
|`hat_y_i = 1`|Final decision to decode request `i`|Native running-request scheduled-work formula and `allocate_slots()`|Available with approximation|Translate only for simple generative decode-ready requests initially|Must preserve decode batching and spec/async semantics by falling back when uncertain|
|`hat_z_i = 1`|Final decision to preempt request `i`|`_preempt_request(request, timestamp)` after removing from `running`|Directly available|Implement real preemption last|Highest-risk state transition|
|all zero|Request receives no action this step|Leave request in appropriate queue/state|Available with approximation|Do nothing to request this iteration|Must avoid losing requests or changing queue order accidentally|
|`LPActionPlan`|Internal plan produced by LP extraction|Future policy-local data structure|Not available yet|Create and test before real action translation|Not a valid vLLM scheduler output|
|Scheduled outputs|Data structure returned by vLLM `Scheduler.schedule()` path|`SchedulerOutput` with native fields and invariants|Directly available|Future `_schedule_impl()` must return same type as default scheduler|Must preserve exact return contract and post-schedule updates|

## Preemption and recomputation assumptions

|LaTeX variable / expression|Meaning in candidate LP algorithm|Candidate vLLM object / field / method|Availability status|Chosen approximation for first LP scheduler|Risk or caveat|
|---|---|---|---|---|---|
|`z_i=1`|Preempt request `i`|`_preempt_request(request, timestamp)`|Directly available|Treat as delete-and-recompute preemption|Caller must follow native running-queue removal semantics|
|`c_i^Z`|Memory recovered under preemption|KV blocks currently held by request `i`|Available with approximation|Recover allocated-block estimate|Shared/cached block accounting can differ from simple counts|
|swapping|Alternative preemption mode|vLLM swap support if present|Out of scope for first LP scheduler|Do not use swapping|Revisit only after recomputation preemption works|
|recomputation|Request can be resumed by recomputing prompt/KV|Native `_preempt_request()` resets `num_computed_tokens` and status to `PREEMPTED`, then prepends to waiting|Directly available|Assume preempted sequences are deleted and recomputed|Real preemption should come after dry-run LP validation|

## Fallback and unsupported-state policy

The first integrated LP scheduler should fail safely. If the LP path sees
unsupported request states, solver failure, allocation failure, or uncertain
action translation, it should delegate to default scheduling rather than
corrupt scheduler state.

The first LP path should support only simple generative requests. Treat the
following as unsupported initially:

- speculative decoding,

- async placeholders,

- pooling,

- KV-transfer edge cases,

- multimodal encoder complications,

- other advanced states where `_classify_request_action_space()` cannot derive
  a conservative action set.

## Instrumentation mapping

|LaTeX variable / expression|Meaning in candidate LP algorithm|Candidate vLLM object / field / method|Availability status|Chosen approximation for first LP scheduler|Risk or caveat|
|---|---|---|---|---|---|
|LP construction|Time to convert scheduler state to LP input|New JSONL fields from scheduler policy|Not available yet|Log `lp_build_time_ms`|Must keep overhead low|
|LP solve|Time spent inside SciPy solver|New JSONL fields from scheduler policy|Not available yet|Log `lp_solve_time_ms`|Solver time may dominate scheduler overhead|
|LP extraction|Time spent in integral/fractional extraction|New JSONL fields from scheduler policy|Not available yet|Log `lp_extract_time_ms`|Helps isolate rounding overhead|
|Action translation|Time spent converting LP output to vLLM actions|New JSONL fields from scheduler policy|Not available yet|Log `lp_translate_time_ms`|Translation may be complex|
|Objective value|Relaxed LP objective|SciPy solver result|Not available yet|Log `lp_objective_value`|Useful for debugging only|
|Solver status|Success/failure of SciPy solve|SciPy solver result|Not available yet|Log `lp_solver_status`|Failure fallback must be defined before implementation|
|`|U_t|`|Number of optimized requests|Scheduler snapshot size|Available with approximation|
|`|U_int|`|Number of integral requests|Derived from LP solution|Not available yet|
|`|U_frac|`|Number of fractional requests|Derived from LP solution|Not available yet|
|Number of preemptions|Count of final `hat_z_i=1`|Derived from action plan|Not available yet|Log `lp_num_preemptions`|Must distinguish dominant vs forced preemptions|

## Remaining items requiring design before implementation

Phase 12.1 verified the core scheduler field names. The following items still
need conservative helper design before coding:

|Needed concept|What must be inspected|
|---|---|
|advanced decode eligibility|exact conservative rules for spec decode, async placeholders, pooling, KV transfer, and multimodal cases|
|block-cost estimation|best helper for predicting incremental blocks without mutating state|
|action translation|how to build a full native-equivalent `SchedulerOutput` without missing connector/zeroing/post-update invariants|
|safe allocation failure response|whether to shrink, skip, preempt, or delegate when `allocate_slots()` returns `None`|
|dry-run logging format|shape of the future `LPActionPlan` diagnostic record|

## Explicitly not mapped in the first LP scheduler

| Concept                               | Reason                                                                                               |
| ------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| byte-level or raw per-token memory    | Phase 12.1 resolved the vLLM integration unit to KV blocks                                           |
| swapping preemption                   | Project assumption is delete-and-recompute preemption only                                           |
| `max_num_partial_prefills` constraint | Not visibly enforced in inspected `schedule()` path; revisit later rather than adding first          |
| advanced vLLM request states          | First LP path should fallback for spec decode, async placeholders, pooling, KV transfer, and MM edge cases |
| algorithms after `Primal Heuristic 1` | User explicitly restricted attention to `\label{subsec:lp_relaxation}`                               |
| native vLLM source edits              | External scheduler package should remain the implementation location unless explicitly changed later |
| overriding `schedule()`               | Phase 10.5 established `_schedule_impl()` as the policy override point                               |
