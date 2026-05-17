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
|Directly available|Expected to map to existing vLLM scheduler state or configuration, though exact field names still need verification before coding.|
|Available with approximation|The concept can be implemented using the project assumptions, but not exactly as written in the mathematical model.|
|Not available yet|The value or operation is not known to be available from verified scheduler-package state and needs code inspection or helper implementation.|
|Out of scope for first LP scheduler|Not planned for the first complete-algorithm implementation attempt.|

## Core objects and sets

|LaTeX variable / expression|Meaning in candidate LP algorithm|Candidate vLLM object / field / method|Availability status|Chosen approximation for first LP scheduler|Risk or caveat|
|---|---|---|---|---|---|
|`t`|Current scheduler iteration / decision epoch|Scheduler call count or logical iteration index from instrumentation|Available with approximation|Use scheduler-call index or monotonic counter maintained by policy/instrumentation|Wall-clock time and scheduler iteration count are not identical|
|`U_t`|All requests considered by the LP at scheduler step `t`|Union of scheduler-visible non-finished requests: waiting, partially-prefilled, running/decode-ready, and preemptible|Available with approximation|Include all requests present in the scheduler except fully processed requests|Exact vLLM containers for all relevant request states must be verified|
|`i in U_t`|One request in the current optimization set|vLLM request object / scheduler request state|Directly available|Use existing request objects once exact fields are verified|Request object API may differ across vLLM versions|
|`t_i^{arrive}`|Arrival time of request `i`|Request arrival timestamp if stored; otherwise policy-maintained insertion timestamp|Not available yet|Prefer existing request arrival metadata; otherwise record arrival order/time when first seen|Must avoid changing request semantics while adding metadata|
|`P_i^{rem}(t)`|Remaining un-prefilled prompt tokens for request `i`|Request prompt progress / number of computed prompt tokens / remaining prompt tokens|Not available yet|Compute from total prompt length minus already-computed prompt tokens, once fields are verified|Incorrect prompt-progress accounting would break decode causality|
|`1{P_i^{rem}(t)=0}`|Decode eligibility indicator|Request is fully prefilled / ready to decode|Not available yet|Derive from prompt-progress state|Must match vLLM's internal definition of decode-ready|

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
|`B_max`|Per-step token budget|vLLM scheduler token budget / max batched tokens budget|Directly available|Map to vLLM's per-step token budget|Exact field name must be verified|
|`S_max`|Max number of active scheduled request/sequence actions|`max_num_seqs`|Directly available|Use `max_num_seqs`|Need verify whether vLLM counts sequences, requests, or groups in this path|
|`M_t^{free}`|Current free KV-cache capacity|vLLM KV-cache manager / block manager available capacity|Available with approximation|Map to current available KV-cache capacity|Exact capacity units must be normalized to per-token model|
|`W_t`|Memory safety margin / watermark|vLLM memory safety margin if exposed; otherwise policy parameter|Available with approximation|Use vLLM safety margin if verified, else configure a conservative watermark|Must avoid over-admitting and causing KV allocation failure|
|`C_max`|Max prefill chunk size|vLLM chunked-prefill limit / max model chunk size / scheduler chunk cap|Not available yet|Map to verified chunked-prefill cap or set from scheduler config|Wrong value may break chunked prefill behavior|
|`c_i^P`|KV memory consumed per prefill token for request `i`|Per-token KV memory estimate|Available with approximation|Use scalar per-token memory model|Ignores block/page allocation effects|
|`c_i^D`|KV memory consumed by one decode token for request `i`|Per-token KV memory estimate|Available with approximation|Use scalar per-token memory model|Decode may only allocate at block boundary in real vLLM|
|`c_i^Z`|KV memory recovered by preempting request `i`|Currently allocated KV memory for request `i`|Available with approximation|Use number of resident tokens times per-token memory|Must match deletion/recomputation preemption semantics|
|`M_t^{free} - W_t`|Usable KV capacity after reserving safety margin|Free KV capacity minus safety watermark|Available with approximation|Compute scalar usable capacity before LP construction|Unit mismatch risk: tokens, bytes, or blocks|

## LP objective and constraints

|LaTeX variable / expression|Meaning in candidate LP algorithm|Candidate vLLM object / field / method|Availability status|Chosen approximation for first LP scheduler|Risk or caveat|
|---|---|---|---|---|---|
|`max sum_i alpha_i y_i + beta_i x_i - gamma_i z_i`|One-step utility objective|SciPy LP objective vector|Not available yet|Build objective coefficients from scheduler-computed weights|SciPy minimizes by default, so signs must be handled carefully|
|`sum_i (x_i + y_i) <= B_max`|Token budget constraint|vLLM per-step token budget|Directly available|Add as LP inequality|Must include both prefill and decode tokens|
|`sum_i (I_i^P + y_i) <= S_max`|Sequence/concurrency budget constraint|`max_num_seqs`|Directly available|Add as LP inequality|May need align with vLLM sequence-group semantics|
|`sum_i(c_i^P x_i + c_i^D y_i - c_i^Z z_i) <= M_t^{free} - W_t`|KV memory feasibility constraint|KV-cache capacity accounting|Available with approximation|Add scalar per-token memory inequality|Ignores block/page granularity|
|`x_i <= min(P_i^{rem}, C_max) I_i^P`|Chunked prefill coupling|Prompt progress and chunk cap|Available with approximation|Add local LP inequality for each request|Requires reliable prompt progress|
|`y_i <= 1{P_i^{rem}=0}`|Decode causality|Request decode-ready status|Available with approximation|Set upper bound of `y_i` to 0 if not prefill-complete|Must match vLLM decode readiness|
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
|`M_rem`|Remaining usable memory after integral decisions|Derived scalar|Not available yet|Compute using scalar memory model|Must stay nonnegative or trigger extraction safety logic|

## ExtractFractionals mapping

|LaTeX variable / expression|Meaning in candidate LP algorithm|Candidate vLLM object / field / method|Availability status|Chosen approximation for first LP scheduler|Risk or caveat|
|---|---|---|---|---|---|
|`ExtractFractionals`|Deterministic conversion of fractional LP solution to feasible integer decisions|Future helper called by `SolveLPRelaxation`|Not available yet|Implement later as internal helper|Correctness depends on capacity accounting|
|`argmax(tilde{y}_i, tilde{I}_i^P, tilde{z}_i)`|Dominant relaxed action for fractional request|Derived from SciPy solution|Not available yet|Use largest relaxed action component|Tie-breaking must be deterministic|
|Dominant preemption|Fractional request mostly wants preemption|Future extracted action|Not available yet|Set `hat_z_i = 1`|Must call safe vLLM preemption/delete path|
|`M_curr += c_i^Z`|Memory recovered by preemption|Scalar memory accounting|Available with approximation|Add recovered per-token memory estimate|Must match actual KV state deletion|
|`M_curr < 0`|Memory still infeasible after dominant preemptions|Scalar memory check|Available with approximation|Trigger forced safety preemption|If scalar model is wrong, safety may be insufficient|
|Largest `tilde{z}_i` forced preemption|Conservative memory feasibility fallback|Future extracted action|Not available yet|Preempt fractional request with largest relaxed preemption variable|May reduce objective but protects memory|
|`argmax(tilde{y}_i, tilde{I}_i^P)`|Dominant admission action after preemption phase|Derived from SciPy solution|Not available yet|Choose decode or prefill among non-preempted fractionals|Must be deterministic on ties|
|Decode extraction|Schedule one decode token if capacity permits|Future action translation|Not available yet|Set `hat_y_i=1` if token and memory capacity remain|Must also satisfy decode causality|
|Prefill extraction|Schedule a feasible prefill chunk|Future action translation|Not available yet|Set `hat_x_i` using min of remaining prompt, chunk cap, token budget, and memory budget|Requires vLLM to honor chosen prefill length|
|`floor(M_curr / c_i^P)`|Memory-limited prefill chunk size|Scalar memory accounting|Available with approximation|Use per-token memory model|Division invalid if `c_i^P <= 0`; must guard|
|Fluid chunking|Truncate final prefill chunk to fit residual capacity|vLLM chunked prefill scheduling|Available with approximation|Schedule smaller chunk if needed|Exact API for chunk length must be verified|

## Translation to vLLM actions

|LaTeX variable / expression|Meaning in candidate LP algorithm|Candidate vLLM object / field / method|Availability status|Chosen approximation for first LP scheduler|Risk or caveat|
|---|---|---|---|---|---|
|`hat_x_i > 0` and `hat_I_i^P = 1`|Final decision to prefill request `i`|vLLM prefill/chunked-prefill scheduling action|Not available yet|Schedule exactly `hat_x_i` prefill tokens|Exact vLLM method/field must be verified|
|`hat_y_i = 1`|Final decision to decode request `i`|vLLM decode scheduling action|Not available yet|Schedule one decode token|Must preserve decode batching semantics|
|`hat_z_i = 1`|Final decision to preempt request `i`|vLLM recomputation preemption/delete action|Not available yet|Delete/preempt sequence and recompute later; no swapping|Highest-risk part of implementation|
|all zero|Request receives no action this step|Leave request in appropriate queue/state|Not available yet|Do nothing to request this iteration|Must avoid losing requests|
|Scheduled outputs|Data structure returned by vLLM `Scheduler.schedule()` path|Existing scheduler output type|Directly available|Future `_schedule_impl()` must return same type as default scheduler|Must preserve exact return contract|

## Preemption and recomputation assumptions

|LaTeX variable / expression|Meaning in candidate LP algorithm|Candidate vLLM object / field / method|Availability status|Chosen approximation for first LP scheduler|Risk or caveat|
|---|---|---|---|---|---|
|`z_i=1`|Preempt request `i`|vLLM preemption path|Not available yet|Treat as delete-and-recompute preemption|Must not implement swapping|
|`c_i^Z`|Memory recovered under preemption|KV memory currently held by request `i`|Available with approximation|Recover resident-token memory under scalar model|Actual KV freeing may be block/page based|
|swapping|Alternative preemption mode|vLLM swap support if present|Out of scope for first LP scheduler|Do not use swapping|Revisit only after recomputation preemption works|
|recomputation|Request can be resumed by recomputing prompt/KV|vLLM recomputation preemption semantics|Available with approximation|Assume preempted sequences are deleted and recomputed|Need verify exact state transition|

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

## Items requiring code inspection before implementation

The following mappings should not be treated as verified field names yet:

|Needed concept|What must be inspected|
|---|---|
|all non-finished requests|vLLM scheduler containers for waiting, running, partially-prefilled, and preemptible requests|
|request arrival time|whether vLLM stores arrival timestamp/order|
|remaining prompt tokens|request fields tracking prompt length and computed prompt progress|
|decode-ready status|vLLM condition for moving from prefill to decode|
|per-step token budget|exact scheduler budget object/field|
|`max_num_seqs`|exact config path available inside scheduler|
|free KV capacity|cache/block manager field or method|
|memory safety margin|whether vLLM exposes a watermark/safety margin|
|recomputation preemption|exact method/state transition used by default vLLM|
|chunked prefill length control|how default scheduler chooses and records chunk size|
|scheduler return type|exact object returned by `Scheduler.schedule()` in vLLM v0.20.2|

## Explicitly not mapped in the first LP scheduler

| Concept                               | Reason                                                                                               |
| ------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| KV block/page rounding                | Project assumption is per-token memory for first implementation target                               |
| swapping preemption                   | Project assumption is delete-and-recompute preemption only                                           |
| algorithms after `Primal Heuristic 1` | User explicitly restricted attention to `\label{subsec:lp_relaxation}`                               |
| native vLLM source edits              | External scheduler package should remain the implementation location unless explicitly changed later |
| overriding `schedule()`               | Phase 10.5 established `_schedule_impl()` as the policy override point                               |
