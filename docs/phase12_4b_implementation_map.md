# Phase 12.4b Implementation Map

**Inspection date:** 2026-05-22  
**Branch:** master  
**vLLM branch:** unity-phase4-v0.20.2-cu130  
**vLLM commit:** bc150f50299199599673614f80d12a196f377655  
**No files changed. Read-only inspection only.**

---

## 1. Exact Class Design

### Class name
```
PrimalLPDryRunScheduler
```

### Module path
```
vllm_scheduler_policies/primal_lp_dry_run.py
```

### MRO
```python
class PrimalLPDryRunScheduler(InstrumentedSchedulerMixin, Scheduler):
    ...
```

MRO resolution (C3 linearization):
```
PrimalLPDryRunScheduler
  -> InstrumentedSchedulerMixin
  -> Scheduler
  -> SchedulerInterface
  -> object
```

This mirrors `SimplePolicy1Scheduler` exactly and is the established pattern in this codebase.

### Method to override
```python
def _schedule_impl(self):
```

`schedule()` **must not** be overridden. `InstrumentedSchedulerMixin.schedule()` is the outer
timing/logging wrapper and handles wall-time measurement, call indexing, and per-call JSONL
emission. All policy logic must live in `_schedule_impl()`.

### How it delegates to default scheduling unchanged

The dry-run scheduler does the following inside `_schedule_impl()`:

1. Collect a non-mutating state snapshot.
2. Classify all requests (supported vs. unsupported).
3. If any unsupported state is detected, emit a fallback log record and call `Scheduler.schedule(self)`.
4. Otherwise: run LP build -> solve -> extract -> produce `LPActionPlan` -> emit a `lp_dry_run`
   log record -> **then call `Scheduler.schedule(self)` and return its output unchanged**.

The real vLLM scheduling action is always `Scheduler.schedule(self)`. The LP result is logged and
discarded in 12.4b. No queue mutations, no `allocate_slots()`, no `SchedulerOutput` construction.

```python
from vllm.v1.core.sched.scheduler import Scheduler  # explicit import required

def _schedule_impl(self):
    snapshot, unsupported_reason = self._collect_lp_state_snapshot()
    if unsupported_reason:
        self._write_lp_dry_run_record(unsupported=True, reason=unsupported_reason)
        return Scheduler.schedule(self)  # NOT super().schedule() — see warning below
    plan = self._run_lp(snapshot)
    self._write_lp_dry_run_record(plan=plan, snapshot=snapshot)
    return Scheduler.schedule(self)      # NOT super().schedule() — see warning below
```

### Delegation warning: `super().schedule()` is forbidden inside `_schedule_impl()`

**`super().schedule()` must never appear inside `PrimalLPDryRunScheduler._schedule_impl()`.**
It causes infinite recursion via the following verified call chain:

1. External call → `InstrumentedSchedulerMixin.schedule(self)` (only `schedule()` in MRO above
   `Scheduler`; verified instrumentation.py:94).
2. Wrapper calls `self._schedule_impl()` (instrumentation.py:111) → dispatches to
   `PrimalLPDryRunScheduler._schedule_impl(self)` because `self` is a
   `PrimalLPDryRunScheduler` instance.
3. `super()` inside `PrimalLPDryRunScheduler._schedule_impl()` resolves to
   `InstrumentedSchedulerMixin` — the next class in `PrimalLPDryRunScheduler`'s MRO, **not**
   `Scheduler`.
4. `super().schedule()` therefore calls `InstrumentedSchedulerMixin.schedule(self)` — the same
   function as step 1.
5. Step 2 repeats → **`RecursionError`**.

This differs from `InstrumentedSchedulerMixin._schedule_impl()` at instrumentation.py:168, where
`super()` resolves to `Scheduler` (the next class in `InstrumentedSchedulerMixin`'s own MRO) and
the delegation is not recursive.

The correct delegation from a subclass override of `_schedule_impl()` is always an explicit
unbound call:

```python
return Scheduler.schedule(self)
```

This goes directly to native vLLM scheduling, bypasses `InstrumentedSchedulerMixin.schedule()`,
and lets the outer wrapper's wall-time measurement and `scheduler_call` JSONL record complete
normally after `_schedule_impl()` returns.

`return super()._schedule_impl()` is a technically valid alternative (it chains through
`InstrumentedSchedulerMixin._schedule_impl()` → `Scheduler.schedule(self)`), but it is more
indirect and breaks if `InstrumentedSchedulerMixin._schedule_impl()` is ever changed. Prefer the
explicit form.

---

## 2. Exact Non-Mutating Snapshot Plan

### Iterating `self.waiting`

`self.waiting` is a `RequestQueue` instance — either `FCFSRequestQueue` (a `deque` subclass) or
`PriorityRequestQueue` (heap-backed). Both implement `__iter__` safely:

- `FCFSRequestQueue.__iter__` -> `deque.__iter__()` — non-destructive.
- `PriorityRequestQueue.__iter__` -> iterates over `self._heap[:]` (a copy of the heap list) —
  non-destructive.

Safe snapshot call:
```python
waiting_requests = list(self.waiting)
```

**Do not** call `peek_request()` or `pop_request()`. Do not mutate order.

### Iterating `self.skipped_waiting`

Same type and same `__iter__` safety guarantee as `self.waiting`:
```python
skipped_requests = list(self.skipped_waiting)
```

### Iterating `self.running`

`self.running` is `list[Request]`. Standard iteration is non-destructive:
```python
running_requests = list(self.running)
```

### Request fields safe to read (non-mutating)

| Field | Type | Note |
|---|---|---|
| `request.request_id` | `str` | Stable key |
| `request.arrival_time` | `float` | Wall-clock timestamp |
| `request.status` | `RequestStatus` | Use to classify |
| `request.num_computed_tokens` | `int` | Prefill progress |
| `request.num_prompt_tokens` | `int` | Total prompt length |
| `request.num_tokens` | `int` (property) | `len(_all_token_ids)`, includes output tokens |
| `request.num_tokens_with_spec` | `int` (property) | `len(_all_token_ids) + len(spec_token_ids)` |
| `request.num_output_placeholders` | `int` | Async scheduling placeholder count |
| `request.spec_token_ids` | `list[int]` | Non-empty -> speculative decoding |
| `request.pooling_params` | `PoolingParams or None` | Non-None -> pooling request |
| `request.has_encoder_inputs` | `bool` (property) | True -> multimodal |
| `request.kv_transfer_params` | `dict or None` | Non-None -> KV transfer |
| `request.num_preemptions` | `int` | Preemption history |

### Fields that must not be mutated

All of the above. Additionally, do **not** write to:
- `request.status`
- `request.num_computed_tokens`
- `request.spec_token_ids`
- `request.num_output_placeholders`
- `request.is_prefill_chunk`

Do **not** call any method that mutates request or scheduler state:
- `request_queue.pop_request()`
- `request_queue.prepend_request()`
- `self.kv_cache_manager.allocate_slots()`
- `self.kv_cache_manager.free()`
- `self._preempt_request()`
- `self.running.append()` / `.remove()` / `.pop()`

---

## 3. Conservative `LPRequestSnapshot` Mapping

The existing `LPRequestSnapshot` dataclass at `vllm_scheduler_policies/primal_lp/types.py` is the
target type. The following maps each field from a live vLLM `Request`.

### `request_id`
```python
request_id = request.request_id
```
Directly available. Stable string key across all scheduler state dicts.

### `arrival_time`
```python
arrival_time = request.arrival_time
```
Directly available. Wall-clock float from `time.time()` at request creation. Used for age-based
weights `beta_i(t)` and `gamma_i(t)`.

### `remaining_prefill_tokens`

For a **waiting or preempted** request (not yet running, or restarting):
```python
remaining_prefill_tokens = max(request.num_tokens - request.num_computed_tokens, 0)
```

Use `request.num_tokens` (not `request.num_prompt_tokens`) because resumed requests include output
tokens in `_all_token_ids`, and the native scheduler uses `num_tokens - num_computed_tokens` as
the scheduled-work formula (verified in Phase 12.1, scheduler.py:677).

For a **running** request in decode phase, `remaining_prefill_tokens = 0` (decode_eligible will be
True and prefill_eligible will be False for those requests).

### `max_prefill_chunk_tokens`

Conservative planning estimate (not an exact vLLM computed value):
```python
threshold = self.scheduler_config.long_prefill_token_threshold
if threshold and threshold > 0:
    max_prefill_chunk_tokens = min(threshold, self.max_num_scheduled_tokens)
else:
    max_prefill_chunk_tokens = self.max_num_scheduled_tokens
```

This mirrors the native scheduler's `num_new_tokens = threshold if 0 < threshold < num_new_tokens`
logic at scheduler.py:413-414 and scheduler.py:678-680. This is a planning upper bound only;
actual native allocation may further truncate it.

### `prefill_eligible`
```python
prefill_eligible = (
    request.status in (RequestStatus.WAITING, RequestStatus.PREEMPTED)
    or (
        request.status == RequestStatus.RUNNING
        and request.num_computed_tokens < request.num_tokens
    )
) and remaining_prefill_tokens > 0
```

Only simple generative non-encoder non-pooling requests reach this point (after unsupported-state
filtering). `PREEMPTED` requests have `num_computed_tokens == 0` after `_preempt_request()`
resets it (scheduler.py:977).

### `decode_eligible`

Conservative decode-eligibility for simple generative requests:
```python
decode_eligible = (
    request.status == RequestStatus.RUNNING
    and request.num_output_placeholders == 0   # no async placeholders
    and len(request.spec_token_ids) == 0        # no speculative tokens
    and (
        request.num_tokens_with_spec
        + request.num_output_placeholders
        - request.num_computed_tokens
    ) > 0
    and request.num_computed_tokens >= request.num_prompt_tokens
)
```

The third condition matches the running-loop formula from scheduler.py:408-412. The
`num_computed_tokens >= num_prompt_tokens` check is the primary decode-readiness signal for simple
generative requests (verified in Phase 12.1 as item 7). Do not use
`num_prompt_tokens - num_computed_tokens == 0` directly; it is insufficient for resumed requests.

### `preemptible`
```python
preemptible = request.status == RequestStatus.RUNNING
```

Only RUNNING requests can be preempted. `_preempt_request()` asserts
`request.status == RequestStatus.RUNNING` at scheduler.py:971.

### `prefill_memory_blocks_per_token`

Conservative per-token block cost estimate:
```python
prefill_memory_blocks_per_token = 1.0 / self.block_size
```

`self.block_size` is set at scheduler.py:153 from the `block_size` constructor argument. One KV
block holds `block_size` tokens, so each token costs `1/block_size` blocks on average. This is a
planning approximation; actual allocation may differ at block boundaries due to prefix caching
hits. Acceptable for Phase 12.4b dry-run.

### `decode_memory_blocks`

Conservative per-decode-step block cost:
```python
decode_memory_blocks = 1.0 / self.block_size
```

In the worst case, a decode step crosses a block boundary and allocates one new block. In the
average case, no new block is needed. `1/block_size` is a conservative planning figure.

### `preempt_recoverable_blocks`

Count currently allocated blocks for the request across all KV cache groups:
```python
kv_blocks = self.kv_cache_manager.get_blocks(request.request_id)
# KVCacheBlocks.blocks is tuple[list[KVCacheBlock], ...]
preempt_recoverable_blocks = float(
    sum(len(group_blocks) for group_blocks in kv_blocks.blocks)
)
```

`kv_cache_manager.get_blocks(request_id)` is verified in Phase 12.1 (item 12) and implemented at
kv_cache_manager.py:526-528. It calls `coordinator.get_blocks(request_id)` which reads from
`manager.req_to_blocks` (kv_cache_coordinator.py:253-260, single_type_kv_cache_manager.py:73).
This is a read-only access path.

**Caveat**: For requests in `self.waiting` or `self.skipped_waiting` that have not been allocated
yet, `get_blocks()` returns an empty `KVCacheBlocks` — `preempt_recoverable_blocks` will be 0.0,
which is correct (nothing to recover from a request with no KV allocation).

---

## 4. KV-Block Planning Map

### Exact free KV-block API
```python
free_blocks = self.kv_cache_manager.block_pool.get_num_free_blocks()
```

`block_pool` is assigned at kv_cache_manager.py:150 from `self.coordinator.block_pool`.
`get_num_free_blocks()` is defined at block_pool.py:478-484 and reads
`self.free_block_queue.num_free_blocks`. This is a read-only integer access.

### Exact per-request allocated-block helper
```python
kv_blocks: KVCacheBlocks = self.kv_cache_manager.get_blocks(request_id)
total_allocated = sum(len(group) for group in kv_blocks.blocks)
```

`kv_cache_manager.get_blocks(request_id)` at kv_cache_manager.py:526 returns a `KVCacheBlocks`
with `.blocks: tuple[list[KVCacheBlock], ...]`. No mutation occurs. If the request has no blocks
yet, returns `empty_kv_cache_blocks` (all empty groups).

### Block size and block-boundary estimates

`self.block_size` is a plain `int` attribute set in `Scheduler.__init__()` at scheduler.py:153.
Available directly from a subclass.

Planning estimates:
- `c_i^P` (prefill): `1.0 / self.block_size` per token (conservative upper bound).
- `c_i^D` (decode): `1.0 / self.block_size` (conservative worst-case: block boundary).
- `c_i^Z` (preemption recovery): exact integer count from `get_blocks()`.

No helper computes exact incremental block cost for a candidate chunk without calling
`allocate_slots()`. The conservative per-token rate is the correct Phase 12.4b approximation.

### Fallback approximation

If `get_blocks()` raises or returns unexpected results for a request, fall back to
`preempt_recoverable_blocks = 0.0` and set a diagnostic field `kv_block_read_error: true` in the
log record. This prevents the dry-run from crashing on unexpected KV manager state.

---

## 5. Unsupported-State Rules

The dry-run LP should skip solving and log `lp_fallback: true` with a `lp_unsupported_reason`
field whenever **any** of the following conditions is detected during snapshot collection.

These checks are applied **per-request**. If any single request triggers an unsupported rule, the
entire batch falls back to default scheduling (do not attempt partial LP).

| Condition | Reason | Check |
|---|---|---|
| Speculative decoding | `spec_token_ids` active | `len(request.spec_token_ids) > 0` |
| Async output placeholders | Async scheduling in progress | `request.num_output_placeholders > 0` |
| Pooling request | Not a generative decode action | `request.pooling_params is not None` |
| KV transfer in-flight | Async KV recv pending | `request.status == RequestStatus.WAITING_FOR_REMOTE_KVS` |
| KV transfer params | Remote KV params present | `request.kv_transfer_params is not None` |
| Multimodal / encoder inputs | Encoder cache changes chunk sizing | `request.has_encoder_inputs` |
| Structured output grammar pending | Extra wait state | `request.status == RequestStatus.WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR` |
| Other non-simple status | Streaming wait or unknown | status not in `{WAITING, RUNNING, PREEMPTED}` |
| Missing fields | AttributeError on required field | catch `AttributeError`; log `lp_unsupported_reason: missing_field` |
| Zero-request universe | Nothing to optimize | `len(U_t) == 0`; not unsupported, but log `lp_num_requests: 0` |

Additionally, after the LP solve:

| Condition | Reason |
|---|---|
| Solver failure (`solution.success == False`) | SciPy/HiGHS returned infeasible or error; log `lp_solver_status` and fall back |
| `lp_fractional_rule_violation == True` (> 3 fractional requests) | LP structure mismatch or numerical issue; log as warning but do not fall back in 12.4b (dry-run only) |

---

## 6. Logging Plan

### Outer wrapper

`InstrumentedSchedulerMixin.schedule()` already emits one `scheduler_call` JSONL record per step
with `scheduler_wall_time_ms`, queue lengths, and output counts. Do **not** modify this path.

### LP dry-run record

Inside `_schedule_impl()`, after the LP path completes (or falls back), call
`self._instrumentation_write()` with a second record. This emits to the same file handle opened
for `SCHEDULER_POLICIES_ITER_LOG`, using the existing `_instrumentation_write()` method at
instrumentation.py:84-91.

```python
def _write_lp_dry_run_record(self, *, call_index: int, ...) -> None:
    record = {
        "event": "lp_dry_run",
        "call_index": call_index,         # same index as scheduler_call record
        "lp_enabled": True,
        # solver fields
        "lp_solver_status": ...,
        "lp_solver_message": ...,
        "lp_objective_value": ...,
        # request count fields
        "lp_num_requests": ...,
        "lp_num_integral_requests": ...,
        "lp_num_fractional_requests": ...,
        "lp_fractional_rule_violation": ...,
        # action count fields
        "lp_num_decode_actions": ...,
        "lp_num_prefill_actions": ...,
        "lp_num_prefill_tokens": ...,
        "lp_num_preemptions": ...,
        "lp_num_forced_preemptions": ...,
        # residual capacity fields
        "lp_token_budget_remaining": ...,
        "lp_sequence_budget_remaining": ...,
        "lp_memory_remaining": ...,
        # timing fields (milliseconds)
        "lp_build_time_ms": ...,
        "lp_solve_time_ms": ...,
        "lp_extract_time_ms": ...,
        # fallback fields
        "lp_fallback": False,
        "lp_unsupported_reason": None,
        "lp_dry_run_error": None,
    }
    self._instrumentation_write(record)
```

For fallback cases:
```python
record = {
    "event": "lp_dry_run",
    "call_index": call_index,
    "lp_enabled": True,
    "lp_fallback": True,
    "lp_unsupported_reason": reason_string,  # e.g. "spec_decode", "encoder_inputs"
}
```

For unexpected exceptions inside `_schedule_impl()` LP logic:
```python
"lp_dry_run_error": type(exc).__name__ + ": " + str(exc)
```

The LP record uses `"event": "lp_dry_run"` to distinguish it from `"event": "scheduler_call"` in
the same JSONL file.

### No new env var needed

`SCHEDULER_POLICIES_ITER_LOG` is the single logging gate. If it is unset,
`_instrumentation_write()` is a no-op (instrumentation.py:84-90), so the LP dry-run record is
also silently dropped.

---

## 7. Test Plan

All tests must run with `~/vllm-sched/.venv/bin/python`.
No server startup, no curl, no benchmarks.

### T1 — Import and compile check
```bash
~/vllm-sched/.venv/bin/python -c "
import vllm_scheduler_policies
from vllm_scheduler_policies.primal_lp_dry_run import PrimalLPDryRunScheduler
print('import ok')
"

~/vllm-sched/.venv/bin/python -m compileall vllm_scheduler_policies/primal_lp_dry_run.py
```

### T2 — MRO and method resolution check
```bash
~/vllm-sched/.venv/bin/python -c "
from vllm_scheduler_policies.primal_lp_dry_run import PrimalLPDryRunScheduler
from vllm_scheduler_policies.instrumentation import InstrumentedSchedulerMixin
from vllm.v1.core.sched.scheduler import Scheduler
assert issubclass(PrimalLPDryRunScheduler, InstrumentedSchedulerMixin)
assert issubclass(PrimalLPDryRunScheduler, Scheduler)
mro_names = [c.__name__ for c in PrimalLPDryRunScheduler.__mro__]
assert mro_names.index('InstrumentedSchedulerMixin') < mro_names.index('Scheduler')
assert hasattr(PrimalLPDryRunScheduler, '_schedule_impl')
print('MRO ok:', mro_names)
"
```

### T2b — Static source safety checks (no server required)

These checks verify that `_schedule_impl()` does not contain forbidden calls that would cause
recursion, mutation, or incorrect delegation. Run as part of the unit test suite.

```python
import inspect
from vllm.v1.core.sched.scheduler import Scheduler
from vllm_scheduler_policies.primal_lp_dry_run import PrimalLPDryRunScheduler

def test_schedule_impl_forbidden_strings():
    src = inspect.getsource(PrimalLPDryRunScheduler._schedule_impl)

    # Recursion guard: super().schedule() inside _schedule_impl() re-enters
    # InstrumentedSchedulerMixin.schedule() and causes infinite recursion.
    assert "super().schedule()" not in src, (
        "_schedule_impl must not call super().schedule() — use Scheduler.schedule(self)"
    )

    # Mutation guards: these APIs mutate KV state, queues, or request status.
    for forbidden in (
        "allocate_slots(",
        "_preempt_request(",
        "SchedulerOutput(",
        "pop_request(",
        "prepend_request(",
        "VLLM_SCHEDULER_ITER_LOG",
    ):
        assert forbidden not in src, (
            f"_schedule_impl must not contain '{forbidden}'"
        )

def test_schedule_impl_uses_explicit_delegation():
    src = inspect.getsource(PrimalLPDryRunScheduler._schedule_impl)
    # Delegation to native vLLM must be explicit.
    assert "Scheduler.schedule(self)" in src, (
        "_schedule_impl must delegate with Scheduler.schedule(self)"
    )
```

### T3 — Unit test: `_collect_lp_state_snapshot()` with fake Request objects

Create a minimal fake `Request`-like object and verify the snapshot helper produces the correct
`LPRequestSnapshot` values without accessing vLLM's real scheduler infrastructure.

Key assertions:
- A RUNNING request with `num_computed_tokens >= num_prompt_tokens` produces
  `decode_eligible=True`, `prefill_eligible=False`.
- A WAITING request with `num_computed_tokens=0` produces `decode_eligible=False`,
  `prefill_eligible=True`.
- A request with `has_encoder_inputs=True` is flagged unsupported.
- A request with `pooling_params is not None` is flagged unsupported.
- A request with `len(spec_token_ids) > 0` is flagged unsupported.
- A request with `status == WAITING_FOR_REMOTE_KVS` is flagged unsupported.

These tests do **not** need a live `Scheduler` object; they only test the classification/snapshot
helper in isolation using `SimpleNamespace` or `dataclass` fakes.

### T4 — Unit test: LP solve on synthetic snapshot
```bash
~/vllm-sched/.venv/bin/python -c "
from vllm_scheduler_policies.primal_lp.types import (
    LPRequestSnapshot, LPCapacities, LPUtilityWeights, RequestUtilityWeights,
)
from vllm_scheduler_policies.primal_lp.solver import solve_lp_relaxation
snap = LPRequestSnapshot(
    request_id='r0',
    remaining_prefill_tokens=0,
    decode_eligible=True,
    preemptible=True,
    decode_memory_blocks=0.0625,
    preempt_recoverable_blocks=4.0,
)
caps = LPCapacities(token_budget=8, sequence_budget=4, free_memory_blocks=16.0)
weights = LPUtilityWeights({'r0': RequestUtilityWeights(alpha=1e6, beta=1.0, gamma=1e9)})
plan = solve_lp_relaxation([snap], caps, weights)
print('solver ok, status:', plan.solver_status, 'decode:', plan.decode)
"
```

### T5 — No server, no curl

No further tests are needed for 12.4b. The dry-run scheduler logs LP plans without changing
outputs. Behavioral correctness testing requires a live server and is deferred to Phase 12.4c+.

The T2b static checks must run before any other test. If `test_schedule_impl_forbidden_strings`
fails, the implementation has a critical delegation error and no further testing is meaningful.

---

## 8. Explicit 12.4b Patch Boundaries

### Allowed files (new or modified)

| File | Action | Purpose |
|---|---|---|
| `vllm_scheduler_policies/primal_lp_dry_run.py` | **new** | `PrimalLPDryRunScheduler` class |
| `vllm_scheduler_policies/__init__.py` | may edit | re-export if needed (optional) |
| `tests/test_primal_lp_dry_run.py` | **new** | T3/T4 unit tests (no server) |

### Forbidden files (must not edit)

| File | Reason |
|---|---|
| `vllm_scheduler_policies/instrumentation.py` | Established outer wrapper; break = break all policies |
| `vllm_scheduler_policies/simple_policy_1.py` | Passthrough baseline must remain untouched |
| `vllm_scheduler_policies/primal_lp/solver.py` | Frozen LP layer; changes require separate phase |
| `vllm_scheduler_policies/primal_lp/extraction.py` | Same |
| `vllm_scheduler_policies/primal_lp/types.py` | Same |
| `vllm_scheduler_policies/primal_lp/weights.py` | Same |
| Any file under `~/vllm-sched/vllm/` | Never edit native vLLM source |

### Forbidden APIs (must not call from 12.4b scheduler)

| API | Why forbidden |
|---|---|
| `super().schedule()` inside `_schedule_impl()` | **Causes infinite recursion** — re-enters `InstrumentedSchedulerMixin.schedule()` which calls `self._schedule_impl()` again. Use `Scheduler.schedule(self)` instead. |
| `self.kv_cache_manager.allocate_slots(...)` | Mutates KV state; dry-run must be read-only |
| `self._preempt_request(request, timestamp)` | Mutates request status, KV, and queues |
| `SchedulerOutput(...)` constructor (manual) | Invalid without all native invariants; return value must come from `Scheduler.schedule(self)` |
| `request_queue.pop_request()` | Destructively removes from live queue |
| `request_queue.prepend_request()` | Mutates live queue |
| `self.running.append()` / `.remove()` | Mutates live running list |
| `VLLM_SCHEDULER_ITER_LOG` | Forbidden by AGENTS.md; use `SCHEDULER_POLICIES_ITER_LOG` |

### Validation commands

```bash
# T1: import check
~/vllm-sched/.venv/bin/python -c \
  "from vllm_scheduler_policies.primal_lp_dry_run import PrimalLPDryRunScheduler; print('ok')"

# T2: compile check
~/vllm-sched/.venv/bin/python -m compileall vllm_scheduler_policies/primal_lp_dry_run.py

# T2b: static source safety checks (run before T3/T4)
~/vllm-sched/.venv/bin/python -m pytest tests/test_primal_lp_dry_run.py \
  -k "forbidden_strings or explicit_delegation" -v

# T3/T4: unit tests (no server)
~/vllm-sched/.venv/bin/python -m pytest tests/test_primal_lp_dry_run.py -v

# Package import smoke
~/vllm-sched/.venv/bin/python -c "import vllm_scheduler_policies; print('ok')"
```

---

## Summary of Key Findings

### What already exists and is ready to use

- `InstrumentedSchedulerMixin` with `_instrumentation_write()` handles all JSONL output — the LP
  record can be emitted by calling this directly from `_schedule_impl()`.
- `solve_lp_relaxation()` and `extract_fractionals()` in the primal LP layer are complete and
  tested.
- `LPRequestSnapshot`, `LPCapacities`, `LPActionPlan` types are complete with validation.
- Both `RequestQueue` subclasses have safe `__iter__` implementations — `for req in self.waiting`
  is non-mutating in both FCFS and priority modes.

### What 12.4b must implement

- `PrimalLPDryRunScheduler._schedule_impl()` with the snapshot -> classify -> LP -> log ->
  delegate pattern.
- `_collect_lp_state_snapshot()` helper that reads `waiting`, `skipped_waiting`, and `running`
  and returns `(list[LPRequestSnapshot], LPCapacities, LPUtilityWeights, unsupported_reason)`.
- Unsupported-state classifier using the six field checks in Section 5.
- `preempt_recoverable_blocks` computation via `kv_cache_manager.get_blocks()`.
- `_write_lp_dry_run_record()` helper.

### Highest-risk item for 12.4b

`decode_eligible` derivation — must match the running-loop formula
(`num_tokens_with_spec + num_output_placeholders - num_computed_tokens > 0` and
`num_computed_tokens >= num_prompt_tokens`) while conservatively excluding all
non-simple-generative cases.

### AGENTS.md delegation guidance is ambiguous

AGENTS.md states:

> Custom policies may either:
> - run policy-specific logic inside `_schedule_impl()` and then delegate to `super().schedule()`

This sentence accurately describes `InstrumentedSchedulerMixin._schedule_impl()` itself, where
`super()` resolves to `Scheduler`. It must **not** be interpreted as permitting `super().schedule()`
inside a subclass override of `_schedule_impl()`, where `super()` resolves to
`InstrumentedSchedulerMixin` and the call is always recursive.

AGENTS.md should be updated in a future pass to read "delegate to `Scheduler.schedule(self)`"
rather than `super().schedule()` to make this safe for subclass authors.

---

**No files changed.**  
`git status --short`: *(empty — clean working tree)*
