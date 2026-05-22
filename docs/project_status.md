# Project Status

This document is the concise current-state reference for the Unity vLLM
scheduler research project. Facts below are limited to repository contents and
the read-only Unity-local artifact audit.

## Repository State

Observed by read-only Unity-local audit:

- vLLM repo:
  - branch: `unity-phase4-v0.20.2-cu130`
  - commit: `bc150f50299199599673614f80d12a196f377655`
  - status: clean
- scheduler package:
  - branch: `master`
  - commit: `169a86289b26d1dc508a88ab74a1616fd0716f15`
  - status: clean
  - latest commits observed:
    - `169a862 documentation updates`
    - `405c574 Clarify scheduler delegation guidance`
    - `d4a0796 Add primal LP dry-run scheduler bridge`

## Environment Baseline

Phase 3/4 setup facts:

- vLLM target: `v0.20.2`
- vLLM commit short: `bc150f5`
- CUDA backend: `cu130`
- Unity CUDA module: `cuda/13.1`
- Python: `/home/atjoshi_umass_edu/vllm-sched/.venv/bin/python`
- Python version: `3.12.13`
- vLLM version: `0.20.2+precompiled`
- torch version: `2.11.0+cu130`
- `torch.version.cuda == 13.0`
- GPU: `NVIDIA A16`
- FlashInfer import succeeded.
- FlashInfer was not disabled.

## Implemented Scheduler Classes

- `BaselinePassthroughScheduler`: behavior-preserving subclass of native vLLM
  `Scheduler`; it overrides no methods.
- `SimplePolicy1Scheduler`: behavior-preserving instrumented scheduler.
- `PrimalLPDryRunScheduler`: dry-run diagnostic LP bridge that logs LP
  diagnostics and delegates to native scheduling unchanged.

## Instrumentation

`InstrumentedSchedulerMixin.schedule()` is the outer scheduler wrapper for
timing and JSONL logging.

`SCHEDULER_POLICIES_ITER_LOG` is the correct scheduler JSONL environment
variable.

`VLLM_SCHEDULER_ITER_LOG` is forbidden except in explicit warnings or historical
failure notes. A stale Phase 9 run using `VLLM_SCHEDULER_ITER_LOG` produced an
"Unknown vLLM environment variable" warning and should not be used as canonical
evidence.

## Critical Delegation Invariant

Future instrumented policies override `_schedule_impl()`, not `schedule()`.

Subclass `_schedule_impl()` implementations must not call `super().schedule()`.
They should delegate to native/default scheduling with `Scheduler.schedule(self)`
or a helper that does exactly that.

Calling `super().schedule()` from a subclass `_schedule_impl()` recursively
re-enters `InstrumentedSchedulerMixin.schedule()`, which calls
`self._schedule_impl()` again.

## Validated Experiment Milestones

### Phase 5 Standard vLLM Demo

- `Qwen/Qwen3-0.6B` served as `qwen3-0.6b`.
- Benchmark completed `8` successful requests and `0` failed requests.
- Request throughput: `0.9816 req/s`.
- Mean TTFT: `19.54 ms`.
- Mean TPOT: `8.46 ms`.
- GPU returned to `0 MiB` after corrected cleanup.

### Phase 7 External Scheduler Passthrough

- `BaselinePassthroughScheduler` loaded via `--scheduler-cls`.
- Default and passthrough servers each answered one chat request.
- Strict failure scans were clean.
- GPU returned to `0 MiB`.

### Phase 8 Launcher

- `default` omits `--scheduler-cls`.
- `passthrough` maps to
  `vllm_scheduler_policies.baseline.BaselinePassthroughScheduler`.
- Default and passthrough tiny smoke benchmarks each had `4` successful
  requests and `0` failed requests.

### Phase 9 JSONL Instrumentation

- Final Phase 9 scheduler commit:
  `bcef5ae0801d8af3c67c7a63739a4c0e8c463fb4`.
- `SCHEDULER_POLICIES_ITER_LOG` was used.
- Tiny benchmark had `4` successful requests and `0` failed requests.
- JSONL contained `101` records, all `ok: True`.
- Total scheduled tokens: `252`.
- Mean scheduler wall time: `0.033352091807023726 ms`.
- Max scheduler wall time: `0.19292300567030907 ms`.
- No preemptions.

### Phase 10.5 Template Method

- Template-method instrumentation was validated.
- `InstrumentedSchedulerMixin.schedule()` calls `_schedule_impl()`.
- `SimplePolicy1Scheduler` remains behavior-preserving.
- `compileall`, MRO check, `git diff --check`, one chat request, and tiny
  benchmark passed.
- JSONL contained `101` records and `0` errors.
- Total scheduled tokens: `250`.
- Max scheduler wall time: `0.2827920252457261 ms`.

## Primal LP Status

The pure-Python LP helper layer exists under
`vllm_scheduler_policies/primal_lp/`.

`LPActionPlan` is internal and is not a valid vLLM `SchedulerOutput`.

`PrimalLPDryRunScheduler` snapshots state, runs/logs the LP planner, and returns
native scheduler output unchanged.

The dry-run bridge must not call `allocate_slots()`, call `_preempt_request()`,
mutate queues/request/KV state, or construct `SchedulerOutput`.

Real action translation remains future work and is high-risk.

## Phase 12 Validation Evidence Status

Repo docs report that Phase 12.3/12.4b targeted tests passed.

The read-only Unity-local audit found no independent `logs`, `results`, or
`tmp` artifacts corroborating Phase 12.3/12.4 validation outside the repo docs.

External evidence for Phase 12.3/12.4 validation is therefore unknown.

## Next Recommended Phase

1. D1/D2 doc consolidation first.
2. Then create `docs/scheduler_experiment_reference.md`.
3. Then sanitize the Phase 12 report.
4. Only after docs are canonical, run a Phase 12.4c/12.5 server experiment to
   measure dry-run LP overhead before attempting real action translation.
