# Project Status

This document is the concise current-state reference for the Unity vLLM
scheduler research project. Facts below are limited to repository contents and
the read-only Unity-local artifact audit.

## Repository State

### Unity-local Audited Runtime Baseline

- vLLM repo:
  - branch: `unity-phase4-v0.20.2-cu130`
  - commit: `bc150f50299199599673614f80d12a196f377655`
  - status: clean at audit time
- scheduler package:
  - branch: `master`
  - commit observed during read-only Unity-local audit:
    `169a86289b26d1dc508a88ab74a1616fd0716f15`
  - status: clean at audit time
  - latest scheduler commits observed during audit:
    - `169a862 documentation updates`
    - `405c574 Clarify scheduler delegation guidance`
    - `d4a0796 Add primal LP dry-run scheduler bridge`

### Current Documentation Cleanup State

Documentation cleanup baseline observed from GitHub before this status update:

- `b884102a88e0a9f9001b9a7c773fa20de2ff21c3`
- message: `Archive superseded documentation`

Documentation cleanup commits verified locally in `git log --oneline -10`:

- `d04ba5a Add project status documentation`
- `07456af Add scheduler experiment reference`
- `55b0bbd Clarify scheduler experiment reference aliases`
- `2f91931 phase_12_report updated`
- `4ac5d8f phase_12_report updated`
- `8c8cb80 Update agent documentation guidance`
- `b884102 Archive superseded documentation`

Superseded phase-specific docs have been moved to `docs/archive/` and are
historical. Canonical docs remain in `docs/project_status.md`,
`docs/scheduler_experiment_reference.md`, `docs/phase_12_report.md`, and
`docs/primal_lp_relaxation_scheduler.md`.

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
`tmp` artifacts corroborating Phase 12.3/12.4b validation outside the repo docs.

External evidence for Phase 12.3/12.4b validation is therefore unknown.

## Phase 13 Direction: Experiment Management Workflow

Phase 13 is a workflow and reproducibility phase.

The immediate goal is to stop relying on copy-pasted mega-scripts and to move
future experiment protocol, helper scripts, configs, and curated summaries into
tracked repository paths under `experiments/`.

Phase 13 should keep raw run artifacts inside the repository tree for easy local
navigation, but raw artifacts should be ignored by Git by default. Examples
include server logs, benchmark logs, scheduler JSONL files, pid files, transient
health-check output, and per-run scratch files.

The initial experiment-management design is documented in
`experiments/README.md`.

Phase 13 should not change scheduler behavior, implement real LP action
translation, or broaden benchmark coverage beyond small harness-validation runs
unless a later phase explicitly chooses to do so.

For `PrimalLPDryRunScheduler` overhead measurements, summaries must distinguish
between total dry-run scheduler overhead and isolated LP solver time. Unless a
dedicated LP-specific timing field exists, the measured timing field is
`scheduler_wall_time_ms`.
