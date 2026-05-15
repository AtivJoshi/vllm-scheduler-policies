# Agent Instructions for vLLM Scheduler Policies

These instructions apply to AI-assisted work in this external scheduler package:

`~/vllm-sched/vllm-scheduler-policies`

## Project

This package contains external scheduler classes for a vLLM scheduler research project on Unity HPC.

The current goal is incremental, reproducible scheduler experimentation for vLLM v0.20.2. The first custom scheduler must preserve default vLLM scheduling behavior unless a task explicitly says otherwise.

## Relevant files

Primary external package files:

- `vllm_scheduler_policies/baseline.py`
- `vllm_scheduler_policies/simple_policy_1.py`
- `vllm_scheduler_policies/instrumentation.py`
- `vllm_scheduler_policies/common.py`
- `scripts/serve.sh`
- `scripts/bench.sh`
- `pyproject.toml`

Relevant vLLM reference files, read-only unless explicitly instructed:

- `../vllm/vllm/v1/core/sched/scheduler.py`
- `../vllm/vllm/config/scheduler.py`
- relevant vLLM v1 scheduler tests under `../vllm/tests/`

## Hard rules

- Do not edit CUDA, C++, Triton, or compiled-kernel code.
- Do not modify native vLLM scheduler files unless explicitly instructed.
- Prefer external scheduler classes in this package.
- Keep patches small and reviewable.
- Preserve default scheduling behavior unless the task explicitly asks for a behavior change.
- Do not change launcher semantics unless explicitly asked.
- Do not use or introduce the old `VLLM_SCHEDULER_ITER_LOG` environment variable.
- Use `SCHEDULER_POLICIES_ITER_LOG` for scheduler JSONL instrumentation.
- Do not disable FlashInfer as part of scheduler work.
- Do not run broad test suites or long benchmarks unless explicitly asked.

## Scheduler instrumentation design

`InstrumentedSchedulerMixin.schedule()` is the outer template-method wrapper for scheduler timing and JSONL logging.

Future policy classes should override `_schedule_impl()`, not `schedule()`.

The default `_schedule_impl()` delegates to `super().schedule()`, preserving vLLM default scheduler behavior. Custom policies may either:

- run policy-specific logic inside `_schedule_impl()` and then delegate to `super().schedule()`, or
- fully implement `_schedule_impl()` and return a valid vLLM `SchedulerOutput`.

Do not bypass `InstrumentedSchedulerMixin.schedule()` in instrumented policy classes, because doing so excludes custom policy work from `scheduler_wall_time_ms`.

## Environment rules

Use the existing project environment:
```bash
    source ~/vllm-sched/docs/unity_cuda_13_1_env.sh
    source ~/vllm-sched/.venv/bin/activate
```
Use the project Python: `~/vllm-sched/.venv/bin/python`

Do not use system `python3`, bare `pip`, or global installs.

Avoid shell patterns that have caused remote instability, especially: `set -euo pipefail`

## First Codex tasks must be read-only

For initial inspection tasks:

- inspect scheduler control flow
- identify extension points
- explain queue and scheduler state variables
- explain how `--scheduler-cls` loads custom schedulers
- propose minimal changes only in prose
- do not edit files

Only edit files after the human explicitly asks for an implementation patch.

## Validation preference

Prefer targeted smoke checks: `~/vllm-sched/.venv/bin/python -c "import vllm_scheduler_policies; print('ok')"`

For server/benchmark checks, use the existing launcher scripts and save outputs under `~/vllm-sched/results/`.

Do not benchmark on login nodes.
