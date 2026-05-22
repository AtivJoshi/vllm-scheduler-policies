# Agent Instructions for vLLM Scheduler Policies

These instructions apply to AI-assisted work in this external scheduler package:

`~/vllm-sched/vllm-scheduler-policies`

## Project

This package contains external scheduler classes for a vLLM scheduler research project on Unity HPC.

The current goal is incremental, reproducible scheduler experimentation for vLLM v0.20.2. The first custom scheduler must preserve default vLLM scheduling behavior unless a task explicitly says otherwise.

## Engineering principles

This is a research codebase. Optimize for correctness, reproducibility, and clarity for future researchers.

- Prefer the simplest implementation that solves the current research problem.
- Make minimal, localized changes. Avoid unrelated refactors.
- Prefer existing project patterns over introducing new abstractions.
- Write explicit, readable code with clear names and straightforward control flow.
- Do not add dependencies unless they are necessary for the current task and justified.
- Do not optimize for performance until correctness is established and a bottleneck is measured.
- Document intent, assumptions, and non-obvious research decisions; do not comment obvious mechanics.
- Do not silently change scientific meaning, metrics, evaluation protocols, scheduling semantics, or experiment parameters.
- Keep generated artifacts, caches, checkpoints, logs, and results separate from source code.
- Do not fabricate experimental results, benchmark numbers, logs, citations, or file contents.

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

The default `InstrumentedSchedulerMixin._schedule_impl()` delegates to `super().schedule()`, preserving vLLM default scheduler behavior. This is safe only inside the mixin's own method, where `super()` reaches native vLLM `Scheduler`.

Custom policy subclasses such as `class MyScheduler(InstrumentedSchedulerMixin, Scheduler)` must not call `super().schedule()` from their own `_schedule_impl()` overrides, because that re-enters `InstrumentedSchedulerMixin.schedule()` and recursively calls `self._schedule_impl()`.

Custom policies may either:

- run policy-specific logic inside `_schedule_impl()` and then delegate to native/default scheduling with `Scheduler.schedule(self)` or a local helper that does exactly that, or
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

## Initial tasks must be read-only

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

For changes that affect algorithms, scheduler behavior, experiments, or measurements:

- Add or update only targeted tests that clarify intended behavior or prevent realistic regressions.
- Do not hard-code behavior merely to satisfy tests while violating the intended algorithm.
- Run the most relevant validation available after changes.
- If validation cannot be run because tools, dependencies, GPUs, datasets, or environment variables are missing, report exactly what could not be run and why.
- Do not claim validation succeeded unless it actually ran.
- Include the exact command that should be run in a correctly configured environment.
