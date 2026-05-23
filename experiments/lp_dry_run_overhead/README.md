# LP Dry-Run Overhead Experiment

This experiment directory is for measuring the overhead of
`PrimalLPDryRunScheduler` relative to `simple_policy_1`.

The initial Phase 13 goal is not to run a broad benchmark campaign. The goal is
to create a reproducible, inspectable harness for small scheduler experiments.

## Scientific question

How much scheduler-side overhead is added by the LP dry-run planning path when
compared with the behavior-preserving instrumented baseline?

## Compared schedulers

- `simple_policy_1`: behavior-preserving instrumented baseline.
- `primal_lp_dry_run`: diagnostic LP dry-run scheduler.

`PrimalLPDryRunScheduler` must return native scheduler output unchanged. This
experiment should not change scheduler behavior.

## Measurement caveat

Unless a dedicated LP-specific timing field exists in the scheduler JSONL
records, this experiment measures total dry-run scheduler overhead using
`scheduler_wall_time_ms`.

It must not claim isolated LP solver time unless the logs contain a real
LP-specific timing field.

## Directory layout

```text
configs/     tracked small run configs
scripts/     tracked experiment wrapper scripts
summaries/   tracked curated summaries
runs/        gitignored raw run artifacts
```

## Raw run policy

Raw run artifacts should be kept under:

```text
runs/<timestamp>_<hostname>/
```

Raw run artifacts are useful for local inspection but should not be committed by
default.

Examples include:

```text
server.log
server.pid
bench_*.log
scheduler_iter.jsonl
analysis.json
analysis.txt
record.md
```

Curated summaries may be copied or regenerated into `summaries/` and committed.

## Safety boundaries

Do not run vLLM server or benchmark jobs on login nodes.

Do not edit scheduler behavior in this experiment phase.

Do not broaden the benchmark matrix without an explicit later-phase decision.

Do not kill arbitrary processes. Stop only the server identified by the current
run directory's pid file, unless a human has inspected the situation and decided
otherwise.