# Experiment Management Workflow

This directory contains tracked experiment code, lightweight configs, curated
summaries, and documentation for Unity vLLM scheduler experiments.

Raw run artifacts should live under per-experiment `runs/` directories inside
this tree for easy navigation, but they should not be tracked by Git by default.
Examples of raw artifacts include server logs, benchmark logs, scheduler JSONL
files, pid files, temporary health-check output, and transient run records.

## Goals

The experiment workflow should make future scheduler experiments:

- reproducible from tracked scripts and configs;
- inspectable through GitHub without pasting large logs into chat;
- safe to run on Unity compute/GPU nodes;
- explicit about what was measured and what was not measured;
- small enough to avoid becoming a general experiment framework too early.

## Non-goals

Phase 13 does not change scheduler behavior.

Phase 13 does not implement real LP action translation.

Phase 13 does not broaden benchmark coverage beyond small harness-validation
runs unless a later phase explicitly chooses to do so.

Phase 13 does not make raw logs canonical Git artifacts.

## Recommended layout

```text
experiments/
  README.md
  common/
    env.sh
    run_lib.sh
    analyze_scheduler_jsonl.py
    make_record.py
  lp_dry_run_overhead/
    README.md
    configs/
    scripts/
    summaries/
    runs/
```

`experiments/common/` should contain small reusable helper scripts.

Each experiment directory should contain its own purpose, protocol, configs,
scripts, curated summaries, and gitignored raw runs.

## Artifact policy

Tracked by default:

- experiment README files;

- reusable helper scripts;

- small configuration files;

- analysis and record-generation code;

- curated summaries in `summaries/`.


Ignored by default:

- raw run directories;

- `*.log`;

- `*.jsonl`;

- `*.pid`;

- transient health-check and response files;

- server and benchmark stdout/stderr captures.


A raw run directory should be self-describing enough for local inspection. A
typical run may contain:

```text
manifest.json
commands.sh
env.txt
git_state.txt
server.log
server.pid
bench_*.log
scheduler_iter.jsonl
analysis.json
analysis.txt
record.md
```

Only compact curated summaries should be copied into tracked `summaries/`
files.

## Measurement discipline

`SCHEDULER_POLICIES_ITER_LOG` is the scheduler JSONL logging environment
variable.

`VLLM_SCHEDULER_ITER_LOG` must not be used for new experiments.

When comparing `PrimalLPDryRunScheduler` against `simple_policy_1`, the dry-run
scheduler returns native scheduler output unchanged. Therefore, unless a
dedicated LP-specific timing field exists, the measured quantity is total
dry-run scheduler overhead via `scheduler_wall_time_ms`, not isolated LP solver
time.

Any generated summary must state whether LP-specific timing is available and
which timing field was used.

## Unity safety rules

Do not run vLLM server or benchmark jobs on login nodes.

Do not use `set -euo pipefail` in Unity remote commands for this project.

Do not kill arbitrary Python, vLLM, or engine processes from generic process
searches. Stop only the server recorded by the current run's pid file unless a
human has inspected the situation and decided otherwise.

## Agent usage rules

Codex CLI or Claude CLI may help create or refactor experiment harness files,
docs, configs, and analysis scripts.

They should not autonomously broaden benchmark matrices.

They should not edit scheduler behavior during Phase 13.

They should not run server or benchmark jobs on login nodes.

They should not kill arbitrary processes without explicit instructions.
