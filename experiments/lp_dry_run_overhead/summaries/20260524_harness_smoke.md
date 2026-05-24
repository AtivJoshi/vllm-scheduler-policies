# Phase 13.7 Harness Smoke Summary

Date: 2026-05-24
Host: `gpu048`
Experiment: `lp_dry_run_overhead`
Config: `experiments/lp_dry_run_overhead/configs/qwen3_0_6b_tiny.env`

## Purpose

This smoke test validated the Phase 13 experiment harness on a live Unity GPU
node.

The goal was not to run a statistically robust benchmark campaign. The goal was
to verify that the new tracked harness can:

- start a vLLM server with a selected scheduler;
- run one tiny benchmark workload;
- write scheduler JSONL to the intended raw run directory;
- analyze the JSONL with the tracked analyzer;
- generate a compact run record;
- stop the server cleanly;
- keep raw artifacts out of Git.

## Compared schedulers

- `simple_policy_1`
- `primal_lp_dry_run`

Both runs used:

- model: `Qwen/Qwen3-0.6B`
- served model name: `qwen3-0.6b`
- random workload: 4 prompts
- random input length: 32
- random output length: 16
- max concurrency: 2
- seed: 0

## Raw run directories

Baseline run:

```text
experiments/lp_dry_run_overhead/runs/20260524T043922Z_gpu048_simple_policy_1
```

LP dry-run run:

```text
experiments/lp_dry_run_overhead/runs/20260524T045827Z_gpu048_primal_lp_dry_run
```

These raw run directories are intentionally gitignored.

## Baseline result: `simple_policy_1`

Benchmark result:

```text
Successful requests: 4
Failed requests: 0
Request throughput (req/s): 5.33
Mean TTFT (ms): 130.32
Mean TPOT (ms): 16.10
```

Scheduler JSONL analysis:

```text
num_records_total=68
event_counts={"scheduler_call": 68}
num_scheduler_call_records=68
num_lp_dry_run_records=0
all_scheduler_calls_ok=True
scheduler_wall_time_ms.mean=0.0380811
scheduler_wall_time_ms.median=0.034057
scheduler_wall_time_ms.p95=0.0766469
scheduler_wall_time_ms.max=0.199063
total_scheduled_tokens=220
total_preemptions=0
```

## LP dry-run result: `primal_lp_dry_run`

Benchmark result:

```text
Successful requests: 4
Failed requests: 0
Request throughput (req/s): 1.68
Mean TTFT (ms): 239.62
Mean TPOT (ms): 62.63
```

Scheduler JSONL analysis:

```text
num_records_total=136
event_counts={"lp_dry_run": 68, "scheduler_call": 68}
num_scheduler_call_records=68
num_lp_dry_run_records=68
all_scheduler_calls_ok=True
scheduler_wall_time_ms.mean=12.7449
scheduler_wall_time_ms.median=13.6326
scheduler_wall_time_ms.p95=27.4164
scheduler_wall_time_ms.max=41.0502
total_scheduled_tokens=220
total_preemptions=0
lp_fallback_counts={"False": 68}
lp_unsupported_reason_counts={"<none>": 68}
lp_solver_success_counts={"True": 68}
lp_solver_message_counts={"Optimization terminated successfully. (HiGHS Status 7: Optimal)": 67, "no_requests": 1}
lp_error_type_counts={"<none>": 68}
```

## Measurement caveat

No LP-specific timing field was present in the JSONL records.

Therefore, this smoke test compares total scheduler wall time via:

```text
scheduler_wall_time_ms
```

It does not measure isolated LP solver time.

## Interpretation

The harness worked end to end for both schedulers.

The baseline run produced only `scheduler_call` records, as expected for
`simple_policy_1`.

The LP dry-run run produced one `lp_dry_run` record for each `scheduler_call`
record, as expected for `PrimalLPDryRunScheduler`.

Both runs completed 4 successful requests and 0 failed requests. Both produced
68 scheduler-call records and 220 scheduled tokens. The LP dry-run run produced
68 LP diagnostic records, no LP errors, and no LP fallback records.

The observed mean scheduler wall time increased from approximately `0.038 ms`
for `simple_policy_1` to approximately `12.745 ms` for `primal_lp_dry_run`.
This is useful smoke-test evidence that the harness can expose dry-run overhead,
but it should not be treated as a statistically robust benchmark.

## Cleanup

Both servers were stopped using the run-directory pid file.

GPU memory returned to:

```text
NVIDIA A16, 0 MiB
```

## Follow-up fixes discovered

The initial `record.md` generator included `record.md` in its own artifact
inventory before writing the file, causing records to say `record.md` was
missing. This was fixed by removing `record.md` from the artifact inventory list
in `experiments/common/make_record.py`.

The live server log also warned about `VLLM_SCHED_VENV` being an unknown vLLM
environment variable. This is not the forbidden scheduler JSONL variable
`VLLM_SCHEDULER_ITER_LOG`, but it is a minor harness hygiene issue worth fixing
in a later cleanup step.