# Phase 13.8 Small Overhead Matrix Summary

Date: 2026-05-24
Host: `gpu048`
Experiment: `lp_dry_run_overhead`
Matrix config: `experiments/lp_dry_run_overhead/configs/qwen3_0_6b_small_matrix.env`

## Purpose

This run used the Phase 13 experiment harness to execute a small reproducible
overhead matrix.

The goal was to move from a one-off harness smoke test to a tiny repeated
comparison across three seeds. This is still not a broad benchmark campaign.

## Matrix

Schedulers:

```text
simple_policy_1
primal_lp_dry_run
```

Seeds:

```text
0 1 2
```

Shared workload:

```text
model=Qwen/Qwen3-0.6B
served_model_name=qwen3-0.6b
num_prompts=4
random_input_len=32
random_output_len=16
max_concurrency=2
request_rate=inf
```

Raw matrix root:

```text
experiments/lp_dry_run_overhead/runs/20260524T054916Z_gpu048_matrix
```

The raw run directory is intentionally gitignored.

## Cell summary

```text
cell,scheduler,seed,success,failed,throughput,wall_mean_ms,wall_p95_ms,wall_max_ms,scheduler_calls,lp_records
primal_lp_dry_run_seed0,primal_lp_dry_run,0,4,0,5.06,1.68407,3.43679,10.593,68,68
primal_lp_dry_run_seed1,primal_lp_dry_run,1,4,0,4.73,1.69183,4.7294,12.7569,67,67
primal_lp_dry_run_seed2,primal_lp_dry_run,2,4,0,4.72,1.64057,3.70371,5.68476,68,68
simple_policy_1_seed0,simple_policy_1,0,4,0,5.42,0.0370955,0.0702168,0.182228,68,0
simple_policy_1_seed1,simple_policy_1,1,4,0,5.97,0.0373553,0.0737731,0.212744,68,0
simple_policy_1_seed2,simple_policy_1,2,4,0,5.8,0.0377146,0.0768679,0.210992,68,0
```

## Grouped summary

### `simple_policy_1`

```text
num_cells=3
seeds=['0', '1', '2']
all_scheduler_calls_ok=True
total_failed_requests=0
mean_of_scheduler_wall_time_ms_mean=0.0373885
mean_of_scheduler_wall_time_ms_p95=0.0736193
mean_request_throughput_req_s=5.73
lp_specific_timing_available_any=False
overhead_metric_used=['scheduler_wall_time_ms']
```

### `primal_lp_dry_run`

```text
num_cells=3
seeds=['0', '1', '2']
all_scheduler_calls_ok=True
total_failed_requests=0
mean_of_scheduler_wall_time_ms_mean=1.67216
mean_of_scheduler_wall_time_ms_p95=3.95663
mean_request_throughput_req_s=4.83667
lp_specific_timing_available_any=False
overhead_metric_used=['scheduler_wall_time_ms']
```

## Comparison

```text
mean_wall_time_delta_ms=1.63477
mean_wall_time_ratio=44.7239
```

Interpreted narrowly, the LP dry-run scheduler added approximately `1.63 ms`
of mean scheduler wall time per scheduler call relative to `simple_policy_1`
on this tiny three-seed matrix.

The mean scheduler wall-time ratio was approximately `44.7x`, but the absolute
baseline is very small, so the ratio is less informative than the millisecond
delta.

## Measurement caveat

No LP-specific timing field was present in the JSONL records.

Therefore this matrix compares total scheduler wall time via:

```text
scheduler_wall_time_ms
```

It does not measure isolated LP solver time.

## Interpretation

The Phase 13 matrix harness worked end to end.

All six matrix cells completed successfully:

```text
successful requests per cell: 4
failed requests per cell: 0
```

The baseline cells produced only `scheduler_call` records. The LP dry-run cells
produced paired `scheduler_call` and `lp_dry_run` records.

The LP dry-run cells did not expose LP-specific timing. Therefore, the measured
difference includes all additional work inside the dry-run scheduler path,
including LP snapshotting, LP planning, dry-run diagnostic logging, and any
solver overhead.

## Comparison with Phase 13.7 smoke result

The earlier one-off Phase 13.7 LP smoke run reported a much larger mean
`scheduler_wall_time_ms` of approximately `12.745 ms`.

This Phase 13.8 three-seed matrix reported a lower LP dry-run mean of
approximately `1.672 ms`.

This difference means the Phase 13.7 one-off run should be treated as harness
validation evidence, not as a stable overhead estimate. The Phase 13.8 matrix is
a better small-run estimate, but it is still tiny and should not be treated as a
full benchmark campaign.

## Cleanup

The matrix completed with return code `0`.

GPU memory returned to:

```text
NVIDIA A16, 0 MiB
```

## Follow-up

Recommended next steps:

1. Commit the matrix summarizer and this tracked summary.

2. Optionally add a future LP-specific timing field so solver/planner time can
    be separated from total scheduler wall time.

3. Keep any larger benchmark matrix as a later explicit phase, not as an
    automatic expansion of Phase 13.
