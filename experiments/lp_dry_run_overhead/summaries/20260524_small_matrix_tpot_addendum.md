# Phase 13.9 Addendum: TPOT-Aware Reanalysis of Phase 13.8 Matrix

Date: 2026-05-24
Original matrix: `experiments/lp_dry_run_overhead/summaries/20260524_small_matrix.md`
Raw matrix root: `experiments/lp_dry_run_overhead/runs/20260524T054916Z_gpu048_matrix`

## Purpose

This addendum reanalyzes the existing Phase 13.8 three-seed matrix with the
updated Phase 13.9 matrix summarizer.

The original Phase 13.8 summary focused mainly on scheduler wall-time overhead
and request throughput. The updated summarizer also reports TPOT, TTFT, ITL,
output-token throughput, total scheduler wall-time, and derived policy
comparisons.

No new benchmark run was performed for this addendum.

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

## Updated cell summary

```text
cell,scheduler,seed,success,failed,req_throughput,mean_ttft_ms,mean_tpot_ms,median_tpot_ms,p99_tpot_ms,wall_mean_ms,wall_p95_ms,wall_sum_ms,wall_max_ms,scheduler_calls,lp_records
primal_lp_dry_run_seed0,primal_lp_dry_run,0,4,0,5.06,125.46,17.66,17.31,19.2,1.68407,3.43679,114.517,10.593,68,68
primal_lp_dry_run_seed1,primal_lp_dry_run,1,4,0,4.73,139.01,18.75,18.84,19.57,1.69183,4.7294,113.353,12.7569,67,67
primal_lp_dry_run_seed2,primal_lp_dry_run,2,4,0,4.72,146.42,18.21,17.75,20.22,1.64057,3.70371,111.559,5.68476,68,68
simple_policy_1_seed0,simple_policy_1,0,4,0,5.42,103.95,17.46,17.07,19.38,0.0370955,0.0702168,2.5225,0.182228,68,0
simple_policy_1_seed1,simple_policy_1,1,4,0,5.97,98.84,15.56,15.37,18.59,0.0373553,0.0737731,2.54016,0.212744,68,0
simple_policy_1_seed2,simple_policy_1,2,4,0,5.8,106.18,15.73,15.53,18.67,0.0377146,0.0768679,2.56459,0.210992,68,0
```

## Updated grouped summary

### `simple_policy_1`

```text
num_cells=3
seeds=['0', '1', '2']
all_scheduler_calls_ok=True
total_failed_requests=0
request_throughput_req_s: mean=5.73, median=5.8, stdev=0.281603, min=5.42, max=5.97
output_token_throughput_tok_s: mean=91.67, median=92.75, stdev=4.44942, min=86.78, max=95.48
mean_ttft_ms: mean=102.99, median=103.95, stdev=3.76299, min=98.84, max=106.18
mean_tpot_ms: mean=16.25, median=15.73, stdev=1.05133, min=15.56, max=17.46
median_tpot_ms: mean=15.99, median=15.53, stdev=0.938723, min=15.37, max=17.07
p99_tpot_ms: mean=18.88, median=18.67, stdev=0.434856, min=18.59, max=19.38
scheduler_wall_time_ms_mean: mean=0.0373885, median=0.0373553, stdev=0.000310854, min=0.0370955, max=0.0377146
scheduler_wall_time_ms_p95: mean=0.0736193, median=0.0737731, stdev=0.00332819, min=0.0702168, max=0.0768679
scheduler_wall_time_ms_sum: mean=2.54242, median=2.54016, stdev=0.0211381, min=2.5225, max=2.56459
num_scheduler_call_records: mean=68, median=68, stdev=0, min=68, max=68
num_lp_dry_run_records: mean=0, median=0, stdev=0, min=0, max=0
lp_specific_timing_available_any=False
overhead_metric_used=['scheduler_wall_time_ms']
```

### `primal_lp_dry_run`

```text
num_cells=3
seeds=['0', '1', '2']
all_scheduler_calls_ok=True
total_failed_requests=0
request_throughput_req_s: mean=4.83667, median=4.73, stdev=0.193477, min=4.72, max=5.06
output_token_throughput_tok_s: mean=77.42, median=75.75, stdev=3.1204, min=75.49, max=81.02
mean_ttft_ms: mean=136.963, median=139.01, stdev=10.6288, min=125.46, max=146.42
mean_tpot_ms: mean=18.2067, median=18.21, stdev=0.545008, min=17.66, max=18.75
median_tpot_ms: mean=17.9667, median=17.75, stdev=0.787676, min=17.31, max=18.84
p99_tpot_ms: mean=19.6633, median=19.57, stdev=0.516366, min=19.2, max=20.22
scheduler_wall_time_ms_mean: mean=1.67216, median=1.68407, stdev=0.0276297, min=1.64057, max=1.69183
scheduler_wall_time_ms_p95: mean=3.95663, median=3.70371, stdev=0.682414, min=3.43679, max=4.7294
scheduler_wall_time_ms_sum: mean=113.143, median=113.353, stdev=1.49017, min=111.559, max=114.517
num_scheduler_call_records: mean=67.6667, median=68, stdev=0.57735, min=67, max=68
num_lp_dry_run_records: mean=67.6667, median=68, stdev=0.57735, min=67, max=68
lp_specific_timing_available_any=False
overhead_metric_used=['scheduler_wall_time_ms']
```

## Updated comparison

```text
mean_tpot_ms_baseline_mean=16.25
mean_tpot_ms_treatment_mean=18.2067
mean_tpot_ms_delta=1.95667
mean_tpot_ms_ratio=1.12041
mean_tpot_delta_per_scheduler_mean_delta=1.19691

mean_ttft_ms_baseline_mean=102.99
mean_ttft_ms_treatment_mean=136.963
mean_ttft_ms_delta=33.9733
mean_ttft_ms_ratio=1.32987

request_throughput_req_s_baseline_mean=5.73
request_throughput_req_s_treatment_mean=4.83667
request_throughput_req_s_delta=-0.893333
request_throughput_req_s_ratio=0.844095

output_token_throughput_tok_s_baseline_mean=91.67
output_token_throughput_tok_s_treatment_mean=77.42
output_token_throughput_tok_s_delta=-14.25
output_token_throughput_tok_s_ratio=0.844551

scheduler_wall_time_ms_mean_baseline_mean=0.0373885
scheduler_wall_time_ms_mean_treatment_mean=1.67216
scheduler_wall_time_ms_mean_delta=1.63477
scheduler_wall_time_ms_mean_ratio=44.7239

scheduler_wall_time_ms_sum_baseline_mean=2.54242
scheduler_wall_time_ms_sum_treatment_mean=113.143
scheduler_wall_time_ms_sum_delta=110.6
scheduler_wall_time_ms_sum_ratio=44.5021
```

## Interpretation

The updated summary resolves the apparent mismatch between the Phase 13.7
one-off smoke TPOT result and the Phase 13.8 matrix scheduler-overhead result.

For the three-seed matrix, `primal_lp_dry_run` increased mean scheduler wall time
by approximately `1.63 ms` per scheduler call relative to `simple_policy_1`.

The corresponding mean TPOT increase was approximately `1.96 ms/token`.

Thus, for this three-seed matrix, the TPOT gap is much smaller than the one-off
Phase 13.7 smoke result and is roughly comparable to the scheduler-call overhead
increase.

The larger remaining latency effect is TTFT. Mean TTFT increased by
approximately `33.97 ms`, from `102.99 ms` to `136.96 ms`.

Request throughput and output-token throughput both decreased by about 15.5%.
This is consistent with additional scheduler-side overhead affecting the small
serving workload, but this addendum does not prove a causal mechanism.

## Measurement caveat

No LP-specific timing field was available.

Therefore, `scheduler_wall_time_ms` measures total scheduler path overhead for
`primal_lp_dry_run`, including snapshotting, LP planning, diagnostic logging, and
any solver overhead. It does not isolate LP solver time.

## Next step

Run the Phase 13.9 five-seed matrix using the updated summarizer and the new
5-seed config, then compare whether the TPOT, TTFT, throughput, and scheduler
wall-time deltas remain consistent across five seeds.