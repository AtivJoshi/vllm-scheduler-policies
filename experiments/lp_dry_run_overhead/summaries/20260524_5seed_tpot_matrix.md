# Phase 13.9 Five-Seed TPOT-Aware Matrix Summary

Date: 2026-05-24
Host: `gpu048`
Experiment: `lp_dry_run_overhead`
Config: `experiments/lp_dry_run_overhead/configs/qwen3_0_6b_5seed_matrix.env`
Raw matrix root: `experiments/lp_dry_run_overhead/runs/20260524T170432Z_gpu048_matrix`

## Purpose

This run compares `simple_policy_1` and `primal_lp_dry_run` over five seeds
using the Phase 13 experiment harness.

The goal is to measure not only scheduler-side overhead, but also whether the
scheduler overhead corresponds to differences in TPOT, TTFT, request throughput,
and output-token throughput.

This is still a small controlled matrix, not a broad benchmark campaign.

## Matrix

| Field | Value |
|---|---|
| Schedulers | `simple_policy_1`, `primal_lp_dry_run` |
| Seeds | `0`, `1`, `2`, `3`, `4` |
| Model | `Qwen/Qwen3-0.6B` |
| Served model name | `qwen3-0.6b` |
| Number of prompts | `4` |
| Random input length | `32` |
| Random output length | `16` |
| Max concurrency | `2` |
| Request rate | `inf` |

## Cell-level summary

| Cell | Scheduler | Seed | Success | Failed | Req/s | Mean TTFT ms | Mean TPOT ms | Median TPOT ms | P99 TPOT ms | Scheduler mean ms | Scheduler p95 ms | Scheduler total ms | Scheduler calls | LP records |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `primal_lp_dry_run_seed0` | `primal_lp_dry_run` | 0 | 4 | 0 | 5.14 | 121.84 | 17.60 | 17.48 | 18.97 | 1.72941 | 3.68742 | 117.600 | 68 | 68 |
| `primal_lp_dry_run_seed1` | `primal_lp_dry_run` | 1 | 4 | 0 | 4.88 | 126.55 | 18.76 | 18.38 | 20.49 | 1.93401 | 5.97568 | 129.579 | 67 | 67 |
| `primal_lp_dry_run_seed2` | `primal_lp_dry_run` | 2 | 4 | 0 | 4.99 | 125.84 | 18.20 | 17.95 | 19.16 | 1.27905 | 1.89849 | 85.6961 | 67 | 67 |
| `primal_lp_dry_run_seed3` | `primal_lp_dry_run` | 3 | 4 | 0 | 4.60 | 151.23 | 18.74 | 18.39 | 19.85 | 1.69970 | 5.52313 | 113.880 | 67 | 67 |
| `primal_lp_dry_run_seed4` | `primal_lp_dry_run` | 4 | 4 | 0 | 4.65 | 127.86 | 19.94 | 19.52 | 21.36 | 2.74936 | 7.16700 | 184.207 | 67 | 67 |
| `simple_policy_1_seed0` | `simple_policy_1` | 0 | 4 | 0 | 5.59 | 107.74 | 15.98 | 16.01 | 19.35 | 0.0375383 | 0.0750842 | 2.55261 | 68 | 0 |
| `simple_policy_1_seed1` | `simple_policy_1` | 1 | 4 | 0 | 5.91 | 105.35 | 15.34 | 15.06 | 18.99 | 0.0385415 | 0.0780791 | 2.62083 | 68 | 0 |
| `simple_policy_1_seed2` | `simple_policy_1` | 2 | 4 | 0 | 5.88 | 107.22 | 15.35 | 15.06 | 19.05 | 0.0378282 | 0.0748541 | 2.57232 | 68 | 0 |
| `simple_policy_1_seed3` | `simple_policy_1` | 3 | 4 | 0 | 5.61 | 119.73 | 15.61 | 15.21 | 19.45 | 0.0370570 | 0.0702979 | 2.51988 | 68 | 0 |
| `simple_policy_1_seed4` | `simple_policy_1` | 4 | 4 | 0 | 5.08 | 142.80 | 16.43 | 16.03 | 19.50 | 0.0376145 | 0.0761859 | 2.55779 | 68 | 0 |

## Grouped scheduler summary

### Request and throughput metrics

| Metric | `simple_policy_1` | `primal_lp_dry_run` |
|---|---:|---:|
| Cells | 5 | 5 |
| Failed requests | 0 | 0 |
| Successful requests, mean | 4.000 | 4.000 |
| Request throughput, mean req/s | 5.614 | 4.852 |
| Request throughput, stdev req/s | 0.333 | 0.228 |
| Output-token throughput, mean tok/s | 89.804 | 77.604 |
| Output-token throughput, stdev tok/s | 5.317 | 3.636 |

### Latency metrics

| Metric | `simple_policy_1` | `primal_lp_dry_run` |
|---|---:|---:|
| Mean TTFT, mean ms | 116.568 | 130.664 |
| Mean TTFT, stdev ms | 15.726 | 11.714 |
| Median TTFT, mean ms | 109.884 | 126.624 |
| P99 TTFT, mean ms | 224.962 | 245.376 |
| Mean TPOT, mean ms | 15.742 | 18.648 |
| Mean TPOT, stdev ms | 0.464 | 0.864 |
| Median TPOT, mean ms | 15.474 | 18.344 |
| P99 TPOT, mean ms | 19.268 | 19.966 |
| Mean ITL, mean ms | 14.756 | 17.482 |
| Median ITL, mean ms | 14.516 | 17.708 |
| P99 ITL, mean ms | 29.806 | 32.650 |

### Scheduler and LP diagnostic metrics

| Metric | `simple_policy_1` | `primal_lp_dry_run` |
|---|---:|---:|
| Scheduler wall time mean, mean ms | 0.0377159 | 1.87831 |
| Scheduler wall time mean, stdev ms | 0.000540885 | 0.541993 |
| Scheduler wall time p95, mean ms | 0.0749002 | 4.85034 |
| Scheduler wall time p95, stdev ms | 0.00287122 | 2.07073 |
| Total scheduler wall time, mean ms | 2.56468 | 126.193 |
| Total scheduler wall time, stdev ms | 0.0367802 | 36.2028 |
| Scheduler-call records, mean | 68.0 | 67.2 |
| LP dry-run records, mean | 0.0 | 67.2 |
| All scheduler calls OK | true | true |
| LP-specific timing available | false | false |
| Timing metric used | `scheduler_wall_time_ms` | `scheduler_wall_time_ms` |

## Direct policy comparison

| Metric | Baseline: `simple_policy_1` | Treatment: `primal_lp_dry_run` | Delta | Ratio |
|---|---:|---:|---:|---:|
| Mean scheduler wall time ms | 0.0377159 | 1.87831 | +1.84059 | 49.8014 |
| Scheduler p95 wall time ms | 0.0749002 | 4.85034 | +4.77544 | 64.7574 |
| Total scheduler wall time ms | 2.56468 | 126.193 | +123.628 | 49.2039 |
| Mean TPOT ms | 15.742 | 18.648 | +2.906 | 1.18460 |
| Median TPOT ms | 15.474 | 18.344 | +2.870 | 1.18547 |
| P99 TPOT ms | 19.268 | 19.966 | +0.698 | 1.03623 |
| Mean TTFT ms | 116.568 | 130.664 | +14.096 | 1.12093 |
| Request throughput req/s | 5.614 | 4.852 | -0.762 | 0.864268 |
| Output-token throughput tok/s | 89.804 | 77.604 | -12.200 | 0.864149 |

Additional derived comparison:

| Derived metric | Value |
|---|---:|
| Mean TPOT delta per scheduler-mean delta | 1.57884 |

## Interpretation

The five-seed matrix confirms that `primal_lp_dry_run` adds measurable scheduler
overhead relative to `simple_policy_1`.

The mean scheduler wall time increased by approximately **1.84 ms per scheduler
call**, from `0.0377 ms` to `1.8783 ms`.

Mean TPOT increased by approximately **2.91 ms/token**, from `15.742 ms` to
`18.648 ms`.

The TPOT increase is therefore moderate and roughly comparable to the
scheduler-call overhead, not orders of magnitude larger. The derived ratio

```text
mean_tpot_delta_per_scheduler_mean_delta=1.57884
```

suggests that each additional millisecond of mean scheduler-call overhead
corresponded to about `1.58 ms` of mean TPOT increase in this tiny workload.

The mean TTFT increase was approximately **14.10 ms**, from `116.568 ms` to
`130.664 ms`.

Throughput decreased by about **13.6%**:

- request throughput ratio: `0.864268`;

- output-token throughput ratio: `0.864149`.


This is consistent with scheduler-side overhead affecting small-workload serving
performance, but it does not by itself prove the precise causal mechanism.

## Relationship to earlier Phase 13 results

The original Phase 13.7 one-off smoke run showed a much larger TPOT gap. That
gap did not persist in the repeated matrix.

The Phase 13.8 three-seed matrix already suggested that the TPOT gap was much
smaller than the one-off smoke run. This Phase 13.9 five-seed matrix strengthens
that conclusion.

The current best small-run estimate is:

```text
scheduler overhead delta: ~1.84 ms per scheduler call
mean TPOT delta:          ~2.91 ms/token
mean TTFT delta:          ~14.10 ms
throughput ratio:         ~0.864x
```

## Measurement caveat

No LP-specific timing field was available.

Therefore, `scheduler_wall_time_ms` measures total scheduler-path overhead for
`primal_lp_dry_run`, including:

- snapshotting;

- LP planning;

- diagnostic logging;

- solver overhead;

- any other Python work inside the dry-run scheduler path.


It does **not** isolate LP solver time.

## Recommended next steps

1. Add LP-specific timing fields if we need to separate solver time from
    snapshot/logging overhead.

2. Keep this five-seed matrix as the current small-run estimate.

3. Do not broaden workload dimensions until the scheduler timing fields are more
    diagnostic.

4. If broadening later, vary one dimension at a time: concurrency, output length,
    or number of prompts.
