## Phase 14: LP dry-run timing attribution

### Goal

Phase 14 added component-level timing attribution for `PrimalLPDryRunScheduler`.
The scheduler remains diagnostic-only: it snapshots scheduler state, solves the
synthetic LP, logs timing/plan summaries, and returns native vLLM scheduling
output unchanged.

### Code changes

Implemented timing fields for:

- `scheduler_wall_time_ms`: full instrumented scheduler path.
- `native_schedule_wall_time_ms`: time spent in native `Scheduler.schedule(self)`.
- `lp_dry_run_total_wall_time_ms`: LP dry-run path excluding native scheduling.
- `lp_snapshot_wall_time_ms`: read-only scheduler/KV snapshot construction.
- `lp_weight_wall_time_ms`: default-like LP utility weight construction.
- `lp_relaxed_lp_build_wall_time_ms`: LP variable/objective/constraint setup.
- `lp_highs_solve_wall_time_ms`: SciPy/HiGHS solve time only.
- `lp_relaxed_solution_parse_wall_time_ms`: conversion of SciPy result to relaxed solution.
- `lp_extract_wall_time_ms`: extraction of integral/fractional action plan.
- `lp_plan_wall_time_ms`: solve plus extraction path.
- `lp_summary_wall_time_ms`: diagnostic summary generation.
- `lp_log_prepare_wall_time_ms`: JSONL record preparation.

The analyzers now aggregate these fields explicitly and preserve backward
compatibility with old logs that only contain `scheduler_wall_time_ms`.

### Validation

#### Tiny real-run validation

Run root:

```text
experiments/lp_dry_run_overhead/runs/20260621T183653Z_gpu048_matrix
```

This run used two cells:

```text
simple_policy_1 seed0
primal_lp_dry_run seed0
```

Result:

```text
simple_policy_1:
  scheduler_call records = 67
  lp_dry_run records = 0

primal_lp_dry_run:
  scheduler_call records = 67
  lp_dry_run records = 67
  lp_specific_timing_available = True
```

#### Five-seed timing matrix

Run root:

```text
experiments/lp_dry_run_overhead/runs/20260621T184424Z_gpu048_matrix
```

Configuration:

```text
model = Qwen/Qwen3-0.6B
schedulers = simple_policy_1 primal_lp_dry_run
seeds = 0 1 2 3 4
num_prompts = 4
random_input_len = 32
random_output_len = 16
max_concurrency = 2
request_rate = inf
node = gpu048
GPU = NVIDIA A16
```

Grouped scheduler timing:

```text
simple_policy_1 scheduler_wall_time_ms.mean:     0.0366086 ms
primal_lp_dry_run scheduler_wall_time_ms.mean:   1.24181 ms
scheduler_wall_time_ms delta:                    1.20521 ms

primal_lp_dry_run native_schedule_wall_time_ms.mean:  0.0401173 ms
primal_lp_dry_run lp_dry_run_total_wall_time_ms.mean: 1.10158 ms
primal_lp_dry_run lp_highs_solve_wall_time_ms.mean:   1.02567 ms
```

Component attribution:

```text
lp_snapshot_wall_time_ms.mean:                 0.0281774 ms
lp_weight_wall_time_ms.mean:                   0.0056243 ms
lp_relaxed_lp_build_wall_time_ms.mean:         0.0210228 ms
lp_highs_solve_wall_time_ms.mean:              1.02567 ms
lp_relaxed_solution_parse_wall_time_ms.mean:   0.00755857 ms
lp_extract_wall_time_ms.mean:                  0.0145378 ms
lp_plan_wall_time_ms.mean:                     1.0537 ms
lp_summary_wall_time_ms.mean:                  0.00323194 ms
lp_log_prepare_wall_time_ms.mean:              0.00455914 ms
lp_dry_run_total_wall_time_ms.mean:            1.10158 ms
```

Interpretation:

Most of the LP dry-run overhead comes from the SciPy/HiGHS solve:

```text
lp_highs_solve / lp_dry_run_total ≈ 1.02567 / 1.10158 ≈ 93.1%
```

The LP dry-run accounts for most of the full scheduler wall time:

```text
lp_dry_run_total / scheduler_wall_time ≈ 1.10158 / 1.24181 ≈ 88.7%
```

The native vLLM scheduler call remains small:

```text
native_schedule_wall_time_ms.mean ≈ 0.0401173 ms
```

### Caveat

This is a timing-attribution experiment, not a broad serving-performance  
benchmark. The workload is intentionally tiny. The throughput, TTFT, and TPOT  
numbers should be used only as sanity checks at this stage.  
