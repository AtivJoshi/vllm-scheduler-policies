# Scheduler Experiment Reference

This is the canonical practical reference for Unity-local vLLM scheduler
experiments. It supersedes
`docs/scheduler_experiment_reference_phase_9_report.md`, which remains a
historical Phase 9 report only.

## 1. Purpose and Scope

Use this guide for small, reproducible scheduler smoke experiments under
`~/vllm-sched`. Keep source code, logs, and results separate. Save run artifacts
under `~/vllm-sched/results/`.

Do not run server or benchmark jobs on login nodes. Broad benchmarks, long
benchmarks, profiling runs, or algorithmic behavior measurements require an
explicit plan before execution.

## 2. Known-Good Environment Setup

Use the Unity CUDA and project virtual environment:

```bash
cd ~/vllm-sched/vllm-scheduler-policies
source ~/vllm-sched/vllm-scheduler-policies/scripts/unity_cuda_13_1_env.sh
source ~/vllm-sched/.venv/bin/activate
```

Use the project Python:

```bash
~/vllm-sched/.venv/bin/python
```

Known Phase 3/4 baseline:

- vLLM target: `v0.20.2`, commit short `bc150f5`
- CUDA backend: `cu130`
- Unity CUDA module: `cuda/13.1`
- Python: `3.12.13`
- vLLM: `0.20.2+precompiled`
- torch: `2.11.0+cu130`
- `torch.version.cuda == 13.0`
- GPU: `NVIDIA A16`
- FlashInfer import succeeded and FlashInfer was not disabled

Avoid shell patterns that have caused Unity instability, especially
`set -euo pipefail`.

## 3. Pre-Run Checks

Record repo and environment state before starting a server:

```bash
cd ~/vllm-sched/vllm-scheduler-policies
git status --short
git rev-parse --abbrev-ref HEAD
git rev-parse HEAD

cd ~/vllm-sched/vllm
git status --short
git rev-parse --abbrev-ref HEAD
git rev-parse HEAD

~/vllm-sched/.venv/bin/python -c "import vllm_scheduler_policies; print('ok')"
nvidia-smi
```

The read-only audit observed:

- vLLM branch `unity-phase4-v0.20.2-cu130`, commit
  `bc150f50299199599673614f80d12a196f377655`, clean
- scheduler package branch `master`, commit
  `169a86289b26d1dc508a88ab74a1616fd0716f15`, clean

## 4. Run Directory Layout

Create a timestamped run directory under results:

```bash
RUN_ID="$(date +%Y%m%d_%H%M%S)_scheduler_smoke_$(hostname)"
RUN_DIR="$HOME/vllm-sched/results/${RUN_ID}"
mkdir -p "${RUN_DIR}"
echo "RUN_DIR=${RUN_DIR}"
```

Recommended files:

```text
env.txt
server.log
server.pid
health.txt
single_chat_response.json
bench_tiny.log
scheduler_iter.jsonl
scheduler_log_path.txt
record.md
```

## 5. Scheduler Aliases and Classes

- `default`: no `--scheduler-cls`
- `passthrough`:
  `vllm_scheduler_policies.baseline.BaselinePassthroughScheduler`
- `simple_policy_1`:
  `vllm_scheduler_policies.simple_policy_1.SimplePolicy1Scheduler`
- `primal_lp_dry_run`:
  `vllm_scheduler_policies.primal_lp_dry_run.PrimalLPDryRunScheduler`

Current `scripts/serve.sh` has built-in mappings for `default`, `passthrough`,
and `simple_policy_1`; `scripts/scheduler_lib.sh` also contains historical
placeholder aliases that should not be used unless implemented and validated.
Until `primal_lp_dry_run` is added to the script map, pass its class path through
the extra vLLM arguments after `--`.

## 6. Starting a Server Safely

For an instrumented policy, export the scheduler JSONL path before server start:

```bash
export SCHEDULER_POLICIES_ITER_LOG="${RUN_DIR}/scheduler_iter.jsonl"
echo "SCHEDULER_POLICIES_ITER_LOG=${SCHEDULER_POLICIES_ITER_LOG}" \
  > "${RUN_DIR}/scheduler_log_path.txt"
```

Start `simple_policy_1`:

```bash
cd ~/vllm-sched/vllm-scheduler-policies

scripts/serve.sh \
  --scheduler simple_policy_1 \
  --model Qwen/Qwen3-0.6B \
  --served-model-name qwen3-0.6b \
  --host 127.0.0.1 \
  --port 8000 \
  --max-model-len 2048 \
  --max-num-batched-tokens 2048 \
  --max-num-seqs 64 \
  --gpu-memory-utilization 0.80 \
  > "${RUN_DIR}/server.log" 2>&1 &

echo "$!" > "${RUN_DIR}/server.pid"
```

Start `primal_lp_dry_run` with a direct class path:

```bash
scripts/serve.sh \
  --scheduler default \
  --model Qwen/Qwen3-0.6B \
  --served-model-name qwen3-0.6b \
  --host 127.0.0.1 \
  --port 8000 \
  --max-model-len 2048 \
  --max-num-batched-tokens 2048 \
  --max-num-seqs 64 \
  --gpu-memory-utilization 0.80 \
  -- --scheduler-cls vllm_scheduler_policies.primal_lp_dry_run.PrimalLPDryRunScheduler \
  > "${RUN_DIR}/server.log" 2>&1 &

echo "$!" > "${RUN_DIR}/server.pid"
```

## 7. Health Check and One Curl Smoke Request

Wait for `/health`:

```bash
i=0
while [ "$i" -lt 24 ]; do
  if curl -fsS http://127.0.0.1:8000/health > "${RUN_DIR}/health.txt" 2>/tmp/vllm_health.err; then
    echo "health_ok_at_attempt=$i"
    break
  fi
  i=$((i + 1))
  sleep 5
done

if [ "$i" -ge 24 ]; then
  echo "health_failed_after_120s"
  cat /tmp/vllm_health.err
fi
```

Send one OpenAI-compatible chat request:

```bash
curl -sS http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3-0.6b",
    "messages": [
      {"role": "user", "content": "Reply with exactly six words about schedulers."}
    ],
    "max_tokens": 16,
    "temperature": 0
  }' \
  > "${RUN_DIR}/single_chat_response.json"
```

Check the server log:

```bash
grep -E "scheduler_cls|Using custom scheduler class|Unknown vLLM environment variable|Traceback|ERROR" \
  "${RUN_DIR}/server.log" || true
```

## 8. Tiny Benchmark Only

Use this tiny benchmark for smoke validation. Do not scale it up without an
explicit benchmark plan.

```bash
BEFORE_LINES=0
if [ -f "${RUN_DIR}/scheduler_iter.jsonl" ]; then
  BEFORE_LINES="$(wc -l < "${RUN_DIR}/scheduler_iter.jsonl")"
fi

scripts/bench.sh \
  --scheduler simple_policy_1 \
  --model qwen3-0.6b \
  --tokenizer Qwen/Qwen3-0.6B \
  --host 127.0.0.1 \
  --port 8000 \
  --endpoint /v1/chat/completions \
  --num-prompts 4 \
  --random-input-len 32 \
  --random-output-len 16 \
  --max-concurrency 2 \
  --request-rate inf \
  --seed 0 \
  > "${RUN_DIR}/bench_tiny.log" 2>&1

BENCH_RC="$?"
echo "bench_rc=${BENCH_RC}"

AFTER_LINES=0
if [ -f "${RUN_DIR}/scheduler_iter.jsonl" ]; then
  AFTER_LINES="$(wc -l < "${RUN_DIR}/scheduler_iter.jsonl")"
fi

echo "scheduler_jsonl_lines_before=${BEFORE_LINES}"
echo "scheduler_jsonl_lines_after=${AFTER_LINES}"
echo "scheduler_jsonl_new_lines=$((AFTER_LINES - BEFORE_LINES))"
```

Expected smoke outcome:

```text
bench_rc=0
Successful requests: 4
Failed requests: 0
scheduler_jsonl_new_lines > 0 for instrumented schedulers
```

## 9. Stopping Server and Verifying GPU Cleanup

Stop only the server process for the current run:

```bash
SERVER_PID="$(cat "${RUN_DIR}/server.pid")"
echo "Stopping SERVER_PID=${SERVER_PID}"
kill "${SERVER_PID}" 2>/dev/null || true

sleep 10
ps -fp "${SERVER_PID}" || true
nvidia-smi
```

Expected cleanup:

```text
server process gone
GPU memory returns to 0 MiB
No VLLM::EngineCore process remains
```

If memory remains, inspect before killing anything:

```bash
nvidia-smi
ps aux | grep -E "vllm|EngineCore" | grep -v grep
```

## 10. Scheduler JSONL Instrumentation and Statistics

`InstrumentedSchedulerMixin.schedule()` is the outer timing/logging wrapper.
Instrumented policies override `_schedule_impl()`.

`SCHEDULER_POLICIES_ITER_LOG` is the only valid scheduler JSONL environment
variable. `VLLM_SCHEDULER_ITER_LOG` is forbidden except as a historical caution:
a stale Phase 9 run using it produced an "Unknown vLLM environment variable"
warning.

Subclass `_schedule_impl()` must delegate to native scheduling with
`Scheduler.schedule(self)` or a helper that does exactly that.
`super().schedule()` inside subclass `_schedule_impl()` is forbidden because it
recursively re-enters `InstrumentedSchedulerMixin.schedule()`.

Basic JSONL summary:

```bash
LOG="${RUN_DIR}/scheduler_iter.jsonl"

LOG="$LOG" ~/vllm-sched/.venv/bin/python - <<'PY'
import json
import os
from pathlib import Path
from statistics import mean

path = Path(os.environ["LOG"])
records = []

with path.open("r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            records.append(json.loads(line))

scheduler_calls = [r for r in records if r.get("event") == "scheduler_call"]
wall = [r.get("scheduler_wall_time_ms") or 0.0 for r in scheduler_calls]

print("path:", path)
print("num_records_total:", len(records))
print("num_scheduler_call_records:", len(scheduler_calls))
print("scheduler_classes:", sorted({r.get("scheduler_class") for r in scheduler_calls}))
print("all_scheduler_calls_ok:", all(r.get("ok") for r in scheduler_calls))
print("total_scheduled_tokens:", sum(r.get("num_scheduled_tokens") or 0 for r in scheduler_calls))
print("total_preemptions:", sum(r.get("num_preemptions") or 0 for r in scheduler_calls))
print("mean_scheduler_wall_time_ms:", mean(wall) if wall else None)
print("max_scheduler_wall_time_ms:", max(wall) if wall else None)
PY
```

`scheduler_wall_time_ms` measures scheduler-call time only. It does not measure
full engine iteration time, GPU model execution time, tokenization,
detokenization, or end-to-end request latency.

## 11. LP Dry-Run JSONL Records

Instrumented JSONL may contain:

- `scheduler_call`: emitted by `InstrumentedSchedulerMixin.schedule()` for the
  outer scheduler call.
- `lp_dry_run`: emitted by `PrimalLPDryRunScheduler` for LP planner diagnostics.

`PrimalLPDryRunScheduler` snapshots scheduler state, runs/logs the LP planner,
and returns native scheduler output unchanged. The dry-run bridge must not call
`allocate_slots()`, call `_preempt_request()`, mutate queues/request/KV state, or
construct `SchedulerOutput`.

Count dry-run records:

```bash
LOG="${RUN_DIR}/scheduler_iter.jsonl"

LOG="$LOG" ~/vllm-sched/.venv/bin/python - <<'PY'
import json
import os
from collections import Counter
from pathlib import Path

path = Path(os.environ["LOG"])
counts = Counter()

with path.open("r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            counts[json.loads(line).get("event")] += 1

print(dict(counts))
PY
```

## 12. Common Failure Modes

Server health fails:

```bash
tail -n 200 "${RUN_DIR}/server.log"
ps -fp "$(cat "${RUN_DIR}/server.pid")" || true
nvidia-smi
```

JSONL not created:

```text
No generation request has run yet.
SCHEDULER_POLICIES_ITER_LOG was not exported before server start.
The selected scheduler is not instrumented.
The server failed before EngineCore initialized.
```

Unknown vLLM environment variable warning:

```text
Do not use VLLM_SCHEDULER_ITER_LOG.
Use SCHEDULER_POLICIES_ITER_LOG.
```

Scheduler timing appears too small:

```text
Check whether the policy overrides schedule() directly or bypasses
InstrumentedSchedulerMixin.schedule(). Instrumented policy work belongs in
_schedule_impl().
```

## 13. Validation Commands

Documentation-only checks:

```bash
git diff --check
git diff --stat
git status --short
```

Targeted package checks:

```bash
~/vllm-sched/.venv/bin/python -c "import vllm_scheduler_policies; print('ok')"
~/vllm-sched/.venv/bin/python -m compileall vllm_scheduler_policies
```

For scheduler behavior or measurement changes, add only targeted tests that
clarify intended behavior and run the most relevant validation available. Do not
claim validation succeeded unless it actually ran.
