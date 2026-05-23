Below is a markdown reference you can save as something like:

```text
~/vllm-sched/docs/scheduler_experiment_reference.md
```

---

# vLLM Scheduler Experiment Reference on Unity

This document is a working reference for running scheduler experiments in the Unity HPC setup. It assumes Phases 1–9 are complete and that the external scheduler package is installed editable into the project virtual environment.

Phase 9 added lightweight scheduler-call JSONL instrumentation using `SCHEDULER_POLICIES_ITER_LOG`, with one JSON record per `Scheduler.schedule()` call. The validated Phase 9 run produced 101 scheduler records, `all_ok: True`, 4 successful benchmark requests, 0 failed requests, and clean GPU shutdown.

---

## 1. Current known-good state

```text
workspace:
  ~/vllm-sched/

vLLM repo:
  ~/vllm-sched/vllm

scheduler package:
  ~/vllm-sched/vllm-scheduler-policies

virtualenv:
  ~/vllm-sched/.venv

CUDA setup:
  source ~/vllm-sched/vllm-scheduler-policies/scripts/unity_cuda_13_1_env.sh

vLLM branch:
  unity-phase4-v0.20.2-cu130

vLLM commit:
  bc150f50299199599673614f80d12a196f377655

scheduler package commit after Phase 9:
  bcef5ae0801d8af3c67c7a63739a4c0e8c463fb4
```

Validated scheduler mappings:

```text
default
  no --scheduler-cls

passthrough
  --scheduler-cls vllm_scheduler_policies.baseline.BaselinePassthroughScheduler

simple_policy_1
  --scheduler-cls vllm_scheduler_policies.simple_policy_1.SimplePolicy1Scheduler
```

Current `simple_policy_1` is behavior-preserving. It does not implement a new scheduling algorithm. It wraps the default vLLM scheduler with JSONL timing instrumentation.

---

## 2. Important instrumentation concept

The current Phase 9 scheduler class is:

```python
class SimplePolicy1Scheduler(InstrumentedSchedulerMixin, Scheduler):
    """Behavior-preserving scheduler with Phase 9 JSONL instrumentation."""
```

The method resolution order is:

```text
SimplePolicy1Scheduler
→ InstrumentedSchedulerMixin
→ vLLM Scheduler
```

Since `SimplePolicy1Scheduler` currently does not define `schedule()`, vLLM calls:

```text
InstrumentedSchedulerMixin.schedule()
```

which then calls:

```python
super().schedule()
```

That delegates to vLLM’s original `Scheduler.schedule()`.

So the current flow is:

```text
vLLM calls scheduler.schedule()
→ InstrumentedSchedulerMixin.schedule()
→ start timer
→ vLLM Scheduler.schedule()
→ stop timer
→ write one JSONL record
→ return original SchedulerOutput unchanged
```

This measures the default vLLM scheduler path.

---

## 3. Critical caveat for future custom policies

Do **not** casually override `schedule()` in future policy classes.

If a future class does this:

```python
class MyPolicyScheduler(InstrumentedSchedulerMixin, Scheduler):
    def schedule(self):
        ...
```

then Python calls `MyPolicyScheduler.schedule()` first. This bypasses `InstrumentedSchedulerMixin.schedule()` unless the custom method explicitly calls `super().schedule()`.

Even if it calls:

```python
return super().schedule()
```

any policy logic before that call is **not included** in the current Phase 9 timer.

Recommended future design:

```python
class InstrumentedSchedulerMixin:
    def schedule(self):
        start = time.perf_counter()
        try:
            output = self._schedule_impl()
            return output
        finally:
            end = time.perf_counter()
            write_jsonl(...)

    def _schedule_impl(self):
        return super().schedule()
```

Then future policies should override `_schedule_impl()`, not `schedule()`:

```python
class MyPolicyScheduler(InstrumentedSchedulerMixin, Scheduler):
    def _schedule_impl(self):
        self.apply_my_policy()
        return super().schedule()
```

This makes the timer cover:

```text
custom policy logic
+
default vLLM Scheduler.schedule()
```

This design also supports fully custom schedulers that do **not** call `Scheduler.schedule()`, as long as they return a valid `SchedulerOutput` and maintain vLLM scheduler invariants.

---

## 4. Standard environment setup

Start from a Unity shell on the allocated node:

```bash
cd ~/vllm-sched

source ~/vllm-sched/vllm-scheduler-policies/scripts/unity_cuda_13_1_env.sh
source ~/vllm-sched/.venv/bin/activate

hostname
nvidia-smi
python - <<'PY'
import importlib.metadata
import os
import sys
import torch

print("python:", sys.executable)
print("torch:", torch.__version__)
print("torch.version.cuda:", torch.version.cuda)
print("vllm:", importlib.metadata.version("vllm"))
print("CUDA_HOME:", os.environ.get("CUDA_HOME"))
PY
```

Expected broad state:

```text
torch: 2.11.0+cu130
torch.version.cuda: 13.0
vllm: 0.20.2+precompiled
GPU: NVIDIA A16
GPU memory before run: 0 MiB
```

---

## 5. Pre-run git checks

Before any experiment:

```bash
echo "=== scheduler package ==="
cd ~/vllm-sched/vllm-scheduler-policies
git status --short
git rev-parse --abbrev-ref HEAD
git rev-parse HEAD

echo
echo "=== vLLM repo ==="
cd ~/vllm-sched/vllm
git status --short
git rev-parse --abbrev-ref HEAD
git rev-parse HEAD
```

Expected:

```text
scheduler package status: clean
vLLM repo status: clean
```

For Phase 9 and later scheduler-package-only experiments, do not modify native vLLM files unless explicitly planned.

---

## 6. Create a run directory

Use one directory per run:

```bash
RUN_ID="$(date +%Y%m%d_%H%M%S)_scheduler_smoke_simple_policy_1_qwen3_0p6b_$(hostname)"
RUN_DIR="$HOME/vllm-sched/results/${RUN_ID}"
mkdir -p "${RUN_DIR}"

echo "RUN_DIR=${RUN_DIR}"
```

Recommended artifacts per run:

```text
server.log
bench.log or bench_tiny.log
scheduler_iter.jsonl
scheduler_log_path.txt
single_chat_response.json
env.txt
summary.json or record.md
server.pid
```

---

## 7. Run an instrumented scheduler server

For `simple_policy_1`, enable JSONL logging with:

```bash
export SCHEDULER_POLICIES_ITER_LOG="${RUN_DIR}/scheduler_iter.jsonl"
echo "SCHEDULER_POLICIES_ITER_LOG=${SCHEDULER_POLICIES_ITER_LOG}" \
  | tee "${RUN_DIR}/scheduler_log_path.txt"
```

Start server:

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
echo "SERVER_PID=$(cat "${RUN_DIR}/server.pid")"
```

Wait for health:

```bash
i=0
while [ "$i" -lt 24 ]; do
  if curl -fsS http://127.0.0.1:8000/health >/tmp/vllm_health.txt 2>/tmp/vllm_health.err; then
    echo "health_ok_at_attempt=$i"
    cat /tmp/vllm_health.txt
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

Check server log:

```bash
grep -E "scheduler_cls|Using custom scheduler class|Unknown vLLM environment variable|Traceback|ERROR" \
  "${RUN_DIR}/server.log" || true

tail -n 120 "${RUN_DIR}/server.log"
```

Expected:

```text
scheduler_cls=vllm_scheduler_policies.simple_policy_1.SimplePolicy1Scheduler
Using custom scheduler class vllm_scheduler_policies.simple_policy_1.SimplePolicy1Scheduler
no Traceback
no ERROR
no Unknown vLLM environment variable warning
```

---

## 8. Send one smoke request

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
  | tee "${RUN_DIR}/single_chat_response.json"
```

Check JSONL exists:

```bash
ls -lh "${RUN_DIR}/scheduler_iter.jsonl"
head -n 5 "${RUN_DIR}/scheduler_iter.jsonl"
```

Expected:

```text
scheduler_iter.jsonl exists
records contain scheduler_class = vllm_scheduler_policies.simple_policy_1.SimplePolicy1Scheduler
records contain ok = true
```

---

## 9. Run a tiny benchmark

```bash
cd ~/vllm-sched/vllm-scheduler-policies

BEFORE_LINES=0
if [ -f "${RUN_DIR}/scheduler_iter.jsonl" ]; then
  BEFORE_LINES="$(wc -l < "${RUN_DIR}/scheduler_iter.jsonl")"
fi
echo "scheduler_jsonl_lines_before=${BEFORE_LINES}"

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

echo "scheduler_jsonl_lines_after=${AFTER_LINES}"
echo "scheduler_jsonl_new_lines=$((AFTER_LINES - BEFORE_LINES))"
```

Check benchmark result:

```bash
cat "${RUN_DIR}/bench_tiny.log"

grep -E "Successful requests|Failed requests|Benchmark duration|Request throughput|Mean TTFT|Mean TPOT|Mean ITL" \
  "${RUN_DIR}/bench_tiny.log" || true
```

Expected:

```text
bench_rc=0
Successful requests: 4
Failed requests: 0
scheduler_jsonl_new_lines > 0
```

---

## 10. Analyze scheduler JSONL

Basic summary:

```bash
LOG="${RUN_DIR}/scheduler_iter.jsonl"

LOG="$LOG" python - <<'PY'
import json
import os
from pathlib import Path
from statistics import mean, median

path = Path(os.environ["LOG"])
records = []

with path.open("r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            records.append(json.loads(line))

def pct(xs, q):
    xs = sorted(xs)
    if not xs:
        return None
    i = int(round((len(xs) - 1) * q / 100.0))
    return xs[i]

wall = [r.get("scheduler_wall_time_ms") or 0.0 for r in records]
tokens = [r.get("num_scheduled_tokens") or 0 for r in records]
req_events = [r.get("num_scheduled_requests") or 0 for r in records]

print("path:", path)
print("num_records_total:", len(records))
print("scheduler_class:", records[0].get("scheduler_class") if records else None)
print("all_ok:", all(r.get("ok") for r in records))
print("total_scheduled_tokens:", sum(tokens))
print("total_scheduled_request_events:", sum(req_events))
print("total_preemptions:", sum(r.get("num_preemptions") or 0 for r in records))
print("max_waiting_before:", max((r.get("waiting_before") or 0) for r in records))
print("max_running_before:", max((r.get("running_before") or 0) for r in records))
print("min_scheduler_wall_time_ms:", min(wall) if wall else None)
print("mean_scheduler_wall_time_ms:", mean(wall) if wall else None)
print("median_scheduler_wall_time_ms:", median(wall) if wall else None)
print("p90_scheduler_wall_time_ms:", pct(wall, 90))
print("p95_scheduler_wall_time_ms:", pct(wall, 95))
print("p99_scheduler_wall_time_ms:", pct(wall, 99))
print("max_scheduler_wall_time_ms:", max(wall) if wall else None)
PY
```

Interpretation:

```text
scheduler_wall_time_ms:
  elapsed wall-clock time inside the scheduler call

num_scheduled_tokens:
  total tokens scheduled in that scheduler call

num_scheduled_requests:
  number of requests that received scheduled tokens in that call

waiting_before:
  waiting queue size before scheduling

running_before:
  active/running queue size before scheduling

num_preemptions:
  requests preempted in that scheduler call

ok:
  whether schedule() returned normally
```

Important limitation:

```text
scheduler_wall_time_ms measures scheduler-call time only.
It does not measure full engine iteration time, GPU model execution time,
HTTP latency, tokenization, detokenization, or end-to-end request latency.
```

---

## 11. Group scheduler time by concurrency

Group by `running_before`:

```bash
LOG="${RUN_DIR}/scheduler_iter.jsonl"

LOG="$LOG" python - <<'PY'
import json
import os
from pathlib import Path
from collections import defaultdict
from statistics import mean, median

path = Path(os.environ["LOG"])
groups = defaultdict(list)

with path.open("r", encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        groups[r.get("running_before")].append(r.get("scheduler_wall_time_ms") or 0.0)

print("running_before,calls,mean_ms,median_ms,max_ms")
for k in sorted(groups, key=lambda x: (-1 if x is None else x)):
    xs = groups[k]
    print(f"{k},{len(xs)},{mean(xs)},{median(xs)},{max(xs)}")
PY
```

Group by `waiting_before`:

```bash
LOG="${RUN_DIR}/scheduler_iter.jsonl"

LOG="$LOG" python - <<'PY'
import json
import os
from pathlib import Path
from collections import defaultdict
from statistics import mean, median

path = Path(os.environ["LOG"])
groups = defaultdict(list)

with path.open("r", encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        groups[r.get("waiting_before")].append(r.get("scheduler_wall_time_ms") or 0.0)

print("waiting_before,calls,mean_ms,median_ms,max_ms")
for k in sorted(groups, key=lambda x: (-1 if x is None else x)):
    xs = groups[k]
    print(f"{k},{len(xs)},{mean(xs)},{median(xs)},{max(xs)}")
PY
```

This helps answer:

```text
Does scheduler overhead increase with active requests?
Does scheduler overhead increase with waiting queue size?
```

A tiny benchmark with concurrency 2 is only a smoke test. For real scaling, run separate directories for:

```text
max_concurrency = 1, 2, 4, 8, 16
```

---

## 12. Find slow scheduler calls

```bash
LOG="${RUN_DIR}/scheduler_iter.jsonl"

LOG="$LOG" python - <<'PY'
import json
import os
from pathlib import Path

path = Path(os.environ["LOG"])
records = [json.loads(line) for line in path.open("r", encoding="utf-8") if line.strip()]
records.sort(key=lambda r: r.get("scheduler_wall_time_ms") or 0.0, reverse=True)

print("top 10 slowest scheduler calls")
for r in records[:10]:
    print({
        "call_index": r.get("call_index"),
        "scheduler_wall_time_ms": r.get("scheduler_wall_time_ms"),
        "waiting_before": r.get("waiting_before"),
        "running_before": r.get("running_before"),
        "num_scheduled_requests": r.get("num_scheduled_requests"),
        "num_scheduled_tokens": r.get("num_scheduled_tokens"),
        "num_preemptions": r.get("num_preemptions"),
        "ok": r.get("ok"),
    })
PY
```

---

## 13. Export JSONL to CSV

```bash
LOG="${RUN_DIR}/scheduler_iter.jsonl"
CSV="${RUN_DIR}/scheduler_iter.csv"

LOG="$LOG" CSV="$CSV" python - <<'PY'
import csv
import json
import os
from pathlib import Path

path = Path(os.environ["LOG"])
out = Path(os.environ["CSV"])

fields = [
    "call_index",
    "start_time_unix_s",
    "scheduler_wall_time_ms",
    "waiting_before",
    "waiting_after",
    "running_before",
    "running_after",
    "skipped_waiting_before",
    "skipped_waiting_after",
    "num_scheduled_requests",
    "num_scheduled_tokens",
    "num_scheduled_new_reqs",
    "num_preemptions",
    "ok",
    "scheduler_class",
    "hostname",
    "pid",
]

with path.open("r", encoding="utf-8") as f, out.open("w", newline="", encoding="utf-8") as g:
    writer = csv.DictWriter(g, fieldnames=fields)
    writer.writeheader()
    for line in f:
        if not line.strip():
            continue
        r = json.loads(line)
        writer.writerow({k: r.get(k) for k in fields})

print(out)
PY
```

---

## 14. Stop server cleanly

```bash
SERVER_PID="$(cat "${RUN_DIR}/server.pid")"

echo "Stopping SERVER_PID=${SERVER_PID}"
kill "${SERVER_PID}" 2>/dev/null || true

sleep 10

echo "=== process status ==="
ps -fp "${SERVER_PID}" || true

echo
echo "=== GPU status ==="
nvidia-smi
```

Expected:

```text
server process gone
GPU memory returns to 0 MiB
No VLLM::EngineCore process remains
```

---

## 15. Create a run record

```bash
cd ~/vllm-sched/vllm-scheduler-policies

RECORD="${RUN_DIR}/scheduler_experiment_record_$(date +%Y%m%d_%H%M%S)_$(hostname).md"

{
  echo "# Scheduler experiment record"
  echo
  echo "## Run"
  echo
  echo "- run_dir: ${RUN_DIR}"
  echo "- hostname: $(hostname)"
  echo "- timestamp: $(date -Is)"
  echo
  echo "## Scheduler package"
  echo
  echo '```text'
  git status --short
  git rev-parse --abbrev-ref HEAD
  git rev-parse HEAD
  git log --oneline -3
  echo '```'
  echo
  echo "## vLLM repo"
  echo
  echo '```text'
  cd ~/vllm-sched/vllm
  git status --short
  git rev-parse --abbrev-ref HEAD
  git rev-parse HEAD
  echo '```'
  echo
  echo "## Environment"
  echo
  echo '```text'
  python - <<'PY'
import importlib.metadata
import os
import sys
import torch

print("python:", sys.executable)
print("torch:", torch.__version__)
print("torch.version.cuda:", torch.version.cuda)
print("vllm:", importlib.metadata.version("vllm"))
print("CUDA_HOME:", os.environ.get("CUDA_HOME"))
PY
  nvcc --version || true
  nvidia-smi
  echo '```'
  echo
  echo "## Scheduler log path"
  echo
  echo '```text'
  cat "${RUN_DIR}/scheduler_log_path.txt" 2>/dev/null || true
  echo '```'
  echo
  echo "## Benchmark summary"
  echo
  echo '```text'
  grep -E "Successful requests|Failed requests|Benchmark duration|Request throughput|Mean TTFT|Mean TPOT|Mean ITL" \
    "${RUN_DIR}/bench_tiny.log" 2>/dev/null || true
  echo '```'
  echo
  echo "## Scheduler JSONL summary"
  echo
  echo '```text'
  LOG="${RUN_DIR}/scheduler_iter.jsonl" python - <<'PY'
import json
import os
from pathlib import Path
from statistics import mean, median

path = Path(os.environ["LOG"])
if not path.exists():
    print("missing:", path)
    raise SystemExit(0)

records = [json.loads(line) for line in path.open("r", encoding="utf-8") if line.strip()]
wall = [r.get("scheduler_wall_time_ms") or 0.0 for r in records]

print("path:", path)
print("num_records_total:", len(records))
print("scheduler_class:", records[0].get("scheduler_class") if records else None)
print("all_ok:", all(r.get("ok") for r in records))
print("total_scheduled_tokens:", sum(r.get("num_scheduled_tokens") or 0 for r in records))
print("total_scheduled_request_events:", sum(r.get("num_scheduled_requests") or 0 for r in records))
print("max_waiting_before:", max((r.get("waiting_before") or 0) for r in records))
print("max_running_before:", max((r.get("running_before") or 0) for r in records))
print("mean_scheduler_wall_time_ms:", mean(wall) if wall else None)
print("median_scheduler_wall_time_ms:", median(wall) if wall else None)
print("max_scheduler_wall_time_ms:", max(wall) if wall else None)
print("num_records_with_preemptions:", sum(1 for r in records if (r.get("num_preemptions") or 0) > 0))
PY
  echo '```'
  echo
  echo "## Warning/error scan"
  echo
  echo '```text'
  grep -E "Traceback|ERROR|Exception|Unknown vLLM environment variable" \
    "${RUN_DIR}/server.log" "${RUN_DIR}/bench_tiny.log" 2>/dev/null || true
  echo '```'
  echo
  echo "## Artifacts"
  echo
  echo "- server.log"
  echo "- bench_tiny.log"
  echo "- scheduler_iter.jsonl"
  echo "- scheduler_iter.csv, if exported"
  echo "- single_chat_response.json"
} > "${RECORD}"

echo "RECORD=${RECORD}"
tail -n 80 "${RECORD}"
```

---

## 16. Recommended experiment progression

For every scheduler policy, run this progression:

```text
1. import/compile check
2. dry-run launcher check
3. server start
4. health check
5. one curl request
6. tiny benchmark
7. scheduler JSONL summary
8. stop server and verify GPU memory returns to 0 MiB
9. create run record
10. commit only after clean validation
```

Do not jump directly to large benchmarks.

---

## 17. Suggested benchmark matrix

Start with smoke:

```text
num_prompts=4
input_len=32
output_len=16
max_concurrency=2
```

Then scaling:

```text
num_prompts=32 or 64
input_len=32
output_len=64
max_concurrency=1,2,4,8,16
```

Prefill-heavy:

```text
input_len=512 or 1024
output_len=16
max_concurrency=2,4,8
```

Decode-heavy:

```text
input_len=32
output_len=128 or 256
max_concurrency=2,4,8
```

Mixed:

```text
input_len=256
output_len=128
max_concurrency=4,8
```

Always compare:

```text
default
passthrough
simple_policy_1 or future policy
```

For scheduler-overhead studies, use an instrumented scheduler class. The default scheduler alone will not produce `scheduler_iter.jsonl`.

---

## 18. What the current instrumentation answers

It answers:

```text
How long does Scheduler.schedule() take?
How many scheduler calls happened?
How many tokens were scheduled per call?
How many requests were scheduled per call?
How many requests were waiting/running before and after each call?
Did preemption happen?
Did schedule() return successfully?
Does scheduler time grow with waiting/running queue size?
```

It does not yet answer:

```text
scheduler time / total engine iteration time
scheduler time / model execution time
scheduler time / end-to-end request latency
GPU kernel timeline
sampling overhead
HTTP/server overhead
```

Those require additional engine-level instrumentation or profiler correlation in a later phase.

---

## 19. Policy implementation rule going forward

For future policies, prefer:

```python
class MyPolicyScheduler(InstrumentedSchedulerMixin, Scheduler):
    def _schedule_impl(self):
        self.apply_policy_logic()
        return super().schedule()
```

Avoid:

```python
class MyPolicyScheduler(InstrumentedSchedulerMixin, Scheduler):
    def schedule(self):
        ...
```

unless the class explicitly owns both scheduling and instrumentation.

The long-term goal is:

```text
InstrumentedSchedulerMixin.schedule()
  owns timing/logging

Policy._schedule_impl()
  owns scheduling policy logic
```

This keeps experiments comparable across policies.

---

## 20. Common failure modes

### Server health fails

Check:

```bash
tail -n 200 "${RUN_DIR}/server.log"
ps -fp "$(cat "${RUN_DIR}/server.pid")" || true
nvidia-smi
```

### JSONL not created

Likely causes:

```text
No generation request has run yet.
SCHEDULER_POLICIES_ITER_LOG was not exported before server start.
The selected scheduler is not instrumented.
The server failed before EngineCore initialized.
```

### Unknown vLLM environment variable warning

Do not use:

```text
VLLM_SCHEDULER_ITER_LOG
```

Use:

```text
SCHEDULER_POLICIES_ITER_LOG
```

### GPU memory remains after stop

Check for lingering engine process:

```bash
nvidia-smi
ps aux | grep -E "vllm|EngineCore" | grep -v grep
```

Kill only your own stale vLLM processes.

### Future policy timing seems too small

Check whether the policy overrides `schedule()` directly. If yes, it may be bypassing or partially bypassing the mixin. Use the `_schedule_impl()` pattern instead.

---