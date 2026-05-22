# Fresh-start plan for vLLM scheduler experiments on Unity

## Context and guiding choices

Start from a clean directory, not from the old `vllm-sched`. Use **one pinned vLLM release/commit**, not floating `main`. As of May 12, 2026, the latest GitHub release is **vLLM v0.20.2**, a patch release on the v0.20.x line. v0.20.0 changed the default CUDA wheel to CUDA 13.0 and upgraded to PyTorch 2.11, so CUDA/toolkit alignment must be checked before committing to v0.20.x. ([GitHub](https://github.com/vllm-project/vllm/releases "Releases · vllm-project/vllm · GitHub"))

The initial goal is **not** to implement the full LaTeX algorithm. The goal is:

1. Build a clean, reproducible Unity + vLLM environment.
    
2. Run standard vLLM serving, benchmarking, and profiling without disabling FlashInfer.
    
3. Create a scheduler-swapping setup.
    
4. Verify a custom scheduler that is behaviorally identical to the default scheduler.
    
5. Add instrumentation for scheduler runtime and scheduler-vs-iteration overhead.
    
6. Then use Codex to implement simple schedulers incrementally.
    

---

## Phase 1: Unity working environment

Set up the Unity workflow before installing vLLM.

Use `tmux` on the Unity login node before requesting a GPU allocation. Unity documents that `salloc` jobs are tied to the SSH session and can be killed if the SSH session closes; Unity recommends `tmux` with `salloc` to keep a session alive across disconnects. ([docs.unity.rc.umass.edu](https://docs.unity.rc.umass.edu/documentation/jobs/?utm_source=chatgpt.com "Submitting Jobs"))

Configure:

```text
- tmux large scrollback
- mouse scrolling
- persistent bash history
- a standard results/logs directory convention
```

Recommended mental model:

```text
Login node:
  tmux session lives here

Inside tmux:
  salloc requests a GPU node

Local VS Code:
  Remote-SSH into the allocated compute node

Inside VS Code terminal:
  run vLLM, Codex, tests, benchmarks
```

Use `salloc` for interactive setup/debugging and `sbatch` later for unattended sweeps. Unity’s docs describe `salloc` as the interactive path and `sbatch` as better for non-interactive repeatable jobs. ([docs.unity.rc.umass.edu](https://docs.unity.rc.umass.edu/documentation/jobs/salloc/?utm_source=chatgpt.com "Interactive CLI Jobs"))

---

## Phase 2: CUDA/toolchain sanity check

Before installing vLLM, inspect the allocated GPU and CUDA modules.

Record:

```text
- hostname
- GPU model
- nvidia-smi driver version
- available CUDA modules
- loaded CUDA module
- nvcc version
- whether curand.h exists
```

The previous setup had:

```text
torch.version.cuda = 13.0
CUDA_HOME = CUDA 12.6
GPU = NVIDIA A16
```

That mismatch allowed basic execution but caused FlashInfer JIT problems. In the new setup, explicitly aim for one of these clean states:

```text
Preferred:
  vLLM/PyTorch CUDA 13.x
  CUDA toolkit 13.x loaded
  FlashInfer works without disabling anything

Acceptable:
  vLLM/PyTorch CUDA 12.9
  CUDA toolkit 12.9 loaded
  FlashInfer works without disabling anything

Fallback:
  choose an older vLLM release compatible with the newest complete CUDA toolkit
  available on Unity
```

Do not start by setting:

```text
VLLM_USE_FLASHINFER_SAMPLER=0
```

Only use that as a temporary debugging workaround. The clean target is standard vLLM with FlashInfer enabled.

---

## Phase 3: choose vLLM version

Start with **latest stable release v0.20.2** only if Unity has a compatible CUDA toolkit and driver setup. The v0.20 release line defaults to CUDA 13.0 and recommends `uv` with `--torch-backend=cu129` when using CUDA 12.9. ([GitHub](https://github.com/vllm-project/vllm/releases "Releases · vllm-project/vllm · GitHub"))

Decision rule:

```text
If Unity has CUDA 13.x:
  use v0.20.2 with CUDA 13-compatible install.

If Unity has CUDA 12.9:
  use v0.20.2 with cu129 backend if the install validates cleanly.

If Unity only has CUDA 12.6:
  do not force v0.20.2 blindly.
  Either request a newer CUDA module from Unity,
  or pin an earlier vLLM stable release whose install validates cleanly.
```

Whatever version is chosen, pin it and record it in every result directory.

---

## Phase 4: install vLLM cleanly

Create a fresh project root, e.g.:

```text
~/vllm-sched/
  vllm/                         # forked vLLM checkout
  vllm-scheduler-policies/      # external scheduler package
  results/
  logs/
  profiles/
  docs/
```

Fork vLLM on GitHub. Clone your fork, add official vLLM as `upstream`, fetch tags, and checkout the chosen stable tag into a new experiment branch.

Use `uv` and a fresh Python virtual environment. vLLM’s current docs recommend the Python-only editable install for Python-only changes: `VLLM_USE_PRECOMPILED=1 uv pip install --editable . --torch-backend=auto`; this reuses prebuilt compiled libraries while reflecting Python source changes. ([vLLM](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/?utm_source=chatgpt.com "GPU - vLLM"))

Validation gate after install:

```text
- import vllm succeeds
- import torch succeeds
- torch.cuda.is_available() is true
- torch.version.cuda matches the intended backend
- CUDA_HOME points to the intended toolkit
- nvcc exists
- curand.h exists
- nvidia-smi sees the allocated GPU
```

Do not proceed to scheduler work until this is clean.

---

## Phase 5: run standard vLLM demo

Run a small model first, e.g. `Qwen/Qwen3-0.6B`.

Validation sequence:

```text
1. Start vLLM server.
2. Send one curl/OpenAI-compatible completion request.
3. Run vllm bench serve with a small synthetic/random workload.
4. Verify server logs have no FlashInfer/CUDA JIT failures.
5. Verify no environment workaround like VLLM_USE_FLASHINFER_SAMPLER=0 is needed.
```

Use conservative serving parameters first:

```text
max_model_len: small, e.g. 2048
max_num_batched_tokens: 2048
max_num_seqs: 64
gpu_memory_utilization: 0.80
```

Save every successful baseline’s environment:

```text
vLLM commit/tag
torch version
torch CUDA version
CUDA_HOME
nvcc version
nvidia-smi
GPU model
server command
benchmark command
server log
benchmark log
```

---

## Phase 6: run standard profiling

Use two profiling modes.

### 6.1 vLLM/PyTorch profiler

Use vLLM’s built-in profiler configuration. vLLM’s docs say to start the server with `--profiler-config` and run `vllm bench serve --profile`; they also warn to use only a few requests because traces get large and to increase `VLLM_RPC_TIMEOUT` because profiler flushing can be slow. ([vLLM](https://docs.vllm.ai/en/stable/contributing/profiling/?utm_source=chatgpt.com "Profiling vLLM - vLLM"))

Use this for:

```text
- CPU/PyTorch operator trace
- CUDA kernel timeline
- identifying large CPU gaps
- seeing attention, sampling, matmul, and memory operation timing
```

Do not use profiler runs as final benchmark numbers; profiling adds overhead.

### 6.2 Nsight Systems later

Use Nsight after basic PyTorch profiling works. Nsight is better for lower-level CUDA timeline analysis, but it is not the first debugging tool.

---

## Phase 7: create external scheduler package

Do **not** modify native vLLM scheduler files initially.

Use vLLM’s `scheduler_cls` mechanism. vLLM’s scheduler config docs state that `scheduler_cls` can be the default scheduler or a class path of the form `mod.custom_class`. ([vLLM](https://docs.vllm.ai/en/latest/api/vllm/config/scheduler/?utm_source=chatgpt.com "scheduler - vLLM"))

Create:

```text
~/vllm-sched/vllm-scheduler-policies/
  pyproject.toml
  vllm_scheduler_policies/
    __init__.py
    baseline.py
    instrumentation.py
    simple_policy_1.py
    common.py
  scripts/
    serve.sh
    bench.sh
```

Install this package editable into the same vLLM virtual environment.

First scheduler:

```text
BaselinePassthroughScheduler:
  subclass vllm.v1.core.sched.scheduler.Scheduler
  no behavior changes
  only verifies external scheduler loading
```

Run:

```text
vLLM default scheduler
vs.
BaselinePassthroughScheduler via --scheduler-cls
```

They should produce statistically similar results.

---

## Phase 8: scheduler swapping from bash

Create a small experiment launcher that accepts:

```text
scheduler name:
  default
  passthrough
  simple_policy_1
  later_policy_2
  latex_policy_v1

model
port
benchmark size
output directory
profiling on/off
```

The launcher should translate:

```text
scheduler=default
  → no --scheduler-cls

scheduler=passthrough
  → --scheduler-cls vllm_scheduler_policies.baseline.BaselinePassthroughScheduler

scheduler=simple_policy_1
  → --scheduler-cls vllm_scheduler_policies.simple_policy_1.SimplePolicy1Scheduler
```

Do not use bash scripts that overwrite vLLM source filgites or patch `scheduler.py` before each run. That makes experiments hard to reproduce.

---

## Phase 9: add scheduler instrumentation

Before implementing real algorithms, instrument scheduler overhead.

Target measurements:

```text
per scheduler call:
  scheduler_start_time
  scheduler_end_time
  scheduler_wall_time_ms
  number of waiting requests
  number of running requests
  number of scheduled requests
  number of scheduled tokens
  number of preemptions if available cheaply

per engine iteration if accessible:
  total iteration wall time
  model execution time
  scheduler time / total iteration time
  scheduler time / decode iteration time
```

The goal is to answer:

```text
How long does my scheduler take?
How does that compare to default scheduler?
How much of each decode iteration is scheduler overhead?
Does scheduler overhead grow with concurrency?
```

Use lightweight timing in Python first:

```text
time.perf_counter()
JSONL iteration log
one line per scheduler call
```

Do **not** rely only on PyTorch profiler for scheduler runtime. PyTorch profiler is useful for GPU/operator traces, but custom JSONL logging around the scheduler gives cleaner scheduler-specific measurements.

Later, correlate:

```text
scheduler JSONL logs
vLLM benchmark metrics
PyTorch profiler traces
server logs
```

---

## Phase 10: Codex setup

Use Codex as a constrained coding assistant, not as a whole-repo oracle.

Recommended workflow:

```text
Local VS Code
  Remote-SSH into Unity compute node

Inside remote terminal
  tmux
  vLLM virtual environment
  Codex CLI inside the repo
```

Create an `AGENTS.md` at the vLLM repo root or scheduler package root. It should tell Codex:

```text
Project:
  vLLM scheduler research

Relevant files:
  vllm/v1/core/sched/scheduler.py
  vllm/config/scheduler.py
  vllm-scheduler-policies/
  tests related to vLLM v1 scheduler

Rules:
  do not edit CUDA/C++ kernels
  do not modify native scheduler files unless explicitly asked
  prefer external scheduler classes
  keep patches small
  preserve default behavior
  add tests/smoke checks
  run targeted tests only
```

First Codex tasks should be read-only:

```text
1. Inspect current vLLM scheduler control flow.
2. Identify extension points.
3. Explain queue/state variables.
4. Explain how --scheduler-cls loads a custom scheduler.
5. Propose minimal passthrough scheduler.
6. Do not edit code yet.
```

Only after that allow it to edit the external scheduler package.

---

## Phase 11: handling the LaTeX algorithm

Do **not** feed the full mathematical LaTeX document to Codex initially.

Create a short implementation document:

```text
docs/simple_scheduler_v1.md
```

This should contain only:

```text
- one-paragraph algorithm goal
- exact vLLM state variables needed
- what score to compute
- how to order requests
- what behavior to preserve
- complexity target
- what to log
- known approximations
- what is explicitly out of scope
```

Also create:

```text
docs/latex_to_code_mapping.md
```

This maps:

```text
paper variable → vLLM object/field/method → directly available? → approximation?
```

Only after simple policies work should Codex read selected LaTeX excerpts.

Initial algorithm should be deliberately simple, for example:

```text
Policy 1:
  inherit default scheduler behavior
  only log queue sizes and timing

Policy 2:
  reorder waiting queue by a simple scalar score
  then delegate to default scheduling

Policy 3:
  add decode/prefill-aware priority
  still delegate most KV-cache and token-budget logic to default scheduler
```

Avoid implementing the full optimization formulation at first.