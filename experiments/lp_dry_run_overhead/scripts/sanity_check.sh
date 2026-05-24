#!/usr/bin/env bash

# Sanity check for LP dry-run overhead experiment setup.
#
# This script does not start a server and does not run a benchmark.
# It checks repository state, Unity environment basics, Python imports,
# scheduler logging environment variables, and GPU visibility.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

source "${REPO_ROOT}/experiments/common/env.sh"
source "${REPO_ROOT}/experiments/common/run_lib.sh"

echo "=== Phase 13.3 sanity check ==="
echo "repo_root=${REPO_ROOT}"
echo "hostname=$(hostname)"
echo "date_utc=$(date -u '+%Y-%m-%dT%H:%M:%SZ')"

echo
echo "=== git state ==="
echo "branch=$(git -C "${REPO_ROOT}" branch --show-current)"
echo "commit=$(git -C "${REPO_ROOT}" rev-parse HEAD)"
echo "status_short:"
git -C "${REPO_ROOT}" status --short

echo
echo "=== scheduler JSONL environment ==="
if [ -n "${SCHEDULER_POLICIES_ITER_LOG:-}" ]; then
  echo "SCHEDULER_POLICIES_ITER_LOG=${SCHEDULER_POLICIES_ITER_LOG}"
else
  echo "SCHEDULER_POLICIES_ITER_LOG is not set"
fi

if [ -n "${VLLM_SCHEDULER_ITER_LOG:-}" ]; then
  echo "WARNING: VLLM_SCHEDULER_ITER_LOG is set but forbidden for new experiments"
  echo "VLLM_SCHEDULER_ITER_LOG=${VLLM_SCHEDULER_ITER_LOG}"
else
  echo "VLLM_SCHEDULER_ITER_LOG is not set"
fi

echo
echo "=== Python ==="
echo "PYTHON_BIN=${PYTHON_BIN}"
"${PYTHON_BIN}" --version

echo
echo "=== package imports ==="
"${PYTHON_BIN}" - <<'PY'
import importlib

modules = [
    "vllm_scheduler_policies",
    "vllm_scheduler_policies.simple_policy_1",
    "vllm_scheduler_policies.primal_lp_dry_run",
    "vllm_scheduler_policies.primal_lp",
]

for name in modules:
    importlib.import_module(name)
    print(f"ok import {name}")
PY

echo
echo "=== GPU visibility ==="
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader || true
else
  echo "nvidia-smi not found"
fi

echo
echo "=== compute-node hint ==="
if experiment_require_compute_context_hint; then
  echo "host does not look like a login node"
else
  echo "compute-node hint failed; this is a warning for future server/benchmark runs"
fi

echo
echo "=== result ==="
echo "sanity_check_completed"