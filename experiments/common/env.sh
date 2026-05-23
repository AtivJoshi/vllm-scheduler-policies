#!/usr/bin/env bash

# Common environment bootstrap for Unity vLLM scheduler experiments.
#
# Source this file from experiment scripts. Do not execute it directly if the
# caller needs environment changes to persist in the current shell.

if [ -n "${EXPERIMENTS_COMMON_ENV_SH_LOADED:-}" ]; then
  return 0 2>/dev/null || exit 0
fi
EXPERIMENTS_COMMON_ENV_SH_LOADED=1
export EXPERIMENTS_COMMON_ENV_SH_LOADED

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export REPO_ROOT

UNITY_ENV_SCRIPT="${REPO_ROOT}/scripts/unity_cuda_13_1_env.sh"

if [ ! -f "${UNITY_ENV_SCRIPT}" ]; then
  echo "ERROR: missing Unity environment script: ${UNITY_ENV_SCRIPT}" >&2
  return 1 2>/dev/null || exit 1
fi

# This script intentionally does not use 'set -euo pipefail'. That combination
# has caused remote Unity session instability in this project.
source "${UNITY_ENV_SCRIPT}"

VLLM_SCHED_VENV="${VLLM_SCHED_VENV:-/home/atjoshi_umass_edu/vllm-sched/.venv}"
export VLLM_SCHED_VENV

PYTHON_BIN="${PYTHON_BIN:-${VLLM_SCHED_VENV}/bin/python}"
export PYTHON_BIN

if [ ! -x "${PYTHON_BIN}" ]; then
  echo "ERROR: Python executable not found or not executable: ${PYTHON_BIN}" >&2
  return 1 2>/dev/null || exit 1
fi