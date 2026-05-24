#!/usr/bin/env bash

# Start one vLLM server for the LP dry-run overhead experiment.
#
# This wrapper creates a run directory, records metadata, sets
# SCHEDULER_POLICIES_ITER_LOG, and launches scripts/serve.sh in the background.
# It does not use 'set -euo pipefail'.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

source "${REPO_ROOT}/experiments/common/env.sh"
source "${REPO_ROOT}/experiments/common/run_lib.sh"
source "${REPO_ROOT}/scripts/scheduler_lib.sh"

CONFIG="${EXPERIMENT_DIR}/configs/qwen3_0_6b_tiny.env"
SCHEDULER="simple_policy_1"
RUN_DIR=""
DRY_RUN="0"
EXTRA_SERVE_ARGS=()

usage() {
  cat <<'EOF'
Usage:
  start_server.sh [options] [-- extra scripts/serve.sh args...]

Options:
  --config PATH       shell config file; default: configs/qwen3_0_6b_tiny.env
  --scheduler NAME    simple_policy_1 | primal_lp_dry_run
  --run-dir PATH      run directory; default: runs/<timestamp>_<host>_<scheduler>
  --dry-run           print actions and command, but do not start server
  -h, --help          show this help
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --config)
      CONFIG="$2"
      shift 2
      ;;
    --scheduler)
      SCHEDULER="$2"
      shift 2
      ;;
    --run-dir)
      RUN_DIR="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN="1"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      EXTRA_SERVE_ARGS=("$@")
      break
      ;;
    *)
      echo "ERROR: unknown argument for start_server.sh: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [ ! -f "${CONFIG}" ]; then
  echo "ERROR: config file not found: ${CONFIG}" >&2
  exit 1
fi

# shellcheck disable=SC1090
source "${CONFIG}"

if [ "${SCHEDULER}" != "simple_policy_1" ] && [ "${SCHEDULER}" != "primal_lp_dry_run" ]; then
  echo "ERROR: unsupported scheduler for this experiment: ${SCHEDULER}" >&2
  echo "Expected: simple_policy_1 or primal_lp_dry_run" >&2
  exit 2
fi

if ! experiment_warn_if_forbidden_scheduler_env_set; then
  exit 1
fi

if [ -z "${RUN_DIR}" ]; then
  RUN_ID="$(experiment_make_run_id)_${SCHEDULER}"
  RUN_DIR="${EXPERIMENT_DIR}/runs/${RUN_ID}"
fi

SCHEDULER_JSONL="${RUN_DIR}/scheduler_iter.jsonl"
SERVER_LOG="${RUN_DIR}/server.log"
SERVER_PID="${RUN_DIR}/server.pid"
COMMANDS_FILE="${RUN_DIR}/commands.sh"
GIT_STATE_FILE="${RUN_DIR}/git_state.txt"
ENV_FILE="${RUN_DIR}/env.txt"

SERVE_CMD=(
  "${REPO_ROOT}/scripts/serve.sh"
  --scheduler "${SCHEDULER}"
  --model "${MODEL}"
  --served-model-name "${SERVED_MODEL_NAME}"
  --host "${HOST}"
  --port "${PORT}"
  --max-model-len "${MAX_MODEL_LEN}"
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}"
  --max-num-seqs "${MAX_NUM_SEQS}"
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
)

if [ "${#EXTRA_SERVE_ARGS[@]}" -gt 0 ]; then
  SERVE_CMD+=(-- "${EXTRA_SERVE_ARGS[@]}")
fi

echo "=== start_server plan ==="
echo "config=${CONFIG}"
echo "scheduler=${SCHEDULER}"
echo "run_dir=${RUN_DIR}"
echo "scheduler_jsonl=${SCHEDULER_JSONL}"
echo "server_log=${SERVER_LOG}"
echo "server_pid=${SERVER_PID}"
echo "command:"
phase8_print_command "${SERVE_CMD[@]}"

if [ "${DRY_RUN}" = "1" ]; then
  echo "dry_run=1"
  exit 0
fi

if ! experiment_require_compute_context_hint; then
  exit 1
fi

mkdir -p "${RUN_DIR}"

experiment_write_git_state "${GIT_STATE_FILE}" || exit 1

{
  echo "CONFIG=${CONFIG}"
  echo "SCHEDULER=${SCHEDULER}"
  echo "RUN_DIR=${RUN_DIR}"
  echo "MODEL=${MODEL}"
  echo "SERVED_MODEL_NAME=${SERVED_MODEL_NAME}"
  echo "TOKENIZER=${TOKENIZER}"
  echo "HOST=${HOST}"
  echo "PORT=${PORT}"
  echo "MAX_MODEL_LEN=${MAX_MODEL_LEN}"
  echo "MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS}"
  echo "MAX_NUM_SEQS=${MAX_NUM_SEQS}"
  echo "GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION}"
  echo "SCHEDULER_POLICIES_ITER_LOG=${SCHEDULER_JSONL}"
} > "${ENV_FILE}"

{
  echo "# Reconstructed server command"
  echo "export SCHEDULER_POLICIES_ITER_LOG=${SCHEDULER_JSONL@Q}"
  phase8_print_command "${SERVE_CMD[@]}"
} > "${COMMANDS_FILE}"

export SCHEDULER_POLICIES_ITER_LOG="${SCHEDULER_JSONL}"

echo "starting server..."
"${SERVE_CMD[@]}" > "${SERVER_LOG}" 2>&1 &
SERVER_PID_VALUE="$!"
echo "${SERVER_PID_VALUE}" > "${SERVER_PID}"

echo "server_pid=${SERVER_PID_VALUE}"
echo "run_dir=${RUN_DIR}"
echo "server_log=${SERVER_LOG}"
echo "scheduler_jsonl=${SCHEDULER_JSONL}"
echo "start_server_completed"