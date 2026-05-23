#!/usr/bin/env bash

# Run one tiny benchmark workload against an already-running server.
#
# This script does not start or stop the server.
# It intentionally does not use 'set -euo pipefail'.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

source "${REPO_ROOT}/experiments/common/env.sh"
source "${REPO_ROOT}/experiments/common/run_lib.sh"
source "${REPO_ROOT}/scripts/scheduler_lib.sh"

CONFIG="${EXPERIMENT_DIR}/configs/qwen3_0_6b_tiny.env"
SCHEDULER="simple_policy_1"
RUN_DIR=""
WORKLOAD_LABEL="tiny"
DRY_RUN="0"
EXTRA_BENCH_ARGS=()

usage() {
  cat <<'EOF'
Usage:
  run_one_workload.sh --run-dir PATH [options] [-- extra scripts/bench.sh args...]

Options:
  --run-dir PATH       existing run directory created by start_server.sh
  --config PATH        shell config file; default: configs/qwen3_0_6b_tiny.env
  --scheduler NAME     label for output; simple_policy_1 | primal_lp_dry_run
  --workload-label X   label used in benchmark log filename; default: tiny
  --dry-run            print command, but do not run benchmark
  -h, --help           show this help
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --run-dir)
      RUN_DIR="$2"
      shift 2
      ;;
    --config)
      CONFIG="$2"
      shift 2
      ;;
    --scheduler)
      SCHEDULER="$2"
      shift 2
      ;;
    --workload-label)
      WORKLOAD_LABEL="$2"
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
      EXTRA_BENCH_ARGS=("$@")
      break
      ;;
    *)
      echo "ERROR: unknown argument for run_one_workload.sh: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [ -z "${RUN_DIR}" ]; then
  echo "ERROR: --run-dir is required" >&2
  usage >&2
  exit 2
fi

if [ ! -f "${CONFIG}" ]; then
  echo "ERROR: config file not found: ${CONFIG}" >&2
  exit 1
fi

# shellcheck disable=SC1090
source "${CONFIG}"

if [ ! -d "${RUN_DIR}" ] && [ "${DRY_RUN}" != "1" ]; then
  echo "ERROR: run directory does not exist: ${RUN_DIR}" >&2
  exit 1
fi

SCHEDULER_JSONL="${RUN_DIR}/scheduler_iter.jsonl"
BENCH_LOG="${RUN_DIR}/bench_${WORKLOAD_LABEL}.log"

BENCH_CMD=(
  "${REPO_ROOT}/scripts/bench.sh"
  --scheduler "${SCHEDULER}"
  --model "${SERVED_MODEL_NAME}"
  --tokenizer "${TOKENIZER}"
  --host "${HOST}"
  --port "${PORT}"
  --num-prompts "${NUM_PROMPTS}"
  --random-input-len "${RANDOM_INPUT_LEN}"
  --random-output-len "${RANDOM_OUTPUT_LEN}"
  --max-concurrency "${MAX_CONCURRENCY}"
  --request-rate "${REQUEST_RATE}"
  --seed "${SEED}"
)

if [ "${#EXTRA_BENCH_ARGS[@]}" -gt 0 ]; then
  BENCH_CMD+=(-- "${EXTRA_BENCH_ARGS[@]}")
fi

before_lines="0"
if [ -f "${SCHEDULER_JSONL}" ]; then
  before_lines="$(wc -l < "${SCHEDULER_JSONL}")"
fi

echo "=== run_one_workload plan ==="
echo "config=${CONFIG}"
echo "scheduler=${SCHEDULER}"
echo "run_dir=${RUN_DIR}"
echo "workload_label=${WORKLOAD_LABEL}"
echo "bench_log=${BENCH_LOG}"
echo "scheduler_jsonl=${SCHEDULER_JSONL}"
echo "scheduler_jsonl_lines_before=${before_lines}"
echo "command:"
phase8_print_command "${BENCH_CMD[@]}"

if [ "${DRY_RUN}" = "1" ]; then
  echo "dry_run=1"
  exit 0
fi

if ! experiment_require_compute_context_hint; then
  exit 1
fi

"${BENCH_CMD[@]}" > "${BENCH_LOG}" 2>&1
bench_rc="$?"

after_lines="0"
if [ -f "${SCHEDULER_JSONL}" ]; then
  after_lines="$(wc -l < "${SCHEDULER_JSONL}")"
fi

echo "bench_rc=${bench_rc}"
echo "scheduler_jsonl_lines_before=${before_lines}"
echo "scheduler_jsonl_lines_after=${after_lines}"
echo "bench_log=${BENCH_LOG}"

if [ "${bench_rc}" != "0" ]; then
  echo "run_one_workload_failed"
  exit "${bench_rc}"
fi

echo "run_one_workload_completed"