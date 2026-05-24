#!/usr/bin/env bash
# Phase 8 vLLM serve launcher.
# Intentionally avoids "set -euo pipefail" for interactive Unity stability.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/scheduler_lib.sh"

SCHEDULER="default"
MODEL="Qwen/Qwen3-0.6B"
SERVED_MODEL_NAME="qwen3-0.6b"
HOST="127.0.0.1"
PORT="8000"
MAX_MODEL_LEN="2048"
MAX_NUM_BATCHED_TOKENS="2048"
MAX_NUM_SEQS="64"
GPU_MEMORY_UTILIZATION="0.80"
DRY_RUN="0"

usage() {
  cat <<'EOF'
Usage:
  scripts/serve.sh [options] [-- extra vllm serve args...]

Options:
  --scheduler NAME              default | passthrough | simple_policy_1 | primal_lp_dry_run | later_policy_2 | latex_policy_v1
  --model NAME                  Hugging Face model name/path
  --served-model-name NAME      OpenAI API model alias
  --host HOST                   default: 127.0.0.1
  --port PORT                   default: 8000
  --max-model-len N             default: 2048
  --max-num-batched-tokens N    default: 2048
  --max-num-seqs N              default: 64
  --gpu-memory-utilization X    default: 0.80
  --dry-run                     print command only
  -h, --help                    show this help
EOF
}

EXTRA_ARGS=()

while [ "$#" -gt 0 ]; do
  case "$1" in
    --scheduler)
      SCHEDULER="$2"
      shift 2
      ;;
    --model)
      MODEL="$2"
      shift 2
      ;;
    --served-model-name)
      SERVED_MODEL_NAME="$2"
      shift 2
      ;;
    --host)
      HOST="$2"
      shift 2
      ;;
    --port)
      PORT="$2"
      shift 2
      ;;
    --max-model-len)
      MAX_MODEL_LEN="$2"
      shift 2
      ;;
    --max-num-batched-tokens)
      MAX_NUM_BATCHED_TOKENS="$2"
      shift 2
      ;;
    --max-num-seqs)
      MAX_NUM_SEQS="$2"
      shift 2
      ;;
    --gpu-memory-utilization)
      GPU_MEMORY_UTILIZATION="$2"
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
      EXTRA_ARGS=("$@")
      break
      ;;
    *)
      echo "ERROR: unknown argument for serve.sh: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

phase8_bootstrap_env

SCHEDULER_CLS="$(phase8_scheduler_class "${SCHEDULER}")"
SCHEDULER_RC="$?"

if [ "${SCHEDULER_RC}" != "0" ]; then
  exit "${SCHEDULER_RC}"
fi

CMD=(
  vllm serve "${MODEL}"
  --served-model-name "${SERVED_MODEL_NAME}"
  --host "${HOST}"
  --port "${PORT}"
  --max-model-len "${MAX_MODEL_LEN}"
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}"
  --max-num-seqs "${MAX_NUM_SEQS}"
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
)

if [ -n "${SCHEDULER_CLS}" ]; then
  CMD+=(--scheduler-cls "${SCHEDULER_CLS}")
fi

CMD+=("${EXTRA_ARGS[@]}")

echo "Phase 8 serve launcher"
echo "scheduler=${SCHEDULER}"
echo "scheduler_cls=${SCHEDULER_CLS}"
echo "command:"
phase8_print_command "${CMD[@]}"

if [ "${DRY_RUN}" = "1" ]; then
  exit 0
fi

exec "${CMD[@]}"