#!/usr/bin/env bash
# Phase 8 vLLM benchmark launcher.
# Intentionally avoids "set -euo pipefail" for interactive Unity stability.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/scheduler_lib.sh"

SCHEDULER="default"
MODEL_ALIAS="qwen3-0.6b"
TOKENIZER="Qwen/Qwen3-0.6B"
HOST="127.0.0.1"
PORT="8000"
ENDPOINT="/v1/chat/completions"
NUM_PROMPTS="4"
RANDOM_INPUT_LEN="32"
RANDOM_OUTPUT_LEN="16"
MAX_CONCURRENCY="2"
REQUEST_RATE="inf"
SEED="0"
PROFILE="0"
DRY_RUN="0"

usage() {
  cat <<'EOF'
Usage:
  scripts/bench.sh [options] [-- extra vllm bench serve args...]

Options:
  --scheduler NAME           label only; default: default
  --model NAME               served model alias; default: qwen3-0.6b
  --tokenizer NAME           tokenizer path/name; default: Qwen/Qwen3-0.6B
  --host HOST                default: 127.0.0.1
  --port PORT                default: 8000
  --endpoint PATH            default: /v1/chat/completions
  --num-prompts N            default: 4
  --random-input-len N       default: 32
  --random-output-len N      default: 16
  --max-concurrency N        default: 2
  --request-rate X           default: inf
  --seed N                   default: 0
  --profile                  add vllm bench serve --profile
  --dry-run                  print command only
  -h, --help                 show this help
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
      MODEL_ALIAS="$2"
      shift 2
      ;;
    --tokenizer)
      TOKENIZER="$2"
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
    --endpoint)
      ENDPOINT="$2"
      shift 2
      ;;
    --num-prompts)
      NUM_PROMPTS="$2"
      shift 2
      ;;
    --random-input-len)
      RANDOM_INPUT_LEN="$2"
      shift 2
      ;;
    --random-output-len)
      RANDOM_OUTPUT_LEN="$2"
      shift 2
      ;;
    --max-concurrency)
      MAX_CONCURRENCY="$2"
      shift 2
      ;;
    --request-rate)
      REQUEST_RATE="$2"
      shift 2
      ;;
    --seed)
      SEED="$2"
      shift 2
      ;;
    --profile)
      PROFILE="1"
      shift
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
      echo "ERROR: unknown argument for bench.sh: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

phase8_bootstrap_env

CMD=(
  vllm bench serve
  --backend openai-chat
  --base-url "http://${HOST}:${PORT}"
  --endpoint "${ENDPOINT}"
  --model "${MODEL_ALIAS}"
  --tokenizer "${TOKENIZER}"
  --dataset-name random
  --random-input-len "${RANDOM_INPUT_LEN}"
  --random-output-len "${RANDOM_OUTPUT_LEN}"
  --num-prompts "${NUM_PROMPTS}"
  --request-rate "${REQUEST_RATE}"
  --max-concurrency "${MAX_CONCURRENCY}"
  --seed "${SEED}"
)

if [ "${PROFILE}" = "1" ]; then
  CMD+=(--profile)
fi

CMD+=("${EXTRA_ARGS[@]}")

echo "Phase 8 bench launcher"
echo "scheduler_label=${SCHEDULER}"
echo "command:"
phase8_print_command "${CMD[@]}"

if [ "${DRY_RUN}" = "1" ]; then
  exit 0
fi

exec "${CMD[@]}"