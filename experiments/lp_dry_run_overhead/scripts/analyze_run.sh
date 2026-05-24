#!/usr/bin/env bash

# Analyze one LP dry-run overhead experiment run directory.
#
# This script does not start a server and does not run a benchmark.
# It reads scheduler_iter.jsonl and writes analysis.json plus analysis.txt.
# It intentionally does not use 'set -euo pipefail'.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

source "${REPO_ROOT}/experiments/common/env.sh"
source "${REPO_ROOT}/experiments/common/run_lib.sh"

RUN_DIR=""
JSONL_PATH=""
JSON_OUT=""
TEXT_OUT=""

usage() {
  cat <<'EOF'
Usage:
  analyze_run.sh --run-dir PATH [options]

Options:
  --run-dir PATH      run directory containing scheduler_iter.jsonl
  --jsonl PATH        override scheduler JSONL path
  --json-out PATH     override output JSON path
  --text-out PATH     override output text path
  -h, --help          show this help
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --run-dir)
      RUN_DIR="$2"
      shift 2
      ;;
    --jsonl)
      JSONL_PATH="$2"
      shift 2
      ;;
    --json-out)
      JSON_OUT="$2"
      shift 2
      ;;
    --text-out)
      TEXT_OUT="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "ERROR: unknown argument for analyze_run.sh: $1" >&2
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

if [ -z "${JSONL_PATH}" ]; then
  JSONL_PATH="${RUN_DIR}/scheduler_iter.jsonl"
fi

if [ -z "${JSON_OUT}" ]; then
  JSON_OUT="${RUN_DIR}/analysis.json"
fi

if [ -z "${TEXT_OUT}" ]; then
  TEXT_OUT="${RUN_DIR}/analysis.txt"
fi

if [ ! -d "${RUN_DIR}" ]; then
  echo "ERROR: run directory does not exist: ${RUN_DIR}" >&2
  exit 1
fi

if [ ! -f "${JSONL_PATH}" ]; then
  echo "ERROR: scheduler JSONL file does not exist: ${JSONL_PATH}" >&2
  exit 1
fi

echo "=== analyze_run ==="
echo "run_dir=${RUN_DIR}"
echo "jsonl=${JSONL_PATH}"
echo "json_out=${JSON_OUT}"
echo "text_out=${TEXT_OUT}"

"${PYTHON_BIN}" "${REPO_ROOT}/experiments/common/analyze_scheduler_jsonl.py" \
  "${JSONL_PATH}" \
  --json-out "${JSON_OUT}" \
  --text-out "${TEXT_OUT}"

echo
echo "analysis_json=${JSON_OUT}"
echo "analysis_txt=${TEXT_OUT}"
echo "analyze_run_completed"