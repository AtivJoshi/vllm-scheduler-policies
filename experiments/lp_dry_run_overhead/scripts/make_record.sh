#!/usr/bin/env bash

# Generate record.md for one LP dry-run overhead experiment run directory.
#
# This script does not start a server and does not run a benchmark.
# It intentionally does not use 'set -euo pipefail'.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

source "${REPO_ROOT}/experiments/common/env.sh"
source "${REPO_ROOT}/experiments/common/run_lib.sh"

RUN_DIR=""
OUT_PATH=""
TITLE="LP Dry-Run Overhead Run Record"

usage() {
  cat <<'EOF'
Usage:
  make_record.sh --run-dir PATH [options]

Options:
  --run-dir PATH      run directory to summarize
  --out PATH          output Markdown path; default: <run-dir>/record.md
  --title TEXT        Markdown title
  -h, --help          show this help
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --run-dir)
      RUN_DIR="$2"
      shift 2
      ;;
    --out)
      OUT_PATH="$2"
      shift 2
      ;;
    --title)
      TITLE="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "ERROR: unknown argument for make_record.sh: $1" >&2
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

if [ ! -d "${RUN_DIR}" ]; then
  echo "ERROR: run directory does not exist: ${RUN_DIR}" >&2
  exit 1
fi

CMD=(
  "${PYTHON_BIN}"
  "${REPO_ROOT}/experiments/common/make_record.py"
  --run-dir "${RUN_DIR}"
  --title "${TITLE}"
)

if [ -n "${OUT_PATH}" ]; then
  CMD+=(--out "${OUT_PATH}")
fi

echo "=== make_record ==="
echo "run_dir=${RUN_DIR}"
if [ -n "${OUT_PATH}" ]; then
  echo "out=${OUT_PATH}"
else
  echo "out=${RUN_DIR}/record.md"
fi
echo "title=${TITLE}"

"${CMD[@]}"