#!/usr/bin/env bash

# Stop the server recorded by a specific run directory.
#
# This script only uses the pid file in the provided run directory.
# It does not kill arbitrary vLLM, Python, or engine processes.
# It intentionally does not use 'set -euo pipefail'.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

source "${REPO_ROOT}/experiments/common/env.sh"
source "${REPO_ROOT}/experiments/common/run_lib.sh"

RUN_DIR=""
WAIT_SECONDS="20"
DRY_RUN="0"

usage() {
  cat <<'EOF'
Usage:
  stop_server.sh --run-dir PATH [options]

Options:
  --run-dir PATH      run directory containing server.pid
  --wait-seconds N    seconds to wait before SIGKILL; default: 20
  --dry-run           print intended action, but do not signal process
  -h, --help          show this help
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --run-dir)
      RUN_DIR="$2"
      shift 2
      ;;
    --wait-seconds)
      WAIT_SECONDS="$2"
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
    *)
      echo "ERROR: unknown argument for stop_server.sh: $1" >&2
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

PID_FILE="${RUN_DIR}/server.pid"

if [ ! -f "${PID_FILE}" ]; then
  echo "ERROR: pid file not found: ${PID_FILE}" >&2
  exit 1
fi

SERVER_PID="$(cat "${PID_FILE}")"

case "${SERVER_PID}" in
  ''|*[!0-9]*)
    echo "ERROR: invalid server pid in ${PID_FILE}: ${SERVER_PID}" >&2
    exit 1
    ;;
esac

echo "=== stop_server plan ==="
echo "run_dir=${RUN_DIR}"
echo "pid_file=${PID_FILE}"
echo "server_pid=${SERVER_PID}"
echo "wait_seconds=${WAIT_SECONDS}"

if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
  echo "server process is not running"
  echo "stop_server_completed"
  exit 0
fi

if [ "${DRY_RUN}" = "1" ]; then
  echo "dry_run=1"
  echo "would send SIGTERM to pid ${SERVER_PID}"
  exit 0
fi

echo "sending SIGTERM to pid ${SERVER_PID}"
kill "${SERVER_PID}" 2>/dev/null || true

elapsed="0"
while kill -0 "${SERVER_PID}" 2>/dev/null; do
  if [ "${elapsed}" -ge "${WAIT_SECONDS}" ]; then
    echo "server still running after ${WAIT_SECONDS}s; sending SIGKILL to pid ${SERVER_PID}"
    kill -9 "${SERVER_PID}" 2>/dev/null || true
    break
  fi

  sleep 1
  elapsed="$((elapsed + 1))"
done

echo "stop_server_completed"