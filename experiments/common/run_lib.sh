#!/usr/bin/env bash

# Shared shell helpers for Unity vLLM scheduler experiment scripts.
#
# This file defines functions only. It intentionally does not use
# 'set -euo pipefail' for this project.

experiment_die() {
  echo "ERROR: $*" >&2
  return 1
}

experiment_note() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

experiment_repo_root() {
  if [ -n "${REPO_ROOT:-}" ] && [ -d "${REPO_ROOT}/.git" ]; then
    printf '%s\n' "${REPO_ROOT}"
    return 0
  fi

  git rev-parse --show-toplevel 2>/dev/null
}

experiment_require_repo_root() {
  local root
  root="$(experiment_repo_root)" || {
    experiment_die "could not determine repository root"
    return 1
  }

  if [ ! -f "${root}/pyproject.toml" ]; then
    experiment_die "repository root does not look like vllm-scheduler-policies: ${root}"
    return 1
  fi

  printf '%s\n' "${root}"
}

experiment_timestamp_utc() {
  date -u '+%Y%m%dT%H%M%SZ'
}

experiment_hostname_short() {
  hostname -s 2>/dev/null || hostname
}

experiment_make_run_id() {
  printf '%s_%s\n' "$(experiment_timestamp_utc)" "$(experiment_hostname_short)"
}

experiment_write_git_state() {
  local out_file="$1"
  local root
  root="$(experiment_require_repo_root)" || return 1

  {
    echo "repo_root=${root}"
    echo "branch=$(git -C "${root}" branch --show-current)"
    echo "commit=$(git -C "${root}" rev-parse HEAD)"
    echo
    echo "status_short:"
    git -C "${root}" status --short
  } > "${out_file}"
}

experiment_warn_if_forbidden_scheduler_env_set() {
  if [ -n "${VLLM_SCHEDULER_ITER_LOG:-}" ]; then
    echo "WARNING: VLLM_SCHEDULER_ITER_LOG is set but is forbidden for new experiments." >&2
    echo "Use SCHEDULER_POLICIES_ITER_LOG instead." >&2
    return 1
  fi

  return 0
}

experiment_require_compute_context_hint() {
  local host
  host="$(experiment_hostname_short)"

  case "${host}" in
    login*|unity-login*|ghlogin*|ood*)
      experiment_die "this appears to be a login node (${host}); do not run server or benchmark jobs here"
      return 1
      ;;
  esac

  return 0
}