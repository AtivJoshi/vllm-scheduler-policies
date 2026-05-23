#!/usr/bin/env bash
# Shared helpers for Phase 8 scheduler-swapping launchers.
# Intentionally avoids "set -euo pipefail" for interactive Unity stability.

phase8_script_dir() {
  cd "$(dirname "${BASH_SOURCE[0]}")" && pwd
}

phase8_package_root() {
  cd "$(phase8_script_dir)/.." && pwd
}

phase8_workspace_root() {
  cd "$(phase8_package_root)/.." && pwd
}

phase8_bootstrap_env() {
  local ws
  ws="${VLLM_SCHED_WORKSPACE:-$(phase8_workspace_root)}"

  if [ -f "${ws}/vllm-scheduler-policies/scripts/unity_cuda_13_1_env.sh" ]; then
    source "${ws}/vllm-scheduler-policies/scripts/unity_cuda_13_1_env.sh"
  fi

  if [ -f "${ws}/.venv/bin/activate" ]; then
    source "${ws}/.venv/bin/activate"
  fi
}

phase8_scheduler_class() {
  local scheduler
  scheduler="$1"

  case "${scheduler}" in
    default)
      return 0
      ;;
    passthrough)
      printf '%s\n' "vllm_scheduler_policies.baseline.BaselinePassthroughScheduler"
      ;;
    simple_policy_1)
      printf '%s\n' "vllm_scheduler_policies.simple_policy_1.SimplePolicy1Scheduler"
      ;;
    later_policy_2)
      printf '%s\n' "vllm_scheduler_policies.later_policy_2.LaterPolicy2Scheduler"
      ;;
    latex_policy_v1)
      printf '%s\n' "vllm_scheduler_policies.latex_policy_v1.LatexPolicyV1Scheduler"
      ;;
    *)
      echo "ERROR: unknown scheduler '${scheduler}'" >&2
      echo "Known schedulers: default, passthrough, simple_policy_1, later_policy_2, latex_policy_v1" >&2
      return 2
      ;;
  esac
}

phase8_print_command() {
  local x
  for x in "$@"; do
    printf '%q ' "$x"
  done
  printf '\n'
}