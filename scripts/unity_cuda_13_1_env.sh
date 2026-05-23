# Source this inside a Unity GPU allocation before CUDA/JIT/vLLM work.
# Usage:
#   source ~/vllm-sched/vllm-scheduler-policies/scripts/unity_cuda_13_1_env.sh

module load cuda/13.1

export CUDA_PATH="$CUDA_HOME"

_CUDA_LIBDIR="$CUDA_HOME/targets/x86_64-linux/lib"
if [ ! -d "$_CUDA_LIBDIR" ]; then
  echo "ERROR: expected CUDA library directory not found: $_CUDA_LIBDIR" >&2
  return 1 2>/dev/null || exit 1
fi

case ":${LD_LIBRARY_PATH:-}:" in
  *":$_CUDA_LIBDIR:"*) ;;
  *) export LD_LIBRARY_PATH="$_CUDA_LIBDIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" ;;
esac

unset _CUDA_LIBDIR
