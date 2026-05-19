"""External scheduler policies for vLLM scheduler research."""

__all__ = ["BaselinePassthroughScheduler"]


def __getattr__(name: str):
    if name == "BaselinePassthroughScheduler":
        from vllm_scheduler_policies.baseline import BaselinePassthroughScheduler

        return BaselinePassthroughScheduler
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
