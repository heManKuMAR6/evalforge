"""EvalForge v2.0 — parallel judge swarm.

Fan-out 10 metrics across 5 judges in parallel via asyncio. One trace goes from
~12s sequential to ~1.5s. Backward compatible with v1.0 — existing
``evalforge.run()`` API is untouched.
"""
from .orchestrator import (
    SwarmResult,
    JudgeResult,
    run_swarm,
    run_swarm_async,
    select_metrics_for_trace,
)
from .consensus import (
    build_consensus,
    detect_disagreements,
    weighted_score,
    METRIC_WEIGHTS,
)
from .model_router import ModelRouter, JudgeConfig, load_config
from . import judges

__all__ = [
    "SwarmResult",
    "JudgeResult",
    "run_swarm",
    "run_swarm_async",
    "select_metrics_for_trace",
    "build_consensus",
    "detect_disagreements",
    "weighted_score",
    "METRIC_WEIGHTS",
    "ModelRouter",
    "JudgeConfig",
    "load_config",
    "judges",
]
