"""Swarm orchestrator — pick applicable judges, fan-out, build consensus.

Public entry points:

  - :func:`run_swarm`         synchronous wrapper, useful in tests
  - :func:`run_swarm_async`   the real async path used by the CLI
  - :func:`select_metrics_for_trace` — picks which judges to run from trace contents
"""
from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from .judges import JUDGE_METRICS, JudgeResult, run_all_judges
from .consensus import build_consensus
from .model_router import ModelRouter


# Public re-export so callers don't have to import from .judges
JudgeResult = JudgeResult   # noqa: PLW0127 — intentional alias for re-export


@dataclass
class SwarmResult:
    """Structured form of the swarm output. ``raw`` is the JSON-serialisable
    dict produced by :func:`build_consensus`."""
    trace_id: str
    passed: bool
    weighted_score: float
    scores: dict[str, float] = field(default_factory=dict)
    judge_timings_ms: dict[str, int] = field(default_factory=dict)
    anomalies: list = field(default_factory=list)
    human_review_flags: list = field(default_factory=list)
    failure_reason: str | None = None
    total_elapsed_ms: int = 0
    raw: dict = field(default_factory=dict)

    @classmethod
    def from_consensus(cls, consensus: dict, total_elapsed_ms: int) -> "SwarmResult":
        sr = consensus["swarm_result"]
        return cls(
            trace_id=consensus["trace_id"],
            passed=sr["overall"] == "PASS",
            weighted_score=sr["weighted_score"],
            scores=sr["scores"],
            judge_timings_ms=sr["judge_timings_ms"],
            anomalies=sr["anomalies"],
            human_review_flags=sr["human_review_flags"],
            failure_reason=sr["failure_reason"],
            total_elapsed_ms=total_elapsed_ms,
            raw=consensus,
        )


def select_metrics_for_trace(trace: dict) -> list[str]:
    """Pick which judges to run based on trace contents.

    Heuristics:
      - tool_call steps present       → judge D (tool_accuracy)
      - retrieval / tool outputs      → judges A, C
      - any code fences in output     → judges D (code_security), E
      - always                        → judge B (goal_completion + g_eval)

    Returns the list of judge IDs (subset of "A".."E"). If nothing else
    matches, B is always included so a trace produces *some* result.
    """
    selected: set[str] = {"B"}   # always evaluate goal completion + g_eval

    steps = trace.get("steps", []) or []
    has_tool_calls = any(s.get("type") == "tool_call" for s in steps)
    if has_tool_calls:
        # tool outputs feed faithfulness/hallucination context and retrieval
        # context for context_precision/answer_relevance
        selected.update({"A", "C", "D"})

    answer = trace.get("output", {}).get("answer", "") or ""
    contains_code = "```" in answer or any(
        "```" in (s.get("content") or "") for s in steps
    )
    if contains_code:
        selected.update({"D", "E"})

    return sorted(selected)


def _load_trace(trace_input: str | dict | Path) -> dict:
    """Accepts a file path, an already-parsed dict, or a JSON string."""
    if isinstance(trace_input, dict):
        return trace_input
    if isinstance(trace_input, Path):
        return json.loads(trace_input.read_text())
    # str — could be a path or JSON
    if trace_input.startswith("{"):
        return json.loads(trace_input)
    return json.loads(Path(trace_input).read_text())


async def run_swarm_async(
    trace: str | dict | Path,
    *,
    model: str | None = None,
    mock: bool = False,
    threshold: float = 0.7,
    consensus_threshold: float = 0.3,
    rubric: str | None = None,
    judges_to_run: list[str] | None = None,
) -> SwarmResult:
    """Run the swarm against a single trace, in parallel across judges."""
    trace_dict = _load_trace(trace)
    trace_id = trace_dict.get("trace_id", "unknown")

    selected = judges_to_run or select_metrics_for_trace(trace_dict)
    router = None if mock else ModelRouter(model=model)

    start = time.perf_counter()
    judge_results = await run_all_judges(
        trace_dict,
        router=router,
        judges_to_run=selected,
        mock=mock,
        rubric=rubric,
    )
    total_elapsed_ms = int((time.perf_counter() - start) * 1000)

    consensus = build_consensus(
        judge_results,
        trace_id=trace_id,
        threshold=threshold,
        consensus_threshold=consensus_threshold,
    )
    return SwarmResult.from_consensus(consensus, total_elapsed_ms)


def run_swarm(
    trace: str | dict | Path,
    *,
    model: str | None = None,
    mock: bool = False,
    threshold: float = 0.7,
    consensus_threshold: float = 0.3,
    rubric: str | None = None,
    judges_to_run: list[str] | None = None,
) -> SwarmResult:
    """Synchronous wrapper around :func:`run_swarm_async`."""
    return asyncio.run(run_swarm_async(
        trace,
        model=model,
        mock=mock,
        threshold=threshold,
        consensus_threshold=consensus_threshold,
        rubric=rubric,
        judges_to_run=judges_to_run,
    ))


async def run_swarm_batch_async(
    traces_dir: str | Path,
    *,
    model: str | None = None,
    mock: bool = False,
    threshold: float = 0.7,
    consensus_threshold: float = 0.3,
    output: str | Path | None = None,
) -> list[SwarmResult]:
    """Run the swarm against every JSON trace in a directory.

    Each trace is processed in parallel-internally, but traces are processed
    sequentially so output ordering is stable. (Two layers of fan-out tend to
    hammer rate limits.)
    """
    base = Path(traces_dir)
    paths = sorted(p for p in base.glob("*.json"))
    results: list[SwarmResult] = []
    for path in paths:
        try:
            sr = await run_swarm_async(
                path,
                model=model,
                mock=mock,
                threshold=threshold,
                consensus_threshold=consensus_threshold,
            )
            results.append(sr)
            if output is not None:
                out_dir = Path(output)
                out_dir.mkdir(parents=True, exist_ok=True)
                out_path = out_dir / f"{sr.trace_id}.json"
                out_path.write_text(json.dumps(sr.raw, indent=2))
        except Exception as exc:  # noqa: BLE001 — keep batch running on failures
            print(f"[swarm] skipping {path.name}: {exc}")
    return results
