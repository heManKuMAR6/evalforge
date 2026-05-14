"""Async judges — five judges, each scoring two related metrics in one LLM call.

  Judge A: faithfulness + hallucination      (tool-output context)
  Judge B: goal_completion + g_eval          (task context)
  Judge C: context_precision + answer_relevance  (retrieval context)
  Judge D: tool_accuracy + code_security     (deterministic, no LLM call)
  Judge E: code_correctness + code_quality   (code context)

Every judge has a ``mock=True`` path that returns deterministic scores so the
swarm runs end-to-end without API keys (used by ``--mock`` and by tests).
"""
from __future__ import annotations

import asyncio
import json
import re
import time
from dataclasses import dataclass, field
from typing import Any

from .model_router import ModelRouter, JudgeConfig

# Per-item character limit for trace content sent to LLM judges. DeepSeek (and
# other OpenAI-compatible providers) occasionally return empty strings when the
# prompt is too large; capping each tool output keeps prompts predictable.
# Tightened from 500 → 300 after observing DeepSeek Judge A take 21s on
# sample_trace.json — prompt size dominates latency at this scale.
TOOL_OUTPUT_CHAR_LIMIT = 300
# Code blocks for judge E get a more generous cap — code is dense signal.
CODE_FIELD_CHAR_LIMIT = 4000
# Per-attempt HTTP timeout. Worst case ≈ 3 × (15s + 1s delay) ≈ 48s per judge.
JUDGE_HTTP_TIMEOUT_SECONDS = 15
# Retries are triggered only by empty responses (the source of the original
# "Expecting value: line 1 column 1 (char 0)" error). HTTP errors fail fast.
JUDGE_MAX_RETRIES = 3
JUDGE_RETRY_DELAY_SECONDS = 1.0

# Deterministic mock scores, matching the Rust CLI's mock values so v1.0 and
# v2.0 produce identical output in mock mode (eases regression testing).
MOCK_SCORES: dict[str, float] = {
    "faithfulness": 0.91,
    "hallucination": 0.95,
    "goal_completion": 0.85,
    "g_eval": 0.88,
    "context_precision": 0.80,
    "answer_relevance": 0.95,
    "tool_accuracy": 1.0,
    "code_security": 0.95,
    "code_correctness": 0.85,
    "code_quality": 0.80,
}

MOCK_REASONS: dict[str, str] = {
    "faithfulness": "Mock — answer aligns with retrieved context",
    "hallucination": "Mock — no hallucinations detected",
    "goal_completion": "Mock — goal appears completed",
    "g_eval": "Mock — response meets rubric criteria",
    "context_precision": "Mock — retrieved context was relevant",
    "answer_relevance": "Mock — answer addresses the question",
    "tool_accuracy": "Mock — all expected tools used",
    "code_security": "Mock — no security issues found",
    "code_correctness": "Mock — code appears correct",
    "code_quality": "Mock — code quality is good",
}


@dataclass
class JudgeResult:
    """One judge's verdict: per-metric scores + reasoning + timing."""
    judge: str                  # "A".."E"
    metrics: list[str]
    scores: dict[str, float] = field(default_factory=dict)
    reasons: dict[str, str] = field(default_factory=dict)
    elapsed_ms: int = 0
    model: str = ""
    error: str | None = None


# ---------------------------------------------------------------------------
# Trace field extractors — pull only the parts a judge needs from the trace.
# ---------------------------------------------------------------------------


def _truncate(text: str, limit: int) -> str:
    """Cut text to ``limit`` chars; mark the truncation so the LLM can see it."""
    if len(text) <= limit:
        return text
    return f"{text[:limit]}... [truncated, {len(text) - limit} more chars]"


def _tool_output_context(trace: dict, limit: int = TOOL_OUTPUT_CHAR_LIMIT) -> str:
    """Concatenated outputs of all tool_call steps. Used by judges A and C.

    Each tool output is independently capped at ``limit`` chars so a single
    very-large output (e.g. a 50 KB web-search dump) can't crowd out signal
    from later steps or push DeepSeek into its empty-response failure mode.
    """
    parts: list[str] = []
    for step in trace.get("steps", []) or []:
        if step.get("type") == "tool_call" and step.get("output") is not None:
            parts.append(_truncate(json.dumps(step["output"]), limit))
    return "\n".join(parts)


def _tool_inputs_context(trace: dict, limit: int = TOOL_OUTPUT_CHAR_LIMIT) -> str:
    """Concatenated tool_call inputs, used to judge retrieval relevance."""
    parts: list[str] = []
    for step in trace.get("steps", []) or []:
        if step.get("type") == "tool_call" and step.get("input") is not None:
            parts.append(_truncate(json.dumps(step["input"]), limit))
    return "\n".join(parts)


def _code_blocks(text: str) -> str:
    """Extract fenced code blocks from a string. Empty if none."""
    if not text:
        return ""
    blocks = re.findall(r"```(?:\w+)?\n?(.*?)```", text, re.DOTALL)
    return "\n\n".join(b.strip() for b in blocks)


def _all_text(trace: dict) -> str:
    """All free-form text in the trace, for code metric extraction."""
    chunks: list[str] = []
    chunks.append(trace.get("output", {}).get("answer", "") or "")
    for step in trace.get("steps", []) or []:
        if step.get("content"):
            chunks.append(step["content"])
        if step.get("output") and isinstance(step["output"], dict):
            for v in step["output"].values():
                if isinstance(v, str):
                    chunks.append(v)
    return "\n".join(c for c in chunks if c)


# ---------------------------------------------------------------------------
# Prompt construction & response parsing
# ---------------------------------------------------------------------------


def _build_prompt(judge: str, metrics: list[str], fields: dict[str, str]) -> str:
    """Build a single prompt that asks the LLM to score multiple metrics."""
    field_block = "\n\n".join(
        f"<{name}>\n{value}\n</{name}>" for name, value in fields.items() if value
    )
    metric_lines = "\n".join(f"  - {m}" for m in metrics)
    return (
        f"You are EvalForge Judge {judge}. Score the following trace on these "
        f"metrics:\n{metric_lines}\n\n"
        f"Each score is a float between 0.0 and 1.0. Provide a one-sentence "
        f"reason per metric.\n\nTrace fields:\n{field_block}\n\n"
        f"Respond in this exact JSON format:\n"
        f"{{{', '.join(f'\"{m}\": {{\"score\": 0.0-1.0, \"reason\": \"...\"}}' for m in metrics)}}}"
    )


def _parse_response(text: str, metrics: list[str]) -> tuple[dict, dict]:
    """Parse a judge LLM response into (scores, reasons). Robust to code fences."""
    cleaned = text.strip()
    # Strip ```json / ``` fences if present
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    parsed = json.loads(cleaned)
    scores: dict[str, float] = {}
    reasons: dict[str, str] = {}
    for m in metrics:
        if m not in parsed:
            raise ValueError(f"judge response missing metric '{m}'")
        entry = parsed[m]
        scores[m] = float(entry["score"])
        reasons[m] = str(entry.get("reason", ""))
    return scores, reasons


async def _call_llm(
    cfg: JudgeConfig,
    router: ModelRouter,
    prompt: str,
    session: Any | None = None,
) -> str:
    """POST the prompt to the configured provider. Returns assistant text.

    ``session`` is an optional ``aiohttp.ClientSession`` so a swarm can reuse
    a single connection pool. Imported lazily so the swarm module can be
    imported without aiohttp installed (mock-only environments).

    Reliability:
      - 30s HTTP timeout per attempt
      - Up to ``JUDGE_MAX_RETRIES`` attempts (1s delay) on empty responses —
        DeepSeek returns ``""`` on overlong prompts, which previously surfaced
        as "Expecting value: line 1 column 1 (char 0)" downstream. HTTP errors
        still fail fast.
    """
    import aiohttp  # noqa: WPS433 — local import: optional in mock-only setups

    req = router.build_request(cfg, prompt)
    request_timeout = aiohttp.ClientTimeout(total=JUDGE_HTTP_TIMEOUT_SECONDS)

    async def _post(s: Any) -> str:
        for attempt in range(1, JUDGE_MAX_RETRIES + 1):
            async with s.post(
                req["url"],
                headers=req["headers"],
                json=req["body"],
                timeout=request_timeout,
            ) as r:
                if r.status >= 400:
                    raise RuntimeError(f"judge HTTP {r.status}: {await r.text()}")
                payload = await r.json()
                text = ModelRouter.parse_response(cfg, payload)
                if text and text.strip():
                    return text
            # Empty / whitespace-only response — back off and retry.
            if attempt < JUDGE_MAX_RETRIES:
                await asyncio.sleep(JUDGE_RETRY_DELAY_SECONDS)
        raise RuntimeError(
            f"judge returned empty response after {JUDGE_MAX_RETRIES} attempts "
            f"(model={cfg.model}); prompt may be too large"
        )

    if session is not None:
        return await _post(session)
    async with aiohttp.ClientSession() as new_session:
        return await _post(new_session)


# ---------------------------------------------------------------------------
# Mock-mode helpers
# ---------------------------------------------------------------------------


def _mock_result(judge_id: str, metrics: list[str], elapsed_ms: int = 1) -> JudgeResult:
    return JudgeResult(
        judge=judge_id,
        metrics=metrics,
        scores={m: MOCK_SCORES[m] for m in metrics},
        reasons={m: MOCK_REASONS[m] for m in metrics},
        elapsed_ms=elapsed_ms,
        model="mock",
    )


async def _run_judge(
    judge_id: str,
    metrics: list[str],
    fields: dict[str, str],
    *,
    router: ModelRouter | None,
    session: Any | None,
    mock: bool,
) -> JudgeResult:
    """Shared body for LLM-backed judges (A, B, C, E)."""
    if mock:
        return _mock_result(judge_id, metrics)

    assert router is not None, "router required when mock=False"
    cfg = router.resolve()
    start = time.perf_counter()
    try:
        prompt = _build_prompt(judge_id, metrics, fields)
        text = await _call_llm(cfg, router, prompt, session=session)
        scores, reasons = _parse_response(text, metrics)
        elapsed_ms = int((time.perf_counter() - start) * 1000)
        return JudgeResult(
            judge=judge_id,
            metrics=metrics,
            scores=scores,
            reasons=reasons,
            elapsed_ms=elapsed_ms,
            model=cfg.model,
        )
    except Exception as exc:  # noqa: BLE001 — surface error in result, not raise
        elapsed_ms = int((time.perf_counter() - start) * 1000)
        return JudgeResult(
            judge=judge_id,
            metrics=metrics,
            scores={m: 0.0 for m in metrics},
            reasons={m: f"judge error: {exc}" for m in metrics},
            elapsed_ms=elapsed_ms,
            model=cfg.model,
            error=str(exc),
        )


# ---------------------------------------------------------------------------
# Judge A — faithfulness + hallucination (shared tool-output context)
# ---------------------------------------------------------------------------


async def judge_a(
    trace: dict,
    *,
    router: ModelRouter | None = None,
    session: Any | None = None,
    mock: bool = False,
) -> JudgeResult:
    metrics = ["faithfulness", "hallucination"]
    fields = {
        "question": trace.get("input", {}).get("user", ""),
        "tool_outputs": _tool_output_context(trace),
        "answer": trace.get("output", {}).get("answer", ""),
    }
    return await _run_judge("A", metrics, fields, router=router, session=session, mock=mock)


# ---------------------------------------------------------------------------
# Judge B — goal_completion + g_eval (shared task context)
# ---------------------------------------------------------------------------


async def judge_b(
    trace: dict,
    *,
    router: ModelRouter | None = None,
    session: Any | None = None,
    mock: bool = False,
    rubric: str | None = None,
) -> JudgeResult:
    metrics = ["goal_completion", "g_eval"]
    fields = {
        "goal": trace.get("input", {}).get("user", ""),
        "system": trace.get("input", {}).get("system", ""),
        "answer": trace.get("output", {}).get("answer", ""),
        "rubric": rubric or "Default rubric: answer should be accurate, relevant, and complete.",
    }
    return await _run_judge("B", metrics, fields, router=router, session=session, mock=mock)


# ---------------------------------------------------------------------------
# Judge C — context_precision + answer_relevance (shared retrieval context)
# ---------------------------------------------------------------------------


async def judge_c(
    trace: dict,
    *,
    router: ModelRouter | None = None,
    session: Any | None = None,
    mock: bool = False,
) -> JudgeResult:
    metrics = ["context_precision", "answer_relevance"]
    fields = {
        "question": trace.get("input", {}).get("user", ""),
        "retrieval_queries": _tool_inputs_context(trace),
        "retrieved_context": _tool_output_context(trace),
        "answer": trace.get("output", {}).get("answer", ""),
    }
    return await _run_judge("C", metrics, fields, router=router, session=session, mock=mock)


# ---------------------------------------------------------------------------
# Judge D — tool_accuracy + code_security (deterministic, no LLM)
# ---------------------------------------------------------------------------


def _score_tool_accuracy(trace: dict) -> tuple[float, str]:
    expected = trace.get("eval_hints", {}).get("expected_tools", []) or []
    if not expected:
        return 1.0, "no expected_tools specified — trivially accurate"
    used = {
        s.get("tool")
        for s in trace.get("steps", []) or []
        if s.get("type") == "tool_call" and s.get("tool")
    }
    matched = [t for t in expected if t in used]
    rate = len(matched) / len(expected)
    return rate, f"{len(matched)}/{len(expected)} expected tools used"


# Patterns that frequently flag real security issues. Kept conservative — false
# positives are worse than missed signals here, since the LLM judges catch the
# rest.
_INSECURE_PATTERNS = [
    (re.compile(r"\beval\s*\("), "uses eval()"),
    (re.compile(r"\bexec\s*\("), "uses exec()"),
    (re.compile(r"shell\s*=\s*True"), "shell=True in subprocess"),
    (re.compile(r"verify\s*=\s*False"), "SSL verification disabled"),
    (re.compile(r"pickle\.loads?"), "unsafe pickle deserialization"),
    (re.compile(r"['\"]password['\"]\s*:\s*['\"][^'\"]{1,40}['\"]"), "hardcoded password literal"),
]


def _score_code_security(trace: dict) -> tuple[float, str]:
    code = _code_blocks(_all_text(trace))
    if not code:
        return 1.0, "no code blocks found — nothing to flag"
    findings = [msg for pat, msg in _INSECURE_PATTERNS if pat.search(code)]
    if not findings:
        return 1.0, "no insecure patterns detected"
    # Each finding costs 0.25, floored at 0.
    score = max(0.0, 1.0 - 0.25 * len(findings))
    return score, "; ".join(findings)


async def judge_d(
    trace: dict,
    *,
    router: ModelRouter | None = None,    # unused — kept for signature parity
    session: Any | None = None,
    mock: bool = False,
) -> JudgeResult:
    start = time.perf_counter()
    metrics = ["tool_accuracy", "code_security"]

    if mock:
        return _mock_result("D", metrics)

    ta_score, ta_reason = _score_tool_accuracy(trace)
    cs_score, cs_reason = _score_code_security(trace)
    elapsed_ms = int((time.perf_counter() - start) * 1000)
    return JudgeResult(
        judge="D",
        metrics=metrics,
        scores={"tool_accuracy": ta_score, "code_security": cs_score},
        reasons={"tool_accuracy": ta_reason, "code_security": cs_reason},
        elapsed_ms=elapsed_ms,
        model="deterministic",
    )


# ---------------------------------------------------------------------------
# Judge E — code_correctness + code_quality (shared code context)
# ---------------------------------------------------------------------------


async def judge_e(
    trace: dict,
    *,
    router: ModelRouter | None = None,
    session: Any | None = None,
    mock: bool = False,
) -> JudgeResult:
    metrics = ["code_correctness", "code_quality"]
    code = _code_blocks(_all_text(trace))
    code_field = code or trace.get("output", {}).get("answer", "")
    fields = {
        "task": trace.get("input", {}).get("user", ""),
        "code": _truncate(code_field, CODE_FIELD_CHAR_LIMIT),
    }
    return await _run_judge("E", metrics, fields, router=router, session=session, mock=mock)


# Index used by the orchestrator to look up judge functions by id.
JUDGES = {
    "A": judge_a,
    "B": judge_b,
    "C": judge_c,
    "D": judge_d,
    "E": judge_e,
}

JUDGE_METRICS = {
    "A": ["faithfulness", "hallucination"],
    "B": ["goal_completion", "g_eval"],
    "C": ["context_precision", "answer_relevance"],
    "D": ["tool_accuracy", "code_security"],
    "E": ["code_correctness", "code_quality"],
}


async def run_all_judges(
    trace: dict,
    *,
    router: ModelRouter | None = None,
    judges_to_run: list[str] | None = None,
    mock: bool = False,
    rubric: str | None = None,
) -> list[JudgeResult]:
    """Fan-out the selected judges concurrently via asyncio.gather.

    A shared aiohttp session is created for LLM-backed judges so they share a
    single connection pool. In mock mode we skip the session entirely.
    """
    selected = judges_to_run or ["A", "B", "C", "D", "E"]

    if mock:
        tasks = []
        for jid in selected:
            if jid == "B":
                tasks.append(judge_b(trace, mock=True, rubric=rubric))
            else:
                tasks.append(JUDGES[jid](trace, mock=True))
        return await asyncio.gather(*tasks)

    import aiohttp  # noqa: WPS433
    async with aiohttp.ClientSession() as session:
        tasks = []
        for jid in selected:
            if jid == "B":
                tasks.append(judge_b(trace, router=router, session=session, rubric=rubric))
            elif jid == "D":
                tasks.append(judge_d(trace))   # deterministic — no LLM call
            else:
                tasks.append(JUDGES[jid](trace, router=router, session=session))
        return await asyncio.gather(*tasks)
