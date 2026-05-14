"""Tests for the v2.0 swarm layer.

All tests run in mock mode — no API keys or network access required.
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest

import evalforge
from evalforge.swarm import (
    METRIC_WEIGHTS,
    ModelRouter,
    SwarmResult,
    build_consensus,
    detect_disagreements,
    run_swarm,
    select_metrics_for_trace,
    weighted_score,
)
from evalforge.swarm.judges import (
    JUDGE_METRICS,
    JudgeResult,
    judge_a,
    judge_b,
    judge_c,
    judge_d,
    judge_e,
    run_all_judges,
)
from evalforge.swarm.model_router import DEFAULT_MODEL
from evalforge.ci.github import (
    build_diff_table,
    diff_results,
    load_swarm_result,
)
from evalforge.ci import github as ci_github


_WORKSPACE_ROOT = Path(__file__).parent.parent.parent.parent
SAMPLE_TRACE = _WORKSPACE_ROOT / "tests/fixtures/sample_trace.json"
SIMPLE_TRACE = _WORKSPACE_ROOT / "tests/fixtures/simple_trace.json"


def _load(p: Path) -> dict:
    return json.loads(p.read_text())


# ---------------------------------------------------------------------------
# model_router
# ---------------------------------------------------------------------------


def test_router_default_is_claude_haiku():
    r = ModelRouter(env={})
    cfg = r.resolve()
    assert cfg.provider == "anthropic"
    assert cfg.model == "claude-haiku-4-5"
    assert cfg.api_model_name == "claude-haiku-4-5-20251001"


def test_router_explicit_deepseek():
    r = ModelRouter(model="deepseek-v4-flash", env={"DEEPSEEK_API_KEY": "sk-test"})
    cfg = r.resolve()
    assert cfg.provider == "deepseek"
    assert "Authorization" in cfg.headers
    assert cfg.base_url.endswith("/v1")


def test_router_ollama_uses_local_base_url():
    r = ModelRouter(model="ollama/qwen3.5", env={})
    cfg = r.resolve()
    assert cfg.provider == "ollama"
    assert "localhost" in cfg.base_url
    assert cfg.api_model_name == "qwen3.5"


def test_router_env_var_overrides_default():
    r = ModelRouter(env={"EVALFORGE_JUDGE_MODEL": "deepseek-v4-flash",
                         "DEEPSEEK_API_KEY": "sk-test"})
    assert r.chosen_model() == "deepseek-v4-flash"


def test_router_config_file_used_when_no_env():
    r = ModelRouter(config={"judge_model": "ollama/qwen3.5"}, env={})
    assert r.chosen_model() == "ollama/qwen3.5"


def test_router_explicit_overrides_env_and_config():
    r = ModelRouter(
        model="claude-haiku-4-5",
        config={"judge_model": "ollama/qwen3.5"},
        env={"EVALFORGE_JUDGE_MODEL": "deepseek-v4-flash"},
    )
    assert r.chosen_model() == "claude-haiku-4-5"


def test_router_request_shape_anthropic():
    r = ModelRouter(model="claude-haiku-4-5", env={"ANTHROPIC_API_KEY": "key"})
    cfg = r.resolve()
    req = r.build_request(cfg, "hello")
    assert req["url"].endswith("/messages")
    assert req["body"]["messages"][0]["content"] == "hello"


def test_router_request_shape_openai_compatible():
    r = ModelRouter(model="deepseek-v4-flash", env={"DEEPSEEK_API_KEY": "k"})
    cfg = r.resolve()
    req = r.build_request(cfg, "hello")
    assert req["url"].endswith("/chat/completions")
    assert req["body"]["messages"][0]["content"] == "hello"


def test_router_parse_response_handles_both_providers():
    r = ModelRouter(env={})
    anthropic_cfg = r.resolve("claude-haiku-4-5")
    openai_cfg = r.resolve("deepseek-v4-flash")
    assert ModelRouter.parse_response(
        anthropic_cfg, {"content": [{"text": "ok"}]}
    ) == "ok"
    assert ModelRouter.parse_response(
        openai_cfg, {"choices": [{"message": {"content": "ok"}}]}
    ) == "ok"


# ---------------------------------------------------------------------------
# judges (mock mode)
# ---------------------------------------------------------------------------


def test_judge_a_mock_returns_two_metrics():
    trace = _load(SAMPLE_TRACE)
    r: JudgeResult = asyncio.run(judge_a(trace, mock=True))
    assert sorted(r.scores) == sorted(JUDGE_METRICS["A"])
    assert all(0 <= v <= 1 for v in r.scores.values())


def test_judge_b_mock_returns_two_metrics():
    trace = _load(SAMPLE_TRACE)
    r = asyncio.run(judge_b(trace, mock=True, rubric="be helpful"))
    assert sorted(r.scores) == sorted(JUDGE_METRICS["B"])


def test_judge_c_mock_returns_two_metrics():
    trace = _load(SAMPLE_TRACE)
    r = asyncio.run(judge_c(trace, mock=True))
    assert sorted(r.scores) == sorted(JUDGE_METRICS["C"])


def test_judge_d_deterministic_tool_accuracy_full_match():
    """sample_trace has both expected tools — tool_accuracy = 1.0."""
    trace = _load(SAMPLE_TRACE)
    r = asyncio.run(judge_d(trace, mock=False))
    assert r.scores["tool_accuracy"] == 1.0
    assert r.model == "deterministic"


def test_judge_d_security_flags_eval():
    trace = {
        "trace_id": "t",
        "steps": [],
        "input": {"user": "", "system": ""},
        "output": {"answer": "Here is code:\n```python\neval('1+1')\n```"},
        "eval_hints": {"expected_tools": [], "expected_answer": None, "context_documents": []},
    }
    r = asyncio.run(judge_d(trace, mock=False))
    assert r.scores["code_security"] < 1.0
    assert "eval" in r.reasons["code_security"]


def test_judge_d_security_clean_code_passes():
    trace = {
        "trace_id": "t",
        "steps": [],
        "input": {"user": "", "system": ""},
        "output": {"answer": "```python\nx = 1 + 2\n```"},
        "eval_hints": {"expected_tools": [], "expected_answer": None, "context_documents": []},
    }
    r = asyncio.run(judge_d(trace, mock=False))
    assert r.scores["code_security"] == 1.0


def test_judge_e_mock_returns_two_metrics():
    trace = _load(SAMPLE_TRACE)
    r = asyncio.run(judge_e(trace, mock=True))
    assert sorted(r.scores) == sorted(JUDGE_METRICS["E"])


def test_run_all_judges_mock_fans_out_five():
    trace = _load(SAMPLE_TRACE)
    results = asyncio.run(run_all_judges(trace, mock=True))
    assert len(results) == 5
    assert {r.judge for r in results} == {"A", "B", "C", "D", "E"}


# ---------------------------------------------------------------------------
# orchestrator + select_metrics_for_trace
# ---------------------------------------------------------------------------


def test_select_metrics_with_tool_calls_includes_a_c_d():
    trace = _load(SAMPLE_TRACE)
    judges = select_metrics_for_trace(trace)
    assert "A" in judges and "C" in judges and "D" in judges


def test_select_metrics_no_tools_only_b():
    trace = _load(SIMPLE_TRACE)
    judges = select_metrics_for_trace(trace)
    assert judges == ["B"]


def test_select_metrics_with_code_includes_d_e():
    trace = {
        "trace_id": "t",
        "steps": [],
        "input": {"user": "write code", "system": ""},
        "output": {"answer": "```python\nprint(1)\n```"},
        "eval_hints": {"expected_tools": [], "expected_answer": None, "context_documents": []},
    }
    judges = select_metrics_for_trace(trace)
    assert "D" in judges and "E" in judges


def test_run_swarm_mock_passes_for_sample_trace():
    result = run_swarm(SAMPLE_TRACE, mock=True)
    assert isinstance(result, SwarmResult)
    assert result.passed is True
    assert result.weighted_score > 0.7
    assert result.raw["evalforge_version"] == "2.0"


def test_run_swarm_mock_total_elapsed_small():
    """Mock swarm should complete in milliseconds, not seconds."""
    result = run_swarm(SAMPLE_TRACE, mock=True)
    assert result.total_elapsed_ms < 2000


def test_run_swarm_mock_simple_trace_only_runs_judge_b():
    result = run_swarm(SIMPLE_TRACE, mock=True)
    # Only judge B ran, so only goal_completion + g_eval should be scored.
    assert set(result.scores) == {"goal_completion", "g_eval"}


def test_run_swarm_persists_raw_schema():
    result = run_swarm(SAMPLE_TRACE, mock=True)
    raw = result.raw
    assert raw["evalforge_version"] == "2.0"
    assert "swarm_result" in raw
    sr = raw["swarm_result"]
    for key in ("overall", "scores", "weighted_score", "threshold",
                "anomalies", "human_review_flags", "failure_reason",
                "judge_timings_ms"):
        assert key in sr, f"missing key {key}"


# ---------------------------------------------------------------------------
# consensus
# ---------------------------------------------------------------------------


def test_weighted_score_uses_metric_weights():
    # hallucination + faithfulness; hallucination weighted 1.5x
    scores = {"faithfulness": 0.6, "hallucination": 1.0}
    expected = (1.0 * 0.6 + 1.5 * 1.0) / (1.0 + 1.5)
    assert abs(weighted_score(scores) - expected) < 1e-9


def test_weighted_score_no_weights_is_arithmetic_mean():
    scores = {"goal_completion": 0.8, "g_eval": 0.9}
    assert abs(weighted_score(scores) - 0.85) < 1e-9


def test_detect_disagreements_flags_wide_gap():
    jr = JudgeResult(
        judge="A",
        metrics=["faithfulness", "hallucination"],
        scores={"faithfulness": 0.95, "hallucination": 0.20},
        reasons={"faithfulness": "", "hallucination": ""},
    )
    flags = detect_disagreements([jr], consensus_threshold=0.3)
    assert len(flags) == 1
    assert flags[0]["judge"] == "A"
    assert flags[0]["delta"] > 0.3


def test_detect_disagreements_ignores_close_scores():
    jr = JudgeResult(
        judge="A",
        metrics=["faithfulness", "hallucination"],
        scores={"faithfulness": 0.92, "hallucination": 0.90},
        reasons={"faithfulness": "", "hallucination": ""},
    )
    assert detect_disagreements([jr]) == []


def test_build_consensus_pass_path():
    jr_a = JudgeResult(
        judge="A",
        metrics=["faithfulness", "hallucination"],
        scores={"faithfulness": 0.9, "hallucination": 0.95},
        reasons={"faithfulness": "ok", "hallucination": "ok"},
        elapsed_ms=120,
    )
    out = build_consensus([jr_a], trace_id="t1", threshold=0.7)
    assert out["swarm_result"]["overall"] == "PASS"
    assert out["swarm_result"]["failure_reason"] is None
    assert out["swarm_result"]["judge_timings_ms"]["A"] == 120


def test_build_consensus_fail_path_provides_failure_reason():
    jr_a = JudgeResult(
        judge="A",
        metrics=["faithfulness", "hallucination"],
        scores={"faithfulness": 0.4, "hallucination": 0.95},
        reasons={"faithfulness": "answer adds facts not in context", "hallucination": "ok"},
    )
    out = build_consensus([jr_a], trace_id="t1", threshold=0.7)
    assert out["swarm_result"]["overall"] == "FAIL"
    assert "faithfulness" in out["swarm_result"]["failure_reason"]
    assert "0.40" in out["swarm_result"]["failure_reason"]


def test_build_consensus_anomaly_high_overall_one_low_metric():
    jr_a = JudgeResult(
        judge="A",
        metrics=["faithfulness", "hallucination"],
        scores={"faithfulness": 0.99, "hallucination": 1.0},
        reasons={"faithfulness": "", "hallucination": ""},
    )
    jr_b = JudgeResult(
        judge="B",
        metrics=["goal_completion", "g_eval"],
        scores={"goal_completion": 1.0, "g_eval": 0.99},
        reasons={"goal_completion": "", "g_eval": ""},
    )
    jr_d = JudgeResult(
        judge="D",
        metrics=["tool_accuracy", "code_security"],
        scores={"tool_accuracy": 1.0, "code_security": 0.20},
        reasons={"tool_accuracy": "", "code_security": "eval() used"},
    )
    out = build_consensus([jr_a, jr_b, jr_d], trace_id="t1", threshold=0.7)
    sr = out["swarm_result"]
    # Weighted score is high because most metrics are 1.0, but code_security low
    assert sr["weighted_score"] > 0.7
    # However the trace still FAILs because code_security is below threshold
    assert sr["overall"] == "FAIL"
    # And the low-metric anomaly is surfaced even though overall is high
    metrics_flagged = [a["metric"] for a in sr["anomalies"]]
    assert "code_security" in metrics_flagged


def test_metric_weights_constants():
    assert METRIC_WEIGHTS["hallucination"] == 1.5
    assert METRIC_WEIGHTS["code_security"] == 1.5


def test_build_consensus_judge_error_fails_trace():
    jr_a = JudgeResult(
        judge="A",
        metrics=["faithfulness", "hallucination"],
        scores={"faithfulness": 0.0, "hallucination": 0.0},
        reasons={"faithfulness": "err", "hallucination": "err"},
        error="connection refused",
    )
    out = build_consensus([jr_a], trace_id="t1")
    assert out["swarm_result"]["overall"] == "FAIL"


# ---------------------------------------------------------------------------
# ci/github diff
# ---------------------------------------------------------------------------


def _result_doc(scores, weighted=None, trace_id="t"):
    return {
        "evalforge_version": "2.0",
        "trace_id": trace_id,
        "swarm_result": {
            "overall": "PASS",
            "scores": scores,
            "weighted_score": weighted if weighted is not None else sum(scores.values()) / len(scores),
            "threshold": 0.7,
            "anomalies": [],
            "human_review_flags": [],
            "failure_reason": None,
            "judge_timings_ms": {},
        },
    }


def test_diff_results_computes_per_metric_deltas():
    before = _result_doc({"faithfulness": 0.72, "hallucination": 0.95})
    after = _result_doc({"faithfulness": 0.91, "hallucination": 0.65})
    d = diff_results(before, after)
    by_metric = {r["metric"]: r for r in d["rows"]}
    assert abs(by_metric["faithfulness"]["delta"] - 0.19) < 1e-6
    assert by_metric["faithfulness"]["symbol"] == "✅"
    assert abs(by_metric["hallucination"]["delta"] + 0.30) < 1e-6
    assert by_metric["hallucination"]["symbol"] == "⚠"


def test_diff_table_renders_expected_columns():
    before = _result_doc({"faithfulness": 0.80})
    after = _result_doc({"faithfulness": 0.92})
    table = build_diff_table(before, after)
    assert "| Metric | Before | After | Δ |" in table
    assert "faithfulness" in table
    assert "**Overall**" in table


def test_load_swarm_result_handles_single_file(tmp_path):
    doc = _result_doc({"faithfulness": 0.9})
    p = tmp_path / "r.json"
    p.write_text(json.dumps(doc))
    loaded = load_swarm_result(p)
    assert loaded["trace_id"] == "t"


def test_load_swarm_result_handles_directory(tmp_path):
    d1 = _result_doc({"faithfulness": 0.8}, trace_id="a")
    d2 = _result_doc({"faithfulness": 1.0}, trace_id="b")
    (tmp_path / "a.json").write_text(json.dumps(d1))
    (tmp_path / "b.json").write_text(json.dumps(d2))
    merged = load_swarm_result(tmp_path)
    # average of 0.8 and 1.0 = 0.9
    assert abs(merged["swarm_result"]["scores"]["faithfulness"] - 0.9) < 1e-9


def test_diff_table_handles_missing_metric():
    before = _result_doc({"faithfulness": 0.8})
    after = _result_doc({"faithfulness": 0.85, "g_eval": 0.9})
    table = build_diff_table(before, after)
    assert "g_eval" in table
    # Missing-side cells should render as em-dash
    assert "—" in table


def test_post_pr_comment_requires_token(monkeypatch):
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    with pytest.raises(RuntimeError, match="GITHUB_TOKEN"):
        ci_github.post_pr_comment("body", repo="o/r", pr_number=1, token=None)


# ---------------------------------------------------------------------------
# Backward compatibility — v1.0 API surface still works
# ---------------------------------------------------------------------------


def test_v1_run_still_works_after_v2_import():
    # Importing v2 swarm shouldn't break the v1 client.run path
    result = evalforge.run(str(SAMPLE_TRACE), metrics=["faithfulness"], mock=True)
    assert result.passed is True
    assert result.metrics[0].score == 0.91


def test_v1_public_api_exports_unchanged():
    for name in ("run", "demo", "EvalResult", "MetricResult",
                 "analyze_run_trend", "adapters"):
        assert hasattr(evalforge, name), f"missing v1.0 public attr: {name}"


# ---------------------------------------------------------------------------
# Reliability: truncation + retry + timeout (fixes for DeepSeek empty-response)
# ---------------------------------------------------------------------------


def test_tool_output_context_truncates_long_outputs():
    from evalforge.swarm.judges import (
        _tool_output_context,
        TOOL_OUTPUT_CHAR_LIMIT,
    )
    big = "x" * 5000
    trace = {
        "steps": [{
            "step_id": 1,
            "type": "tool_call",
            "tool": "search",
            "input": {"q": "hi"},
            "output": {"result": big},
        }],
    }
    ctx = _tool_output_context(trace)
    # JSON wrapping adds a few chars; ellipsis marker adds ~40. Generous bound.
    assert len(ctx) <= TOOL_OUTPUT_CHAR_LIMIT + 80
    assert "truncated" in ctx


def test_tool_inputs_context_truncates_long_inputs():
    from evalforge.swarm.judges import _tool_inputs_context, TOOL_OUTPUT_CHAR_LIMIT
    big = "y" * 5000
    trace = {
        "steps": [{
            "step_id": 1,
            "type": "tool_call",
            "tool": "search",
            "input": {"q": big},
        }],
    }
    ctx = _tool_inputs_context(trace)
    assert len(ctx) <= TOOL_OUTPUT_CHAR_LIMIT + 80
    assert "truncated" in ctx


def test_short_outputs_are_not_truncated():
    from evalforge.swarm.judges import _tool_output_context
    trace = {
        "steps": [{
            "step_id": 1,
            "type": "tool_call",
            "tool": "search",
            "output": {"r": "short"},
        }],
    }
    ctx = _tool_output_context(trace)
    assert "truncated" not in ctx
    assert "short" in ctx


def test_truncate_helper_appends_marker_only_when_cut():
    from evalforge.swarm.judges import _truncate
    assert _truncate("abc", 10) == "abc"
    out = _truncate("a" * 100, 10)
    assert out.startswith("aaaaaaaaaa")
    assert "truncated" in out


def test_judge_e_code_field_is_capped():
    """A massive code block in the trace must not blow past CODE_FIELD_CHAR_LIMIT."""
    from evalforge.swarm.judges import CODE_FIELD_CHAR_LIMIT
    # We exercise this indirectly: build a trace with a huge code block, run
    # judge_e in mock mode, then verify the live (non-mock) field-builder
    # produces a capped value.
    big_code = "print(1)\n" * 10000
    trace = {
        "trace_id": "t",
        "steps": [],
        "input": {"user": "review this", "system": ""},
        "output": {"answer": f"```python\n{big_code}```"},
        "eval_hints": {"expected_tools": [], "expected_answer": None, "context_documents": []},
    }
    # In mock mode the field isn't actually built — instead inspect the helper.
    from evalforge.swarm.judges import _code_blocks, _all_text, _truncate
    code = _code_blocks(_all_text(trace))
    capped = _truncate(code, CODE_FIELD_CHAR_LIMIT)
    assert len(capped) <= CODE_FIELD_CHAR_LIMIT + 80


def test_retry_constants_set_reasonably():
    from evalforge.swarm.judges import (
        JUDGE_HTTP_TIMEOUT_SECONDS,
        JUDGE_MAX_RETRIES,
        JUDGE_RETRY_DELAY_SECONDS,
    )
    assert JUDGE_MAX_RETRIES == 3
    assert JUDGE_RETRY_DELAY_SECONDS == 1.0
    assert JUDGE_HTTP_TIMEOUT_SECONDS == 15


class _FakeResponse:
    """Async-context-manager response stand-in for aiohttp."""

    def __init__(self, status: int, payload: dict | str):
        self.status = status
        self._payload = payload

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def json(self):
        return self._payload

    async def text(self):
        return str(self._payload)


class _FakeSession:
    """Records calls; returns scripted responses. Mimics aiohttp.ClientSession."""

    def __init__(self, responses: list[_FakeResponse]):
        self._responses = list(responses)
        self.calls: list[dict] = []

    def post(self, url, headers=None, json=None, timeout=None):
        self.calls.append({"url": url, "json": json, "timeout": timeout})
        return self._responses.pop(0)


def test_call_llm_retries_on_empty_then_succeeds():
    """Two empty responses followed by a real one → returns the real one."""
    from evalforge.swarm import judges as J

    cfg = ModelRouter(env={}).resolve("claude-haiku-4-5")
    router = ModelRouter(env={})
    session = _FakeSession([
        _FakeResponse(200, {"content": [{"text": ""}]}),
        _FakeResponse(200, {"content": [{"text": "   "}]}),
        _FakeResponse(200, {"content": [{"text": '{"faithfulness": {"score": 0.9, "reason": "ok"}}'}]}),
    ])

    # Make asyncio.sleep instant so the test doesn't actually wait 2s.
    orig_sleep = asyncio.sleep

    async def fast_sleep(_):
        return None

    asyncio.sleep = fast_sleep
    try:
        result = asyncio.run(J._call_llm(cfg, router, "prompt", session=session))
    finally:
        asyncio.sleep = orig_sleep

    assert "faithfulness" in result
    assert len(session.calls) == 3


def test_call_llm_raises_after_max_retries_on_persistent_empty():
    from evalforge.swarm import judges as J

    cfg = ModelRouter(env={}).resolve("claude-haiku-4-5")
    router = ModelRouter(env={})
    session = _FakeSession([
        _FakeResponse(200, {"content": [{"text": ""}]}),
        _FakeResponse(200, {"content": [{"text": ""}]}),
        _FakeResponse(200, {"content": [{"text": ""}]}),
    ])

    orig_sleep = asyncio.sleep

    async def fast_sleep(_):
        return None

    asyncio.sleep = fast_sleep
    try:
        with pytest.raises(RuntimeError, match="empty response"):
            asyncio.run(J._call_llm(cfg, router, "prompt", session=session))
    finally:
        asyncio.sleep = orig_sleep

    assert len(session.calls) == 3  # exactly JUDGE_MAX_RETRIES


def test_call_llm_passes_timeout_to_aiohttp():
    """Every POST should carry a 30s ClientTimeout, not run unbounded."""
    from evalforge.swarm import judges as J

    cfg = ModelRouter(env={}).resolve("claude-haiku-4-5")
    router = ModelRouter(env={})
    session = _FakeSession([
        _FakeResponse(200, {"content": [{"text": "{}"}]}),
    ])
    asyncio.run(J._call_llm(cfg, router, "prompt", session=session))
    sent_timeout = session.calls[0]["timeout"]
    assert sent_timeout is not None
    # aiohttp.ClientTimeout has a `total` attribute
    assert sent_timeout.total == J.JUDGE_HTTP_TIMEOUT_SECONDS


def test_judge_a_splits_into_two_parallel_calls():
    """Non-mock Judge A should make exactly 2 LLM calls (one per metric)."""
    from evalforge.swarm import judges as J

    session = _FakeSession([
        _FakeResponse(200, {"content": [{"text": '{"faithfulness": {"score": 0.88, "reason": "ok"}}'}]}),
        _FakeResponse(200, {"content": [{"text": '{"hallucination": {"score": 0.94, "reason": "ok"}}'}]}),
    ])
    router = ModelRouter(env={"ANTHROPIC_API_KEY": "k"})
    trace = _load(SAMPLE_TRACE)
    result = asyncio.run(J.judge_a(trace, router=router, session=session, mock=False))

    assert len(session.calls) == 2, "judge A must split into 2 single-metric calls"
    assert result.scores == {"faithfulness": 0.88, "hallucination": 0.94}
    assert result.error is None

    # Each prompt should mention only one metric, not both
    prompts = [c["json"]["messages"][0]["content"] for c in session.calls]
    assert sum("faithfulness" in p for p in prompts) >= 1
    assert sum("hallucination" in p for p in prompts) >= 1
    # No single prompt should ask for both metrics together (the whole point of the split)
    for p in prompts:
        assert not ("faithfulness" in p and "hallucination" in p), \
            "split should produce focused single-metric prompts"


def test_judge_a_partial_failure_surfaces_in_error():
    """If one of the two parallel calls fails, the other's score is still recorded."""
    from evalforge.swarm import judges as J

    session = _FakeSession([
        _FakeResponse(200, {"content": [{"text": '{"faithfulness": {"score": 0.9, "reason": "ok"}}'}]}),
        _FakeResponse(500, "downstream error"),
    ])
    router = ModelRouter(env={"ANTHROPIC_API_KEY": "k"})
    trace = _load(SAMPLE_TRACE)
    result = asyncio.run(J.judge_a(trace, router=router, session=session, mock=False))

    # Faithfulness still scored from the successful call
    assert result.scores["faithfulness"] == 0.9
    # Hallucination defaulted to 0 because its call failed
    assert result.scores["hallucination"] == 0.0
    assert result.error is not None
    assert "500" in result.error


def test_call_llm_does_not_retry_on_http_error():
    """A 500 should fail fast — only empty responses trigger retries."""
    from evalforge.swarm import judges as J

    cfg = ModelRouter(env={}).resolve("claude-haiku-4-5")
    router = ModelRouter(env={})
    session = _FakeSession([
        _FakeResponse(500, "internal server error"),
    ])
    with pytest.raises(RuntimeError, match="HTTP 500"):
        asyncio.run(J._call_llm(cfg, router, "prompt", session=session))
    assert len(session.calls) == 1
