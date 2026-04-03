from pathlib import Path

import evalforge
from evalforge.client import EvalResult, MetricResult


# Resolve fixture paths relative to this file so tests work regardless of
# which directory pytest is invoked from.
_WORKSPACE_ROOT = Path(__file__).parent.parent.parent.parent
SAMPLE_TRACE = str(_WORKSPACE_ROOT / "tests/fixtures/sample_trace.json")
SIMPLE_TRACE = str(_WORKSPACE_ROOT / "tests/fixtures/simple_trace.json")


def test_mock_run():
    result = evalforge.run(SAMPLE_TRACE, metrics=["faithfulness"], mock=True)
    assert result.passed is True
    assert len(result.metrics) == 1
    assert result.metrics[0].score == 0.91


def test_simple_trace_mock():
    result = evalforge.run(SIMPLE_TRACE, metrics=["faithfulness"], mock=True)
    assert result.passed is True
    assert len(result.metrics) == 1
    assert result.metrics[0].score == 0.91


def test_threshold_boundary():
    mr = MetricResult(metric="faithfulness", score=0.7, passed=True, reason="at threshold")
    assert mr.passed is True


def test_metric_result_fields():
    mr = MetricResult(metric="faithfulness", score=0.85, passed=True, reason="looks good")
    assert mr.metric == "faithfulness"
    assert mr.score == 0.85
    assert mr.passed is True
    assert mr.reason == "looks good"

    er = EvalResult(
        trace_id="trace-001",
        framework="langchain",
        metrics=[mr],
        passed=True,
    )
    assert er.trace_id == "trace-001"
    assert er.framework == "langchain"
    assert er.passed is True
    assert er.metrics[0].metric == "faithfulness"


def test_run_trend_stable():
    """Stable scores produce no regression."""
    import tempfile, json
    from evalforge.trend import analyze_run_trend

    with tempfile.TemporaryDirectory() as tmpdir:
        for i, score in enumerate([0.91, 0.90, 0.92, 0.91, 0.90]):
            data = {
                "trace_id": f"run-{i}",
                "metrics": [{"metric": "faithfulness", "score": score, "passed": True}]
            }
            Path(tmpdir, f"run_{i:03d}.json").write_text(json.dumps(data))

        report = analyze_run_trend(tmpdir, metrics=["faithfulness"])
        assert report.any_regression is False
        assert report.trends[0].direction == "stable"


def test_run_trend_regression():
    """Degrading scores trigger regression detection."""
    import tempfile, json
    from evalforge.trend import analyze_run_trend

    with tempfile.TemporaryDirectory() as tmpdir:
        for i, score in enumerate([0.91, 0.85, 0.79, 0.73, 0.67]):
            data = {
                "trace_id": f"run-{i}",
                "metrics": [{"metric": "faithfulness", "score": score, "passed": score >= 0.7}]
            }
            Path(tmpdir, f"run_{i:03d}.json").write_text(json.dumps(data))

        report = analyze_run_trend(tmpdir, metrics=["faithfulness"])
        assert report.any_regression is True
        assert report.trends[0].direction == "degrading"
        assert report.trends[0].slope < -0.02


def test_run_trend_requires_two_files():
    """Raises ValueError with fewer than 2 run files."""
    import tempfile, json
    from evalforge.trend import analyze_run_trend

    with tempfile.TemporaryDirectory() as tmpdir:
        Path(tmpdir, "run_001.json").write_text(json.dumps({
            "metrics": [{"metric": "faithfulness", "score": 0.91, "passed": True}]
        }))
        try:
            analyze_run_trend(tmpdir, metrics=["faithfulness"])
            assert False, "Should have raised ValueError"
        except ValueError:
            pass
