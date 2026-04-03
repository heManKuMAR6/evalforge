from dataclasses import dataclass, field
from pathlib import Path
import json
import statistics


@dataclass
class RunPoint:
    run_id: str
    timestamp: str
    metric: str
    score: float
    passed: bool


@dataclass
class MetricTrend:
    metric: str
    slope: float
    window_size: int
    regression_detected: bool
    scores: list[float]
    direction: str  # "improving", "stable", "degrading"


@dataclass
class RunTrendReport:
    agent_id: str
    window: int
    trends: list[MetricTrend] = field(default_factory=list)
    any_regression: bool = False

    def summary(self) -> str:
        lines = [
            f"RunTrendReport — {self.agent_id}",
            f"Window: {self.window} runs",
            "─" * 40,
        ]
        for t in self.trends:
            icon = "↑" if t.direction == "improving" else "↓" if t.direction == "degrading" else "→"
            reg = " ⚠ REGRESSION" if t.regression_detected else ""
            lines.append(
                f"{t.metric:<20} slope={t.slope:+.4f}  {icon}{reg}"
            )
            lines.append(f"  scores: {[round(s, 2) for s in t.scores]}")
        lines.append("─" * 40)
        lines.append(
            "Overall: REGRESSION DETECTED" if self.any_regression else "Overall: STABLE"
        )
        return "\n".join(lines)


def analyze_run_trend(
    history_dir: str,
    metrics: list[str],
    window: int = 10,
    regression_threshold: float = -0.02,
) -> RunTrendReport:
    """
    Analyze trend across sequential eval run JSON outputs.

    history_dir should contain JSON files saved by:
        evalforge run --trace ... --output history/run_001.json

    Each file must have:
        {"metrics": [{"metric": "faithfulness", "score": 0.91, ...}]}
    """
    paths = sorted(Path(history_dir).glob("*.json"))

    if len(paths) < 2:
        raise ValueError(
            f"Need at least 2 run files in {history_dir}. "
            f"Found {len(paths)}. "
            f"Run evalforge with --output to save results."
        )

    metric_scores: dict[str, list[float]] = {m: [] for m in metrics}

    for path in paths[-window:]:
        try:
            data = json.loads(path.read_text())
            for mr in data.get("metrics", []):
                if mr["metric"] in metric_scores:
                    metric_scores[mr["metric"]].append(float(mr["score"]))
        except (json.JSONDecodeError, KeyError):
            continue

    trends = []
    for metric, scores in metric_scores.items():
        if len(scores) < 2:
            continue

        xs = list(range(len(scores)))
        slope, _ = statistics.linear_regression(xs, scores)
        slope = round(slope, 4)

        if slope > 0.01:
            direction = "improving"
        elif slope < -0.01:
            direction = "degrading"
        else:
            direction = "stable"

        trends.append(MetricTrend(
            metric=metric,
            slope=slope,
            window_size=len(scores),
            regression_detected=slope < regression_threshold,
            scores=scores,
            direction=direction,
        ))

    return RunTrendReport(
        agent_id=history_dir,
        window=window,
        trends=trends,
        any_regression=any(t.regression_detected for t in trends),
    )
