"""GitHub PR-comment integration for the swarm diff command.

Builds a before/after score-diff table and posts it as a PR comment. The table
matches the format specified in the v2.0 brief:

    | Metric            | Before | After | Δ      |
    |-------------------|--------|-------|--------|
    | faithfulness      | 0.72   | 0.91  | +0.19 ✅|
    | hallucination     | 0.95   | 0.65  | -0.30 ⚠ |
    | ...               | ...    | ...   | ...    |
    | **Overall**       | 0.81   | 0.83  | +0.02 →|

Network I/O is gated behind an explicit ``token`` argument, so unit tests can
exercise table-building without making real HTTP calls.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


def load_swarm_result(path: str | Path) -> dict:
    """Load a SwarmResult JSON file. Tolerates either a single trace's JSON or
    a directory of them (in which case results are merged by averaging)."""
    p = Path(path)
    if p.is_dir():
        files = sorted(p.glob("*.json"))
        if not files:
            raise FileNotFoundError(f"no JSON files in {p}")
        return _merge_directory(files)
    return json.loads(p.read_text())


def _merge_directory(files: list[Path]) -> dict:
    """Average scores across multiple SwarmResult files in a directory."""
    sums: dict[str, float] = {}
    counts: dict[str, int] = {}
    weighted_sum = 0.0
    weighted_count = 0
    threshold = 0.7
    for f in files:
        data = json.loads(f.read_text())
        sr = data.get("swarm_result", data)
        for metric, score in sr.get("scores", {}).items():
            sums[metric] = sums.get(metric, 0.0) + score
            counts[metric] = counts.get(metric, 0) + 1
        if "weighted_score" in sr:
            weighted_sum += sr["weighted_score"]
            weighted_count += 1
        threshold = sr.get("threshold", threshold)
    averaged = {m: sums[m] / counts[m] for m in sums}
    return {
        "evalforge_version": "2.0",
        "trace_id": f"<aggregate of {len(files)} traces>",
        "swarm_result": {
            "overall": "PASS" if all(s >= threshold for s in averaged.values()) else "FAIL",
            "scores": averaged,
            "weighted_score": weighted_sum / weighted_count if weighted_count else 0.0,
            "threshold": threshold,
            "anomalies": [],
            "human_review_flags": [],
            "failure_reason": None,
            "judge_timings_ms": {},
        },
    }


def _delta_symbol(delta: float) -> str:
    if delta > 0.05:
        return "✅"
    if delta < -0.05:
        return "⚠"
    return "→"


def diff_results(before: dict, after: dict) -> dict:
    """Compute per-metric and overall deltas between two swarm results."""
    b_sr = before.get("swarm_result", before)
    a_sr = after.get("swarm_result", after)
    b_scores = b_sr.get("scores", {})
    a_scores = a_sr.get("scores", {})

    metrics = sorted(set(b_scores) | set(a_scores))
    rows: list[dict] = []
    for m in metrics:
        b = b_scores.get(m)
        a = a_scores.get(m)
        if b is None or a is None:
            delta = None
            symbol = "—"
        else:
            delta = round(a - b, 4)
            symbol = _delta_symbol(delta)
        rows.append({
            "metric": m,
            "before": b,
            "after": a,
            "delta": delta,
            "symbol": symbol,
        })

    b_overall = b_sr.get("weighted_score", 0.0)
    a_overall = a_sr.get("weighted_score", 0.0)
    overall_delta = round(a_overall - b_overall, 4)
    return {
        "rows": rows,
        "overall": {
            "before": b_overall,
            "after": a_overall,
            "delta": overall_delta,
            "symbol": _delta_symbol(overall_delta),
        },
    }


def build_diff_table(before: dict, after: dict, title: str = "EvalForge Swarm — PR Diff") -> str:
    """Build the markdown table posted to PRs."""
    d = diff_results(before, after)
    lines = [
        f"## {title}",
        "",
        "| Metric | Before | After | Δ |",
        "|--------|--------|-------|---|",
    ]
    for row in d["rows"]:
        before = "—" if row["before"] is None else f"{row['before']:.2f}"
        after = "—" if row["after"] is None else f"{row['after']:.2f}"
        if row["delta"] is None:
            delta = "—"
        else:
            sign = "+" if row["delta"] >= 0 else ""
            delta = f"{sign}{row['delta']:.2f} {row['symbol']}"
        lines.append(f"| {row['metric']} | {before} | {after} | {delta} |")

    o = d["overall"]
    sign = "+" if o["delta"] >= 0 else ""
    lines.append(
        f"| **Overall** | {o['before']:.2f} | {o['after']:.2f} | "
        f"{sign}{o['delta']:.2f} {o['symbol']} |"
    )
    return "\n".join(lines)


def post_pr_comment(
    body: str,
    *,
    repo: str | None = None,
    pr_number: int | str | None = None,
    token: str | None = None,
) -> dict:
    """POST a comment to the GitHub PR. Returns the parsed response body.

    Parameter resolution:
      - ``repo``       defaults to ``GITHUB_REPOSITORY``
      - ``pr_number``  defaults to the issue number parsed from ``GITHUB_REF``
                       or ``GITHUB_PR_NUMBER``
      - ``token``      defaults to ``GITHUB_TOKEN``

    Raises ``RuntimeError`` if any of the three can't be resolved.
    """
    repo = repo or os.environ.get("GITHUB_REPOSITORY")
    pr_number = pr_number or os.environ.get("GITHUB_PR_NUMBER") or _pr_from_ref()
    token = token or os.environ.get("GITHUB_TOKEN")

    if not repo:
        raise RuntimeError("GITHUB_REPOSITORY not set and --repo not provided")
    if not pr_number:
        raise RuntimeError("PR number not set — pass --pr or set GITHUB_PR_NUMBER")
    if not token:
        raise RuntimeError("GITHUB_TOKEN not set")

    url = f"https://api.github.com/repos/{repo}/issues/{pr_number}/comments"
    payload = json.dumps({"body": body}).encode()
    req = urllib.request.Request(
        url,
        data=payload,
        method="POST",
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "Content-Type": "application/json",
            "User-Agent": "evalforge-swarm",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"GitHub API {exc.code}: {exc.read().decode()}") from exc


def _pr_from_ref() -> str | None:
    """Try to extract a PR number from ``GITHUB_REF`` (refs/pull/123/merge)."""
    ref = os.environ.get("GITHUB_REF", "")
    parts = ref.split("/")
    if len(parts) >= 3 and parts[1] == "pull":
        return parts[2]
    return None
