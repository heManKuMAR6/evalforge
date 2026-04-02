from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class MetricResult:
    metric: str
    score: float
    passed: bool
    reason: str


@dataclass
class EvalResult:
    trace_id: str
    framework: str
    metrics: list[MetricResult] = field(default_factory=list)
    passed: bool = False


def _find_binary() -> str:
    # 1. Explicit override via environment variable
    env_bin = os.environ.get("EVALFORGE_BIN")
    if env_bin and Path(env_bin).is_file():
        return env_bin

    # 2. Walk up from client.py looking for target/debug/evalforge (up to 5 levels)
    current = Path(__file__).resolve().parent
    for _ in range(5):
        candidate = current / "target" / "debug" / "evalforge"
        if candidate.exists():
            return str(candidate)
        current = current.parent

    # 3. System PATH
    import shutil
    path_bin = shutil.which("evalforge")
    if path_bin:
        return path_bin

    raise RuntimeError(
        f"EvalForge binary not found. "
        f"Searched up to 5 parent dirs from {Path(__file__).resolve()}. "
        f"Set EVALFORGE_BIN env var to the binary path, or run 'cargo build' first."
    )


def run(
    trace: str,
    metrics: list[str],
    threshold: float = 0.7,
    mock: bool = False,
    api_key: str | None = None,
) -> EvalResult:
    binary = _find_binary()

    cmd = [
        binary,
        "run",
        "--trace", trace,
        "--metrics", ",".join(metrics),
        "--threshold", str(threshold),
    ]
    if mock:
        cmd.append("--mock")

    env = os.environ.copy()
    if api_key is not None:
        env["ANTHROPIC_API_KEY"] = api_key

    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)

    if proc.returncode not in (0, 1):
        raise RuntimeError(
            f"evalforge exited with code {proc.returncode}.\n"
            f"stdout: {proc.stdout}\nstderr: {proc.stderr}"
        )

    output = proc.stdout

    # Parse trace summary fields
    trace_id = _extract_field(output, "Trace ID")
    framework = _extract_field(output, "Framework")

    # Parse scoring result lines and the Reason: line that follows each.
    # e.g.:
    #   faithfulness     0.91   PASS
    #   Reason: Mock score — skipping live API call
    metric_results: list[MetricResult] = []
    pattern = re.compile(
        r"^(\w+)\s+([\d.]+)\s+(PASS|FAIL)\s*\nReason:\s*(.+)$",
        re.MULTILINE,
    )
    for m in pattern.finditer(output):
        metric_results.append(
            MetricResult(
                metric=m.group(1),
                score=float(m.group(2)),
                passed=m.group(3) == "PASS",
                reason=m.group(4).strip(),
            )
        )

    overall_passed = proc.returncode == 0 and all(r.passed for r in metric_results)

    return EvalResult(
        trace_id=trace_id,
        framework=framework,
        metrics=metric_results,
        passed=overall_passed,
    )


def _extract_field(output: str, label: str) -> str:
    pattern = re.compile(rf"^{re.escape(label)}:\s+(.+)$", re.MULTILINE)
    m = pattern.search(output)
    return m.group(1).strip() if m else ""
