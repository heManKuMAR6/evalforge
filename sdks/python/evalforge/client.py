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

    # 2. Compiled Rust binary relative to this file (local dev)
    local_bin = Path(__file__).parent / "../../../target/debug/evalforge"
    local_bin = local_bin.resolve()
    if local_bin.is_file():
        return str(local_bin)

    # 3. System PATH
    import shutil
    path_bin = shutil.which("evalforge")
    if path_bin:
        return path_bin

    raise RuntimeError(
        "evalforge binary not found. Set EVALFORGE_BIN, run `cargo build` in the "
        "workspace root, or install the binary to PATH."
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

    # Parse scoring result lines: e.g. "faithfulness     0.91   PASS"
    metric_results: list[MetricResult] = []
    pattern = re.compile(r"^(\w+)\s+([\d.]+)\s+(PASS|FAIL)\s*$", re.MULTILINE)
    for m in pattern.finditer(output):
        metric_results.append(
            MetricResult(
                metric=m.group(1),
                score=float(m.group(2)),
                passed=m.group(3) == "PASS",
                reason="",  # CLI doesn't surface per-metric reasons in stdout
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
