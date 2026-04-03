from __future__ import annotations

import os
import re
import subprocess
import sys
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
    # 1. Explicit env var
    if "EVALFORGE_BIN" in os.environ:
        return os.environ["EVALFORGE_BIN"]

    # 2. Bundled inside pip package (maturin puts it in scripts/)
    scripts_dir = Path(sys.prefix) / "bin" / "evalforge"
    if scripts_dir.exists():
        return str(scripts_dir)

    # 3. Walk up from client.py looking for target/debug/evalforge
    current = Path(__file__).resolve().parent
    for _ in range(6):
        candidate = current / "target" / "debug" / "evalforge"
        if candidate.exists():
            return str(candidate)
        candidate2 = current / "target" / "release" / "evalforge"
        if candidate2.exists():
            return str(candidate2)
        current = current.parent

    # 4. On PATH
    import shutil
    which = shutil.which("evalforge")
    if which:
        return which

    raise RuntimeError(
        "\nEvalForge binary not found.\n\n"
        "Option 1 — Build from source:\n"
        "  git clone https://github.com/heManKuMAR6/evalforge\n"
        "  cd evalforge && cargo build --release\n"
        "  export EVALFORGE_BIN=/path/to/evalforge/target/release/evalforge\n\n"
        "Option 2 — Set binary path manually:\n"
        "  export EVALFORGE_BIN=/path/to/evalforge/binary\n\n"
        "Full instructions: github.com/heManKuMAR6/evalforge\n"
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


def demo() -> EvalResult:
    """Run EvalForge on a built-in sample trace. No file needed."""
    import tempfile
    import json

    sample_trace = {
        "evalforge_version": "0.1",
        "trace_id": "demo-trace-001",
        "timestamp": "2026-04-02T10:00:00Z",
        "metadata": {
            "framework": "langchain",
            "model": "gpt-4o",
            "agent_name": "demo-agent",
            "duration_ms": 3421,
            "total_tokens": 1820
        },
        "input": {
            "user": "What is the capital of Australia?",
            "system": "You are a helpful assistant."
        },
        "steps": [
            {
                "step_id": 1,
                "type": "thought",
                "content": "The user wants to know Australia's capital."
            },
            {
                "step_id": 2,
                "type": "tool_call",
                "tool": "web_search",
                "input": {"query": "capital of Australia"},
                "output": {"result": "Canberra is the capital of Australia."},
                "duration_ms": 800
            }
        ],
        "output": {
            "answer": "The capital of Australia is Canberra."
        },
        "eval_hints": {
            "expected_tools": ["web_search"],
            "expected_answer": "Canberra",
            "context_documents": []
        }
    }

    with tempfile.NamedTemporaryFile(
        mode='w', suffix='.json', delete=False
    ) as f:
        json.dump(sample_trace, f)
        tmp_path = f.name

    return run(tmp_path, metrics=["faithfulness"], mock=True)


def _extract_field(output: str, label: str) -> str:
    pattern = re.compile(rf"^{re.escape(label)}:\s+(.+)$", re.MULTILINE)
    m = pattern.search(output)
    return m.group(1).strip() if m else ""
