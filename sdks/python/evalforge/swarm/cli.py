"""CLI entrypoint shared by the Rust ``evalforge swarm`` / ``evalforge diff``
subcommands.

Invoked as ``python3 -m evalforge.swarm.cli <subcommand> ...``. The Rust binary
forwards its CLI args here so we keep one implementation rather than
duplicating the orchestration logic in two languages.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

from .orchestrator import run_swarm_async
from ..ci.github import (
    build_diff_table,
    diff_results,
    load_swarm_result,
    post_pr_comment,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="evalforge-swarm")
    sub = parser.add_subparsers(dest="cmd", required=True)

    swarm = sub.add_parser("swarm", help="Run the parallel judge swarm")
    swarm.add_argument("--traces", required=True, help="Directory of trace JSON files")
    swarm.add_argument("--output", help="Directory to save per-trace SwarmResult JSON")
    swarm.add_argument("--model", default=None, help="Judge model (default: from config / env)")
    swarm.add_argument("--threshold", type=float, default=0.7)
    swarm.add_argument("--consensus-threshold", type=float, default=0.3)
    swarm.add_argument("--mock", action="store_true", help="Use deterministic mock scores")
    swarm.add_argument("--rubric", default=None, help="Rubric for g_eval metric")

    diff = sub.add_parser("diff", help="Diff two swarm result files / dirs")
    diff.add_argument("--before", required=True, help="Path to baseline result(s)")
    diff.add_argument("--after", required=True, help="Path to new result(s)")
    diff.add_argument("--post-github-comment", action="store_true",
                      help="POST the diff table as a PR comment")
    diff.add_argument("--repo", default=None, help="owner/repo (default: $GITHUB_REPOSITORY)")
    diff.add_argument("--pr", default=None, help="PR number (default: derived from env)")
    diff.add_argument("--output", default=None,
                      help="Write the markdown table to this file as well as stdout")

    return parser


async def _cmd_swarm(args: argparse.Namespace) -> int:
    traces_dir = Path(args.traces)
    if not traces_dir.is_dir():
        print(f"error: {traces_dir} is not a directory", file=sys.stderr)
        return 1

    trace_paths = sorted(traces_dir.glob("*.json"))
    if not trace_paths:
        print(f"error: no trace JSON files found in {traces_dir}", file=sys.stderr)
        return 1

    out_dir = Path(args.output) if args.output else None
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)

    print("EvalForge v2.0 — Swarm")
    print(f"Traces:  {len(trace_paths)} found in {traces_dir}")
    mode_label = "mock" if args.mock else f"live ({args.model or 'default'})"
    print(f"Mode:    {mode_label}")
    print("─" * 60)

    all_passed = True
    total_start = time.perf_counter()

    for path in trace_paths:
        try:
            result = await run_swarm_async(
                path,
                model=args.model,
                mock=args.mock,
                threshold=args.threshold,
                consensus_threshold=args.consensus_threshold,
                rubric=args.rubric,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"{path.name:<32}  ERROR: {exc}")
            all_passed = False
            continue

        verdict = "PASS" if result.passed else "FAIL"
        per_judge = " ".join(f"{j}:{ms}ms" for j, ms in result.judge_timings_ms.items())
        print(
            f"{path.name:<32}  weighted={result.weighted_score:.2f}  "
            f"total={result.total_elapsed_ms}ms  {per_judge}  {verdict}"
        )
        if result.failure_reason:
            print(f"    failure_reason: {result.failure_reason}")
        if result.human_review_flags:
            for f in result.human_review_flags:
                print(f"    review: judge {f['judge']} metrics {f['metrics']} delta={f['delta']}")

        if out_dir is not None:
            out_path = out_dir / f"{result.trace_id}.json"
            out_path.write_text(json.dumps(result.raw, indent=2))

        if not result.passed:
            all_passed = False

    total_elapsed_ms = int((time.perf_counter() - total_start) * 1000)
    print("─" * 60)
    print(f"Total elapsed: {total_elapsed_ms}ms across {len(trace_paths)} traces "
          f"({total_elapsed_ms / max(len(trace_paths), 1):.0f}ms/trace)")
    print(f"Swarm result: {'PASS' if all_passed else 'FAIL'}")
    return 0 if all_passed else 1


def _cmd_diff(args: argparse.Namespace) -> int:
    before = load_swarm_result(args.before)
    after = load_swarm_result(args.after)

    table = build_diff_table(before, after)
    print(table)

    if args.output:
        Path(args.output).write_text(table)

    if args.post_github_comment:
        pr = args.pr or None
        try:
            post_pr_comment(table, repo=args.repo, pr_number=pr)
            print("\nPosted PR comment.")
        except RuntimeError as exc:
            print(f"\nerror posting PR comment: {exc}", file=sys.stderr)
            return 2

    # Non-zero exit if overall regressed by >0.05
    d = diff_results(before, after)
    return 1 if d["overall"]["delta"] < -0.05 else 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.cmd == "swarm":
        return asyncio.run(_cmd_swarm(args))
    if args.cmd == "diff":
        return _cmd_diff(args)
    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
