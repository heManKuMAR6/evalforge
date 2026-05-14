"""CI integrations for EvalForge — GitHub PR comments and diff reporting."""
from .github import (
    post_pr_comment,
    build_diff_table,
    load_swarm_result,
    diff_results,
)

__all__ = ["post_pr_comment", "build_diff_table", "load_swarm_result", "diff_results"]
