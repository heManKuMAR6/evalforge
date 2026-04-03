"""
LangChain trace adapter for EvalForge.

Converts LangChain AgentExecutor output to EvalForge trace format.

Usage:
    from langchain.agents import AgentExecutor
    from evalforge.adapters import from_langchain
    import evalforge

    # Run your LangChain agent
    result = agent_executor.invoke({"input": "Your question"})

    # Convert to EvalForge trace
    trace = from_langchain(
        result=result,
        model="gpt-4o",
        agent_name="my-agent"
    )

    # Evaluate
    eval_result = evalforge.run(trace, metrics=["faithfulness"])
"""

import json
import tempfile
import uuid
from datetime import datetime, timezone
from typing import Any


def from_langchain(
    result: dict[str, Any],
    model: str = "unknown",
    agent_name: str = "langchain-agent",
    intermediate_steps: list | None = None,
) -> str:
    """
    Convert LangChain AgentExecutor output to EvalForge trace JSON file.

    Args:
        result: The dict returned by agent_executor.invoke()
                Must have "input" and "output" keys.
        model: The LLM model used (e.g. "gpt-4o")
        agent_name: Name for this agent
        intermediate_steps: Optional list of (AgentAction, observation) tuples
                            from result.get("intermediate_steps", [])

    Returns:
        Path to a temporary trace JSON file ready for evalforge.run()
    """
    steps = []
    step_id = 1
    total_tokens = 0

    if intermediate_steps is None:
        intermediate_steps = result.get("intermediate_steps", [])

    for action, observation in intermediate_steps:
        tool_name = getattr(action, "tool", "unknown_tool")
        tool_input = getattr(action, "tool_input", {})
        if isinstance(tool_input, str):
            tool_input = {"query": tool_input}

        steps.append({
            "step_id": step_id,
            "type": "thought",
            "content": getattr(action, "log", "").strip(),
        })
        step_id += 1

        steps.append({
            "step_id": step_id,
            "type": "tool_call",
            "tool": tool_name,
            "input": tool_input,
            "output": {"result": str(observation)},
            "duration_ms": 0,
        })
        step_id += 1

    trace = {
        "evalforge_version": "0.1",
        "trace_id": str(uuid.uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "metadata": {
            "framework": "langchain",
            "model": model,
            "agent_name": agent_name,
            "duration_ms": 0,
            "total_tokens": total_tokens,
        },
        "input": {
            "user": result.get("input", ""),
            "system": "",
        },
        "steps": steps,
        "output": {
            "answer": result.get("output", ""),
        },
        "eval_hints": {
            "expected_tools": [],
            "expected_answer": None,
            "context_documents": [],
        },
    }

    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False
    )
    json.dump(trace, tmp)
    tmp.close()
    return tmp.name
