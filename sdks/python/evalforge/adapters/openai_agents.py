"""
OpenAI Agents SDK adapter for EvalForge.

Usage:
    from openai_agents import Agent, Runner
    from evalforge.adapters import from_openai_agents
    import evalforge

    agent = Agent(name="my-agent", instructions="You are helpful.")
    result = Runner.run_sync(agent, "What is the capital of France?")

    trace = from_openai_agents(result, agent_name="my-agent")
    eval_result = evalforge.run(trace, metrics=["faithfulness", "goal_completion"])
"""

import json
import tempfile
import uuid
from datetime import datetime, timezone
from typing import Any


def from_openai_agents(
    result: Any,
    agent_name: str = "openai-agent",
    model: str = "gpt-4o",
) -> str:
    """
    Convert OpenAI Agents SDK result to EvalForge trace JSON file.

    Args:
        result: The result from Runner.run_sync() or await Runner.run()
                Supports RunResult with .final_output and .new_messages
        agent_name: Name for this agent
        model: The model used

    Returns:
        Path to a temporary trace JSON file ready for evalforge.run()
    """
    final_output = ""
    user_input = ""
    steps = []
    step_id = 1

    # Extract from RunResult object
    if hasattr(result, "final_output"):
        final_output = str(result.final_output)

    # Extract messages
    messages = []
    if hasattr(result, "new_messages"):
        messages = result.new_messages
    elif hasattr(result, "messages"):
        messages = result.messages
    elif isinstance(result, list):
        messages = result

    for msg in messages:
        role = getattr(msg, "role", None) or msg.get("role", "assistant")
        content = getattr(msg, "content", None) or msg.get("content", "")

        if isinstance(content, list):
            content = " ".join(
                c.get("text", "") if isinstance(c, dict) else str(c)
                for c in content
            )

        if role == "user" and not user_input:
            user_input = str(content)
            continue

        # Check for tool calls
        tool_calls = getattr(msg, "tool_calls", None)
        if tool_calls:
            for tc in tool_calls:
                tool_name = getattr(tc, "function", {})
                if hasattr(tool_name, "name"):
                    tool_name = tool_name.name
                else:
                    tool_name = str(tool_name)

                steps.append({
                    "step_id": step_id,
                    "type": "tool_call",
                    "tool": tool_name,
                    "input": {},
                    "output": {"result": str(content)},
                    "duration_ms": 0,
                })
                step_id += 1
        else:
            steps.append({
                "step_id": step_id,
                "type": "thought",
                "content": str(content),
            })
            step_id += 1

    if not final_output and steps:
        final_output = steps[-1].get("content", "")

    trace = {
        "evalforge_version": "0.1",
        "trace_id": str(uuid.uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "metadata": {
            "framework": "openai-agents",
            "model": model,
            "agent_name": agent_name,
            "duration_ms": 0,
            "total_tokens": 0,
        },
        "input": {
            "user": user_input,
            "system": "",
        },
        "steps": steps,
        "output": {
            "answer": final_output,
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
