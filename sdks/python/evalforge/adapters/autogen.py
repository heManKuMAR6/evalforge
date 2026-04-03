"""
AutoGen trace adapter for EvalForge.

Usage:
    from autogen import AssistantAgent, UserProxyAgent
    from evalforge.adapters import from_autogen
    import evalforge

    assistant = AssistantAgent("assistant", llm_config={...})
    user_proxy = UserProxyAgent("user_proxy")

    chat_result = user_proxy.initiate_chat(
        assistant,
        message="Your question"
    )

    trace = from_autogen(chat_result, model="gpt-4o")
    eval_result = evalforge.run(trace, metrics=["faithfulness"])
"""

import json
import tempfile
import uuid
from datetime import datetime, timezone
from typing import Any


def from_autogen(
    chat_result: Any,
    model: str = "unknown",
    agent_name: str = "autogen-agent",
) -> str:
    """
    Convert AutoGen chat result to EvalForge trace JSON file.

    Args:
        chat_result: The result from user_proxy.initiate_chat()
        model: The LLM model used
        agent_name: Name for this agent

    Returns:
        Path to a temporary trace JSON file ready for evalforge.run()
    """
    messages = []
    if hasattr(chat_result, "chat_history"):
        messages = chat_result.chat_history
    elif isinstance(chat_result, list):
        messages = chat_result

    steps = []
    user_input = ""
    final_output = ""

    for i, msg in enumerate(messages):
        role = msg.get("role", "assistant")
        content = msg.get("content", "")

        if i == 0:
            user_input = content
            continue

        steps.append({
            "step_id": i,
            "type": "thought",
            "content": f"[{role}]: {content}",
        })
        final_output = content

    trace = {
        "evalforge_version": "0.1",
        "trace_id": str(uuid.uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "metadata": {
            "framework": "autogen",
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
