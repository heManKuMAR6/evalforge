"""
CrewAI trace adapter for EvalForge.

Usage:
    from crewai import Crew
    from evalforge.adapters import from_crewai
    import evalforge

    crew = Crew(agents=[...], tasks=[...])
    result = crew.kickoff()

    trace = from_crewai(result, crew_name="my-crew")
    eval_result = evalforge.run(trace, metrics=["faithfulness", "goal_completion"])
"""

import json
import tempfile
import uuid
from datetime import datetime, timezone
from typing import Any


def from_crewai(
    result: Any,
    crew_name: str = "crewai-agent",
    model: str = "unknown",
    task_description: str = "",
) -> str:
    """
    Convert CrewAI kickoff result to EvalForge trace JSON file.

    Args:
        result: The result returned by crew.kickoff()
        crew_name: Name for this crew
        model: The LLM model used
        task_description: The original task description given to the crew

    Returns:
        Path to a temporary trace JSON file ready for evalforge.run()
    """
    final_output = str(result) if not isinstance(result, str) else result

    steps = [{
        "step_id": 1,
        "type": "thought",
        "content": f"CrewAI crew '{crew_name}' executed task.",
    }]

    trace = {
        "evalforge_version": "0.1",
        "trace_id": str(uuid.uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "metadata": {
            "framework": "crewai",
            "model": model,
            "agent_name": crew_name,
            "duration_ms": 0,
            "total_tokens": 0,
        },
        "input": {
            "user": task_description,
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
