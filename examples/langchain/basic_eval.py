"""
EvalForge + LangChain Example

This example shows how to evaluate a LangChain agent with EvalForge.
The trace is manually constructed here to show the expected format.
In production, use LangChain callbacks to capture traces automatically.
"""

import json
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../sdks/python'))

import evalforge

# In production this trace would be captured automatically from LangChain callbacks
# Here we construct it manually to show the expected format
trace = {
    "evalforge_version": "0.1",
    "trace_id": "langchain-example-001",
    "timestamp": "2026-04-02T10:00:00Z",
    "metadata": {
        "framework": "langchain",
        "model": "gpt-4o",
        "agent_name": "research-agent",
        "duration_ms": 2100,
        "total_tokens": 950
    },
    "input": {
        "user": "What is the capital of Australia?",
        "system": "You are a helpful assistant."
    },
    "steps": [
        {
            "step_id": 1,
            "type": "thought",
            "content": "The user is asking about Australia's capital. I know this is Canberra, not Sydney."
        },
        {
            "step_id": 2,
            "type": "tool_call",
            "tool": "web_search",
            "input": {"query": "capital of Australia"},
            "output": {"result": "Canberra is the capital of Australia. It became the capital in 1913."},
            "duration_ms": 800
        }
    ],
    "output": {
        "answer": "The capital of Australia is Canberra. It has been the capital since 1913."
    },
    "eval_hints": {
        "expected_tools": ["web_search"],
        "expected_answer": "Canberra",
        "context_documents": []
    }
}

# Save trace to temp file
trace_path = "/tmp/langchain_example_trace.json"
with open(trace_path, "w") as f:
    json.dump(trace, f)

print("Running EvalForge on LangChain agent trace...")
print()

result = evalforge.run(trace_path, metrics=["faithfulness"], mock=True)

print(f"Framework:  {trace['metadata']['framework']}")
print(f"Agent:      {trace['metadata']['agent_name']}")
print(f"Passed:     {result.passed}")
for m in result.metrics:
    print(f"  {m.metric}: {m.score} ({'PASS' if m.passed else 'FAIL'})")
