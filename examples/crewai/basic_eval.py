"""
EvalForge + CrewAI Example

This example shows how to evaluate a CrewAI agent with EvalForge.
The trace is manually constructed here to show the expected format.
In production, use CrewAI callbacks to capture traces automatically.
"""

import json
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../sdks/python'))

import evalforge

# In production this trace would be captured automatically from CrewAI callbacks
# Here we construct it manually to show the expected format
trace = {
    "evalforge_version": "0.1",
    "trace_id": "crewai-example-001",
    "timestamp": "2026-04-02T10:00:00Z",
    "metadata": {
        "framework": "crewai",
        "model": "gpt-4o",
        "agent_name": "content-writer",
        "duration_ms": 1850,
        "total_tokens": 1120
    },
    "input": {
        "user": "Write a short summary of transformer architecture",
        "system": "You are a technical content writer."
    },
    "steps": [
        {
            "step_id": 1,
            "type": "thought",
            "content": "The user wants a concise summary of transformer architecture. I should look up the key concepts to ensure accuracy."
        },
        {
            "step_id": 2,
            "type": "tool_call",
            "tool": "knowledge_base",
            "input": {"query": "transformer architecture self-attention encoder decoder"},
            "output": {
                "result": (
                    "Transformers use self-attention mechanisms to process sequences in parallel. "
                    "The architecture consists of an encoder and decoder, each built from stacked layers "
                    "of multi-head self-attention and feed-forward networks. Introduced in 'Attention Is All You Need' (2017)."
                )
            },
            "duration_ms": 620
        }
    ],
    "output": {
        "answer": (
            "The transformer architecture, introduced in 2017, processes sequences using self-attention "
            "rather than recurrence. It consists of an encoder and decoder, each made up of stacked layers "
            "containing multi-head self-attention and feed-forward sub-layers. This parallel processing "
            "approach makes transformers highly efficient and scalable, forming the foundation of modern "
            "large language models."
        )
    },
    "eval_hints": {
        "expected_tools": ["knowledge_base"],
        "expected_answer": None,
        "context_documents": []
    }
}

# Save trace to temp file
trace_path = "/tmp/crewai_example_trace.json"
with open(trace_path, "w") as f:
    json.dump(trace, f)

print("Running EvalForge on CrewAI agent trace...")
print()

result = evalforge.run(trace_path, metrics=["faithfulness"], mock=True)

print(f"Framework:  {trace['metadata']['framework']}")
print(f"Agent:      {trace['metadata']['agent_name']}")
print(f"Passed:     {result.passed}")
for m in result.metrics:
    print(f"  {m.metric}: {m.score} ({'PASS' if m.passed else 'FAIL'})")
