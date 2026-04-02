"""
EvalForge + AutoGen Example

This example shows how to evaluate an AutoGen agent with EvalForge.
The trace is manually constructed here to show the expected format.
In production, use AutoGen message hooks to capture traces automatically.
"""

import json
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../sdks/python'))

import evalforge

# In production this trace would be captured automatically from AutoGen message hooks
# Here we construct it manually to show the expected format
trace = {
    "evalforge_version": "0.1",
    "trace_id": "autogen-example-001",
    "timestamp": "2026-04-02T10:00:00Z",
    "metadata": {
        "framework": "autogen",
        "model": "gpt-4o",
        "agent_name": "coding-agent",
        "duration_ms": 1430,
        "total_tokens": 780
    },
    "input": {
        "user": "What is a binary search tree?",
        "system": "You are a coding assistant that explains computer science concepts clearly."
    },
    "steps": [
        {
            "step_id": 1,
            "type": "thought",
            "content": "The user wants an explanation of binary search trees. I'll retrieve a reference definition and then craft a clear explanation."
        },
        {
            "step_id": 2,
            "type": "tool_call",
            "tool": "code_executor",
            "input": {"code": "# Retrieve BST definition from knowledge store\nprint(get_concept('binary_search_tree'))"},
            "output": {
                "result": (
                    "A binary search tree (BST) is a node-based data structure where each node has at most two children. "
                    "For every node, all values in the left subtree are less than the node's value, "
                    "and all values in the right subtree are greater. "
                    "Search, insertion, and deletion are O(log n) on average."
                )
            },
            "duration_ms": 390
        }
    ],
    "output": {
        "answer": (
            "A binary search tree (BST) is a data structure where each node has at most two children — "
            "a left child and a right child. The key property is ordering: every value in a node's left subtree "
            "is smaller than the node's value, and every value in its right subtree is larger. "
            "This makes search, insertion, and deletion operations O(log n) on average, making BSTs "
            "efficient for sorted data storage and retrieval."
        )
    },
    "eval_hints": {
        "expected_tools": ["code_executor"],
        "expected_answer": None,
        "context_documents": []
    }
}

# Save trace to temp file
trace_path = "/tmp/autogen_example_trace.json"
with open(trace_path, "w") as f:
    json.dump(trace, f)

print("Running EvalForge on AutoGen agent trace...")
print()

result = evalforge.run(trace_path, metrics=["faithfulness"], mock=True)

print(f"Framework:  {trace['metadata']['framework']}")
print(f"Agent:      {trace['metadata']['agent_name']}")
print(f"Passed:     {result.passed}")
for m in result.metrics:
    print(f"  {m.metric}: {m.score} ({'PASS' if m.passed else 'FAIL'})")
