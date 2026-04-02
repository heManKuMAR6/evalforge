"""
EvalForge + OpenAI Agents SDK Example

This example shows how to evaluate an OpenAI Agents SDK agent with EvalForge.
The trace is manually constructed here to show the expected format.
In production, use OpenAI Agents SDK tracing hooks to capture traces automatically.
"""

import json
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../sdks/python'))

import evalforge

# In production this trace would be captured automatically from OpenAI Agents SDK tracing hooks
# Here we construct it manually to show the expected format
trace = {
    "evalforge_version": "0.1",
    "trace_id": "openai-agents-example-001",
    "timestamp": "2026-04-02T10:00:00Z",
    "metadata": {
        "framework": "openai-agents",
        "model": "gpt-4o-mini",
        "agent_name": "customer-support-agent",
        "duration_ms": 1200,
        "total_tokens": 640
    },
    "input": {
        "user": "How do I reset my password?",
        "system": "You are a customer support agent. Answer questions using the knowledge base."
    },
    "steps": [
        {
            "step_id": 1,
            "type": "thought",
            "content": "The user wants to reset their password. I should look up the password reset procedure in the knowledge base."
        },
        {
            "step_id": 2,
            "type": "tool_call",
            "tool": "knowledge_base",
            "input": {"query": "password reset procedure"},
            "output": {
                "result": (
                    "To reset your password: (1) Go to the login page and click 'Forgot Password'. "
                    "(2) Enter your registered email address. "
                    "(3) Check your email for a reset link — it expires in 30 minutes. "
                    "(4) Click the link and enter your new password. "
                    "(5) Log in with your new password."
                )
            },
            "duration_ms": 410
        }
    ],
    "output": {
        "answer": (
            "To reset your password, follow these steps:\n"
            "1. Go to the login page and click 'Forgot Password'.\n"
            "2. Enter your registered email address.\n"
            "3. Check your email for a reset link (it expires in 30 minutes).\n"
            "4. Click the link and enter your new password.\n"
            "5. Log in with your new password.\n\n"
            "If you don't receive the email within a few minutes, check your spam folder."
        )
    },
    "eval_hints": {
        "expected_tools": ["knowledge_base"],
        "expected_answer": None,
        "context_documents": []
    }
}

# Save trace to temp file
trace_path = "/tmp/openai_agents_example_trace.json"
with open(trace_path, "w") as f:
    json.dump(trace, f)

print("Running EvalForge on OpenAI Agents SDK agent trace...")
print()

result = evalforge.run(trace_path, metrics=["faithfulness"], mock=True)

print(f"Framework:  {trace['metadata']['framework']}")
print(f"Agent:      {trace['metadata']['agent_name']}")
print(f"Passed:     {result.passed}")
for m in result.metrics:
    print(f"  {m.metric}: {m.score} ({'PASS' if m.passed else 'FAIL'})")
