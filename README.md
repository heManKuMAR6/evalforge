```markdown
# EvalForge

> Framework-agnostic LLM agent evaluation harness. Score any agent, any framework, in CI.

[![CI](https://github.com/heManKuMAR6/evalforge/actions/workflows/ci.yml/badge.svg)](https://github.com/heManKuMAR6/evalforge/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## The Problem

Every agent framework — LangChain, CrewAI, AutoGen, OpenAI Agents SDK, Mastra — has its own
trace format, its own logging, its own evaluation story (if any). There is no single tool that
can take a run from any of these, score it on quality metrics, and give you a pass/fail in CI.

You cannot improve agents you cannot measure.

## What EvalForge Does

EvalForge reads a standard trace JSON, scores it using research-backed metrics, and outputs
a result your CI pipeline understands — regardless of which framework produced the trace.

```bash
evalforge run --trace my_agent_run.json --metrics faithfulness
```

Output:
```json
{
  "metrics": {
    "faithfulness": { "score": 0.91, "pass": true, "reason": "The answer accurately reflects the retrieved context." }
  },
  "overall": "PASS"
}
```

## Supported Frameworks

| Framework | Language | Status |
|-----------|----------|--------|
| LangChain / LangGraph | Python | ✅ v0.1 |
| CrewAI | Python | ✅ v0.1 |
| AutoGen / AG2 | Python | ✅ v0.1 |
| OpenAI Agents SDK | Python | ✅ v0.1 |
| Mastra | TypeScript | 🔜 Planned |
| Vercel AI SDK | TypeScript | 🔜 Planned |

## Installation

```bash
# Python SDK
pip install evalforge

# CLI — build from source
git clone https://github.com/heManKuMAR6/evalforge
cd evalforge
cargo build --release
```

## Quick Start

```python
import evalforge

result = evalforge.run(
    trace="my_agent_run.json",
    metrics=["faithfulness"]
)

print(result.passed)              # True
print(result.metrics[0].score)   # 0.91
print(result.metrics[0].reason)  # "The answer accurately reflects..."
```

## CLI Usage

```bash
# Basic scoring
evalforge run --trace my_trace.json --metrics faithfulness

# Custom threshold
evalforge run --trace my_trace.json --metrics faithfulness --threshold 0.8

# Mock mode (no API key needed — for testing)
evalforge run --trace my_trace.json --metrics faithfulness --mock
```

## Trace Format

EvalForge uses a simple universal trace format that any framework can map to:

```json
{
  "evalforge_version": "0.1",
  "trace_id": "my-run-001",
  "metadata": {
    "framework": "langchain",
    "model": "gpt-4o",
    "agent_name": "my-agent",
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
      "content": "I need to look this up."
    },
    {
      "step_id": 2,
      "type": "tool_call",
      "tool": "web_search",
      "input": { "query": "capital of Australia" },
      "output": { "result": "Canberra is the capital." },
      "duration_ms": 800
    }
  ],
  "output": {
    "answer": "The capital of Australia is Canberra."
  },
  "eval_hints": {
    "expected_tools": ["web_search"],
    "expected_answer": "Canberra",
    "context_documents": []
  }
}
```

## Metrics

| Metric | Description | Status |
|--------|-------------|--------|
| `faithfulness` | Did the answer stay true to retrieved context? | ✅ v0.1 |
| `tool_accuracy` | Did the agent use the right tools correctly? | 🔜 v0.2 |
| `goal_completion` | Did the agent complete the assigned task? | 🔜 v0.2 |
| `hallucination` | Did the agent make up facts? | 🔜 v0.2 |

## CI/CD Integration

Add EvalForge to your GitHub Actions pipeline:

```yaml
- name: Evaluate agent
  run: evalforge run --trace agent_run.json --metrics faithfulness
  env:
    ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
```

Exit code 0 = all metrics pass. Exit code 1 = one or more metrics fail. 
Plug directly into any CI pipeline.

## Architecture

```
evalforge/
├── crates/
│   ├── evalforge-core/     # Rust core — trace parsing, scoring
│   └── evalforge-cli/      # CLI binary
├── sdks/
│   ├── python/             # pip install evalforge
│   └── js/                 # npm install evalforge (coming soon)
└── examples/               # Working examples per framework
    ├── langchain/
    ├── crewai/
    ├── autogen/
    └── openai-agents/
```

## Examples

See the `examples/` folder for working examples with each framework:

```bash
python examples/langchain/basic_eval.py
python examples/crewai/basic_eval.py
python examples/autogen/basic_eval.py
python examples/openai-agents/basic_eval.py
```

## Roadmap

- [x] v0.1 — CLI + trace parser + faithfulness metric + Python SDK
- [ ] v0.2 — tool_accuracy + goal_completion + hallucination metrics
- [ ] v0.3 — CI/CD integrations (GitHub Actions, GitLab CI)
- [ ] v0.4 — JS SDK + Mastra support
- [ ] v0.5 — Trace adapters (auto-convert from LangChain, CrewAI natively)
- [ ] v1.0 — Full framework adapters + web dashboard

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). We welcome issues, PRs, and feedback.

If you find EvalForge useful, please star the repo — it helps others discover it.

## License

MIT © 2026 Hemanth Kumar
```

---

Once you've pasted it into `README.md`, commit and push:

```bash
git add .
git commit -m "docs: update README with CI badge, real examples, and full usage docs"
git push
```
