# EvalForge

> Framework-agnostic LLM agent evaluation harness. Score any agent, any framework, in CI.

[![CI](https://github.com/YOUR_USERNAME/evalforge/actions/workflows/ci.yml/badge.svg)](https://github.com/YOUR_USERNAME/evalforge/actions)
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
evalforge run --trace my_agent_run.json --metrics faithfulness,tool_accuracy
```

Output:
```json
{
  "metrics": {
    "faithfulness": { "score": 0.91, "pass": true },
    "tool_accuracy": { "score": 0.78, "pass": false }
  },
  "overall": "FAIL"
}
```

## Supported Frameworks

| Framework | Language | Status |
|-----------|----------|--------|
| LangChain / LangGraph | Python | 🚧 Coming v0.1 |
| CrewAI | Python | 🚧 Coming v0.1 |
| AutoGen / AG2 | Python | 🚧 Coming v0.1 |
| OpenAI Agents SDK | Python | 🚧 Coming v0.1 |
| Mastra | TypeScript | 🔜 Planned |
| Vercel AI SDK | TypeScript | 🔜 Planned |

## Installation
```bash
# Python SDK
pip install evalforge

# CLI (coming soon)
curl -LsSf https://evalforge.dev/install.sh | sh
```

## Quick Start
```python
import evalforge

result = evalforge.run(
    trace="my_agent_run.json",
    metrics=["faithfulness", "tool_accuracy"]
)

print(result.score)       # 0.85
print(result.passed)      # True
```

## Metrics

| Metric | Description |
|--------|-------------|
| `faithfulness` | Did the answer stay true to retrieved context? |
| `tool_accuracy` | Did the agent use the right tools correctly? |
| `goal_completion` | Did the agent complete the assigned task? |
| `hallucination` | Did the agent make up facts? |

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
```

## Roadmap

- [ ] v0.1 — CLI + trace parser + faithfulness metric + Python SDK
- [ ] v0.2 — tool_accuracy + goal_completion metrics
- [ ] v0.3 — CI/CD integrations (GitHub Actions, GitLab CI)
- [ ] v0.4 — JS SDK + Mastra support
- [ ] v1.0 — Full framework adapters + dashboard

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). We welcome issues, PRs, and feedback.

## License

MIT © 2026 Hemanth Kumar
