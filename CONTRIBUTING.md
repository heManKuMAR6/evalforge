# Contributing to EvalForge

Thank you for your interest in contributing!

## Getting Started

1. Fork the repo
2. Clone your fork: `git clone https://github.com/YOUR_USERNAME/evalforge`
3. Create a branch: `git checkout -b feat/your-feature`
4. Make your changes
5. Run tests: `cargo test`
6. Push and open a PR

## Development Setup

Requirements:
- Rust 1.75+
- Python 3.11+
- uv
```bash
# Install Rust deps
cargo build

# Install Python SDK deps
cd sdks/python
uv sync
```

## Code Style

- Rust: `cargo fmt` before committing
- Python: `ruff format` before committing

## Reporting Issues

Use the GitHub Issues tab. Please include:
- Your OS and version
- The trace file (redact sensitive data)
- The command you ran
- The output you got vs what you expected
