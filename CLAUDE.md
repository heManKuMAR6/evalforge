# EvalForge — Claude Code Guide

## What this project is
EvalForge is a framework-agnostic LLM agent evaluation harness.
- Rust core (`crates/`) — trace parsing, metrics, CLI binary
- Python SDK (`sdks/python/`) — pip install evalforge
- JS/TS SDK (`sdks/js/`) — npm install evalforge

## Architecture
```
crates/
  evalforge-core/   — library: trace parsing + all metrics
  evalforge-cli/    — binary: CLI that reads traces and scores them
sdks/
  python/           — thin wrapper calling the CLI via subprocess
  js/               — TypeScript SDK calling the CLI via spawnSync
tests/fixtures/     — sample trace JSON files for testing
```

## Key rules when making changes

1. Every new metric needs:
   - A new file in `crates/evalforge-core/src/metrics/`
   - Added to `crates/evalforge-core/src/metrics/mod.rs`
   - Handler added to `crates/evalforge-cli/src/main.rs`
   - At least 2 unit tests

2. Never change the trace format in `tests/fixtures/` without updating:
   - The Rust structs in `crates/evalforge-core/src/trace.rs`
   - Both fixture files (`sample_trace.json` and `simple_trace.json`)

3. Always run `cargo test` after any Rust changes

4. Always run `cd sdks/python && uv run pytest tests/ -v` after Python changes

5. The CLI output format is: `metric_name     score   PASS/FAIL`
   Never change this format — the Python and JS SDKs parse it.

## How to add a new metric

1. Create `crates/evalforge-core/src/metrics/my_metric.rs`
2. Add `pub mod my_metric;` to `crates/evalforge-core/src/metrics/mod.rs`
3. Add handler in `crates/evalforge-cli/src/main.rs` match block
4. Add mock result for `--mock` flag
5. Run `cargo test` — must pass

## How to add a new framework adapter (Python)

1. Create `sdks/python/evalforge/adapters/my_framework.py`
2. Export from `sdks/python/evalforge/adapters/__init__.py`
3. Add tests to `sdks/python/tests/test_adapters.py`
4. Run `uv run pytest tests/ -v` — must pass

## How to add a new framework adapter (JS)

1. Create `sdks/js/src/adapters/my_framework.ts`
2. Export from `sdks/js/src/adapters/index.ts`
3. Add tests to `sdks/js/tests/adapters.test.js`
4. Run `npm test` — must pass

## Versioning

All three must match:
- `crates/evalforge-core/Cargo.toml`
- `crates/evalforge-cli/Cargo.toml`
- `sdks/python/pyproject.toml`
- `sdks/js/package.json`

## Running tests

```bash
# Rust
cargo test

# Python
cd sdks/python && uv run pytest tests/ -v

# JS
cd sdks/js && npm test

# All at once
cargo test && cd sdks/python && uv run pytest tests/ -v && cd ../js && npm test
```

## Common mistakes to avoid

- Do NOT change the CLI output format — SDKs depend on it
- Do NOT add unwrap() in Rust without handling the error
- Do NOT commit API keys
- Do NOT skip tests when adding new metrics
- Do NOT change trace JSON field names without updating Rust structs
