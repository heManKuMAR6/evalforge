"""Model routing for swarm judges.

Selects which LLM to call based on flag + config + env. Three supported models:
  - deepseek-v4-flash   — primary, OpenAI-compatible API
  - ollama/<name>       — fallback, OpenAI-compatible API at local base_url
  - claude-haiku-4-5    — original v1.0 judge (Anthropic API), default

Config resolution order:
  1. Explicit ``model`` argument to ``ModelRouter`` constructor
  2. ``--judge-model`` CLI flag (passed via ``model`` arg from caller)
  3. ``EVALFORGE_JUDGE_MODEL`` env var
  4. ``evalforge.config.json`` in CWD (or any parent up to repo root)
  5. Default: ``claude-haiku-4-5``
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Supported model identifiers — anything else falls back to claude-haiku-4-5.
SUPPORTED_MODELS = ("deepseek-v4-flash", "claude-haiku-4-5")
DEFAULT_MODEL = "claude-haiku-4-5"


@dataclass
class JudgeConfig:
    """Resolved configuration for a single judge LLM call."""
    model: str
    provider: str           # "deepseek" | "ollama" | "anthropic"
    base_url: str
    api_key: str
    headers: dict = field(default_factory=dict)
    api_model_name: str = ""   # actual model id sent in API body

    def is_anthropic(self) -> bool:
        return self.provider == "anthropic"

    def is_openai_compatible(self) -> bool:
        return self.provider in ("deepseek", "ollama")


def load_config(start_dir: str | Path | None = None) -> dict:
    """Load evalforge.config.json by walking up from start_dir to filesystem root.

    Returns an empty dict if no config is found. Never raises for missing files —
    only for malformed JSON.
    """
    if start_dir is None:
        start_dir = Path.cwd()
    current = Path(start_dir).resolve()
    for _ in range(8):
        candidate = current / "evalforge.config.json"
        if candidate.is_file():
            return json.loads(candidate.read_text())
        if current.parent == current:
            break
        current = current.parent
    return {}


def _normalize_model(name: str) -> str:
    """Strip ``ollama/`` prefix and lowercase. Returns the canonical key."""
    n = name.strip().lower()
    if n.startswith("ollama/"):
        return "ollama"
    return n


class ModelRouter:
    """Resolves model name + config + env into a ready-to-use JudgeConfig.

    Construction is cheap and side-effect-free. Use :meth:`resolve` to produce
    the JudgeConfig that the judges will pass into the HTTP layer.
    """

    def __init__(
        self,
        model: str | None = None,
        config: dict | None = None,
        env: dict | None = None,
    ):
        self._env = env if env is not None else os.environ
        self._config = config if config is not None else load_config()
        self._explicit_model = model

    def chosen_model(self) -> str:
        """Resolve which model to use, applying the priority chain."""
        if self._explicit_model:
            return self._explicit_model
        env_val = self._env.get("EVALFORGE_JUDGE_MODEL")
        if env_val:
            return env_val
        cfg_val = self._config.get("judge_model")
        if cfg_val:
            return cfg_val
        return DEFAULT_MODEL

    def resolve(self, model: str | None = None) -> JudgeConfig:
        """Build the JudgeConfig for the chosen (or overridden) model."""
        name = model or self.chosen_model()
        key = _normalize_model(name)

        if key == "deepseek-v4-flash":
            api_key = (
                self._env.get("DEEPSEEK_API_KEY")
                or self._config.get("deepseek_api_key", "")
            )
            base_url = (
                self._env.get("DEEPSEEK_BASE_URL")
                or self._config.get("deepseek_base_url")
                or "https://api.deepseek.com/v1"
            )
            return JudgeConfig(
                model=name,
                provider="deepseek",
                base_url=base_url.rstrip("/"),
                api_key=api_key,
                headers={"Authorization": f"Bearer {api_key}"} if api_key else {},
                api_model_name="deepseek-v4-flash",
            )

        if key == "ollama":
            base_url = (
                self._env.get("OLLAMA_BASE_URL")
                or self._config.get("ollama_base_url")
                or "http://localhost:11434/v1"
            )
            api_model = name.split("/", 1)[1] if "/" in name else "qwen3.5"
            return JudgeConfig(
                model=name,
                provider="ollama",
                base_url=base_url.rstrip("/"),
                api_key="",
                headers={},
                api_model_name=api_model,
            )

        # Default + claude-haiku-4-5 path — original Anthropic-style judge
        api_key = self._env.get("ANTHROPIC_API_KEY", "")
        return JudgeConfig(
            model="claude-haiku-4-5",
            provider="anthropic",
            base_url="https://api.anthropic.com/v1",
            api_key=api_key,
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            api_model_name="claude-haiku-4-5-20251001",
        )

    def build_request(self, cfg: JudgeConfig, prompt: str) -> dict[str, Any]:
        """Return (url, body, headers) shaped for the configured provider."""
        if cfg.is_anthropic():
            return {
                "url": f"{cfg.base_url}/messages",
                "headers": cfg.headers,
                "body": {
                    "model": cfg.api_model_name,
                    "max_tokens": 1024,
                    "messages": [{"role": "user", "content": prompt}],
                },
            }
        # OpenAI-compatible (DeepSeek + Ollama use the same shape)
        return {
            "url": f"{cfg.base_url}/chat/completions",
            "headers": cfg.headers,
            "body": {
                "model": cfg.api_model_name,
                "max_tokens": 1024,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0,
            },
        }

    @staticmethod
    def parse_response(cfg: JudgeConfig, payload: dict) -> str:
        """Extract the assistant text from a provider response body."""
        if cfg.is_anthropic():
            return payload["content"][0]["text"]
        return payload["choices"][0]["message"]["content"]
