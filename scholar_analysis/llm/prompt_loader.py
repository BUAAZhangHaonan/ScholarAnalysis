"""Load and cache YAML prompt templates from the prompts/ directory."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

_loader_cache: dict[str, "PromptTemplate"] = {}


@dataclass
class PromptTemplate:
    name: str
    language: str
    system: str
    user: str
    output_sections: list[str]


def load_prompt(
    name: str,
    language: str = "en",
    prompts_dir: str | Path = "prompts",
) -> PromptTemplate:
    """Load a prompt template by name and language.

    Falls back to English if the requested language file is missing.
    """
    cache_key = f"{name}:{language}"
    if cache_key in _loader_cache:
        return _loader_cache[cache_key]

    base = Path(prompts_dir)
    target = base / f"{name}.{language}.yaml"
    fallback = base / f"{name}.en.yaml"

    path = target if target.exists() else fallback
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {target} or {fallback}")

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError(
            f"Prompt file {path} did not parse into a dict (got {type(data).__name__}). "
            f"Check YAML syntax."
        )

    system_text = data.get("system")
    user_text = data.get("user")
    if not system_text or not user_text:
        raise ValueError(
            f"Prompt file {path} is missing 'system' or 'user' text. "
            f"Got keys: {list(data.keys())}"
        )

    template = PromptTemplate(
        name=data.get("name", name),
        language=data.get("language", language),
        system=system_text,
        user=user_text,
        output_sections=data.get("output_format", {}).get("sections", []),
    )

    _loader_cache[cache_key] = template
    logger.info("Loaded prompt %s (%s) from %s (system=%d chars, user=%d chars)", name, language, path, len(template.system), len(template.user))
    return template
