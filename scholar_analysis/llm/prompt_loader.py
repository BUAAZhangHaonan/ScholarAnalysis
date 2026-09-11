"""Load packaged prompts or an explicitly configured prompt directory."""
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
import yaml

@dataclass
class PromptTemplate:
    name: str
    language: str
    system: str
    user: str
    output_sections: list[str]

_loader_cache = {}

def load_prompt(name, language="en", prompts_dir=""):
    if "/" in name or ".." in name or language not in ("en", "zh"):
        raise ValueError("Unsupported prompt name or language")
    base = Path(prompts_dir) if prompts_dir else files("scholar_analysis").joinpath("prompts")
    # Historical default 'prompts' meant the shipped assets, not a mandatory cwd.
    if str(prompts_dir) == "prompts" and not Path(prompts_dir).exists():
        base = files("scholar_analysis").joinpath("prompts")
    key = (str(base), name, language)
    if key in _loader_cache:
        return _loader_cache[key]
    target = base.joinpath(f"{name}.{language}.yaml")
    if not target.is_file():
        target = base.joinpath(f"{name}.en.yaml")
    data = yaml.safe_load(target.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("system"), str) or not isinstance(data.get("user"), str):
        raise ValueError("Prompt must contain system and user text")
    prompt = PromptTemplate(data.get("name", name), data.get("language", language),
                            data["system"], data["user"], [])
    _loader_cache[key] = prompt
    return prompt
