"""Configuration via environment variables and .env file."""

from __future__ import annotations

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def _split_csv(value: str) -> list[str]:
    """Split a comma-separated string into a stripped list (empty items preserved)."""
    if not value:
        return []
    return [item.strip() for item in value.split(",")]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="SCHOLAR_ANALYSIS_",
        env_file=".env",
        env_file_encoding="utf-8",
    )

    host: str = "0.0.0.0"
    port: int = 8005
    transport: str = "sse"
    access_token: str = ""
    log_level: str = "INFO"

    # Backend APIs
    arxiv_mirror_base_url: str = "http://127.0.0.1:8900/api/v1"
    arxiv_mirror_data_dir: str = ""

    # MinerU — legacy single-URL fields (kept for backward compatibility)
    mineru_base_url: str | None = None
    mineru_username: str = ""
    mineru_password: str = ""

    # MinerU — multi-endpoint pool (preferred). Comma-separated lists paired by index.
    # Endpoints without auth should have empty username/password at the matching index.
    mineru_endpoints: str = ""
    mineru_usernames: str = ""
    mineru_passwords: str = ""

    http_timeout: float = 600.0

    # Temp files
    temp_dir: str = "/tmp/scholar-analysis"
    request_max_age_seconds: float = 1800.0
    cleanup_interval_seconds: float = 300.0

    # Prompts
    prompts_dir: str = "prompts"
    default_language: str = "en"

    # DeepSeek (primary)
    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com/v1/chat/completions"
    deepseek_model: str = "deepseek-v4-flash"
    deepseek_max_concurrent: int = 3
    deepseek_thinking: bool = False
    deepseek_context_tokens: int = 256000

    # GLM (fallback)
    bigmodel_api_key: str = ""
    bigmodel_base_url: str = "https://open.bigmodel.cn/api/coding/paas/v4/chat/completions"
    bigmodel_model: str = "glm-5-turbo"
    bigmodel_max_concurrent: int = 3

    # Qwen (local fallback)
    qwen_api_key: str = ""
    qwen_base_url: str = "http://localhost:8000/v1/chat/completions"
    qwen_model: str = "Qwen/Qwen3.5-27B-GPTQ-Int4"
    qwen_max_concurrent: int = 5

    # Concurrency
    max_concurrent_pipelines: int = 5
    max_concurrent_parses: int = 3

    # LLM
    response_headroom_tokens: int = 16000

    # Model pool
    model_pool_cooldown_seconds: float = 30.0

    @model_validator(mode="after")
    def _normalise_mineru_endpoints(self) -> "Settings":
        """Resolve mineru_endpoints_list / mineru_creds_list.

        Priority:
          1. mineru_endpoints (comma-separated multi-URL form)
          2. legacy mineru_base_url (single URL, wrapped as a one-element list)

        For (2), credentials come from mineru_username/mineru_password.
        """
        endpoints = _split_csv(self.mineru_endpoints)
        if endpoints:
            usernames = _split_csv(self.mineru_usernames)
            passwords = _split_csv(self.mineru_passwords)
            creds: list[tuple[str, str]] = []
            for i, url in enumerate(endpoints):
                user = usernames[i] if i < len(usernames) else ""
                pwd = passwords[i] if i < len(passwords) else ""
                creds.append((user or "", pwd or ""))
            self.mineru_endpoints_list = list(endpoints)
            self.mineru_creds_list = creds
        elif self.mineru_base_url:
            self.mineru_endpoints_list = [self.mineru_base_url]
            self.mineru_creds_list = [(self.mineru_username or "", self.mineru_password or "")]
        else:
            self.mineru_endpoints_list = []
            self.mineru_creds_list = []
        return self

    # Populated by the model_validator above. Declared here for type hints.
    mineru_endpoints_list: list[str] = []
    mineru_creds_list: list[tuple[str, str]] = []


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
