"""Configuration via environment variables and .env file."""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


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
    mineru_base_url: str = "http://localhost:8888"
    mineru_username: str = ""
    mineru_password: str = ""
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
    deepseek_thinking: bool = True
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


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
