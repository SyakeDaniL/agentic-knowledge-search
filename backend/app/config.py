"""環境変数から設定を読み込む。"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default)


@dataclass(frozen=True)
class Settings:
    # Azure OpenAI
    openai_endpoint: str = field(default_factory=lambda: _env("AZURE_OPENAI_ENDPOINT"))
    openai_api_key: str = field(default_factory=lambda: _env("AZURE_OPENAI_API_KEY"))
    openai_api_version: str = field(
        default_factory=lambda: _env("AZURE_OPENAI_API_VERSION", "2024-10-21")
    )
    chat_deployment: str = field(
        default_factory=lambda: _env("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-4.1-mini")
    )
    embedding_deployment: str = field(
        default_factory=lambda: _env("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-small")
    )
    embedding_dimensions: int = field(
        default_factory=lambda: int(_env("AZURE_OPENAI_EMBEDDING_DIMENSIONS", "1536"))
    )

    # Azure AI Search
    search_endpoint: str = field(default_factory=lambda: _env("AZURE_SEARCH_ENDPOINT"))
    search_api_key: str = field(default_factory=lambda: _env("AZURE_SEARCH_API_KEY"))
    search_index: str = field(default_factory=lambda: _env("AZURE_SEARCH_INDEX", "knowledge"))

    # エージェント
    max_iterations: int = field(default_factory=lambda: int(_env("AGENT_MAX_ITERATIONS", "3")))
    top_k: int = field(default_factory=lambda: int(_env("SEARCH_TOP_K", "5")))

    # 公開デモの保護
    daily_request_limit: int = field(default_factory=lambda: int(_env("DAILY_REQUEST_LIMIT", "30")))
    allowed_origins: tuple[str, ...] = field(
        default_factory=lambda: tuple(
            o.strip()
            for o in _env("ALLOWED_ORIGINS", "http://localhost:5173").split(",")
            if o.strip()
        )
    )


def get_settings() -> Settings:
    return Settings()
