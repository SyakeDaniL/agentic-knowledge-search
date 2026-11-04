"""Azure OpenAI の呼び出しをまとめる。テストでは FakeLLM に差し替える。"""

from __future__ import annotations

import json
from typing import Any, Protocol

from .config import Settings
from .models import Usage


class LLM(Protocol):
    def chat(
        self, messages: list[dict[str, str]], *, json_mode: bool = False
    ) -> tuple[str, Usage]: ...

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class AzureOpenAILLM:
    def __init__(self, settings: Settings) -> None:
        from openai import AzureOpenAI

        self._settings = settings
        self._client = AzureOpenAI(
            azure_endpoint=settings.openai_endpoint,
            api_key=settings.openai_api_key,
            api_version=settings.openai_api_version,
        )

    def chat(self, messages: list[dict[str, str]], *, json_mode: bool = False) -> tuple[str, Usage]:
        kwargs: dict[str, Any] = {
            "model": self._settings.chat_deployment,
            "messages": messages,
            "temperature": 0,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        res = self._client.chat.completions.create(**kwargs)
        usage = Usage(
            prompt_tokens=res.usage.prompt_tokens if res.usage else 0,
            completion_tokens=res.usage.completion_tokens if res.usage else 0,
            llm_calls=1,
        )
        return res.choices[0].message.content or "", usage

    def embed(self, texts: list[str]) -> list[list[float]]:
        res = self._client.embeddings.create(
            model=self._settings.embedding_deployment,
            input=texts,
            dimensions=self._settings.embedding_dimensions,
        )
        return [d.embedding for d in res.data]


def parse_json(text: str) -> dict[str, Any]:
    """LLM の JSON 出力を読む。壊れていたら空の dict を返す。"""
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            return {}
        try:
            data = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return {}
    return data if isinstance(data, dict) else {}
