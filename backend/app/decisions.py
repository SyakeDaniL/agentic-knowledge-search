"""Azure OpenAI Decisions API（2026年10月 GA）のクライアント。

文章を生成せず、確率つきの型付きの答え（predicate / choice）だけを返す API。
振り分けと「根拠が足りるか」の判定に使う。出力トークンが課金されず、
確率をしきい値で扱えるのが利点。環境変数で有効化したときだけ使う。

エンドポイント: POST {endpoint}/openai/v1/decisions
対応モデル: gpt-6-luna（リージョンが限られるため、別リソースに置ける設定にしている）
"""

from __future__ import annotations

from typing import Any, Protocol

from .config import Settings
from .models import Usage


class Decider(Protocol):
    def choose(
        self, text: str, *, name: str, instructions: str, choices: list[tuple[str, str]]
    ) -> tuple[dict[str, float], Usage]:
        """choices の各値の確率を返す。"""
        ...

    def probability(self, text: str, *, name: str, instructions: str) -> tuple[float, Usage]:
        """条件が成り立つ確率（0〜1）を返す。"""
        ...


class AzureDecisionsClient:
    def __init__(self, settings: Settings, *, timeout: float = 15.0, transport=None) -> None:
        import httpx

        self._deployment = settings.decisions_deployment
        self._client = httpx.Client(
            base_url=settings.decisions_endpoint.rstrip("/"),
            headers={"api-key": settings.decisions_api_key},
            timeout=timeout,
            transport=transport,  # テストでは httpx.MockTransport を渡す
        )

    def _post(self, text: str, question: dict[str, Any]) -> tuple[dict[str, Any], Usage]:
        res = self._client.post(
            "/openai/v1/decisions",
            json={"model": self._deployment, "input": text, "questions": [question]},
        )
        res.raise_for_status()
        body = res.json()
        usage_raw = body.get("usage") or {}
        usage = Usage(prompt_tokens=int(usage_raw.get("input_tokens", 0)), llm_calls=1)
        answers = body.get("answers") or [{}]
        return answers[0], usage

    def choose(
        self, text: str, *, name: str, instructions: str, choices: list[tuple[str, str]]
    ) -> tuple[dict[str, float], Usage]:
        answer, usage = self._post(
            text,
            {
                "type": "choice",
                "name": name,
                "instructions": instructions,
                "choices": [{"value": v, "description": d} for v, d in choices],
            },
        )
        probs = {
            str(p.get("value")): float(p.get("probability", 0.0))
            for p in answer.get("probabilities", [])
        }
        if not probs and answer.get("choice") is not None:
            probs = {str(answer["choice"]): 1.0}
        return probs, usage

    def probability(self, text: str, *, name: str, instructions: str) -> tuple[float, Usage]:
        answer, usage = self._post(
            text, {"type": "predicate", "name": name, "instructions": instructions}
        )
        return float(answer.get("probability", 0.0)), usage
