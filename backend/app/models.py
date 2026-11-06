"""API とエージェントで共通に使うデータ型。"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class Chunk(BaseModel):
    """検索でヒットした文書の断片。"""

    id: str
    title: str
    content: str
    source: str
    score: float = 0.0


class Citation(BaseModel):
    """回答の根拠として示す出典。"""

    number: int
    title: str
    source: str
    excerpt: str


class TraceStep(BaseModel):
    """画面に表示するエージェントの処理ステップ。"""

    kind: Literal["route", "plan", "search", "judge", "answer"]
    detail: str
    queries: list[str] = Field(default_factory=list)
    hits: int = 0
    elapsed_ms: int = 0


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    llm_calls: int = 0

    def add(self, other: Usage) -> None:
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens
        self.llm_calls += other.llm_calls


Mode = Literal["auto", "agent", "baseline"]
Route = Literal["simple", "complex"]


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=500)
    mode: Mode = "auto"


class AskResponse(BaseModel):
    answer: str
    citations: list[Citation]
    trace: list[TraceStep]
    usage: Usage
    latency_ms: int
    mode: Mode
    route: Route | None = None  # auto モードで振り分けた先
    decision_engine: Literal["llm", "decisions_api"] = "llm"  # 振り分け・判定に使った仕組み
