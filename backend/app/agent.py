"""エージェント型RAG、比較用の単純なRAG、その2つを振り分ける自動モード。

エージェントの流れ:
  1. plan   : 質問を分解して検索クエリを作る
  2. search : 各クエリでハイブリッド検索する
  3. judge  : 根拠が足りるか判定し、足りなければ 2 に戻る（最大 max_iterations 回）
  4. answer : 集めた文書だけを根拠に、出典つきで回答する

自動モード（auto）は、最初に質問を simple / complex に分類し、
simple は単純なRAG、complex はエージェントで答える。精度と速度・費用の両立が目的。
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

from . import prompts
from .llm import LLM, parse_json
from .models import AskResponse, Chunk, Citation, TraceStep, Usage
from .search import Retriever

MAX_CONTEXT_CHUNKS = 12


@dataclass
class _State:
    chunks: list[Chunk]
    trace: list[TraceStep]
    usage: Usage
    seen_queries: set[str]
    hits_by_query: list[list[Chunk]] = field(default_factory=list)


class KnowledgeAgent:
    def __init__(
        self, llm: LLM, retriever: Retriever, *, max_iterations: int = 3, top_k: int = 5
    ) -> None:
        self._llm = llm
        self._retriever = retriever
        self._max_iterations = max_iterations
        self._top_k = top_k

    # ---- 公開メソッド ----

    def ask(self, question: str, mode: str = "auto") -> AskResponse:
        if mode == "baseline":
            return self.ask_baseline(question)
        if mode == "agent":
            return self.ask_agent(question)
        return self.ask_auto(question)

    def ask_auto(self, question: str) -> AskResponse:
        """質問を振り分けてから、単純なRAGかエージェントで答える。"""
        started = time.perf_counter()
        route, reason, usage = self._route(question)
        res = self.ask_agent(question) if route == "complex" else self.ask_baseline(question)
        label = "複雑な質問 → エージェント" if route == "complex" else "単純な質問 → 単純なRAG"
        res.trace.insert(0, TraceStep(kind="route", detail=f"{label}（{reason}）"))
        res.usage.add(usage)
        res.mode = "auto"
        res.route = route
        res.latency_ms = int((time.perf_counter() - started) * 1000)
        return res

    def ask_agent(self, question: str) -> AskResponse:
        started = time.perf_counter()
        state = _State(chunks=[], trace=[], usage=Usage(), seen_queries=set())

        queries = self._plan(question, state)
        for iteration in range(self._max_iterations):
            self._search(queries, state)
            if iteration == self._max_iterations - 1:
                break
            sufficient, next_queries = self._judge(question, state)
            if sufficient or not next_queries:
                break
            queries = next_queries

        answer, citations = self._answer(question, state)
        return AskResponse(
            answer=answer,
            citations=citations,
            trace=state.trace,
            usage=state.usage,
            latency_ms=int((time.perf_counter() - started) * 1000),
            mode="agent",
        )

    def ask_baseline(self, question: str) -> AskResponse:
        """比較用：質問そのままで1回だけ検索して答える。"""
        started = time.perf_counter()
        state = _State(chunks=[], trace=[], usage=Usage(), seen_queries=set())
        self._search([question], state)
        answer, citations = self._answer(question, state)
        return AskResponse(
            answer=answer,
            citations=citations,
            trace=state.trace,
            usage=state.usage,
            latency_ms=int((time.perf_counter() - started) * 1000),
            mode="baseline",
        )

    # ---- 各ステップ ----

    def _route(self, question: str) -> tuple[str, str, Usage]:
        text, usage = self._llm.chat(
            [
                {"role": "system", "content": prompts.ROUTE_SYSTEM},
                {"role": "user", "content": question},
            ],
            json_mode=True,
        )
        data = parse_json(text)
        # 判定に失敗したら、取りこぼしを避けるためエージェントに回す
        route = "simple" if data.get("route") == "simple" else "complex"
        reason = str(data.get("reason", "")).strip() or "判定なし"
        return route, reason, usage

    def _plan(self, question: str, state: _State) -> list[str]:
        text, usage = self._llm.chat(
            [
                {"role": "system", "content": prompts.PLAN_SYSTEM},
                {"role": "user", "content": question},
            ],
            json_mode=True,
        )
        state.usage.add(usage)
        queries = _clean_queries(parse_json(text).get("queries")) or [question]
        state.trace.append(
            TraceStep(kind="plan", detail="質問を分解して検索を計画", queries=queries)
        )
        return queries

    def _search(self, queries: list[str], state: _State) -> None:
        new_queries = [q for q in queries if q not in state.seen_queries]
        known_ids = {c.id for c in state.chunks}
        added = 0
        for q in new_queries:
            state.seen_queries.add(q)
            hits = self._retriever.search(q, self._top_k)
            state.hits_by_query.append(hits)
            for chunk in hits:
                if chunk.id not in known_ids:
                    state.chunks.append(chunk)
                    known_ids.add(chunk.id)
                    added += 1
        state.trace.append(
            TraceStep(
                kind="search",
                detail=f"ハイブリッド検索で新しい文書を{added}件取得",
                queries=new_queries,
                hits=added,
            )
        )

    def _judge(self, question: str, state: _State) -> tuple[bool, list[str]]:
        text, usage = self._llm.chat(
            [
                {"role": "system", "content": prompts.JUDGE_SYSTEM},
                {
                    "role": "user",
                    "content": f"質問: {question}\n\n文書:\n"
                    + prompts.format_context(_select_context(state)),
                },
            ],
            json_mode=True,
        )
        state.usage.add(usage)
        data = parse_json(text)
        sufficient = bool(data.get("sufficient", True))
        next_queries = _clean_queries(data.get("next_queries"))
        detail = "根拠は十分" if sufficient else f"不足: {data.get('missing', '')}".strip()
        state.trace.append(
            TraceStep(
                kind="judge",
                detail=detail,
                queries=[] if sufficient else next_queries,
            )
        )
        return sufficient, next_queries

    def _answer(self, question: str, state: _State) -> tuple[str, list[Citation]]:
        context_chunks = _select_context(state)
        if not context_chunks:
            state.trace.append(TraceStep(kind="answer", detail="根拠なし"))
            return "社内文書からは確認できませんでした", []

        text, usage = self._llm.chat(
            [
                {"role": "system", "content": prompts.ANSWER_SYSTEM},
                {
                    "role": "user",
                    "content": f"文書:\n{prompts.format_context(context_chunks)}"
                    f"\n\n質問: {question}",
                },
            ],
        )
        state.usage.add(usage)
        citations = _citations_from(text, context_chunks)
        state.trace.append(TraceStep(kind="answer", detail=f"出典{len(citations)}件で回答を作成"))
        return text.strip(), citations


def _select_context(state: _State) -> list[Chunk]:
    """各クエリの上位から1件ずつ順に取り、上限まで集める。

    先頭から切るだけだと、後から追加で検索した結果（改定のお知らせなど）が
    上限からあふれて judge にも answer にも渡らないため。
    """
    picked: list[Chunk] = []
    seen: set[str] = set()
    depth = max((len(h) for h in state.hits_by_query), default=0)
    for rank in range(depth):
        for hits in state.hits_by_query:
            if rank < len(hits) and hits[rank].id not in seen:
                picked.append(hits[rank])
                seen.add(hits[rank].id)
                if len(picked) == MAX_CONTEXT_CHUNKS:
                    return picked
    return picked


def _clean_queries(raw: object) -> list[str]:
    if not isinstance(raw, list):
        return []
    out: list[str] = []
    for q in raw:
        if isinstance(q, str) and q.strip() and q.strip() not in out:
            out.append(q.strip())
    return out[:3]


def _citations_from(answer: str, chunks: list[Chunk]) -> list[Citation]:
    numbers = sorted({int(n) for n in re.findall(r"\[(\d+)\]", answer)})
    citations = []
    for n in numbers:
        if 1 <= n <= len(chunks):
            c = chunks[n - 1]
            citations.append(
                Citation(number=n, title=c.title, source=c.source, excerpt=c.content[:160])
            )
    return citations
