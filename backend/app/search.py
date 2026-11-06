"""Azure AI Search でのハイブリッド検索（キーワード＋ベクトル）。"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Protocol

from .config import Settings
from .llm import LLM
from .models import Chunk

MAX_PARALLEL_SEARCHES = 5


class Retriever(Protocol):
    def search(self, query: str, top_k: int) -> list[Chunk]: ...


def search_all(retriever: Retriever, queries: list[str], top_k: int) -> list[list[Chunk]]:
    """複数クエリを検索し、クエリと同じ順で結果を返す。

    retriever が search_many を持っていればそれを使う（まとめてベクトル化＋並列検索）。
    持っていなければ（テスト用の偽物など）1件ずつ検索する。
    """
    if not queries:
        return []
    many = getattr(retriever, "search_many", None)
    if callable(many):
        return many(queries, top_k)
    return [retriever.search(q, top_k) for q in queries]


class AzureSearchRetriever:
    def __init__(self, settings: Settings, llm: LLM) -> None:
        from azure.core.credentials import AzureKeyCredential
        from azure.search.documents import SearchClient

        self._llm = llm
        self._client = SearchClient(
            endpoint=settings.search_endpoint,
            index_name=settings.search_index,
            credential=AzureKeyCredential(settings.search_api_key),
        )

    def search(self, query: str, top_k: int) -> list[Chunk]:
        return self.search_many([query], top_k)[0]

    def search_many(self, queries: list[str], top_k: int) -> list[list[Chunk]]:
        # ベクトル化は1回の呼び出しにまとめ、検索はスレッドで並列に実行する
        vectors = self._llm.embed(queries)
        workers = min(len(queries), MAX_PARALLEL_SEARCHES)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            return list(
                pool.map(
                    lambda qv: self._hybrid(qv[0], qv[1], top_k), zip(queries, vectors, strict=True)
                )
            )

    def _hybrid(self, query: str, vector: list[float], top_k: int) -> list[Chunk]:
        from azure.search.documents.models import VectorizedQuery

        results = self._client.search(
            search_text=query,
            vector_queries=[
                VectorizedQuery(vector=vector, k_nearest_neighbors=top_k, fields="content_vector")
            ],
            select=["id", "title", "content", "source"],
            top=top_k,
        )
        return [
            Chunk(
                id=r["id"],
                title=r["title"],
                content=r["content"],
                source=r["source"],
                score=float(r.get("@search.score", 0.0)),
            )
            for r in results
        ]
