"""Azure AI Search でのハイブリッド検索（キーワード＋ベクトル）。"""

from __future__ import annotations

from typing import Protocol

from .config import Settings
from .llm import LLM
from .models import Chunk


class Retriever(Protocol):
    def search(self, query: str, top_k: int) -> list[Chunk]: ...


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
        from azure.search.documents.models import VectorizedQuery

        vector = self._llm.embed([query])[0]
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
