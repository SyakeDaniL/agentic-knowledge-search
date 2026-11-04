"""data/docs の Markdown を分割・ベクトル化して Azure AI Search に登録する。

使い方（backend フォルダで）:
    python -m scripts.ingest --docs ../data/docs
"""

from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path

from app.config import get_settings
from app.llm import AzureOpenAILLM

MAX_CHARS = 800


def split_markdown(text: str, max_chars: int = MAX_CHARS) -> list[tuple[str, str]]:
    """見出しごとに分け、長い節はさらに段落で区切る。(見出し, 本文) を返す。"""
    sections: list[tuple[str, str]] = []
    heading = ""
    buf: list[str] = []
    for line in text.splitlines():
        if re.match(r"^#{1,3} ", line):
            if "".join(buf).strip():
                sections.append((heading, "\n".join(buf).strip()))
            heading, buf = line.lstrip("# ").strip(), []
        else:
            buf.append(line)
    if "".join(buf).strip():
        sections.append((heading, "\n".join(buf).strip()))

    chunks: list[tuple[str, str]] = []
    for head, body in sections:
        part = ""
        for para in re.split(r"\n\s*\n", body):
            if part and len(part) + len(para) > max_chars:
                chunks.append((head, part.strip()))
                part = ""
            part += para + "\n\n"
        if part.strip():
            chunks.append((head, part.strip()))
    return chunks


def ensure_index(settings) -> None:
    from azure.core.credentials import AzureKeyCredential
    from azure.search.documents.indexes import SearchIndexClient
    from azure.search.documents.indexes.models import (
        HnswAlgorithmConfiguration,
        SearchableField,
        SearchField,
        SearchFieldDataType,
        SearchIndex,
        SimpleField,
        VectorSearch,
        VectorSearchProfile,
    )

    client = SearchIndexClient(
        settings.search_endpoint, AzureKeyCredential(settings.search_api_key)
    )
    index = SearchIndex(
        name=settings.search_index,
        fields=[
            SimpleField(name="id", type=SearchFieldDataType.String, key=True),
            SearchableField(
                name="title", type=SearchFieldDataType.String, analyzer_name="ja.microsoft"
            ),
            SearchableField(
                name="content", type=SearchFieldDataType.String, analyzer_name="ja.microsoft"
            ),
            SimpleField(name="source", type=SearchFieldDataType.String, filterable=True),
            SearchField(
                name="content_vector",
                type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                searchable=True,
                vector_search_dimensions=settings.embedding_dimensions,
                vector_search_profile_name="default",
            ),
        ],
        vector_search=VectorSearch(
            algorithms=[HnswAlgorithmConfiguration(name="hnsw")],
            profiles=[VectorSearchProfile(name="default", algorithm_configuration_name="hnsw")],
        ),
    )
    client.create_or_update_index(index)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs", default="../data/docs")
    args = parser.parse_args()

    from azure.core.credentials import AzureKeyCredential
    from azure.search.documents import SearchClient

    settings = get_settings()
    llm = AzureOpenAILLM(settings)
    ensure_index(settings)

    docs = []
    for path in sorted(Path(args.docs).glob("*.md")):
        text = path.read_text(encoding="utf-8")
        doc_title = text.splitlines()[0].lstrip("# ").strip() if text else path.stem
        for i, (head, body) in enumerate(split_markdown(text)):
            title = f"{doc_title} / {head}" if head and head != doc_title else doc_title
            chunk_id = hashlib.sha1(f"{path.name}-{i}".encode()).hexdigest()
            docs.append({"id": chunk_id, "title": title, "content": body, "source": path.name})

    print(f"{len(docs)} 件のチャンクをベクトル化します")
    for start in range(0, len(docs), 16):
        batch = docs[start : start + 16]
        vectors = llm.embed([f"{d['title']}\n{d['content']}" for d in batch])
        for d, v in zip(batch, vectors, strict=True):
            d["content_vector"] = v

    client = SearchClient(
        settings.search_endpoint, settings.search_index, AzureKeyCredential(settings.search_api_key)
    )
    client.upload_documents(docs)
    print("登録しました")


if __name__ == "__main__":
    main()
