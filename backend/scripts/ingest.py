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
# 「経費精算規程 第4条（…）の改定」のように、別の規程の条文を指す見出し
ARTICLE_REF = re.compile(r"^(\S+規程) (第\d+条)")


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


def link_related(docs: list[dict]) -> None:
    """改定のお知らせなどの各節に、参照先の条文の本文を検索用の文脈として付ける。

    改定のお知らせは「上限を3万円から5万円に」のように変更点しか書かないため、
    条文側の言葉（モニター・キーボードなど）で検索すると上位に出てこない。
    related は検索とベクトル化にだけ使い、LLM には渡さない。
    """
    articles: dict[tuple[str, str], str] = {}
    for d in docs:
        doc_title, _, head = d["title"].partition(" / ")
        m = re.match(r"^(第\d+条)", head)
        if m:
            articles[(doc_title, m.group(1))] = f"{head}\n{d['content']}"
    for d in docs:
        _, _, head = d["title"].partition(" / ")
        m = ARTICLE_REF.match(head)
        d["related"] = articles.get((m.group(1), m.group(2)), "") if m else ""


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
            SearchableField(
                name="related", type=SearchFieldDataType.String, analyzer_name="ja.microsoft"
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

    link_related(docs)
    linked = sum(1 for d in docs if d["related"])
    print(f"{len(docs)} 件のチャンクをベクトル化します（関連条文つき: {linked} 件）")
    for start in range(0, len(docs), 16):
        batch = docs[start : start + 16]
        vectors = llm.embed([f"{d['title']}\n{d['content']}\n{d['related']}" for d in batch])
        for d, v in zip(batch, vectors, strict=True):
            d["content_vector"] = v

    client = SearchClient(
        settings.search_endpoint, settings.search_index, AzureKeyCredential(settings.search_api_key)
    )
    client.upload_documents(docs)
    print("登録しました")


if __name__ == "__main__":
    main()
