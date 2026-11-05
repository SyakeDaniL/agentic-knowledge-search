"""評価スクリプトの採点・集計のテスト（Azure 不要）。"""

from __future__ import annotations

import json
from pathlib import Path

from scripts.run_eval import Record, citation_ok, summarize

EVAL_SET = Path(__file__).resolve().parents[2] / "eval" / "eval_set.jsonl"
DOCS = Path(__file__).resolve().parents[2] / "data" / "docs"


def test_citation_ok_handles_alternatives_and_unanswerable():
    assert citation_ok(["a.md", "b.md|c.md"], ["a.md", "c.md"]) is True
    assert citation_ok(["a.md", "b.md|c.md"], ["a.md"]) is False
    assert citation_ok([], ["a.md"]) is None


def test_summarize_reports_each_mode_and_route_split():
    records = [
        Record("q1", "single", "baseline", True, True, 1000, 100, None),
        Record("q1", "single", "auto", True, True, 1100, 120, "simple"),
        Record("q2", "hard", "baseline", False, False, 1000, 100, None),
        Record("q2", "hard", "auto", True, True, 5000, 900, "complex"),
    ]
    text = summarize(records, ["baseline", "auto"])

    assert "| 単純なRAG | 50%（1/2） | 50%（1/2） | 1.0秒 | 100 |" in text
    assert "| 自動（振り分け） | 100%（2/2）" in text
    assert "2 問中 1 問をエージェントに" in text
    assert "| q2 | 惑わせる文書あり | ×（×） | ○（○） complex |" in text


def test_eval_set_sources_point_to_existing_docs():
    doc_names = {p.name for p in DOCS.glob("*.md")}
    items = [json.loads(line) for line in EVAL_SET.read_text(encoding="utf-8").splitlines()]
    assert len({i["id"] for i in items}) == len(items)
    for item in items:
        for req in item["sources"]:
            for alt in req.split("|"):
                assert alt in doc_names, f"{item['id']}: {alt} がありません"
        if item["type"] == "unanswerable":
            assert item["sources"] == []
