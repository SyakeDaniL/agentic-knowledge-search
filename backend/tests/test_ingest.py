"""取り込み処理のうち、Azure に接続しない部分を確かめる。"""

from __future__ import annotations

from scripts.ingest import link_related


def test_revision_section_gets_text_of_the_article_it_changes():
    docs = [
        {
            "title": "経費精算規程 / 第4条（在宅勤務で使う備品）",
            "content": "モニターなどは3万円まで。",
        },
        {
            "title": "規程改定のお知らせ / 経費精算規程 第4条（在宅勤務で使う備品）の改定",
            "content": "上限を5万円に引き上げ。",
        },
        {"title": "規程改定のお知らせ / 存在しない規程 第9条の改定", "content": "…"},
    ]

    link_related(docs)

    assert docs[0]["related"] == ""
    assert "モニター" in docs[1]["related"]
    assert docs[2]["related"] == ""
