"""Azure に接続せず、偽の LLM と検索でエージェントの動きを確かめる。"""

from __future__ import annotations

import json

from fastapi.testclient import TestClient

from app.agent import KnowledgeAgent
from app.config import Settings
from app.main import create_app, get_agent
from app.models import Chunk, Usage


class FakeLLM:
    """呼ばれた順に決められた応答を返す。"""

    def __init__(self, replies: list[str]) -> None:
        self.replies = list(replies)
        self.calls: list[list[dict[str, str]]] = []

    def chat(self, messages, *, json_mode=False):
        self.calls.append(messages)
        return self.replies.pop(0), Usage(prompt_tokens=10, completion_tokens=5, llm_calls=1)

    def embed(self, texts):
        return [[0.0] * 3 for _ in texts]


class FakeRetriever:
    def __init__(self, table: dict[str, list[Chunk]]) -> None:
        self.table = table
        self.queries: list[str] = []

    def search(self, query, top_k):
        self.queries.append(query)
        return self.table.get(query, [])[:top_k]


EXPENSE = Chunk(
    id="e1", title="経費精算規程", content="在宅勤務手当は月3,000円。", source="expense.md"
)
SECURITY = Chunk(
    id="s1", title="情報セキュリティ規程", content="私物PCでの業務は禁止。", source="security.md"
)


def test_agent_searches_again_when_evidence_is_missing():
    llm = FakeLLM(
        [
            json.dumps({"queries": ["在宅勤務手当"]}),
            json.dumps(
                {"sufficient": False, "missing": "私物PCの扱い", "next_queries": ["私物PC 業務"]}
            ),
            json.dumps({"sufficient": True, "missing": "", "next_queries": []}),
            "在宅勤務手当は月3,000円です[1]。私物PCでの業務は禁止です[2]。",
        ]
    )
    retriever = FakeRetriever({"在宅勤務手当": [EXPENSE], "私物PC 業務": [SECURITY]})
    agent = KnowledgeAgent(llm, retriever, max_iterations=3)

    res = agent.ask_agent("在宅勤務の手当と、私物PCを使ってよいか教えて")

    assert retriever.queries == ["在宅勤務手当", "私物PC 業務"]
    assert [s.kind for s in res.trace] == ["plan", "search", "judge", "search", "judge", "answer"]
    assert [c.number for c in res.citations] == [1, 2]
    assert res.usage.llm_calls == 4


def test_agent_says_unknown_when_nothing_is_found():
    llm = FakeLLM(
        [
            json.dumps({"queries": ["社員食堂 メニュー"]}),
            json.dumps({"sufficient": False, "missing": "情報なし", "next_queries": []}),
        ]
    )
    agent = KnowledgeAgent(llm, FakeRetriever({}), max_iterations=3)

    res = agent.ask_agent("社員食堂の今日のメニューは？")

    assert res.answer == "社内文書からは確認できませんでした"
    assert res.citations == []


def test_baseline_searches_once_with_the_raw_question():
    llm = FakeLLM(["在宅勤務手当は月3,000円です[1]。"])
    retriever = FakeRetriever({"在宅勤務手当は？": [EXPENSE]})
    agent = KnowledgeAgent(llm, retriever)

    res = agent.ask_baseline("在宅勤務手当は？")

    assert retriever.queries == ["在宅勤務手当は？"]
    assert res.mode == "baseline"
    assert res.citations[0].title == "経費精算規程"


def test_auto_routes_simple_question_to_baseline():
    llm = FakeLLM(
        [
            json.dumps({"route": "simple", "reason": "1つの事実"}),
            "在宅勤務手当は月3,000円です[1]。",
        ]
    )
    retriever = FakeRetriever({"在宅勤務手当は？": [EXPENSE]})
    agent = KnowledgeAgent(llm, retriever)

    res = agent.ask("在宅勤務手当は？", "auto")

    assert res.mode == "auto"
    assert res.route == "simple"
    assert [s.kind for s in res.trace] == ["route", "search", "answer"]
    assert res.usage.llm_calls == 2


def test_auto_routes_complex_question_to_agent():
    llm = FakeLLM(
        [
            json.dumps({"route": "complex", "reason": "論点が2つ"}),
            json.dumps({"queries": ["在宅勤務手当", "私物PC 業務"]}),
            json.dumps({"sufficient": True, "missing": "", "next_queries": []}),
            "手当は月3,000円です[1]。私物PCは禁止です[2]。",
        ]
    )
    retriever = FakeRetriever({"在宅勤務手当": [EXPENSE], "私物PC 業務": [SECURITY]})
    agent = KnowledgeAgent(llm, retriever)

    res = agent.ask("手当と私物PCについて", "auto")

    assert res.route == "complex"
    assert [s.kind for s in res.trace][:2] == ["route", "plan"]


def test_auto_falls_back_to_agent_when_route_is_unreadable():
    llm = FakeLLM(
        [
            "わかりません",
            json.dumps({"queries": ["在宅勤務手当"]}),
            json.dumps({"sufficient": True, "missing": "", "next_queries": []}),
            "月3,000円です[1]。",
        ]
    )
    agent = KnowledgeAgent(llm, FakeRetriever({"在宅勤務手当": [EXPENSE]}))

    assert agent.ask("在宅勤務手当は？", "auto").route == "complex"


def test_api_health_and_rate_limit():
    app = create_app(Settings(daily_request_limit=1))
    llm = FakeLLM(["回答です[1]。"])
    app.dependency_overrides[get_agent] = lambda: KnowledgeAgent(
        llm, FakeRetriever({"質問": [EXPENSE]})
    )
    client = TestClient(app)

    assert client.get("/health").json() == {"status": "ok"}
    first = client.post("/api/ask", json={"question": "質問", "mode": "baseline"})
    assert first.status_code == 200
    second = client.post("/api/ask", json={"question": "質問", "mode": "baseline"})
    assert second.status_code == 429


def test_follow_up_search_results_reach_the_answer_even_when_context_is_full():
    first = [Chunk(id=f"r{i}", title=f"規程{i}", content="…", source=f"r{i}.md") for i in range(10)]
    revision = Chunk(id="rev", title="改定のお知らせ", content="5万円に引き上げ。", source="rev.md")
    llm = FakeLLM(
        [
            json.dumps({"queries": ["備品 補助"]}),
            json.dumps({"sufficient": False, "missing": "改定", "next_queries": ["備品 改定"]}),
            json.dumps({"sufficient": True, "missing": "", "next_queries": []}),
            "5万円です[1]。",
        ]
    )
    retriever = FakeRetriever({"備品 補助": first, "備品 改定": [revision]})
    agent = KnowledgeAgent(llm, retriever, max_iterations=3, top_k=10)

    agent.ask_agent("モニターの補助はいくら？")

    answer_prompt = llm.calls[-1][-1]["content"]
    assert "改定のお知らせ" in answer_prompt
