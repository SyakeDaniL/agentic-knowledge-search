"""Decisions API の組み込み・検索の並列化・所要時間の記録のテスト（Azure 不要）。"""

from __future__ import annotations

import json

import httpx

from app.agent import KnowledgeAgent
from app.config import Settings
from app.decisions import AzureDecisionsClient
from app.models import Usage
from tests.test_agent import EXPENSE, SECURITY, FakeLLM, FakeRetriever


class FakeDecider:
    def __init__(self, route_probs: dict[str, float], sufficient: list[float]) -> None:
        self.route_probs = route_probs
        self.sufficient = list(sufficient)
        self.calls: list[str] = []

    def choose(self, text, *, name, instructions, choices):
        self.calls.append(name)
        return self.route_probs, Usage(prompt_tokens=20, llm_calls=1)

    def probability(self, text, *, name, instructions):
        self.calls.append(name)
        return self.sufficient.pop(0), Usage(prompt_tokens=50, llm_calls=1)


class BatchRetriever(FakeRetriever):
    """search_many を持つ検索。まとめて呼ばれたかを記録する。"""

    def __init__(self, table):
        super().__init__(table)
        self.batches: list[list[str]] = []

    def search_many(self, queries, top_k):
        self.batches.append(list(queries))
        return [self.table.get(q, [])[:top_k] for q in queries]


def test_route_uses_decision_probability_and_threshold():
    llm = FakeLLM(["月3,000円です[1]。"])
    decider = FakeDecider({"simple": 0.7, "complex": 0.3}, [])
    agent = KnowledgeAgent(
        llm, FakeRetriever({"手当は？": [EXPENSE]}), decider=decider, route_complex_threshold=0.4
    )

    res = agent.ask("手当は？", "auto")

    assert res.route == "simple"  # 0.3 < 0.4
    assert res.decision_engine == "decisions_api"
    assert "0.30" in res.trace[0].detail
    assert len(llm.calls) == 1  # LLM は回答の1回だけ（振り分けは Decisions API）


def test_low_threshold_sends_borderline_questions_to_agent():
    llm = FakeLLM(
        [
            json.dumps({"queries": ["手当"]}),
            "月3,000円です[1]。",
        ]
    )
    decider = FakeDecider({"simple": 0.55, "complex": 0.45}, [0.9])
    agent = KnowledgeAgent(
        llm, FakeRetriever({"手当": [EXPENSE]}), decider=decider, route_complex_threshold=0.4
    )

    res = agent.ask("手当は？", "auto")

    assert res.route == "complex"  # 0.45 >= 0.4


def test_sufficient_decision_skips_llm_judge():
    llm = FakeLLM(
        [
            json.dumps({"queries": ["在宅勤務手当", "私物PC 業務"]}),
            "手当は月3,000円です[1]。私物PCは禁止です[2]。",
        ]
    )
    decider = FakeDecider({}, [0.92])
    retriever = FakeRetriever({"在宅勤務手当": [EXPENSE], "私物PC 業務": [SECURITY]})
    agent = KnowledgeAgent(llm, retriever, decider=decider)

    res = agent.ask_agent("手当と私物PCは？")

    judge = [s for s in res.trace if s.kind == "judge"]
    assert len(judge) == 1 and "0.92" in judge[0].detail
    assert len(llm.calls) == 2  # plan と answer だけ。judge の生成は省略
    assert decider.calls == ["evidence_sufficient"]


def test_insufficient_decision_falls_back_to_llm_for_next_queries():
    llm = FakeLLM(
        [
            json.dumps({"queries": ["在宅勤務手当"]}),
            json.dumps({"sufficient": False, "missing": "私物PC", "next_queries": ["私物PC 業務"]}),
            "手当は月3,000円です[1]。私物PCは禁止です[2]。",
        ]
    )
    decider = FakeDecider({}, [0.2, 0.95])
    retriever = FakeRetriever({"在宅勤務手当": [EXPENSE], "私物PC 業務": [SECURITY]})
    agent = KnowledgeAgent(llm, retriever, decider=decider)

    res = agent.ask_agent("手当と私物PCは？")

    assert retriever.queries == ["在宅勤務手当", "私物PC 業務"]
    assert [s.kind for s in res.trace].count("judge") == 2


def test_queries_are_searched_in_one_batch_and_steps_are_timed():
    llm = FakeLLM(
        [
            json.dumps({"queries": ["在宅勤務手当", "私物PC 業務"]}),
            json.dumps({"sufficient": True, "missing": "", "next_queries": []}),
            "回答[1][2]。",
        ]
    )
    retriever = BatchRetriever({"在宅勤務手当": [EXPENSE], "私物PC 業務": [SECURITY]})
    agent = KnowledgeAgent(llm, retriever)

    res = agent.ask_agent("手当と私物PCは？")

    assert retriever.batches == [["在宅勤務手当", "私物PC 業務"]]
    assert all(s.elapsed_ms >= 0 for s in res.trace)
    assert res.decision_engine == "llm"


def test_decisions_client_parses_choice_and_predicate():
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert request.url.path == "/openai/v1/decisions"
        assert request.headers["api-key"] == "k"
        assert body["model"] == "luna"
        q = body["questions"][0]
        if q["type"] == "choice":
            assert [c["value"] for c in q["choices"]] == ["simple", "complex"]
            answer = {
                "type": "choice",
                "name": q["name"],
                "choice": "complex",
                "probabilities": [
                    {"value": "simple", "probability": 0.2},
                    {"value": "complex", "probability": 0.8},
                ],
                "confidence": 0.8,
            }
        else:
            answer = {"type": "predicate", "name": q["name"], "probability": 0.65}
        return httpx.Response(
            200,
            json={
                "model": "luna",
                "answers": [answer],
                "usage": {"input_tokens": 42, "output_tokens": 0},
            },
        )

    settings = Settings(
        decisions_endpoint="https://example.openai.azure.com/",
        decisions_api_key="k",
        decisions_deployment="luna",
    )
    client = AzureDecisionsClient(settings, transport=httpx.MockTransport(handler))

    probs, usage = client.choose(
        "質問", name="route", instructions="分類", choices=[("simple", "a"), ("complex", "b")]
    )
    assert probs == {"simple": 0.2, "complex": 0.8}
    assert usage.prompt_tokens == 42 and usage.completion_tokens == 0

    p, _ = client.probability("材料", name="evidence_sufficient", instructions="十分か")
    assert p == 0.65


def test_decisions_enabled_only_when_all_settings_present():
    assert not Settings(decisions_endpoint="https://x", decisions_api_key="k").decisions_enabled
    assert Settings(
        decisions_endpoint="https://x", decisions_api_key="k", decisions_deployment="d"
    ).decisions_enabled
