"""評価セットで、単純なRAG・エージェント型RAG・自動モードを比べる。

使い方（backend フォルダで）:
    python -m scripts.run_eval
    python -m scripts.run_eval --top-k 3            # 検索で取る件数を絞って難しくする
    python -m scripts.run_eval --modes baseline,auto --limit 10

測るもの:
- 正答率: LLM に「回答が正解と同じ内容か」を採点させる
- 出典の正しさ: 評価セットの sources に書いた文書を、回答の出典に含めているか
  （"a.md|b.md" はどちらか一方でよい。答えがない質問は対象外）
- 平均応答時間・平均トークン数
- 自動モードが、どれだけの質問をエージェントに回したか
- 段階ごと（振り分け・計画・検索・判定・回答）の平均時間

Decisions API の環境変数を設定していれば、振り分けと判定はそちらで行われる（結果の先頭に表示）。
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from app.llm import parse_json

MODES = ("baseline", "agent", "auto")
MODE_LABELS = {"baseline": "単純なRAG", "agent": "エージェント型RAG", "auto": "自動（振り分け）"}
STEP_LABELS = {
    "route": "振り分け",
    "plan": "計画",
    "search": "検索",
    "judge": "判定",
    "answer": "回答",
}
TYPE_LABELS = {
    "single": "1つの文書",
    "multi_hop": "複数の文書",
    "revision": "改定の反映",
    "hard": "惑わせる文書あり",
    "unanswerable": "答えがない",
}

JUDGE_PROMPT = """あなたは採点者です。質問・正解・回答を読み、回答が正解と同じ内容を含んでいれば
{"correct": true}、そうでなければ {"correct": false} を JSON で返してください。
正解が「回答できない」の場合は、回答が推測せず確認できないと述べていれば正解です。
正解に含まれない余分な説明があっても、正解と矛盾しなければ正解とします。"""


@dataclass
class Record:
    id: str
    type: str
    mode: str
    correct: bool
    citation_ok: bool | None  # None は採点の対象外
    latency_ms: int
    tokens: int
    route: str | None
    step_ms: dict[str, int] = field(default_factory=dict)  # 段階ごとの合計時間


def citation_ok(required: list[str], cited_sources: list[str]) -> bool | None:
    """必要な出典がすべて含まれているか。required が空なら対象外（None）。"""
    if not required:
        return None
    cited = set(cited_sources)
    return all(any(alt in cited for alt in req.split("|")) for req in required)


def _pct(values: list[bool]) -> str:
    return f"{sum(values) / len(values):.0%}（{sum(values)}/{len(values)}）" if values else "-"


def summarize(records: list[Record], modes: list[str]) -> str:
    by_mode: dict[str, list[Record]] = defaultdict(list)
    for r in records:
        by_mode[r.mode].append(r)

    lines = [
        "## 全体",
        "",
        "| モード | 正答率 | 出典の正しさ | 平均応答時間 | 平均トークン |",
        "| --- | --- | --- | --- | --- |",
    ]
    for m in modes:
        rs = by_mode[m]
        if not rs:
            continue
        cites = [r.citation_ok for r in rs if r.citation_ok is not None]
        latency = sum(r.latency_ms for r in rs) / len(rs) / 1000
        tokens = sum(r.tokens for r in rs) / len(rs)
        lines.append(
            f"| {MODE_LABELS[m]} | {_pct([r.correct for r in rs])} | {_pct(cites)} "
            f"| {latency:.1f}秒 | {tokens:,.0f} |"
        )

    types = [t for t in TYPE_LABELS if any(r.type == t for r in records)]
    lines += [
        "",
        "## 質問の種類ごとの正答率",
        "",
        "| 種類 | " + " | ".join(MODE_LABELS[m] for m in modes) + " |",
        "| --- |" + " --- |" * len(modes),
    ]
    for t in types:
        cells = [_pct([r.correct for r in by_mode[m] if r.type == t]) for m in modes]
        lines.append(f"| {TYPE_LABELS[t]} | " + " | ".join(cells) + " |")

    lines += [
        "",
        "## 段階ごとの平均時間（その段階を通った質問だけで平均）",
        "",
        "| モード | " + " | ".join(STEP_LABELS.values()) + " |",
        "| --- |" + " --- |" * len(STEP_LABELS),
    ]
    for m in modes:
        cells = []
        for kind in STEP_LABELS:
            values = [r.step_ms[kind] for r in by_mode[m] if kind in r.step_ms]
            cells.append(f"{sum(values) / len(values) / 1000:.2f}秒" if values else "-")
        lines.append(f"| {MODE_LABELS[m]} | " + " | ".join(cells) + " |")

    auto = by_mode.get("auto", [])
    if auto:
        complex_n = sum(r.route == "complex" for r in auto)
        lines += [
            "",
            f"自動モードは {len(auto)} 問中 {complex_n} 問をエージェントに、"
            f"{len(auto) - complex_n} 問を単純なRAGに振り分けた。",
        ]

    ids = sorted({r.id for r in records})
    index = {(r.id, r.mode): r for r in records}
    lines += [
        "",
        "## 質問ごとの結果（○正解 ×不正解、括弧内は出典）",
        "",
        "| ID | 種類 | " + " | ".join(MODE_LABELS[m] for m in modes) + " |",
        "| --- | --- |" + " --- |" * len(modes),
    ]
    for qid in ids:
        cells = []
        qtype = ""
        for m in modes:
            r = index.get((qid, m))
            if r is None:
                cells.append("-")
                continue
            qtype = TYPE_LABELS.get(r.type, r.type)
            mark = "○" if r.correct else "×"
            if r.citation_ok is not None:
                mark += "（○）" if r.citation_ok else "（×）"
            if r.route:
                mark += f" {r.route}"
            cells.append(mark)
        lines.append(f"| {qid} | {qtype} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def _step_ms(trace) -> dict[str, int]:
    totals: dict[str, int] = {}
    for step in trace:
        totals[step.kind] = totals.get(step.kind, 0) + step.elapsed_ms
    return totals


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", default="../eval/eval_set.jsonl")
    parser.add_argument("--out", default="../eval/results.md")
    parser.add_argument("--modes", default=",".join(MODES))
    parser.add_argument("--top-k", type=int, default=None, help="検索で取る件数を上書きする")
    parser.add_argument("--limit", type=int, default=None, help="先頭から n 問だけ実行する")
    args = parser.parse_args()

    from app.main import get_agent

    modes = [m.strip() for m in args.modes.split(",") if m.strip() in MODES]
    agent = get_agent()
    if args.top_k:
        agent._top_k = args.top_k
    llm = agent._llm  # 採点にも同じモデルを使う

    items = [
        json.loads(line)
        for line in Path(args.eval).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ][: args.limit]

    records: list[Record] = []
    for item in items:
        for mode in modes:
            res = agent.ask(item["question"], mode)
            verdict, _ = llm.chat(
                [
                    {"role": "system", "content": JUDGE_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            f"質問: {item['question']}\n正解: {item['answer']}\n回答: {res.answer}"
                        ),
                    },
                ],
                json_mode=True,
            )
            record = Record(
                id=item["id"],
                type=item.get("type", "single"),
                mode=mode,
                correct=bool(parse_json(verdict).get("correct")),
                citation_ok=citation_ok(item.get("sources", []), [c.source for c in res.citations]),
                latency_ms=res.latency_ms,
                tokens=res.usage.prompt_tokens + res.usage.completion_tokens,
                route=res.route,
                step_ms=_step_ms(res.trace),
            )
            records.append(record)
            print(record)

    engine = "Decisions API" if agent.decision_engine == "decisions_api" else "LLM（JSON出力）"
    header = (
        f"# 評価結果\n\n問題数: {len(items)}、検索件数（top_k）: {agent._top_k}、"
        f"振り分け・判定: {engine}\n\n"
    )
    Path(args.out).write_text(header + summarize(records, modes), encoding="utf-8")
    print(f"結果を {args.out} に書き出しました")


if __name__ == "__main__":
    main()
