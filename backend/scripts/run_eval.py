"""評価セットで、エージェント型RAGと単純なRAGを比べる。

使い方（backend フォルダで）:
    python -m scripts.run_eval --eval ../eval/eval_set.jsonl --out ../eval/results.md

各質問について両方のモードで回答を作り、LLM に「正解と同じ内容か」を判定させて
正答率・平均応答時間・平均トークン数を Markdown の表で出力する。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.llm import parse_json
from app.main import get_agent

JUDGE_PROMPT = """あなたは採点者です。質問・正解・回答を読み、回答が正解と同じ内容を含んでいれば
{"correct": true}、そうでなければ {"correct": false} を JSON で返してください。
正解が「回答できない」の場合は、回答が推測せず確認できないと述べていれば正解です。"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", default="../eval/eval_set.jsonl")
    parser.add_argument("--out", default="../eval/results.md")
    args = parser.parse_args()

    agent = get_agent()
    llm = agent._llm  # 採点にも同じモデルを使う
    items = [
        json.loads(line)
        for line in Path(args.eval).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    stats = {m: {"correct": 0, "latency": 0, "tokens": 0} for m in ("baseline", "agent")}
    rows = []
    for item in items:
        row = {"id": item["id"]}
        for mode in ("baseline", "agent"):
            res = (
                agent.ask_baseline(item["question"])
                if mode == "baseline"
                else agent.ask_agent(item["question"])
            )
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
            ok = bool(parse_json(verdict).get("correct"))
            stats[mode]["correct"] += ok
            stats[mode]["latency"] += res.latency_ms
            stats[mode]["tokens"] += res.usage.prompt_tokens + res.usage.completion_tokens
            row[mode] = "○" if ok else "×"
        rows.append(row)
        print(row)

    n = len(items) or 1
    lines = [
        "| モード | 正答率 | 平均応答時間 | 平均トークン |",
        "| --- | --- | --- | --- |",
    ]
    for mode, label in (("baseline", "単純なRAG"), ("agent", "エージェント型RAG")):
        s = stats[mode]
        accuracy = s["correct"] / n
        latency_sec = s["latency"] / n / 1000
        tokens = s["tokens"] / n
        lines.append(f"| {label} | {accuracy:.0%} | {latency_sec:.1f}秒 | {tokens:,.0f} |")
    lines += ["", "| ID | 単純なRAG | エージェント型RAG |", "| --- | --- | --- |"]
    lines += [f"| {r['id']} | {r['baseline']} | {r['agent']} |" for r in rows]
    Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"結果を {args.out} に書き出しました")


if __name__ == "__main__":
    main()
