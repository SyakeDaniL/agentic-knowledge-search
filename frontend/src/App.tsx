import { useState, type FormEvent } from "react";
import { ask } from "./api";
import { TraceView } from "./components/TraceView";
import type { AskResponse, Mode } from "./types";

const EXAMPLES = [
  "在宅勤務用にモニターを買いたいです。補助はありますか？私物PCを業務に使ってもいい？",
  "カフェでリモートワークをしながら、社外の生成AIに顧客名を入れてもいいですか？",
  "経費の申請期限はいつまで？",
];

export default function App() {
  const [question, setQuestion] = useState("");
  const [mode, setMode] = useState<Mode>("auto");
  const [result, setResult] = useState<AskResponse | null>(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  async function submit(e: FormEvent) {
    e.preventDefault();
    if (!question.trim()) return;
    setLoading(true);
    setError("");
    setResult(null);
    try {
      setResult(await ask(question.trim(), mode));
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="container">
      <header>
        <h1>Agentic Knowledge Search</h1>
        <p className="lead">
          架空の会社「Contoso Japan」の社内規程を、AIが検索を計画・検索し直してから出典つきで答えます。自動モードでは、簡単な質問は速い単純なRAGで、複雑な質問だけエージェントで答えます。
        </p>
      </header>

      <form onSubmit={submit} className="ask-form">
        <textarea
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder="社内規程について質問してください"
          rows={3}
          maxLength={500}
        />
        <div className="controls">
          <label>
            <input type="radio" checked={mode === "auto"} onChange={() => setMode("auto")} />
            自動（おすすめ）
          </label>
          <label>
            <input type="radio" checked={mode === "agent"} onChange={() => setMode("agent")} />
            エージェント型RAG
          </label>
          <label>
            <input
              type="radio"
              checked={mode === "baseline"}
              onChange={() => setMode("baseline")}
            />
            単純なRAG（比較用）
          </label>
          <button type="submit" disabled={loading || !question.trim()}>
            {loading ? "考え中…" : "質問する"}
          </button>
        </div>
        <div className="examples">
          {EXAMPLES.map((ex) => (
            <button type="button" key={ex} onClick={() => setQuestion(ex)}>
              {ex}
            </button>
          ))}
        </div>
      </form>

      {error && <p className="error">{error}</p>}

      {result && (
        <section className="result">
          <h2>回答</h2>
          <p className="answer">{result.answer}</p>

          {result.citations.length > 0 && (
            <>
              <h3>出典</h3>
              <ul className="citations">
                {result.citations.map((c) => (
                  <li key={c.number}>
                    <strong>
                      [{c.number}] {c.title}
                    </strong>
                    <span className="source">{c.source}</span>
                    <p>{c.excerpt}…</p>
                  </li>
                ))}
              </ul>
            </>
          )}

          <h3>処理の流れ</h3>
          <TraceView steps={result.trace} />

          <p className="meta">
            {(result.latency_ms / 1000).toFixed(1)}秒・LLM呼び出し{result.usage.llm_calls}回・
            {(result.usage.prompt_tokens + result.usage.completion_tokens).toLocaleString()}
            トークン
          </p>
        </section>
      )}
    </main>
  );
}
