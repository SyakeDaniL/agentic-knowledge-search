import type { TraceStep } from "../types";

const LABELS: Record<TraceStep["kind"], string> = {
  route: "振分",
  plan: "計画",
  search: "検索",
  judge: "判定",
  answer: "回答",
};

export function TraceView({ steps }: { steps: TraceStep[] }) {
  return (
    <ol className="trace">
      {steps.map((step, i) => (
        <li key={i} className={`trace-step trace-${step.kind}`}>
          <span className="trace-label">{LABELS[step.kind]}</span>
          <span>{step.detail}</span>
          {step.queries.length > 0 && (
            <ul className="trace-queries">
              {step.queries.map((q) => (
                <li key={q}>{q}</li>
              ))}
            </ul>
          )}
        </li>
      ))}
    </ol>
  );
}
