export type Mode = "auto" | "agent" | "baseline";

export interface Citation {
  number: number;
  title: string;
  source: string;
  excerpt: string;
}

export interface TraceStep {
  kind: "route" | "plan" | "search" | "judge" | "answer";
  detail: string;
  queries: string[];
  hits: number;
  elapsed_ms: number;
}

export interface Usage {
  prompt_tokens: number;
  completion_tokens: number;
  llm_calls: number;
}

export interface AskResponse {
  answer: string;
  citations: Citation[];
  trace: TraceStep[];
  usage: Usage;
  latency_ms: number;
  mode: Mode;
  route: "simple" | "complex" | null;
  decision_engine: "llm" | "decisions_api";
}
