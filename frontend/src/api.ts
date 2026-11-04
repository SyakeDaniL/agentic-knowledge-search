import type { AskResponse, Mode } from "./types";

// 本番では VITE_API_BASE_URL に Container Apps の URL を入れる。ローカルは Vite のプロキシを使う。
const BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "";

export async function ask(question: string, mode: Mode): Promise<AskResponse> {
  const res = await fetch(`${BASE_URL}/api/ask`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, mode }),
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail ?? `エラーが発生しました（${res.status}）`);
  }
  return res.json();
}
