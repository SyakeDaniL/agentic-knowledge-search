# CLAUDE.md

このリポジトリで Claude Code が作業するときのメモ。

## 構成
- `backend/` FastAPI。エージェントの本体は `app/agent.py`（plan → search → judge → answer）
- `frontend/` React + TypeScript（Vite）
- `infra/` Bicep（Azure OpenAI / AI Search Free / Container Apps / Static Web Apps）
- `data/docs/` 架空の会社 Contoso Japan の社内文書（Markdown）
- `eval/` 評価セットと結果

## よく使うコマンド
- バックエンドのテスト: `cd backend && pytest -q`
- 整形と静的チェック: `cd backend && ruff format . && ruff check .`
- フロントのビルド: `cd frontend && npm run build`
- Bicep の確認: `az bicep build --file infra/main.bicep`

## ルール
- LLM と検索は `LLM` / `Retriever` のプロトコル越しに使う。テストでは偽物に差し替え、Azure に接続しない
- キーやエンドポイントをコードに書かない。`.env`（コミットしない）か環境変数で渡す
- `data/docs/` には実在の会社・個人の情報を入れない
- プロンプトを変えたら `scripts/run_eval.py` で正答率が下がっていないか確かめる
