# Agentic Knowledge Search

社内文書を、AIが**検索を計画し、足りなければ検索し直してから**出典つきで答える「エージェント型RAG」のデモです。
題材は架空の会社「Contoso Japan 株式会社」の社内規程・FAQ です。

> 🚧 開発中（2026年12月公開予定）。デモURL・評価結果・動画は公開時に追記します。

## 特徴

| | 単純なRAG | このプロジェクト（エージェント型RAG） |
| --- | --- | --- |
| 検索 | 質問をそのまま1回だけ検索 | 質問を論点に分解して検索を計画 |
| 足りないとき | そのまま答える（推測が混ざる） | 根拠が足りるか判定し、足りなければ検索し直す |
| 検索方式 | ベクトル検索のみが多い | キーワード＋ベクトルのハイブリッド検索 |
| 根拠 | 示さないことが多い | 出典番号つきで回答、根拠がなければ「確認できない」 |
| 見える化 | なし | 計画・検索・判定の流れを画面に表示 |
| 品質 | 感覚で確認 | 評価セットで正答率・速度・トークン数を比較 |

## 構成

```
ユーザー → React（Static Web Apps）→ FastAPI（Container Apps）
                                        ├─ Azure OpenAI（gpt-4.1-mini / text-embedding-3-small）
                                        └─ Azure AI Search（ハイブリッド検索）
架空の社内文書 → 取り込みスクリプト（分割→ベクトル化→登録）→ Azure AI Search
GitHub Actions ＋ Bicep で自動デプロイ
```

エージェントの流れ（`backend/app/agent.py`）:

1. **plan** 質問を1〜3個の検索クエリに分解する
2. **search** 各クエリでハイブリッド検索し、新しい文書だけを集める
3. **judge** 根拠が足りるか判定し、足りなければ新しいクエリで 2 に戻る（最大3回）
4. **answer** 集めた文書だけを根拠に、出典番号つきで答える

## 評価

`eval/eval_set.jsonl` の30問で、単純なRAGと比べます。質問は4種類です。

| 種類 | 問題数 | 例 |
| --- | --- | --- |
| 1つの文書で答えられる | 17 | 経費の申請期限はいつまで？ |
| 複数の文書をまたぐ | 8 | カフェで作業しながら、社外の生成AIに顧客名を入れてもいい？ |
| 規程の改定を反映する必要がある | 2 | 在宅勤務手当はいくら？（2026年10月に増額） |
| 文書に答えがない | 3 | 退職金はどうやって計算される？ |

| モード | 正答率 | 平均応答時間 | 平均トークン |
| --- | --- | --- | --- |
| 単純なRAG | 97%（29/30） | 3.1秒 | 529 |
| エージェント型RAG | 100%（30/30） | 8.7秒 | 1,381 |

gpt-4.1-mini（Azure OpenAI, japaneast）で2026年11月に計測。差が出たのは q03（在宅勤務の備品補助と私物PC）で、単純なRAGは2026年10月の改定のお知らせを拾えず、改定前の金額（3万円）で答えた。
採点も同じモデルで行っているため、数値は目安です（`python -m scripts.run_eval` で再現できます）。

## ローカルで動かす

必要なもの: Python 3.11+、Node.js 20+、Azure OpenAI と Azure AI Search のリソース

```bash
# 1. バックエンド
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env   # キーとエンドポイントを記入
set -a && source .env && set +a

# 2. 文書を取り込む（インデックス作成→ベクトル化→登録）
python -m scripts.ingest --docs ../data/docs

# 3. API を起動
uvicorn app.main:app --reload

# 4. 別のターミナルでフロントを起動 → http://localhost:5173
cd frontend && npm install && npm run dev
```

テスト（Azure 不要）: `cd backend && pytest -q`

評価: `cd backend && python -m scripts.run_eval`

## デプロイ

1. Azure にリソースグループを作る（例: `az group create -n rg-aks-demo -l japaneast`）
2. GitHub Actions 用に OIDC でログインできるアプリ登録を作り、リポジトリの Secrets に
   `AZURE_CLIENT_ID` `AZURE_TENANT_ID` `AZURE_SUBSCRIPTION_ID` を、Variables に `AZURE_RESOURCE_GROUP` を登録する
3. Actions の「Deploy」を手動実行する
4. 初回は ghcr.io のパッケージ `agentic-knowledge-search-api` を Public にする（Container Apps が取得できるように）
5. 文書の取り込みはローカルから `python -m scripts.ingest` を実行する

## 費用

文書約100件・質問月1,000回の想定で **月600円前後** です（2026年11月時点の料金）。

- Azure OpenAI gpt-4.1-mini：入力 0.40ドル／出力 1.60ドル（100万トークンあたり）
- Azure AI Search：Free（50MB・インデックス3つまで）
- Container Apps：毎月の無料枠内（使わないときは0台に縮小）
- Static Web Apps：Free

公開デモは IP ごとに 1日30回までに制限しています。本番で使うなら AI Search は Basic 以上、
回数制限は Redis などの共有ストアに置き換えます。

## 開発の進め方

設計・実装・レビューに Claude Code を使っています。リポジトリのルールは `CLAUDE.md` にまとめています。

## ライセンス

MIT。`data/docs/` の文書はすべて架空のもので、実在の会社・人物とは関係ありません。
