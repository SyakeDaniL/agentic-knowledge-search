# 文書の取り込みと評価（検索件数5件・3件）をまとめて実行する（Windows PowerShell 用）。
# 使い方（リポジトリのルートで）:
#   powershell -ExecutionPolicy Bypass -File backend\scripts\run_all.ps1
# 結果は eval\results\<日付>_top5.md と <日付>_top3.md に保存される。

$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

# .env を読み込む（KEY=VALUE の行だけ。# から始まる行と空行は無視）
Get-Content ".env" | ForEach-Object {
    if ($_ -match '^\s*([A-Z_][A-Z0-9_]*)\s*=\s*(.*)\s*$') {
        [Environment]::SetEnvironmentVariable($Matches[1], $Matches[2])
    }
}

$python = ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    Write-Host "仮想環境を作成します"
    python -m venv .venv
}
& $python -m pip install -q -r requirements-dev.txt

$engine = if ($env:AZURE_DECISIONS_DEPLOYMENT) { "decisions" } else { "llm" }
$date = Get-Date -Format "yyyy-MM-dd"
New-Item -ItemType Directory -Force "..\eval\results" | Out-Null

Write-Host "1/3 文書を取り込みます"
& $python -m scripts.ingest --docs ..\data\docs

Write-Host "2/3 評価（検索件数5件）"
& $python -m scripts.run_eval --top-k 5 --out "..\eval\results\${date}_top5_${engine}.md"

Write-Host "3/3 評価（検索件数3件）"
& $python -m scripts.run_eval --top-k 3 --out "..\eval\results\${date}_top3_${engine}.md"

Write-Host "完了: eval\results\${date}_top5_${engine}.md と ${date}_top3_${engine}.md"
