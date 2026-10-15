<#
.SYNOPSIS
  One script from a fresh clone to a scored result (Windows PowerShell 5.1 or newer).

.DESCRIPTION
  Run it from the repository folder:
    .\setup.ps1              # check tools, install the Python environment, run the tests, score the committed answers
    .\setup.ps1 -Run dev     # ... and pull the models and rerun the dev set (~25 min on a 5 GB GPU)
    .\setup.ps1 -Run heldout # ... and rerun the held-out set (~20 min)
    .\setup.ps1 -Run all     # both
  If uv or Ollama is missing the script stops and tells you how to install it.
  If PowerShell refuses to run scripts:  powershell -ExecutionPolicy Bypass -File .\setup.ps1
#>
param(
    [ValidateSet('check', 'dev', 'heldout', 'all')] [string] $Run = 'check',
    [string] $VisionModel = 'qwen3.5:9b',   # Apple Silicon: qwen3.5:9b-mlx; tiny GPU: qwen3.5:4b
    [int] $Limit = 0                        # smoke test: only the first N instructions (results file is then incomplete)
)
$LimitArgs = @(); if ($Limit -gt 0) { $LimitArgs = @('--limit', $Limit) }
$ErrorActionPreference = 'Stop'
$Apertus = 'hf.co/Colby/apertus-v1.5-8b-text-Q4_K_M-GGUF'
$missing = $false

function Step($text) { Write-Host "`n== $text" -ForegroundColor Cyan }
function Ok($text) { Write-Host "   ok: $text" -ForegroundColor Green }
function Fail($text) { Write-Host "   MISSING: $text" -ForegroundColor Red; $script:missing = $true }

if (-not (Test-Path 'run.py') -or -not (Test-Path 'pyproject.toml')) {
    Write-Host 'Run this script from the repository folder (the one with run.py).' -ForegroundColor Red
    exit 1
}

Step 'Tools'
if (Get-Command uv -ErrorAction SilentlyContinue) {
    Ok "uv $(uv --version)"
} else {
    Fail 'uv (Python project manager; it downloads Python itself)'
    Write-Host @'
   Install uv, then open a new terminal and run this script again:
     winget install --id astral-sh.uv -e
   or, without winget:
     powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
'@
}
if (Get-Command ollama -ErrorAction SilentlyContinue) {
    Ok "ollama $(ollama --version 2>$null)"
} else {
    Fail 'ollama (serves both models locally)'
    Write-Host @'
   Install Ollama, then open a new terminal and run this script again:
     winget install --id Ollama.Ollama -e
   or download the installer from https://ollama.com/download
   (The desktop app starts the server automatically; otherwise run `ollama serve` in a second terminal.)
'@
}
if ($missing) { exit 1 }

Step 'Python environment (uv sync)'
uv sync
if ($LASTEXITCODE -ne 0) { exit 1 }

Step 'Tests without a model'
uv run python tests/test_pipeline.py
if ($LASTEXITCODE -ne 0) { exit 1 }

Step 'Score the committed answers (must print SCRIPT TOTAL 65.0 / 65)'
uv run python starter/evaluate.py results.json --data dataset
if ($LASTEXITCODE -ne 0) { exit 1 }

if ($Run -eq 'check') {
    Write-Host "`nDone. To rerun the pipeline: .\setup.ps1 -Run dev | heldout | all" -ForegroundColor Cyan
    exit 0
}

Step 'Ollama server'
function Serving { try { Invoke-WebRequest -UseBasicParsing -Uri 'http://localhost:11434/api/tags' -TimeoutSec 3 | Out-Null; $true } catch { $false } }
if (-not (Serving)) {
    Write-Host '   not running, starting `ollama serve` in the background'
    Start-Process ollama -ArgumentList 'serve' -WindowStyle Hidden
    $t = 0
    while (-not (Serving) -and $t -lt 30) { Start-Sleep -Seconds 1; $t++ }
}
if (Serving) { Ok 'Ollama answers on localhost:11434' } else {
    Write-Host '   Ollama is not reachable. Start it with `ollama serve` in another terminal and run this script again.' -ForegroundColor Red
    exit 1
}

Step 'Models (pulled only if missing; 12 GB in total, takes 5-30 min depending on the connection)'
$have = (ollama list) -join "`n"
foreach ($m in @($Apertus, $VisionModel)) {
    $short = $m.Split('/')[-1].Split(':')[0]
    if ($have -match [regex]::Escape($short)) { Ok "$m present" } else { Write-Host "   pulling $m"; ollama pull $m; if ($LASTEXITCODE -ne 0) { exit 1 } }
}

if ($Run -in @('dev', 'all')) {
    Step 'Dev set: 10 scenes, 50 instructions (~25 min on a 5 GB GPU)'
    uv run python run.py --data dataset --out results.json --vision-model $VisionModel @LimitArgs
    if ($LASTEXITCODE -ne 0) { exit 1 }
    uv run python starter/evaluate.py results.json --data dataset --report report.json
}
if ($Run -in @('heldout', 'all')) {
    Step 'Held-out set: 5 scenes, 40 instructions (~20 min)'
    uv run python run.py --data heldout --out results_heldout.json --vision-model $VisionModel @LimitArgs
    if ($LASTEXITCODE -ne 0) { exit 1 }
    Write-Host '   results_heldout.json written (no local scoring: the held-out answers are not published)'
}
Write-Host "`nDone." -ForegroundColor Cyan
