#!/usr/bin/env bash
# One script from a fresh clone to a scored result (macOS, Linux; also Git Bash on Windows).
# Run it from the repository folder:
#   ./setup.sh              # check tools, install the Python environment, run the tests, score the committed answers
#   ./setup.sh dev          # ... and pull the models and rerun the dev set (~25 min on a 5 GB GPU, faster on Apple Silicon)
#   ./setup.sh heldout      # ... and rerun the held-out set (~20 min)
#   ./setup.sh all          # both
# If uv or Ollama is missing the script stops and tells you how to install it.
# VISION_MODEL=qwen3.5:9b-mlx ./setup.sh dev   uses the Apple-Silicon build of the vision model (tiny GPU: qwen3.5:4b).
# LIMIT=3 ./setup.sh heldout                   smoke test: only the first 3 instructions (results file is then incomplete).
set -u
RUN="${1:-check}"
APERTUS="hf.co/Colby/apertus-v1.5-8b-text-Q4_K_M-GGUF"
VISION_MODEL="${VISION_MODEL:-qwen3.5:9b}"
LIMIT_ARGS=(); [ -n "${LIMIT:-}" ] && LIMIT_ARGS=(--limit "$LIMIT")   # expanded with ${LIMIT_ARGS[@]+...}: empty arrays are "unbound" in bash 3.2 (macOS)
missing=0

step() { printf '\n== %s\n' "$1"; }
ok()   { printf '   ok: %s\n' "$1"; }
miss() { printf '   MISSING: %s\n' "$1"; missing=1; }

[ -f run.py ] && [ -f pyproject.toml ] || { echo "Run this script from the repository folder (the one with run.py)."; exit 1; }
case "$RUN" in check|dev|heldout|all) ;; *) echo "usage: ./setup.sh [check|dev|heldout|all]"; exit 1 ;; esac

step "Tools"
if command -v uv >/dev/null 2>&1; then ok "uv $(uv --version)"; else
    miss "uv (Python project manager; it downloads Python itself)"
    cat <<'EOF'
   Install uv, then open a new terminal and run this script again:
     curl -LsSf https://astral.sh/uv/install.sh | sh        # macOS and Linux
     brew install uv                                        # macOS with Homebrew
EOF
fi
if command -v ollama >/dev/null 2>&1; then ok "ollama $(ollama --version 2>/dev/null | head -n1)"; else
    miss "ollama (serves both models locally)"
    cat <<'EOF'
   Install Ollama, then open a new terminal and run this script again:
     macOS:  brew install ollama     or download from https://ollama.com/download
     Linux:  curl -fsSL https://ollama.com/install.sh | sh
   (The desktop app starts the server automatically; otherwise run `ollama serve` in a second terminal.)
EOF
fi
[ "$missing" -eq 0 ] || exit 1

step "Python environment (uv sync)"
uv sync || exit 1

step "Tests without a model"
uv run python tests/test_pipeline.py || exit 1

step "Score the committed answers (must print SCRIPT TOTAL 65.0 / 65)"
uv run python starter/evaluate.py results.json --data dataset || exit 1

if [ "$RUN" = check ]; then
    printf '\nDone. To rerun the pipeline: ./setup.sh dev | heldout | all\n'
    exit 0
fi

step "Ollama server"
serving() { curl -fs --max-time 3 http://localhost:11434/api/tags >/dev/null 2>&1; }
if ! serving; then
    echo "   not running, starting 'ollama serve' in the background"
    nohup ollama serve >/dev/null 2>&1 &
    for _ in $(seq 1 30); do serving && break; sleep 1; done
fi
serving && ok "Ollama answers on localhost:11434" || { echo "   Ollama is not reachable. Start it with 'ollama serve' in another terminal and run this script again."; exit 1; }

step "Models (pulled only if missing; 12 GB in total, takes 5-30 min depending on the connection)"
have="$(ollama list 2>/dev/null | awk 'NR > 1 {print $1}')"   # exact tags, so a present qwen3.5:4b does not stand in for the 9b
for m in "$APERTUS" "$VISION_MODEL"; do
    if printf '%s\n' "$have" | grep -qxF -e "$m" -e "$m:latest"; then ok "$m present"; else echo "   pulling $m"; ollama pull "$m" || exit 1; fi
done

if [ "$RUN" = dev ] || [ "$RUN" = all ]; then
    step "Dev set: 10 scenes, 50 instructions (~25 min on a 5 GB GPU)"
    uv run python run.py --data dataset --out results.json --vision-model "$VISION_MODEL" ${LIMIT_ARGS[@]+"${LIMIT_ARGS[@]}"} || exit 1
    uv run python starter/evaluate.py results.json --data dataset --report report.json
fi
if [ "$RUN" = heldout ] || [ "$RUN" = all ]; then
    step "Held-out set: 5 scenes, 40 instructions (~20 min)"
    uv run python run.py --data heldout --out results_heldout.json --vision-model "$VISION_MODEL" ${LIMIT_ARGS[@]+"${LIMIT_ARGS[@]}"} || exit 1
    echo "   results_heldout.json written (no local scoring: the held-out answers are not published)"
fi
printf '\nDone.\n'
