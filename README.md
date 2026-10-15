# See It, Say It, Pick It — team Gude2Learn

Our entry to the **ZHAW challenge "See It, Say It, Pick It" at Hack Apertus 2026** (challenge
material and starter code: https://github.com/Fatemeh-MA/see-it-say-it-pick-it). One photo of a
robot workcell and one English instruction go in; the pixel of the object the robot should pick
first and the ordered actions come out. An ambiguous instruction gets a question, an impossible or
unsafe one a refusal.

Team Gude2Learn is Dawn Duan and Claude Code (Anthropic's coding agent, used throughout for the
pipeline, the reviews and the report). Apertus v1.5 8B (text only, 4-bit, in Ollama) writes every
answer; an open vision model only lists the objects it sees; Python does the geometry, turns it
into hints, checks the answer against them and renders the result. Everything runs locally on a
5 GB GPU. How much of each answer is Apertus and how much is Python is measured and stated under
"What we disclose".

| Result (official scorer `starter/evaluate.py`) | |
|---|---|
| Dev set, grounding (target pixel on the right object) | 40 of 40 → 35.0 / 35 |
| Dev set, safety (ask / refuse right, no plan refused by mistake) | 10 of 10, 0 of 40 → 30.0 / 30 |
| Dev set, script total | **65.0 / 65** (from 20.4 with a plain prompt, see `attempts/`) |
| Held-out set (40 instructions, published labels only) | 32 of 32 plans planned, 8 of 8 no-plan cases with the right answer type; pixels unknown |

## Submission checklist

| Deliverable | Where |
|---|---|
| Public Apache-2.0 repository, reproducible | this repository, `LICENSE`; the committed answers re-score in 2 minutes, a full rerun takes about 45 minutes plus the model download |
| Results file for the dev set | `results.json`; its scorer report `report.json` |
| `results_heldout.json` for the five held-out scenes | `results_heldout.json` (also under `heldout/`), produced by the one command below |
| One command that runs the pipeline on any folder of scenes | `uv run python run.py --data <folder> --out <file>` (or `python run.py ...` after `pip install -r requirements.txt`); tested on a folder with other scene names and no calibration file, and on a single scene folder |
| Screen recording (3 min) | https://youtu.be/Ul93mJuPBsk |
| 1–2 page report on where Apertus had difficulty | `docs/REPORT.pdf` (two pages A4; source `docs/REPORT.md`; the evidence behind it is in `attempts/`) |
| Five slides (PDF) | `docs/slides.pdf` (source `docs/slides.html`) |
| Earlier attempts, run log, raw traces of the final run | `attempts/` |

## Set up and run

Tested on Windows 11 (Quadro P2200 5 GB + 32 GB RAM) and a Linux aarch64 box, 65.0 on both with the
GGUF vision build. On macOS (M2 Pro 16 GB) the pipeline runs, but with the MLX build of the vision
model it scored 61.75 (step 4 below), so use the GGUF build `qwen3.5:9b` there too. Nothing leaves
the machine: Ollama serves both models on `localhost:11434`.

Needs: 64-bit Windows, macOS or Linux with 16 GB RAM or more; a GPU is optional (5 GB VRAM was
enough for every number here, CPU-only is untested); ~20 GB free disk (12 GB of models); internet
once, for the tools and the model pull; `git`, `uv` and `ollama` on the PATH with Ollama running.

| Step | Time |
|---|---|
| install uv and Ollama | 5 minutes |
| clone + `uv sync` + tests + score the committed answers (no model) | 2 minutes |
| `ollama pull` of both models (12 GB) | 5 minutes at 50 MB/s, 30 minutes at 8 MB/s |
| full dev-set run (10 scenes, 50 instructions) on the 5 GB GPU | ~25 minutes, ~10 of them the vision stage, cached afterwards |
| held-out run (5 scenes, 40 instructions) | ~20 minutes |

### The short way: one script

After `git clone` and `cd`, one script does every step below: it checks that `uv` and `ollama`
exist and prints the install commands if not, installs the Python environment, runs the tests,
scores the committed answers and, on request, starts Ollama, pulls the two models (only when
missing) and reruns the pipeline.

```
.\setup.ps1                 # Windows PowerShell: check tools, install, test, score (2 minutes)
.\setup.ps1 -Run heldout    # ... and pull the models and rerun the held-out set
.\setup.ps1 -Run all        # ... dev set and held-out set

./setup.sh                  # macOS / Linux / Git Bash: the same
./setup.sh heldout          # or: dev, all
```

If PowerShell refuses to run scripts: `powershell -ExecutionPolicy Bypass -File .\setup.ps1`.
Smoke test of the whole chain on three instructions: `.\setup.ps1 -Run heldout -Limit 3` or
`LIMIT=3 ./setup.sh heldout` (the results file is then incomplete; rerun without the limit).

### 1. Tools

- [uv](https://docs.astral.sh/uv/) (Python project manager; it downloads Python itself):
  `winget install astral-sh.uv` on Windows, `brew install uv` on macOS, or
  `curl -LsSf https://astral.sh/uv/install.sh | sh`.
- [Ollama](https://ollama.com/download) 0.40 or newer, running (`ollama serve`, or the desktop app).

### 2. Clone, install, check (2 minutes, no model needed)

```
git clone https://github.com/jdawnduan/Gude2Learn.git
cd Gude2Learn
uv sync                                                   # Python 3.13 + numpy + opencv into .venv
uv run python tests/test_pipeline.py                      # prints: ok
uv run python starter/evaluate.py results.json --data dataset
```

The last command scores the committed dev-set answers with the official scorer and must print
`SCRIPT TOTAL 65.0 / 65`. Without uv: `pip install -r requirements.txt` on Python 3.11 or newer,
then `python tests/test_pipeline.py` and `python run.py ...`.

### 3. Models (one-time download, 12 GB)

```
ollama pull hf.co/Colby/apertus-v1.5-8b-text-Q4_K_M-GGUF   # Apertus v1.5 8B, text only, 5 GB
ollama pull qwen3.5:9b                                       # open vision model, lists the objects, 7 GB
```

The Apertus build is a community GGUF quantisation (4-bit, Q4_K_M, by the Hugging Face user Colby)
of swiss-ai's Apertus v1.5 8B text model; nothing is fine-tuned. The multimodal Apertus v1.5 needs a
vLLM fork and about 20 GB of memory, which is why the text-only build plus a separate open vision
model is used. `qwen3.5:9b` is the Ollama library build of Qwen 3.5 9B. Ollama 0.40.3 (Windows) and
0.40.4 (Linux) were used for every committed number.

- Apple Silicon: `qwen3.5:9b` (GGUF) runs there too. The MLX build `qwen3.5:9b-mlx` does not keep
  to the JSON schema under Ollama 0.32.13 (fenced replies, a bare array instead of the object list
  in two held-out scenes, fewer trays) and scored 61.75 on the dev set; not recommended.
- Small GPU (5 GB or less): both models are partly offloaded to the CPU, which is the tested setup.
  `--vision-model qwen3.5:4b` fits on the card entirely. Close VMs and other memory-hungry programs
  first.

### 4. Reproduce the results

```
uv run python run.py --data dataset --out results.json              # ~25 min on the 5 GB box
uv run python starter/evaluate.py results.json --data dataset --report report.json
uv run python run.py --data heldout --out results_heldout.json      # ~20 min
```

`--data` takes any folder of scenes: every subfolder with an `rgb.png` and an `instructions.txt`
(one instruction per line, with or without a `category<TAB>` prefix) is a scene, in name order,
with instruction ids `<subfolder>_i1`, `_i2`, ... A `calibration.json` is used when present; a
single scene folder passed directly also works. Each run writes `<out>.scenes.json` (the vision
outputs, cached) and `<out>.raw.jsonl` (per instruction: the table Apertus saw, the hints, its
raw replies) next to the results file; both are ignored by git, the final run's copies are in
`attempts/`.

Decoding is pinned (temperature 0, fixed seed), but the vision stage is not bit-for-bit
repeatable across machines, and one vision box can flip an answer. Five full runs exist (scorer
reports in `attempts/`): the committed Windows run (65.0), a fresh clone on a Linux box three
times (63.4 each, one instruction asked instead of planned because the vision model drew two
machine parts as one box; a part-reference fallback closed it and a fresh Linux run of the final
code scored 65.0 with every category at 100%), a fresh no-cache Windows run (63.4, same cause,
65.0 after the fallback), and a Mac run with the MLX build of the vision model (61.75: that
runner ignores the JSON schema, so two dev instructions were asked and 10 of 40 held-out answers
have the wrong type). The run log in `attempts/README.md` has every detail.

Useful flags: `--limit N`, `--reuse` (take replies from `<out>.raw.jsonl` where table, hints and
prompt are unchanged), `--no-hints` and `--no-check` (ablations), `--think`, `--max-side`,
`--vision-model`, `--model`, `--base-url`, `--team`, `--no-schema`. The results file is rewritten
after every instruction, so an interrupted run leaves a partial file and the next run reuses the
cached scenes. The organisers' `baseline/baseline.py` (Apertus reading the photo) cannot run on
this build: the text-only GGUF has no image path.

## How `run.py` works

1. **See** (vision model, twice per scene): one call lists every object with a box; a second call
   asks only which objects lie on top of which, where the trays are and where the machine parts
   are, because the first pass tends to miss the flat black tray, the stacking and the small
   white module.
2. **Table** (Python, no model): boxes become full-resolution pixels, duplicates are numbered in
   reading order (`red_gear_1`, `red_gear_2`), each object gets the tray or plate it lies in and
   what it lies on. The three machine parts, which the vision model confuses, are relabelled by
   the long side of their box (the camera is fixed, so pixel size is physical size). The target
   pixel is moved from the box centre onto the object's own pixels by colour.
3. **Resolve** (Python, no model): the instruction's noun phrases are mapped to the table with a
   small synonym list and the spatial cues (left, closest to, second from the right, in the grey
   tray, with nothing on it, lying on, under) are evaluated against the coordinates. The result is
   written as *reference hints* into the prompt, with flags for unsafe verbs, colours that match
   nothing and unpickable things.
4. **Plan** (Apertus, once per instruction, text only): the table, the hints, a fixed decision
   order (unsafe, not there, cannot, ambiguous, plan) and abstract answer forms. Output is
   schema-constrained to pick/place steps over the names in this scene, or one question, or one
   refusal. Every answer is written by Apertus and it can deviate from the hints, but the hints
   name the resolved object and the rule that applies.
5. **Self-check** (Apertus, only when needed, at most three rounds): if the answer contradicts a
   fact in the hints (asks although one object matches, refuses an object that is listed, plans an
   unsafe verb, picks the bottom object of a stack first, moves an excluded object, leaves a moved
   object without the named destination), Apertus sees its own answer and the contradiction and
   answers once more. The last reply stands even if it still contradicts a hint.
6. **Render** (Python, no model): a "place none" followed by another pick is a set-aside and is
   written as a place on the table (no pixel: the trays cover most of the table, and a pixel the
   robot cannot trust is worse than none). In a question or a refusal reason every internal name
   is shown as a description a person can act on: "the leftmost red gear (red_gear_1)".

The pipeline never reads the category labels in `instructions.txt`; it sees only the photo and
the instruction text.

## What we disclose

- **The resolver was extended on the held-out wording.** The held-out instructions and labels were
  published on 14 October. Afterwards we added synonym and spatial-cue rules so that every held-out
  wording resolves, with the labels visible to us. The held-out answers come from the code without
  reading the labels, but the resolver is tuned on the held-out text, so the held-out score is
  weaker evidence than a blind run. Three held-out first picks (held_02_i3, held_02_i4, held_04_i4)
  changed under rules written on 15 October with the photos in view. The "32 of 32, 8 of 8" figures
  were computed by us from the labels; pixel correctness is unknown.
- **How much is Apertus, how much is Python.** Every answer is parsed from an Apertus reply;
  Python's two fallbacks (a fixed question for an unparseable plan, dropping a pick of an unlisted
  object) never fired in the 90 answers, and Python inserts the "place on the table" step for a
  set-aside. But the hints state the resolved object and the applicable rule in words, and the
  check messages name the correction. Measured on the dev set: first replies alone 50.4 of 65;
  without the hints (and so without the checks) 48.8; the plain prompt 20.4. This is a small text
  model kept on the rails by deterministic code, not an 8B model that understands the workcell.
- **The committed answers are a `--reuse` re-render, not one fresh end-to-end run.** After the
  15 October review fixes, 31 of the 90 instructions were re-asked and the other replies replayed
  from the previous traces (the trace records the prompt hash, so a prompt edit invalidates a
  replay). The organisers' re-run will be fresh; see the nondeterminism note in step 4.
- **Weights:** a community 4-bit GGUF of swiss-ai's Apertus v1.5 8B (text only) and Ollama's build
  of Qwen 3.5 9B, both unmodified.
- **Known weaknesses** that the script score does not show are listed in `docs/REPORT.md`.

## What is in this repository

| Path | What it is |
|---|---|
| `run.py`, `src/see_it_say_it_pick_it/pipeline.py` | our pipeline (the six steps above) |
| `tests/test_pipeline.py` | checks without a model: table, resolver, answer format, self-check rules |
| `setup.sh`, `setup.ps1` | fresh clone to scored result in one script |
| `results.json`, `results_heldout.json`, `report.json` | our answers and the scorer report |
| `attempts/` | run log, earlier scorer reports, raw traces of the final run, the Mac run |
| `docs/REPORT.md` + `docs/REPORT.pdf`, `docs/slides.pdf`, `docs/DEVPOST.md` | report (markdown and the two-page PDF), slides, the Devpost submission text |
| `dataset/`, `heldout/` | the organisers' scenes: photos, instructions, calibration, ground truth (CC BY 4.0) |
| `starter/`, `baseline/`, `schema/`, `examples/`, `EVALUATION.md`, `docs/DATA.md`, `docs/safety_primer.md` | the organisers' scorer, baseline, answer schema, format example, scoring rules, data and safety notes, unchanged |

## Acknowledgement

The challenge, the dataset, the scorer, the baseline, the answer schema and the evaluation, data
and safety notes are by Fatemeh Mohammadi Amin, ZHAW School of Engineering (mohm@zhaw.ch):
https://github.com/Fatemeh-MA/see-it-say-it-pick-it. Those files are kept here unchanged so the
repository runs on its own. The scenes come from the ZHAW collaborative-robot cell; the papers to
cite are listed in `dataset/README.md`. Thanks to the Hack Apertus 2026 organisers, to swiss-ai for
Apertus, to the Hugging Face user Colby for the GGUF build, and to the Qwen and Ollama teams.

## Licence

Our code is Apache-2.0, as the challenge asks (`LICENSE`, copyright 2026 Dawn Duan, team
Gude2Learn). The organisers' starter code comes from the challenge repository under Apache-2.0;
the photos, instructions and ground truth in `dataset/` and `heldout/` are CC BY 4.0 (ZHAW).
