# Pre-submission sanity check

Team Apertus-2A-ZHAW, "See It, Say It, Pick It", Hack Apertus 2026.
Third pass, written 2026-10-15 against commit `922f1a1` ("Second review pass"), which is pushed and
equals `origin/main`. Deadline: 2026-10-16 12:00 CEST.

Two fresh, independent read-only reviews were run on this commit (integrity, rules and deliverables;
jury view on pipeline logic and code quality) and every finding that matters was re-verified by hand.
Earlier passes ran on `d5c7608` and `10f78ac`; section 1 shows the status of each earlier finding.
Line numbers refer to `src/see_it_say_it_pick_it/pipeline.py` at `922f1a1` unless another file is named.

## Verdict in one paragraph

**No cheating found. The submission is ready apart from two documentation corrections.** The code never
reads the ground truth or the category labels, every organiser file is byte-identical to upstream, all
90 answers re-derive exactly from the raw Apertus traces (including every check round and every action
pixel), the only network call is to local Ollama, the dev set re-scores at 65.0 / 65, and both test
runners pass. All six items from the second pass that could be fixed in the code or the docs are fixed:
held_04_i4 is now physically right after one check round, the "makes every decision" overclaim is
gone, the `--reuse` provenance and the three changed held-out pixels are disclosed. **Two things remain:**
four places call the Linux box run "the submitted code" although it ran on the previous commit; and the
new "place table" set-aside pixel is described as a bare-table spot but lands on the image's bottom
border band in 8 of 15 scenes and, in held_05, most likely inside the black tray that vision did not
detect. Expected jury view today: pipeline logic about 16 / 20, code quality about 4 / 5.

## What was verified directly on `922f1a1`

| Check | Result |
|---|---|
| `uv run python tests/test_pipeline.py` | prints `ok` |
| `uv run pytest -q tests` | 1 passed |
| `uv run python starter/evaluate.py results.json --data dataset` | `SCRIPT TOTAL 65.0 / 65`, 40 of 40, 10 of 10, 0 of 40 refused by mistake |
| `report.json` | identical to a fresh `--report` regeneration |
| Plan structure script, 72 plans | 0 consecutive picks, 0 dangling picks; step-by-step simulation against each scene table: nothing is ever picked while something still lies on it |
| held_04_i4 | yellow_gear aside, red_cube aside, then both red gears to the grey tray; matches the two stacks in the scene table; one check round |
| Overclaim strings | "makes every decision", "one decision maker", "third round fixed held_04_i4", "no Python path" no longer occur in README, report, run log or slides |
| `git diff upstream/main` on organiser files | empty |
| Commits touching `dataset/` or `heldout/` | only the organisers' commits and the upstream import |
| Schema validity, 90 answers | 0 errors; every action pixel inside 2448 x 2048 |
| Answers vs traces | last reply of every trace re-rendered with the pipeline's own functions: 0 mismatches in 90, pixels included; hints and tables re-derived: identical; every logged check-round issue equals the check function's output on the preceding answer |
| First-reply score, dev | 50.4 / 65, matches the claim |
| Self-check counts | dev 8 of 50 fired, 2 needed a second round; held-out 11 of 40, 3 a second, 1 a third; match the docs |
| Re-ask provenance | vs `d5c7608`: 31 of 90 re-asked, as claimed; `922f1a1` re-asked 2 more rounds (held_04_i4, held_05_i8) |
| Root `results*.raw.jsonl` / `*.scenes.json` | SHA256-identical to the `attempts/` copies |
| `git status -sb` on main | clean, equal to `origin/main` |

## 1. Status of earlier findings

| # | Finding | Status on `922f1a1` |
|---|---|---|
| A | held_04_i4 lifted a gear with the cube on it; docs claimed it fixed | **fixed**: the stacking check names every unmoved top in one message (817-827), test at `tests/test_pipeline.py:115-127`; run log sentence now true |
| B | "makes every decision" in README intro, slide 1, slide 2 heading | **fixed**: now "writes every answer" |
| C | Linux reproduction predated the review fixes | **partly**: a second box run on `10f78ac` was recorded (`attempts/2026-10-15_linux_box_10f78ac.report.json`, 63.375, same flip scene_09_i3). But README line 136, report line 88, run-log table line 11 and slide 3 line 244 call it "the submitted code"; the submitted code is `922f1a1`, which changed the stacking message, added the set-aside pixel and bumped the cache version after that run |
| D | state that results are a `--reuse` re-render | **fixed** (README 224-228, report, run log, slide 5) |
| E | "no Python path emits an answer type" | **fixed**: both fallbacks named and verified never to have fired |
| F | verbatim held-out sentences in tests, comments, run log, slides | **partly**: the test and one comment were paraphrased; comments at lines 602 and 627 still say "at the top of the grey tray", the run log line 29 and slide 3 line 240 still quote "take it off the gear" as a rule name, run log lines 33 and 94 and report line 62 quote other held-out sentences; README 220-221 says "the test file and one comment", which undercounts |
| G | name the three changed held-out pixels | **fixed and verified**: exactly held_02_i3, held_02_i4, held_04_i4 changed vs `d5c7608`, none since |
| H | README example for scene_06_i4 stale | **fixed**, matches `results.json` |
| I | report length | **partly**: about 1630 words, still roughly three pages |
| J-b | black tray missing from 7 of 15 scene tables | unchanged, disclosed; photo sampling shows the tray is present in every photo |
| J-c | containment by box centre (banana "in plate" held_02, grey part "in plate" held_05) | unchanged, disclosed |
| J-d | held_05_i8 redundant set-aside | unchanged: the check demands "place none" even when the top object itself goes to the destination (825-827) |
| J-e | wrong rule labels (held_03_i8, scene_09_i5), unpickable choices (scene_05_i5), garbled wording (scene_07_i5, held_04_i5) | unchanged, disclosed |
| J-f | `place table` step had no pixel | **fixed with a new problem**, see 3.1 |
| J-h | code-quality items | partly: pytest-compatible tests, exact-tag model match in setup scripts, README example fixed; module now 1041 lines, `resolve` still 221 lines, bare JSON errors on a corrupt cache, `--reuse` still blind to prompt edits |

## 2. Integrity

### 2.1 Ground truth, labels, hard-coding: clean, disclosed where it matters

The category column is discarded at line 934 before anything else runs; the prompt (63-103) receives
only table, hints and instruction text. No team file references the ground truth, the masks, the CSV,
scene ids, instruction ids or literal pixels. No lookup keyed on sentences. The new commit added no
held-out-specific rule: the direction-refines-container change (699) altered no committed hint, and the
other two changes are the stacking message and the set-aside pixel. The held-out-driven rules from 14
and 15 October are generic in form and disclosed in README "What we disclose" and the report.

### 2.2 Committed answers come from the code: clean

Every answer re-derives from the last reply of its trace. All changes between `10f78ac` and `922f1a1`
are the set-aside pixel on five plans and the held_04_i4 plan, both explained by the code diff. No commit
has ever changed a results file without also changing the pipeline. Only the organisers' re-run can
show the traces came from the model, and the README invites it.

### 2.3 How much Apertus decides: stated honestly

First replies alone score 50.4 / 65 on the dev set; on the held-out set 35 of 40 first replies already
had the right answer type (not claimed anywhere; it would be a fair number to add). The hints name the
resolved object and the rule, and the check messages name the correction; README step 4, the disclosure
section, the report and slide 2 say so. Python never overrides the final reply; the two fallbacks never
fired.

### 2.4 Models, APIs, rules: clean

Apertus community GGUF and `qwen3.5:9b` through Ollama at localhost (918-921), one endpoint, no keys;
only pytest and its dependencies were added to the lockfile. Pixel convention, one-action ask/refuse,
local-only, and label-free held-out answers all hold.

## 3. Jury view: pipeline logic (estimated 16 / 20)

Good: every committed plan executes in order against its scene table, including the two-stack case;
every dev target pixel is at least 36 px inside its mask; the ten dev multi-step plans match the
organisers' reference plans; questions name the choices with positions; all 15 scene tables stay in
vocabulary with plausible counts and box sizes.

### 3.1 The set-aside pixel (new)

`aside_pixel` (861-873) picks one point per scene, the grid point farthest from every detected box and
the image border. In 8 of 15 scenes that is [104, 1936], the bottom-left corner: 104 px is about 25 mm
at the calibration's 0.24 mm per pixel, while a gear is about 85 mm wide, so a set-aside object there
overlaps the grey tray's box and the image edge. Pixel statistics show that row lies in a distinct dark
band about 190 px high along the bottom of every photo, not on the table surface. In held_04_i4 both
set-asides use the same pixel, so the cube would be stacked on the yellow gear. In held_05 the point
[1528, 232] sits in a large dark region at the top right where the black tray lies, and held_05 is one
of the scenes whose table lacks the black tray, so the set-aside most likely lands inside the undetected
tray. The reviewers could not view the images; look at `heldout/held_05/rgb.png` at (1528, 232) and
`dataset/scene_01/rgb.png` at (104, 1936) before deciding. The README wording "bare-table spot farthest
from every box" (190-191) is not what the code guarantees.

### 3.2 Other findings

- held_05_i8 sets the grey part aside and then moves it to the tray; one move too many.
- The stacking message handles any number of independent stacks in one round, but a chained stack (A
  on B on C) needs two rounds and the first message is itself wrong (only supports of picked objects are
  inspected, 819-827). No such scene exists in either set.
- Black tray present in every photo but absent from 7 of 15 tables; "from the black tray" then filters
  nothing. Containment by box centre gives doubtful "in plate" readings in held_02 and held_05.
- Wrong rule labels in held_03_i8 and scene_09_i5; scene_05_i5 offers unpickables; scene_07_i5 and
  held_04_i5 wording garbled. All replayed by `--reuse`, all disclosed.
- Unsafe and contradiction detection are regex lists. Disclosed.

## 4. Jury view: code quality (estimated 4 / 5)

Good: six-stage design with an explanatory docstring; every answer auditable through the traces and a
cache keyed on model, size and version; all 13 flags documented; honest run log; both test runners pass;
`uv.lock` with a dev group; Apache-2.0 with a real copyright line; scorer reproduces in seconds.

Costing the point:
- One 1041-line module; `resolve` 221 lines (520-740) with `m2` rebound five times; `check` 106 lines;
  `main` 124 lines; `PARTS` and `ASIDE` used before definition; area expression repeated eight times.
- Unexplained thresholds remain at 599, 678-679, 687, 172.
- A corrupt `.scenes.json` or `.raw.jsonl` raises a bare JSON traceback (936, 975); `aside_pixel` has
  no box clipping and its centre fallback can land inside a box (866-869).
- `--reuse` keys on table and hints only (985), so a prompt-header edit would replay stale replies, and
  it reads the git-ignored root trace file; the README does not say to copy the `attempts/` files back.
- `setup.sh` lines 15, 76, 81: an empty array expanded under `set -u` is "unbound variable" in bash
  below 4.4, so `./setup.sh dev` dies on stock macOS bash 3.2 unless `LIMIT` is set.
- `setup.ps1` line 46: `ollama --version` with stderr redirected under a stop-on-error preference is a
  terminating error in Windows PowerShell 5.1 when Ollama is installed but not running, so the no-model
  path fails on exactly the machine the script is meant to help.
- One monolithic pytest test; `refine_pixel`, the schema builder, the cache and the round loop are
  untested.

## 5. Deliverables

| Deliverable | Status |
|---|---|
| Public Apache-2.0 repository | pushed, `origin/main` = `922f1a1` |
| `results.json` + `report.json` | present, 65.0 / 65, report identical to a fresh regeneration |
| `results_heldout.json` | 40 answers, valid, traceable |
| One command | `uv run python run.py --data <folder> --out <file>`, plus the setup scripts (see 4 for two platform bugs) |
| 1-2 page report | `docs/REPORT.md`, about 1630 words, roughly three pages |
| Five slides PDF | 5 pages, generated 14 s after the HTML in the same commit |
| Run log, traces, box report | `attempts/` |
| Devpost entry | cannot be checked from here |

## 6. Action list

### Must do before submitting

1. **Stop calling the box run "the submitted code"** in README line 136, report line 88, run-log table
   line 11 and slide 3 line 244, then regenerate `slides.pdf`. Say it ran on `10f78ac`, one commit
   before the submission, and that the last commit changed only the stacking message and the set-aside
   pixel. Or rerun `./setup.sh dev` on the box at `922f1a1` and commit that report.
2. **Decide on the set-aside pixel** after looking at the two photo spots named in 3.1. Either remove
   the pixel from the `place table` step (the schema allows a place without a pixel) or change the
   README wording to what the code does, and in both cases give the second set-aside in one plan a
   different spot. Re-render with `--reuse`; no model calls needed for the pixel.

### Should do

3. Fix the two setup-script bugs: `setup.sh` use `${LIMIT_ARGS[@]+"${LIMIT_ARGS[@]}"}`; `setup.ps1`
   wrap `ollama --version` in try/catch or drop the stderr redirect. Ten minutes, no model.
4. Paraphrase the remaining verbatim held-out sentences (comments 602 and 627, run log 29, 33, 94,
   report 62, slide 3 line 240) and correct README 220-221 to "the prompt synonyms, two comments and
   the run log".
5. Let the stacking check accept a top object that goes straight to the destination first, then
   re-ask held_05_i8 (one call), or list it under known weaknesses.
6. In the nondeterminism note, say the box's scene_06_i2 plan also differed (extra set-aside) and one
   banana pixel moved about 230 px, instead of "same plan with a slightly different pixel" for all 12.
7. Add the `922f1a1` re-ask count (2 of 90) next to "31 of 90" and the held-out first-reply figure
   (35 of 40 right types) next to the dev 50.4.
8. Catch the JSON error on a corrupt cache or trace file and name the file; store a hash of the prompt
   header in each raw record so `--reuse` can detect prompt edits.

### Nice to have

9. Trim the report toward two pages; the run-log table and the nondeterminism paragraph duplicate the
   README and run log.
10. Align the held-out wording list in report line 100 with README line 206.
11. Add a commit field to future scorer-report files.
12. `pyproject.toml` publishes a personal work e-mail in a public repository; the author's choice.

Any change to results must come from regenerating the files with the code and refreshing the
`attempts/` traces in the same commit, never from editing a results file by hand.

## 7. Weaknesses to disclose

Already disclosed: resolver tuned on held-out wording; 50.4 first replies and the check messages that
name the correction; `--reuse` re-render; three changed held-out pixels; black tray missing in half
the scenes; keyword-based unsafe detection; wrong rule labels; nondeterminism; weight provenance; the
two Python fallbacks.

Not yet disclosed: the set-aside pixel is a per-scene constant on the image border in 8 of 15 scenes
and probably inside the undetected black tray in held_05, and held_04_i4 uses it twice; held_05_i8's
redundant move; chained stacks need two rounds; the two setup-script platform bugs; the box run was
on the previous commit.
