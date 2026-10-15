# Pre-submission sanity check

Team Apertus-2A-ZHAW, "See It, Say It, Pick It", Hack Apertus 2026.
Second pass, written 2026-10-15 against commit `10f78ac` ("Review fixes"), which is pushed and equals
`origin/main`. Deadline: 2026-10-16 12:00 CEST.

Two fresh, independent read-only reviews were run on this commit (integrity, rules and deliverables;
jury view on pipeline logic and code quality) and their findings were re-verified by hand where it
matters. A first pass on commit `d5c7608` is summarised in section 1 so the status of each earlier
finding is visible. Line numbers refer to `src/see_it_say_it_pick_it/pipeline.py` at `10f78ac` unless
another file is named.

## Verdict in one paragraph

**No cheating found, and the submission is in a materially better state than yesterday.** The code
never reads the ground truth or the category labels, every organiser file is byte-identical to upstream,
all 90 committed answers re-derive exactly from the raw Apertus traces and the recomputed vision tables,
the only network call is to local Ollama, and the dev-set score re-scores at 65.0 / 65. The three
systematic plan errors from the first pass (two picks in a row, "except" ignored, destinations dropped)
are gone, and the held-out tuning, the first-reply score and the weight provenance are now disclosed.
**Three things still need fixing before you submit, all documentation:** the run log and the report say a
third check round fixed held_04_i4, but the committed plan still picks a gear with the cube on it; the
README intro and slide 1 still say Apertus "makes every decision", which the repository's own disclosure
contradicts; and the second-machine reproduction is presented as a run of the submitted code although it
ran before the review fixes. Expected jury view today: pipeline logic about 15 / 20, code quality about
4 / 5.

## What was verified directly on `10f78ac`

| Check | Result |
|---|---|
| `uv run python tests/test_pipeline.py` | prints `ok` |
| `uv run python starter/evaluate.py results.json --data dataset` | `SCRIPT TOTAL 65.0 / 65`, 40 of 40, 10 of 10, 0 of 40 refused by mistake |
| Plan structure script over all 72 plans | 0 plans with two picks in a row, 0 dangling picks (was 8 and 3) |
| `git diff upstream/main` on organiser files | empty: scorer, schema, dataset, ground truth, held-out scenes untouched |
| Commits touching `dataset/` or `heldout/` | only the organisers' commits and the upstream import |
| Schema validity, both results files | all 90 valid; `place` with object `table` is allowed (schema only requires a non-empty string) |
| Instruction coverage | 50 dev ids, 40 held-out ids, no duplicates |
| Answers vs traces | every answer re-derived from the last Apertus reply plus cached pixels: 0 mismatches in 90; scene tables and refined pixels recomputed from the raw vision replies and photos: identical |
| Root `results*.raw.jsonl` / `*.scenes.json` | SHA256-identical to the `attempts/` copies |
| First-reply score (dev, no self-check) | 50.4 / 65, matches the README and report claim |
| `git status -sb` on main | clean, equal to `origin/main` |

## 1. Status of the first-pass findings

| # | Finding on `d5c7608` | Status on `10f78ac` |
|---|---|---|
| P1 | two commits unpushed | fixed, main equals origin |
| P2 | resolver and prompt tuned on held-out wording after the 14 Oct release; literal held-out sentences in comments and a test | disclosed in README, report, run log and slide 5; the comment at the old line 538 and 646 and the test sentence were replaced. But the fix commit added new literals: `tests/test_pipeline.py:163` tests the exact sentence of held_02_i3, and the comment at line 604 quotes held_02_i4. See 2.3 |
| P3 | slide 2 and README step 4 overclaimed Apertus's role | README step 4, the disclosure section, the report and slide 2 are now accurate and give the 50.4 figure. README lines 6-7 and slide 1 (line 167) still say "makes every decision"; slide 2 heading still says "one decision maker" |
| P4 | weight provenance undisclosed | fixed in README step 3, report, slide 2 |
| P5 | hygiene (pyproject, LICENSE, .gitignore, flags, requirements, timing) | fixed |
| J1 | 8 plans with two picks in a row | fixed: a set-aside before another pick is rendered as `place table` (lines 877-886); 6 plans carry it |
| J2 | "except X" not enforced | fixed: hint (710-715, 727-729) plus check rule (813-816); the three affected plans no longer pick X |
| J3 | five wrong held-out plans | four fixed (held_02_i3, held_02_i4, held_03_i2, held_05_i8). **held_04_i4 still wrong**, see 3.1 |
| J4 | black tray missing in 7 of 15 scene tables | unchanged, disclosed |
| J5 | unsafe and contradiction detection are keyword lists | unchanged, disclosed |
| J6 | wrong rule labels in refusals, unpickable choices in one question | unchanged (held_03_i8, scene_09_i5, scene_05_i5), disclosed |
| J7 | banana pixel 3 px from the mask edge | fixed: concave-blob rule in `refine_pixel` (436-444), now 70 px from the edge; no dev pixel is within 36 px of an edge |
| J8 | code-quality items | mostly fixed: shared `_inter`/`_overlap`, thresholds commented, HTTP and timeout errors handled, cache keyed on vision model and size, results written after every instruction, pyproject and LICENSE filled. Still open: 1020-line module, `resolve` 221 lines, tests not pytest-compatible |

## 2. Integrity

### 2.1 Ground truth and labels: clean

Instructions are loaded by the organisers' `starter/scene_io.py`, which splits `category<TAB>text`;
the pipeline keeps only the text at line 914 and nothing else runs before that. No team file references
`ground_truth`, `instructions.csv`, the masks, scene ids, instruction ids or literal pixels. The scorer
is imported only for its schema checker. The scene tables and every target pixel were recomputed from
the raw vision replies and the photos and match the cached values exactly, so pixels are code-derived.

### 2.2 Committed answers come from the code: clean, with one thing to state

Every answer in both results files re-derives from the last Apertus reply in the trace of the same
instruction. Every change between `d5c7608` and `10f78ac` (12 dev, 8 held-out) maps to a hunk in the
`pipeline.py` diff of the same commit; none looks hand-made. Results files have never been changed in a
commit that did not also change the pipeline and tests.

The final files are a `--reuse` re-render, not one fresh end-to-end run: 59 of 90 first replies are
byte-identical to the previous traces and 31 instructions were re-asked, exactly as the run log says.
The prompt header (lines 1-101) is unchanged, so the reused replies are valid. The organisers' re-run
will be fresh and may differ by the disclosed one-answer nondeterminism; say explicitly in the README
that the committed held-out file was re-rendered with `--reuse`.

### 2.3 Hard-coding and held-out tuning: disclosed, keep it that way

No per-scene or per-instruction constants and no lookup from sentence to answer. The disclosure in
README "What we disclose", `docs/REPORT.md` and slide 5 is accurate: synonym and spatial-cue rules were
extended on the published held-out wording with the labels visible, and the 32/32, 8/8 figures come
from the labels. Three held-out target pixels (held_02_i3, held_02_i4, held_04_i4) changed as a result
of rules written on 15 October with the held-out text and photos in view; the disclosure should name
that consequence. The fix commit also reintroduced two verbatim held-out sentences,
`tests/test_pipeline.py:163` and the comment at line 604, and the run log and slide 3 still quote "take
it off the gear" as a rule name. Either remove them or add one sentence that held-out sentences appear
in comments and tests.

Legitimate domain knowledge, fine to keep: object vocabulary and unpickables (40-46), gripper rules in
the prompt (82-83), part sizes for a fixed camera (299-301), colour priors (490-494), the unsafe verb
list from the safety primer (495-496), cube-never-under-gear prior (266-269), reading-order numbering.

### 2.4 How much Apertus decides: now stated honestly, two leftovers

Apertus writes every JSON answer; the last reply stands after at most three rounds and Python never
overrides the answer type. On 71 of 90 instructions the first reply already matched the hints; on 19
it was re-asked with messages that name the required action, sometimes the whole plan ("the plan must
start with pick X", "exactly three moves, in this order"). First replies alone score 50.4 / 65 on the dev
set; the check rounds add the rest. README step 4, the disclosure section, the report and slide 2 say
this. Two leftovers contradict it: README lines 6-7 and slide 1 line 167 ("makes every decision"),
slide 2 heading ("one decision maker"). Also "no Python path emits an answer type" (README 204, report
106, slide 2) is not literally true: line 891 returns a fallback question for an unparseable plan and
lines 874-875 drop picks of unlisted objects. Neither fired in the 90 committed answers; say "never
fired" rather than "no path".

### 2.5 Models and APIs: clean

Apertus `hf.co/Colby/apertus-v1.5-8b-text-Q4_K_M-GGUF` and `qwen3.5:9b` via Ollama at localhost
(898-900); one endpoint, temperature 0, seed 0; no keys, no other hosts. Provenance disclosed.

## 3. Jury view: pipeline logic (estimated 15 / 20)

Good: all 15 scene tables use only the vocabulary with plausible counts and box sizes; stacking is right
in 5 of 6 stacked scenes; 71 of 72 plans are executable in order including every set-aside; the dev
multi-step plans match the organisers' reference plans; questions name the choices with positions.

### 3.1 held_04_i4 is still physically impossible and the docs say it was fixed

Committed plan: pick yellow_gear, place table, pick red_gear_2, place grey_tray, pick red_gear_1, place
grey_tray. The scene table for held_04 says red_cube lies on red_gear_1, so the last pick lifts a gear
with the cube on it. The trace shows three rounds oscillating: round 1 "yellow_gear lies on red_gear_2,
start with pick yellow_gear", round 2 "red_cube lies on red_gear_1, start with pick red_cube" (reply
drops the yellow gear), round 3 repeats round 1. Cause: `check` reports only the first stack violation
and words it as "must start with" (817-822, the `break`), so two stacks give contradictory corrections.
`attempts/README.md:68` ("A third check round fixed held_04_i4") and `docs/REPORT.md:67` ("a destination
check and a third round fixed both") are wrong. A jury that reads the plan will find this.

### 3.2 Other findings

- Black tray missing from the vision table in 7 of 15 scenes; "from the black tray" then filters
  nothing (scene_05_i4 is right by coincidence). Disclosed.
- Containment is a box-centre test: banana "in plate" (held_02) and grey_part "in plate" (held_05) are
  doubtful and can add a needless move (held_05_i4).
- held_05_i8 has a redundant set-aside of the grey part before moving it to the tray, because the
  destination check says "add that move" and Apertus added rather than replaced.
- Refusal labels wrong in held_03_i8 ("NOT THERE" for impossible, repeated identically through three
  rounds) and scene_09_i5 ("unsafe" for a contradiction); scene_05_i5 offers the plate, saucer and tray
  as things to give; held_04_i5 repeats the gear description twice; scene_07_i5 wording is garbled.
- The `place table` step carries no pixel while every other step does.
- Unsafe, contradiction and "inside a solid object" rules are regex lists. Disclosed.
- Duplicate numbering is top-row first, so pipeline names differ from the reference names in the
  scene_06 and scene_09 reports (harmless, the scorer uses pixels).

## 4. Jury view: code quality (estimated 4 / 5)

Good: six-stage design with an explanatory docstring; every answer auditable through traces and a
versioned, model-keyed cache; accurate README with all flags; honest run log; tests run without a model;
`uv.lock`; Apache-2.0 with a real copyright line; setup scripts that fail loudly and pull models only
when missing; actionable Ollama error messages.

Costing the point: one 1020-line module, `resolve` 221 lines (520-740) with `m2` rebound five times,
`check` 101 lines, `main` 123 lines; `PARTS` and `ASIDE` used before definition; a few unexplained
thresholds remain (678, 687, 599); a corrupt cache or trace file raises a bare JSON error (916, 955);
`--reuse` keys on table and hints only, so a prompt edit would silently replay stale replies; tests are
not pytest-compatible (`test_resolve(objs)` takes a positional parameter); `README.md:231-233` shows an
example answer for scene_06_i4 that is not the committed one; the setup scripts' model-presence test
matches on the bare tag so `qwen3.5:4b` present would skip pulling the 9b.

## 5. Deliverables

| Deliverable | Status |
|---|---|
| Public Apache-2.0 repository | pushed, `origin/main` = `10f78ac` |
| `results.json` + `report.json` | present, 65.0 / 65 |
| `results_heldout.json` | 40 answers, valid, traceable |
| One command | `uv run python run.py --data <folder> --out <file>`, plus `setup.ps1` / `setup.sh` |
| 1-2 page report | `docs/REPORT.md`, about 1850 words, probably 3-4 pages; trim or accept |
| Five slides PDF | `docs/slides.pdf`, 5 pages, regenerated 4 s after the HTML in the same commit |
| Run log and traces | `attempts/` |
| Devpost entry | cannot be checked from here |

## 6. Action list

### Must do before submitting (documentation only, no model time needed)

1. **Correct the held_04_i4 claim** in `attempts/README.md:68` and `docs/REPORT.md:67`: it is still
   wrong (cube left on red_gear_1) because the stacking check reports one stack at a time. Or fix it:
   drop the `break` at 822 and list every unmoved top in one message, re-ask that instruction with
   `--reuse`, regenerate the results and traces together, re-score.
2. **Soften "makes every decision"** in README lines 6-7 and slide 1 (`docs/slides.html:167`), and
   "one decision maker" (slide 2 heading), to match the disclosure; regenerate `slides.pdf`.
3. **Date the second-machine run**: the 63.4 Linux reproduction ran on the pre-review-fix code. Say so
   in README (lines 132-135), the report (91-93), the run log line 31, and the slide 3 footnote (line
   246), or rerun `./setup.sh dev` on that machine with `10f78ac`.

### Should do

4. State in the README that the committed held-out file is a `--reuse` re-render (31 of 90 re-asked).
5. Replace "no Python path emits an answer type" with "the fallback paths never fired in the 90
   committed answers" (README 204, report 106, slide 2).
6. Remove the new verbatim held-out sentences (`tests/test_pipeline.py:163`, comment at line 604) and
   the "take it off the gear" rule name in the run log and slide 3, or extend the disclosure.
7. Add to the disclosure that three held-out target pixels changed as a result of the 15 Oct rules.
8. Fix the README example answer for scene_06_i4 (lines 231-233) to the committed one.
9. Label fixes at render time: when the hints say "(rule 3)" and the reason starts with "NOT THERE" or
   "unsafe", prefix the right label; exclude unpickables from "no clue" question choices (check line
   756). Both need no model call for the label part.

### Nice to have

10. Make the tests pytest-compatible and add pytest to dev dependencies.
11. Give the `place table` step a pixel; note in README step 6 that it mirrors the reference plans'
    "place aside".
12. Slide 3 heading says "nine runs" over a ten-row table.
13. Add the Linux run's scorer output to `attempts/` so the 63.4 claim has an artifact.
14. Trim `docs/REPORT.md` toward two pages.

Any change to results must come from regenerating the files with the code and refreshing the
`attempts/` traces in the same commit, never from editing a results file by hand.

## 7. Weaknesses to disclose

Already disclosed: resolver tuned on held-out wording; first replies 50.4 / 65 and the check messages
that name the correction; black tray missing in half the scenes; keyword-based unsafe detection; wrong
rule labels; nondeterminism and the 63.4 reproduction; weight provenance.

Not yet disclosed: held_04_i4 is physically impossible as committed and the self-check oscillates when
two objects are stacked on two targets; the committed results are a `--reuse` re-render; three
held-out target pixels changed under the 15 Oct rules; containment by box centre can add a needless
move; Python inserts `place table` and would emit a fallback question on an unparseable plan.
