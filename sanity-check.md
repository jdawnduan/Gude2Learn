# Pre-submission sanity check

Team Apertus-2A-ZHAW, "See It, Say It, Pick It", Hack Apertus 2026.
Written 2026-10-15 against commit `d5c7608` (clean working tree). Deadline: 2026-10-16 12:00 CEST.

Three independent read-only reviews were run on the repository and combined here: an integrity audit
(ground-truth leakage, hard-coding, trace consistency, git history), a rules-and-deliverables audit, and
a jury-style review of pipeline logic and code quality. Line numbers refer to
`src/see_it_say_it_pick_it/pipeline.py` at commit `d5c7608` unless another file is named.

## Verdict in one paragraph

**No cheating found.** The committed answers are reproduced exactly from the raw Apertus traces, the
code never reads the ground truth or the category labels, no organiser file (scorer, schema, data,
ground truth, held-out scenes) was modified, and the only network call is to local Ollama. The dev-set
score of 65.0 / 65 re-scores with the official scorer. **Two things must be done before submitting:**
push the two unpushed commits (the public repository still lacks `docs/REPORT.md`), and add a short
disclosure that the resolver and prompt were extended on the wording of the public held-out instructions
after their release on 14 October. **One thing should be fixed:** eight submitted plans contain two picks
in a row because "set it aside" moves lose their place step; this costs nothing in the script score but
is the first thing a jury reading the plans will notice. Expected jury view today: pipeline logic about
13 / 20, code quality about 4 / 5.

## What was verified directly

| Check | Result |
|---|---|
| `uv run python tests/test_pipeline.py` | prints `ok` |
| `uv run python starter/evaluate.py results.json --data dataset` | `SCRIPT TOTAL 65.0 / 65`, grounding 40 of 40, safety 10 of 10, 0 of 40 plans refused by mistake, every category 100% |
| `report.json` totals | automatic_score 65.0, matches README, slides, `attempts/README.md`, `docs/REPORT.md` |
| `git diff upstream/main -- starter/ EVALUATION.md schema/ dataset/ heldout/ baseline/ docs/DATA.md docs/safety_primer.md` | empty: every organiser file is byte-identical to upstream |
| Commits touching `dataset/ground_truth` | only the organisers' initial commit |
| Schema validity of both results files | all entries valid: plans start with pick and carry an in-range `[u, v]`; ask/refuse are one action with `target_pixel: null`; `team` present |
| Instruction coverage | `results.json` has exactly the 50 dev ids; `results_heldout.json` has exactly `held_01_i1` .. `held_05_i8`, no duplicates |
| Answers vs raw traces | re-deriving every answer from the last Apertus reply in `attempts/final_*.raw.jsonl` plus the cached pixels reproduces both results files with 0 mismatches in 90 |
| Root `results*.raw.jsonl` / `results*.scenes.json` | SHA256-identical to the `attempts/` copies |
| `git status -sb` | `main...origin/main [ahead 2]`: `5d276ca` (REPORT.md) and `d5c7608` (second-machine run) are not on GitHub |

## 1. Integrity: is the score earned by the code?

### 1.1 Ground truth and labels: clean

- Instructions are loaded with the organisers' `starter/scene_io.py`, which splits `category<TAB>text`;
  the pipeline keeps only the text: line 836, `jobs = [(d, iid, text) for ... for iid, _, text in load_scene(d).instructions]`.
  The instruction string stored in every trace line equals the text column of `instructions.txt` for
  all 90 instructions.
- No file in `run.py`, `pipeline.py`, `tests/`, `setup.*` references `ground_truth`, `instructions.csv`,
  the masks, scene ids, instruction ids, or literal pixels. The only reader of the ground truth is the
  organisers' scorer, and it loads it inside its own `main()`, so importing `answer_type` from it
  (line 36) does not touch the answers.
- The numeric priors in the code are vision-derived, not mask-derived: part size thresholds
  `PART_LONG_SIDE` (296-298), the reading-order band (187), the stacking overlap (285).

### 1.2 Committed answers come from the code: clean

Every answer in `results.json` and `results_heldout.json` is reproduced from the final reply in the
raw trace of the same run, including questions, reasons and pixels. `results.json` was changed in three
commits and each one also changed `pipeline.py` and the tests, with diffs that match the commit
messages; there is no results-only edit. The held-out file and its trace were written in the same
second. What cannot be verified offline is that the trace files themselves came from the model rather
than being typed; only the organisers' re-run can show that, and the README already invites it.

### 1.3 Hard-coding: nothing per scene or per instruction, but tuned to held-out wording (disclose)

There is no lookup from instruction text to answers and no scene-specific constant. The resolver is
generic in form. However, these rules, prompt lines, a comment and a test match held-out instructions
verbatim, none of which occur in the dev set, and git shows they were all committed after the held-out
scenes were imported (`e5b8646`, 14 Oct 14:21): in `0870db3` (14 Oct 16:03), `b926e4c` (14 Oct 17:01)
and `77c3b86` (15 Oct 10:30).

| Code | Held-out instruction it fits |
|---|---|
| line 445 `gears? (that|which) (is|are) not red` -> yellow_gear; prompt line 75 | held_01_i7 "pick up the gear that is not red" |
| lines 449-450 `orange fruit`, `long yellow fruit`; prompt line 73 | held_04_i1, held_02_i1 |
| lines 530-533 "solid object, nothing fits inside it" (fires on no dev instruction) | held_03_i8 "put the mug inside the small white part" |
| line 538 comment `("off the gear")`; `tests/test_pipeline.py:121` tests that literal sentence | held_04_i8 "take it off the gear" |
| prompt line 86 lists `"that part"` as an ambiguity example | held_03_i5 "put that part in the tray" |
| line 646 comment `("both grey parts and the small white part")` | held_03_i4, verbatim |
| line 631 `(to the |just |right |directly )?`; lines 479, 661-664 image-edge words | held_05_i6 "just below the mug", held_03_i6 "bottom of the image", held_02_i3 "top of the grey tray" |

The organisers published the held-out text and labels on 14 October, so reading them breaks no rule,
and the answers do come from the code. But EVALUATION.md says the held-out score "is what shows whether
a pipeline really works", and a resolver extended sentence by sentence to the 40 held-out wordings with
the labels in view weakens that evidence. The honest move is to say so (see section 6). The held-out
summary "32 of 32 plans, 8 of 8 no-plan" in `attempts/README.md` and on the slides was necessarily
computed by hand from the labels in `heldout/*/instructions.txt`; say that too.

Legitimate domain knowledge that is fine to keep, and worth listing in the report as such: the object
vocabulary (lines 40-41, from `dataset/README.md`), the 85 mm rule and unpickables (44, 82-83), the
synonym list (442-467), colour priors (471-474), the unsafe verb list (475-476, from
`docs/safety_primer.md`), the part size thresholds for a fixed camera, the cube-never-under-gear prior.

### 1.4 Does Apertus decide? Yes in form, mostly Python in substance (disclose accurately)

- Every answer is parsed from an Apertus reply (lines 887, 906); no path emits a plan without the model,
  even when the resolver finds exactly one match. Python never overwrites the answer type: after at most
  two rounds the last reply stands, and two held-out answers were committed although they still
  violated a check.
- But the hints state the answer type and object in words ("exactly one object, so plan, do not ask",
  line 682; "gives no clue which: ask (rule 4)", 686; the full three-move order, 569-570), and the
  self-check messages name the required answer ("give the plan", 707; "refuse", 728-732; "Pick {target}",
  744; "exactly three moves, in this order", 772-773). The prompt says "Only ask when the hints say
  'no clue'" (89).
- Measured on the final traces: the final answer type equals the type implied by the hints on 50 of 50
  dev and 39 of 40 held-out instructions, and every plan picks the hint-resolved object. Re-scoring
  Apertus's *first* replies on the dev set gives 51.25 / 65; the check loop added the rest. Self-check
  fired on 9 of 50 dev and 11 of 40 held-out instructions.
- `docs/REPORT.md` already says "the 8B model does not get there alone, and we do not claim it does".
  Slide 2 ("Nothing in Python decides an answer type; the self-check only shows Apertus a contradiction")
  and README step 4 ("Apertus makes every decision; the hints only save it from doing geometry") are
  stronger than the code supports and should be softened to match the report.

### 1.5 Models and APIs: clean

Apertus `hf.co/Colby/apertus-v1.5-8b-text-Q4_K_M-GGUF` and `qwen3.5:9b`, both via Ollama at
`http://localhost:11434` (lines 820-823); the only HTTP call is `ollama_chat` (176-179). No proprietary
API, no keys, no other hosts. The organisers' `baseline/baseline.py` contains an OpenAI-compatible call
to localhost:8000, but the pipeline imports only `encode` and `parse` from it. Disclose that the Apertus
build is a community GGUF quantisation of swiss-ai's Apertus v1.5 8B, so the jury can judge provenance.

## 2. Rules compliance

| Rule | Status | Evidence |
|---|---|---|
| Apertus does the language reasoning | OK, with the caveat in 1.4 | vision model called twice per scene with object/layout prompts only (lines 852, 858); it never sees the instruction |
| Open vision model may help it see | OK | qwen3.5 9B, open weights |
| Everything local, no proprietary model or API | OK | section 1.5 |
| `[u, v]` = column, row, origin top-left, full 2448 x 2048 photo | OK | boxes scaled to `sc.size` (line 225); `refine_pixel` returns `[x1 + xs[i], y1 + ys[i]]`; 40 of 40 grounding confirms it |
| Left/right/top/bottom as seen in the photo | OK | resolver uses image coordinates |
| One `ask_user` or one `refuse`, `target_pixel: null` | OK | schema check passed on all 90 |
| Held-out answers must come from the code | OK | section 1.2; README gives the exact command; the setup scripts run it |
| Category labels not used by the pipeline | OK | line 836 |
| Licence | OK | unmodified Apache 2.0 text; data CC BY 4.0 as the organisers state |

## 3. Deliverables

| Deliverable | Status |
|---|---|
| Public Apache-2.0 repository | present locally; **2 commits not pushed** |
| `results.json` + `report.json` | present, re-score to 65.0 / 65 |
| `results_heldout.json`, 40 answers | present, schema-valid, traceable |
| One command on any folder of scenes | `uv run python run.py --data <folder> --out <file>` |
| 1-2 page report on Apertus difficulties | `docs/REPORT.md`, about 1450 words, added in `5d276ca` (unpushed) |
| Five slides as PDF | `docs/slides.pdf` has 5 pages and is newer than `docs/slides.html` |
| Run log and raw traces | `attempts/` |
| Devpost entry with repository link and `results_heldout.json` | cannot be checked from here |

## 4. Jury view: pipeline logic (estimated 13 / 20)

What is good: every object name in all 15 scene tables is in the vocabulary, duplicates are numbered,
stacks are mostly captured, questions are specific and answerable ("the leftmost red gear (red_gear_1),
the middle ..."), and the dev plans match the ground-truth reference plans.

What a jury will find, all verified on the committed files:

1. **Eight plans contain two picks in a row.** `to_answer` (lines 803-808) drops Apertus's
   `"place": "none"` step, so every "set it aside" move is submitted as pick X, pick Y. For a one-object
   gripper that is impossible as written. Dev: scene_06_i4, scene_07_i4, scene_08_i4, scene_09_i4.
   Held-out: held_02_i7, held_04_i3, held_04_i4, held_05_i8. The organisers' reference plans spell
   the move as "place aside".
2. **"Except X" is not enforced.** The check at line 750 adds any object named in any hint to the
   allowed set, so the excluded object is picked anyway, as a dangling pick at the end:
   scene_10_i4 (yellow gear), held_01_i8 (red cube), held_02_i7 (yellow gear). The dev case still
   scores because the first pick is right.
3. **Five held-out plans are semantically wrong** even where the first pick is right: held_01_i8 and
   held_02_i7 (excluded object picked, and held_02_i7 places nothing on the plate); held_02_i3 (the
   lower red gear chosen for "at the top of the grey tray", because the direction cue is skipped once a
   container cue is set, line 660; this is a likely grounding miss); held_02_i4 ("both fruits on the
   plate" lost the mandarin, because the destination "on the plate" was read as a location filter,
   lines 583-589, and the check then told Apertus to drop it); held_04_i4 (the yellow gear on red_gear_2
   is never moved; the two-round budget ran out). Two more are incomplete: held_03_i2 and held_05_i8
   drop a destination. The 65 / 65 reflects the first pick and the answer type, not the whole plan.
4. **The black tray is missing from the vision table in 7 of 15 scenes** (scene_02/03/04/05,
   held_02/03/05). "From the black tray" is then silently skipped; scene_05_i4 is right only because
   "everything" happened to coincide.
5. **Unsafe and contradiction detection are keyword lists**, not physical reasoning: "pick up the red
   cube without moving the mug" would be refused even if feasible (line 526); "the cube under my hand"
   is not caught by `UNSAFE_RE` (475-476). scene_07_i5 passes on the keyword alone because vision missed
   the mug-on-gear stack.
6. **Refusal labels are wrong in about a quarter of refusals** while the content is right:
   held_03_i8 says "NOT THERE" for an impossible request; scene_09_i5 says "unsafe" for a contradiction;
   held_04_i5's reason is bloated by name rendering. scene_05_i5's question offers the plate, saucer and
   tray as things to "give".
7. **Target pixels can sit at an object's edge.** The banana pixel (scene_05_i1, i2) is 3 px from the
   mask edge because `refine_pixel` (399-425) takes the blob pixel nearest the centroid, which for a
   concave object lies on the inner edge. EVALUATION.md's robot check prefers the middle of the top
   surface. Median edge distance over the dev set is otherwise good.

## 5. Jury view: code quality (estimated 4 / 5)

Good: a clear six-stage design with an explanatory module docstring; every answer auditable through
the raw trace and the versioned scene cache; accurate README and time budget; honest run log; tests
run without a model; `uv.lock` pins; one setup script per platform that fails loudly and pulls models
only when missing.

Costing the point: one 928-line module with a 195-line `resolve` (500-693) of nested regex branches;
box intersection written three times (282-284, 301-307, 609-612); unexplained thresholds (1.5 at 241,
0.3/0.8 at 215, 0.6 at 285, 40 px at 324, `//12` at 410); only `URLError` caught (919) so a timeout or
a 404 "model not found" gives a traceback or a misleading message; a malformed vision reply becomes an
empty table silently (860); the scene cache key ignores `--vision-model` and `--max-side`;
`results.json` written only at the end (921); tests are a script, not pytest-compatible;
`pyproject.toml` still says `description = "Add your description here"`; LICENSE appendix still reads
`Copyright [yyyy] [name of copyright owner]`; the organisers' `requirements.txt` ("Python 3.11, vllm")
contradicts `pyproject` (>= 3.13); `.venv/` is kept out of git only by uv's own `.venv/.gitignore`;
README does not list `--model`, `--base-url`, `--team`, `--no-schema`.

## 6. Action list

### Must do before submitting

1. **Push** `main` (two commits ahead of `origin/main`). Then open the GitHub page and confirm
   `docs/REPORT.md` is visible and the repository is public.
2. **Add a disclosure paragraph** to `docs/REPORT.md` (and a line in the README) along these lines:
   "The held-out instructions and their category labels were published on 14 October. Between 14 and
   15 October we extended the resolver's synonym and spatial-cue rules so that every held-out wording
   resolves (for example 'the gear that is not red', 'long yellow fruit', 'just below', 'inside the
   small white part'), with the labels visible. The held-out answers are produced by the code without
   reading the labels, but the resolver was tuned on the held-out text. The 32/32 and 8/8 held-out
   figures were computed by us from the published labels; pixel correctness is unknown."
3. **Soften slide 2 and README step 4** to match `docs/REPORT.md`: Apertus writes every answer and can
   deviate, but the hints state the resolved object and the applicable rule, and the self-check messages
   name the required correction. Give the number: first-reply dev score 51.25 / 65, final 65.0 / 65.
4. **State the weight provenance**: Apertus v1.5 8B as a community Q4_K_M GGUF (hf.co/Colby), text-only.

### Should do if there is time (each is small; re-render or rerun only the affected instructions)

5. **Emit the set-aside move as a place** (`to_answer`, 803-808): when `place` is `none` and another
   pick follows, append `{"action": "place", "object": "table"}`. Fixes the pick-pick sequence in
   eight plans. Needs no model call: `--reuse` re-renders from the logged replies in seconds. Re-score
   the dev set afterwards and regenerate `results_heldout.json` and the `attempts/` copies together.
6. **Close the "except" loophole** (`check`, line 750): `resolve` already computes `ex_names` (674);
   remove them from the allowed set and add a hint "do not move X". Fixes scene_10_i4, held_01_i8,
   half of held_02_i7.
7. **Destination fixes**: skip the container cue when the container is the object of put/move/place
   (583-589, fixes held_02_i4); let a direction refine after a container cue (660, fixes held_02_i3);
   challenge a missing destination when the instruction says "put ... in/on" and the move has
   `place none` (fixes held_02_i7, held_03_i2, held_05_i8).
8. **Pixel refinement**: use the maximum of `cv2.distanceTransform` on the blob instead of the pixel
   nearest the centroid (422-425); moves the banana pixel to the middle of the object. Cache-only
   recompute, no model.
9. **Remove the literal held-out sentences** from the comments at lines 538 and 646 and from
   `tests/test_pipeline.py:121`, replacing the test with a dev-style sentence.

Any change under 5-9 must be followed by regenerating the results files from the code, never by
editing a results file by hand, and the `attempts/` traces must be refreshed in the same commit.

### Nice to have (15 minutes total)

10. Fill `pyproject.toml` description and the LICENSE copyright line; add `.venv/` to `.gitignore`.
11. Document `--team`, `--model`, `--base-url`, `--no-schema` in the README flag list.
12. Replace or annotate the organisers' `requirements.txt` so pip users are not told Python 3.11 + vllm.
13. Catch `TimeoutError` and `HTTPError` at line 919 with distinct messages; write results incrementally.
14. Reconcile the dev-run timing on slide 5 with the README time budget, or drop one of them.
15. Record the exact Ollama version used in the README so the organisers' re-run has the same stack.

## 7. Known weaknesses to disclose honestly

- Set-aside moves are submitted as consecutive picks (8 of 72 plans) unless item 5 is applied.
- Held-out plans are wrong or incomplete in 7 of 32 cases even where the first-pick pixel is right.
- "Except X" is not enforced; X is picked anyway in 3 plans.
- The black tray is missing from the vision table in 7 of 15 scenes.
- Unsafe and self-contradiction detection are keyword lists.
- Refusal rule labels are wrong in about a quarter of refusals; the content is right.
- Target pixels can lie at the edge of concave or occluded objects.
- Two full runs are not bit-identical; the second-machine reproduction scored 63.4 with one flipped
  answer (already disclosed).
- Without the Python hints and self-check the same Apertus scores 20.4 (already disclosed).
- The resolver was extended on held-out wording after the held-out release (not yet disclosed).
