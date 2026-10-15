# Attempts: run log and evidence

What is in this folder:

| File | What it is |
|---|---|
| `2026-10-13_mac_plain_prompt.report.json` | scorer report of the first two-stage pipeline with a plain prompt, 23.3/65 (M2 Pro Mac) |
| `2026-10-13_mac_claude_prompt.report.json` | same day, a longer prompt variant, 17.7/65 (kept as a negative result) |
| `final_dev.raw.jsonl`, `final_heldout.raw.jsonl` | the final run, one line per instruction: the scene table Apertus saw, the reference hints, its raw reply, the self-check issues and its corrected replies. This is the evidence for the jury's "pipeline logic" question |
| `final_dev.scenes.json`, `final_heldout.scenes.json` | the final run's vision outputs (raw replies of both vision calls) and the derived table per scene |
| `../report.json` | scorer report of the final dev-set run, 65.0/65 |
| `../results.json`, `../results_heldout.json` | the submitted answers |

Everything below is the run log: what was changed, what it scored, and what Apertus got wrong.

## Run log

Dev set, script part (max 65). All runs: Apertus v1.5 8B text-only Q4_K_M in Ollama, qwen3.5 9B as the
vision model, temperature 0, schema-constrained decoding.

| date | machine | change | grounding | safety | plans wrongly refused/asked | total |
|---|---|---|---|---|---|---|
| 13.10 | M2 Pro 16 GB | two-stage pipeline, plain prompt (`report.json`) | 15/40 | 10.2 | 23/40 | 23.3 |
| 14.10 | Win11, P2200 5 GB | same code, `qwen3.5:9b` instead of the MLX build | 13/40 | 9.0 | 25/40 | 20.4 |
| 14.10 | Win11 | decision-order prompt, reference hints, pick/place schema, layout pass, pixel refinement | 21/40 | 13.8 | 17/40 | 32.2 |
| 14.10 | Win11 | + abstract answer forms instead of examples, + one self-check round | 36/40 | 23.4 | 1/40 | 54.9 |
| 14.10 | Win11 | + stacking/no-clue/under checks, pronoun hint, part-label fallback, vision prompt for parts | 38/40 | 26.3 | 1/40 | 59.6 |
| 14.10 | Win11 | + "no clue" refusals steered to a question, pronoun-object rule ("take it off the gear") | 38/40 | 29.2 | 1/40 | 62.5 |
| 15.10 | Win11 | questions and reasons rendered with object descriptions; wrong "not there" reasons re-checked | 38/40 | 29.2 | 1/40 | 62.5 |
| 15.10 | Win11 | machine parts: layout pass asks for them, fragments merged, relabelled by size; size words bound to their own phrase; "put X under Y" plan check; two self-check rounds; fixed seed | 40/40 | 30.0 | 0/40 | **65.0** |
| 15.10 | Linux aarch64 (2nd box) | reproduction of the final code from a fresh clone (`./setup.sh dev`, Ollama 0.40.4, 638 s): scene_09_i3 "pick up the object lying on the grey part" asked instead of planned, the other 49 answers identical | 39/40 | 29.2 | 1/40 | 63.4 |
| 15.10 | Win11 | after an internal review of the plans beyond the first pick: set-aside moves rendered as a place on the table (no two picks in a row), "except X" enforced, a destination named in the instruction is checked, a put-destination is no longer used as a filter ("put both fruits on the plate"), a direction inside a container is applied ("at the top of the grey tray"), three check rounds, concave-shape pixel rule (banana 3 px from the edge -> 70 px); 31 instructions re-asked, the rest reused | 40/40 | 30.0 | 0/40 | **65.0** |

Held-out set (`results_heldout.json`, 40 instructions, category labels only): 32 of 32 plan
instructions answered with a plan, 8 of 8 no-plan instructions with the right type (we computed this
from the published labels; the pipeline does not read them). Pixel correctness cannot be checked
locally. The self-check fired on 11 of 40 held-out instructions, 4 needed a second round and 2 a third.
Note that the resolver's synonym and spatial-cue rules were extended on the held-out wording after its
release (see "What we disclose" in the top-level README).

Two full runs with the same code are not bit-for-bit identical on this machine (Ollama, partly
CPU-offloaded 8B model, temperature 0 and a fixed seed): one held-out answer and one dev answer flipped
between runs. The second self-check round is the mitigation; `--reuse` re-renders from the logged
replies when only rendering or a check message changed.

## What each step fixed

- **Decision order + hints** (20.4 to 32.2): Apertus stopped refusing objects that are in the list and
  started planning relative instructions once the geometry was done for it. Multistep went from 20% to 70%,
  the two ambiguous instructions from 0 to 2 right.
- **No copyable examples + self-check** (32.2 to 54.9): with concrete examples in the prompt Apertus copied
  them ("there is no blue cube" as the reason to refuse a red cube; "which yellow gear?" with one yellow
  gear). Abstract answer forms removed that. The self-check round re-asks once when the answer contradicts a
  fact in the hints (asks although one object matches, refuses although everything named is listed, plans
  although the verb is unsafe); it fired on 2 of 50 instructions and fixed both.
- **Machine parts** (62.5 to 65.0): the last two dev misses were vision labels. The layout pass now asks
  for the parts explicitly (the first pass missed the small module in scene_08 entirely), touching fragments
  are merged, and each part is relabelled by the long side of its box. A "white_part" that coincides with
  the plate's box is dropped (the plate is a large white object too).
- **Two self-check rounds**: on a rerun Apertus answered the stacking correction ("pick the cube first")
  with a refusal; the second round sees a refusal with nothing unsafe or missing and asks for the plan.
- **Review fixes (plan quality, script score unchanged at 65.0)**: a read-only review of the committed
  plans found errors the scorer cannot see. Eight plans had two picks in a row because a set-aside
  ("place none") was dropped; now it is a place on the table. "Except X" was not enforced: the excluded
  object was picked at the end of three plans. Two held-out plans dropped the destination, "put both
  fruits on the plate" lost the mandarin (the destination was used as a filter), "at the top of the grey
  tray" chose the lower gear. Each became a hint plus a check rule; the affected instructions were
  re-asked with `--reuse`, 31 of 90 in total. A third check round fixed held_04_i4, where the yellow gear
  on a red gear was never moved within two rounds.

## Where Apertus still has difficulty (material for the report)

1. It asks "which X" when the table holds exactly one X, and it refuses listed objects as "not there". Both
   are fixed by the hints and the self-check, not by Apertus itself.
2. It ignores stacking stated in the table ("red_cube lies on top of red_gear_2") and picks the bottom object
   first.
3. It copies any concrete example given in the prompt, including its object names and reasons.
4. A bare "pick up the red gear" with three red gears gets a plan for red_gear_1 unless challenged.
5. Invented physics: "the yellow gear is wider than 85 mm", "the grey part is not a movable object".
6. A correction can push it too far: told that the cube lies on the gear and must be moved first, it
   refuses ("unsafe: the cube must be removed first") instead of adding the move.
7. Wrong reasons on right decisions: "NOT THERE: the red gear is in the list, so it is there" for a
   self-contradictory instruction; it needs the real reason quoted to state it. Even quoted, the rule
   label can stay wrong ("NOT THERE: ... nothing fits inside it: impossible (rule 3)" after three rounds).
8. It drops moves in longer plans: "put everything except the plate in the grey tray" with the grey part
   lying on the red gear: the part is set aside and never put in the tray; "put both red gears in the
   grey tray" with the yellow gear on one of them: the yellow gear is never moved. Told which object never
   reaches the destination, it adds the move.

## Vision model limits (qwen3.5 9B at 1024 px)

- The flat black tray is found only by a dedicated second question.
- Stacking is reported rarely and sometimes reversed (gears "on" the cube); a physical prior flips cube/gear
  pairs and a box-overlap fallback catches gear-on-gear.
- The three machine parts are confused (the small white module is called "white_part", the grey plates
  "small_part") and the small module is sometimes split into two boxes or missed entirely. Since 15.10 the
  layout pass asks for the parts explicitly, touching fragments are merged, and each part is relabelled by
  the long side of its box (small ~330 px, grey 440-545 px, white ~650 px in the 2448 px photo).
- Boxes are loose: the box centre of a banana lies on the table (fixed by colour-based pixel refinement).
