# Run log

Dev set, script part (max 65). All runs: Apertus v1.5 8B text-only Q4_K_M in Ollama, qwen3.5 9B as the
vision model, temperature 0, schema-constrained decoding.

| date | machine | change | grounding | safety | plans wrongly refused/asked | total |
|---|---|---|---|---|---|---|
| 13.10 | M2 Pro 16 GB | two-stage pipeline, plain prompt (`report.json`) | 15/40 | 10.2 | 23/40 | 23.3 |
| 14.10 | Win11, P2200 5 GB | same code, `qwen3.5:9b` instead of the MLX build | 13/40 | 9.0 | 25/40 | 20.4 |
| 14.10 | Win11 | decision-order prompt, reference hints, pick/place schema, layout pass, pixel refinement | 21/40 | 13.8 | 17/40 | 32.2 |
| 14.10 | Win11 | + abstract answer forms instead of examples, + one self-check round | 36/40 | 23.4 | 1/40 | 54.9 |
| 14.10 | Win11 | + stacking/no-clue/under checks, pronoun hint, part-label fallback, vision prompt for parts | 38/40 | 26.3 | 1/40 | 59.6 |
| 14.10 | Win11 | + "no clue" refusals steered to a question, pronoun-object rule ("take it off the gear") | 38/40 | 29.2 | 1/40 | **62.5** |

15.10: questions and refusal reasons render every internal name as a description built from the geometry
("the red gear on the left, in the grey tray (red_gear_1)"), and a refusal whose reason says "not there"
although everything named is listed gets one more self-check round. Scores unchanged by design; see the
table below for the re-run.

The two remaining grounding misses are vision labels: the three machine parts (grey, white, small white)
are confused by the vision model, so "the grey part closest to the mug" and "the small white part" land on
the wrong part or are reported missing.

## What each step fixed

- **Decision order + hints** (20.4 to 32.2): Apertus stopped refusing objects that are in the list and
  started planning relative instructions once the geometry was done for it. Multistep went from 20% to 70%,
  the two ambiguous instructions from 0 to 2 right.
- **No copyable examples + self-check** (32.2 to 54.9): with concrete examples in the prompt Apertus copied
  them ("there is no blue cube" as the reason to refuse a red cube; "which yellow gear?" with one yellow
  gear). Abstract answer forms removed that. The self-check round re-asks once when the answer contradicts a
  fact in the hints (asks although one object matches, refuses although everything named is listed, plans
  although the verb is unsafe); it fired on 2 of 50 instructions and fixed both.

## Where Apertus still has difficulty (material for the report)

1. It asks "which X" when the table holds exactly one X, and it refuses listed objects as "not there". Both
   are fixed by the hints and the self-check, not by Apertus itself.
2. It ignores stacking stated in the table ("red_cube lies on top of red_gear_2") and picks the bottom object
   first.
3. It copies any concrete example given in the prompt, including its object names and reasons.
4. A bare "pick up the red gear" with three red gears gets a plan for red_gear_1 unless challenged.
5. Invented physics: "the yellow gear is wider than 85 mm", "the grey part is not a movable object".

## Vision model limits (qwen3.5 9B at 1024 px)

- The flat black tray is found only by a dedicated second question.
- Stacking is reported rarely and sometimes reversed (gears "on" the cube); a physical prior flips cube/gear
  pairs and a box-overlap fallback catches gear-on-gear.
- The small white part is often labelled as a grey part; the resolver falls back to the smallest part seen.
- Boxes are loose: the box centre of a banana lies on the table (fixed by colour-based pixel refinement).
