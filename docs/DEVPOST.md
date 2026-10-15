# Devpost submission text: See It, Say It, Pick It (team Gude2Learn)

Plain text for the Devpost form. Numbers and claims match `README.md`, `docs/REPORT.md` and `attempts/README.md`.

## Inspiration

The ZHAW challenge at Hack Apertus 2026: one photo of a robot workcell and one English sentence
go in, the pixel of the object to pick first and the ordered actions come out; an ambiguous
instruction gets a question, an impossible or unsafe one a refusal. We wanted the smallest Apertus
(8B, 4-bit, text only) to write every answer on a laptop or a 5 GB GPU, with nothing leaving the
machine, and to measure honestly how much of each answer is the model and how much is Python.

## What it does

Input: a 2448 x 2048 photo and an instruction such as "put the red gear under the yellow gear in
the grey tray". Output: the target pixel of the first object picked plus pick/place steps, or one
question naming the choices, or one refusal with the rule and the reason. Dev set (50
instructions, official scorer): 65.0 of 65, grounding 40/40, safety 30/30, up from 20.4 with a
plain prompt. Held-out set (40 instructions): 32 of 32 plans answered with a plan, 8 of 8 no-plan
cases with the right answer type, computed by us from the published labels; pixel correctness
there is unknown. Everything runs locally in Ollama; one command runs any folder of scenes.

## How we built it

Six steps, one model writes the answer. (1) See: qwen3.5 9B, an open vision model, lists the
objects with boxes; a second call asks only about stacking, trays and the machine parts. (2)
Table: Python turns boxes into a scene table (duplicates numbered in reading order, which tray
each object lies in, what lies on it, the pixel moved onto the object by colour). (3) Resolve:
Python maps the noun phrases and spatial cues ("on the left", "closest to the mug", "in the grey
tray", "with nothing on it") to the table and writes them as reference hints. (4) Plan: Apertus
v1.5 8B text-only (community Q4_K_M GGUF of the swiss-ai weights) reads the table, the hints and
a fixed decision order (unsafe, not there, cannot, ambiguous, plan) and answers under a per-scene
JSON schema. (5) Self-check: when the answer contradicts a fact in the hints, Apertus sees its
own answer and the contradiction and answers again, at most three rounds. (6) Render: steps
become the submission format, a set-aside becomes a place on the table, and every internal name
in a question or reason becomes a description a person can act on. `setup.sh` / `setup.ps1`
go from a fresh clone to a scored result. Claude Code was used throughout, for the pipeline,
the reviews and the report; the team is Dawn Duan and Claude Code.

## Challenges we ran into

- Multimodal Apertus does not fit a 16 GB Mac or a 5 GB GPU (vLLM fork, about 20 GB), so the
  text-only build plus a separate open vision model.
- Apertus 8B over-refuses with confident false reasons ("the red cube is not in the list" when it
  is), cannot do geometry from numbers (7% of relative instructions right), copies any example in
  the prompt (few-shot made the score worse, 23.3 to 17.7), ignores stacking stated in the table,
  loses moves in long plans, and over-corrects into a refusal after a correction.
- The vision model confuses the three machine parts, misses the flat black tray in 7 of 15
  tables and reports stacking rarely; one merged box flipped fresh runs on other machines from
  65.0 to 63.4 until a part-reference fallback closed it.
- On the Mac, the MLX build of the vision model under Ollama 0.32 ignores the JSON schema (fenced
  replies, a bare array instead of the object list, missing trays): 61.75 instead of 65.0 and 10
  of 40 held-out answer types wrong. Recorded, GGUF build recommended, fix left as future work.
- Reproducibility is a claim you have to earn: five full runs on three machines, every one in
  the run log with its score and cause.

## Accomplishments that we're proud of

- 20.4 to 65.0 on the dev set with an 8B 4-bit text-only model on a 5 GB GPU; a fresh clone on
  a Linux box reproduces 65.0 with every category at 100%.
- Every answer is written by Apertus (no malformed answer in 90); Python's two fallbacks never
  fired. Measured ablations: no self-check 50.4, no hints 48.8, plain prompt 20.4.
- Questions and refusals a person can act on: "Which red gear should I pick up: the leftmost red
  gear (red_gear_1), the middle red gear (red_gear_2), or the rightmost red gear (red_gear_3)?"
- Full disclosure in the README and report: the resolver was extended on the held-out wording,
  the committed answers are a `--reuse` re-render, what is Apertus and what is Python,
  cross-machine nondeterminism, the Mac run.

## What we learned

- A small model decides well when counting, geometry and the answer grammar are taken off its
  plate. Show it its own contradiction instead of a correct example: abstract answer forms plus
  the self-check were the largest gain, 32.2 to 54.9.
- Constrained decoding is load-bearing: a runner that ignores the schema loses the trays, two
  dev answers and most of two held-out scenes.
- Vision is the ceiling: Apertus cannot repair a wrong label, and one box decides between 65.0
  and 63.4.
- The script score hides plan quality: a read-through of the plans found pick-pick sequences,
  "except" objects moved and dropped destinations, each now a hint plus a check rule.

## What's next for Gude2Learn

Geometric tray detection (fixed camera, known tray size), containment and stacking from pixels
instead of boxes, a parser that takes a bare array and a stack guard (the two Mac fixes), an
out-of-distribution wording check, Apertus 70B, think mode, a stronger vision model, and a split
of the one module into see, resolve, plan and render. Details in `docs/REPORT.md`, "Future work".
Code: github.com/jdawnduan/Apertus-2A-ZHAW (Apache-2.0).
