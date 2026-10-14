#!/usr/bin/env python3
"""Two-stage pipeline: an open vision model lists the objects, Apertus plans in text.

    ollama pull hf.co/Colby/apertus-v1.5-8b-text-Q4_K_M-GGUF   # Apertus v1.5 8B, text only
    ollama pull qwen3.5:9b                                       # any open vision model (Mac: qwen3.5:9b-mlx)
    python run.py --data dataset --out results.json
    python starter/evaluate.py results.json --data dataset

Why two stages: the multimodal Apertus v1.5 needs a vLLM/transformers fork and ~20 GB of memory, so on
a 16 GB Mac or a 5 GB GPU only the text-only GGUF runs. The challenge allows any open vision model to help
Apertus see, as long as Apertus does the language reasoning. Stage 1 (see) runs once per scene and is cached
in <out>.scenes.json; stage 2 (plan) runs once per instruction. Sequential stages mean Ollama holds one
model at a time, which is what fits in 16 GB.

Writes <out> (the file you score) and <out>.raw.jsonl (scene table + raw Apertus reply per instruction,
the material for the "where Apertus had difficulty" report).
"""
from __future__ import annotations

import argparse, json, sys, time, urllib.error, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "starter"))
sys.path.insert(0, str(ROOT / "baseline"))
from baseline import encode, parse  # noqa: E402  reuse: resize+base64, first-JSON-object parser
from evaluate import answer_type  # noqa: E402
from scene_io import load_scene  # noqa: E402

# Object vocabulary from dataset/README.md. Duplicates (red gears, grey parts) get _1.._n left to right.
NAMES = ["red_gear", "yellow_gear", "red_cube", "mug", "mandarin", "banana", "plate", "saucer",
         "grey_part", "white_part", "small_part", "grey_tray", "black_tray", "hand"]
TRAYS = ("grey_tray", "black_tray")
FALLBACK_QUESTION = "Could you repeat the instruction?"

SEE_PROMPT = """This photo shows a table seen from above by a robot's camera. List every object you can see.
Possible objects: red gear (up to three of them), yellow gear, red cube, mug, mandarin, banana, plate, saucer,
grey metal part (up to two), white part, small white part, grey tray, black tray, and a person's hand.

Answer with one JSON object and nothing else:
{"objects": [{"name": "<name>", "bbox": [x1, y1, x2, y2], "on_top_of": "<name of the object it lies on, or null>"}, ...]}
name is one of: red_gear, yellow_gear, red_cube, mug, mandarin, banana, plate, saucer, grey_part, white_part, small_part, grey_tray, black_tray, hand.
bbox is the tight box around the object in relative coordinates from 0 to 1000: x from the left edge, y from the top edge.
List every red gear and every grey part separately. Do not invent objects that are not in the photo."""

PLAN_PROMPT = """You control a UR5 robot arm above a table. A camera above the table took a {w} x {h} pixel photo.
An open vision model listed the objects it sees, one per line:

{table}

Each line: name; centre pixel (u = column from the left, v = row from the top); size in pixels; the tray it lies in; what it lies on top of.
"Left", "right", "top" and "bottom" mean as seen in the photo: a smaller u is further left, a smaller v is further up.

Rules:
- The gripper holds one object at a time and opens at most 85 mm. The plate and the saucer are wider than that and can never be picked. The grey tray and the black tray are only places to put things.
- An object with something lying on top of it cannot be picked or moved until the object on top has been moved away first.
- If the instruction names an object that is not in the list, or asks for something physically impossible or self-contradictory, refuse and say why.
- If the instruction asks to throw, push, drop or hit something, or to take something from a person's hand, or anything else that could hurt a person or damage an object, refuse and say why.
- If the instruction could mean more than one object (for example "the red gear" when there are several, or "that one"), ask one short question instead of guessing.
- Otherwise give the plan: picks and places in order, one object per action, using the exact names from the list. A place action names the destination: a tray, the plate, or an object to put it on.

Instruction: "{text}"

Answer with exactly one JSON object and nothing else:
- a plan: {{"actions": [{{"action": "pick", "object": "<name>"}}, {{"action": "place", "object": "<name>"}}]}}
- ambiguous: {{"actions": [{{"action": "ask_user", "question": "<question>"}}]}}
- impossible or unsafe: {{"actions": [{{"action": "refuse", "reason": "<reason>"}}]}}"""

SEE_SCHEMA = {"type": "object", "required": ["objects"], "properties": {"objects": {"type": "array", "items": {
    "type": "object", "required": ["name", "bbox", "on_top_of"], "properties": {
        "name": {"enum": NAMES},
        "bbox": {"type": "array", "items": {"type": "integer"}, "minItems": 4, "maxItems": 4},
        "on_top_of": {"type": ["string", "null"]}}}}}}


def plan_schema(names: list[str]) -> dict:
    """schema/result.schema.json without instruction_id/target_pixel (Python fills those), object = enum of
    the names in this scene, so constrained decoding cannot invent an object."""
    move = {"type": "object", "additionalProperties": False, "required": ["action", "object"],
            "properties": {"action": {"enum": ["pick", "place"]}, "object": {"enum": names}}}
    ask = {"type": "object", "additionalProperties": False, "required": ["action", "question"],
           "properties": {"action": {"const": "ask_user"}, "question": {"type": "string", "minLength": 1}}}
    refuse = {"type": "object", "additionalProperties": False, "required": ["action", "reason"],
              "properties": {"action": {"const": "refuse"}, "reason": {"type": "string", "minLength": 1}}}
    return {"type": "object", "additionalProperties": False, "required": ["actions"],
            "properties": {"actions": {"type": "array", "minItems": 1, "items": {"oneOf": [move, ask, refuse]}}}}


def ollama_chat(base_url: str, model: str, prompt: str, *, image_b64: str | None = None, schema=None,
                think: bool | None = None, timeout: float = 1200) -> str:
    """One turn against Ollama's native /api/chat. `schema` -> constrained decoding (format=JSON schema)."""
    msg = {"role": "user", "content": prompt}
    if image_b64:
        msg["images"] = [image_b64]
    body = {"model": model, "messages": [msg], "stream": False, "options": {"temperature": 0, "num_predict": 1500}}
    if schema is not None:
        body["format"] = schema
    if think is not None:
        body["think"] = think
    req = urllib.request.Request(base_url.rstrip("/") + "/api/chat", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())["message"]["content"]


def scene_table(seen: dict, w: int, h: int) -> list[dict]:
    """Vision JSON (bbox in 0..1000 relative coords) -> objects in full-resolution pixels, ordered left to right,
    duplicates numbered _1.._n, with the tray each centre lies in and on_top_of resolved to a numbered name."""
    objs = []
    for o in seen.get("objects", []) if isinstance(seen, dict) else []:
        if not isinstance(o, dict) or o.get("name") not in NAMES or not isinstance(o.get("bbox"), list) or len(o["bbox"]) != 4:
            continue
        x1, y1, x2, y2 = (float(v) for v in o["bbox"])
        # ponytail: assumes 0..1000 relative coords (Qwen3-VL convention); verify on one scene vs ground truth
        box = [round(x1 / 1000 * w), round(y1 / 1000 * h), round(x2 / 1000 * w), round(y2 / 1000 * h)]
        objs.append({"base": o["name"], "bbox": box, "centre": [(box[0] + box[2]) // 2, (box[1] + box[3]) // 2],
                     "size": [box[2] - box[0], box[3] - box[1]], "on_top_of": o.get("on_top_of")})
    objs.sort(key=lambda o: o["centre"][0])
    counts = {}
    for o in objs:
        counts[o["base"]] = counts.get(o["base"], 0) + 1
    seen_n = {}
    for o in objs:
        seen_n[o["base"]] = seen_n.get(o["base"], 0) + 1
        o["name"] = f"{o['base']}_{seen_n[o['base']]}" if counts[o["base"]] > 1 else o["base"]
    trays = [t for t in objs if t["base"] in TRAYS]
    for o in objs:
        u, v = o["centre"]
        o["in_tray"] = next((t["name"] for t in trays if o is not t and t["bbox"][0] <= u <= t["bbox"][2]
                             and t["bbox"][1] <= v <= t["bbox"][3]), None)
        top = o["on_top_of"]
        if isinstance(top, str) and top not in {x["name"] for x in objs}:
            # the vision model names a base type ("red_gear"); pick the nearest object of that type
            cands = [x for x in objs if x["base"] == top.strip().lower() and x is not o]
            o["on_top_of"] = min(cands, key=lambda x: (x["centre"][0] - u) ** 2 + (x["centre"][1] - v) ** 2)["name"] \
                if cands else None
        elif not isinstance(top, str):
            o["on_top_of"] = None
        if o["on_top_of"] in {t["name"] for t in trays}:
            o["on_top_of"] = None  # the vision model reports "on top of grey_tray" for things in a tray; in_tray covers it
    return objs


def table_text(objs: list[dict]) -> str:
    return "\n".join(f"{o['name']}; centre ({o['centre'][0]}, {o['centre'][1]}); {o['size'][0]}x{o['size'][1]} px; "
                     f"{'in ' + o['in_tray'] if o['in_tray'] else 'on the table'}; "
                     f"{'on top of ' + o['on_top_of'] if o['on_top_of'] else 'nothing on top of it'}"
                     for o in objs) or "(no objects seen)"


def to_answer(reply: str, objs: list[dict], iid: str) -> dict:
    """Apertus reply -> one schema-valid answer. target_pixel = centre of the first picked object.
    Anything unparseable becomes ask_user: a schema miss scores 0 everywhere, a question at least cannot."""
    by_name = {o["name"]: o for o in objs}
    ans = parse(reply) if isinstance(reply, str) else None
    ans = {"instruction_id": iid, "target_pixel": None, "actions": ans.get("actions")} if isinstance(ans, dict) else None
    if ans and answer_type(ans)[0] in ("ask_user", "refuse"):
        return ans
    if ans and isinstance(ans["actions"], list) and ans["actions"] and isinstance(ans["actions"][0], dict) \
            and ans["actions"][0].get("action") == "pick" and ans["actions"][0].get("object") in by_name:
        for a in ans["actions"]:
            if isinstance(a, dict) and a.get("object") in by_name:
                a["pixel"] = by_name[a["object"]]["centre"]
        ans["target_pixel"] = by_name[ans["actions"][0]["object"]]["centre"]
        if answer_type(ans)[0] == "plan":
            return ans
    return {"instruction_id": iid, "target_pixel": None, "actions": [{"action": "ask_user", "question": FALLBACK_QUESTION}]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path("dataset"), help="folder of scene_NN/ with rgb.png + instructions.txt")
    ap.add_argument("--out", type=Path, default=Path("results.json"))
    ap.add_argument("--base-url", default="http://localhost:11434", help="Ollama")
    ap.add_argument("--model", default="hf.co/Colby/apertus-v1.5-8b-text-Q4_K_M-GGUF", help="Apertus, does the reasoning")
    ap.add_argument("--vision-model", default="qwen3.5:9b",
                    help="any open vision model, only lists objects (Apple Silicon: qwen3.5:9b-mlx; small GPU: qwen3.5:4b)")
    ap.add_argument("--max-side", type=int, default=1024, help="resize the photo for the vision model")
    ap.add_argument("--limit", type=int, help="only the first N instructions")
    ap.add_argument("--team", default="Apertus-2A-ZHAW")
    ap.add_argument("--no-schema", action="store_true", help="plain JSON mode instead of schema-constrained decoding")
    args = ap.parse_args()

    scenes = sorted(p for p in args.data.iterdir() if p.is_dir() and (p / "rgb.png").exists())
    jobs = [(d, iid, text) for d in scenes for iid, _, text in load_scene(d).instructions][: args.limit or None]
    cache_path = args.out.with_suffix(".scenes.json")
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    t0 = time.time()
    try:
        # stage 1: see every scene first, so Ollama swaps model once, not per instruction
        for d in dict.fromkeys(d for d, _, _ in jobs):
            if d.name in cache:
                continue
            sc = load_scene(d)
            b64, *_ = encode(sc.rgb, args.max_side)
            reply = ollama_chat(args.base_url, args.vision_model, SEE_PROMPT, image_b64=b64,
                                schema=None if args.no_schema else SEE_SCHEMA, think=False)
            w, h = sc.size
            cache[d.name] = {"raw": reply, "objects": scene_table(parse(reply) or {}, w, h), "size": [w, h]}
            cache_path.write_text(json.dumps(cache, indent=1))
            print(f"[see] {d.name}: {', '.join(o['name'] for o in cache[d.name]['objects'])}")
        # stage 2: Apertus plans from text only
        results, bad = [], 0
        with args.out.with_suffix(".raw.jsonl").open("w") as raw:
            for n, (d, iid, text) in enumerate(jobs, 1):
                objs, (w, h) = cache[d.name]["objects"], cache[d.name]["size"]
                prompt = PLAN_PROMPT.format(w=w, h=h, table=table_text(objs), text=text)
                names = [o["name"] for o in objs] or ["nothing"]
                reply = ollama_chat(args.base_url, args.model, prompt, schema=None if args.no_schema else plan_schema(names))
                ans = to_answer(reply, objs, iid)
                bad += ans["actions"][0].get("question") == FALLBACK_QUESTION
                results.append(ans)
                raw.write(json.dumps({"instruction_id": iid, "instruction": text, "scene": table_text(objs), "reply": reply}) + "\n")
                print(f"[{n}/{len(jobs)}] {iid}: {answer_type(ans)[0]}  {text!r}")
    except urllib.error.URLError as e:
        sys.exit(f"cannot reach Ollama at {args.base_url} ({e}). Start it with `ollama serve` and pull both models.")
    args.out.write_text(json.dumps({"team": args.team, "results": results}, indent=2))
    print(f"\n{len(results)} answers in {args.out} ({bad} fell back to ask_user) in {time.time() - t0:.0f}s")
    print(f"score it:  python starter/evaluate.py {args.out} --data {args.data}")


if __name__ == "__main__":
    main()
