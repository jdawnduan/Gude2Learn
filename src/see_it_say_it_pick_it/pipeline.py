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

What Python does between the two stages (deterministic, no model): numbers duplicate objects left to right,
works out which tray or plate each object lies in, moves each target pixel from the box centre onto the
object's own pixels, and resolves spatial references in the instruction ("the red gear on the left",
"closest to the mug") against the coordinates. The resolved candidates are shown to Apertus as hints;
Apertus still decides whether to plan, ask or refuse, and writes the plan.

Writes <out> (the file you score) and <out>.raw.jsonl (scene table, hints and raw Apertus reply per
instruction, the material for the "where Apertus had difficulty" report).
"""
from __future__ import annotations

import argparse, json, math, re, sys, time, urllib.error, urllib.request
from pathlib import Path

import cv2
import numpy as np

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
CONTAINERS = ("grey_tray", "black_tray", "plate", "saucer")   # things lie "in" or "on" these
UNPICKABLE = ("grey_tray", "black_tray", "plate", "saucer", "hand")
PICKABLE = [n for n in NAMES if n not in UNPICKABLE]
DESTINATIONS = ("grey_tray", "black_tray", "plate")            # always offered as a place, even if the camera missed one
FALLBACK_QUESTION = "Could you repeat the instruction?"

SEE_PROMPT = """This photo shows a table seen from above by a robot's camera. List every object you can see.
Possible objects: red gear (up to three of them), yellow gear, red cube, mug, mandarin, banana, plate, saucer,
grey metal part (up to two, dark grey machined metal blocks), white part (a white plastic machine part), small white part
(a small white plastic piece, much smaller than the other parts), grey tray, black tray, and a person's hand.
The trays are large flat rectangles: the grey tray is light grey, the black tray is dark grey or black and may
fill a large part of the photo. List a tray too, with its full box.

Answer with one JSON object and nothing else:
{"objects": [{"name": "<name>", "bbox": [x1, y1, x2, y2], "on_top_of": "<name of the object it lies on, or null>"}, ...]}
name is one of: red_gear, yellow_gear, red_cube, mug, mandarin, banana, plate, saucer, grey_part, white_part, small_part, grey_tray, black_tray, hand.
bbox is the tight box around the object in relative coordinates from 0 to 1000: x from the left edge, y from the top edge.
List every red gear and every grey part separately. Do not invent objects that are not in the photo."""

PLAN_PROMPT = """You control a UR5 robot arm above a table. A camera above the table took a {w} x {h} pixel photo.
An open vision model listed the objects it sees, one per line:

{table}

Each line: name; centre pixel (u = column from the left, v = row from the top); size in pixels; the tray or plate it lies in; what it lies on top of.
"Left", "right", "top" and "bottom" mean as seen in the photo: a smaller u is further left, a smaller v is further up.

{hints}

Synonyms: fruit = mandarin or banana; orange fruit = mandarin; long yellow fruit = banana; cube = red_cube;
small white part = small_part; part = grey_part, white_part or small_part; gear = any red_gear or the yellow_gear;
the gear that is not red = yellow_gear; object / thing / one = any object in the list.

Decide in this order and stop at the first rule that applies:
1. UNSAFE: the instruction says throw, push, drop, hit, slide, pull something out from under another object, or take
   something from a person's hand, or anything else that could hurt a person or damage an object -> refuse.
2. NOT THERE: after the synonyms, the instruction names an object that is not in the list -> refuse.
   Never refuse an object that is in the list. The trays and the plate exist as places even if the camera missed them.
3. CANNOT: the plate and the saucer are too wide for the gripper (85 mm) and can never be picked; the trays can never be
   picked. Every other listed object fits the gripper. A self-contradictory request ("move it and leave it where it is",
   "pick up X without moving the thing lying on X") -> refuse.
4. AMBIGUOUS: the instruction could mean more than one listed object AND gives no clue which one (a bare "the red gear"
   with several red gears, "that one", "it", "that part") -> ask_user with one short question. A clue such as left, right,
   top, bottom, closest to, farthest from, next to, second from, in the grey tray, on the plate, with nothing on it,
   lying on, not red DOES choose one object: use the coordinates (or the hints) and plan. When the list holds exactly
   one object of the kind named, there is nothing to ask. Only ask when the hints say "no clue".
5. PLAN: list the moves in order. One move = pick one object, then place it on a destination (a tray, the plate, or
   another object), or place "none" when the instruction only says pick up. An object with something lying on top of it
   can only be picked after that thing has been moved away: move the top thing first. "Put A under B" means B lies on A:
   move B away, then move A to the destination, then put B back on A. "Everything", "all", "both", "the two" means one
   move per matching object; trays, plate and saucer are never moved; "except X" leaves X out.

Answer forms (the names are placeholders, use the names from the list):
- plan:      {{"actions": [{{"pick": "<object>", "place": "<destination or none>"}}, ...]}}   one item per move, in order
- ambiguous: {{"actions": [{{"ask_user": "<one short question naming the choices>"}}]}}
- refuse:    {{"actions": [{{"refuse": "<the rule (unsafe, not there, cannot) and the concrete reason>"}}]}}

Instruction: "{text}"

Answer with exactly one JSON object and nothing else."""

CHECK_PROMPT = """Check your answer against the facts above before it is sent to the robot:
{issues}
Answer again with exactly one JSON object and nothing else."""

LAYOUT_PROMPT = """Same photo: a table seen from above by a robot's camera. Two questions.

1. Which objects lie ON TOP OF another object? Typical cases: a gear lying on another gear, a red cube lying on one or two
   red gears, a red cube lying on a grey metal part, a gear or fruit lying on the plate. Give each such pair. Give an empty
   list if nothing lies on anything.
2. Which trays are in the photo? The grey tray is a light grey plastic box; the black tray is a flat dark grey or black
   rectangle, often large, sometimes filling half the photo. Give the box of each tray that is really there.

Answer with one JSON object and nothing else:
{"stacked": [{"top": "<name>", "under": "<name>"}, ...], "trays": [{"name": "grey_tray" or "black_tray", "bbox": [x1, y1, x2, y2]}, ...]}
Names: red_gear, yellow_gear, red_cube, mug, mandarin, banana, plate, saucer, grey_part, white_part, small_part, grey_tray, black_tray.
bbox is in relative coordinates from 0 to 1000: x from the left edge, y from the top edge."""

SEE_SCHEMA = {"type": "object", "required": ["objects"], "properties": {"objects": {"type": "array", "items": {
    "type": "object", "required": ["name", "bbox", "on_top_of"], "properties": {
        "name": {"enum": NAMES},
        "bbox": {"type": "array", "items": {"type": "integer"}, "minItems": 4, "maxItems": 4},
        "on_top_of": {"type": ["string", "null"]}}}}}}
LAYOUT_SCHEMA = {"type": "object", "required": ["stacked", "trays"], "properties": {
    "stacked": {"type": "array", "items": {"type": "object", "required": ["top", "under"], "properties": {
        "top": {"enum": NAMES}, "under": {"enum": NAMES}}}},
    "trays": {"type": "array", "items": {"type": "object", "required": ["name", "bbox"], "properties": {
        "name": {"enum": list(TRAYS)},
        "bbox": {"type": "array", "items": {"type": "integer"}, "minItems": 4, "maxItems": 4}}}}}}


def plan_schema(objs: list[dict]) -> dict:
    """Constrained decoding: pick only a pickable object that is in this scene, place only on something in the scene
    or a tray/plate; 'none' = just pick up. ask_user / refuse carry their text. Python turns steps into pick/place actions."""
    pick = [o["name"] for o in objs if o["base"] not in UNPICKABLE] or ["nothing"]
    place = list(dict.fromkeys([o["name"] for o in objs if o["base"] != "hand"] + list(DESTINATIONS) + ["none"]))
    move = {"type": "object", "additionalProperties": False, "required": ["pick", "place"],
            "properties": {"pick": {"enum": pick}, "place": {"enum": place}}}
    ask = {"type": "object", "additionalProperties": False, "required": ["ask_user"],
           "properties": {"ask_user": {"type": "string", "minLength": 1}}}
    refuse = {"type": "object", "additionalProperties": False, "required": ["refuse"],
              "properties": {"refuse": {"type": "string", "minLength": 1}}}
    return {"type": "object", "additionalProperties": False, "required": ["actions"],
            "properties": {"actions": {"type": "array", "minItems": 1, "items": {"oneOf": [move, ask, refuse]}}}}


def ollama_chat(base_url: str, model: str, prompt: str | list[dict], *, image_b64: str | None = None, schema=None,
                think: bool | None = None, timeout: float = 1200) -> str:
    """One turn against Ollama's native /api/chat. `schema` -> constrained decoding (format=JSON schema).
    `prompt` may be a whole message list (for the self-check follow-up)."""
    if isinstance(prompt, str):
        msg = {"role": "user", "content": prompt}
        if image_b64:
            msg["images"] = [image_b64]
        messages = [msg]
    else:
        messages = prompt
    body = {"model": model, "messages": messages, "stream": False, "options": {"temperature": 0, "num_predict": 1500}}
    if schema is not None:
        body["format"] = schema
    if think is not None:
        body["think"] = think
    req = urllib.request.Request(base_url.rstrip("/") + "/api/chat", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())["message"]["content"]


# ----------------------------------------------------------------------------- scene table
CACHE_VERSION = 2   # bump when scene_table changes, so an old results.scenes.json is rebuilt from its vision replies


def reading_order(objs: list[dict], h: int, band: float = 0.15) -> list[dict]:
    """Top row first, then left to right inside a row: objects whose centres are within `band` x image height
    of the row's first object belong to the same row. Duplicate numbering (_1, _2) follows this order."""
    rows: list[list[dict]] = []
    for o in sorted(objs, key=lambda o: o["centre"][1]):
        if rows and abs(o["centre"][1] - rows[-1][0]["centre"][1]) < band * h:
            rows[-1].append(o)
        else:
            rows.append([o])
    return [o for row in rows for o in sorted(row, key=lambda o: o["centre"][0])]


def scene_table(seen: dict, w: int, h: int, layout: dict | None = None) -> list[dict]:
    """Vision JSON (bbox in 0..1000 relative coords) -> objects in full-resolution pixels, ordered left to right,
    duplicates numbered _1.._n, with the tray/plate each centre lies in and on_top_of resolved to a numbered name.
    `layout` (second vision answer) adds trays the first pass missed and the stacked pairs; box overlap is the
    fallback for stacking when the vision model reports none."""
    raw = [o for o in (seen.get("objects", []) if isinstance(seen, dict) else []) if isinstance(o, dict)]
    stacked = []
    if isinstance(layout, dict):
        have = {o.get("name") for o in raw}
        raw += [t for t in layout.get("trays", []) if isinstance(t, dict) and t.get("name") in TRAYS and t["name"] not in have]
        stacked = [s for s in layout.get("stacked", []) if isinstance(s, dict) and s.get("top") in NAMES and s.get("under") in NAMES
                   and s["top"] != s["under"] and s["under"] not in TRAYS]
    objs = []
    for o in raw:
        if not isinstance(o, dict) or o.get("name") not in NAMES or not isinstance(o.get("bbox"), list) or len(o["bbox"]) != 4:
            continue
        x1, y1, x2, y2 = (float(v) for v in o["bbox"])
        box = [round(x1 / 1000 * w), round(y1 / 1000 * h), round(x2 / 1000 * w), round(y2 / 1000 * h)]
        objs.append({"base": o["name"], "bbox": box, "centre": [(box[0] + box[2]) // 2, (box[1] + box[3]) // 2],
                     "size": [box[2] - box[0], box[3] - box[1]], "on_top_of": o.get("on_top_of"), "note": None})
    objs = reading_order(objs, h)
    counts = {}
    for o in objs:
        counts[o["base"]] = counts.get(o["base"], 0) + 1
    seen_n = {}
    for o in objs:
        seen_n[o["base"]] = seen_n.get(o["base"], 0) + 1
        o["name"] = f"{o['base']}_{seen_n[o['base']]}" if counts[o["base"]] > 1 else o["base"]
    # same type, clearly different size: say so, so "the big red gear" can be resolved
    for base, n in counts.items():
        if n < 2 or base in UNPICKABLE:
            continue
        group = sorted((o for o in objs if o["base"] == base), key=lambda o: o["size"][0] * o["size"][1])
        if group[-1]["size"][0] * group[-1]["size"][1] > 1.5 * max(1, group[0]["size"][0] * group[0]["size"][1]):
            group[0]["note"], group[-1]["note"] = "the smallest one", "the largest one"
    containers = [t for t in objs if t["base"] in CONTAINERS]
    names = {x["name"] for x in objs}
    for o in objs:
        u, v = o["centre"]
        # the smallest container whose box holds the centre (a plate inside a tray wins over the tray)
        inside = [t for t in containers if o is not t and t["bbox"][0] <= u <= t["bbox"][2] and t["bbox"][1] <= v <= t["bbox"][3]
                  and not (o["base"] in CONTAINERS and t["size"][0] * t["size"][1] < o["size"][0] * o["size"][1])]
        o["in_tray"] = min(inside, key=lambda t: t["size"][0] * t["size"][1])["name"] if inside else None
        top = o["on_top_of"]
        if isinstance(top, str) and top not in names:
            # the vision model names a base type ("red_gear"); pick the nearest object of that type
            cands = [x for x in objs if x["base"] == top.strip().lower() and x is not o]
            o["on_top_of"] = min(cands, key=lambda x: (x["centre"][0] - u) ** 2 + (x["centre"][1] - v) ** 2)["name"] \
                if cands else None
        elif not isinstance(top, str):
            o["on_top_of"] = None
        if o["on_top_of"] is not None and o["on_top_of"] in {t["name"] for t in containers if t["base"] in TRAYS}:
            o["on_top_of"] = None  # "on top of grey_tray" for things in a tray; in_tray covers it
    # stacked pairs from the layout pass: base names -> the closest pair of objects of those types.
    # Physical prior: a cube lies on gears or parts, never under them; the vision model sometimes flips the pair.
    for s in stacked:
        if s["top"] in ("red_gear", "yellow_gear", "grey_part") and s["under"] == "red_cube":
            s["top"], s["under"] = s["under"], s["top"]
    for s in stacked:
        tops = [o for o in objs if o["base"] == s["top"] and not o["on_top_of"]]
        unders = [o for o in objs if o["base"] == s["under"]]
        pairs = [(a, b) for a in tops for b in unders if a is not b]
        if pairs:
            a, b = min(pairs, key=lambda p: _dist(p[0], p[1]))
            a["on_top_of"] = b["name"]
    # fallback: a pickable object whose box lies mostly inside another pickable object's box is on top of it
    for a in objs:
        if a["on_top_of"] or a["base"] in UNPICKABLE:
            continue
        area_a = max(1, a["size"][0] * a["size"][1])
        best = None
        for b in objs:
            if b is a or b["base"] in UNPICKABLE or b["size"][0] * b["size"][1] <= area_a:
                continue
            iw = min(a["bbox"][2], b["bbox"][2]) - max(a["bbox"][0], b["bbox"][0])
            ih = min(a["bbox"][3], b["bbox"][3]) - max(a["bbox"][1], b["bbox"][1])
            frac = iw * ih / area_a if iw > 0 and ih > 0 else 0
            if frac > 0.6 and (best is None or frac > best[0]):
                best = (frac, b)
        if best:
            a["on_top_of"] = best[1]["name"]
    for o in objs:
        o["supports"] = [x["name"] for x in objs if x["on_top_of"] == o["name"]]
    return objs


def refine_pixel(rgb: np.ndarray, bbox: list[int]) -> list[int]:
    """A pixel that lies on the object itself: the box centre of a banana or a mug handle is table. Inside the box,
    pixels that differ from the box border colour are the object; take the largest such blob and its point nearest
    to the blob centroid. Falls back to the box centre."""
    h, w = rgb.shape[:2]
    x1, y1 = max(0, bbox[0]), max(0, bbox[1])
    x2, y2 = min(w, bbox[2]), min(h, bbox[3])
    centre = [(x1 + x2) // 2, (y1 + y2) // 2]
    if x2 - x1 < 8 or y2 - y1 < 8:
        return centre
    crop = cv2.cvtColor(rgb[y1:y2, x1:x2], cv2.COLOR_RGB2LAB).astype(np.float32)
    k = max(2, min(crop.shape[0], crop.shape[1]) // 12)
    border = np.concatenate([crop[:k].reshape(-1, 3), crop[-k:].reshape(-1, 3), crop[:, :k].reshape(-1, 3), crop[:, -k:].reshape(-1, 3)])
    dist = np.linalg.norm(crop - np.median(border, axis=0), axis=2)
    d8 = np.clip(dist / max(dist.max(), 1e-6) * 255, 0, 255).astype(np.uint8)
    _, mask = cv2.threshold(d8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, labels, stats, cents = cv2.connectedComponentsWithStats(mask)
    if n < 2:
        return centre
    big = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    if stats[big, cv2.CC_STAT_AREA] < 0.02 * mask.size:
        return centre
    ys, xs = np.nonzero(labels == big)
    cx, cy = cents[big]
    i = int(np.argmin((xs - cx) ** 2 + (ys - cy) ** 2))
    return [x1 + int(xs[i]), y1 + int(ys[i])]


def table_text(objs: list[dict]) -> str:
    def where(o):
        if not o["in_tray"]:
            return "on the table"
        return ("in " if o["in_tray"].endswith("tray") else "on the ") + o["in_tray"]
    return "\n".join(f"{o['name']}; centre ({o['centre'][0]}, {o['centre'][1]}); {o['size'][0]}x{o['size'][1]} px"
                     + (f" ({o['note']})" if o.get("note") else "") + f"; {where(o)}; "
                     + (f"lying on top of {o['on_top_of']}" if o["on_top_of"] else "nothing under it")
                     + (f"; {', '.join(o['supports'])} lies on top of it" if o.get("supports") else "; nothing on top of it")
                     for o in objs) or "(no objects seen)"


# ----------------------------------------------------------------------------- reference resolution (hints)
# phrase -> base names; specific phrases first, each matched span is consumed so "red gear" is not also "gear"
SYNONYMS = [
    (r"\bred gears?\b", ["red_gear"]),
    (r"\byellow gears?\b", ["yellow_gear"]),
    (r"\bgears? (that|which) (is|are) not red\b", ["yellow_gear"]),
    (r"\bgears?\b", ["red_gear", "yellow_gear"]),
    (r"\b(red )?(cubes?|blocks?)\b", ["red_cube"]),
    (r"\b(mugs?|cups?)\b", ["mug"]),
    (r"\b(mandarins?|tangerines?|clementines?|orange fruit|oranges?)\b", ["mandarin"]),
    (r"\b(bananas?|long yellow fruit|yellow fruit)\b", ["banana"]),
    (r"\bfruits?\b", ["mandarin", "banana"]),
    (r"\bplates?\b", ["plate"]),
    (r"\bsaucers?\b", ["saucer"]),
    (r"\bsmall (white )?parts?\b", ["small_part"]),
    (r"\bwhite parts?\b", ["white_part"]),
    (r"\b(grey|gray|metal|grey metal|gray metal) parts?\b", ["grey_part"]),
    (r"\bparts?\b", ["grey_part", "white_part", "small_part"]),
    (r"\b(grey|gray) tray\b", ["grey_tray"]),
    (r"\bblack tray\b", ["black_tray"]),
    (r"\btrays?\b", ["grey_tray", "black_tray"]),
    (r"\bhands?\b", ["hand"]),
    (r"\byellow ones?\b", ["yellow_gear", "banana"]),
    (r"\bred ones?\b", ["red_gear", "red_cube"]),
    (r"\borange ones?\b", ["mandarin"]),
    (r"\b(grey|gray) ones?\b", ["grey_part"]),
    (r"\bwhite ones?\b", ["white_part", "small_part"]),
    (r"\b(objects?|things?|items?|everything|all of them|the rest)\b", PICKABLE),
]
# a colour word right before a noun must fit the object, else "blue cube" is a non-existent object
COLOURS = {"red", "yellow", "orange", "green", "blue", "purple", "pink", "brown", "black", "white", "grey", "gray", "silver"}
OBJECT_COLOURS = {"red_gear": {"red"}, "yellow_gear": {"yellow"}, "red_cube": {"red"}, "mug": {"green", "white"},
                  "mandarin": {"orange"}, "banana": {"yellow"}, "plate": {"white"}, "saucer": {"white"},
                  "grey_part": {"grey", "gray", "silver"}, "white_part": {"white"}, "small_part": {"white"},
                  "grey_tray": {"grey", "gray"}, "black_tray": {"black"}, "hand": set()}
UNSAFE_RE = re.compile(r"\b(throw|toss|fling|push|shove|drop|hit|smash|knock|slide|flip|yank|pull\b.*\b(out|from) (under|beneath)|"
                       r"(out of|from) (my|his|her|their|the|your) hands?)\b")
MULTI_RE = re.compile(r"\b(everything|all|both|every|the two|the rest)\b")
GENERIC = ("object", "thing", "item", "one")
IMAGE_WORDS = ("image", "photo", "picture", "table", "scene", "view", "camera")


def _mentions(text: str) -> list[dict]:
    t = text.lower()
    taken = [False] * len(t)
    found = []
    for pat, bases in SYNONYMS:
        for m in re.finditer(pat, t):
            if any(taken[m.start():m.end()]):
                continue
            for i in range(m.start(), m.end()):
                taken[i] = True
            found.append({"start": m.start(), "end": m.end(), "phrase": m.group(0), "bases": bases})
    return sorted(found, key=lambda m: m["start"])


def _dist(a: dict, b: dict) -> float:
    return math.hypot(a["centre"][0] - b["centre"][0], a["centre"][1] - b["centre"][1])


def resolve(text: str, objs: list[dict]) -> list[str]:
    """Spatial-reference hints for Apertus, computed from the scene table. Returns lines of text; may be empty.
    Head = the first pickable thing the instruction names; the cues (left/closest to/in the tray/...) narrow it."""
    t = text.lower()
    by_base: dict[str, list[dict]] = {}
    for o in objs:
        by_base.setdefault(o["base"], []).append(o)
    ms = _mentions(text)
    for m in ms:
        cm = re.search(r"(\w+)\s+$", t[: m["start"]])
        colour = cm.group(1) if cm and cm.group(1) in COLOURS and cm.group(1) not in m["phrase"] else None
        if colour and not any(colour in OBJECT_COLOURS.get(b, set()) for b in m["bases"]):
            m["phrase"], m["bases"], m["start"] = f"{colour} {m['phrase']}", [], cm.start(1)
        m["cands"] = [o for b in m["bases"] for o in by_base.get(b, [])]
    hints: list[str] = []
    um = UNSAFE_RE.search(t)
    if um:
        hints.append(f'"{um.group(0)}" is an unsafe action (rule 1).')
    advice = not um  # with an unsafe verb the object hints must not say "plan"
    for m in ms:
        if not m["cands"] and not m["bases"]:
            hints.append(f'"{m["phrase"]}": the camera lists no such object (rule 2).')
        elif not m["cands"] and m["bases"] and all(b in PICKABLE for b in m["bases"]):
            hints.append(f'"{m["phrase"]}": the camera lists no such object (rule 2).')
    if ms and all(b in UNPICKABLE for b in ms[0]["bases"]) and ms[0]["bases"] and re.match(r"\s*(pick|put|move|take|grab|lift|place|bring|give|hand)", t):
        hints.append(f'"{ms[0]["phrase"]}" cannot be picked: too wide for the gripper or a fixed place (rule 3).')
    cm = re.search(r"\b(leave it where it is|leave it there|without moving|without touching|without lifting|"
                   r"(and|but) (do not|don't|dont) move|while it stays|keep it where it is)\b", t)
    if cm:
        hints.append(f'"{cm.group(0)}" contradicts the move that the instruction asks for: self-contradictory (rule 3).')
    im = re.search(r"\b(in|inside|into) (the )?", t)
    m_in = next((m for m in ms if im and m["start"] == im.end() and m["bases"] and all(b not in CONTAINERS and b != "mug" for b in m["bases"])), None)
    if m_in and im.start() > 0 and any(m["end"] <= im.start() for m in ms):
        hints.append(f'"{m_in["phrase"]}" is a solid object, nothing fits inside it: impossible (rule 3).')
    pick_ms = [m for m in ms if any(b in PICKABLE for b in m["bases"])]
    if not pick_ms:
        if not ms and re.search(r"\b(that|this|it|those|these|them|one)\b", t) and advice:
            hints.append('The instruction names no object, only "that/this/it": no clue which object is meant: ask (rule 4).')
        return hints
    head = pick_ms[0]
    cands = [o for o in head["cands"] if o["base"] in PICKABLE]
    parts = ("grey_part", "white_part", "small_part")
    if not cands and all(b in parts for b in head["bases"]):
        # the camera labels the three parts unreliably: fall back to the parts it did see
        cands = [o for o in objs if o["base"] in parts]
        if cands and head["bases"] == ["small_part"]:
            cands = [min(cands, key=lambda o: o["size"][0] * o["size"][1])]
        if cands:
            hints = [h for h in hints if head["phrase"] not in h]
            hints.append(f'"{head["phrase"]}" is not labelled as such, but the camera labels the parts unreliably; '
                         f'the part(s) it sees: {", ".join(o["name"] for o in cands)}.')
    if not cands:
        return hints
    # "put X under Y": Y ends up lying on X, so Y is moved first (it usually lies on X already)
    pm = re.search(r"\b(under|underneath|beneath) (the )?", t)
    m_y = next((m for m in ms if pm and m["start"] == pm.end() and m["cands"] and m is not head and m["cands"][0]["base"] in PICKABLE), None)
    if m_y and re.match(r"\s*(put|place|move|set|lay)\b", t):
        y = m_y["cands"][0]["name"] if len(m_y["cands"]) == 1 else None
        if y:
            hints.append(f"first pick {y} and put it aside (it lies on or must end up on {head['phrase']}), then move the "
                         f"{head['phrase']}, then place {y} on it.")
    multi = bool(MULTI_RE.search(t[: head["end"] + 1])) or head["phrase"] in ("everything", "all of them", "the rest")
    others = [m for m in ms if m is not head]

    def ref(pattern: str) -> dict | None:
        """The mention that follows `pattern` (e.g. 'closest to the <X>'), or None."""
        mm = re.search(pattern, t)
        if not mm:
            return None
        return next((m for m in others if m["start"] >= mm.end() - 1 and m["start"] - mm.end() <= 6), None)

    cue = None
    # inside a container: "in the grey tray", "from the black tray", "on the plate", "that is on the plate"
    for mm in re.finditer(r"\b(in|inside|from|on|that is on|that are on|out of|(top|bottom|left|right|middle|centre|center|side) of) (the )?", t):
        m2 = next((m for m in others if m["start"] == mm.end() and any(b in CONTAINERS for b in m["bases"])), None)
        if m2 and m2["cands"] and mm.start() > head["start"] - 1:
            inside = [o for o in cands if o["in_tray"] in {c["name"] for c in m2["cands"]} or o["on_top_of"] in {c["name"] for c in m2["cands"]}]
            if inside:
                cands, cue = inside, f"in/on {m2['phrase']}"
            break
    # "the red gear under the cube": the one the cube lies on; the cube must move first
    um2 = re.search(r"\b(under|underneath|beneath|below) (the )?", t)
    m2 = next((m for m in others if um2 and m["start"] == um2.end() and m["cands"]), None)
    if m2 and not any(b in CONTAINERS for b in m2["bases"]) and um2.start() > head["start"]:
        under = [o for o in cands if any(c["on_top_of"] == o["name"] for c in m2["cands"])]
        if under:
            top = next(c["name"] for c in m2["cands"] if c["on_top_of"] == under[0]["name"])
            cands, cue = under, f"under {m2['phrase']}; {top} lies on it and must be moved first"
    if re.search(r"\bwith nothing on (it|top)\b|\bnothing lying on\b", t):
        free = [o for o in cands if not o["supports"]]
        if free:
            cands, cue = free, "nothing on it"
    m2 = ref(r"\b(lying|sitting|placed|that is|which is)? ?on( top of)? (the )?(other )?")
    if m2 and m2 is not head and not any(b in CONTAINERS for b in m2["bases"]) and m2["cands"]:
        on = [o for o in cands if o["on_top_of"] in {c["name"] for c in m2["cands"]}]
        if on:
            cands, cue = on, f"lying on {m2['phrase']}"
        else:
            # the camera reported nothing lying on it: the object whose box overlaps it most, else the nearest one
            def overlap(o, c):
                iw = min(o["bbox"][2], c["bbox"][2]) - max(o["bbox"][0], c["bbox"][0])
                ih = min(o["bbox"][3], c["bbox"][3]) - max(o["bbox"][1], c["bbox"][1])
                return iw * ih if iw > 0 and ih > 0 else 0
            pool = [o for o in cands if o not in m2["cands"]]
            if pool:
                best = max(pool, key=lambda o: (max(overlap(o, c) for c in m2["cands"]), -min(_dist(o, c) for c in m2["cands"])))
                cands, cue = [best], f"the camera reported nothing lying on {m2['phrase']}; the object overlapping it most"
    elif re.search(r"\b(lying|sitting) on the other\b|\bon top of the other\b|\bon the other\b", t):
        on = [o for o in cands if o["on_top_of"]]
        if on:
            cands, cue = on, "lying on another object"
    m2 = ref(r"\b(closest|nearest|next|near|adjacent|beside|close)( to)? (the )?")
    if m2 and m2["cands"]:
        pool = [o for o in cands if o not in m2["cands"]] or cands
        best = min(pool, key=lambda o: min(_dist(o, c) for c in m2["cands"]))
        cands, cue = [best], f"closest to {m2['phrase']}"
    m2 = ref(r"\b(farthest|furthest|far(thest)? away) (from )?(the )?")
    if m2 and m2["cands"]:
        pool = [o for o in cands if o not in m2["cands"]] or cands
        best = max(pool, key=lambda o: min(_dist(o, c) for c in m2["cands"]))
        cands, cue = [best], f"farthest from {m2['phrase']}"
    dm = re.search(r"\b(to the |just |right |directly )?(left|right|above|below|under|beneath|underneath|on top) (of )?(the )?", t)
    m2 = next((m for m in others if m["start"] == dm.end()), None) if dm else None
    if m2 and m2["cands"] and dm.group(2) in ("left", "right", "above", "below") and (len(cands) > 1 or head["phrase"] in GENERIC):
        d = dm.group(2)
        anchor = m2["cands"][0]  # "the mug" is usually unique

        def side(o):
            du, dv = o["centre"][0] - anchor["centre"][0], o["centre"][1] - anchor["centre"][1]
            return {"left": du < 0 and abs(du) >= abs(dv) * 0.5, "right": du > 0 and abs(du) >= abs(dv) * 0.5,
                    "above": dv < 0 and abs(dv) >= abs(du) * 0.5, "below": dv > 0 and abs(dv) >= abs(du) * 0.5}[d]
        pool = [o for o in cands if o is not anchor and side(o)]
        if pool:
            best = min(pool, key=lambda o: _dist(o, anchor))
            cands, cue = [best], f"{d} of {m2['phrase']}"
    sm = re.search(r"\b(big|bigger|biggest|large|larger|largest|small|smaller|smallest|little|tiny)\b", t)
    if sm and len(cands) > 1 and not (head["start"] <= sm.start() < head["end"]):
        area = lambda o: o["size"][0] * o["size"][1]
        best = max(cands, key=area) if sm.group(1)[0] in "bl" else min(cands, key=area)
        cands, cue = [best], sm.group(1)
    om = re.search(r"\b(second|third|2nd|3rd)\b.*\bfrom the (left|right|top|bottom)\b", t)
    if om and len(cands) > 1:
        idx = {"second": 1, "2nd": 1, "third": 2, "3rd": 2}[om.group(1)]
        key = (lambda o: o["centre"][0]) if om.group(2) in ("left", "right") else (lambda o: o["centre"][1])
        order = sorted(cands, key=key, reverse=om.group(2) in ("right", "bottom"))
        if idx < len(order):
            cands, cue = [order[idx]], f"{om.group(1)} from the {om.group(2)}"
    elif len(cands) > 1 and not cue:
        em = re.search(r"\b(left|right|top|bottom|upper|lower|leftmost|rightmost|topmost|highest|lowest)(most)?\b", t)
        rel = re.match(r"\w+ (of|side of) the (\w+)", t[em.start():]) if em else None
        if em and not (rel and rel.group(2) not in IMAGE_WORDS and not any(m["start"] == em.start() + rel.start(2) and any(
                b in CONTAINERS for b in m["bases"]) for m in others)):
            d = em.group(1).replace("most", "")
            d = {"upper": "top", "lower": "bottom", "highest": "top", "lowest": "bottom"}.get(d, d)
            key = (lambda o: o["centre"][0]) if d in ("left", "right") else (lambda o: o["centre"][1])
            best = max(cands, key=key) if d in ("right", "bottom") else min(cands, key=key)
            cands, cue = [best], f"{d}"
    # except X
    xm = re.search(r"\bexcept( for)? (the )?", t)
    if xm and multi:
        excl = [m for m in others if m["start"] >= xm.end() - 1]
        ex_names = {c["name"] for m in excl for c in m["cands"]}
        cands = [o for o in cands if o["name"] not in ex_names]
    names = [o["name"] for o in cands]
    phrase = head["phrase"]
    if multi:
        hints.append(f'"{phrase}" (all of them{", " + cue if cue else ""}) = {", ".join(names) or "nothing"}'
                     + (": one move per object." if advice else "."))
    elif len(names) == 1:
        hints.append(f'"{phrase}"{" (" + cue + ")" if cue else ""} = {names[0]}' + (": exactly one object, so plan, do not ask." if advice else "."))
    elif cue:
        hints.append(f'"{phrase}" ({cue}) = {", ".join(names)}.')
    else:
        hints.append(f'"{phrase}" could be {", ".join(names)}' + (" and the instruction gives no clue which: ask (rule 4)." if advice else "."))
    for m in others:
        if m["cands"] and m is not head and not any(m["phrase"] in h for h in hints):
            hints.append(f'"{m["phrase"]}" = {", ".join(o["name"] for o in m["cands"])}.')
    for o in cands:
        if o["supports"]:
            hints.append(f'{", ".join(o["supports"])} lies on top of {o["name"]}: pick {", ".join(o["supports"])} first and put it aside.')
    return hints


def check(ans: dict, hints: list[str], objs: list[dict]) -> list[str]:
    """Self-check: where the answer contradicts a fact in the hints, the issues to put to Apertus once more.
    Empty list = consistent."""
    kind = answer_type(ans)[0]
    joined = " ".join(hints)
    issues = []
    unsafe = "unsafe action" in joined
    missing = "no such object" in joined
    cannot = "(rule 3)" in joined
    resolved = any("exactly one object" in h for h in hints) or any("all of them" in h for h in hints)
    if kind == "ask_user" and resolved and "no clue" not in joined and not (unsafe or missing or cannot):
        issues.append("The hints resolve the instruction to exactly the listed object(s); there is nothing to ask. "
                      "Apply rule 5 and give the plan, or refuse only with a concrete rule 1 or rule 3 reason.")
    if kind == "refuse" and not (unsafe or missing or cannot):
        named = sorted({o["name"] for o in objs if any(o["name"] in h for h in hints)})
        issues.append(f"Every object the instruction names is in the list ({', '.join(named) or 'see hints'}) and fits the gripper, "
                      "so rule 2 does not apply. Refuse only for a concrete rule 1 or rule 3 reason, otherwise plan.")
    if kind == "plan" and unsafe:
        issues.append("The instruction contains an unsafe action (rule 1): refuse.")
    if kind == "plan" and missing:
        issues.append("The instruction names an object the camera does not list (rule 2): refuse.")
    if kind == "plan" and cannot:
        issues.append("The hints flag a rule 3 problem (cannot / self-contradictory): refuse.")
    if kind == "plan" and "no clue" in joined and not resolved:
        issues.append("The hints say the instruction gives no clue which object is meant (rule 4): ask one short question "
                      "that names the choices instead of guessing.")
    if kind == "plan":
        by_name = {o["name"]: o for o in objs}
        picked = [a["object"] for a in ans["actions"] if a["action"] == "pick"]
        for i, p in enumerate(picked):
            tops = [t for t in (by_name.get(p, {}).get("supports") or []) if t not in picked[:i]]
            if tops:
                issues.append(f"{', '.join(tops)} lies on top of {p}: pick {tops[0]} and put it aside before picking {p}.")
                break
        fm = next((h for h in hints if h.startswith("first pick ")), None)
        if fm and picked and picked[0] != fm.split()[2]:
            issues.append(f"The instruction puts something under {fm.split()[2]}: {fm}")
    return issues


def hints_text(hints: list[str]) -> str:
    if not hints:
        return "Reference hints: none."
    return "Reference hints (computed from the coordinates above; they are reliable):\n" + "\n".join("- " + h for h in hints)


# ----------------------------------------------------------------------------- answer
def to_answer(reply: str, objs: list[dict], iid: str) -> dict:
    """Apertus reply (steps: pick/place, ask_user, refuse) -> one schema-valid answer.
    target_pixel = a pixel on the first picked object. Anything unparseable becomes ask_user:
    a schema miss scores 0 everywhere, a question at least cannot."""
    by_name = {o["name"]: o for o in objs}
    ans = parse(reply) if isinstance(reply, str) else None
    steps = ans.get("actions") if isinstance(ans, dict) else None
    if isinstance(steps, list) and steps and all(isinstance(s, dict) for s in steps):
        s0 = steps[0]
        if "ask_user" in s0 and isinstance(s0["ask_user"], str) and s0["ask_user"].strip():
            return {"instruction_id": iid, "target_pixel": None, "actions": [{"action": "ask_user", "question": s0["ask_user"].strip()}]}
        if "refuse" in s0 and isinstance(s0["refuse"], str) and s0["refuse"].strip():
            return {"instruction_id": iid, "target_pixel": None, "actions": [{"action": "refuse", "reason": s0["refuse"].strip()}]}
        actions = []
        for s in steps:
            p = s.get("pick")
            if p not in by_name or by_name[p]["base"] in UNPICKABLE:
                continue
            actions.append({"action": "pick", "object": p, "pixel": by_name[p].get("pixel", by_name[p]["centre"])})
            d = s.get("place")
            if isinstance(d, str) and d not in ("none", "", p):
                a = {"action": "place", "object": d}
                if d in by_name:
                    a["pixel"] = by_name[d].get("pixel", by_name[d]["centre"])
                actions.append(a)
        if actions:
            out = {"instruction_id": iid, "target_pixel": list(actions[0]["pixel"]), "actions": actions}
            if answer_type(out)[0] == "plan":
                return out
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
    ap.add_argument("--no-hints", action="store_true", help="do not give Apertus the resolved references (ablation)")
    ap.add_argument("--think", action="store_true", help="let Apertus think before answering (slower)")
    ap.add_argument("--no-check", action="store_true", help="skip the self-check round (ablation)")
    args = ap.parse_args()

    scenes = sorted(p for p in args.data.iterdir() if p.is_dir() and (p / "rgb.png").exists())
    jobs = [(d, iid, text) for d in scenes for iid, _, text in load_scene(d).instructions][: args.limit or None]
    cache_path = args.out.with_suffix(".scenes.json")
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    t0 = time.time()
    try:
        # stage 1: see every scene first, so Ollama swaps model once, not per instruction
        for d in dict.fromkeys(d for d, _, _ in jobs):
            if d.name in cache and cache[d.name].get("version") == CACHE_VERSION:
                continue
            sc = load_scene(d)
            w, h = sc.size
            b64 = None
            if d.name in cache:
                reply = cache[d.name]["raw"]  # older cache: reuse the first vision reply
            else:
                b64, *_ = encode(sc.rgb, args.max_side)
                reply = ollama_chat(args.base_url, args.vision_model, SEE_PROMPT, image_b64=b64,
                                    schema=None if args.no_schema else SEE_SCHEMA, think=False)
            if d.name in cache and "layout" in cache[d.name]:
                layout = cache[d.name]["layout"]
            else:
                b64 = b64 or encode(sc.rgb, args.max_side)[0]
                layout = ollama_chat(args.base_url, args.vision_model, LAYOUT_PROMPT, image_b64=b64,
                                     schema=None if args.no_schema else LAYOUT_SCHEMA, think=False)
            objs = scene_table(parse(reply) or {}, w, h, parse(layout) or {})
            for o in objs:
                o["pixel"] = refine_pixel(sc.rgb, o["bbox"])
            cache[d.name] = {"version": CACHE_VERSION, "raw": reply, "layout": layout, "objects": objs, "size": [w, h]}
            cache_path.write_text(json.dumps(cache, indent=1))
            print(f"[see] {d.name}: {', '.join(o['name'] + (' on ' + o['on_top_of'] if o['on_top_of'] else '') for o in objs)}")
        # stage 2: Apertus plans from text only
        results, bad = [], 0
        with args.out.with_suffix(".raw.jsonl").open("w") as raw:
            for n, (d, iid, text) in enumerate(jobs, 1):
                objs, (w, h) = cache[d.name]["objects"], cache[d.name]["size"]
                for o in objs:
                    o.setdefault("supports", [x["name"] for x in objs if x.get("on_top_of") == o["name"]])
                hints = [] if args.no_hints else resolve(text, objs)
                prompt = PLAN_PROMPT.format(w=w, h=h, table=table_text(objs), hints=hints_text(hints), text=text)
                schema = None if args.no_schema else plan_schema(objs)
                reply = ollama_chat(args.base_url, args.model, prompt, schema=schema, think=True if args.think else None)
                ans = to_answer(reply, objs, iid)
                issues = check(ans, hints, objs) if hints and not args.no_check else []
                reply2 = None
                if issues:  # one self-check round: Apertus sees its answer and the contradiction, answers again
                    msgs = [{"role": "user", "content": prompt}, {"role": "assistant", "content": reply},
                            {"role": "user", "content": CHECK_PROMPT.format(issues="\n".join("- " + i for i in issues))}]
                    reply2 = ollama_chat(args.base_url, args.model, msgs, schema=schema, think=True if args.think else None)
                    ans = to_answer(reply2, objs, iid)
                bad += ans["actions"][0].get("question") == FALLBACK_QUESTION
                results.append(ans)
                raw.write(json.dumps({"instruction_id": iid, "instruction": text, "scene": table_text(objs), "hints": hints,
                                      "reply": reply, "check": issues, "reply2": reply2}) + "\n")
                print(f"[{n}/{len(jobs)}] {iid}: {answer_type(ans)[0]}{' (after check)' if issues else ''}  {text!r}")
    except urllib.error.URLError as e:
        sys.exit(f"cannot reach Ollama at {args.base_url} ({e}). Start it with `ollama serve` and pull both models.")
    args.out.write_text(json.dumps({"team": args.team, "results": results}, indent=2))
    print(f"\n{len(results)} answers in {args.out} ({bad} fell back to ask_user) in {time.time() - t0:.0f}s")
    print(f"score it:  python starter/evaluate.py {args.out} --data {args.data}")


if __name__ == "__main__":
    main()
