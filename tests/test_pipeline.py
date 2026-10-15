"""Checks without a model: vision JSON -> full-res table (numbering, tray, stacking), reference resolution,
and Apertus reply -> schema-valid answer."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "starter"))
from evaluate import answer_type  # noqa: E402

from see_it_say_it_pick_it.pipeline import FALLBACK_QUESTION, aside_pixel, check, humanize, resolve, scene_table, to_answer  # noqa: E402

SEEN = {"objects": [
    {"name": "red_gear", "bbox": [500, 500, 600, 600], "on_top_of": None},
    {"name": "red_gear", "bbox": [100, 500, 200, 600], "on_top_of": None},
    {"name": "grey_tray", "bbox": [50, 400, 300, 700], "on_top_of": None},
    {"name": "yellow_gear", "bbox": [520, 520, 580, 580], "on_top_of": None},
    {"name": "mug", "bbox": [700, 100, 800, 200], "on_top_of": None},
]}
LAYOUT = {"stacked": [], "trays": [{"name": "black_tray", "bbox": [400, 50, 900, 300]}]}


def check_table():
    objs = scene_table(SEEN, 2448, 2048, LAYOUT)
    names = [o["name"] for o in objs]
    # reading order: top row (black_tray, mug) first, then the lower row left to right
    assert names == ["black_tray", "mug", "red_gear_1", "grey_tray", "red_gear_2", "yellow_gear"]
    by = {o["name"]: o for o in objs}
    assert by["red_gear_1"]["centre"] == [367, 1126] and by["red_gear_1"]["in_tray"] == "grey_tray"
    assert by["yellow_gear"]["on_top_of"] == "red_gear_2"          # box-overlap fallback for stacking
    assert by["red_gear_2"]["supports"] == ["yellow_gear"]
    assert by["mug"]["in_tray"] == "black_tray"                     # tray added by the layout pass
    # descriptions a person can read: duplicates by their order, unique objects by their region of the photo
    assert by["red_gear_1"]["desc"] == "the left red gear, in the grey tray"
    assert by["red_gear_2"]["desc"] == "the right red gear, with the yellow gear on it"
    assert by["yellow_gear"]["desc"] == "the yellow gear in the centre, lying on the red gear"
    assert by["mug"]["desc"] == "the mug at the top right, in the black tray"
    q = humanize("Which red gear: red_gear_1 or red_gear_2?", objs)
    assert q == ("Which red gear: the left red gear, in the grey tray (red_gear_1) or "
                 "the right red gear, with the yellow gear on it (red_gear_2)?")
    three = scene_table({"objects": [{"name": "red_gear", "bbox": [400 + 100 * i, 450, 460 + 100 * i, 550], "on_top_of": None}
                                     for i in range(3)]}, 1000, 1000)
    assert [o["desc"] for o in three] == ["the leftmost red gear", "the middle red gear", "the rightmost red gear"]
    four = scene_table({"objects": [{"name": "red_gear", "bbox": [400, 100 + 200 * i, 460, 160 + 200 * i], "on_top_of": None}
                                    for i in range(4)]}, 1000, 1000)
    assert [o["desc"] for o in four] == ["the top red gear", "the 2nd from the top red gear", "the 3rd from the top red gear",
                                         "the bottom red gear"]
    # machine parts relabelled by size: two "small" plates of 512 and 506 px are grey parts, a 650 px plate is the white
    # part, and two touching fragments of 100x223 and 242x129 px are one small white part
    parts = scene_table({"objects": [
        {"name": "small_part", "bbox": [582, 149, 792, 271], "on_top_of": None},      # 514x250 px -> grey
        {"name": "small_part", "bbox": [820, 176, 915, 423], "on_top_of": None},      # 233x506 px -> grey
        {"name": "grey_part", "bbox": [556, 407, 823, 701], "on_top_of": None},       # 654x602 px -> white
        {"name": "white_part", "bbox": [296, 241, 337, 350], "on_top_of": None},      # fragment 100x223
        {"name": "small_part", "bbox": [335, 240, 434, 303], "on_top_of": None}]},    # fragment 242x129, touching
        2448, 2048)
    assert sorted(o["name"] for o in parts) == ["grey_part_1", "grey_part_2", "small_part", "white_part"]
    small = next(o for o in parts if o["name"] == "small_part")
    assert small["bbox"] == [725, 492, 1062, 717]
    # a part the first pass missed comes from the layout pass; one that overlaps a known part is not added twice
    lay = {"stacked": [], "trays": [], "parts": [{"name": "white_part", "bbox": [280, 230, 420, 350]},
                                                 {"name": "grey_part", "bbox": [585, 150, 790, 270]}]}
    got = scene_table({"objects": [{"name": "small_part", "bbox": [582, 149, 792, 271], "on_top_of": None}]}, 2448, 2048, lay)
    assert sorted(o["name"] for o in got) == ["grey_part", "small_part"]     # the layout "white_part" is 343 px long: small
    lay = {"stacked": [], "trays": [], "parts": [{"name": "white_part", "bbox": [567, 456, 881, 828]}]}   # = the plate's box
    got = scene_table({"objects": [{"name": "plate", "bbox": [566, 456, 881, 829], "on_top_of": None}]}, 2448, 2048, lay)
    assert [o["name"] for o in got] == ["plate"]
    # "the" written by Apertus is not doubled; in a reason only duplicated types get the description
    assert humanize("put the mug on the plate", objs) == \
        "put the mug at the top right, in the black tray (mug) on the plate"   # no plate seen: the word stays
    assert humanize("the red_cube lies on red_gear_1; put the mug aside", objs, full=False) == \
        "the red_cube lies on the left red gear, in the grey tray (red_gear_1); put the mug aside"
    return objs


def check_resolve(objs):
    one = lambda text: [h for h in resolve(text, objs) if "exactly one" in h]
    assert "red_gear_1" in one("pick up the red gear on the left")[0]
    assert "red_gear_2" in one("pick up the second red gear from the left")[0]
    assert "red_gear_1" in one("pick up the red gear with nothing on it")[0]
    assert "yellow_gear" in one("pick up the gear lying on the other gear")[0]
    assert "red_gear_2" in one("pick up the red gear closest to the mug")[0]
    assert "red_gear_1" in one("pick up the red gear farthest from the mug")[0]
    assert "red_gear_2" in one("pick up the red gear under the yellow gear")[0]
    big = scene_table({"objects": [{"name": "red_gear", "bbox": [100, 100, 200, 200], "on_top_of": None},
                                   {"name": "red_gear", "bbox": [500, 100, 700, 300], "on_top_of": None}]}, 1000, 1000)
    assert big[1]["note"] == "the largest one"
    assert "red_gear_2" in [h for h in resolve("pick up the big red gear", big) if "exactly one" in h][0]
    # a size word applies only to the phrase it stands in front of
    both = [h for h in resolve("put both red gears and the little yellow gear in the black tray", objs) if "all of them" in h][0]
    assert "red_gear_1" in both and "red_gear_2" in both
    assert any("no clue" in h for h in resolve("pick up the red gear", objs))
    assert any("no such object" in h for h in resolve("pick up the blue cube", objs))
    assert any("unsafe" in h for h in resolve("throw the mug into the tray", objs))
    assert not any("so plan" in h for h in resolve("throw the mug into the tray", objs))
    assert any("cannot be picked" in h for h in resolve("pick up the grey tray", objs))
    allh = [h for h in resolve("put everything from the grey tray in the black tray", objs) if "all of them" in h][0]
    assert "red_gear_1" in allh and "mug" not in allh


def check_answer(objs):
    assert to_answer("garbage", objs, "x")["actions"][0]["question"] == FALLBACK_QUESTION
    ans = to_answer('{"actions": [{"pick": "red_gear_2", "place": "grey_tray"}, {"pick": "mug", "place": "none"}]}', objs, "x")
    by = {o["name"]: o for o in objs}
    assert ans["target_pixel"] == by["red_gear_2"]["centre"] and answer_type(ans) == ("plan", None)
    assert [a["action"] for a in ans["actions"]] == ["pick", "place", "pick"]   # a final bare pick stays a pick
    # a set-aside before another pick is put down on the table: no two picks in a row for a one-object gripper
    ans = to_answer('{"actions": [{"pick": "yellow_gear", "place": "none"}, {"pick": "red_gear_2", "place": "grey_tray"}]}', objs, "x")
    assert [(a["action"], a["object"]) for a in ans["actions"]] == \
        [("pick", "yellow_gear"), ("place", "table"), ("pick", "red_gear_2"), ("place", "grey_tray")]
    # the set-aside spot is bare table: inside the photo and inside no box
    aside = aside_pixel(objs, 2448, 2048)
    assert 0 <= aside[0] < 2448 and 0 <= aside[1] < 2048
    assert not any(o["bbox"][0] <= aside[0] <= o["bbox"][2] and o["bbox"][1] <= aside[1] <= o["bbox"][3] for o in objs)
    ans = to_answer('{"actions": [{"pick": "yellow_gear", "place": "none"}, {"pick": "red_gear_2", "place": "grey_tray"}]}', objs, "x", aside)
    assert ans["actions"][1] == {"action": "place", "object": "table", "pixel": aside}
    # two targets each carrying something: one message names both tops (one at a time made Apertus alternate)
    stacks = scene_table({"objects": [{"name": "red_gear", "bbox": [100, 450, 160, 510], "on_top_of": None},
                                      {"name": "red_cube", "bbox": [110, 460, 150, 500], "on_top_of": "red_gear"},
                                      {"name": "red_gear", "bbox": [300, 450, 360, 510], "on_top_of": None},
                                      {"name": "yellow_gear", "bbox": [310, 460, 350, 500], "on_top_of": "red_gear"},
                                      {"name": "grey_tray", "bbox": [50, 50, 500, 300], "on_top_of": None}]}, 1000, 1000)
    hints = resolve("put both red gears in the grey tray", stacks)
    issues = check(to_answer('{"actions": [{"pick": "red_gear_1", "place": "grey_tray"}, {"pick": "red_gear_2", "place": "grey_tray"}]}',
                             stacks, "x"), hints, stacks)
    assert len(issues) == 1 and "pick red_cube, place none; pick yellow_gear, place none" in issues[0]
    assert not check(to_answer('{"actions": [{"pick": "red_cube", "place": "none"}, {"pick": "yellow_gear", "place": "none"}, '
                               '{"pick": "red_gear_1", "place": "grey_tray"}, {"pick": "red_gear_2", "place": "grey_tray"}]}',
                               stacks, "x"), hints, stacks)
    ans = to_answer('{"actions": [{"refuse": "no blue cube"}]}', objs, "x")
    assert ans["target_pixel"] is None and answer_type(ans) == ("refuse", None)
    ans = to_answer('{"actions": [{"pick": "grey_tray", "place": "plate"}]}', objs, "x")  # unpickable -> fallback
    assert ans["actions"][0]["action"] == "ask_user"
    # self-check: a question although the hints resolved one object; a plan although the verb is unsafe
    hints = resolve("pick up the red gear on the left", objs)
    assert check(to_answer('{"actions": [{"ask_user": "which?"}]}', objs, "x"), hints, objs)
    assert not check(to_answer('{"actions": [{"pick": "red_gear_1", "place": "none"}]}', objs, "x"), hints, objs)
    hints = resolve("throw the mug", objs)
    assert check(to_answer('{"actions": [{"pick": "mug", "place": "none"}]}', objs, "x"), hints, objs)
    hints = resolve("pick up the red gear under the yellow gear", objs)
    assert check(to_answer('{"actions": [{"pick": "red_gear_2", "place": "none"}]}', objs, "x"), hints, objs)  # yellow first
    assert any("contradictory" in h for h in resolve("put the mug on the plate and leave it where it is", objs))
    hints = resolve("pick up the red gear", objs)   # no clue -> a plan must be challenged
    assert check(to_answer('{"actions": [{"pick": "red_gear_1", "place": "none"}]}', objs, "x"), hints, objs)
    assert any("no clue" in h for h in resolve("give me that one", objs))
    assert any("no clue" in h for h in resolve("move it away from the mug", objs))   # pronoun target, a noun as reference
    assert not any("exactly one" in h for h in resolve("take it off the mug", objs))
    hints = resolve("give me that one", objs)   # a refusal here must be steered to a question, not a plan
    issues = check(to_answer('{"actions": [{"refuse": "not there"}]}', objs, "x"), hints, objs)
    assert issues and "ask" in issues[0]
    hints = resolve("pick up the red gear closest to the mug", objs)   # resolved to red_gear_2, plan picks the mug
    assert check(to_answer('{"actions": [{"pick": "mug", "place": "none"}]}', objs, "x"), hints, objs)
    # "except X": X is excluded from the set and a plan that moves it anyway is challenged
    flat = scene_table({"objects": [{"name": "red_gear", "bbox": [100, 450, 160, 500], "on_top_of": None},
                                    {"name": "red_gear", "bbox": [200, 450, 260, 500], "on_top_of": None},
                                    {"name": "yellow_gear", "bbox": [600, 450, 640, 490], "on_top_of": None},
                                    {"name": "black_tray", "bbox": [50, 50, 500, 300], "on_top_of": None}]}, 1000, 1000)
    hints = resolve("put all the gears in the black tray except the yellow one", flat)
    assert any(h.startswith('"except": yellow_gear') for h in hints)
    assert "yellow_gear" not in [h for h in hints if "all of them" in h][0].split(" = ")[1]
    issues = check(to_answer('{"actions": [{"pick": "red_gear_1", "place": "black_tray"}, {"pick": "red_gear_2", "place": "black_tray"}, '
                             '{"pick": "yellow_gear", "place": "black_tray"}]}', flat, "x"), hints, flat)
    assert issues and "except yellow_gear" in issues[0]
    assert not check(to_answer('{"actions": [{"pick": "red_gear_1", "place": "black_tray"}, {"pick": "red_gear_2", "place": "black_tray"}]}',
                               flat, "x"), hints, flat)
    # ... unless it lies on an object that must be moved and has to be set aside first (yellow_gear on red_gear_2 in `objs`)
    hints = resolve("put all the gears in the black tray except the yellow one", objs)
    assert not check(to_answer('{"actions": [{"pick": "red_gear_1", "place": "black_tray"}, {"pick": "yellow_gear", "place": "none"}, '
                               '{"pick": "red_gear_2", "place": "black_tray"}]}', objs, "x"), hints, objs)
    # with a put-verb "on the plate" / "in the black tray" is the destination, not a filter; "from the grey tray" filters
    hints = resolve("put both red gears in the black tray", objs)
    assert any("all of them) = red_gear_1, red_gear_2" in h for h in hints) and any(h.startswith('Destination "black tray" = black_tray') for h in hints)
    issues = check(to_answer('{"actions": [{"pick": "red_gear_1", "place": "black_tray"}, {"pick": "red_gear_2", "place": "none"}]}', objs, "x"), hints, objs)
    assert any("destination is black_tray" in i and "red_gear_2" in i for i in issues)
    assert not check(to_answer('{"actions": [{"pick": "red_gear_1", "place": "black_tray"}, {"pick": "yellow_gear", "place": "none"}, '
                               '{"pick": "red_gear_2", "place": "black_tray"}]}', objs, "x"), hints, objs)
    assert "red_gear_1" in [h for h in resolve("put the object from the grey tray in the black tray", objs) if "exactly one" in h][0]
    assert not any(h.startswith("Destination") for h in resolve("put the red gear on the left on the plate and the mug in the black tray", objs))
    assert not any(h.startswith("Destination") for h in resolve("pick up the red gear closest to the grey tray", objs))
    # "at the top of the <container>": the one nearest that edge among those inside it
    two = scene_table({"objects": [{"name": "red_gear", "bbox": [100, 450, 160, 500], "on_top_of": None},
                                   {"name": "red_gear", "bbox": [100, 620, 160, 680], "on_top_of": None},
                                   {"name": "grey_tray", "bbox": [50, 400, 300, 700], "on_top_of": None}]}, 1000, 1000)
    assert "red_gear_1" in [h for h in resolve("pick up the upper red gear in the grey tray", two) if "exactly one" in h][0]
    assert "red_gear_1" in [h for h in resolve("pick up the red gear at the top of the tray", two) if "exactly one" in h][0]
    assert "red_gear_2" in [h for h in resolve("pick up the red gear at the bottom of the tray", two) if "exactly one" in h][0]
    hints = resolve("put everything from the grey tray in the black tray", objs)  # = red_gear_1 only; mug is extra
    assert check(to_answer('{"actions": [{"pick": "red_gear_1", "place": "black_tray"}, {"pick": "mug", "place": "black_tray"}]}', objs, "x"), hints, objs)
    hints = resolve("put both red gears and the mug in the black tray", objs)     # the mug is named too: not extra
    assert not check(to_answer('{"actions": [{"pick": "yellow_gear", "place": "none"}, {"pick": "red_gear_2", "place": "black_tray"}, '
                               '{"pick": "red_gear_1", "place": "black_tray"}, {"pick": "mug", "place": "black_tray"}]}', objs, "x"), hints, objs)
    hints = resolve("pick up the red gear on the left without moving the mug", objs)   # rule 3, but the reason says "not there"
    issues = check(to_answer('{"actions": [{"refuse": "NOT THERE: red_gear_1 is in the list, so it is there."}]}', objs, "x"), hints, objs)
    assert issues and "reason says" in issues[0]
    hints = resolve("put the red gear under the mug in the black tray", objs)
    assert any(h.startswith("first pick mug") for h in hints)
    assert check(to_answer('{"actions": [{"pick": "red_gear_1", "place": "black_tray"}]}', objs, "x"), hints, objs)
    hints = resolve("put the red gear on the left under the mug in the black tray", objs)
    full = '{"actions": [{"pick": "mug", "place": "none"}, {"pick": "red_gear_1", "place": "black_tray"}, {"pick": "mug", "place": "red_gear_1"}]}'
    assert not check(to_answer(full, objs, "x"), hints, objs)
    short = '{"actions": [{"pick": "mug", "place": "none"}, {"pick": "red_gear_1", "place": "none"}, {"pick": "mug", "place": "black_tray"}]}'
    assert any("three moves" in i for i in check(to_answer(short, objs, "x"), hints, objs))


def test_pipeline():
    """The one test pytest collects (`uv run pytest`); `python tests/test_pipeline.py` runs it too."""
    objs = check_table()
    check_resolve(objs)
    check_answer(objs)


if __name__ == "__main__":
    test_pipeline()
    print("ok")
