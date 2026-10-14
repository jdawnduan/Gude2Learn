"""Checks without a model: vision JSON -> full-res table (numbering, tray, stacking), reference resolution,
and Apertus reply -> schema-valid answer."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "starter"))
from evaluate import answer_type  # noqa: E402

from see_it_say_it_pick_it.pipeline import FALLBACK_QUESTION, check, resolve, scene_table, to_answer  # noqa: E402

SEEN = {"objects": [
    {"name": "red_gear", "bbox": [500, 500, 600, 600], "on_top_of": None},
    {"name": "red_gear", "bbox": [100, 500, 200, 600], "on_top_of": None},
    {"name": "grey_tray", "bbox": [50, 400, 300, 700], "on_top_of": None},
    {"name": "yellow_gear", "bbox": [520, 520, 580, 580], "on_top_of": None},
    {"name": "mug", "bbox": [700, 100, 800, 200], "on_top_of": None},
]}
LAYOUT = {"stacked": [], "trays": [{"name": "black_tray", "bbox": [400, 50, 900, 300]}]}


def test_table():
    objs = scene_table(SEEN, 2448, 2048, LAYOUT)
    names = [o["name"] for o in objs]
    # reading order: top row (black_tray, mug) first, then the lower row left to right
    assert names == ["black_tray", "mug", "red_gear_1", "grey_tray", "red_gear_2", "yellow_gear"]
    by = {o["name"]: o for o in objs}
    assert by["red_gear_1"]["centre"] == [367, 1126] and by["red_gear_1"]["in_tray"] == "grey_tray"
    assert by["yellow_gear"]["on_top_of"] == "red_gear_2"          # box-overlap fallback for stacking
    assert by["red_gear_2"]["supports"] == ["yellow_gear"]
    assert by["mug"]["in_tray"] == "black_tray"                     # tray added by the layout pass
    return objs


def test_resolve(objs):
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
    assert any("no clue" in h for h in resolve("pick up the red gear", objs))
    assert any("no such object" in h for h in resolve("pick up the blue cube", objs))
    assert any("unsafe" in h for h in resolve("throw the mug into the tray", objs))
    assert not any("so plan" in h for h in resolve("throw the mug into the tray", objs))
    assert any("cannot be picked" in h for h in resolve("pick up the grey tray", objs))
    allh = [h for h in resolve("put everything from the grey tray in the black tray", objs) if "all of them" in h][0]
    assert "red_gear_1" in allh and "mug" not in allh


def test_answer(objs):
    assert to_answer("garbage", objs, "x")["actions"][0]["question"] == FALLBACK_QUESTION
    ans = to_answer('{"actions": [{"pick": "red_gear_2", "place": "grey_tray"}, {"pick": "mug", "place": "none"}]}', objs, "x")
    by = {o["name"]: o for o in objs}
    assert ans["target_pixel"] == by["red_gear_2"]["centre"] and answer_type(ans) == ("plan", None)
    assert [a["action"] for a in ans["actions"]] == ["pick", "place", "pick"]
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
    assert any("no clue" in h for h in resolve("take it off the gear", objs))
    assert not any("exactly one" in h for h in resolve("take it off the mug", objs))
    hints = resolve("give me that one", objs)   # a refusal here must be steered to a question, not a plan
    issues = check(to_answer('{"actions": [{"refuse": "not there"}]}', objs, "x"), hints, objs)
    assert issues and "ask" in issues[0]
    hints = resolve("pick up the red gear closest to the mug", objs)   # resolved to red_gear_2, plan picks the mug
    assert check(to_answer('{"actions": [{"pick": "mug", "place": "none"}]}', objs, "x"), hints, objs)
    hints = resolve("put all the red gears in the black tray except the one on the right", objs)
    hints = resolve("put everything from the grey tray in the black tray", objs)  # = red_gear_1 only; mug is extra
    assert check(to_answer('{"actions": [{"pick": "red_gear_1", "place": "black_tray"}, {"pick": "mug", "place": "black_tray"}]}', objs, "x"), hints, objs)
    hints = resolve("put the red gear under the mug in the black tray", objs)
    assert any(h.startswith("first pick mug") for h in hints)
    assert check(to_answer('{"actions": [{"pick": "red_gear_1", "place": "black_tray"}]}', objs, "x"), hints, objs)


def test_pipeline():
    objs = test_table()
    test_resolve(objs)
    test_answer(objs)


if __name__ == "__main__":
    test_pipeline()
    print("ok")
