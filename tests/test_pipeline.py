"""One check: vision JSON -> full-res table (numbering, tray), and Apertus reply -> schema-valid answer."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "starter"))
from evaluate import answer_type  # noqa: E402

from see_it_say_it_pick_it.pipeline import FALLBACK_QUESTION, scene_table, to_answer  # noqa: E402

SEEN = {"objects": [
    {"name": "red_gear", "bbox": [500, 500, 600, 600], "on_top_of": None},
    {"name": "red_gear", "bbox": [100, 500, 200, 600], "on_top_of": None},
    {"name": "grey_tray", "bbox": [50, 400, 300, 700], "on_top_of": None},
    {"name": "yellow_gear", "bbox": [520, 520, 580, 580], "on_top_of": "red_gear"},
]}


def test_pipeline():
    objs = scene_table(SEEN, 2448, 2048)
    assert [o["name"] for o in objs] == ["red_gear_1", "grey_tray", "red_gear_2", "yellow_gear"]  # left -> right
    assert objs[0]["centre"] == [367, 1126] and objs[0]["in_tray"] == "grey_tray"
    assert objs[3]["on_top_of"] == "red_gear_2"  # base name resolved to the nearest numbered object
    assert to_answer("garbage", objs, "x")["actions"][0]["question"] == FALLBACK_QUESTION
    ans = to_answer('{"actions": [{"action": "pick", "object": "red_gear_2"}, {"action": "place", "object": "grey_tray"}]}', objs, "x")
    assert ans["target_pixel"] == objs[2]["centre"] and answer_type(ans) == ("plan", None)
    ans = to_answer('{"actions": [{"action": "refuse", "reason": "no blue cube"}]}', objs, "x")
    assert ans["target_pixel"] is None and answer_type(ans) == ("refuse", None)


if __name__ == "__main__":
    test_pipeline()
    print("ok")
