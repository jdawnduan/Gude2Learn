#!/usr/bin/env python3
"""One command for any folder of scenes: python run.py --data heldout --out results_heldout.json"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))   # works without `uv sync` too (pip install -r requirements.txt)
from see_it_say_it_pick_it.pipeline import main  # noqa: E402

main()
