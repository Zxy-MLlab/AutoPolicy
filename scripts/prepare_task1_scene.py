#!/usr/bin/env python3
"""Render the tracked task1 MJCF template for this checkout location."""

from __future__ import annotations

import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "data/real/task1/model/task1_yam_bottle.template.xml"
SCENE = ROOT / "data/real/task1/model/task1_yam_bottle.xml"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="validate an existing rendered scene")
    args = parser.parse_args()
    expected = TEMPLATE.read_text(encoding="utf-8").replace("__AUTOPOLICY_ROOT__", str(ROOT))
    if args.check:
        if not SCENE.is_file() or SCENE.read_text(encoding="utf-8") != expected:
            raise SystemExit(f"task1 scene is missing or stale; run {__file__}")
    else:
        SCENE.write_text(expected, encoding="utf-8")
    print(SCENE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
