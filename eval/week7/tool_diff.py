"""Diff of the tool descriptions and parameter enums.

    python eval/week7/tool_diff.py        # writes eval/week7/third_tool.diff

Two diffs:
  week7_base -> final        what adding get_jurisdiction_rules did to the whole registry
  v2a_first_cut -> final     the sharpening forced by thrash observed in the race dry-run
No lexical overlap score: search_handbook and get_jurisdiction_rules rightly share topic words
(PTO, notice, sick) and differ by scope, which word counts cannot see. Overlap is judged by
behaviour instead - which tool the agent picks for one jurisdiction's figure (see README).
"""

import difflib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from app.agent.tools import TOOL_SPECS  # noqa: E402

SPECS = os.path.join(HERE, "tool_specs")


def lines(specs):
    return json.dumps(specs, indent=2, ensure_ascii=False).splitlines(keepends=True)


def main():
    base = json.load(open(os.path.join(SPECS, "week7_base.json"), encoding="utf-8"))
    first = json.load(open(os.path.join(SPECS, "v2a_first_cut.json"), encoding="utf-8"))
    with open(os.path.join(SPECS, "final.json"), "w", encoding="utf-8") as f:
        json.dump(TOOL_SPECS, f, indent=2, ensure_ascii=False)

    out = list(difflib.unified_diff(lines(base), lines(TOOL_SPECS), "tool_specs/week7_base.json", "tool_specs/final.json"))
    out.append("\n")
    out += difflib.unified_diff(lines(first), lines(TOOL_SPECS), "tool_specs/v2a_first_cut.json", "tool_specs/final.json")
    text = "".join(out)
    with open(os.path.join(HERE, "third_tool.diff"), "w", encoding="utf-8") as f:
        f.write(text)
    print(text)


if __name__ == "__main__":
    main()
