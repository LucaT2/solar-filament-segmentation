"""Print the key results from a downloaded Kaggle notebook log (tuning, classifier, comparison with the baseline).

Usage:
    python scripts/summarize_kaggle_log.py outputs/kaggle_runs/<run>/solar-fillament.log
"""

from __future__ import annotations

import json
import re
import sys

KEEP = re.compile(
    r"^={5,}|best: PostConfig|honest \(nested\)|^--- vs baseline|^[AB] = |common images|^\s*(PQ|SQ|RQ|TP|FP|FN)\s"
    r"|ΔPQ|^by site|^\s*site\s|^\s+[BCLMTU]\s+\d|filaments on|validation:"
)


def main() -> None:
    raw = open(sys.argv[1], encoding="utf-8").read()
    try:  # Kaggle stores the log as a JSON list of stream chunks
        text = "".join(chunk.get("data", "") for chunk in json.loads(raw))
    except json.JSONDecodeError:
        text = raw
    text = re.sub(r"\x1b\[[0-9;]*m", "", text)  # strip terminal colours
    for line in text.splitlines():
        if KEEP.search(line) and "wandb" not in line:
            print(line.rstrip())


if __name__ == "__main__":
    main()
