"""Release dates for every model in the comparison.

OpenRouter records the date each model was first listed, which is the closest
public proxy for a release date and is consistent across companies. For models
called directly on OpenAI the id is mapped to its OpenRouter equivalent first.

Snapshotted to results/model_release_dates.json so the tables stay stable.
"""
from __future__ import annotations

import datetime
import json
import os
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import pricing   # noqa: E402

OUT = os.path.join(ROOT, "results", "model_release_dates.json")


def fetch(refresh=False):
    if os.path.exists(OUT) and not refresh:
        return json.load(open(OUT))
    with urllib.request.urlopen("https://openrouter.ai/api/v1/models", timeout=60) as r:
        data = json.load(r)["data"]
    created = {m["id"]: m.get("created") for m in data if m.get("created")}
    table = {}
    for mid, ts in created.items():
        table[mid] = datetime.date.fromtimestamp(ts).isoformat()
    # models we called directly on OpenAI, under the name we called them by
    for direct, alias in pricing.DIRECT_TO_OPENROUTER.items():
        if alias in table:
            table[direct] = table[alias]
    with open(OUT, "w") as fh:
        json.dump(table, fh, indent=0, sort_keys=True)
    return table


def date_for(model, table=None):
    table = table if table is not None else fetch()
    if model in table:
        return table[model]
    alias = pricing.DIRECT_TO_OPENROUTER.get(model)
    return table.get(alias)


if __name__ == "__main__":
    t = fetch(refresh=True)
    print(f"snapshotted {len(t)} release dates")
    import glob, csv
    for path in sorted(glob.glob(os.path.join(ROOT, "results", "*llm_*_summary.txt"))):
        model = next((l.split(":", 1)[1].strip() for l in open(path)
                      if l.startswith("model")), None)
        if model:
            print(f"  {model:42s} {date_for(model, t) or 'unknown'}")
