"""Dollar costs for every LLM run, from OpenRouter's public price list.

The OpenAI runs were made directly against OpenAI, which does not return a cost
with the response -- only token counts. OpenRouter publishes list prices for the
same models on an endpoint that needs no key, so costs for those runs are
computed as tokens x list price and labelled as such. Runs made *through*
OpenRouter carry the exact charged amount, which the runner records per call.

Prices are snapshotted to results/pricing_snapshot.json so the numbers in the
report stay reproducible after the list changes.
"""
from __future__ import annotations

import json
import os
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SNAPSHOT = os.path.join(ROOT, "results", "pricing_snapshot.json")

# Models we called directly on OpenAI -> the OpenRouter id that lists the price.
DIRECT_TO_OPENROUTER = {
    "gpt-5.6-sol": "openai/gpt-5.6-sol",
    "gpt-5.6-terra": "openai/gpt-5.6-terra",
    "gpt-5.6-luna": "openai/gpt-5.6-luna",
    "gpt-5": "openai/gpt-5",
    "gpt-4o": "openai/gpt-4o",
    "gpt-4-turbo": "openai/gpt-4-turbo",
    "gpt-4.1": "openai/gpt-4.1",
    "o3": "openai/o3",
}


def fetch(refresh=False):
    if os.path.exists(SNAPSHOT) and not refresh:
        with open(SNAPSHOT) as fh:
            return json.load(fh)
    with urllib.request.urlopen("https://openrouter.ai/api/v1/models", timeout=60) as r:
        data = json.load(r)["data"]
    table = {}
    for m in data:
        p = m.get("pricing", {})
        try:
            table[m["id"]] = {"prompt": float(p.get("prompt", 0)),
                              "completion": float(p.get("completion", 0))}
        except (TypeError, ValueError):
            continue
    os.makedirs(os.path.dirname(SNAPSHOT), exist_ok=True)
    with open(SNAPSHOT, "w") as fh:
        json.dump(table, fh, indent=0, sort_keys=True)
    return table


def price_for(model, table):
    """Returns (prompt_per_token, completion_per_token, source) or None."""
    if model in table:
        return table[model]["prompt"], table[model]["completion"], "openrouter list"
    alias = DIRECT_TO_OPENROUTER.get(model)
    if alias and alias in table:
        return table[alias]["prompt"], table[alias]["completion"], "openrouter list"
    return None


def estimate(model, in_tokens, out_tokens, table=None):
    table = table if table is not None else fetch()
    got = price_for(model, table)
    if not got:
        return None, "unknown"
    pin, pout, src = got
    return in_tokens * pin + out_tokens * pout, src


if __name__ == "__main__":
    t = fetch(refresh=True)
    print(f"snapshotted {len(t)} model prices to {SNAPSHOT}")
    for m in DIRECT_TO_OPENROUTER:
        got = price_for(m, t)
        if got:
            print(f"  {m:16s} ${got[0]*1e6:6.2f}/M in  ${got[1]*1e6:7.2f}/M out")
