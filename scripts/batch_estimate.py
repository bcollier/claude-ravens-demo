"""What would these runs have cost in batch mode?

Every major provider sells an asynchronous batch tier at a discount: you upload
a file of requests, the provider works through them within a deadline (24 hours
is typical), and you pay less per token. OpenRouter publishes the batch price as
a separate `:batch` model id, so this needs no guessing -- it multiplies the
token counts already recorded by the published batch rate.

Nothing is actually submitted. This is an estimate from real token counts and
real list prices.

Usage:  python scripts/batch_estimate.py
"""
from __future__ import annotations

import csv
import glob
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import pricing   # noqa: E402

RESULTS = os.path.join(ROOT, "results")


def runs():
    """(model, in_tokens, out_tokens, correct, charged_cost_or_None)."""
    out = []
    for path in sorted(glob.glob(os.path.join(RESULTS, "*llm_*_answers.csv"))):
        rows = list(csv.DictReader(open(path)))
        summary = path.replace("_answers.csv", "_summary.txt")
        model = os.path.basename(path)
        if os.path.exists(summary):
            for line in open(summary):
                if line.startswith("model"):
                    model = line.split(":", 1)[1].strip()
        in_t = sum(int(r["InputTokens"]) for r in rows)
        out_t = sum(int(r["OutputTokens"]) for r in rows)
        if not in_t:
            continue
        charged = sum(float(r.get("CostUSD") or 0) for r in rows) or None
        out.append((model, in_t, out_t, sum(int(r["Correct"]) for r in rows), charged))
    return out


def main():
    table = pricing.fetch()
    resolved = {}
    for model, *_ in runs():
        base = model if model in table else pricing.DIRECT_TO_OPENROUTER.get(model)
        resolved[model] = (base, (base + ":batch") if base and base + ":batch" in table
                           else None)

    lines = []
    w = lines.append
    w("| Model | Score | Tokens in / out | Standard | Batch | Saving |")
    w("|---|---|---|---|---|---|")
    tot_std = tot_batch = 0.0
    missing = []
    for model, in_t, out_t, correct, charged in runs():
        base, batch = resolved[model]
        if not base:
            missing.append(model)
            continue
        p = table[base]
        std = in_t * p["prompt"] + out_t * p["completion"]
        if batch:
            b = table[batch]
            bat = in_t * b["prompt"] + out_t * b["completion"]
            saving = f"{(1 - bat / std) * 100:.0f}%" if std else "—"
            bs = f"${bat:.2f}"
        else:
            bat, bs, saving = std, "—", "no batch tier"
            missing.append(model)
        tot_std += std
        tot_batch += bat
        w(f"| `{model}` | {correct}/96 | {in_t:,} / {out_t:,} | ${std:.2f} | {bs} | {saving} |")
    w(f"| **total** | | | **${tot_std:.2f}** | **${tot_batch:.2f}** | "
      f"**{(1 - tot_batch / tot_std) * 100:.0f}%** |")
    out = "\n".join(lines)
    print(out)
    if missing:
        print("\nno published batch variant: " + ", ".join(sorted(set(missing))))
    with open(os.path.join(RESULTS, "epilogue_batch_estimate.md"), "w") as fh:
        fh.write(out + "\n")
        if missing:
            fh.write("\nNo published batch variant: "
                     + ", ".join(f"`{m}`" for m in sorted(set(missing))) + "\n")
    print(f"\nwrote {RESULTS}/epilogue_batch_estimate.md")


if __name__ == "__main__":
    main()
