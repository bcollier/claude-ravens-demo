"""What did the relation network actually learn?

Two diagnostics, both cheap, both worth running on any model that scores badly
for reasons you cannot name:

  1. Context-blind test. Erase every context panel and re-score. If accuracy
     holds up, the model is reading something in the answer options rather than
     solving the matrix -- a classic artefact of how distractors are generated.

  2. Duplicate-shortcut test. On the real problems, how often does the model
     pick the option most similar to a panel already visible, and how often is
     that actually the right answer?

Usage:  python 04_neural/diagnose.py [--tag relationnet]
"""
from __future__ import annotations

import argparse
import os
import random
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, os.path.join(ROOT, "common"), os.path.join(ROOT, "02_classical_ai")]

import ravens            # noqa: E402
import imageops as io    # noqa: E402
import wren              # noqa: E402
import solver as nsolver # noqa: E402


def context_blind(model, dev, n=512, seed=4242):
    out = {}
    for blank in (False, True):
        rng = random.Random(seed)
        P, S, M, O, Y, NC = nsolver.synth_batch(rng, n)
        if blank:
            P = P.copy()
            P[:, :8] = 0.0
        t = nsolver.to_torch((P, S, M, O, Y, NC), dev)
        with torch.no_grad():
            lg = model(t[0], t[1], t[2], n_ctx=8).masked_fill(t[3] == 0, -1e4)
        out["blind" if blank else "full"] = (lg.argmax(1) == t[4]).float().mean().item()
    return out


def duplicate_shortcut(scores):
    ps = ravens.load_all()
    pick_dup = truth_dup = 0
    for i, p in enumerate(ps):
        givens = [p.image(g) for g in p.givens]
        dup = np.array([max(io.sim(p.image(c), g) for g in givens) for c in p.choices])
        most = int(np.argmax(dup))
        pick_dup += int(np.argmax(scores[i][:p.n_choices])) == most
        truth_dup += (p.answer - 1) == most
    return pick_dup / len(ps), truth_dup / len(ps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="relationnet")
    args = ap.parse_args()

    dev = nsolver.device()
    model = wren.RelationNet().to(dev)
    ck = torch.load(os.path.join(HERE, f"{args.tag}.pt"), map_location=dev)
    model.load_state_dict(ck["state"])
    model.eval()

    cb = context_blind(model, dev)
    scores = np.load(os.path.join(ROOT, "results", f"neural_{args.tag}_scores.npy"))
    pick, truth = duplicate_shortcut(scores)

    lines = [
        f"context-blind test (synthetic)",
        f"  full problem                      {cb['full']:.1%}",
        f"  every context panel erased        {cb['blind']:.1%}   (chance ~13%)",
        f"",
        f"duplicate-shortcut test (real 96)",
        f"  model picks the option most like a visible panel   {pick:.0%}",
        f"  that option is actually correct                    {truth:.0%}",
    ]
    out = "\n".join(lines)
    print(out)
    with open(os.path.join(ROOT, "results", f"epilogue_neural_{args.tag}_diagnostics.txt"),
              "w") as fh:
        fh.write(out + "\n")


if __name__ == "__main__":
    main()
