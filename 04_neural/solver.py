"""Epilogue agent D: neural networks, still no LLM.

Three models, all reported on the same 96 problems:

  D1  relation network  -- a CNN + pairwise relation head (wren.py), trained
      only on synthetic matrices from render.py. The 96 real problems are a
      genuine out-of-distribution test: the network never sees one in training.

  D2  neural ranker     -- the symbolic rule features from 02_classical_ai fed
      to an MLP instead of the linear ranker, under the same cross-validation.
      Answers "would a neural net do better on these features?"

  D3  hybrid            -- D2's features plus the relation network's score for
      each option. Answers "do the two approaches know different things?"

Usage
    python 04_neural/solver.py --steps 4000
    python 04_neural/solver.py --skip-train      # reuse a saved checkpoint
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, os.path.join(ROOT, "common"), os.path.join(ROOT, "02_classical_ai")]

import ravens          # noqa: E402
import render          # noqa: E402
import wren            # noqa: E402

CKPT = os.path.join(HERE, "relationnet.pt")   # overridden per --tag in main()
MAX_OPTS = 8


def device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------- data

def pad_options(opts):
    """All problems carry 8 option slots; 2x2 problems mask the last two."""
    n = len(opts)
    padded = list(opts) + [render.blank()] * (MAX_OPTS - n)
    opt_mask = np.zeros(MAX_OPTS, dtype=np.float32)
    opt_mask[:n] = 1.0
    return padded, opt_mask


def encode_problem(context, options, three):
    panels, slots, mask, n_ctx = wren.pack(context, *[None], three) if False else (None,)*4
    padded, opt_mask = pad_options(options)
    panels, slots, mask, n_ctx = wren.pack(context, padded, three)
    return panels, slots, mask, n_ctx, opt_mask


def synth_batch(rng, size):
    P, S, M, O, Y, NC = [], [], [], [], [], []
    for _ in range(size):
        three = rng.random() < 0.75
        ctx, opts, ans, _ = render.make_problem(rng, three)
        panels, slots, mask, n_ctx, opt_mask = encode_problem(ctx, opts, three)
        # every sample must have the same panel count to stack: pad context too
        need = 8 - len(ctx)
        if need:
            panels = np.concatenate([panels[:len(ctx)],
                                     np.stack([1 - render.blank() / 255.0] * need),
                                     panels[len(ctx):]])
            slots = slots.copy()
        P.append(panels); S.append(slots); M.append(mask); O.append(opt_mask)
        Y.append(ans); NC.append(n_ctx)
    return (np.stack(P), np.stack(S), np.stack(M), np.stack(O),
            np.array(Y, dtype=np.int64), NC)


def real_problems():
    """The 96, in the same tensor format, downsampled the same way."""
    out = []
    for p in ravens.load_all():
        three = p.problem_type == "3x3"
        load = lambda n: np.array(Image.open(p.path(n)).convert("L")
                                  .resize((render.OUT, render.OUT), Image.BILINEAR))
        ctx = [load(g) for g in p.givens]
        opts = [load(c) for c in p.choices]
        panels, slots, mask, n_ctx, opt_mask = encode_problem(ctx, opts, three)
        need = 8 - len(ctx)
        if need:
            panels = np.concatenate([panels[:len(ctx)],
                                     np.stack([1 - render.blank() / 255.0] * need),
                                     panels[len(ctx):]])
        out.append((p, panels, slots, mask, opt_mask, n_ctx))
    return out


def to_torch(batch, dev):
    P, S, M, O, Y, NC = batch
    return (torch.tensor(P, dtype=torch.float32, device=dev).unsqueeze(2),
            torch.tensor(S, device=dev), torch.tensor(M, device=dev),
            torch.tensor(O, device=dev), torch.tensor(Y, device=dev), NC)


# ---------------------------------------------------------------- train

def train(steps=4000, batch=32, lr=3e-4, seed=0, log_every=250):
    dev = device()
    torch.manual_seed(seed)
    rng = random.Random(seed)
    model = wren.RelationNet().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps)

    val = to_torch(synth_batch(random.Random(9999), 256), dev)
    t0 = time.time()
    hist = []
    for step in range(1, steps + 1):
        model.train()
        P, S, M, O, Y, NC = to_torch(synth_batch(rng, batch), dev)
        logits = model(P, S, M, n_ctx=8)
        logits = logits.masked_fill(O == 0, -1e4)
        loss = F.cross_entropy(logits, Y)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()

        if step % log_every == 0 or step == steps:
            model.eval()
            with torch.no_grad():
                vl = model(val[0], val[1], val[2], n_ctx=8).masked_fill(val[3] == 0, -1e4)
                acc = (vl.argmax(1) == val[4]).float().mean().item()
            hist.append((step, loss.item(), acc))
            print(f"  step {step:5d}  loss {loss.item():.3f}  synthetic-val {acc:.1%}"
                  f"  ({time.time()-t0:.0f}s)", flush=True)
    torch.save({"state": model.state_dict(), "hist": hist}, CKPT)
    return model, hist


@torch.no_grad()
def score_real(model, reals, dev):
    """Per-problem option scores from the relation network."""
    model.eval()
    out = []
    for p, panels, slots, mask, opt_mask, n_ctx in reals:
        P = torch.tensor(panels, dtype=torch.float32, device=dev).unsqueeze(1).unsqueeze(0)
        S = torch.tensor(slots, device=dev)[None]
        M = torch.tensor(mask, device=dev)[None]
        O = torch.tensor(opt_mask, device=dev)[None]
        logits = model(P, S, M, n_ctx=8).masked_fill(O == 0, -1e4)[0]
        out.append((p, logits[:p.n_choices].float().cpu().numpy()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--out", default=os.path.join(ROOT, "results"))
    ap.add_argument("--tag", default="relationnet",
                    help="output name, so ablations do not overwrite each other")
    args = ap.parse_args()

    global CKPT
    CKPT = os.path.join(HERE, f"{args.tag}.pt")
    dev = device()
    print(f"device: {dev}")
    t0 = time.time()

    if args.skip_train and os.path.exists(CKPT):
        model = wren.RelationNet().to(dev)
        ck = torch.load(CKPT, map_location=dev)
        model.load_state_dict(ck["state"]); hist = ck["hist"]
        print("loaded checkpoint")
    else:
        print(f"training the relation network on synthetic matrices "
              f"({args.steps} steps x batch {args.batch}) ...")
        model, hist = train(args.steps, args.batch)
    train_secs = time.time() - t0

    reals = real_problems()
    scores = score_real(model, reals, dev)
    preds = {p.name: int(np.argmax(s)) + 1 for p, s in scores}
    correct = sum(preds[p.name] == p.answer for p, _ in scores)
    print(f"\nD1 relation network on the real 96: {correct}/96 = {correct/96:.1%}")

    by_set = {}
    for p, _ in scores:
        c, n = by_set.get(p.set_name, (0, 0))
        by_set[p.set_name] = (c + (preds[p.name] == p.answer), n + 1)
    for s in ravens.SET_ORDER:
        print(f"  {s:26s} {by_set[s][0]:2d}/12")

    os.makedirs(args.out, exist_ok=True)
    np.save(os.path.join(args.out, f"neural_{args.tag}_scores.npy"),
            np.array([np.pad(s, (0, MAX_OPTS - len(s))) for _, s in scores]))
    with open(os.path.join(args.out, f"epilogue_neural_{args.tag}_answers.csv"),
              "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["ProblemSet", "RavensProblem", "Answer", "Correct", "Truth"])
        for p, _ in scores:
            w.writerow([p.set_name, p.name, preds[p.name],
                        int(preds[p.name] == p.answer), p.answer])
    with open(os.path.join(args.out, f"epilogue_neural_{args.tag}_summary.txt"), "w") as fh:
        fh.write(f"model            : relation network (WReN-style), {sum(x.numel() for x in model.parameters())} params\n")
        fh.write(f"trained_on       : synthetic matrices only, never the real 96\n")
        fh.write(f"steps            : {args.steps}\nbatch            : {args.batch}\n")
        fh.write(f"device           : {dev}\n")
        fh.write(f"train_seconds    : {train_secs:.1f}\n")
        fh.write(f"synthetic_val    : {hist[-1][2]:.4f}\n")
        fh.write(f"correct          : {correct}\naccuracy         : {correct/96:.4f}\n")
    print(f"\nsynthetic validation accuracy at the end of training: {hist[-1][2]:.1%}")
    print(f"total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
