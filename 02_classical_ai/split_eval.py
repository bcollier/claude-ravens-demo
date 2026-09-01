"""Epilogue: re-score the learnable agents under a plain 70/30 train/test split.

The in-class headline was nested leave-one-problem-SET-out: train on seven sets,
test on the eighth. That is a harder test of transfer (the held-out set is a
problem family the model has never seen) but it trains on 84 problems. A random
stratified 70/30 split is the more conventional protocol and trains on 67. Both
are reported here, over many random seeds, so the spread is visible.

Three rankers, identical protocol:

  linear   the in-class pairwise logistic ranker
  mlp      a small neural network scoring each option, trained listwise
  hybrid   the same MLP with the relation network's score added as a feature

IMPORTANT CAVEAT, stated here because no split can fix it: the *rule families
and features* in features.py were designed while looking at all 96 problems --
set D's failures were rendered and inspected, and attribute descriptors were
added because of what they showed. Model weights and hyper-parameters are
honestly held out; the feature design is not. Treat these numbers as an upper
bound on what this feature set would score on genuinely unseen problems.

Usage:  python 02_classical_ai/split_eval.py --seeds 30
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, os.path.join(ROOT, "common")]

import ravens          # noqa: E402
import features        # noqa: E402
import solver          # noqa: E402
import torch           # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F   # noqa: E402

TAUS, GAMMAS, CS = solver.TAUS, solver.GAMMAS, solver.CS


# ---------------------------------------------------------------- rankers

class MLPRanker:
    """Listwise neural ranker: score each option, softmax over the group.

    Small on purpose -- 67 training problems is not much, so the network is
    sized to the data rather than to ambition."""

    def __init__(self, n_features, hidden=48, epochs=400, lr=3e-3, wd=3e-3,
                 dropout=0.3, seed=0):
        torch.manual_seed(seed)
        self.net = nn.Sequential(nn.Linear(n_features, hidden), nn.ReLU(),
                                 nn.Dropout(dropout), nn.Linear(hidden, 1))
        self.epochs, self.lr, self.wd = epochs, lr, wd

    @staticmethod
    def _pack(Xs, answers=None):
        """Pad every problem to 8 options so the whole training set is one tensor."""
        n_feat = Xs[0].shape[1]
        X = torch.zeros(len(Xs), 8, n_feat)
        mask = torch.zeros(len(Xs), 8, dtype=torch.bool)
        for i, Xi in enumerate(Xs):
            X[i, :len(Xi)] = torch.tensor(Xi, dtype=torch.float32)
            mask[i, :len(Xi)] = True
        y = torch.tensor([a - 1 for a in answers]) if answers is not None else None
        return X, mask, y

    def fit(self, Xs, answers):
        X, mask, y = self._pack(Xs, answers)
        opt = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=self.wd)
        self.net.train()
        for _ in range(self.epochs):
            opt.zero_grad()
            logits = self.net(X).squeeze(-1).masked_fill(~mask, -1e4)
            F.cross_entropy(logits, y).backward()
            opt.step()
        return self

    def predict(self, X):
        self.net.eval()
        with torch.no_grad():
            return int(self.net(torch.tensor(X, dtype=torch.float32)).squeeze(-1).argmax()) + 1


def fit_predict(kind, Xs_tr, ans_tr, Xs_te, C, seed):
    if kind == "linear":
        r = solver.PairwiseRanker(C).fit(Xs_tr, ans_tr)
        return [r.predict(X) for X in Xs_te]
    r = MLPRanker(Xs_tr[0].shape[1], seed=seed).fit(Xs_tr, ans_tr)
    return [r.predict(X) for X in Xs_te]


# ---------------------------------------------------------------- protocol

def stratified_split(problems, rng, frac=0.7):
    """Keep each problem set's proportion in both halves, and hit the target
    train size exactly (12 per set does not divide 70/30 cleanly)."""
    by_set = {}
    for i, p in enumerate(problems):
        by_set.setdefault(p.set_name, []).append(i)
    order = sorted(by_set)
    shuffled = {}
    for s in order:
        idx = list(by_set[s])
        rng.shuffle(idx)
        shuffled[s] = idx
    per_set = int(len(problems) * frac) // len(order)          # 8 of each 12
    extra = int(len(problems) * frac) - per_set * len(order)   # 3 left over
    bonus = set(rng.sample(order, extra))
    tr, te = [], []
    for s in order:
        k = per_set + (1 if s in bonus else 0)
        tr += shuffled[s][:k]
        te += shuffled[s][k:]
    return sorted(tr), sorted(te)


def pick_hyper(problems, Xgrid, train_idx, folds, kind, seed):
    """Choose (tau, gamma, C) using ONLY the training half."""
    inner = sorted({folds[i] for i in train_idx})
    best, best_acc = None, -1.0
    for (tau, gamma), Xs in Xgrid.items():
        for C in (CS if kind == "linear" else [0.1]):
            hits = tot = 0
            for g in inner:
                itr = [i for i in train_idx if folds[i] != g]
                ite = [i for i in train_idx if folds[i] == g]
                if not ite or not itr:
                    continue
                pr = fit_predict(kind, [Xs[i] for i in itr], [problems[i].answer for i in itr],
                                 [Xs[i] for i in ite], C, seed)
                hits += sum(a == problems[i].answer for a, i in zip(pr, ite))
                tot += len(ite)
            acc = hits / max(tot, 1)
            if acc > best_acc:
                best_acc, best = acc, (tau, gamma, C)
    return best


def run(problems, Xgrid, kind, seeds, frac=0.7):
    folds = [p.set_name for p in problems]
    accs, sizes = [], None
    for seed in range(seeds):
        rng = np.random.default_rng(seed)
        pyr = __import__("random").Random(seed)
        train_idx, test_idx = stratified_split(problems, pyr, frac)
        sizes = (len(train_idx), len(test_idx))
        # (tau, gamma, C) are chosen with the cheap linear ranker on the training
        # half only; the MLP reuses that setting rather than re-searching, which
        # would cost 128 network fits per seed for no measurable gain.
        tau, gamma, C = pick_hyper(problems, Xgrid, train_idx, folds, "linear", seed)
        Xs = Xgrid[(tau, gamma)]
        pr = fit_predict(kind, [Xs[i] for i in train_idx],
                         [problems[i].answer for i in train_idx],
                         [Xs[i] for i in test_idx], C, seed)
        accs.append(sum(a == problems[i].answer for a, i in zip(pr, test_idx)) / len(test_idx))
    return np.array(accs), sizes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--kinds", nargs="*", default=["linear", "mlp"])
    ap.add_argument("--out", default=os.path.join(ROOT, "results"))
    args = ap.parse_args()

    t0 = time.time()
    problems = ravens.load_all()
    cache = solver.score_all(problems)
    print("building feature grid ...")
    Xgrid = {(t, g): [features.feature_matrix(p, cache[p.name], t, g) for p in problems]
             for t in TAUS for g in GAMMAS}

    # optional hybrid feature: the relation network's score per option
    npy = os.path.join(args.out, "neural_relationnet_scores.npy")
    if os.path.exists(npy) and "hybrid" in args.kinds:
        raw = np.load(npy)
        hyb = {}
        for key, Xs in Xgrid.items():
            new = []
            for i, p in enumerate(problems):
                col = features._znorm(raw[i][:p.n_choices])[:, None]
                new.append(np.hstack([Xs[i], col]))
            hyb[key] = new
        Xgrid_h = hyb
    else:
        Xgrid_h = None
        args.kinds = [k for k in args.kinds if k != "hybrid"]

    results = {}
    for kind in args.kinds:
        grid = Xgrid_h if kind == "hybrid" else Xgrid
        k = "mlp" if kind == "hybrid" else kind
        accs, sizes = run(problems, grid, k, args.seeds)
        results[kind] = {"mean": float(accs.mean()), "std": float(accs.std()),
                         "min": float(accs.min()), "max": float(accs.max()),
                         "seeds": args.seeds, "train_n": sizes[0], "test_n": sizes[1],
                         "per_seed": [round(float(a), 4) for a in accs]}
        print(f"  {kind:8s} 70/30 split, {args.seeds} seeds: "
              f"{accs.mean():.1%} +/- {accs.std():.1%}  "
              f"(min {accs.min():.1%}, max {accs.max():.1%})  "
              f"train {sizes[0]} / test {sizes[1]}")

    results["_runtime_seconds"] = round(time.time() - t0, 1)
    with open(os.path.join(args.out, "epilogue_split_eval.json"), "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nwrote {args.out}/epilogue_split_eval.json  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
