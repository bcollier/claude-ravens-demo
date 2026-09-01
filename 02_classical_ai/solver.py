"""Classical-AI Raven's solver: symbolic rule search + a learned ranker.

  1. Generate-and-test -- enumerate a few hundred candidate rules per problem
     (geometric transforms, pixel set algebra, numeric progressions, Latin
     squares over decomposed attributes, relational patterns).
  2. Validate -- hide a line of the matrix whose answer is visible, make each
     rule recover it against the real distractors, and keep the likelihood it
     assigned to the truth. That is the rule's trust score.
  3. Rank -- each family casts a confidence-scaled vote for every option. A
     pairwise logistic ranker learns how much to trust each family.

Everything is reported under nested leave-one-problem-set-out cross-validation:
hyper-parameters are chosen inside the training folds only, so the held-out set
is never touched by any fitting decision.

Usage
    python solver.py                      # full evaluation, writes ../results
    python solver.py --explain "Basic Problem D-09"
"""
from __future__ import annotations

import argparse
import csv
import os
import pickle
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "..", "common")]

import ravens                                          # noqa: E402
import features                                        # noqa: E402
from sklearn.linear_model import LogisticRegression    # noqa: E402

CACHE = os.path.join(HERE, ".rule_cache.pkl")
TAUS = [0.35, 0.5, 0.8, 1.2]
GAMMAS = [1.0, 2.0, 3.0, 5.0]
CS = [0.01, 0.03, 0.1, 0.3, 1.0]


# --------------------------------------------------------------- rule scoring

def score_all(problems, refresh=False):
    if os.path.exists(CACHE) and not refresh:
        with open(CACHE, "rb") as fh:
            cache = pickle.load(fh)
        if set(cache) == {p.name for p in problems}:
            return cache
    t0 = time.time()
    cache = {}
    for i, p in enumerate(problems, 1):
        cache[p.name] = features.score_rules(p)
        if i % 24 == 0:
            print(f"    scored {i}/{len(problems)} ({time.time()-t0:.0f}s)")
    with open(CACHE, "wb") as fh:
        pickle.dump(cache, fh)
    print(f"  rule generation + validation: {time.time()-t0:.1f}s")
    return cache


# --------------------------------------------------------------- ranker

class PairwiseRanker:
    """Logistic regression over (correct - wrong) feature differences."""

    def __init__(self, C=0.1):
        self.model = LogisticRegression(C=C, fit_intercept=False, max_iter=5000)

    def fit(self, Xs, answers):
        diffs, labels = [], []
        for X, ans in zip(Xs, answers):
            pos = X[ans - 1]
            for k in range(len(X)):
                if k == ans - 1:
                    continue
                d = pos - X[k]
                diffs += [d, -d]
                labels += [1, 0]
        self.model.fit(np.array(diffs), np.array(labels))
        return self

    def predict(self, X):
        return int(np.argmax(X @ self.model.coef_.ravel())) + 1


# --------------------------------------------------------------- evaluation

def accuracy(problems, preds, idx=None):
    idx = range(len(problems)) if idx is None else idx
    idx = list(idx)
    return sum(preds[i] == problems[i].answer for i in idx) / len(idx)


def per_set(problems, preds):
    out = {}
    for i, p in enumerate(problems):
        c, n = out.get(p.set_name, (0, 0))
        out[p.set_name] = (c + (preds[i] == p.answer), n + 1)
    return out


def unsup_preds(problems, cache, tau, gamma):
    return [int(np.argmax(features.unsupervised_score(p, cache[p.name], tau, gamma))) + 1
            for p in problems]


def nested_unsupervised(problems, cache, folds):
    """No training, but tau/gamma still chosen only on the other folds."""
    grid = {(t, g): unsup_preds(problems, cache, t, g) for t in TAUS for g in GAMMAS}
    preds = [0] * len(problems)
    chosen = {}
    for f in sorted(set(folds)):
        tr = [i for i, x in enumerate(folds) if x != f]
        te = [i for i, x in enumerate(folds) if x == f]
        best = max(grid, key=lambda k: accuracy(problems, grid[k], tr))
        chosen[f] = best
        for i in te:
            preds[i] = grid[best][i]
    return preds, chosen


def nested_learned(problems, cache, folds, Xgrid):
    """Outer: leave-one-set-out. Inner: pick (tau, gamma, C) on training folds."""
    preds = [0] * len(problems)
    chosen = {}
    outer = sorted(set(folds))
    for f in outer:
        tr = [i for i, x in enumerate(folds) if x != f]
        te = [i for i, x in enumerate(folds) if x == f]
        inner_folds = sorted({folds[i] for i in tr})

        best_key, best_acc = None, -1.0
        for (tau, gamma), Xs in Xgrid.items():
            for C in CS:
                hits = 0
                for g in inner_folds:
                    itr = [i for i in tr if folds[i] != g]
                    ite = [i for i in tr if folds[i] == g]
                    r = PairwiseRanker(C).fit([Xs[i] for i in itr],
                                              [problems[i].answer for i in itr])
                    hits += sum(r.predict(Xs[i]) == problems[i].answer for i in ite)
                acc = hits / len(tr)
                if acc > best_acc:
                    best_acc, best_key = acc, (tau, gamma, C)
        tau, gamma, C = best_key
        chosen[f] = best_key
        Xs = Xgrid[(tau, gamma)]
        r = PairwiseRanker(C).fit([Xs[i] for i in tr], [problems[i].answer for i in tr])
        for i in te:
            preds[i] = r.predict(Xs[i])
    return preds, chosen


# --------------------------------------------------------------- explain

def explain(problems, cache, name, top=10):
    p = next(x for x in problems if x.name == name)
    _, scored = cache[p.name]
    rows = []
    for fam, rs in scored.items():
        v, a = rs.vconf(features.VALIDATION_TAU), rs.agree
        w = v * (0.7 + 0.3 * a)
        for i, rn in enumerate(rs.names):
            rows.append((w[i], v[i], a[i], fam, rn, int(np.argmax(rs.fits[i])) + 1))
    rows.sort(reverse=True)
    print(f"\n{p.name}   ({p.problem_type}, correct answer = {p.answer})")
    print(f"  {'trust':>6} {'held-out':>9} {'holds':>6}  {'family':<14} {'rule':<26} picks")
    for w, v, a, fam, rn, pick in rows[:top]:
        mark = " <-- correct" if pick == p.answer else ""
        print(f"  {w:6.3f} {v:9.3f} {a:6.3f}  {fam:<14} {rn:<26} {pick}{mark}")
    s = features.unsupervised_score(p, cache[p.name])
    print(f"  combined vote -> {int(np.argmax(s)) + 1}")


# --------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "..", "results"))
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--explain", metavar="PROBLEM")
    args = ap.parse_args()

    t0 = time.time()
    problems = ravens.load_all()
    print(f"Loaded {len(problems)} problems")
    print("Generating and validating rules ...")
    cache = score_all(problems, args.refresh)

    if args.explain:
        explain(problems, cache, args.explain)
        return

    os.makedirs(args.out, exist_ok=True)
    folds = [p.set_name for p in problems]

    print("Building feature grid ...")
    Xgrid = {(t, g): [features.feature_matrix(p, cache[p.name], t, g) for p in problems]
             for t in TAUS for g in GAMMAS}
    print(f"  {len(Xgrid)} (tau, gamma) settings x {Xgrid[(TAUS[0], GAMMAS[0])][0].shape[1]} features")

    print("\nEvaluating (nested leave-one-problem-set-out) ...")
    base_preds, base_choice = nested_unsupervised(problems, cache, folds)
    base_acc = accuracy(problems, base_preds)
    print(f"  [A] rule search only, no training : "
          f"{round(base_acc*96)}/96  {base_acc:.1%}")

    lp, lchoice = nested_learned(problems, cache, folds, Xgrid)
    lacc = accuracy(problems, lp)
    print(f"  [B] rule search + learned ranker  : {round(lacc*96)}/96  {lacc:.1%}")

    # [C] standard leave-one-problem-out, using the settings the nested search
    # picked most often. More training data per fold, so it is the friendlier
    # of the two honest estimates -- reported alongside, not as the headline.
    modal = max(set(lchoice.values()), key=list(lchoice.values()).count)
    tau, gamma, C = modal
    Xs = Xgrid[(tau, gamma)]
    loo = [0] * len(problems)
    for i in range(len(problems)):
        tr = [j for j in range(len(problems)) if j != i]
        r = PairwiseRanker(C).fit([Xs[j] for j in tr], [problems[j].answer for j in tr])
        loo[i] = r.predict(Xs[i])
    loo_acc = accuracy(problems, loo)
    print(f"  [C] same, leave-one-PROBLEM-out   : {round(loo_acc*96)}/96  {loo_acc:.1%}"
          f"   (tau={tau}, gamma={gamma}, C={C})")

    # [D] fitted and scored on the same 96 problems. Not a result -- printed so
    # the gap between it and [B] is visible.
    ins = PairwiseRanker(C).fit(Xs, [p.answer for p in problems])
    ins_preds = [ins.predict(X) for X in Xs]
    ins_acc = accuracy(problems, ins_preds)
    print(f"  [D] in-sample (NOT a fair score)  : {round(ins_acc*96)}/96  {ins_acc:.1%}")

    print("\nPer set (nested leave-one-set-out):")
    pb, pl = per_set(problems, base_preds), per_set(problems, lp)
    print(f"  {'set':26s} {'no training':>12s} {'+ ranker':>10s}")
    for s in ravens.SET_ORDER:
        print(f"  {s:26s} {pb[s][0]:>9d}/12 {pl[s][0]:>7d}/12")

    runtime = time.time() - t0
    with open(os.path.join(args.out, "classical_answers.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["ProblemSet", "RavensProblem", "Answer", "Correct", "Truth",
                    "AnswerNoTraining", "CorrectNoTraining"])
        for i, p in enumerate(problems):
            w.writerow([p.set_name, p.name, lp[i], int(lp[i] == p.answer), p.answer,
                        base_preds[i], int(base_preds[i] == p.answer)])
    with open(os.path.join(args.out, "classical_summary.txt"), "w") as fh:
        fh.write(f"rule_search_only        : {base_acc:.4f}\n")
        fh.write(f"rule_search_plus_ranker : {lacc:.4f}\n")
        fh.write(f"leave_one_problem_out   : {loo_acc:.4f}\n")
        fh.write(f"in_sample_not_a_result  : {ins_acc:.4f}\n")
        fh.write(f"runtime_seconds         : {runtime:.2f}\n")
        fh.write(f"hyperparams_per_fold    : {lchoice}\n")

    # what did the ranker learn to trust? (fit on everything, for the write-up)
    coef = ins.model.coef_.ravel()
    print("\nWhat the ranker learned to trust (top 12):")
    for i in np.argsort(-np.abs(coef))[:12]:
        print(f"  {features.FEATURE_NAMES[i]:24s} {coef[i]:+.3f}")
    np.save(os.path.join(args.out, "classical_coef.npy"), coef)
    print(f"\nTotal runtime {runtime:.1f}s")


if __name__ == "__main__":
    main()
