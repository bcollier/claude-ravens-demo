"""Which problems do the language models get wrong, and do they agree?

Two questions worth separating:

  * Are the failures the *same* problems across labs? Independent models failing
    independently looks like noise. Independent models failing on the same items,
    and picking the same wrong option, means the problem itself is doing
    something the whole class of model handles badly.
  * When they are wrong, are they wrong together? Convergence on one distractor
    is evidence of a systematic misreading; a scatter across options is evidence
    of guessing.

Usage:  python scripts/error_analysis.py [--min-acc 0.7]
"""
from __future__ import annotations

import argparse
import collections
import csv
import glob
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, os.path.join(ROOT, "common")]

import ravens   # noqa: E402

RESULTS = os.path.join(ROOT, "results")


def load():
    runs = {}
    for path in sorted(glob.glob(os.path.join(RESULTS, "*llm_*_answers.csv"))):
        rows = list(csv.DictReader(open(path)))
        if not rows or all(int(r["InputTokens"] or 0) == 0 for r in rows):
            continue
        sm = path.replace("_answers.csv", "_summary.txt")
        model = os.path.basename(path)
        if os.path.exists(sm):
            for line in open(sm):
                if line.startswith("model"):
                    model = line.split(":", 1)[1].strip()
        runs[model] = {r["RavensProblem"]: r for r in rows}
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-acc", type=float, default=0.70,
                    help="only count models at or above this accuracy as 'strong'")
    ap.add_argument("--top", type=int, default=10)
    args = ap.parse_args()

    problems = ravens.load_all()
    truth = {p.name: p.answer for p in problems}
    runs = load()
    acc = {m: sum(int(r["Correct"]) for r in d.values()) / len(d) for m, d in runs.items()}
    strong = [m for m, a in acc.items() if a >= args.min_acc]

    print(f"{len(runs)} runs loaded; {len(strong)} at or above {args.min_acc:.0%}: "
          + ", ".join(sorted(strong)) + "\n")

    rows = []
    for p in problems:
        wrong = [m for m in strong if int(runs[m][p.name]["Correct"]) == 0]
        picks = collections.Counter(int(runs[m][p.name]["Answer"]) for m in wrong)
        modal, modal_n = picks.most_common(1)[0] if picks else (None, 0)
        rows.append({"name": p.name, "truth": truth[p.name], "n_wrong": len(wrong),
                     "n_strong": len(strong), "picks": picks,
                     "modal": modal, "modal_n": modal_n, "wrong_models": wrong})
    rows.sort(key=lambda r: (-r["n_wrong"], -r["modal_n"]))

    print(f"{'problem':28s} {'truth':>5s} {'missed by':>10s}  {'answers given':<26s} agreement")
    print("-" * 92)
    for r in rows[:args.top]:
        if not r["n_wrong"]:
            break
        dist = " ".join(f"{k}x{v}" for k, v in sorted(r["picks"].items(), key=lambda t: -t[1]))
        agree = r["modal_n"] / r["n_wrong"]
        print(f"{r['name']:28s} {r['truth']:>5d} {r['n_wrong']:>4d}/{r['n_strong']:<5d} "
              f"{dist:<26s} {agree:.0%} on option {r['modal']}")

    # convergence: when strong models are wrong, do they agree with each other?
    multi = [r for r in rows if r["n_wrong"] >= 2]
    if multi:
        conv = sum(r["modal_n"] / r["n_wrong"] for r in multi) / len(multi)
        print(f"\nWhen two or more strong models miss the same problem, "
              f"{conv:.0%} of them pick the same wrong option on average.")
        print(f"Random guessing among 7 remaining options would give about 14%.")

    # do the models fail on the SAME problems, beyond what their accuracy implies?
    import itertools, random
    err = {m: {p.name for p in problems if int(runs[m][p.name]["Correct"]) == 0}
           for m in strong}
    obs, exp = [], []
    rng = random.Random(0)
    names = [p.name for p in problems]
    for a, b in itertools.combinations(strong, 2):
        A, B = err[a], err[b]
        if not A or not B:
            continue
        obs.append(len(A & B) / len(A | B))
        # same error counts, but scattered at random
        sims = []
        for _ in range(200):
            ra = set(rng.sample(names, len(A)))
            rb = set(rng.sample(names, len(B)))
            sims.append(len(ra & rb) / len(ra | rb))
        exp.append(sum(sims) / len(sims))
    if obs:
        print(f"\nOverlap of error sets between pairs of strong models (Jaccard): "
              f"{sum(obs)/len(obs):.2f}")
        print(f"If each model failed on a random selection of the same size: "
              f"{sum(exp)/len(exp):.2f}")
        print(f"So the failures are about {(sum(obs)/len(obs))/(sum(exp)/len(exp)):.1f}x "
              f"more shared than independent errors would be.")

    # how concentrated are the failures?
    missed_by = collections.Counter()
    for p in problems:
        k = sum(1 for m in strong if int(runs[m][p.name]["Correct"]) == 0)
        missed_by[k] += 1
    total_errors = sum(k * v for k, v in missed_by.items())
    print(f"\nOf {len(problems)} problems, {missed_by[0]} were solved by every strong "
          f"model. The {sum(v for k, v in missed_by.items() if k >= 2)} problems missed "
          f"by two or more account for "
          f"{sum(k*v for k, v in missed_by.items() if k >= 2)/max(total_errors,1):.0%} "
          f"of all errors.")

    # the stated rules, for the hardest few
    print("\n" + "=" * 92)
    for r in rows[:3]:
        if not r["n_wrong"]:
            break
        p = next(x for x in problems if x.name == r["name"])
        print(f"\n{r['name']}   correct = {r['truth']}   "
              f"missed by {r['n_wrong']}/{r['n_strong']} strong models")
        for m in r["wrong_models"]:
            rec = runs[m][r["name"]]
            print(f"  [{m}] said {rec['Answer']}: {rec['Rule'][:150]}")
        ok = [m for m in strong if int(runs[m][r["name"]]["Correct"]) == 1]
        for m in ok[:2]:
            print(f"  [{m}] CORRECT {runs[m][r['name']]['Answer']}: "
                  f"{runs[m][r['name']]['Rule'][:150]}")

    with open(os.path.join(RESULTS, "epilogue_error_analysis.txt"), "w") as fh:
        fh.write(f"strong models ({args.min_acc:.0%}+): {', '.join(sorted(strong))}\n\n")
        for r in rows:
            if not r["n_wrong"]:
                continue
            dist = " ".join(f"{k}x{v}" for k, v in sorted(r["picks"].items(),
                                                          key=lambda t: -t[1]))
            fh.write(f"{r['name']:28s} truth {r['truth']}  missed {r['n_wrong']}/"
                     f"{r['n_strong']}  answers {dist}\n")
    print(f"\nwrote {RESULTS}/epilogue_error_analysis.txt")


if __name__ == "__main__":
    main()
