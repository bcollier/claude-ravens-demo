"""How much of the ceiling is generation, and how much is selection?

Three numbers, all over the same 96 problems:

  any rule           some rule somewhere picks the right answer
  best-family oracle if an oracle told us which family was in force, and we
                     took that family's most-trusted rule, we would be right
  most-trusted rule  what you get by just believing the single highest-scoring
                     rule in the whole problem

The gap between the second and the third is the part of the problem that is
about *choosing* a rule rather than *finding* one. That gap is what the ranker
in solver.py is for.
"""
from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "..", "common")]

import ravens        # noqa: E402
import features      # noqa: E402
import solver        # noqa: E402


def main():
    problems = ravens.load_all()
    cache = solver.score_all(problems)
    tau = features.VALIDATION_TAU
    stats = {k: {s: 0 for s in ravens.SET_ORDER} for k in
             ("any rule", "best-family oracle", "most-trusted rule")}

    for p in problems:
        _, scored = cache[p.name]
        any_hit = fam_hit = False
        top_w, top_pick = -1.0, None
        for fam, rs in scored.items():
            w = rs.vconf(tau) * (0.7 + 0.3 * rs.agree)
            picks = np.argmax(rs.fits, axis=1) + 1
            if (picks == p.answer).any():
                any_hit = True
            best = int(np.argmax(w))
            if picks[best] == p.answer:
                fam_hit = True
            if w[best] > top_w:
                top_w, top_pick = w[best], picks[best]
        stats["any rule"][p.set_name] += any_hit
        stats["best-family oracle"][p.set_name] += fam_hit
        stats["most-trusted rule"][p.set_name] += (top_pick == p.answer)

    lines = [f"{'':22s}" + "".join(f"{ravens.set_label(s):>13s}" for s in ravens.SET_ORDER)
             + f"{'total':>9s}"]
    for k, d in stats.items():
        lines.append(f"{k:22s}" + "".join(f"{d[s]:>13d}" for s in ravens.SET_ORDER)
                     + f"{sum(d.values()):>7d}/96")
    out = "\n".join(lines)
    print(out)
    path = os.path.join(HERE, "..", "results", "classical_ceiling.txt")
    with open(path, "w") as fh:
        fh.write(out + "\n")
    print(f"\nwrote {os.path.normpath(path)}")


if __name__ == "__main__":
    main()
