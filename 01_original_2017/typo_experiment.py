"""What do the two scoring typos in the 2017 agent actually cost?

Agent.py's weighted-score expression has two slips:

  * `IPR_COL_WT * solution_scores[i, IPR_COL]` appears twice, so the
    intersection-pixel-ratio ROW score -- which the agent computes -- is never
    used, and the COLUMN score is counted double.
  * `IPR_DIAG_WT + solution_scores[i, IPR_DIAG]` uses `+` where every other
    term uses `*`, so the diagonal IPR score enters with weight 1.0 instead of
    the intended 0.15 (about 7x its intended influence).

This script leaves Agent.py untouched. It writes patched copies to a temp
directory, runs each against all 96 problems, and reports the scores, so the
headline number for the original stays the number the original actually
produces.

Usage:  python typo_experiment.py
"""
from __future__ import annotations

import contextlib
import io
import os
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

DUPLICATE = ("IPR_COL_WT * solution_scores[i, IPR_COL] +\n"
             "                                              "
             "IPR_COL_WT * solution_scores[i, IPR_COL] +")
DUPLICATE_FIXED = ("IPR_ROW_WT * solution_scores[i, IPR_ROW] +\n"
                   "                                              "
                   "IPR_COL_WT * solution_scores[i, IPR_COL] +")
PLUS = "IPR_DIAG_WT + solution_scores[i, IPR_DIAG]"
PLUS_FIXED = "IPR_DIAG_WT * solution_scores[i, IPR_DIAG]"

VARIANTS = {
    "as shipped (2017)": [],
    "fix the + typo only": [(PLUS, PLUS_FIXED)],
    "fix the duplicated IPR_COL only": [(DUPLICATE, DUPLICATE_FIXED)],
    "fix both": [(PLUS, PLUS_FIXED), (DUPLICATE, DUPLICATE_FIXED)],
}


def run_variant(patches):
    src = open(os.path.join(HERE, "Agent.py")).read()
    for old, new in patches:
        assert old in src, f"patch target not found:\n{old}"
        src = src.replace(old, new, 1)

    work = tempfile.mkdtemp()
    for f in ("ProblemSet.py", "RavensFigure.py", "RavensGrader.py",
              "RavensObject.py", "RavensProblem.py", "RavensProject.py"):
        shutil.copy(os.path.join(HERE, f), work)
    with open(os.path.join(work, "Agent.py"), "w") as fh:
        fh.write(src)
    os.symlink(os.path.join(ROOT, "Problems"), os.path.join(work, "Problems"))

    cwd, path = os.getcwd(), list(sys.path)
    for mod in ("Agent", "ProblemSet", "RavensProject", "RavensGrader",
                "RavensFigure", "RavensObject", "RavensProblem"):
        sys.modules.pop(mod, None)
    try:
        os.chdir(work)
        sys.path.insert(0, work)
        import RavensProject
        with contextlib.redirect_stdout(io.StringIO()):
            RavensProject.solve()
            RavensProject.grade()
        correct = 0
        with open(os.path.join(work, "SetResults.csv")) as fh:
            next(fh)
            for line in fh:
                correct += int(line.rstrip().split(",")[-3])
        return correct
    finally:
        os.chdir(cwd)
        sys.path[:] = path
        for mod in ("Agent", "ProblemSet", "RavensProject", "RavensGrader",
                    "RavensFigure", "RavensObject", "RavensProblem"):
            sys.modules.pop(mod, None)
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    print(f"{'variant':34s} {'correct':>8s}  {'of 96':>7s}")
    for label, patches in VARIANTS.items():
        n = run_variant(patches)
        print(f"{label:34s} {n:>8d}  {n/96:>6.1%}")
