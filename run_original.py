"""Run the original 2017 KBAI agent, unmodified, over all 8 problem sets.

The 2017 harness resolves "Problems/..." relative to the current working
directory and writes its CSVs into it, so we run with the repo root as CWD and
just put the vendored 2017 modules on sys.path.
"""
import os
import shutil
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "01_original_2017"))
os.chdir(ROOT)

import RavensProject  # noqa: E402


def main():
    start = time.time()
    RavensProject.solve()
    RavensProject.grade()
    elapsed = time.time() - start

    os.makedirs("results", exist_ok=True)
    for src, dst in [("AgentAnswers.csv", "results/original_answers.csv"),
                     ("ProblemResults.csv", "results/original_problem_results.csv"),
                     ("SetResults.csv", "results/original_set_results.csv")]:
        shutil.move(src, dst)

    print("\nTotal runtime: %.1f s" % elapsed)
    with open("results/original_runtime.txt", "w") as fh:
        fh.write("%.2f\n" % elapsed)


if __name__ == "__main__":
    main()
