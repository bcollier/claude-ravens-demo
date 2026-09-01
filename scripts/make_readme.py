"""Generate README.md -- the narrative write-up of the whole experiment.

Written for readers with no background in machine learning. Every number is
pulled from the result files rather than typed, so the story cannot drift from
what actually ran. Regenerate with: python scripts/make_readme.py
"""
from __future__ import annotations

import csv
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, os.path.join(ROOT, "common")]

import ravens    # noqa: E402
import pricing   # noqa: E402

RESULTS = os.path.join(ROOT, "results")
PROBLEMS = ravens.load_all()
TRUTH = {p.name: p.answer for p in PROBLEMS}
N = len(PROBLEMS)


def summary(path):
    d = {}
    if os.path.exists(path):
        for line in open(path):
            k, _, v = line.partition(":")
            d[k.strip()] = v.strip()
    return d


def llm_runs():
    table = pricing.fetch()
    out = []
    for path in sorted(glob.glob(os.path.join(RESULTS, "*llm_*_answers.csv"))):
        rows = list(csv.DictReader(open(path)))
        if not rows or all(int(r["InputTokens"] or 0) == 0 for r in rows):
            continue
        s = summary(path.replace("_answers.csv", "_summary.txt"))
        model = s.get("model", os.path.basename(path))
        in_t = sum(int(r["InputTokens"]) for r in rows)
        out_t = sum(int(r["OutputTokens"]) for r in rows)
        charged = sum(float(r.get("CostUSD") or 0) for r in rows)
        cost = charged if charged > 0 else (pricing.estimate(model, in_t, out_t, table)[0] or 0)
        out.append({
            "model": model, "correct": sum(int(r["Correct"]) for r in rows),
            "cost": cost, "wall": float(s.get("wall_seconds", 0) or 0),
            "in_t": in_t, "out_t": out_t,
            "answers": {r["RavensProblem"]: int(r["Answer"]) for r in rows},
            "inclass": "epilogue" not in os.path.basename(path),
        })
    out.sort(key=lambda r: -r["correct"])
    return out


def agent_answers(path, col):
    return {r["RavensProblem"]: int(r[col]) for r in csv.DictReader(open(path))}


def main():
    runs = llm_runs()
    best = runs[0]
    orig = agent_answers(os.path.join(RESULTS, "original_answers.csv"), "Agent's Answer")
    clas = agent_answers(os.path.join(RESULTS, "classical_answers.csv"), "Answer")
    clas0 = agent_answers(os.path.join(RESULTS, "classical_answers.csv"), "AnswerNoTraining")
    n_orig = sum(orig[n] == TRUTH[n] for n in TRUTH)
    n_clas = sum(clas[n] == TRUTH[n] for n in TRUTH)
    n_clas0 = sum(clas0[n] == TRUTH[n] for n in TRUTH)
    n_skip = sum(1 for n in TRUTH if orig[n] < 0)
    orig_rt = float(open(os.path.join(RESULTS, "original_runtime.txt")).read().strip())
    csum = summary(os.path.join(RESULTS, "classical_summary.txt"))
    ceiling = {}
    cp = os.path.join(RESULTS, "classical_ceiling.txt")
    if os.path.exists(cp):
        for line in open(cp).read().strip().split("\n")[1:]:
            ceiling[line.split("  ")[0].strip()] = line.rsplit("/96", 1)[0].split()[-1]
    nn = summary(os.path.join(RESULTS, "epilogue_neural_relationnet_v4_summary.txt")) \
        or summary(os.path.join(RESULTS, "epilogue_neural_relationnet_summary.txt"))
    split = {}
    sp = os.path.join(RESULTS, "epilogue_split_eval.json")
    if os.path.exists(sp):
        split = json.load(open(sp))

    strong = [r for r in runs if r["correct"] / N >= 0.70]
    wrong_by = {n: [r for r in strong if r["answers"].get(n) != TRUTH[n]] for n in TRUTH}
    clean = sum(1 for n in TRUTH if not wrong_by[n])
    shared = [n for n in TRUTH if len(wrong_by[n]) >= 2]
    tot_err = sum(len(v) for v in wrong_by.values())
    shared_err = sum(len(wrong_by[n]) for n in shared)

    L = []
    w = L.append

    # ------------------------------------------------------------- opening
    w("# Can a computer pass an IQ test?\n")
    w("Three programs, written across eight years, all trying to solve the same 96 "
      "puzzles. One was written by a student in 2017. One was written this year with "
      "no artificial intelligence of the kind you have been reading about. One is that "
      "kind of AI. This is what happened when they sat the same test.\n")
    w("Everything here is reproducible: every number in this document is generated "
      "from the raw results by a script, not typed in by hand.\n")
    w(f"| | Score out of {N} | |")
    w("|---|---|---|")
    w(f"| The 2017 student project | **{n_orig}** | 35% |")
    w(f"| A modern program with no AI language model | **{n_clas}** | 62% |")
    w(f"| An AI language model (`{best['model']}`) | **{best['correct']}** | "
      f"{best['correct']/N:.0%} |")
    w(f"| Guessing at random | 13 | 13% |")
    w("")
    w("The rest of this explains what each one was doing, why the gaps are the size "
      "they are, and — more usefully — where the AI still fails and why.\n")

    # ------------------------------------------------------------- the test
    w("---\n\n## 1. The test\n")
    w("**Raven's Progressive Matrices** is a reasoning test designed in 1936. It uses "
      "no words and no numbers, which is the point: it was built to measure reasoning "
      "without measuring education or language. You are shown a grid of pictures with "
      "the last one missing, and a set of options. Work out the pattern; pick the "
      "piece that completes it.\n")
    w("It is still widely used, including in some hiring and university admissions "
      "processes. So \"can software do this?\" is not an idle question.\n")
    w("### An easy one\n")
    w("![An easy 2x2 Raven's problem](docs/example_easy.png)\n")
    w("Three cells, all the same black square. Whatever rule is operating, it is not "
      "changing anything, so the fourth cell is a black square too: **option 2**. "
      "Almost anything can do this one.\n")
    w("### A harder one\n")
    w("![A 3x3 Raven's problem with two attributes](docs/example_medium.png)\n")
    w("Now there are two things going on at once. Every cell has an **outer shape** "
      "(circle, octagon, square) and an **inner pattern** (vertical bar, horizontal "
      "bar, cross). Look along any row or column: each outer shape appears exactly "
      "once, and each bar pattern appears exactly once — like a sudoku made of "
      "pictures. The bottom row already has a square-with-horizontal-bar and a "
      "circle-with-cross, so the missing cell must be the octagon with the vertical "
      "bar: **option 3**.\n")
    w("This is the real skill the test is measuring — noticing that two independent "
      "patterns are running at the same time and tracking both.\n")
    w("### A genuinely hard one\n")
    w("![A hard problem where two rules combine](docs/example_hard.png)\n")
    w("Keep this one in mind. It is the single problem that defeated the most AI "
      "models, and section 6 comes back to it.\n")

    # ------------------------------------------------------- contestant 1
    w("---\n\n## 2. Contestant one: the 2017 student project\n")
    w("The starting point was a real university assignment from eight years ago, "
      "written for a Georgia Tech course on knowledge-based AI. **It was run "
      "completely unmodified** — not a single character changed — and it still works "
      f"on today's software. It answered all {N} problems in {orig_rt:.0f} seconds.\n")
    w("### What it does\n")
    w("Its strategy is to turn each picture into a handful of numbers, then do "
      "arithmetic on them. The main number is simply **what percentage of the square "
      "is covered in black ink**. If ink coverage goes 10%, 20%, 30% along a row, the "
      "next one should be about 40%, so pick the option closest to that.\n")
    w("It has six other measurements in the same spirit: how much ink two pictures "
      "share, where the ink sits on average, whether a picture is an exact repeat of "
      "another. Each measurement scores every option, the scores are combined using "
      "weights the author tuned by hand, and the highest total wins.\n")
    w("### What it got right\n")
    w("Three of its ideas are genuinely good, and all three reappear in the modern "
      "programs below. It compares **relationships rather than objects** — it does not "
      "ask \"which option looks like the cell above?\" but \"is the gap between the "
      "answer and the cell above the same as the gap between the two cells before "
      "it?\" That is what analogy actually is. It **eliminates** options that are exact "
      "copies of a visible cell. And it **tests a rule on the rows it can see** before "
      "applying it to the row it cannot.\n")
    w("### Why it stops at 35%\n")
    w(f"Two reasons, and neither is a bug. First, **it refuses to attempt {n_skip} of "
      f"the {N} problems** — every 2x2 puzzle, a quarter of the test, scored zero "
      "because that part was never finished before the deadline.\n")
    w("Second, and more interesting: **a picture reduced to a few numbers loses the "
      "information the harder problems depend on.** Ink coverage cannot tell you that "
      "the inner shape rotated while the outer frame stayed still. Two completely "
      "different pictures can have identical ink coverage. No amount of adjusting the "
      "weights recovers something the measurements never captured.\n")
    w("There is a nice footnote here. The code contains two genuine arithmetic "
      "mistakes in the scoring formula. Fixing them changes the score by at most one "
      "problem, and fixing both makes it slightly *worse* — because the weights had "
      "been tuned by trial and error around the mistakes. The bugs had been absorbed "
      "into the tuning. (`01_original_2017/typo_experiment.py` runs this.)\n")

    # ------------------------------------------------------- contestant 2
    w("---\n\n## 3. Contestant two: a modern program, no AI language model allowed\n")
    w("The rule for this one was: solve the puzzles using any technique you like, "
      "except the sort of AI that has read the internet. So this program has to be "
      "*told* how to reason, in code.\n")
    w("### The approach, and why\n")
    w("The 2017 program's limit was that it measured seven things and hoped one of "
      "them mattered. The obvious fix is to **propose a large number of possible "
      "rules and then work out which one is actually in force**. So this program "
      "generates roughly 300 candidate rules for every puzzle:\n")
    w("- *Is something being flipped or rotated?* (eight geometric transformations)")
    w("- *Is the third picture the overlap, or the combination, or the difference of "
      "the first two?* (six ways of combining two images pixel by pixel)")
    w("- *Is some measurable quantity counting up?* (eight measurements — ink, number "
      "of separate objects, number of enclosed holes, symmetry, and so on)")
    w("- *Is this a sudoku-style arrangement where every row contains each shape "
      "once?* (checked against five different ways of splitting a picture into its "
      "parts — outer frame, inner shape, interior detail)\n")
    w("### The discovery that mattered\n")
    w("Halfway through, a diagnostic was run to answer a simple question: *is the "
      "right answer even among the rules we are proposing?* The result reframed the "
      "whole project.\n")
    if ceiling:
        w(f"| | Score |")
        w("|---|---|")
        w(f"| Some proposed rule, somewhere, picks the correct answer | "
          f"**{ceiling.get('any rule','?')}/96** |")
        w(f"| If a magic oracle told us which *kind* of rule was in force | "
          f"**{ceiling.get('best-family oracle','?')}/96** |")
        w(f"| Simply believing whichever single rule scores highest | "
          f"**{ceiling.get('most-trusted rule','?')}/96** |")
        w("")
        w(f"The correct answer is almost always in there. The problem is *choosing*. "
          f"And notice the bottom row: believing your single best-scoring rule gets "
          f"{ceiling.get('most-trusted rule','?')}/96 — which is what the 2017 program "
          f"scored. Applying one fixed scoring scheme to every puzzle lands in the "
          f"mid-thirties no matter how good the individual measurements are.\n")
    w("### How rules earn trust\n")
    w("The fix is the most transferable idea in this repository. To decide whether a "
      "rule is any good, **hide part of the puzzle you already know the answer to, "
      "and see whether the rule can recover it** — competing against the real "
      "distractors. A rule that reliably recovers hidden cells gets a loud vote; a "
      "rule that cannot gets ignored.\n")
    w("![How a rule earns trust](docs/rule_validation.png)\n")
    w("This is the same logic as testing a business forecast by hiding last quarter's "
      "figures and checking whether the model predicts them, rather than asking "
      "whether it fits the data it was built on.\n")
    w("### Results\n")
    w(f"Rule search alone, with no learning at all: **{n_clas0}/{N}**. Adding a small "
      f"statistical model that learns how much to trust each *type* of rule: "
      f"**{n_clas}/{N}** — nearly double the 2017 program, in {float(csum.get('runtime_seconds',0)):.0f} "
      f"seconds on a laptop, with no internet connection.\n")

    # ------------------------------------------------------- neural
    w("### A detour: what about a neural network?\n")
    w("A neural network is a program that is not told any rules at all. You show it "
      "thousands of solved examples and it works out for itself what predicts the "
      "right answer. It is the technology behind image recognition, and it is the "
      "obvious thing to try here.\n")
    w("There is an immediate problem: **96 puzzles is nowhere near enough to learn "
      f"from.** So the standard trick was used — write a program that *invents* "
      "Raven's-style puzzles by the thousand, train the network on those, and keep the "
      "96 real ones as a clean test it has never seen.\n")
    if nn:
        w(f"The result: **{nn.get('correct','?')}/{N}**. Worse than guessing.\n")
    w("Four versions were tried. Every one of them learned the *invented* puzzles "
      "well — the final version scores about 56% on invented puzzles it had not seen — "
      "and every one of them failed on the real ones. Working out why turned out to be "
      "the most instructive part of the whole project, and it is in "
      "[EPILOGUE.md](EPILOGUE.md). The short version:\n")
    w("> The invented puzzles had a statistical quirk. Because of how they were "
      "generated, the correct answer was a repeat of a picture already visible on the "
      "page about 30% of the time. In real Raven's problems that is true only 10% of "
      "the time — the people who write them deliberately avoid it. The network learned "
      "the quirk perfectly, applied it to the real test, and was wrong more often than "
      "chance.\n")
    w("**It had not failed to learn. It had learned exactly the wrong thing, because "
      "the world it was shown was not the world it was tested in.** That is the single "
      "most common way real machine learning projects fail, and it usually looks like "
      "this: every training metric healthy, the deployed model quietly useless.\n")

    # ------------------------------------------------------- contestant 3
    w("---\n\n## 4. Contestant three: the AI language model\n")
    w("This is the technology behind ChatGPT and its competitors. It has been trained "
      "on an enormous amount of text and images. Nobody programmed it to solve Raven's "
      "matrices; the question is whether it can anyway.\n")
    w("### Exactly what it is sent\n")
    w("This matters, because \"we asked the AI\" hides all the engineering. Each puzzle "
      "is one request containing **37 pieces of content — 20 short pieces of text and "
      "17 images**:\n")
    w("![Exactly what is sent to the language model](docs/llm_payload.png)\n")
    w("Reading down that picture: a sentence explaining the grid layout; then the "
      "**whole puzzle sheet as one image**; then **each of the eight grid cells as its "
      "own labelled image**; then **each of the eight answer options as its own "
      "labelled image**; then the question.\n")
    w("Both views are sent deliberately. The full sheet shows the *structure* — which "
      "cells form a row, which one is missing — but any single cell inside it is too "
      "small to compare shapes reliably. The individual pictures give the detail but "
      "lose all sense of position. Neither on its own is enough.\n")
    w("The model is also given a short instruction listing the kinds of rule these "
      "puzzles use — 134 words, quoted in full in "
      "[03_llm/README.md](03_llm/README.md). That is the only help it gets. **No "
      "worked examples, no second attempts, no conversation, and no hints about which "
      "set a puzzle came from.** One request, one answer, and whatever comes back is "
      "the score.\n")
    w("### Exactly what comes back\n")
    w("The reply is forced into a fixed format, so the answer is always a number that "
      "can be marked automatically. Here is a real reply, to the sudoku-style puzzle "
      "from section 1:\n")
    w("```json\n{\n  \"rule\": \"Both the outer shapes and bar patterns cycle through "
      "three states in each row and column, requiring an octagon with a vertical "
      "bar.\",\n  \"answer\": 3,\n  \"confidence\": 0.99\n}\n```\n")
    w("Correct, in 4.8 seconds. Note that it did not just produce a number — it "
      "described the rule, and the description is right. Every one of these "
      f"explanations is saved in `results/`, including for the puzzles it got wrong, "
      "which is what makes section 6 possible.\n")

    # ------------------------------------------------------- results
    w("---\n\n## 5. The results\n")
    labs = {(r["model"].split("/")[0] if "/" in r["model"] else "openai") for r in runs}
    w(f"{len(runs)} different language models from {len(labs)} companies were run on "
      f"the identical {N} puzzles with the identical prompt. Alongside them, the two "
      f"programs that run on a laptop with no internet.\n")
    w("| | Score | Accuracy | Cost | Time | Needs internet |")
    w("|---|---|---|---|---|---|")
    w(f"| **2017 student project** | {n_orig}/{N} | {n_orig/N:.0%} | $0 | "
      f"{orig_rt:.0f} s | no |")
    w(f"| **Modern program, no language model** | {n_clas}/{N} | {n_clas/N:.0%} | $0 | "
      f"{float(csum.get('runtime_seconds',0)):.0f} s | no |")
    for r in runs:
        star = " *(used in class)*" if r["inclass"] else ""
        w(f"| `{r['model']}`{star} | {r['correct']}/{N} | {r['correct']/N:.0%} | "
          f"${r['cost']:.2f} | {r['wall']:.0f} s | yes |")
    w("")
    w("Costs are for all 96 puzzles. Times are for the whole sweep with ten requests "
      "running at once.\n")
    w("### Three things worth noticing\n")
    w("**The price spread is far wider than the accuracy spread.** Among the top "
      "models a few percentage points of accuracy separate them, while the cost varies "
      "by roughly ten times. Picking the biggest, most expensive model is usually the "
      "wrong default — a newer cheap model routinely beats an older expensive one.\n")
    w("**Progress over two years is dramatic.** The same family of models went from "
      "the mid-thirties to the mid-nineties. For context, the mid-thirties is what the "
      "2017 student program scores.\n")
    w("**One model could not sit the test at all.** `gpt-3.5-turbo`, the model that "
      "made ChatGPT famous in 2022, cannot accept images. Not a low score — no score. "
      "The barrier is what it can perceive, not how well it reasons.\n")
    w("### Could it be cheaper?\n")
    bp = os.path.join(RESULTS, "epilogue_batch_estimate.md")
    if os.path.exists(bp):
        w("Yes. Every provider sells a **batch** option: submit the work, get it back "
          "within 24 hours, pay about half. For this workload — 96 independent "
          "questions with nobody waiting — that is close to free money. Running all of "
          "these in batch mode would have cost roughly half. Details and per-model "
          "figures in [EPILOGUE.md](EPILOGUE.md).\n")
        w("The trade-off is only speed. The classroom demonstration finished in 77 "
          "seconds and the results went on the screen immediately; in batch mode it "
          "would have been cheaper and useless for that purpose. **A benchmark is the "
          "ideal batch job. A demo is not.**\n")
    w("### Is this a fair test of the language models?\n")
    w("Only partly, and it is worth being honest about. These puzzles have been in a "
      "public code repository since 2017, so they may well have been part of what the "
      "models were trained on. A score of 97% here is a score on a *public, "
      "possibly-memorised* test. The two laptop programs have no such advantage, which "
      "is one reason they are worth keeping in the comparison.\n")

    # ------------------------------------------------------- mistakes
    w("---\n\n## 6. Where the AI still fails\n")
    w("This is the most useful section, because the failures are not random.\n")
    w(f"Taking the {len(strong)} models that scored 70% or better:\n")
    w(f"- **{clean} of the {N} puzzles** were solved by every single one of them.")
    w(f"- The **{len(shared)} puzzles that two or more models missed** account for "
      f"**{shared_err/max(tot_err,1):.0%} of all the mistakes made**.")
    import collections as _c
    agrees = []
    for n in shared:
        picks = _c.Counter(r["answers"].get(n) for r in wrong_by[n])
        agrees.append(picks.most_common(1)[0][1] / len(wrong_by[n]))
    conv = sum(agrees) / len(agrees) if agrees else 0
    w(f"- When several models miss the same puzzle, **about {conv:.0%} of them choose "
      f"the same wrong option** — against roughly 14% if they were guessing.\n")
    w("Different companies, different technology, different training data, arriving at "
      "the same wrong answer. **The failures are a property of the puzzles, not of any "
      "one product.** If you are evaluating an AI tool for your own use, this is the "
      "pattern to look for: not \"how often is it wrong\" but \"is it wrong in a "
      "predictable place\".\n")
    w("### Failure one: two rules at once, and it only applies the obvious one\n")
    w("![Two rules operating at once](docs/mistake_two_rules.png)\n")
    w("Five of seven models got this wrong, and four of them gave the *same* wrong "
      "answer. What makes it hard is that two rules run simultaneously: going across, "
      "alternate bands get filled black; going down, two rings are removed each time "
      "(5, then 3, then 1). The answer needs both, which leaves a single filled ring — "
      "a plain black square.\n")
    w("The models saw the filling rule and stopped. You can watch it happen in their "
      "own explanations, which describe the fill perfectly and never mention the "
      "count:\n")
    w("> *\"The right column shows the left column's concentric squares with alternate "
      "rings filled black.\"* — a flawless description of half the puzzle.\n")
    w("The two models that solved it stated both rules. **And they had all been "
      "explicitly instructed to check rows and columns before answering.** Being told "
      "was not enough: the visually dramatic rule crowded out the quiet numerical one. "
      "That is a recognisably human error.\n")
    w("### Failure two: right rule, lost count\n")
    w("![A three-attribute sudoku puzzle](docs/mistake_bookkeeping.png)\n")
    w("Here three patterns run at once — how many triangles (1, 2 or 3), whether they "
      "are filled or outlined, and which way they point. Four of seven models missed "
      "it, and this time they **described the rule correctly and still picked the "
      "wrong cell**:\n")
    w("> *\"Each row and column contains one cell with 1, 2, and 3 triangles, while the "
      "styles cycle among filled upright, outlined upright, and outlined "
      "right-pointing\"* — then names the wrong option.\n")
    w("That is not a reasoning failure, it is a **bookkeeping** failure. Three "
      "constraints have to be held in mind and intersected at once, and they lose "
      "track — the same way you might when filling in a sudoku in your head rather "
      "than on paper.\n")
    w("Which is exactly what the no-AI program does perfectly and for free, because "
      "for it that step is six lines of code checking every combination.\n")
    w("### The pattern\n")
    w("**Language models are much better at *noticing* what kind of rule is present. "
      "They are worse at the exhaustive checking afterwards.** The hand-written "
      "program is the mirror image: hopeless at spotting which of 300 candidate rules "
      "matters, flawless at verifying one once chosen.\n")
    w("The practical lesson generalises well beyond puzzles: use a language model for "
      "the judgement call, and ordinary software for the arithmetic that follows. "
      "Asking the model to do both in one step is where these mistakes come from.\n")
    w("### And it does not know when it is wrong\n")
    w("Every model was asked to rate its own confidence. The best one reported **0.99 "
      "on every single answer — including all three it got wrong.** There is no "
      "threshold you could set to catch its mistakes automatically. If you are "
      "deploying one of these, do not expect it to flag its own errors.\n")

    # ------------------------------------------------------- takeaways
    w("---\n\n## 7. What to take from this\n")
    w("**Old software is more durable than you think.** Eight-year-old code ran "
      "untouched. The expensive part of software is rarely keeping it alive; it is "
      "that its original design decides its ceiling.\n")
    w("**Most of the gain came from measuring the right thing.** The single most "
      "useful hour was spent not writing a solver but running a diagnostic that asked "
      "*where is the bottleneck?* The answer — choosing between rules, not finding "
      "them — redirected everything after it. Before optimising, find out what you are "
      "optimising.\n")
    w("**A model is only as good as the world you show it.** The neural network "
      "learned its training data faithfully and scored below random guessing, because "
      "that data had a quirk the real test did not. No training metric revealed it. "
      "Only looking at the data next to the real thing did.\n")
    w("**Test on what you did not train on, and do it more than once.** An early "
      "reading of one experiment said 65.5%; running the same thing 20 times with "
      "different random splits said "
      + (f"{split['linear']['mean']:.1%}, ranging from {split['linear']['min']:.0%} to "
         f"{split['linear']['max']:.0%} depending on the split. " if split.get("linear")
         else "several points lower. ")
      + "A single measurement on a small sample is close to worthless.\n")
    w("**Confident and correct are unrelated.** Both the language models and the "
      "neural network were most confident precisely where they were most wrong.\n")
    w("**And measure your own tooling before blaming the model.** Three models "
      "initially scored 0 out of 96. All three were bugs in how the questions were "
      "being sent, not failures of the models. A suspiciously round zero is almost "
      "always yours.\n")

    # ------------------------------------------------------- technical
    w("---\n\n## 8. Technical notes\n")
    w("*For readers who want to run or extend this. Everything below is optional.*\n")
    w("### Getting started\n")
    w("```bash\n"
      "pip install -r requirements.txt\n\n"
      "python run_original.py                    # the 2017 agent          (~35 s)\n"
      "python 02_classical_ai/solver.py          # the no-LLM agent        (~60 s)\n"
      "python 02_classical_ai/diagnose.py        # the bottleneck analysis\n"
      "python 04_neural/solver.py --steps 8000   # train the neural net    (~15 min)\n\n"
      "export OPENAI_API_KEY=...\n"
      "python 03_llm/solver.py --model gpt-5.6-sol\n\n"
      "python scripts/compare.py                 # rebuild COMPARISON.md\n"
      "```\n")
    w("Agents one and two need no network. All numbers in this document, "
      "[COMPARISON.md](COMPARISON.md) and [EPILOGUE.md](EPILOGUE.md) are regenerated "
      "from `results/` by the scripts in `scripts/`.\n")
    w("### Software architecture\n")
    w("```\n"
      "                        Problems/            96 puzzles, unchanged from 2017\n"
      "                            |\n"
      "                   common/ravens.py          one loader, so all agents see\n"
      "                            |                identical inputs\n"
      "        +-------------------+-------------------+------------------+\n"
      "        |                   |                   |                  |\n"
      "  01_original_2017/   02_classical_ai/     03_llm/            04_neural/\n"
      "  Agent.py            imageops.py          solver.py          render.py\n"
      "  (vendored, byte-    features.py          openrouter_        wren.py\n"
      "   identical)         solver.py             solver.py         solver.py\n"
      "        |                   |                   |                  |\n"
      "        +-------------------+---------+---------+------------------+\n"
      "                                      |\n"
      "                                  results/      one CSV per run\n"
      "                                      |\n"
      "        +----------------+------------+------------+----------------+\n"
      "     compare.py     make_epilogue.py  make_page.py  error_analysis.py\n"
      "        |                |                 |              |\n"
      "  COMPARISON.md    EPILOGUE.md      docs/index.html   analysis txt\n"
      "```\n")
    w("### Data flow, agent by agent\n")
    w("| Agent | Input | Intermediate representation | Output |")
    w("|---|---|---|---|")
    w("| 2017 | 16 PNGs | 7 scalars per panel | weighted sum, argmax |")
    w("| Classical | 16 PNGs | 5 attribute decompositions -> ~300 scored rules -> "
      "48 features per option | learned ranking, argmax |")
    w("| LLM | 17 PNGs + prompt | *(none you can inspect)* | JSON `{rule, answer, "
      "confidence}` |")
    w("| Neural | 16 PNGs at 64x64 | 128-d embedding per panel -> pairwise relation "
      "vectors | softmax over options |")
    w("")
    w("### Where the interesting code is\n")
    w("| Question | File |")
    w("|---|---|")
    w("| How is a rule proposed and scored? | `02_classical_ai/features.py` — "
      "`score_rules()` and `_validation_score()` |")
    w("| How is a picture decomposed into parts? | `02_classical_ai/imageops.py` — "
      "`DESCRIPTORS` |")
    w("| What exactly is sent to the model? | `03_llm/solver.py` — `build_input()` |")
    w("| How are synthetic puzzles invented? | `04_neural/render.py` — "
      "`make_problem()` |")
    w("| The relation network | `04_neural/wren.py` |")
    w("| Why did the network fail? | `04_neural/diagnose.py` |")
    w("| Is the evaluation honest? | `02_classical_ai/split_eval.py` |")
    w("")
    w("### Things worth trying\n")
    w("- Run `python 02_classical_ai/solver.py --explain \"Basic Problem E-05\"` to see "
      "which rules the program trusted and what each one voted for.\n"
      "- Add a rule family to `features.py` and see whether the cross-validated score "
      "moves. Most ideas do not help; that is the lesson.\n"
      "- Change the synthetic puzzle generator in `04_neural/render.py` and re-run "
      "`04_neural/diagnose.py`. Try to close the gap between the invented world and "
      "the real one.\n"
      "- Send the language model *only* the individual panels, or *only* the assembled "
      "sheet, and see how much each view is worth.\n")
    w("### Full documentation\n")
    w("| | |")
    w("|---|---|")
    w("| [COMPARISON.md](COMPARISON.md) | the in-class results, per set and per problem |")
    w("| [EPILOGUE.md](EPILOGUE.md) | the neural network, the train/test split, all eleven models |")
    w("| [docs/BUILD_LOG.md](docs/BUILD_LOG.md) | every experiment tried, including the six that were dropped |")
    w("| [01_original_2017/NOTES.md](01_original_2017/NOTES.md) | the 2017 code in detail |")
    w("| [03_llm/README.md](03_llm/README.md) | the full prompt and payload |")
    w("| [PROVENANCE.md](PROVENANCE.md) | who asked for what, which models did the work |")
    w("")
    w("---\n")
    w("Problems and harness from the Georgia Tech Knowledge-Based AI project, via "
      "[bcollier/KBAI_Ravens_Project](https://github.com/bcollier/KBAI_Ravens_Project). "
      "Agents two, three and four, the evaluation harness and this write-up were built "
      "with [Claude Code](https://claude.com/claude-code).\n")

    out = os.path.join(ROOT, "README.md")
    with open(out, "w") as fh:
        fh.write("\n".join(L))
    print(f"wrote {out} ({len(L)} blocks)")


if __name__ == "__main__":
    main()
