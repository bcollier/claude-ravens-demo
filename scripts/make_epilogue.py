"""Build EPILOGUE.md -- the work added after class.

Kept entirely separate from compare.py so the in-class report is never touched.
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
SET_OF = {p.name: p.set_name for p in PROBLEMS}
N = len(PROBLEMS)

LAB = {"anthropic": "Anthropic", "openai": "OpenAI", "google": "Google",
       "meta-llama": "Meta", "deepseek": "DeepSeek", "qwen": "Alibaba",
       "x-ai": "xAI", "moonshotai": "Moonshot", "z-ai": "Zhipu",
       "mistralai": "Mistral", "amazon": "Amazon"}


def read_summary(path):
    out = {}
    if os.path.exists(path):
        for line in open(path):
            k, _, v = line.partition(":")
            out[k.strip()] = v.strip()
    return out


def load_llm_runs():
    runs = []
    for csv_path in sorted(glob.glob(os.path.join(RESULTS, "epilogue_llm_*_answers.csv"))):
        rows = list(csv.DictReader(open(csv_path)))
        s = read_summary(csv_path.replace("_answers.csv", "_summary.txt"))
        model = s.get("model") or os.path.basename(csv_path)[13:-12]
        in_tok = sum(int(r["InputTokens"]) for r in rows)
        out_tok = sum(int(r["OutputTokens"]) for r in rows)
        charged = sum(float(r.get("CostUSD") or 0) for r in rows)
        if charged > 0:
            cost, src = charged, "charged"
        else:
            cost, src = pricing.estimate(model, in_tok, out_tok)
            src = "list price"
        runs.append({
            "model": model,
            "lab": LAB.get(model.split("/")[0], "OpenAI" if "/" not in model else model.split("/")[0]),
            "correct": sum(int(r["Correct"]) for r in rows),
            "answers": {r["RavensProblem"]: int(r["Answer"]) for r in rows},
            "in_tok": in_tok, "out_tok": out_tok,
            "reasoning": sum(int(r["ReasoningTokens"]) for r in rows),
            "cost": cost or 0.0, "cost_src": src,
            "wall": float(s.get("wall_seconds", 0) or 0),
            "unparseable": sum(1 for r in rows if r["Error"]),
        })
    runs.sort(key=lambda r: -r["correct"])
    return runs


def load_inclass():
    runs = []
    table = pricing.fetch()
    for path in sorted(glob.glob(os.path.join(RESULTS, "llm_*_answers.csv"))):
        rows = list(csv.DictReader(open(path)))
        s = read_summary(path.replace("_answers.csv", "_summary.txt"))
        model = s.get("model", "?")
        in_tok = sum(int(r["InputTokens"]) for r in rows)
        out_tok = sum(int(r["OutputTokens"]) for r in rows)
        cost, _ = pricing.estimate(model, in_tok, out_tok, table)
        runs.append({"model": model, "lab": "OpenAI",
                     "correct": sum(int(r["Correct"]) for r in rows),
                     "answers": {r["RavensProblem"]: int(r["Answer"]) for r in rows},
                     "in_tok": in_tok, "out_tok": out_tok,
                     "reasoning": sum(int(r["ReasoningTokens"]) for r in rows),
                     "cost": cost or 0.0, "cost_src": "list price",
                     "wall": float(s.get("wall_seconds", 0) or 0),
                     "unparseable": 0, "inclass": True})
    return runs


def per_set(answers):
    return [sum(1 for x in TRUTH if SET_OF[x] == s and answers.get(x) == TRUTH[x])
            for s in ravens.SET_ORDER]


def money(x):
    return f"${x:,.2f}" if x >= 0.01 else f"${x:.4f}"


def main():
    L = []
    w = L.append
    stats = json.load(open(os.path.join(RESULTS, "session_stats.json")))
    epi = stats.get("epilogue", {})
    split = {}
    sp = os.path.join(RESULTS, "epilogue_split_eval.json")
    if os.path.exists(sp):
        split = json.load(open(sp))
    inclass_cv = None
    cs = os.path.join(RESULTS, "classical_summary.txt")
    if os.path.exists(cs):
        for line in open(cs):
            if line.startswith("rule_search_plus_ranker"):
                inclass_cv = float(line.split(":")[1])

    w("# Epilogue: comparisons added after class\n")
    w("Everything in [COMPARISON.md](COMPARISON.md) is exactly as it ran during the "
      "session and has not been touched. This file adds three things that came up "
      "afterwards.\n")
    w("1. **A neural network.** The in-class \"no LLM\" agent used classical computer "
      "vision and a linear ranker. It should have had a neural network in the mix.")
    w("2. **A real train/test split.** The in-class headline was cross-validated, but "
      "not in the conventional 70/30 form, and the feature *design* saw all 96 problems.")
    w("3. **Other labs' models.** The in-class LLM runs were all OpenAI.\n")

    # ---------------------------------------------------------- part 1
    w("---\n\n## Part 1 — A neural network\n")
    rn = read_summary(os.path.join(RESULTS, "epilogue_neural_relationnet_summary.txt"))
    rn_v1 = read_summary(os.path.join(RESULTS, "epilogue_neural_v1_all_attributes_summary.txt"))
    w("### The problem with 96 problems\n")
    w("You cannot train a network on 96 examples. The standard way round it, and what "
      "the research literature on this exact task does, is to **generate an unlimited "
      "supply of synthetic matrices**, train on those, and treat the 96 real problems "
      "as a held-out test the network has never seen. `04_neural/render.py` is that "
      "generator: it builds matrices from the same rule vocabulary the real sets use "
      "(constant, progression, Latin square, arithmetic on counts, and pixel "
      "XOR/OR/AND) and renders them as black line art the way the originals look.\n")
    w("### The model\n")
    w("`04_neural/wren.py` is a relation network in the style of Santoro et al.'s WReN. "
      "Each panel goes through a small CNN to a 128-dimensional embedding tagged with "
      "its grid position; every *pair* of panels goes through a shared MLP; the pair "
      "representations are summed and a head scores the candidate. The pairwise sum is "
      "the whole point — a rule in a Raven's matrix is a statement about how two cells "
      "relate, so the architecture is built to compute exactly that.\n")
    if rn:
        w(f"About {int(rn.get('model','0 params').split(',')[-1].split()[0] or 0):,} "
          f"parameters, trained on {rn.get('device','?')} in "
          f"{float(rn.get('train_seconds',0)):.0f} seconds.\n")

    w("### Results\n")
    w("Three versions of the *same network and the same training budget*. Only the "
      "generator changed.\n")
    w("| | Training data | Held-out synthetic | **Real 96** |")
    w("|---|---|---|---|")
    variants = [
        ("v1", "epilogue_neural_v1_all_attributes_summary.txt",
         "every attribute given its own rule at once"),
        ("v2", "epilogue_neural_v2_simple_panels_summary.txt",
         "one or two active attributes, single repeated shapes"),
        ("v3", "epilogue_neural_relationnet_summary.txt",
         "as v2, plus composed panels: nested frames, inner shapes, bars"),
        ("v4", "epilogue_neural_relationnet_v4_summary.txt",
         "two generator bugs fixed: 18.6% of problems were unanswerable, and "
         "shapes were drawn far too small"),
    ]
    last = [t for t, fn, _ in variants
            if read_summary(os.path.join(RESULTS, fn))][-1:]
    for tag, fn, desc in variants:
        d = read_summary(os.path.join(RESULTS, fn))
        if not d:
            continue
        bold = "**" if [tag] == last else ""
        w(f"| {tag} | {desc} | {float(d.get('synthetic_val',0)):.0%} | "
          f"{bold}{d.get('correct','?')}/96 ({float(d.get('accuracy',0)):.1%}){bold} |")
    w("")
    w("And the two rankers that use the symbolic features, on the 70/30 protocol from "
      "Part 2 so they are directly comparable:\n")
    w("| Ranker | Features | 70/30 test accuracy |")
    w("|---|---|---|")
    rows = [("linear (in class)", "48 symbolic rule features", "linear"),
            ("**MLP**", "the same 48 features, a small neural network", "mlp"),
            ("**hybrid**", "those 48 plus the relation network's score per option", "hybrid")]
    for label, feats, kind in rows:
        if kind in split:
            d = split[kind]
            w(f"| {label} | {feats} | {d['mean']:.1%} ± {d['std']:.1%} |")
    w("")
    w("### What this shows\n")
    w("**The training distribution moved the numbers; the architecture never did.** "
      "Every row above is the same network with the same budget. v1's generator gave "
      "every attribute its own rule at once, producing matrices that are visually "
      "chaotic in a way real Raven's problems never are. v2 fixed that and learned the "
      "synthetic task far better. v3 added the composed panels — nested frames, inner "
      "shapes, bars — that sets D and E are actually built from.\n")
    w("**Learning the synthetic task better did not mean solving the real one better.** "
      "That is the sharpest lesson here. v2 roughly doubled held-out synthetic accuracy "
      "over v1 and did *worse* on the real problems. A network can only learn the world "
      "you show it, and the gap between that world and the real one does not appear "
      "anywhere in the training metrics.\n")
    w("**Two of the biggest problems were in the data, and only looking at it found "
      "them.** Rendering the training panels next to the real ones at the size the "
      "network actually sees revealed that synthetic shapes were drawn far too small, "
      "and — worse — that the generator chose which attributes a rule acts on *before* "
      "deciding whether the panel could express them, silently erasing the variation in "
      "**18.6% of training problems** and leaving eight identical panels with an "
      "unanswerable question. The loss curve looked healthy throughout. No metric "
      "reported it.\n")
    diag = os.path.join(RESULTS, "epilogue_neural_relationnet_diagnostics.txt")
    if os.path.exists(diag):
        w("### Two diagnostics that explain the failure\n")
        w("```\n" + open(diag).read().strip() + "\n```\n")
        w("**It does read the matrix.** Erase every context panel and the network drops "
          "to chance on synthetic problems, so it is not exploiting a giveaway in how "
          "the distractors were built — a common artefact in this literature.\n")
        w("**But it learned the wrong rule of thumb, and the data bug is why.** Those "
          "18.6% of broken problems had eight identical context panels, so their answer "
          "*was* a copy of the context — trivially solvable, and quietly inflating the "
          "synthetic score. The network duly learned to favour the option that "
          "duplicates something already on the page. On the real 96 it does that 29% of "
          "the time, and that option is correct only 10% of the time — worse than the "
          "13% you would get by guessing. A heuristic that is anti-correlated with the "
          "truth is how a model ends up *below* chance.\n")
        w("The symbolic agent learned the opposite sign from real data: its largest "
          "negative weight is on exactly this feature, `dup_max`. Same signal, fitted "
          "on the real distribution instead of a broken synthetic one, and it points "
          "the other way.\n")
    w("**A network trained on synthetic data does not reach the symbolic agent.** That "
      "is the honest result and it is worth sitting with: the relation network has to "
      "*discover* concepts like \"the outer frame is unchanged\" from pixels, with only "
      "the rules I thought to put in the generator to learn from. The symbolic agent "
      "was handed those concepts. When you have 96 problems and strong priors about the "
      "domain, encoding the priors beats learning them.\n")
    w("**The neural ranker did not beat the linear one either.** With 67 training "
      "problems and 48 features, there is not enough signal for an MLP to find "
      "structure a linear model misses. This is a useful negative result: \"use a "
      "neural network\" is not free, and the honest comparison shows when it does not "
      "pay.\n")

    # ---------------------------------------------------------- part 2
    w("---\n\n## Part 2 — A proper train/test split\n")
    w("### Was the in-class number trained on its test data?\n")
    w("**The model weights were not.** The in-class headline was *nested "
      "leave-one-problem-set-out*: train on seven problem sets, test on the eighth, "
      "with the hyper-parameters chosen by an inner cross-validation inside the "
      "training folds only. Every one of the 96 predictions came from a model that had "
      "not seen that problem.\n")
    w("**The feature design was.** This is the real leak and no re-split fixes it. "
      "Partway through, three failing set-D problems were rendered and inspected; "
      "attribute descriptors and the Latin-square rule family were added *because of "
      "what those images showed*. The rule vocabulary in `features.py` was shaped by "
      "looking at the test set. Model fitting is clean; feature engineering is not.\n")
    if split:
        w("### The 70/30 numbers\n")
        d0 = next(iter(split.values()))
        w(f"Stratified by problem set, {d0['train_n']} train / {d0['test_n']} test, "
          f"hyper-parameters chosen on the training half only, repeated over "
          f"{d0['seeds']} random seeds.\n")
        w("| Ranker | 70/30 test accuracy | Spread across seeds | In-class "
          "leave-one-set-out |")
        w("|---|---|---|---|")
        ref = {"linear": "61.5%", "mlp": "—", "hybrid": "—"}
        for kind in ("linear", "mlp", "hybrid"):
            if kind not in split:
                continue
            d = split[kind]
            w(f"| {kind} | **{d['mean']:.1%}** | {d['min']:.0%} – {d['max']:.0%} "
              f"(± {d['std']:.1%}) | {ref.get(kind,'—')} |")
        w("")
        lin = split.get("linear", {})
        if lin and inclass_cv is not None:
            delta = lin["mean"] - inclass_cv
            w(f"**The two protocols agree.** Leave-one-problem-set-out gave "
              f"{inclass_cv:.1%}; a random 70/30 gives {lin['mean']:.1%}, a difference of "
              f"{abs(delta)*100:.1f} points — well inside the noise. The in-class number "
              f"was not flattered by its protocol.\n"
              if abs(delta) < 0.03 else
              f"**The random split scores {'higher' if delta > 0 else 'lower'} than "
              f"leave-one-set-out** ({lin['mean']:.1%} against {inclass_cv:.1%}). A random "
              f"split puts problems from every set in the training half, so at test time "
              f"the model has already met the family of rules it is being asked about; "
              f"leave-one-set-out withholds a whole family.\n")
            w(f"**Look at the spread, though.** Across {lin['seeds']} seeds the same "
              f"procedure produced anything from {lin['min']:.0%} to {lin['max']:.0%}. On "
              f"29 test problems one answer is 3.4 percentage points, so a single "
              f"train/test split of a dataset this size tells you almost nothing — the "
              f"honest report is the distribution, not a number. An earlier three-seed "
              f"run of exactly this code read 65.5%; twenty seeds put it at "
              f"{lin['mean']:.1%}.\n")
        if "mlp" in split and "linear" in split:
            m, l = split["mlp"], split["linear"]
            w(f"**The neural ranker is worse and far less stable**: {m['mean']:.1%} "
              f"± {m['std']:.1%} against the linear ranker's {l['mean']:.1%} "
              f"± {l['std']:.1%}, with individual seeds ranging {m['min']:.0%} to "
              f"{m['max']:.0%}. With 67 training problems and 48 features there is not "
              f"enough signal for an MLP to find structure a linear model misses, and "
              f"plenty of room for it to find structure that is not there.\n")

    # ---------------------------------------------------------- part 3
    epi_runs = load_llm_runs()
    inclass = load_inclass()
    if epi_runs:
        w("---\n\n## Part 3 — One model per lab\n")
        w("Identical inputs, identical prompt, one call per problem. Only the model "
          "changes.\n")
        allruns = sorted(epi_runs + inclass, key=lambda r: -r["correct"])
        w("| Model | Lab | Score | Accuracy | Cost | Cost per correct | Wall clock | "
          "Input tok | Output tok | No answer | In class? |")
        w("|---|---|---|---|---|---|---|---|---|---|---|")
        for r in allruns:
            cpc = money(r["cost"] / r["correct"]) if r["correct"] else "—"
            bad = r.get("unparseable", 0)
            w(f"| `{r['model']}` | {r['lab']} | {r['correct']}/{N} | "
              f"**{r['correct']/N:.1%}** | {money(r['cost'])} | {cpc} | "
              f"{r['wall']:.0f} s | {r['in_tok']:,} | {r['out_tok']:,} | "
              f"{bad if bad else '—'} | {'yes' if r.get('inclass') else 'no'} |")
        w("")
        total = sum(r["cost"] for r in epi_runs)
        w(f"Epilogue model spend: **{money(total)}** across {len(epi_runs)} runs of 96 "
          f"problems. Costs for runs made through OpenRouter are the amount actually "
          f"charged; the OpenAI-direct runs are token counts times list price.\n")
        bad_any = [r for r in allruns if r.get("unparseable")]
        if bad_any:
            w("**A \"no answer\" has two very different causes, and they are worth "
              "separating.**\n")
            w("*Yours.* The first pass here produced spectacular false zeros: an "
              "8,000-token output budget let reasoning models spend the whole allowance "
              "thinking and return an empty message, `gpt-4-turbo` rejects any budget "
              "above 4,096, and o-series models reject the `max_tokens` parameter "
              "outright. Three models scored 0/96 until the harness was fixed, and o3 "
              "went from 72/96 with 13 truncations to 76/96 with none. If you are "
              "benchmarking models, assume a suspiciously round zero is your bug before "
              "it is theirs.\n")
            w("*Theirs.* What is left after the fixes is genuine. Gemini 3.1 Pro's two "
              "failures are degeneration loops — the reply is the word \"producing\" "
              "repeated until the budget runs out. That is a real failure mode and it "
              "belongs in the score.\n")

        w("### By problem set\n")
        w("| Model | " + " | ".join(ravens.set_label(s) for s in ravens.SET_ORDER) + " |")
        w("|" + "---|" * (len(ravens.SET_ORDER) + 1))
        for r in allruns:
            w(f"| `{r['model']}` | " + " | ".join(str(x) for x in per_set(r["answers"])) + " |")
        w("")

    # ---------------------------------------------------------- gpt-3.5
    w("### GPT-3.5 could not take the test\n")
    w("`gpt-3.5-turbo` has no image input. There is no way to give it the puzzle at "
      "all — not a low score, an impossible task. That is worth showing students "
      "directly: the model that made ChatGPT famous in 2022 cannot even be entered into "
      "this comparison, and the barrier is modality, not reasoning.\n")

    # ---------------------------------------------------------- error analysis
    ea = os.path.join(RESULTS, "epilogue_error_analysis.txt")
    if os.path.exists(ea) and epi_runs:
        strong = [r for r in (epi_runs + inclass) if r["correct"] / N >= 0.70]
        names = sorted({n for n in TRUTH})
        wrong_by = {n: [r for r in strong if r["answers"].get(n) != TRUTH[n]]
                    for n in names}
        shared = [n for n in names if len(wrong_by[n]) >= 2]
        clean = [n for n in names if not wrong_by[n]]
        total_err = sum(len(v) for v in wrong_by.values())
        shared_err = sum(len(wrong_by[n]) for n in shared)

        w("---\n\n## Part 4 — Do the models fail on the same problems?\n")
        w(f"Yes, decisively. Taking the {len(strong)} models that score 70% or better:\n")
        w(f"- **{len(clean)} of {N} problems** were solved by every one of them.")
        w(f"- The **{len(shared)} problems missed by two or more** account for "
          f"**{shared_err/max(total_err,1):.0%} of all errors**.")
        w(f"- Error sets overlap about **4x more than independent errors would** "
          f"(Jaccard 0.16 observed against 0.04 for random failures of the same size).")
        w(f"- And when several models miss the same problem, **63% of them choose the "
          f"same wrong option** — against roughly 14% for guessing.\n")
        w("Independent labs, different architectures, different training data, "
          "converging on the same wrong answer. That is a property of the problems, "
          "not of any one model.\n")

        w("### The problems that beat them\n")
        w("| Problem | Correct | Missed by | Answers given | Agreement |")
        w("|---|---|---|---|---|")
        import collections as _c
        rows_ea = sorted(shared, key=lambda n: -len(wrong_by[n]))
        for n in rows_ea[:8]:
            picks = _c.Counter(r["answers"].get(n) for r in wrong_by[n])
            modal, cnt = picks.most_common(1)[0]
            dist = ", ".join(f"{k} ({v})" for k, v in picks.most_common())
            w(f"| `{n.replace('Problem ','')}` | {TRUTH[n]} | {len(wrong_by[n])}/"
              f"{len(strong)} | {dist} | {cnt/len(wrong_by[n]):.0%} on option {modal} |")
        w("")

        w("### Why: two rules, and they only apply one\n")
        w("`Challenge B-03` and `B-04` are the same puzzle in squares and circles. "
          "Cell A is five nested outlines; cell B is the same five with alternate bands "
          "filled black; cell C is three nested outlines. Two rules operate at once: "
          "**left-to-right fills alternate bands**, and **top-to-bottom removes two "
          "rings** (5 → 3 → 1).\n")
        w("The correct answer to B-03 needs both: one ring, filled — a plain black "
          "square. Five of seven strong models answered option 1, which is what you get "
          "by applying the fill rule to three rings and never checking the column. Their "
          "stated rules give them away:\n")
        w("> *\"The right column shows the left column\u2019s concentric squares with "
          "alternate rings filled black\"* — a correct description of half the puzzle.\n")
        w("The two that solved it stated both rules:\n")
        w("> *\"The number of nested squares decreases by 2 both across rows and down "
          "columns (5\u21923\u21921), and moving left\u2192right changes thin outlines "
          "to solid black.\"*\n")
        w("The system prompt explicitly says to check rows **and** columns before "
          "committing. Being told is not enough: the fill transformation is visually "
          "loud and the count progression is quiet, and the loud one wins.\n")

        w("### Why: the rule is right and the arithmetic is wrong\n")
        w("`Challenge D-08` is a Latin square over three attributes at once — one, two "
          "or three triangles; filled or outline; upright or right-pointing. Four of "
          "seven missed it, and this time they *state the rule correctly*:\n")
        w("> *\"Each row and column contains one cell with 1, 2, and 3 triangles, while "
          "the styles cycle among filled upright, outlined upright, and outlined "
          "right-pointing\"* — then picks the wrong cell.\n")
        w("That is not a reasoning failure, it is a bookkeeping failure: three "
          "constraints have to be intersected simultaneously and the models lose track. "
          "It is exactly the operation the classical agent does perfectly and for free, "
          "because it is a loop over permutations rather than something held in mind.\n")
        w("Which is the useful shape of the result. The LLMs are far better at "
          "*noticing* what kind of rule is present; they are worse at the exhaustive "
          "checking once they have. Set B\u2019s 2x2 problems, where there is only one "
          "row to learn the rule from and no second line to confirm it against, are "
          "where they lose most ground.\n")

    # ---------------------------------------------------------- batch mode
    bpath = os.path.join(RESULTS, "epilogue_batch_estimate.md")
    if os.path.exists(bpath):
        w("---\n\n## Would batch mode have been cheaper?\n")
        w("Yes, and by a lot. Every provider here sells an asynchronous batch tier: you "
          "upload a file of requests, they are worked through within a deadline "
          "(24 hours is the usual promise, often much sooner), and you pay less per "
          "token. OpenRouter publishes the batch rate as a separate `:batch` model id, "
          "so the table below is not a guess &mdash; it is the token counts actually "
          "recorded, multiplied by the published batch price.\n")
        w("**Nothing was submitted in batch mode.** These runs were all synchronous; "
          "this is what they would have cost.\n")
        w(open(bpath).read().strip() + "\n")
        w("### What you give up\n")
        w("Latency, and nothing else. The models are identical, so accuracy would not "
          "change. But a batch job is asynchronous: you submit and come back later. The "
          "in-class sweep finished in 77 seconds and the results went straight onto the "
          "screen; the same work in batch would have been cheaper and useless for that "
          "purpose.\n")
        w("Which makes the rule fairly clean. **A benchmark sweep is the ideal batch "
          "workload** &mdash; 96 independent calls, no ordering, nobody waiting. **A "
          "demo is the ideal synchronous workload.** This project happened to be both, "
          "and paid synchronous prices for the half that did not need to.\n")
        w("Two details worth noticing in the table. The discount is a flat 50% almost "
          "everywhere, which suggests it is a pricing convention rather than a "
          "measured cost saving. And `gemini-3.7-flash` is the exception at 75% off, "
          "which takes the cheapest good model in this comparison from $1.50 to $0.37 "
          "&mdash; four cents per correct answer, against $5.33 for the year-old "
          "flagship it beats.\n")

    # ---------------------------------------------------------- cost of epilogue
    if epi.get("found"):
        u = epi["usage"]
        w("---\n\n## What the epilogue itself cost\n")
        w("Measured from the Claude Code session transcript, counting only the turns "
          "after the request for this epilogue.\n")
        w("| | |")
        w("|---|---|")
        w(f"| Model doing the work | Claude Opus 5 (1M context) |")
        w(f"| Assistant turns | {epi['model_turns']} |")
        w(f"| Tool calls | {sum(epi['tools'].values())} "
          f"({', '.join(f'{v} {k}' for k, v in sorted(epi['tools'].items(), key=lambda t:-t[1]))}) |")
        w(f"| Output tokens | {u.get('output_tokens',0):,} |")
        w(f"| Fresh input tokens | {u.get('input_tokens',0):,} |")
        w(f"| Cache writes | {u.get('cache_creation_input_tokens',0):,} |")
        w(f"| Cache reads | {u.get('cache_read_input_tokens',0):,} |")
        if epi.get("span_seconds"):
            w(f"| Elapsed | {epi['span_seconds']/60:.0f} minutes |")
        w("")
        w("Regenerate with `python scripts/session_stats.py`; the figures move as the "
          "session continues.\n")

    w("---\n\n## Reproducing this\n")
    w("```bash\n"
      "python 04_neural/solver.py --steps 5000            # train + test the relation network\n"
      "python 02_classical_ai/split_eval.py --seeds 20    # the 70/30 protocol\n"
      "python 03_llm/openrouter_solver.py --all           # every lab's model\n"
      "python scripts/make_epilogue.py                    # rebuild this file\n"
      "```\n")

    out = os.path.join(ROOT, "EPILOGUE.md")
    with open(out, "w") as fh:
        fh.write("\n".join(L))
    print(f"wrote {out} ({len(L)} lines)")


if __name__ == "__main__":
    main()
