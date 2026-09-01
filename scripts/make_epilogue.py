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
    ]
    for tag, fn, desc in variants:
        d = read_summary(os.path.join(RESULTS, fn))
        if not d:
            continue
        bold = "**" if tag == "v3" else ""
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
    w("A diagnostic worth copying: blanking every context panel and re-scoring drops "
      "the network to chance on synthetic problems. So it genuinely reads the matrix "
      "rather than exploiting a giveaway in how the distractors were made — the failure "
      "on real problems is domain gap, not a shortcut.\n")
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
        w("**The random split scores *higher* than leave-one-set-out, and that is "
          "expected.** A random 70/30 puts problems from every set in the training "
          "half, so at test time the model has already seen the family of rules it is "
          "being asked about. Leave-one-set-out withholds a whole family. The gap "
          "between the two is a measure of how much the agent relies on having seen "
          "that kind of problem before — and it is the more useful number if you care "
          "whether the thing generalises.\n")
        w("Note also the spread. On 29 test problems, one problem is 3.4 percentage "
          "points, so a single split's number is nearly meaningless on its own; only "
          "the distribution over seeds means anything.\n")

    # ---------------------------------------------------------- part 3
    epi_runs = load_llm_runs()
    inclass = load_inclass()
    if epi_runs:
        w("---\n\n## Part 3 — One model per lab\n")
        w("Identical inputs, identical prompt, one call per problem. Only the model "
          "changes.\n")
        allruns = sorted(epi_runs + inclass, key=lambda r: -r["correct"])
        w("| Model | Lab | Score | Accuracy | Cost | Cost per correct | Wall clock | "
          "Input tok | Output tok | In class? |")
        w("|---|---|---|---|---|---|---|---|---|---|")
        for r in allruns:
            cpc = money(r["cost"] / r["correct"]) if r["correct"] else "—"
            w(f"| `{r['model']}` | {r['lab']} | {r['correct']}/{N} | "
              f"**{r['correct']/N:.1%}** | {money(r['cost'])} | {cpc} | "
              f"{r['wall']:.0f} s | {r['in_tok']:,} | {r['out_tok']:,} | "
              f"{'yes' if r.get('inclass') else 'no'} |")
        w("")
        total = sum(r["cost"] for r in epi_runs)
        w(f"Epilogue model spend: **{money(total)}** across {len(epi_runs)} runs of 96 "
          f"problems.\n")

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
