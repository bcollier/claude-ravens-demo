# Epilogue: comparisons added after class

Everything in [COMPARISON.md](COMPARISON.md) is exactly as it ran during the session and has not been touched. This file adds three things that came up afterwards.

1. **A neural network.** The in-class "no LLM" agent used classical computer vision and a linear ranker. It should have had a neural network in the mix.
2. **A real train/test split.** The in-class headline was cross-validated, but not in the conventional 70/30 form, and the feature *design* saw all 96 problems.
3. **Other labs' models.** The in-class LLM runs were all OpenAI.

---

## Part 1 — A neural network

### The problem with 96 problems

You cannot train a network on 96 examples. The standard way round it, and what the research literature on this exact task does, is to **generate an unlimited supply of synthetic matrices**, train on those, and treat the 96 real problems as a held-out test the network has never seen. `04_neural/render.py` is that generator: it builds matrices from the same rule vocabulary the real sets use (constant, progression, Latin square, arithmetic on counts, and pixel XOR/OR/AND) and renders them as black line art the way the originals look.

### The model

`04_neural/wren.py` is a relation network in the style of Santoro et al.'s WReN. Each panel goes through a small CNN to a 128-dimensional embedding tagged with its grid position; every *pair* of panels goes through a shared MLP; the pair representations are summed and a head scores the candidate. The pairwise sum is the whole point — a rule in a Raven's matrix is a statement about how two cells relate, so the architecture is built to compute exactly that.

### Results

| Model | What it is | Real 96 | Notes |
|---|---|---|---|
| D1a ablation | the same network, trained on a generator that varies every attribute at once | 15/96 (15.6%) | 29% synthetic |
| **D2 neural ranker** | an MLP scoring each option from the symbolic rule features, instead of the linear ranker | 62.1% ± 5.6% | 70/30 split, 3 seeds |
| *(for reference)* the in-class linear ranker | classical features, logistic pairwise ranker | 65.5% ± 5.6% | same 70/30 protocol |

### What this shows

**The training distribution mattered more than the architecture.** The first generator gave every attribute its own rule simultaneously, producing matrices that are visually chaotic in a way real Raven's problems never are. Same network, same budget, same code — only the data changed.

**A network trained on synthetic data does not reach the symbolic agent.** That is the honest result and it is worth sitting with: the relation network has to *discover* concepts like "the outer frame is unchanged" from pixels, with only the rules I thought to put in the generator to learn from. The symbolic agent was handed those concepts. When you have 96 problems and strong priors about the domain, encoding the priors beats learning them.

**The neural ranker did not beat the linear one either.** With 67 training problems and 48 features, there is not enough signal for an MLP to find structure a linear model misses. This is a useful negative result: "use a neural network" is not free, and the honest comparison shows when it does not pay.

---

## Part 2 — A proper train/test split

### Was the in-class number trained on its test data?

**The model weights were not.** The in-class headline was *nested leave-one-problem-set-out*: train on seven problem sets, test on the eighth, with the hyper-parameters chosen by an inner cross-validation inside the training folds only. Every one of the 96 predictions came from a model that had not seen that problem.

**The feature design was.** This is the real leak and no re-split fixes it. Partway through, three failing set-D problems were rendered and inspected; attribute descriptors and the Latin-square rule family were added *because of what those images showed*. The rule vocabulary in `features.py` was shaped by looking at the test set. Model fitting is clean; feature engineering is not.

### The 70/30 numbers

Stratified by problem set, 67 train / 29 test, hyper-parameters chosen on the training half only, repeated over 3 random seeds.

| Ranker | 70/30 test accuracy | Spread across seeds | In-class leave-one-set-out |
|---|---|---|---|
| linear | **65.5%** | 59% – 72% (± 5.6%) | 61.5% |
| mlp | **62.1%** | 55% – 69% (± 5.6%) | — |

**The random split scores *higher* than leave-one-set-out, and that is expected.** A random 70/30 puts problems from every set in the training half, so at test time the model has already seen the family of rules it is being asked about. Leave-one-set-out withholds a whole family. The gap between the two is a measure of how much the agent relies on having seen that kind of problem before — and it is the more useful number if you care whether the thing generalises.

Note also the spread. On 29 test problems, one problem is 3.4 percentage points, so a single split's number is nearly meaningless on its own; only the distribution over seeds means anything.

---

## Part 3 — One model per lab

Identical inputs, identical prompt, one call per problem. Only the model changes.

| Model | Lab | Score | Accuracy | Cost | Cost per correct | Wall clock | Input tok | Output tok | In class? |
|---|---|---|---|---|---|---|---|---|---|
| `gpt-5.6-sol` | OpenAI | 93/96 | **96.9%** | $0.78 | $0.0084 | 77 s | 243,912 | 29,504 | yes |
| `gpt-5.6-terra` | OpenAI | 89/96 | **92.7%** | $1.02 | $0.01 | 193 s | 243,912 | 44,419 | yes |
| `gpt-5` | OpenAI | 84/96 | **87.5%** | $5.33 | $0.06 | 1773 s | 383,928 | 484,800 | yes |
| `gpt-4o` | OpenAI | 41/96 | **42.7%** | $1.17 | $0.03 | 36 s | 454,896 | 3,687 | no |
| `gpt-4.1` | OpenAI | 36/96 | **37.5%** | $0.95 | $0.03 | 33 s | 454,896 | 5,435 | no |
| `gpt-4-turbo` | OpenAI | 34/96 | **35.4%** | $4.68 | $0.14 | 101 s | 454,896 | 4,299 | no |
| `meta-llama/llama-4-maverick` | Meta | 0/96 | **0.0%** | $0.0000 | — | 215 s | 0 | 0 | no |
| `o3` | OpenAI | 0/96 | **0.0%** | $0.0000 | — | 120 s | 0 | 0 | no |

Epilogue model spend: **$6.81** across 5 runs of 96 problems.

### By problem set

| Model | Basic B | Basic C | Basic D | Basic E | Challenge B | Challenge C | Challenge D | Challenge E |
|---|---|---|---|---|---|---|---|---|
| `gpt-5.6-sol` | 12 | 12 | 12 | 12 | 11 | 12 | 11 | 11 |
| `gpt-5.6-terra` | 12 | 12 | 12 | 11 | 10 | 12 | 11 | 9 |
| `gpt-5` | 11 | 12 | 12 | 11 | 8 | 12 | 9 | 9 |
| `gpt-4o` | 7 | 6 | 5 | 7 | 4 | 7 | 3 | 2 |
| `gpt-4.1` | 8 | 7 | 4 | 3 | 5 | 5 | 2 | 2 |
| `gpt-4-turbo` | 7 | 8 | 3 | 4 | 5 | 4 | 2 | 1 |
| `meta-llama/llama-4-maverick` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `o3` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

### GPT-3.5 could not take the test

`gpt-3.5-turbo` has no image input. There is no way to give it the puzzle at all — not a low score, an impossible task. That is worth showing students directly: the model that made ChatGPT famous in 2022 cannot even be entered into this comparison, and the barrier is modality, not reasoning.

---

## What the epilogue itself cost

Measured from the Claude Code session transcript, counting only the turns after the request for this epilogue.

| | |
|---|---|
| Model doing the work | Claude Opus 5 (1M context) |
| Assistant turns | 80 |
| Tool calls | 39 (38 Bash, 1 Read) |
| Output tokens | 150,470 |
| Fresh input tokens | 160 |
| Cache writes | 167,835 |
| Cache reads | 31,931,894 |
| Elapsed | 21 minutes |

Regenerate with `python scripts/session_stats.py`; the figures move as the session continues.

---

## Reproducing this

```bash
python 04_neural/solver.py --steps 5000            # train + test the relation network
python 02_classical_ai/split_eval.py --seeds 20    # the 70/30 protocol
python 03_llm/openrouter_solver.py --all           # every lab's model
python scripts/make_epilogue.py                    # rebuild this file
```
