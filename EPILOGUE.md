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

About 398,817 parameters, trained on mps in 1079 seconds.

### Results

Three versions of the *same network and the same training budget*. Only the generator changed.

| | Training data | Held-out synthetic | **Real 96** |
|---|---|---|---|
| v1 | every attribute given its own rule at once | 29% | 15/96 (15.6%) |
| v2 | one or two active attributes, single repeated shapes | 59% | 10/96 (10.4%) |
| v3 | as v2, plus composed panels: nested frames, inner shapes, bars | 61% | **11/96 (11.5%)** |

And the two rankers that use the symbolic features, on the 70/30 protocol from Part 2 so they are directly comparable:

| Ranker | Features | 70/30 test accuracy |
|---|---|---|
| linear (in class) | 48 symbolic rule features | 61.6% ± 6.7% |
| **MLP** | the same 48 features, a small neural network | 59.0% ± 9.7% |

### What this shows

**The training distribution moved the numbers; the architecture never did.** Every row above is the same network with the same budget. v1's generator gave every attribute its own rule at once, producing matrices that are visually chaotic in a way real Raven's problems never are. v2 fixed that and learned the synthetic task far better. v3 added the composed panels — nested frames, inner shapes, bars — that sets D and E are actually built from.

**Learning the synthetic task better did not mean solving the real one better.** That is the sharpest lesson here. v2 roughly doubled held-out synthetic accuracy over v1 and did *worse* on the real problems. A network can only learn the world you show it, and the gap between that world and the real one does not appear anywhere in the training metrics.

A diagnostic worth copying: blanking every context panel and re-scoring drops the network to chance on synthetic problems. So it genuinely reads the matrix rather than exploiting a giveaway in how the distractors were made — the failure on real problems is domain gap, not a shortcut.

**A network trained on synthetic data does not reach the symbolic agent.** That is the honest result and it is worth sitting with: the relation network has to *discover* concepts like "the outer frame is unchanged" from pixels, with only the rules I thought to put in the generator to learn from. The symbolic agent was handed those concepts. When you have 96 problems and strong priors about the domain, encoding the priors beats learning them.

**The neural ranker did not beat the linear one either.** With 67 training problems and 48 features, there is not enough signal for an MLP to find structure a linear model misses. This is a useful negative result: "use a neural network" is not free, and the honest comparison shows when it does not pay.

---

## Part 2 — A proper train/test split

### Was the in-class number trained on its test data?

**The model weights were not.** The in-class headline was *nested leave-one-problem-set-out*: train on seven problem sets, test on the eighth, with the hyper-parameters chosen by an inner cross-validation inside the training folds only. Every one of the 96 predictions came from a model that had not seen that problem.

**The feature design was.** This is the real leak and no re-split fixes it. Partway through, three failing set-D problems were rendered and inspected; attribute descriptors and the Latin-square rule family were added *because of what those images showed*. The rule vocabulary in `features.py` was shaped by looking at the test set. Model fitting is clean; feature engineering is not.

### The 70/30 numbers

Stratified by problem set, 67 train / 29 test, hyper-parameters chosen on the training half only, repeated over 20 random seeds.

| Ranker | 70/30 test accuracy | Spread across seeds | In-class leave-one-set-out |
|---|---|---|---|
| linear | **61.6%** | 52% – 76% (± 6.7%) | 61.5% |
| mlp | **59.0%** | 38% – 79% (± 9.7%) | — |

**The two protocols agree.** Leave-one-problem-set-out gave 61.5%; a random 70/30 gives 61.6%, a difference of 0.1 points — well inside the noise. The in-class number was not flattered by its protocol.

**Look at the spread, though.** Across 20 seeds the same procedure produced anything from 52% to 76%. On 29 test problems one answer is 3.4 percentage points, so a single train/test split of a dataset this size tells you almost nothing — the honest report is the distribution, not a number. An earlier three-seed run of exactly this code read 65.5%; twenty seeds put it at 61.6%.

**The neural ranker is worse and far less stable**: 59.0% ± 9.7% against the linear ranker's 61.6% ± 6.7%, with individual seeds ranging 38% to 79%. With 67 training problems and 48 features there is not enough signal for an MLP to find structure a linear model misses, and plenty of room for it to find structure that is not there.

---

## Part 3 — One model per lab

Identical inputs, identical prompt, one call per problem. Only the model changes.

| Model | Lab | Score | Accuracy | Cost | Cost per correct | Wall clock | Input tok | Output tok | No answer | In class? |
|---|---|---|---|---|---|---|---|---|---|---|
| `gpt-5.6-sol` | OpenAI | 93/96 | **96.9%** | $0.78 | $0.0084 | 77 s | 243,912 | 29,504 | — | yes |
| `anthropic/claude-fable-5` | Anthropic | 89/96 | **92.7%** | $4.80 | $0.05 | 146 s | 283,488 | 39,402 | — | no |
| `gpt-5.6-terra` | OpenAI | 89/96 | **92.7%** | $1.02 | $0.01 | 193 s | 243,912 | 44,419 | — | yes |
| `gpt-5` | OpenAI | 84/96 | **87.5%** | $5.33 | $0.06 | 1773 s | 383,928 | 484,800 | — | yes |
| `o3` | OpenAI | 76/96 | **79.2%** | $3.59 | $0.05 | 947 s | 405,120 | 347,404 | — | no |
| `gpt-4o` | OpenAI | 41/96 | **42.7%** | $1.17 | $0.03 | 36 s | 454,896 | 3,687 | — | no |
| `gpt-4.1` | OpenAI | 36/96 | **37.5%** | $0.95 | $0.03 | 33 s | 454,896 | 5,435 | — | no |
| `gpt-4-turbo` | OpenAI | 34/96 | **35.4%** | $4.68 | $0.14 | 101 s | 454,896 | 4,299 | — | no |

Epilogue model spend: **$15.20** across 5 runs of 96 problems. Costs for runs made through OpenRouter are the amount actually charged; the OpenAI-direct runs are token counts times list price.

### By problem set

| Model | Basic B | Basic C | Basic D | Basic E | Challenge B | Challenge C | Challenge D | Challenge E |
|---|---|---|---|---|---|---|---|---|
| `gpt-5.6-sol` | 12 | 12 | 12 | 12 | 11 | 12 | 11 | 11 |
| `anthropic/claude-fable-5` | 12 | 12 | 12 | 12 | 10 | 12 | 10 | 9 |
| `gpt-5.6-terra` | 12 | 12 | 12 | 11 | 10 | 12 | 11 | 9 |
| `gpt-5` | 11 | 12 | 12 | 11 | 8 | 12 | 9 | 9 |
| `o3` | 11 | 9 | 12 | 9 | 7 | 12 | 7 | 9 |
| `gpt-4o` | 7 | 6 | 5 | 7 | 4 | 7 | 3 | 2 |
| `gpt-4.1` | 8 | 7 | 4 | 3 | 5 | 5 | 2 | 2 |
| `gpt-4-turbo` | 7 | 8 | 3 | 4 | 5 | 4 | 2 | 1 |

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
