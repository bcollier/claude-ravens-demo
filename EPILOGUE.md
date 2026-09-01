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
| v3 | as v2, plus composed panels: nested frames, inner shapes, bars | 61% | 11/96 (11.5%) |
| v4 | two generator bugs fixed: 18.6% of problems were unanswerable, and shapes were drawn far too small | 56% | **10/96 (10.4%)** |

And the two rankers that use the symbolic features, on the 70/30 protocol from Part 2 so they are directly comparable:

| Ranker | Features | 70/30 test accuracy |
|---|---|---|
| linear (in class) | 48 symbolic rule features | 61.6% ± 6.7% |
| **MLP** | the same 48 features, a small neural network | 59.0% ± 9.7% |

### What this shows

**The training distribution moved the numbers; the architecture never did.** Every row above is the same network with the same budget. v1's generator gave every attribute its own rule at once, producing matrices that are visually chaotic in a way real Raven's problems never are. v2 fixed that and learned the synthetic task far better. v3 added the composed panels — nested frames, inner shapes, bars — that sets D and E are actually built from.

**Learning the synthetic task better did not mean solving the real one better.** That is the sharpest lesson here. v2 roughly doubled held-out synthetic accuracy over v1 and did *worse* on the real problems. A network can only learn the world you show it, and the gap between that world and the real one does not appear anywhere in the training metrics.

**Two of the biggest problems were in the data, and only looking at it found them.** Rendering the training panels next to the real ones at the size the network actually sees revealed that synthetic shapes were drawn far too small, and — worse — that the generator chose which attributes a rule acts on *before* deciding whether the panel could express them, silently erasing the variation in **18.6% of training problems** and leaving eight identical panels with an unanswerable question. The loss curve looked healthy throughout. No metric reported it.

### Two diagnostics that explain the failure

```
context-blind test (synthetic)
  full problem                      54.7%
  every context panel erased        14.3%   (chance ~13%)

duplicate-shortcut test (real 96)
  model picks the option most like a visible panel   29%
  that option is actually correct                    10%
```

**It does read the matrix.** Erase every context panel and the network drops to chance on synthetic problems, so it is not exploiting a giveaway in how the distractors were built — a common artefact in this literature.

**But it learned the wrong rule of thumb, and the data bug is why.** Those 18.6% of broken problems had eight identical context panels, so their answer *was* a copy of the context — trivially solvable, and quietly inflating the synthetic score. The network duly learned to favour the option that duplicates something already on the page. On the real 96 it does that 29% of the time, and that option is correct only 10% of the time — worse than the 13% you would get by guessing. A heuristic that is anti-correlated with the truth is how a model ends up *below* chance.

The symbolic agent learned the opposite sign from real data: its largest negative weight is on exactly this feature, `dup_max`. Same signal, fitted on the real distribution instead of a broken synthetic one, and it points the other way.

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
| `google/gemini-3.7-flash` | Google | 92/96 | **95.8%** | $1.50 | $0.02 | 144 s | 1,624,800 | 74,659 | — | no |
| `google/gemini-3.1-pro-preview` | Google | 90/96 | **93.8%** | $6.64 | $0.07 | 344 s | 1,602,428 | 285,994 | 2 | no |
| `anthropic/claude-fable-5` | Anthropic | 89/96 | **92.7%** | $4.80 | $0.05 | 146 s | 283,488 | 39,402 | — | no |
| `gpt-5.6-terra` | OpenAI | 89/96 | **92.7%** | $1.02 | $0.01 | 193 s | 243,912 | 44,419 | — | yes |
| `gpt-5` | OpenAI | 84/96 | **87.5%** | $5.33 | $0.06 | 1773 s | 383,928 | 484,800 | — | yes |
| `o3` | OpenAI | 76/96 | **79.2%** | $3.59 | $0.05 | 947 s | 405,120 | 347,404 | — | no |
| `gpt-4o` | OpenAI | 41/96 | **42.7%** | $1.17 | $0.03 | 36 s | 454,896 | 3,687 | — | no |
| `gpt-4.1` | OpenAI | 36/96 | **37.5%** | $0.95 | $0.03 | 33 s | 454,896 | 5,435 | — | no |
| `gpt-4-turbo` | OpenAI | 34/96 | **35.4%** | $4.68 | $0.14 | 101 s | 454,896 | 4,299 | — | no |

Epilogue model spend: **$23.34** across 7 runs of 96 problems. Costs for runs made through OpenRouter are the amount actually charged; the OpenAI-direct runs are token counts times list price.

**A "no answer" has two very different causes, and they are worth separating.**

*Yours.* The first pass here produced spectacular false zeros: an 8,000-token output budget let reasoning models spend the whole allowance thinking and return an empty message, `gpt-4-turbo` rejects any budget above 4,096, and o-series models reject the `max_tokens` parameter outright. Three models scored 0/96 until the harness was fixed, and o3 went from 72/96 with 13 truncations to 76/96 with none. If you are benchmarking models, assume a suspiciously round zero is your bug before it is theirs.

*Theirs.* What is left after the fixes is genuine. Gemini 3.1 Pro's two failures are degeneration loops — the reply is the word "producing" repeated until the budget runs out. That is a real failure mode and it belongs in the score.

### By problem set

| Model | Basic B | Basic C | Basic D | Basic E | Challenge B | Challenge C | Challenge D | Challenge E |
|---|---|---|---|---|---|---|---|---|
| `gpt-5.6-sol` | 12 | 12 | 12 | 12 | 11 | 12 | 11 | 11 |
| `google/gemini-3.7-flash` | 12 | 11 | 12 | 12 | 9 | 12 | 12 | 12 |
| `google/gemini-3.1-pro-preview` | 10 | 11 | 12 | 12 | 9 | 12 | 12 | 12 |
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

## Part 4 — Do the models fail on the same problems?

Yes, decisively. Taking the 7 models that score 70% or better:

- **65 of 96 problems** were solved by every one of them.
- The **14 problems missed by two or more** account for **71% of all errors**.
- Error sets overlap about **4x more than independent errors would** (Jaccard 0.16 observed against 0.04 for random failures of the same size).
- And when several models miss the same problem, **63% of them choose the same wrong option** — against roughly 14% for guessing.

Independent labs, different architectures, different training data, converging on the same wrong answer. That is a property of the problems, not of any one model.

### The problems that beat them

| Problem | Correct | Missed by | Answers given | Agreement |
|---|---|---|---|---|
| `Challenge B-03` | 3 | 5/7 | 1 (4), 5 (1) | 80% on option 1 |
| `Challenge B-04` | 4 | 4/7 | 1 (2), 5 (2) | 50% on option 1 |
| `Challenge D-08` | 1 | 4/7 | 2 (3), 6 (1) | 75% on option 2 |
| `Challenge E-08` | 7 | 4/7 | 5 (2), 3 (2) | 50% on option 5 |
| `Challenge E-09` | 1 | 4/7 | 2 (2), 5 (1), 6 (1) | 50% on option 2 |
| `Basic C-12` | 8 | 3/7 | 5 (1), 7 (1), 6 (1) | 33% on option 5 |
| `Challenge B-02` | 1 | 3/7 | 5 (1), 3 (1), 4 (1) | 33% on option 5 |
| `Challenge D-05` | 2 | 3/7 | 8 (2), 3 (1) | 67% on option 8 |

### Why: two rules, and they only apply one

`Challenge B-03` and `B-04` are the same puzzle in squares and circles. Cell A is five nested outlines; cell B is the same five with alternate bands filled black; cell C is three nested outlines. Two rules operate at once: **left-to-right fills alternate bands**, and **top-to-bottom removes two rings** (5 → 3 → 1).

The correct answer to B-03 needs both: one ring, filled — a plain black square. Five of seven strong models answered option 1, which is what you get by applying the fill rule to three rings and never checking the column. Their stated rules give them away:

> *"The right column shows the left column’s concentric squares with alternate rings filled black"* — a correct description of half the puzzle.

The two that solved it stated both rules:

> *"The number of nested squares decreases by 2 both across rows and down columns (5→3→1), and moving left→right changes thin outlines to solid black."*

The system prompt explicitly says to check rows **and** columns before committing. Being told is not enough: the fill transformation is visually loud and the count progression is quiet, and the loud one wins.

### Why: the rule is right and the arithmetic is wrong

`Challenge D-08` is a Latin square over three attributes at once — one, two or three triangles; filled or outline; upright or right-pointing. Four of seven missed it, and this time they *state the rule correctly*:

> *"Each row and column contains one cell with 1, 2, and 3 triangles, while the styles cycle among filled upright, outlined upright, and outlined right-pointing"* — then picks the wrong cell.

That is not a reasoning failure, it is a bookkeeping failure: three constraints have to be intersected simultaneously and the models lose track. It is exactly the operation the classical agent does perfectly and for free, because it is a loop over permutations rather than something held in mind.

Which is the useful shape of the result. The LLMs are far better at *noticing* what kind of rule is present; they are worse at the exhaustive checking once they have. Set B’s 2x2 problems, where there is only one row to learn the rule from and no second line to confirm it against, are where they lose most ground.

---

## Would batch mode have been cheaper?

Yes, and by a lot. Every provider here sells an asynchronous batch tier: you upload a file of requests, they are worked through within a deadline (24 hours is the usual promise, often much sooner), and you pay less per token. OpenRouter publishes the batch rate as a separate `:batch` model id, so the table below is not a guess &mdash; it is the token counts actually recorded, multiplied by the published batch price.

**Nothing was submitted in batch mode.** These runs were all synchronous; this is what they would have cost.

| Model | Score | Tokens in / out | Standard | Batch | Saving |
|---|---|---|---|---|---|
| `anthropic/claude-fable-5` | 89/96 | 283,488 / 39,402 | $4.80 | $2.40 | 50% |
| `google/gemini-3.1-pro-preview` | 90/96 | 1,602,428 / 285,994 | $6.64 | $3.32 | 50% |
| `google/gemini-3.7-flash` | 92/96 | 1,624,800 / 74,659 | $1.50 | $0.37 | 75% |
| `gpt-4.1` | 36/96 | 454,896 / 5,435 | $0.95 | $0.48 | 50% |
| `gpt-4-turbo` | 34/96 | 454,896 / 4,299 | $4.68 | $2.34 | 50% |
| `gpt-4o` | 41/96 | 454,896 / 3,687 | $1.17 | $0.59 | 50% |
| `o3` | 76/96 | 405,120 / 347,404 | $3.59 | $1.79 | 50% |
| `gpt-5.6-sol` | 93/96 | 243,912 / 29,504 | $0.78 | $0.39 | 50% |
| `gpt-5.6-terra` | 89/96 | 243,912 / 44,419 | $1.02 | $0.51 | 50% |
| `gpt-5` | 84/96 | 383,928 / 484,800 | $5.33 | $2.66 | 50% |
| **total** | | | **$30.47** | **$14.86** | **51%** |

### What you give up

Latency, and nothing else. The models are identical, so accuracy would not change. But a batch job is asynchronous: you submit and come back later. The in-class sweep finished in 77 seconds and the results went straight onto the screen; the same work in batch would have been cheaper and useless for that purpose.

Which makes the rule fairly clean. **A benchmark sweep is the ideal batch workload** &mdash; 96 independent calls, no ordering, nobody waiting. **A demo is the ideal synchronous workload.** This project happened to be both, and paid synchronous prices for the half that did not need to.

Two details worth noticing in the table. The discount is a flat 50% almost everywhere, which suggests it is a pricing convention rather than a measured cost saving. And `gemini-3.7-flash` is the exception at 75% off, which takes the cheapest good model in this comparison from $1.50 to $0.37 &mdash; four cents per correct answer, against $5.33 for the year-old flagship it beats.

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
