# Build log: how the three agents were actually made

A lab notebook rather than a tidy write-up. Sixteen things were tried; six were
dropped or made results worse. The dead ends are the part worth reading.

Everything here is reconstructed from the Claude Code session transcript with
`python scripts/session_stats.py`, which writes
[`results/session_timeline.txt`](../results/session_timeline.txt) and
[`results/session_stats.json`](../results/session_stats.json).

---

## The class questions, answered

Students were asked to predict, before the session:

> **Will Claude Code be able to solve any (1 or more) Raven's Matrices problem
> without the use of an LLM by the end of class?**

**Yes — 9 minutes 34 seconds in.** The session began at **12:58:38**. The first
non-LLM agent to run scored **55 of 96 at 13:08:12**, before any machine
learning was involved at all: pure rule search with confidence-weighted voting.

> **How many out of 96 do you estimate the non-LLM version will solve by the end
> of class?**

**At exactly 1:50 PM the answer was 59 of 96 (61.5%)**, or 49 of 96 with no
training whatsoever. The last non-LLM run before that cutoff finished at
**13:48:53** and reported exactly those two numbers — and they are still the
final numbers. Nothing after 1:50 PM improved the non-LLM agent; the remaining
time went to the LLM sweeps and the write-up.

### Timeline of the non-LLM agent, local time

| time | | score | what changed |
|---|---|---|---|
| 12:58:38 | | — | session starts, repo cloned |
| 13:03:58 | | **34/96** | the 2017 agent, run unmodified — the baseline to beat |
| 13:08:12 | | **55/96** | first non-LLM agent: rule search, no training |
| 13:09:05 | | **59/96** | + learned ranker, leave-one-set-out |
| 13:12:58 | ↓ | 57/96 | added attribute descriptors — *cross-validated score got worse* |
| 13:16:06 | ↓ | 53/96 | switched to held-out rule validation — worse again |
| 13:17:39 | ↓ | 47/96 | validation as a likelihood — worse still |
| 13:22:09 | | **59/96** | scale-free validation + restructured features; nested CV |
| 13:31:36 | | 59/96 | rejected: separate rankers per problem type |
| 13:48:53 | | **59/96** | rejected: top-k family gating, option centrality |
| — | | **59/96** | final |

Three consecutive changes made the number go **down** before the version that
made it go up. That is what the middle of a project looks like.

---

## The sixteen experiments

| # | tried | outcome | kept |
|---|---|---|---|
| 1 | Run the 2017 code unmodified on Python 3.12 | runs; **34/96** | ✓ baseline |
| 2 | Fix its two scoring bugs | 34, 34, **33** | ✗ made it worse |
| 3 | Rule search v1 — transforms, pixel set algebra, numeric progressions, relational patterns; rules scored by how well they fit the visible rows | **55/96** untrained | ✓ |
| 4 | Pairwise logistic ranker over 58 features | LOSO **61.5%**, in-sample only 66.7% | features were the ceiling, not the model |
| 5 | **Looked at the actual set-D images** | found panels are compositions: outer frame × inner shape, arranged as Latin squares | → led to 6 |
| 6 | Five attribute descriptors (silhouette, largest component, smallest component, interior) + attribute transforms + a Latin-square family; 70 features | in-sample rose to 75%, **cross-validated fell to 59.4%** | classic overfitting |
| 7 | **Diagnostic: is the answer even in the rule space?** | any rule **95/96**; best-family oracle **90/96**; single most-trusted rule **34/96** | the finding that reframed everything |
| 8 | Score rules by whether they recover a *hidden* line of the matrix | set D 6→10, but **set E collapsed 8→4** | partial |
| 9 | Diagnosed #8: rule trust was multiplied by fit quality, which punishes exact logic rules (0.899 vs a coarse rule's 1.000) | — | diagnosis |
| 10 | Validation as a softmax likelihood, τ = 0.05 on raw fits | **47/96** — worse | pixel overlap and hole-counts are on different scales |
| 11 | z-score the fits before the softmax, so trust is scale-free | swept τ and γ | ✓ the fix |
| 12 | Collapse 4 features/family → 2 pre-scaled votes (48 features) | nested LOSO **61.5%** | ✓ |
| 13 | Separate rankers for 2×2 and 3×3 problems | 58.3% | ✗ dropped |
| 14 | Vote using only the top-k best-validated families | 34–46% | ✗ dropped, badly |
| 15 | "Distractors cluster around the answer" centrality feature | ±1 problem | ✗ dropped |
| 16 | LLM: pilot three `gpt-5.6` variants on the hardest 24 problems | luna 23/24, sol 23/24 at ¼ the tokens, terra behind | chose `sol` |

### The two that mattered

**Experiment 7** is the one to steal. Before adding any more machinery, ask
whether the machinery you already have *can* express the right answer. It could,
95 times out of 96. That meant every hour spent inventing new rule types would
have been wasted, and the entire problem was rule *selection*. Without that
measurement the natural instinct — #6, add more rules — was actively harmful.

**Experiment 11** is the fix that came out of it. A rule's trust is now the
probability it assigns to a panel it was not allowed to see, competing against
the real answer options, computed on z-scored fits so that "how much ink
overlaps" and "how many holes" are judged on the same scale. That single change
is the difference between a pile of pixel heuristics and a solver.

---

## What it cost

Measured from the session transcript, not estimated.

| | |
|---|---|
| Wall-clock session | ~68 minutes to the final non-LLM number; the rest was LLM sweeps and writing |
| Model turns | 285, all Claude Opus 5 (1M context) |
| Tool calls | 137 — of which **128 were shell commands** |
| Output tokens | ~460,000 (code written + reasoning; only ~8,000 characters were prose shown on screen) |
| Fresh input tokens | 570 |
| Cache writes | ~745,000 |
| Cache reads | ~49,500,000 |
| Python written | ~1,840 lines across 10 files |
| Prose written | ~380 lines of Markdown |

Two things worth noticing in that table. **Cache reads outnumber fresh input
tokens roughly 87,000 to 1** — almost every token the model read was a cached
re-read of the same growing conversation, which is what makes a long agentic
session affordable. And **output tokens dwarf visible prose by about 50×**:
nearly everything the model "said" was code and private reasoning, not text on
the screen.

---

## What a student should take from this

1. **Measure the ceiling before you raise it.** Experiment 7 cost ten minutes
   and redirected the whole project.
2. **Going backwards is normal.** Three consecutive changes made the score
   worse. Each was informative; the third one explained the first two.
3. **Look at the data with your eyes.** The single biggest structural insight
   (experiment 5) came from rendering three failing puzzles and looking at them,
   not from any metric.
4. **Hold something out, always.** Experiment 6 looked like a 9-point
   improvement in-sample and was a 2-point regression in reality.
5. **Delete what does not work.** Four experiments were dropped. The final agent
   is smaller than the largest intermediate version.
