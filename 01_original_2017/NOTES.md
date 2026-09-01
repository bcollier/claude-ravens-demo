# The 2017 agent: does it still run, and how good is it?

Source: [`bcollier/KBAI_Ravens_Project`](https://github.com/bcollier/KBAI_Ravens_Project),
last commit *"Final submission, project 3"*. Vendored here **byte-for-byte** —
`Agent.py` and the six harness files are identical to the upstream repo. Verify with:

```bash
git clone https://github.com/bcollier/KBAI_Ravens_Project /tmp/kbai
for f in Agent.py ProblemSet.py RavensFigure.py RavensGrader.py \
         RavensObject.py RavensProblem.py RavensProject.py; do
  diff -q "/tmp/kbai/$f" "$f" && echo "identical  $f"
done
```

## Does it still run?

**Yes, with zero changes.** It was written for Python 3 with Pillow and numpy, and
it runs unmodified on Python 3.12.7 / Pillow 10.4 / numpy 1.26. No deprecation
errors, no porting, no shims. The only thing added to this directory is
`typo_experiment.py`, which does not touch `Agent.py`. Not vendored: the
upstream repo's `API/` javadoc, `submit.py`, `bonnie/` autograder client,
`imagechops_scratch.py` and `.idea/` — none are needed to run the agent.

```bash
python run_original.py       # from the repo root
```

## How good is it?

**34 / 96 = 35.4%** across all eight problem sets.

| Set | Correct | Incorrect | Skipped |
|---|---|---|---|
| Basic B | 0 | 0 | 12 |
| Basic C | 7 | 5 | 0 |
| Basic D | 8 | 4 | 0 |
| Basic E | 7 | 5 | 0 |
| Challenge B | 0 | 0 | 12 |
| Challenge C | 6 | 6 | 0 |
| Challenge D | 2 | 10 | 0 |
| Challenge E | 4 | 8 | 0 |

Two things drive that number.

**It declines every 2x2 problem.** `Solve()` dispatches to `solve3x3` for `"3x3"`
problems and otherwise returns `-1`, which the grader records as *Skipped*. The
2x2 path (`verbal_solve`, `guess_or_pass`) is commented out in the final commit.
That is 24 problems — a quarter of the test — answered with a shrug. On the 72
problems it does attempt it scores **34/72 = 47.2%**.

**It was tuned on two sets.** The upstream `Problems/ProblemSetList.txt` lists
only `Basic Problems D` and `Basic Problems E`, so those are the sets the weights
were tuned against. It holds up on Basic C/D/E (7–8 of 12) and falls apart on the
Challenge sets it never saw (2/12 on Challenge D). That is the classic symptom of
hand-tuned weights: it generalises about as far as the data it was tuned on.

## What the code is actually doing

Strip away the plumbing and the agent is one idea applied seven ways:

> Reduce each 184&times;184 panel to a single number. Extrapolate that number
> along the row, the column and the diagonal. Score each answer option by how
> close it lands to the prediction. Add the scores up with fixed weights and
> take the best.

The seven measurements:

| # | statistic | what it computes | weight in the final score |
|---|---|---|---|
| 1 | **DPR** — dark pixel ratio | fraction of the panel that is inked | row 0.50, col 0.20, diag 0.50 |
| 2 | **IPR** — intersection pixel ratio | `2 × shared ink / (ink A + ink B)`, i.e. the Dice coefficient | row 0.10\*, col 0.10, diag 0.15\* |
| 3 | **identity check** | is a row/col/diagonal unchanged? then prefer options identical to the last cell | 5 |
| 4 | **elimination** | if a line is *not* unchanged, penalise options identical to any visible panel | −10 |
| 5 | **image addition** | if `A ∪ B = C` and `D ∪ E = F`, look for `G ∪ H = answer` | 5 |
| 6 | **dark pixel centroid** | where the ink sits, extrapolated the same way | **0** |
| 7 | **object count** | number of objects, from the verbal representation | **0** |

\* see the bugs section — the row IPR weight is never actually applied and the
diagonal one is applied as `1.0`.

Extrapolation is always the same linear step: `predict = last + (last − first)`.
For a row that is `H + (H − G)`; for a column `F + (F − C)`. Subtraction and
intersection have column headers, constants and weights reserved for them, but
are never populated — they contribute zero.

## What was effective

Four of the ideas in this file are the right ideas, and all four survive into
the much larger agent in `02_classical_ai/` as whole rule families.

**Comparing relationships rather than objects (IPR).** The agent does not ask
"which option looks like H"; it asks "is the H→option relationship the same as
the G→H relationship". That is analogy at the right level of abstraction, and it
is not the obvious first thing to try. In the rebuilt agent this became the
`simpat` family — and the learned ranker gives it one of the largest positive
weights of any of the 22 families. A 2017 student project found by hand the
feature a fitted model would later rank near the top.

**Elimination as a hard negative.** If nothing in the matrix repeats, then an
option that is pixel-identical to a panel you can already see is probably a
distractor. That is a *constraint*, not a score, and mixing constraints with
scores is the correct instinct. The rebuilt agent carries the same signal as
`dup_max`, and its learned weight is large and **negative** — independently
confirming the 2017 judgement.

**Pixel set algebra (image addition).** Testing `A ∪ B = C` on the rows you can
see before applying it to the row you cannot is exactly the generate-and-test
structure the rebuilt agent uses everywhere. Problem set E is built almost
entirely on this, and the union is the single most common case.

**Linear extrapolation of a measured property.** `H + (H − G)` is a real
hypothesis about how a matrix progresses, and it is right often enough to carry
sets C and D. The rebuilt agent's `num` family is this same idea over eight
measurements instead of one, with the step estimated from the visible rows
rather than assumed.

It is also worth saying plainly: the code is readable, honestly commented, and
it still runs eight years later with no changes. That is not nothing.

## What was lacking

**Representation — the ceiling.** Every statistic collapses a panel to one
number. Two panels with identical ink coverage are, to this agent, the same
panel. That makes an entire class of rule literally inexpressible: *the inner
shape rotates while the outer frame stays put*, *each row contains a circle, a
square and a triangle in some order*. Problem set D is built on exactly those
compositional rules, and no reweighting of seven scalars can reach them. The
rebuilt agent's first real gain came from decomposing a panel into an outer
silhouette, a largest component, a smallest component and an interior — at which
point the Latin-square rules in set D become checkable.

**No per-problem rule selection — the mechanism of the ceiling.** The weights
are constants. Every problem is scored with the same blend of DPR, IPR, identity
and addition, whether or not those rules have anything to do with it. There is
no step that asks *which rule is in force here?*

The rebuilt agent measured what that costs. `diagnose.py` scores the strategy of
picking the single highest-scoring rule and believing it, across a rule space
hundreds of times larger than this one: **34/96**. The 2017 agent scores
**34/96**. The equality of the two numbers is a coincidence; what it points at
is not. Applying one fixed scoring scheme to every problem lands in the
mid-thirties almost regardless of how good the individual measurements are.
Choosing per problem is where the remaining sixty points live.

**No validation signal.** Nothing in the file asks whether a rule actually holds
for the problem in front of it. `check_addition` comes closest — it verifies
`A ∪ B = C` on the visible rows first — but its result is a boolean gate, not a
score, and the other six statistics are applied unconditionally. Without a
per-problem measure of "is this rule any good here", tuning has nowhere to go
except global weights, and global weights are tuned by running the whole set and
nudging. Which is what happened, and why the bugs below did not matter.

**Tuned on a quarter of the test.** The upstream `ProblemSetList.txt` lists only
Basic D and Basic E. The results show it: 7–8 of 12 on the Basic sets, 2 of 12
on Challenge D.

**Unfinished coverage.** 2&times;2 problems return `-1`. Subtraction and
intersection are declared but never written. Centroid and object count are
computed in full and then multiplied by zero. These are the fingerprints of a
graded course project meeting a deadline, not a design flaw — but they cost 24
problems outright.

## Two real bugs in the scoring expression

```python
IPR_COL_WT * solution_scores[i, IPR_COL] +
IPR_COL_WT * solution_scores[i, IPR_COL] +      # duplicated; IPR_ROW never used
...
IPR_DIAG_WT + solution_scores[i, IPR_DIAG] +    # '+' where every other term has '*'
```

The first means the intersection-pixel-ratio **row** score is computed and then
thrown away, while the column score is counted twice. The second means the
diagonal IPR score enters with weight `1.0` instead of the intended `0.15`, about
seven times its intended influence.

`typo_experiment.py` fixes them, one at a time and together, on temporary copies:

| variant | correct |
|---|---|
| as shipped (2017) | 34 / 96 |
| fix the `+` typo only | 34 / 96 |
| fix the duplicated `IPR_COL` only | 34 / 96 |
| fix both | 33 / 96 |

**Fixing the bugs does not help, and fixing both makes it slightly worse.** The
weights were hand-tuned by running the agent and nudging numbers until the score
went up, so the tuning had already absorbed the bugs. The typos were load-bearing.
This is a good argument for the approach in `02_classical_ai/`, where the weights
are fitted rather than nudged, and reported under cross-validation.

## Verdict

The code is honest, readable, well-commented work that still runs a
software generation later. Its ceiling is set by its design, not its bugs: seven
scalar summaries of a whole panel cannot express "the inner shape rotates while the
outer frame stays put", so no amount of weight tuning gets it past the mid-30s on
this test.
