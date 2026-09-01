# The 2017 agent: does it still run, and how good is it?

Source: [`bcollier/KBAI_Ravens_Project`](https://github.com/bcollier/KBAI_Ravens_Project),
last commit *"Final submission, project 3"*. Vendored here **byte-for-byte** —
`Agent.py` and the six harness files are identical to the upstream repo. Verify with:

```bash
diff -r <(git -C /path/to/KBAI_Ravens_Project ls-files) .   # or just diff Agent.py
```

## Does it still run?

**Yes, with zero changes.** It was written for Python 3 with Pillow and numpy, and
it runs unmodified on Python 3.12.7 / Pillow 10.4 / numpy 1.26. No deprecation
errors, no porting, no shims. The only thing added to this directory is
`typo_experiment.py`, which does not touch `Agent.py`.

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

## How it works

Seven pixel statistics, each turned into a 1–10 score per answer option, then
combined with hand-set weights:

- **DPR** — dark-pixel ratio. Extrapolates a linear trend along the row (`H + (H-G)`),
  the column (`F + (F-C)`) and the diagonal, and scores each option by how close
  its ink coverage is to the prediction.
- **IPR** — intersection-pixel ratio. How much ink two panels share, normalised.
  Assumes the G↔H relationship should repeat as H↔answer.
- **Identity check** — if a row, column or diagonal is unchanged, prefer options
  identical to the last cell; otherwise *eliminate* options identical to any given.
- **Dark-pixel centroid** — where the ink sits, extrapolated the same way. Computed,
  then given weight `0.0`.
- **Object count** — from the verbal representation. Computed, then disabled entirely
  ("No longer used in Project 3").
- **Image addition** — if `A+B=C` and `D+E=F` as pixel unions, look for `G+H=answer`.
  Skipped for set D by an explicit name check.
- Subtraction and intersection appear as column headers and weights, but are never
  populated. They contribute zero.

The pixel loops are pure Python `getpixel` calls — the author's comment says as
much ("it is not the most efficient way to do it obviously"). It is still only
~0.5 s per problem, 34 s for all 96.

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
