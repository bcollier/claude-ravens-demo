# Agent 2: classical AI, no LLM

Three stages, none of them a language model and none of them a neural network.

## 1. Generate

`imageops.py` reduces every panel to a boolean ink mask and provides the
vocabulary a rule can be written in: eight dihedral transforms, six pixel
set-algebra operations, eight scalar measurements (ink coverage, connected
components, enclosed holes, bounding-box area, centroid, two symmetry scores),
and five *attribute descriptors* that pull a panel apart into things a rule can
talk about separately:

| descriptor | what it isolates |
|---|---|
| `whole` | the panel, cropped to its ink and rescaled (kills position and size) |
| `sil` | the filled outer silhouette — the panel's overall shape |
| `outer` | the largest connected component — usually the frame |
| `inner` | the smallest connected component — usually the inner shape |
| `core` | ink strictly inside the silhouette — interior detail such as bars |

`features.py` crosses that vocabulary with the geometry of the matrix — rows,
columns, and both **wrapped diagonals** — to produce a few hundred candidate
rules per problem, in six families:

- **transform** — `T` carries each cell of the line to the next (identity covers
  "the line does not change"), compared raw and after centroid alignment
- **set algebra** — the third cell is the union / intersection / XOR / difference
  of the first two
- **numeric** — some measurable property is constant, steps by a constant amount,
  or the first two sum to the third
- **relational** — the *similarities* inside the last line mirror those inside
  the complete lines (a generalisation of the 2017 agent's DPR/IPR idea)
- **attribute** — any of the six transforms above, applied to any of the five
  descriptors rather than the raw pixels
- **Latin square** — every line holds the same set of attribute values in a
  different order; the answer is the value the last line is missing

## 2. Test

The hard part is not generating rules, it is knowing which one is in force. A
diagnostic run showed the correct answer is the top pick of *some* family in
**86 of 96** problems — so selection, not generation, was the bottleneck.

So rules are scored the way you would score any predictor. Hide a line of the
matrix whose answer is already visible, make the rule recover it **competing
against the real answer options**, and record the likelihood it assigned to the
truth:

```
vconf(rule) = mean over hidden lines of  softmax(z-scored fits)[true panel]
```

Two properties make this work. Fits are z-scored first, so a rule measuring
pixel overlap and a rule counting holes are on the same scale. And a rule too
coarse to separate the options scores about `1/k` no matter how perfectly it
"holds" — coarse rules stop drowning out sharp ones.

## 3. Rank

Each family casts a vote for every option, scaled by `vconf^gamma`. Summing
those votes with no training at all answers **49/96**. A pairwise logistic
ranker (RankNet-style, trained on `correct − wrong` feature differences) then
learns how much to trust each family, taking it to **59/96**.

## Honesty

The headline is **nested leave-one-problem-set-out cross-validation**: the model
is trained on seven sets and tested on the eighth, and `tau`, `gamma` and the
regularisation strength are chosen by an inner cross-validation *within the
training folds only*. The held-out set never touches any fitting decision. The
friendlier standard leave-one-problem-out estimate is reported alongside.

## Running it

```bash
python solver.py                            # full evaluation (~60 s, cached after the first run)
python solver.py --refresh                  # rebuild the rule cache
python solver.py --explain "Basic Problem D-09"
```

`--explain` prints the rules the engine trusted most and what each one picked.
Set E is built on pixel logic, and the engine finds it:

```
Basic Problem E-05   (3x3, correct answer = 5)
   trust  held-out  holds  family         rule                       picks
   0.459     0.460  0.987  logic_col      a_not_b                    5 <-- correct
   0.458     0.460  0.984  logic_col      xor                        5 <-- correct
   0.458     0.460  0.984  logic_col      nand_ink                   5 <-- correct
   0.419     0.426  0.943  logic_row      xor                        5 <-- correct
   0.419     0.426  0.943  logic_row      nand_ink                   5 <-- correct
   0.419     0.426  0.943  logic_row      a_not_b                    5 <-- correct
   0.369     0.375  0.940  simpat_col     simpattern                 5 <-- correct
   0.323     0.323  1.000  latin_col      latin:inner                1
   0.308     0.316  0.908  simpat_row     simpattern                 5 <-- correct
   0.286     0.308  0.756  logic_col      or                         8
  combined vote -> 5
```

`held-out` is the likelihood the rule assigned to a panel it was not allowed to
see; `holds` is how well it literally fits the visible lines. `xor` and
`nand_ink` are the same operation on these binary panels, which is why they
score identically.
