# Claude Raven's Demo

Three agents solving the same 96 Raven's Progressive Matrices, built to compare
how far the same problem gets you across three eras of technique:

| | agent | technique |
|---|---|---|
| 1 | [`01_original_2017/`](01_original_2017/) | the original 2017 Georgia Tech KBAI submission, run unmodified |
| 2 | [`02_classical_ai/`](02_classical_ai/) | symbolic rule search + classical CV + a learned ranker — **no LLM, no neural network** |
| 3 | [`03_llm/`](03_llm/) | one OpenAI multimodal reasoning call per problem |

Full write-up with per-set and per-problem breakdowns: **[COMPARISON.md](COMPARISON.md)**.

<!-- HEADLINE:START -->
<!-- HEADLINE:END -->

## The problems

96 puzzles from [`bcollier/KBAI_Ravens_Project`](https://github.com/bcollier/KBAI_Ravens_Project):
Basic and Challenge sets B, C, D and E, twelve problems each. Set B is 2x2 with
six answer options; C, D and E are 3x3 with eight. Every problem ships with panel
images; only Basic B and Basic C also ship with a verbal representation.

**All three agents get the images and nothing else.** The verbal representation is
withheld from everyone, because six of the eight sets do not have one and using it
would make the sets incomparable.

## Running everything

```bash
pip install -r requirements.txt

python run_original.py                      # agent 1  (~35 s)
python 02_classical_ai/solver.py            # agent 2  (~60 s)

export OPENAI_API_KEY=...
python 03_llm/solver.py --model gpt-5.6-sol # agent 3  (~80 s at 10 workers)

python scripts/compare.py                   # regenerate COMPARISON.md
```

Agents 1 and 2 need no network. Agent 3 needs an OpenAI key.

## Layout

```
Problems/            the 96 puzzles, unchanged from the 2017 repo
common/ravens.py     one loader all three agents share, so they see identical inputs
01_original_2017/    vendored 2017 code (byte-identical) + NOTES.md + typo_experiment.py
02_classical_ai/     imageops.py (CV primitives), features.py (rule engine), solver.py
03_llm/solver.py     OpenAI runner
scripts/compare.py   builds COMPARISON.md from results/ — no number is hand-typed
results/             every run's raw CSV output
```

## Reading the results honestly

- **Agent 1** is reported exactly as it behaves, including declining all 24 of the
  2x2 problems. See [NOTES.md](01_original_2017/NOTES.md) for why, and for what
  happens when you fix its two scoring bugs (nothing good).
- **Agent 2** has a supervised component, so its headline is **nested
  leave-one-problem-set-out cross-validation** — trained on seven sets, tested on
  the eighth, with hyper-parameters chosen inside the training folds only. The
  untrained rule-search-only score is reported alongside.
- **Agent 3** is scored on a public benchmark that has been on GitHub since 2017.
  Training-data contamination cannot be ruled out. See
  [03_llm/README.md](03_llm/README.md).
