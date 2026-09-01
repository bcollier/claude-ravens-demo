# Agent 3: an LLM

One `responses.create` call per problem against OpenAI's multimodal reasoning
models. No tools, no retries on a wrong answer, no few-shot examples, no
per-problem-set tuning.

## What the model is given

Exactly what the other two agents get — **the images**. For each problem:

1. the assembled problem sheet, so the spatial layout is visible the way a human
   sees it,
2. every matrix cell as its own labelled image (`Cell A`, `Cell B`, …),
3. every answer option as its own labelled image (`Option 1`, `Option 2`, …).

It is told the grid shape (`A B C / D E F / G H ?`), how many options there are,
and — in the system prompt — the general families of rule that appear in Raven's
matrices. It is **not** told which problem set the puzzle came from, is not given
the verbal representation that ships with Basic B and Basic C, and never sees a
correct answer.

Replies are constrained by a JSON schema to `{rule, answer, confidence}`, so the
answer is always a parseable integer and the stated rule can be inspected
afterwards. The `rule` field is recorded in the results CSV.

## Running it

```bash
export OPENAI_API_KEY=...
python solver.py --model gpt-5.6-sol              # all 96
python solver.py --model gpt-5.6-sol --per-set 2  # quick 16-problem smoke test
python solver.py --model gpt-5.6-terra --workers 10
```

Results land in `../results/llm_<model>_answers.csv` plus a `_summary.txt`.

## Model choice

`gpt-5.6` ships as three variants. A pilot on the two hardest sets (Challenge D
and Challenge E, 24 problems) put `luna` and `sol` level on accuracy at 23/24,
but `sol` used about a quarter of the reasoning tokens and finished in half the
wall clock, so `sol` is the primary model here. `terra` and the 2025-era `gpt-5`
are included as a scaling comparison.

## The honest caveat

These problems are from a public Georgia Tech course repository that has been on
GitHub since 2017, and the underlying Raven's-style matrices are widely
discussed online. There is no way to rule out that some of this material is in
the training data. That does not make the score meaningless — the model still
has to read sixteen images and pick the right one — but "96.9%" should be read
as *96.9% on a public benchmark*, not as a clean measurement of novel visual
reasoning. The classical agent in `02_classical_ai/` has no such exposure, which
is part of why it is worth keeping in the comparison.
