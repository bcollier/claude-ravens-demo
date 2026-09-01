# How this repository was made

A record of who asked for what, which models did the work, and what was decided
along the way — so the results in [COMPARISON.md](COMPARISON.md) can be judged
with the method in view.

- **Date built:** 1 September 2026
- **Requested by:** Ben Collier (`ben@collier.phd`)
- **Built by:** Claude Opus 5 (1M context), model ID `claude-opus-5[1m]`, running
  in [Claude Code](https://claude.com/claude-code) in an interactive terminal
  session on the user's machine
- **Session:** `session_01ULcCZmdUapQwVjBmG1z7XU`

## The original prompt, verbatim

> all right, Claude. I want to build a tool that will solve the Raven's intelligence tests. I'm going to give you a repo, and you can pull that down, and I want you to do three things:
>
> 1. I want you to get the code that's in the repo to work. The code is eight years old, I don't even know if it works anymore. I want you to show how good that code is, how many it gets right in one application.
>
> 2. I want you to build another application that will use any kind of technique except for an LLM. You're not allowed to use an LLM, but you can use any AI technique you want. I want you to go through all 96 of these puzzles and solve them using any kind of ML technique or AI technique that you want.
>
> 3. I want a third program that is going to use an LLM. I will give you an OpenAI key, so you can use the OpenAI libraries. Assume you can use GPT-5.6 or whatever you think is the most intelligent model right now to solve these.
>
> At the end, I'm going to want a comparison of my original code, the code with AI or ML that isn't an LLM, and then an LLM as well. You have my GitHub key, so you can push all of these to new repos. Actually, just make one repo called "Claude Raven's Demo" and push all of that to it.
>
> I'd like a markdown file comparing the performance across all three. If you get stuck, let me know, but I'm going to be teaching, so I don't want you to really ask me a bunch of questions. I'll paste the link right now.  https://github.com/bcollier/KBAI_Ravens_Project   ANy questions before you start?

Three follow-ups, also verbatim:

> yes go ahead, also make sure you record my full original prompt, what model you are currently running etc.

> if possible I would also like to know how many tokens you spent on this task, including input, output, cache, reasoning, etc. I want to know how many lines of code you generated, things like that. Give me visual representations of the architecture of the code, the logic behind why you chose the approach, what you tried along the way, the whole narrative. Everything that would be helpful for students learning about how AI works

> also explain the original code, what was it doing, what was effective about what it was doing and what was lacking

> also I want to know what you had accomplished by exactly 1:50PM today. look through the logs to tell me the answer to the following questions students were asked: Will Claude Code be able to solve any (1 or more) Raven's Matrices problem without the use of an LLM by the end of class? How many out of 96 do you estimate the non-LLM version of this program will be able to solve by the end of class?

Those produced [`docs/BUILD_LOG.md`](docs/BUILD_LOG.md), the deeper analysis in
[`01_original_2017/NOTES.md`](01_original_2017/NOTES.md), the illustrated page in
`docs/index.html`, and `scripts/session_stats.py`, which reads the build's real
token and line counts out of the Claude Code session transcript.

## Which model did what

| Role | Model | Notes |
|---|---|---|
| Wrote all the code, ran the experiments, wrote the write-up | **Claude Opus 5 (1M context)** — `claude-opus-5[1m]` | Anthropic, via Claude Code |
| Agent 3, primary | **`gpt-5.6-sol`** | OpenAI Responses API, `reasoning.effort = high` |
| Agent 3, second tier | **`gpt-5.6-terra`** | same settings |
| Agent 3, prior generation | **`gpt-5`** | same settings, included as a scaling point |

Claude wrote agent 2 and agent 3's harness but **is not part of any agent**. No
Anthropic model is called at solve time by any of the three agents. Agent 2
makes no network calls at all.

`gpt-5.6` was available in three variants (`luna`, `sol`, `terra`). Rather than
guess which was strongest, all three were piloted on the two hardest problem
sets (Challenge D and Challenge E, 24 problems): `luna` and `sol` tied at 23/24,
`terra` was behind. `sol` used roughly a quarter of `luna`'s reasoning tokens and
half its wall clock, so `sol` became the primary model.

## Exact LLM call settings

- OpenAI Python SDK `2.2.0`, `client.responses.create`
- `reasoning={"effort": "high"}`, `max_output_tokens=32000`
- Structured output via `text.format` `json_schema` (strict), returning
  `{rule, answer, confidence}`
- One independent call per problem — no conversation, no few-shot examples, no
  self-consistency voting, no retry on a wrong answer. Retries (up to 4, with
  exponential backoff) fire only on transport or JSON-parse failures.
- 10 concurrent workers
- Input per problem: the assembled problem sheet image, then every matrix cell
  and every answer option as its own labelled image. No verbal representation.

## Environment

| | |
|---|---|
| Machine | Apple M3 Max, 14 cores, macOS 26.6.2 (25G83) |
| Python | 3.12.7 |
| Packages | `numpy 1.26.4`, `Pillow 10.4.0`, `scipy 1.13.1`, `scikit-learn 1.5.1`, `openai 2.2.0`, `matplotlib 3.9.2` |

The 2017 agent was run under exactly this environment with **no changes to its
source**, which is the answer to "does eight-year-old code still work".

## Decisions made without asking

The brief said not to ask a lot of questions, so these were judgement calls.
Each is reversible and each is disclosed where it affects a number.

1. **All three agents get images only.** Sets Basic B and Basic C ship with a
   verbal representation; the other six sets do not. Giving it to some agents on
   some sets would make the sets incomparable, so it is withheld from everyone.
   The 2017 agent's verbal code path was already disabled by its author.
2. **All 96 problems, Basic and Challenge.** The upstream
   `ProblemSetList.txt` listed only Basic D and Basic E; it was replaced with all
   eight sets. This is the single change made to anything from the original repo,
   and it is a configuration file, not code.
3. **Agent 2's headline is nested leave-one-problem-set-out cross-validation.**
   It has a supervised component, so an in-sample number would not be a result.
   The in-sample and leave-one-problem-out figures are reported next to it.
4. **The 2017 agent's skipped problems count as wrong**, because they are: it
   returns `-1` on all 24 2x2 problems. Its score on the 72 it attempts (47.2%)
   is given in [`01_original_2017/NOTES.md`](01_original_2017/NOTES.md).
5. **The 2017 agent's bugs were left in place.** Fixing them is a separate,
   clearly-labelled experiment (`typo_experiment.py`), not the headline.
6. **The repository is public.** This was the one question actually asked, since
   creating a public repository is hard to undo. The user chose public.

## What would make these numbers stronger

- **Contamination.** These puzzles have been in a public GitHub repository since
  2017. There is no way here to rule out that they are in an LLM's training data.
  A clean test would need freshly generated matrices. Agent 2 has no such
  exposure.
- **Single run.** Each LLM answered each problem once. No variance estimate.
- **96 problems** is a small test. The difference between, say, 93/96 and 89/96
  is four problems.
