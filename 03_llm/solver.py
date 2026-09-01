"""LLM Raven's solver -- OpenAI multimodal reasoning models.

Same inputs as the other two agents: the panel images only. No verbal
representation, no hint about which problem set a puzzle came from, one
independent call per problem, no retrying a wrong answer. The model sees the
assembled problem sheet plus each panel individually, and returns JSON with
its answer and the rule it believes is in force.

Usage
    python solver.py --model gpt-5.6-sol
    python solver.py --model gpt-5.6-sol --sets "Basic Problems D" --limit 6
"""
from __future__ import annotations

import argparse
import base64
import csv
import json
import os
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, "..", "common")]

import ravens                                   # noqa: E402
from openai import OpenAI                       # noqa: E402

SYSTEM = """You are solving Raven's Progressive Matrices, a nonverbal reasoning test.

You are shown a matrix of figures with the last cell missing, then a numbered
list of answer options. Work out the rule (or combination of rules) that
governs how the figures change across the rows and down the columns, then
choose the option that completes the matrix.

Rules in these problems include: shapes being reflected, rotated or resized;
objects being added, removed, filled or unfilled; a property changing by a
constant amount along a row or column; the third cell being the pixel union,
intersection or difference of the first two; and Latin-square arrangements
where every row and column contains each shape exactly once.

Check your candidate answer against BOTH the rows and the columns before
committing. Answer with the option number only."""

SCHEMA = {
    "type": "object",
    "properties": {
        "rule": {"type": "string", "description": "The rule governing the matrix, in one sentence."},
        "answer": {"type": "integer", "description": "The number of the chosen answer option."},
        "confidence": {"type": "number", "description": "0-1 confidence in the answer."},
    },
    "required": ["rule", "answer", "confidence"],
    "additionalProperties": False,
}


def data_url(path):
    with open(path, "rb") as fh:
        return "data:image/png;base64," + base64.b64encode(fh.read()).decode()


def build_input(problem):
    grid = "A B C / D E F / G H ?" if problem.problem_type == "3x3" else "A B / C ?"
    content = [{"type": "input_text", "text":
                f"This is a {problem.problem_type} Raven's matrix laid out as {grid}. "
                f"The missing cell is marked with a question mark. There are "
                f"{problem.n_choices} answer options, numbered 1 to {problem.n_choices}.\n\n"
                f"Here is the whole problem sheet:"}]
    if os.path.exists(problem.sheet_path()):
        content.append({"type": "input_image", "image_url": data_url(problem.sheet_path())})
    content.append({"type": "input_text", "text": "Now each panel on its own. Matrix cells:"})
    for name in problem.givens:
        content.append({"type": "input_text", "text": f"Cell {name}:"})
        content.append({"type": "input_image", "image_url": data_url(problem.path(name))})
    content.append({"type": "input_text", "text": "Answer options:"})
    for name in problem.choices:
        content.append({"type": "input_text", "text": f"Option {name}:"})
        content.append({"type": "input_image", "image_url": data_url(problem.path(name))})
    content.append({"type": "input_text", "text":
                    f"Which option (1-{problem.n_choices}) completes the matrix?"})
    return [{"role": "user", "content": content}]


def ask(client, model, problem, effort, max_retries=4):
    kwargs = dict(
        model=model,
        instructions=SYSTEM,
        input=build_input(problem),
        text={"format": {"type": "json_schema", "name": "ravens_answer",
                         "schema": SCHEMA, "strict": True}},
        max_output_tokens=32000,
    )
    if effort:
        kwargs["reasoning"] = {"effort": effort}

    for attempt in range(max_retries):
        try:
            t0 = time.time()
            r = client.responses.create(**kwargs)
            payload = json.loads(r.output_text)
            u = r.usage
            reasoning = getattr(getattr(u, "output_tokens_details", None), "reasoning_tokens", 0) or 0
            return {
                "answer": int(payload["answer"]),
                "rule": payload.get("rule", ""),
                "confidence": payload.get("confidence", ""),
                "latency": round(time.time() - t0, 2),
                "in_tokens": u.input_tokens,
                "out_tokens": u.output_tokens,
                "reasoning_tokens": reasoning,
                "error": "",
            }
        except Exception as exc:                     # transient API / parse failure
            if attempt == max_retries - 1:
                return {"answer": -1, "rule": "", "confidence": "", "latency": 0,
                        "in_tokens": 0, "out_tokens": 0, "reasoning_tokens": 0,
                        "error": f"{type(exc).__name__}: {exc}"[:300]}
            time.sleep(2 ** attempt + random.random())


def run(model, problems, effort, workers, out_csv, quiet=False):
    client = OpenAI()
    lock = threading.Lock()
    done = [0]
    results = [None] * len(problems)

    def work(i):
        r = ask(client, model, problems[i], effort)
        r["correct"] = int(r["answer"] == problems[i].answer)
        results[i] = r
        with lock:
            done[0] += 1
            if not quiet:
                mark = "OK " if r["correct"] else ("ERR" if r["error"] else "-- ")
                print(f"  [{done[0]:3d}/{len(problems)}] {mark} {problems[i].name:26s} "
                      f"said {r['answer']} truth {problems[i].answer} "
                      f"({r['reasoning_tokens']} rtok, {r['latency']}s)"
                      + (f"  {r['error']}" if r["error"] else ""), flush=True)

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(work, range(len(problems))))
    wall = time.time() - t0

    if out_csv:
        os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
        with open(out_csv, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["ProblemSet", "RavensProblem", "Answer", "Correct", "Truth",
                        "Confidence", "LatencySeconds", "InputTokens", "OutputTokens",
                        "ReasoningTokens", "Rule", "Error"])
            for p, r in zip(problems, results):
                w.writerow([p.set_name, p.name, r["answer"], r["correct"], p.answer,
                            r["confidence"], r["latency"], r["in_tokens"], r["out_tokens"],
                            r["reasoning_tokens"], r["rule"].replace("\n", " "), r["error"]])
    return results, wall


def summarise(model, problems, results, wall):
    by_set = {}
    for p, r in zip(problems, results):
        c, n = by_set.get(p.set_name, (0, 0))
        by_set[p.set_name] = (c + r["correct"], n + 1)
    total = sum(r["correct"] for r in results)
    errs = sum(1 for r in results if r["error"])
    print(f"\n{model}: {total}/{len(problems)} = {total/len(problems):.1%}"
          f"   wall {wall:.0f}s   api errors {errs}")
    for s in ravens.SET_ORDER:
        if s in by_set:
            print(f"  {s:26s} {by_set[s][0]:2d}/{by_set[s][1]}")
    print(f"  tokens: {sum(r['in_tokens'] for r in results):,} in, "
          f"{sum(r['out_tokens'] for r in results):,} out "
          f"({sum(r['reasoning_tokens'] for r in results):,} reasoning)")
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt-5.6-sol")
    ap.add_argument("--effort", default="high")
    ap.add_argument("--sets", nargs="*", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--per-set", type=int, default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    problems = ravens.load_all(args.sets)
    if args.per_set:
        keep, seen = [], {}
        for p in problems:
            seen[p.set_name] = seen.get(p.set_name, 0) + 1
            if seen[p.set_name] <= args.per_set:
                keep.append(p)
        problems = keep
    if args.limit:
        problems = problems[:args.limit]

    out = args.out or os.path.join(HERE, "..", "results",
                                   f"llm_{args.model.replace('.', '_').replace('-', '_')}_answers.csv")
    print(f"{args.model} (effort={args.effort}) on {len(problems)} problems, "
          f"{args.workers} workers")
    results, wall = run(args.model, problems, args.effort, args.workers, out)
    total = summarise(args.model, problems, results, wall)
    with open(out.replace("_answers.csv", "_summary.txt"), "w") as fh:
        fh.write(f"model            : {args.model}\n")
        fh.write(f"effort           : {args.effort}\n")
        fh.write(f"problems         : {len(problems)}\n")
        fh.write(f"correct          : {total}\n")
        fh.write(f"accuracy         : {total/len(problems):.4f}\n")
        fh.write(f"wall_seconds     : {wall:.2f}\n")
        fh.write(f"workers          : {args.workers}\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
