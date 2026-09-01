"""Epilogue: run the same 96 problems against many labs' models via OpenRouter.

Deliberately a separate file from solver.py so the in-class run is untouched.
Identical inputs and identical prompt -- only the model changes. OpenRouter
reports the actual dollar cost of each call, which is recorded per problem.

Usage
    python 03_llm/openrouter_solver.py --model anthropic/claude-fable-5
    python 03_llm/openrouter_solver.py --all
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "..", "common")]

import ravens                                   # noqa: E402
from openai import OpenAI                       # noqa: E402
from solver import SYSTEM, data_url as data_uri  # noqa: E402  (same prompt as in class)

BASE_URL = "https://openrouter.ai/api/v1"
OUT_DIR = os.path.join(HERE, "..", "results")

# One flagship vision model per lab, plus two historical OpenAI models.
LINEUP = [
    "anthropic/claude-fable-5",
    "google/gemini-3.1-pro-preview",
    "google/gemini-3.7-flash",
    "x-ai/grok-4.6",
    "qwen/qwen3.8-max",
    "moonshotai/kimi-k3",
    "deepseek/deepseek-v4-flash-vision-exp",
    "meta-llama/llama-4-maverick",
    "z-ai/glm-5v-turbo",
    "openai/gpt-4o",
    "openai/gpt-4-turbo",
]

ASK = ("Reply with ONLY a JSON object of the form "
       '{"rule": "<one sentence>", "answer": <option number>, "confidence": <0-1>}.')


def slug(model):
    return re.sub(r"[^a-z0-9]+", "_", model.lower()).strip("_")


def build_messages(problem):
    grid = "A B C / D E F / G H ?" if problem.problem_type == "3x3" else "A B / C ?"
    content = [{"type": "text", "text":
                f"This is a {problem.problem_type} Raven's matrix laid out as {grid}. "
                f"The missing cell is marked with a question mark. There are "
                f"{problem.n_choices} answer options, numbered 1 to {problem.n_choices}."
                f"\n\nHere is the whole problem sheet:"}]
    if os.path.exists(problem.sheet_path()):
        content.append({"type": "image_url",
                        "image_url": {"url": data_uri(problem.sheet_path())}})
    content.append({"type": "text", "text": "Now each panel on its own. Matrix cells:"})
    for name in problem.givens:
        content.append({"type": "text", "text": f"Cell {name}:"})
        content.append({"type": "image_url", "image_url": {"url": data_uri(problem.path(name))}})
    content.append({"type": "text", "text": "Answer options:"})
    for name in problem.choices:
        content.append({"type": "text", "text": f"Option {name}:"})
        content.append({"type": "image_url", "image_url": {"url": data_uri(problem.path(name))}})
    content.append({"type": "text", "text":
                    f"Which option (1-{problem.n_choices}) completes the matrix? {ASK}"})
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": content}]


def parse_answer(text, n_choices):
    """Models vary in how well they honour a JSON instruction. Be forgiving."""
    if not text:
        return -1, "", ""
    m = re.search(r"\{.*\}", text, re.S)
    if m:
        try:
            obj = json.loads(m.group(0))
            a = int(obj.get("answer", -1))
            if 1 <= a <= n_choices:
                return a, str(obj.get("rule", ""))[:400], str(obj.get("confidence", ""))
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
    m = re.search(r'"?answer"?\s*[:=]\s*"?(\d+)', text, re.I)
    if m and 1 <= int(m.group(1)) <= n_choices:
        return int(m.group(1)), "", ""
    nums = re.findall(r"\b([1-8])\b", text)
    if nums and 1 <= int(nums[-1]) <= n_choices:
        return int(nums[-1]), "", ""
    return -1, "", ""


# Older and reasoning-family models disagree about the token-limit parameter.
# The limit has to cover reasoning tokens too: at 8,000 a reasoning model can
# burn the whole budget thinking and return an empty message. 32,000 matches
# what the in-class runs gave gpt-5.x, so the comparison stays fair. Models that
# cannot accept it are stepped down automatically from the API's error.
DEFAULT_MAX_TOKENS = 32000
TOKEN_PARAM = {}          # model -> ("max_tokens" | "max_completion_tokens", limit)


def ask(client, model, problem, retries=4, route="openrouter"):
    param, limit = TOKEN_PARAM.get(model, ("max_tokens", DEFAULT_MAX_TOKENS))
    extra = {"usage": {"include": True}} if route == "openrouter" else {}
    for attempt in range(retries):
        body = dict(model=model, messages=build_messages(problem), **{param: limit})
        try:
            t0 = time.time()
            r = client.chat.completions.create(**body, extra_body=extra)
            text = (r.choices[0].message.content or "") if r.choices else ""
            ans, rule, conf = parse_answer(text, problem.n_choices)
            u = r.usage
            cost = float(getattr(u, "cost", 0) or 0)
            details = getattr(u, "completion_tokens_details", None)
            reasoning = int(getattr(details, "reasoning_tokens", 0) or 0)
            return {"answer": ans, "rule": rule, "confidence": conf,
                    "latency": round(time.time() - t0, 2),
                    "in_tokens": int(getattr(u, "prompt_tokens", 0) or 0),
                    "out_tokens": int(getattr(u, "completion_tokens", 0) or 0),
                    "reasoning_tokens": reasoning, "cost": cost,
                    "error": "" if ans > 0 else
                             (f"truncated at {u.completion_tokens} output tokens"
                              if not text and getattr(u, "completion_tokens", 0) >= limit
                              else "unparseable: " + text[:120].replace("\n", " "))}
        except Exception as exc:
            msg = str(exc)
            if "max_completion_tokens" in msg and param == "max_tokens":
                param = "max_completion_tokens"
                TOKEN_PARAM[model] = (param, limit)
                continue
            m = re.search(r"supports at most (\d+) completion tokens", msg)
            if m and limit > int(m.group(1)):
                limit = int(m.group(1))
                TOKEN_PARAM[model] = (param, limit)
                continue
            if attempt == retries - 1:
                return {"answer": -1, "rule": "", "confidence": "", "latency": 0,
                        "in_tokens": 0, "out_tokens": 0, "reasoning_tokens": 0, "cost": 0.0,
                        "error": f"{type(exc).__name__}: {exc}"[:200]}
            time.sleep(2 ** attempt + random.random())


def make_client(route):
    if route == "openai":
        return OpenAI(api_key=os.environ["OPENAI_API_KEY"], timeout=600)
    return OpenAI(base_url=BASE_URL, api_key=os.environ["OPENROUTER_API_KEY"], timeout=600,
                  default_headers={
                      "HTTP-Referer": "https://github.com/bcollier/claude-ravens-demo",
                      "X-Title": "Claude Ravens Demo"})


def run_model(model, problems, workers=8, quiet=False, route="openrouter"):
    client = make_client(route)
    results = [None] * len(problems)
    lock, done = threading.Lock(), [0]

    def work(i):
        r = ask(client, model, problems[i], route=route)
        r["correct"] = int(r["answer"] == problems[i].answer)
        results[i] = r
        with lock:
            done[0] += 1
            if not quiet and done[0] % 12 == 0:
                hits = sum(x["correct"] for x in results if x)
                print(f"    {model}: {done[0]}/{len(problems)} done, {hits} correct",
                      flush=True)

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(work, range(len(problems))))
    wall = time.time() - t0

    total = sum(r["correct"] for r in results)
    cost = sum(r["cost"] for r in results)
    errs = sum(1 for r in results if r["error"])
    os.makedirs(OUT_DIR, exist_ok=True)
    base = os.path.join(OUT_DIR, f"epilogue_llm_{slug(model)}")
    with open(base + "_answers.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["ProblemSet", "RavensProblem", "Answer", "Correct", "Truth", "Confidence",
                    "LatencySeconds", "InputTokens", "OutputTokens", "ReasoningTokens",
                    "CostUSD", "Rule", "Error"])
        for p, r in zip(problems, results):
            w.writerow([p.set_name, p.name, r["answer"], r["correct"], p.answer,
                        r["confidence"], r["latency"], r["in_tokens"], r["out_tokens"],
                        r["reasoning_tokens"], f"{r['cost']:.6f}",
                        r["rule"].replace("\n", " "), r["error"]])
    with open(base + "_summary.txt", "w") as fh:
        fh.write(f"model            : {model}\nprovider_route   : {route}\n"
                 f"problems         : {len(problems)}\ncorrect          : {total}\n"
                 f"accuracy         : {total/len(problems):.4f}\n"
                 f"wall_seconds     : {wall:.2f}\nworkers          : {workers}\n"
                 f"cost_usd         : {cost:.4f}\nunparseable      : {errs}\n"
                 f"input_tokens     : {sum(r['in_tokens'] for r in results)}\n"
                 f"output_tokens    : {sum(r['out_tokens'] for r in results)}\n"
                 f"reasoning_tokens : {sum(r['reasoning_tokens'] for r in results)}\n")
    print(f"  {model:44s} {total:2d}/{len(problems)} = {total/len(problems):5.1%}  "
          f"${cost:6.3f}  {wall:5.0f}s  {errs} unparseable", flush=True)
    return total, cost, wall


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--route", default="openrouter", choices=["openrouter", "openai"])
    ap.add_argument("--models", nargs="*")
    args = ap.parse_args()

    problems = ravens.load_all()
    models = args.models or (LINEUP if args.all else [args.model])
    grand = 0.0
    for m in models:
        if not m:
            continue
        print(f"=== {m} ===", flush=True)
        try:
            _, cost, _ = run_model(m, problems, args.workers, route=args.route)
            grand += cost
        except Exception as exc:
            print(f"  {m}: FAILED {type(exc).__name__}: {exc}"[:300], flush=True)
    print(f"\nTotal OpenRouter spend this sweep: ${grand:.2f}")


if __name__ == "__main__":
    main()
