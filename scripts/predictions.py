"""Score the class's predictions against what actually happened.

Two questions were asked before the session:

  Q1  Will Claude Code solve one or more Raven's problems WITHOUT an LLM by the
      end of class?                                  -> actually: YES, at 13:08
  Q2  ...the same question again, with an LLM. A duplicate; ignored.
  Q3  How many of the 96 will the non-LLM version solve by the end of class?
                                                     -> actually: 59

Free-text answers are turned into a number by the rules in PARSE below. Every
judgement call is written down here rather than done silently: percentages are
converted against 96, ranges take the midpoint, "a third"/"half" take the
fraction, and answers that give no quantity are scored but flagged.

The raw export is NOT committed -- it carries student and section identifiers.
Point --csv at it locally.

Usage:  python scripts/predictions.py --csv "../Knowledge-Based AI ....csv"
"""
from __future__ import annotations

import argparse
import csv
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRUTH_Q1 = "yes"          # the first non-LLM agent scored 55/96 at 13:08:12
TRUTH_Q3 = 59             # the non-LLM agent's final score, reported at 13:48:53
TOTAL = 96

# Explicit readings for answers a regex would get wrong. Value, then why.
PARSE = {
    "50% of more":                      (48, "read as '50% or more' -> half of 96"),
    "All of the 96.":                   (96, "all of them"),
    "Maybe around 5-10":                (8,  "midpoint of the range"),
    "1/4th to a half?":                 (36, "midpoint of 24 and 48"),
    "Probably a little under half":     (45, "just under half of 96"),
    "Probably 85%.":                    (82, "85% of 96"),
    "I estimate that it will get around a 65%": (62, "65% of 96"),
    "90+":                              (90, "floor of an open-ended answer"),
    "I think it will be able to solve at least one.": 
        (1,  "answers Q1, gives no quantity"),
    "I believe Claude Code will solve more than half of the 96 problems without the use of an LLM.":
        (49, "smallest integer satisfying 'more than half'"),
    "I think claude can solve over half of them correct, so maybe like 50+.":
        (50, "the number given"),
    "I think it'll be able to solve roughly a third of the 96, but not necessarily the easiest third.":
        (32, "a third of 96"),
    "I would assume that the non-LLM version would be able to solve roughly 30% of the problems which is around 28 problems.":
        (28, "the number given"),
    "I believe that the non-LLM version of this program will be able to solve around 50% by the end of class, so around 48.":
        (48, "the number given"),
}


def parse_estimate(text):
    t = " ".join(text.split())
    if t in PARSE:
        return PARSE[t]
    # "Out of 96, ... solve 50" must not read as 96. Drop the phrase that
    # restates the total before looking for the estimate.
    t = re.sub(r"\b(out\s+)?of\s+(the\s+)?96\b", " ", t, flags=re.I)
    m = re.search(r"(\d+)\s*%", t)
    if m:
        return round(int(m.group(1)) / 100 * TOTAL), f"{m.group(1)}% of {TOTAL}"
    nums = [int(n) for n in re.findall(r"\b(\d{1,3})\b", t) if int(n) <= TOTAL]
    if nums:
        return nums[0], "first number in the answer"
    return None, "no quantity given"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=os.path.join(
        os.path.dirname(ROOT),
        "Knowledge-Based AI Solving Raven's Puzzle Survey Student Analysis Report.csv"))
    args = ap.parse_args()

    rows = list(csv.reader(open(args.csv)))
    header = rows[0]
    i_q1 = next(i for i, h in enumerate(header) if "without the use of an LLM" in h)
    i_q3 = next(i for i, h in enumerate(header) if "How many out of 96" in h)
    i_time = header.index("submitted")

    people = []
    for r in rows[1:]:
        if not r or not r[0].strip():
            continue
        est, how = parse_estimate(r[i_q3])
        people.append({"name": r[0].strip(), "q1": r[i_q1].strip(),
                       "raw": " ".join(r[i_q3].split()), "est": est, "how": how,
                       "submitted": r[i_time]})

    q1_right = [p for p in people if p["q1"].lower().startswith(TRUTH_Q1)]
    print(f"Q1  'will it solve 1+ without an LLM?'   truth: YES")
    print(f"    {len(q1_right)}/{len(people)} correct ({len(q1_right)/len(people):.0%})")
    wrong = [p["name"] for p in people if p not in q1_right]
    print(f"    said no: {', '.join(wrong) if wrong else 'nobody'}\n")

    scored = [p for p in people if p["est"] is not None]
    for p in scored:
        p["err"] = abs(p["est"] - TRUTH_Q3)
    # ties broken by who submitted first
    scored.sort(key=lambda p: (p["err"], p["submitted"]))

    print(f"Q3  'how many of 96 will the non-LLM version solve?'   truth: {TRUTH_Q3}\n")
    print(f"    {'#':>2}  {'name':24s} {'guess':>5s} {'off by':>7s}   reading")
    for i, p in enumerate(scored[:10], 1):
        print(f"    {i:>2}  {p['name']:24s} {p['est']:>5d} {p['err']:>7d}   {p['how']}")

    ests = [p["est"] for p in scored]
    print(f"\n    class median {sorted(ests)[len(ests)//2]}, mean {sum(ests)/len(ests):.0f}, "
          f"range {min(ests)}-{max(ests)}")
    over = sum(1 for e in ests if e > TRUTH_Q3)
    print(f"    {over}/{len(ests)} guessed too high, {len(ests)-over-sum(1 for e in ests if e==TRUTH_Q3)} too low")

    # Two files. The full per-student scoring stays local (it carries every
    # student's name and answer); only aggregates and the podium are committed.
    import json
    podium, place, prev_err = [], 0, None
    for p_ in scored:
        if p_["err"] != prev_err:
            place += 1
            prev_err = p_["err"]
        if place > 4:
            break
        podium.append({"place": place, "name": p_["name"], "estimate": p_["est"],
                       "off_by": p_["err"]})
    summary = {
        "n_students": len(people),
        "q1_truth": "yes", "q1_correct": len(q1_right),
        "q3_truth": TRUTH_Q3,
        "median": sorted(ests)[len(ests) // 2],
        "mean": round(sum(ests) / len(ests)),
        "min": min(ests), "max": max(ests),
        "too_high": over, "too_low": len(ests) - over - sum(1 for e in ests if e == TRUTH_Q3),
        "within_10": sum(1 for e in ests if abs(e - TRUTH_Q3) <= 10),
        "podium": podium,
    }
    js = os.path.join(ROOT, "results", "predictions_summary.json")
    with open(js, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"wrote {js}")

    out = os.path.join(ROOT, "results", "predictions_scored.csv")
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["rank", "name", "q1_answer", "q1_correct", "estimate",
                    "off_by", "reading", "raw_answer"])
        for i, p in enumerate(scored, 1):
            w.writerow([i, p["name"], p["q1"], int(p in q1_right), p["est"],
                        p["err"], p["how"], p["raw"]])
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
