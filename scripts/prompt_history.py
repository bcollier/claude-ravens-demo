"""Extract every instruction the user gave, verbatim, into PROMPT_HISTORY.md.

Read out of the Claude Code session transcript rather than reconstructed from
memory, so the wording is exactly what was typed. Tool results, system
reminders, background-task notifications and local slash-commands are filtered
out; only genuine typed messages remain.

Usage:  python scripts/prompt_history.py [path/to/transcript.jsonl]
"""
from __future__ import annotations

import datetime
import glob
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TZ = datetime.timezone(datetime.timedelta(hours=-4))

NOISE = ("<system-reminder>", "<task-notification>", "<local-command",
         "[SYSTEM NOTIFICATION", "Caveat: The messages below")


def is_only_attachments(text):
    """Pasted screenshots arrive as their own bare '[Image: source: ...]' lines."""
    lines = [ln for ln in text.split("\n") if ln.strip()]
    return bool(lines) and all(ln.strip().startswith("[Image:") for ln in lines)


def find_transcript():
    hits = glob.glob(os.path.expanduser("~/.claude/projects/*/*.jsonl"))
    return max(hits, key=os.path.getmtime) if hits else None


# Skills inject themselves as user-role messages; they were not typed by anyone.
SKILL_MARKERS = ("Base directory for this skill:",
                 "Approach this as the design lead",
                 "Draw as the engineer who has to live with the decision")


def user_messages(path):
    out = []
    for line in open(path):
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue

        # Messages typed while a turn was already running are queued, and appear
        # as queue-operation records rather than as user messages.
        if rec.get("type") == "queue-operation" and rec.get("operation") == "enqueue":
            text = str(rec.get("content") or "").strip()
            if text and not any(n in text for n in NOISE) and not is_only_attachments(text):
                when = rec.get("timestamp")
                stamp = (datetime.datetime.fromisoformat(when.replace("Z", "+00:00"))
                         .astimezone(TZ).strftime("%H:%M") if when else "")
                out.append((stamp, text))
            continue

        msg = rec.get("message") or {}
        if msg.get("role") != "user":
            continue
        content = msg.get("content")
        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            parts = [b.get("text", "") for b in content
                     if isinstance(b, dict) and b.get("type") == "text"]
            if any(isinstance(b, dict) and b.get("type") == "tool_result"
                   for b in content):
                continue
            text = "\n".join(parts)
        else:
            continue
        text = text.strip()
        if not text or any(n in text for n in NOISE):
            continue
        if any(m in text for m in SKILL_MARKERS) or text.startswith("<command-name>"):
            continue
        if is_only_attachments(text):
            continue
        when = rec.get("timestamp")
        stamp = (datetime.datetime.fromisoformat(when.replace("Z", "+00:00"))
                 .astimezone(TZ).strftime("%H:%M") if when else "")
        out.append((stamp, text))
    out.sort(key=lambda t: t[0])
    # the harness can echo a message more than once; keep first occurrences
    seen, uniq = set(), []
    for stamp, text in out:
        key = re.sub(r"\s+", " ", text)[:120]
        if key in seen:
            continue
        seen.add(key)
        uniq.append((stamp, text))
    return uniq


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else find_transcript()
    msgs = user_messages(path)

    L = ["# Every instruction given, in order\n",
         "The complete set of prompts behind this project, quoted exactly as typed. "
         "Extracted from the Claude Code session transcript by "
         "`scripts/prompt_history.py`, not written from memory.\n",
         "It is worth reading as its own artefact. Almost none of these prompts "
         "specify *how* to do anything. They state a goal, and several of the most "
         "consequential ones are corrections or challenges to work already done — "
         "asking for a neural network that had been left out, asking whether the "
         "evaluation was honest, asking what had actually been achieved by a "
         "particular minute. The quality of the result owes more to those than to "
         "the opening brief.\n",
         f"All times are local, on the day of the session. {len(msgs)} messages.\n",
         "---\n"]

    for i, (stamp, text) in enumerate(msgs, 1):
        text = "\n".join(ln for ln in text.split("\n")
                          if not ln.strip().startswith("[Image: source:")).strip()
        L.append(f"## {i}. {stamp}\n")
        L.append("\n".join("> " + ln if ln.strip() else ">"
                           for ln in text.split("\n")))
        L.append("")

    out = os.path.join(ROOT, "PROMPT_HISTORY.md")
    with open(out, "w") as fh:
        fh.write("\n".join(L))
    print(f"wrote {out}  ({len(msgs)} messages)")
    for i, (stamp, text) in enumerate(msgs, 1):
        print(f"  {i:>2}. {stamp}  {' '.join(text.split())[:78]}")


if __name__ == "__main__":
    main()
