#!/usr/bin/env bash
# Full 96-problem runs for the LLM comparison.
set -u
cd "$(dirname "$0")"
for m in gpt-5.6-sol gpt-5.6-terra gpt-5; do
  echo "=================== $m ==================="
  python3 03_llm/solver.py --model "$m" --workers 10 2>&1
done
