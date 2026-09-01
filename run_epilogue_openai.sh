#!/usr/bin/env bash
# Epilogue: historical OpenAI vision models, through the OpenAI key directly.
set -u
cd "$(dirname "$0")"
python3 03_llm/openrouter_solver.py --route openai --workers 8 \
  --models gpt-4-turbo gpt-4o gpt-4.1 o3 2>&1
