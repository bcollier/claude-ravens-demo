# Raw results

Everything in `COMPARISON.md` is generated from these files by
`scripts/compare.py`. Regenerate with `python scripts/compare.py`.

| file | produced by | contents |
|---|---|---|
| `original_answers.csv` | `run_original.py` | the 2017 agent's answer per problem (`-1` = declined) |
| `original_problem_results.csv` | the 2017 course grader | per-problem correct / incorrect / skipped |
| `original_set_results.csv` | the 2017 course grader | per-set totals |
| `original_stdout.log` | `run_original.py` | the agent's own trace, including its elimination decisions |
| `original_runtime.txt` | `run_original.py` | wall clock for all 96 |
| `original_typo_experiment.txt` | `01_original_2017/typo_experiment.py` | score with each scoring bug fixed |
| `classical_answers.csv` | `02_classical_ai/solver.py` | answers with and without the learned ranker |
| `classical_summary.txt` | `02_classical_ai/solver.py` | the four accuracy estimates and the per-fold hyper-parameters |
| `classical_ceiling.txt` | `02_classical_ai/diagnose.py` | how much of the ceiling is rule generation vs rule selection |
| `classical_coef.npy` | `02_classical_ai/solver.py` | learned weight per feature (order: `features.FEATURE_NAMES`) |
| `llm_<model>_answers.csv` | `03_llm/solver.py` | answer, confidence, latency, tokens and the rule the model stated |
| `llm_<model>_summary.txt` | `03_llm/solver.py` | accuracy and wall clock for the sweep |
| `llm_runs.log` | `run_llm_all.sh` | full console trace of every LLM run |

The `Rule` column of the LLM CSVs is worth reading on its own — it is the model's
own one-sentence account of the pattern, including on the problems it got wrong.
