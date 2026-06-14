# Revision experiments — reproducibility

All experiments added during the revision live here. A self-contained virtual
environment is provided (`venv/`, Python 3.10, `anthropic`, `requests`,
`scipy`, `numpy`). Random seeds are fixed (`20260614`).

## 1. Large-scale Wikidata validation + type-aware entity linker
Addresses reviewer points R#1.1, R#1.3, R#4.2, R#5 (entity linking).

```bash
./venv/bin/python validate_corpus.py --per-file 22 --per-model 420 --link-eval 180
```
Validates the existing multi-model corpus (Claude Sonnet 3.7, GPT-4.1, Grok 3)
against Wikidata using a type-aware entity linker, writing:
- `results/per_fact.csv`     — per-fact verdicts (naive vs. type-aware linker)
- `results/summary.json`     — per-model accuracy, coverage, linking P/R/F1
- `results/linking_gold.json`— the entity-linking gold set
All Wikidata lookups are cached under `cache/` (resumable, reproducible).

Then derive the manuscript numbers and the LaTeX macro file:
```bash
./venv/bin/python compute_stats.py      # writes ../revised_manuscript/numbers.tex
```

## 2. Expert-system reasoning (Prolog inference)
Addresses reviewer point R#4.4.
```bash
cd reasoning && swipl -q -g main -t halt rules.pl
```
Loads an LLM-extracted knowledge base, adds a domain-independent inference-rule
layer, and runs multi-hop deduction, negation-as-failure and aggregation
queries. Transcript saved to `reasoning/results_reasoning.txt`.

## 3. Reference validation (zero-hallucination check)
```bash
./venv/bin/python validate_refs.py
```
Mechanically checks every reference added in the revision against Crossref /
arXiv. Entries without a resolvable identifier are flagged, never fabricated.

## 4. Multi-model Claude effort sweep (ready to run; needs API credits)
Addresses the comparison requested across Claude models and effort levels, and
the empirical cost/latency request (R#5.3).
```bash
./venv/bin/python run_claude_models.py --dry-run            # offline wiring check
./venv/bin/python run_claude_models.py --topics-per-config 8   # live run
```
Generates knowledge bases with Claude Haiku 4.5, Sonnet 4.6 and Opus 4.8
(the latter at low and high reasoning effort), instruments per-call latency and
token cost, and re-validates each configuration against Wikidata.

> NOTE: at the time of revision the available `ANTHROPIC_API_KEY` returned
> `invalid_request_error: credit balance is too low`, so the live sweep could
> not be executed. The harness is complete and verified offline (`--dry-run`);
> a funded key reproduces the full model x effort comparison with one command.
