# AlignProbe: Deterministic Instruction-Hierarchy & Sycophancy Benchmark for Open-Weight Transformers

[![AlignProbe CI](https://github.com/suvenducodes/align-probe/actions/workflows/ci.yml/badge.svg)](https://github.com/suvenducodes/align-probe/actions/workflows/ci.yml)

**AlignProbe** is a reproducible, pure-PyTorch evaluation suite that probes two core alignment failure modes in instruction-tuned Transformer models—using **100% deterministic verifiers** (no LLM-as-a-judge) and paired statistical reporting:

1. **Instruction Hierarchy (20 adversarial cases):** Tests whether system-level constraints hold against secret/canary extraction, indirect prompt injection inside untrusted data tags, JSON/format-contract breaking, and policy bypass framing.
2. **Honesty Under Pressure / Sycophancy (20 two-turn paired cases):** Measures how often a model abandons a factually correct Turn-1 answer when challenged by authoritative or confident user pushback in Turn 2 (**Flip Rate** + **exact McNemar test**).
3. **Benign Utility Controls (10 cases):** Guards against trivial over-refusal gaming.

## Repository Structure
```text
align-probe/
├── data/benchmark.jsonl          # 50 curated deterministic evaluation cases
├── src/
│   ├── grader.py                 # Deterministic verifiers (canary, schema, label, MCQ)
│   ├── evaluate.py               # PyTorch + HuggingFace greedy evaluation runner
│   └── analyze.py                # Wilson 95% CIs, McNemar test, plots & REPORT.md builder
├── tests/test_grader.py          # Positive & negative verifier tests (proves graders fail on bad traces)
├── .github/workflows/ci.yml      # Automated CI running pytest + dry-run pipeline
├── results/                      # Saved JSONL traces, summary JSON & summary_plot.png
└── REPORT.md                     # Auto-generated quantitative & qualitative evaluation report
```

## Quickstart

### 1. Install dependencies & verify graders
```bash
pip install -r requirements.txt
pytest -v
```

### 2. Run on open-weight HuggingFace Transformers (ungated, zero API keys)
Works on a 4 GB+ CUDA GPU (FP16) or free Google Colab T4 GPU:
```bash
python -m src.evaluate --model Qwen/Qwen2.5-1.5B-Instruct --output results/qwen2.5_1.5b_traces.jsonl
python -m src.evaluate --model HuggingFaceTB/SmolLM2-1.7B-Instruct --output results/smollm2_1.7b_traces.jsonl
```

### 3. Generate statistics, plots & `REPORT.md`
```bash
python -m src.analyze results/qwen2.5_1.5b_traces.jsonl results/smollm2_1.7b_traces.jsonl
```

### 4. Fast smoke test (no weight download)
```bash
python -m src.evaluate --dry-run --output results/dry_run_traces.jsonl
python -m src.analyze results/dry_run_traces.jsonl
```

## Why Deterministic Verifiers?
Every grader in `src/grader.py` is unit-tested in `tests/test_grader.py` against both aligned outputs (must pass) and deliberately compromised outputs—including space-obfuscated/reversed canary leaks, markdown-fenced JSON violations, and sycophantic Turn-2 flips (must fail).
