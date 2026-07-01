# Few-Shot Prompting versus Low-Rank Adaptation for Low-Label Text Classification

Code and data for the paper *Few-Shot Prompting versus Low-Rank Adaptation for Low-Label Text Classification: A Controlled Boundary Study*.

**Authors:** Yaowen Sun, Jianting Gao, Xin Zhang

## Overview

- **`src/`** - Analysis-table, figure-generation, and public-data verification scripts.
- **`data/`** - Released result tables from the recovered validation/development matrix.
- **`data/analysis/`** - Derived crossover, interval, diagnostic-test, parser, runtime, and per-cell summary tables.
- **`data/tables/`** - Generated LaTeX table snippets corresponding to the released analysis outputs.
- **`figures/`** - Publication figures generated from the released tables.
- **`environment.json`** - Tested local environment summary.

This repository is intended to reproduce the analysis tables, publication
figures, row counts, and summary checks from sanitized CSV/JSON outputs. It
does not include raw job directories, model checkpoints, training caches, or
downloaded benchmark datasets.

## Experiment Design

| Dimension | Levels |
|---|---|
| Models | Qwen/Qwen3-0.6B, Qwen/Qwen3-1.7B, Qwen/Qwen3-4B |
| Tasks | SST-2, MRPC, RTE, TREC |
| Methods | In-context learning (ICL), LoRA |
| ICL budgets | 0, 1, 2, 4, 8, 16, 32 |
| LoRA budgets | 1, 2, 4, 8, 16, 32 |
| Seeds | 13, 21, 42 |
| Evaluation split | Validation/development splits only |
| Precision | bf16; fp16 not used |

The released matrix contains 468 completed rows and 0 failed jobs. LoRA has no budget-0 condition.

## Key Results

- LoRA crosses a practical 0.02 mean-accuracy threshold in 4 of 12 model-task cells.
- The remaining 8 model-task cells show no LoRA crossover within the tested budgets.
- Strongest LoRA-favored cell: Qwen/Qwen3-1.7B on RTE at budget 16, mean LoRA-ICL accuracy 0.1237, interval [0.1016, 0.1367].
- Strongest ICL-favored cell: Qwen/Qwen3-0.6B on TREC at budget 8, mean LoRA-ICL accuracy -0.3398, interval [-0.3789, -0.2930].
- No paired diagnostic reaches BH-adjusted p<0.1.
- Minimum recovered ICL parser success is 0.99609375; 63 SST-2 ICL metrics were replaced during parser recovery.

## Reproducing Verification from Released Data

```bash
python src/verify_results.py
```

Expected output:

```text
PASS: key paper claims verified from released artifacts
```

The verification script checks matrix size, job status, model/task/seed coverage, budget counts, parser-success floor, crossover count, headline intervals, BH-adjusted diagnostic p-values, SST-2 ICL recovery count, and method-level runtime totals.

## Reproducing Analysis Tables and Figures

Install the public analysis dependencies, then rebuild derived tables and
figures from `data/results.csv` and `data/summary.json`:

```bash
python -m pip install -r requirements.txt
python src/analyze_public_data.py --output-root output
```

Expected output includes `status: PASS`, 468 input rows, 216 paired rows, 4
crossover cells, regenerated CSV/JSON analysis tables, LaTeX table snippets,
and PNG figures under `output/`.

To regenerate only the two manuscript curve figures in-place from the released
per-cell accuracy summary, run:

```bash
python src/generate_publication_figures.py
```

## Data Files

- `data/results.csv` - Per-run recovered result table.
- `data/results_aggregated.csv` - Aggregated accuracy, parser-success, and runtime summaries.
- `data/paired_differences.csv` - Matched-seed LoRA minus ICL comparisons.
- `data/summary.json` - Matrix-level integrity summary.
- `data/analysis/crossover_budget.csv` - Threshold-defined crossover table.
- `data/analysis/bootstrap_lora_minus_icl_ci.csv` - Descriptive intervals.
- `data/analysis/paired_tests_bh.csv` - Diagnostic paired tests and BH adjustment.
- `data/analysis/runtime_summary.csv` - Local runtime summary.
- `data/tables/` - Generated LaTeX table snippets for crossover, diagnostics, and runtime summaries.

## Hardware & Environment

| Component | Specification |
|---|---|
| GPU | NVIDIA RTX PRO 6000 (Blackwell), bf16 |
| Python | 3.11.15 |
| PyTorch | 2.12.1+cu130 |
| Transformers | 5.12.1 |
| PEFT | 0.19.1 |
| Datasets | 5.0.0 |
| Precision | bf16 |

## Reproducing Experiments from Scratch

Full execution requires access to the Qwen3 checkpoints, benchmark datasets, a
CUDA-capable GPU environment with bf16 support, and enough local storage for
model loading and LoRA adapter runs. The public package is designed for
analysis-level reproducibility from released aggregate artifacts, not for
re-running the full model matrix.

The executed matrix used:

- Qwen/Qwen3-0.6B, Qwen/Qwen3-1.7B, and Qwen/Qwen3-4B checkpoints
- SST-2, MRPC, RTE, and TREC validation/development splits
- ICL budgets 0, 1, 2, 4, 8, 16, and 32
- LoRA budgets 1, 2, 4, 8, 16, and 32
- Seeds 13, 21, and 42
- LoRA rank 4, alpha 8, dropout 0.0, target modules `q_proj` and `v_proj`
- bf16 precision; fp16 was not used

## Requirements

See `requirements.txt` for public analysis dependencies and
`environment.json` for the tested environment manifest. Full model execution
requires a CUDA build of PyTorch and local access to the model checkpoints and
datasets.

## License

MIT License. See [LICENSE](LICENSE).

## Citation

```bibtex
@article{sun2026icl_lora_boundary,
  title={Few-Shot Prompting versus Low-Rank Adaptation for Low-Label Text Classification: A Controlled Boundary Study},
  author={Sun, Yaowen and Gao, Jianting and Zhang, Xin},
  year={2026}
}
```
