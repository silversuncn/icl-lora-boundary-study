#!/usr/bin/env python3
"""Verify key paper claims from released CSV/JSON artifacts."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
ANALYSIS = DATA / "analysis"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def assert_close(name: str, actual: float, expected: float, tol: float = 1e-9) -> None:
    if abs(actual - expected) > tol:
        raise AssertionError(f"{name}: expected {expected}, got {actual}")


def main() -> None:
    results = read_csv(DATA / "results.csv")
    paired = read_csv(DATA / "paired_differences.csv")
    crossover = read_csv(ANALYSIS / "crossover_budget.csv")
    threshold = read_csv(ANALYSIS / "threshold_sensitivity.csv")
    ci = read_csv(ANALYSIS / "bootstrap_lora_minus_icl_ci.csv")
    tests = read_csv(ANALYSIS / "paired_tests_bh.csv")
    summary = load_json(DATA / "summary.json")

    assert len(results) == 780
    assert len(paired) == 360
    assert summary["completed_jobs"] == 780
    assert summary["failed_jobs"] == 0
    assert all(row["status"] == "PASS" for row in results)

    models = {row["model_id"] for row in results}
    tasks = {row["task"] for row in results}
    seeds = {int(row["seed"]) for row in results}
    assert models == {"Qwen/Qwen3-0.6B", "Qwen/Qwen3-1.7B", "Qwen/Qwen3-4B"}
    assert tasks == {"sst2", "mrpc", "rte", "trec"}
    assert seeds == {7, 13, 21, 42, 55}

    method_budget_counts = Counter((row["method"], int(row["budget"])) for row in results)
    for budget in [0, 1, 2, 4, 8, 16, 32]:
        assert method_budget_counts[("ICL", budget)] == 60
    for budget in [1, 2, 4, 8, 16, 32]:
        assert method_budget_counts[("LoRA", budget)] == 60
    assert ("LoRA", 0) not in method_budget_counts

    assert_close(
        "minimum recovered ICL parser success",
        min(float(row["parser_success"]) for row in results if row["method"] == "ICL"),
        0.9921875,
    )

    crossover_count = sum(1 for row in crossover if row["crossover_budget"])
    assert crossover_count == 4
    assert len(crossover) - crossover_count == 8

    threshold_counts = {
        float(row["threshold"]): int(row["crossover_cell_count_at_threshold"])
        for row in threshold[::12]
    }
    assert threshold_counts == {0.0: 4, 0.01: 4, 0.02: 4, 0.03: 4, 0.05: 2, 0.1: 1}

    ci_by_cell = {
        (row["model_id"], row["task"], int(row["budget"])): row
        for row in ci
    }
    strong_lora = ci_by_cell[("Qwen/Qwen3-1.7B", "rte", 8)]
    assert_close("strongest LoRA-favored mean", float(strong_lora["mean_lora_minus_icl"]), 0.10078125)
    assert_close("strongest LoRA-favored CI low", float(strong_lora["ci_low"]), 0.0859375)
    assert_close("strongest LoRA-favored CI high", float(strong_lora["ci_high"]), 0.1140625)

    strong_icl = ci_by_cell[("Qwen/Qwen3-0.6B", "trec", 32)]
    assert_close("strongest ICL-favored mean", float(strong_icl["mean_lora_minus_icl"]), -0.32265625)
    assert_close("strongest ICL-favored CI low", float(strong_icl["ci_low"]), -0.3703125)
    assert_close("strongest ICL-favored CI high", float(strong_icl["ci_high"]), -0.26875)

    assert_close("minimum BH-adjusted diagnostic p-value", min(float(row["p_bh"]) for row in tests), 0.09183673469387756)
    assert all(float(row["p_bh"]) >= 0.09 for row in tests)

    sst2_icl_rows = [
        row for row in results
        if row["task"] == "sst2" and row["method"] == "ICL"
    ]
    assert len(sst2_icl_rows) == 105

    total_runtime_by_method = Counter()
    for row in results:
        total_runtime_by_method[row["method"]] += float(row["runtime_sec"])
    assert_close("ICL runtime total", total_runtime_by_method["ICL"], 12019.97, tol=0.01)
    assert_close("LoRA runtime total", total_runtime_by_method["LoRA"], 15056.94, tol=0.01)

    print("PASS: five-seed key paper claims verified from released artifacts")


if __name__ == "__main__":
    main()
