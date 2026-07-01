#!/usr/bin/env python3
"""Rebuild public analysis tables and figures from released result files."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean, median, stdev
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


MODELS = ["Qwen/Qwen3-0.6B", "Qwen/Qwen3-1.7B", "Qwen/Qwen3-4B"]
TASKS = ["sst2", "mrpc", "rte", "trec"]
METHODS = ["ICL", "LoRA"]
SEEDS = [13, 21, 42]
ICL_BUDGETS = [0, 1, 2, 4, 8, 16, 32]
LORA_BUDGETS = [1, 2, 4, 8, 16, 32]
MATCHED_BUDGETS = [1, 2, 4, 8, 16, 32]
PRACTICAL_THRESHOLD = 0.02
BOOTSTRAP_N = 5000
BOOTSTRAP_SEED = 20260629


def fnum(value: Any, default: float = math.nan) -> float:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def inum(value: Any, default: int = 0) -> int:
    if value is None or value == "":
        return default
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def group_rows(rows: list[dict[str, Any]], keys: tuple[str, ...]) -> dict[tuple[Any, ...], list[dict[str, Any]]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[k] for k in keys)].append(row)
    return grouped


def pct(values: list[float], q: float) -> float:
    xs = sorted(values)
    if not xs:
        return math.nan
    pos = (len(xs) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return xs[lo]
    return xs[lo] * (hi - pos) + xs[hi] * (pos - lo)


def bootstrap_ci(values: list[float]) -> dict[str, float]:
    vals = [v for v in values if not math.isnan(v)]
    if not vals:
        return {"mean": math.nan, "ci_low": math.nan, "ci_high": math.nan}
    rng = random.Random(BOOTSTRAP_SEED + len(vals) + int(sum(vals) * 100000))
    boot = []
    for _ in range(BOOTSTRAP_N):
        sample = [vals[rng.randrange(len(vals))] for _ in vals]
        boot.append(mean(sample))
    return {"mean": mean(vals), "ci_low": pct(boot, 0.025), "ci_high": pct(boot, 0.975)}


def sign_flip_pvalue(values: list[float]) -> float:
    vals = [v for v in values if not math.isnan(v)]
    if not vals:
        return math.nan
    observed = abs(mean(vals))
    total = 2 ** len(vals)
    extreme = 0
    for mask in range(total):
        signed = [value if (mask >> i) & 1 else -value for i, value in enumerate(vals)]
        if abs(mean(signed)) >= observed - 1e-12:
            extreme += 1
    return extreme / total


def bh_adjust(rows: list[dict[str, Any]], p_key: str = "p_value") -> None:
    valid = [(i, float(row[p_key])) for i, row in enumerate(rows) if not math.isnan(float(row[p_key]))]
    valid.sort(key=lambda item: item[1])
    m = len(valid)
    adjusted = [math.nan] * len(rows)
    running = 1.0
    for rank_from_end, (idx, pval) in enumerate(reversed(valid), start=1):
        rank = m - rank_from_end + 1
        running = min(running, pval * m / rank)
        adjusted[idx] = min(running, 1.0)
    for i, row in enumerate(rows):
        row["p_bh"] = adjusted[i]


def aggregate_accuracy(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    out = []
    for key, items in sorted(group_rows(rows, ("model_id", "task", "method", "budget")).items()):
        accs = [fnum(r["accuracy_strict"]) for r in items]
        parsers = [fnum(r["parser_success"]) for r in items]
        runtimes = [fnum(r["runtime_sec"]) for r in items]
        out.append(
            {
                "method": key[2],
                "model_id": key[0],
                "task": key[1],
                "budget": int(key[3]),
                "n": len(items),
                "accuracy_mean": mean(accs),
                "accuracy_std": stdev(accs) if len(accs) > 1 else 0.0,
                "accuracy_min": min(accs),
                "accuracy_max": max(accs),
                "parser_success_min": min(parsers),
                "parser_success_mean": mean(parsers),
                "runtime_sec_mean": mean(runtimes),
                "runtime_sec_sum": sum(runtimes),
            }
        )
    return out


def paired_rows(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    by_cell = {(r["model_id"], r["task"], inum(r["budget"]), inum(r["seed"]), r["method"]): r for r in rows}
    out = []
    for model_id in MODELS:
        for task in TASKS:
            for budget in MATCHED_BUDGETS:
                for seed in SEEDS:
                    icl = by_cell.get((model_id, task, budget, seed, "ICL"))
                    lora = by_cell.get((model_id, task, budget, seed, "LoRA"))
                    if not icl or not lora:
                        continue
                    out.append(
                        {
                            "model_id": model_id,
                            "task": task,
                            "budget": budget,
                            "seed": seed,
                            "icl_accuracy": fnum(icl["accuracy_strict"]),
                            "lora_accuracy": fnum(lora["accuracy_strict"]),
                            "lora_minus_icl_accuracy": fnum(lora["accuracy_strict"]) - fnum(icl["accuracy_strict"]),
                            "icl_runtime_sec": fnum(icl["runtime_sec"]),
                            "lora_runtime_sec": fnum(lora["runtime_sec"]),
                            "lora_minus_icl_runtime_sec": fnum(lora["runtime_sec"]) - fnum(icl["runtime_sec"]),
                            "lora_train_runtime_sec": fnum(lora.get("train_runtime_sec")),
                            "lora_inference_runtime_sec": fnum(lora.get("inference_runtime_sec")),
                            "lora_model_load_sec": fnum(lora.get("model_load_sec")),
                        }
                    )
    return out


def interval_and_test_rows(paired: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    ci_rows = []
    test_rows = []
    for key, items in sorted(group_rows(paired, ("model_id", "task", "budget")).items()):
        values = [fnum(r["lora_minus_icl_accuracy"]) for r in items]
        ci = bootstrap_ci(values)
        direction = "LoRA" if ci["mean"] > PRACTICAL_THRESHOLD else "ICL" if ci["mean"] < -PRACTICAL_THRESHOLD else "tie"
        base = {
            "model_id": key[0],
            "task": key[1],
            "budget": int(key[2]),
            "n": len(values),
            "mean_lora_minus_icl": ci["mean"],
            "ci_low": ci["ci_low"],
            "ci_high": ci["ci_high"],
            "median_lora_minus_icl": median(values),
            "min_lora_minus_icl": min(values),
            "max_lora_minus_icl": max(values),
            "direction_by_threshold": direction,
        }
        ci_rows.append(base)
        test_rows.append({**base, "p_value": sign_flip_pvalue(values), "test": "exact_sign_flip_two_sided"})
    bh_adjust(test_rows)
    return ci_rows, test_rows


def crossover_table(ci_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    by_model_task: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in ci_rows:
        by_model_task[(row["model_id"], row["task"])].append(row)
    for (model_id, task), items in sorted(by_model_task.items()):
        pts = sorted(items, key=lambda r: r["budget"])
        cross = [r for r in pts if r["mean_lora_minus_icl"] >= PRACTICAL_THRESHOLD]
        best = max(pts, key=lambda r: r["mean_lora_minus_icl"])
        out.append(
            {
                "model_id": model_id,
                "task": task,
                "threshold": PRACTICAL_THRESHOLD,
                "crossover_budget": cross[0]["budget"] if cross else "",
                "best_budget": best["budget"],
                "best_mean_lora_minus_icl": best["mean_lora_minus_icl"],
                "interpretation": "LoRA crosses threshold" if cross else "No LoRA practical crossover in tested budgets",
            }
        )
    return out


def runtime_summary(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    out = []
    for key, items in sorted(group_rows(rows, ("model_id", "task", "method", "budget")).items()):
        runtimes = [fnum(r["runtime_sec"]) for r in items]
        train = [fnum(r.get("train_runtime_sec")) for r in items if not math.isnan(fnum(r.get("train_runtime_sec")))]
        infer = [fnum(r.get("inference_runtime_sec")) for r in items if not math.isnan(fnum(r.get("inference_runtime_sec")))]
        load = [fnum(r.get("model_load_sec")) for r in items if not math.isnan(fnum(r.get("model_load_sec")))]
        out.append(
            {
                "model_id": key[0],
                "task": key[1],
                "method": key[2],
                "budget": int(key[3]),
                "n": len(items),
                "runtime_sec_mean": mean(runtimes),
                "runtime_sec_sum": sum(runtimes),
                "train_runtime_sec_mean": mean(train) if train else "",
                "inference_runtime_sec_mean": mean(infer) if infer else (mean(runtimes) if key[2] == "ICL" else ""),
                "model_load_sec_mean": mean(load) if load else "",
            }
        )
    return out


def parser_summary(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    out = []
    for key, items in sorted(group_rows(rows, ("model_id", "task", "method")).items()):
        parsers = [fnum(r["parser_success"]) for r in items]
        failures = [inum(r["parser_failures"]) for r in items]
        out.append(
            {
                "model_id": key[0],
                "task": key[1],
                "method": key[2],
                "n": len(items),
                "parser_success_min": min(parsers),
                "parser_success_mean": mean(parsers),
                "parser_failure_total": sum(failures),
            }
        )
    return out


def seed_variance_summary(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    out = []
    for key, items in sorted(group_rows(rows, ("model_id", "task", "method", "budget")).items()):
        accs = [fnum(r["accuracy_strict"]) for r in items]
        out.append(
            {
                "model_id": key[0],
                "task": key[1],
                "method": key[2],
                "budget": int(key[3]),
                "n": len(accs),
                "accuracy_std_across_seeds": stdev(accs) if len(accs) > 1 else 0.0,
                "accuracy_range_across_seeds": max(accs) - min(accs),
            }
        )
    return out


def write_latex_tables(table_dir: Path, crossover: list[dict[str, Any]], tests: list[dict[str, Any]], runtime: list[dict[str, Any]]) -> None:
    table_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        r"\begin{tabular}{llrrr}",
        r"\toprule",
        r"Model & Task & Cross. budget & Best budget & Best $\Delta$ \\",
        r"\midrule",
    ]
    for row in crossover:
        model = row["model_id"].split("/")[-1]
        cross = row["crossover_budget"] if row["crossover_budget"] != "" else r"--"
        lines.append(f"{model} & {row['task'].upper()} & {cross} & {row['best_budget']} & {float(row['best_mean_lora_minus_icl']):.4f} \\\\")
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    (table_dir / "table_crossover_boundary.tex").write_text("\n".join(lines), encoding="utf-8")

    selected = [
        ("Qwen/Qwen3-1.7B", "rte", 16),
        ("Qwen/Qwen3-0.6B", "sst2", 1),
        ("Qwen/Qwen3-0.6B", "rte", 32),
        ("Qwen/Qwen3-4B", "mrpc", 32),
        ("Qwen/Qwen3-0.6B", "trec", 8),
        ("Qwen/Qwen3-1.7B", "trec", 32),
        ("Qwen/Qwen3-4B", "sst2", 32),
    ]
    by_key = {(r["model_id"], r["task"], int(r["budget"])): r for r in tests}
    lines = [
        r"\begin{tabular}{lllrlrr}",
        r"\toprule",
        r"Model & Task & Budget & $\Delta$ & Interval & Direction & BH $p$ \\",
        r"\midrule",
    ]
    for key in selected:
        row = by_key[key]
        model = row["model_id"].split("/")[-1]
        interval = f"[{float(row['ci_low']):.4f}, {float(row['ci_high']):.4f}]"
        lines.append(f"{model} & {row['task'].upper()} & {row['budget']} & {float(row['mean_lora_minus_icl']):.4f} & {interval} & {row['direction_by_threshold']} & {float(row['p_bh']):.4f} \\\\")
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    (table_dir / "table_representative_diagnostics.tex").write_text("\n".join(lines), encoding="utf-8")

    totals: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for row in runtime:
        method = row["method"]
        totals[method]["runtime"] += float(row["runtime_sec_sum"])
        if method == "LoRA":
            n = int(row["n"])
            train = fnum(row["train_runtime_sec_mean"], 0.0)
            infer = fnum(row["inference_runtime_sec_mean"], 0.0)
            load = fnum(row["model_load_sec_mean"], 0.0)
            totals[method]["train"] += train * n
            totals[method]["infer"] += infer * n
            totals[method]["load"] += load * n
        else:
            totals[method]["infer"] += float(row["runtime_sec_sum"])
    lines = [
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Method & Total & Training comp. & Inference comp. & Load/overhead comp. \\",
        r"\midrule",
        f"ICL & {totals['ICL']['runtime']:.2f} & -- & {totals['ICL']['infer']:.2f} & -- \\\\",
        f"LoRA & {totals['LoRA']['runtime']:.2f} & {totals['LoRA']['train']:.2f} & {totals['LoRA']['infer']:.2f} & {totals['LoRA']['load']:.2f} \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        "",
    ]
    (table_dir / "table_runtime_summary.tex").write_text("\n".join(lines), encoding="utf-8")


def plot_accuracy_curves(acc_rows: list[dict[str, Any]], fig_dir: Path) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    for task in TASKS:
        for model_id in MODELS:
            fig, ax = plt.subplots(figsize=(5.8, 3.35))
            for method, color, marker, linestyle in [
                ("ICL", "#2563eb", "o", "-"),
                ("LoRA", "#dc2626", "s", "--"),
            ]:
                points = [
                    (int(r["budget"]), float(r["accuracy_mean"]))
                    for r in acc_rows
                    if r["task"] == task and r["model_id"] == model_id and r["method"] == method
                ]
                points.sort()
                if points:
                    ax.plot([x for x, _ in points], [y for _, y in points], label=method, color=color, marker=marker, linestyle=linestyle, linewidth=1.8, markersize=4)
            ax.set_title(f"{task.upper()}, {model_id.split('/')[-1]}", fontsize=11)
            ax.set_xlabel("Label budget", fontsize=9)
            ax.set_ylabel("Accuracy", fontsize=9)
            ax.set_xticks(ICL_BUDGETS)
            ax.grid(True, which="major", alpha=0.28, linewidth=0.55)
            ax.legend(fontsize=8, loc="best")
            fig.tight_layout()
            out_name = f"accuracy_curve_{task}_{model_id.split('/')[-1].replace('.', '_')}.png"
            fig.savefig(fig_dir / out_name, dpi=300, bbox_inches="tight", facecolor="white")
            plt.close(fig)


def plot_heatmap(path: Path, title: str, rows: list[str], cols: list[str], values: dict[tuple[str, str], float], note: str) -> None:
    matrix = [[values.get((row, col), math.nan) for col in cols] for row in rows]
    fig, ax = plt.subplots(figsize=(max(5, 0.65 * len(cols) + 2.4), max(4, 0.34 * len(rows) + 1.8)))
    finite = [v for line in matrix for v in line if not math.isnan(v)]
    vmax = max(abs(v) for v in finite) if finite else 1.0
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_title(title, fontsize=11)
    ax.set_xticks(range(len(cols)), cols, fontsize=8)
    ax.set_yticks(range(len(rows)), rows, fontsize=8)
    for i, row in enumerate(matrix):
        for j, value in enumerate(row):
            label = "" if math.isnan(value) else (f"{int(value)}" if title.lower().startswith("crossover") else f"{value:.3f}")
            ax.text(j, i, label, ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.03)
    fig.text(0.5, 0.02, note, ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def make_figures(acc: list[dict[str, Any]], ci_rows: list[dict[str, Any]], cross: list[dict[str, Any]], runtime: list[dict[str, Any]], parser_rows: list[dict[str, Any]], fig_dir: Path) -> None:
    plot_accuracy_curves(acc, fig_dir)
    rows = [f"{m.split('/')[-1]} {t}" for m in MODELS for t in TASKS]
    budget_cols = [str(b) for b in MATCHED_BUDGETS]
    diff_values = {(f"{r['model_id'].split('/')[-1]} {r['task']}", str(r["budget"])): float(r["mean_lora_minus_icl"]) for r in ci_rows}
    plot_heatmap(fig_dir / "lora_minus_icl_mean_heatmap.png", "Mean LoRA minus ICL accuracy", rows, budget_cols, diff_values, "Green favors LoRA; red favors ICL.")
    cross_values = {(f"{r['model_id'].split('/')[-1]} {r['task']}", "crossover"): float(r["crossover_budget"]) if r["crossover_budget"] != "" else math.nan for r in cross}
    plot_heatmap(fig_dir / "crossover_budget_heatmap.png", "Crossover budget at threshold 0.02", rows, ["crossover"], cross_values, "Blank cells have no LoRA practical crossover.")
    runtime_values = {(f"{r['model_id'].split('/')[-1]} {r['task']}", f"{r['method']} n{r['budget']}"): float(r["runtime_sec_mean"]) for r in runtime if r["budget"] in (1, 8, 32)}
    plot_heatmap(fig_dir / "runtime_mean_heatmap.png", "Mean runtime per job, seconds", rows, [f"{m} n{b}" for m in METHODS for b in (1, 8, 32)], runtime_values, "LoRA includes load, train, and inference time.")
    parser_values = {(f"{r['model_id'].split('/')[-1]} {r['task']}", r["method"]): float(r["parser_success_min"]) for r in parser_rows}
    plot_heatmap(fig_dir / "parser_success_min_heatmap.png", "Minimum parser success", rows, METHODS, parser_values, "Recovered parser-success diagnostic.")


def headline_patterns(ci_rows: list[dict[str, Any]], cross: list[dict[str, Any]], runtime: list[dict[str, Any]], parser_rows: list[dict[str, Any]]) -> dict[str, Any]:
    runtime_by_method: dict[str, float] = defaultdict(float)
    for row in runtime:
        runtime_by_method[row["method"]] += float(row["runtime_sec_sum"])
    return {
        "practical_threshold": PRACTICAL_THRESHOLD,
        "strongest_lora_cells": sorted(ci_rows, key=lambda r: r["mean_lora_minus_icl"], reverse=True)[:8],
        "strongest_icl_cells": sorted(ci_rows, key=lambda r: r["mean_lora_minus_icl"])[:8],
        "crossover_cells": [r for r in cross if r["crossover_budget"] != ""],
        "no_crossover_cells": [r for r in cross if r["crossover_budget"] == ""],
        "runtime_sec_sum_by_method": dict(runtime_by_method),
        "parser_worst_cells": sorted(parser_rows, key=lambda r: r["parser_success_min"])[:8],
    }


def validate_inputs(rows: list[dict[str, str]], summary: dict[str, Any]) -> None:
    if len(rows) != 468:
        raise AssertionError(f"expected 468 rows, got {len(rows)}")
    if summary.get("status") != "PASS":
        raise AssertionError("summary status is not PASS")
    if any(row["status"] != "PASS" for row in rows):
        raise AssertionError("at least one result row is not PASS")
    if sorted({row["model_id"] for row in rows}) != sorted(MODELS):
        raise AssertionError("unexpected model coverage")
    if sorted({row["task"] for row in rows}) != sorted(TASKS):
        raise AssertionError("unexpected task coverage")
    if sorted({inum(row["seed"]) for row in rows}) != SEEDS:
        raise AssertionError("unexpected seed coverage")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, default=Path(__file__).resolve().parents[1] / "data")
    parser.add_argument("--output-root", type=Path, default=Path("output"))
    args = parser.parse_args()

    input_root = args.input_root
    out_data = args.output_root / "data"
    out_analysis = out_data / "analysis"
    out_tables = out_data / "tables"
    out_figures = args.output_root / "figures"

    rows = read_csv(input_root / "results.csv")
    summary = json.loads((input_root / "summary.json").read_text(encoding="utf-8"))
    validate_inputs(rows, summary)

    acc = aggregate_accuracy(rows)
    paired = paired_rows(rows)
    ci_rows, test_rows = interval_and_test_rows(paired)
    cross = crossover_table(ci_rows)
    runtime = runtime_summary(rows)
    parser_rows = parser_summary(rows)
    seed_rows = seed_variance_summary(rows)
    patterns = headline_patterns(ci_rows, cross, runtime, parser_rows)

    write_csv(out_data / "results_aggregated.csv", acc)
    write_csv(out_data / "paired_differences.csv", paired)
    for name, table in [
        ("per_cell_accuracy_summary", acc),
        ("paired_lora_minus_icl", paired),
        ("bootstrap_lora_minus_icl_ci", ci_rows),
        ("paired_tests_bh", test_rows),
        ("crossover_budget", cross),
        ("runtime_summary", runtime),
        ("parser_success_summary", parser_rows),
        ("seed_variance_summary", seed_rows),
    ]:
        write_csv(out_analysis / f"{name}.csv", table)
        write_json(out_analysis / f"{name}.json", table)
    write_json(out_analysis / "headline_patterns.json", patterns)
    write_latex_tables(out_tables, cross, test_rows, runtime)
    make_figures(acc, ci_rows, cross, runtime, parser_rows, out_figures)

    print(
        json.dumps(
            {
                "status": "PASS",
                "input_rows": len(rows),
                "paired_rows": len(paired),
                "crossover_cells": len(patterns["crossover_cells"]),
                "analysis_files": len(list(out_analysis.glob("*"))),
                "table_files": len(list(out_tables.glob("*.tex"))),
                "figure_files": len(list(out_figures.glob("*.png"))),
                "output_root": str(args.output_root),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
