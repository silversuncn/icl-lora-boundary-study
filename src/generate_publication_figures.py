#!/usr/bin/env python3
"""Regenerate publication figures from released analysis tables."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "data" / "analysis"
FIGURES = ROOT / "figures"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def draw_accuracy_curve(
    acc_rows: list[dict[str, str]],
    task: str,
    model_id: str,
    out_name: str,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(5.8, 3.35))
    styles = {
        "ICL": {"color": "#2563eb", "marker": "o", "linestyle": "-"},
        "LoRA": {"color": "#dc2626", "marker": "s", "linestyle": "--"},
    }
    for method in ("ICL", "LoRA"):
        points = [
            (int(r["budget"]), float(r["accuracy_mean"]))
            for r in acc_rows
            if r["task"] == task and r["model_id"] == model_id and r["method"] == method
        ]
        points.sort()
        ax.plot(
            [x for x, _ in points],
            [y for _, y in points],
            label=method,
            linewidth=1.8,
            markersize=4.0,
            **styles[method],
        )

    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Label budget", fontsize=9)
    ax.set_ylabel("Accuracy", fontsize=9)
    ax.set_xticks([0, 1, 2, 4, 8, 16, 32])
    ax.tick_params(labelsize=8)
    ax.grid(True, which="major", alpha=0.28, linewidth=0.55)
    ax.legend(fontsize=8, frameon=True, framealpha=0.92, edgecolor="#d1d5db", loc="best")
    for spine in ax.spines.values():
        spine.set_color("#9ca3af")
        spine.set_linewidth(0.6)

    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / out_name, dpi=450, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    acc_rows = read_csv(ANALYSIS / "per_cell_accuracy_summary.csv")
    draw_accuracy_curve(
        acc_rows,
        "rte",
        "Qwen/Qwen3-1.7B",
        "accuracy_curve_rte_Qwen3-1_7B.png",
        "RTE, Qwen3-1.7B",
    )
    draw_accuracy_curve(
        acc_rows,
        "trec",
        "Qwen/Qwen3-0.6B",
        "accuracy_curve_trec_Qwen3-0_6B.png",
        "TREC, Qwen3-0.6B",
    )
    print("generated_publication_figures=2")


if __name__ == "__main__":
    main()
