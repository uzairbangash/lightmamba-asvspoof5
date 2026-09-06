#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_ROOTS = [
    Path("/media/uzair/Data/asvspoof5_exp_result"),
    Path("/home/uzair/Desktop/baseline/asvspoof5/AST: Audio Spectrogram Transformer"),
]

DISPLAY_NAMES = {
    "WavLM_Large_2Layer_Light_Mamba_Kernel7_Expansion2_With_Augmentation":
        "WavLM-large + 2L Light Mamba k7 e2",
    "WavLM_Large_2Layer_UltraLight_Mamba_Kernel5_Expansion1_With_Augmentation":
        "WavLM-large + 2L Ultra-Light Mamba k5 e1",
    "WavLM_Large_2Layer_Light_Mamba_Kernel5_Expansion2_With_Augmentation":
        "WavLM-large + 2L Light Mamba k5 e2",
    "WavLM_Large_Light_Transformer_With_Augmentation":
        "WavLM-large + Light Transformer",
    "WavLM_Large_3Layer_Light_Mamba_Kernel7_Expansion2_With_Augmentation":
        "WavLM-large + 3L Light Mamba k7 e2",
    "WavLM_Large_Light_Mamba_With_Augmentation":
        "WavLM-large + 4L Light Mamba k7 e2",
    "WavLM_Large_Official_Mamba_With_Augmentation":
        "WavLM-large + 4L official mamba-ssm",
}


def safe_name(text: str) -> str:
    text = text.lower().replace("+", "plus")
    text = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    return text


def iter_final_summaries(roots: list[Path]) -> list[Path]:
    paths: set[Path] = set()
    for root in roots:
        if not root.exists():
            continue
        if root.name == "asvspoof5_exp_result":
            paths.update(root.glob("*/artifacts/final_summary.json"))
        else:
            paths.update(root.glob("*/exp_result/*/artifacts/final_summary.json"))
    return sorted(paths)


def load_row(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text())
    except Exception:
        return None

    eval_metrics = data.get("final_eval_metrics", {})
    dev_metrics = data.get("final_dev_metrics", {})
    required = {"tp", "tn", "fp", "fn", "eer"}
    if not required.issubset(eval_metrics) or not required.issubset(dev_metrics):
        return None

    exp_name = path.parent.parent.name
    display = DISPLAY_NAMES.get(exp_name, exp_name.replace("_", " "))
    calibrated = eval_metrics.get("calibrated", {})
    f1 = calibrated.get("f1_bonafide", eval_metrics.get("f1_bonafide", float("nan")))

    def split_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
        return {
            "eer": float(metrics["eer"]) * 100.0,
            "dcf": float(metrics.get("dcf", float("nan"))),
            "tn": int(metrics["tn"]),
            "fp": int(metrics["fp"]),
            "fn": int(metrics["fn"]),
            "tp": int(metrics["tp"]),
        }

    return {
        "path": path,
        "experiment": exp_name,
        "display": display,
        "best_epoch": data.get("best_epoch"),
        "dev_eer": float(data.get("best_dev_eer", float("nan"))) * 100.0,
        "eval_eer": float(eval_metrics["eer"]) * 100.0,
        "eval_dcf": float(eval_metrics.get("dcf", float("nan"))),
        "f1": float(f1) * 100.0,
        "dev": split_metrics(dev_metrics),
        "eval": split_metrics(eval_metrics),
    }


def annotate_cell(ax, col: int, row: int, count: int, percent: float, max_value: int) -> None:
    color = "white" if count > max_value * 0.55 else "#111827"
    ax.text(
        col,
        row - 0.08,
        f"{count:,}",
        ha="center",
        va="center",
        fontsize=15,
        fontweight="bold",
        color=color,
    )
    ax.text(
        col,
        row + 0.18,
        f"{percent:.2f}%",
        ha="center",
        va="center",
        fontsize=12,
        color=color,
    )


def draw_matrix(row: dict[str, Any], split: str, output_path: Path) -> None:
    metrics = row[split]
    matrix = np.array([[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]], dtype=float)
    row_sums = matrix.sum(axis=1, keepdims=True)
    percents = np.divide(matrix, row_sums, out=np.zeros_like(matrix), where=row_sums > 0) * 100.0

    fig, ax = plt.subplots(figsize=(7.2, 6.2), dpi=220)
    image = ax.imshow(matrix, cmap="Blues")

    for y in range(2):
        for x in range(2):
            annotate_cell(ax, x, y, int(matrix[y, x]), float(percents[y, x]), int(matrix.max()))

    ax.set_xticks([0, 1], labels=["Predicted spoof", "Predicted bonafide"], fontsize=12)
    ax.set_yticks([0, 1], labels=["Actual spoof", "Actual bonafide"], fontsize=12)
    ax.set_xlabel("Predicted", fontsize=13, fontweight="bold", labelpad=10)
    ax.set_ylabel("Actual", fontsize=13, fontweight="bold", labelpad=10)

    ax.set_xticks(np.arange(-0.5, 2, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 2, 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=2.5)
    ax.tick_params(which="minor", bottom=False, left=False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.tight_layout()
    fig.savefig(output_path.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def draw_combined(rows: list[dict[str, Any]], split: str, output_path: Path) -> None:
    cols = min(3, len(rows))
    rows_n = math.ceil(len(rows) / cols)
    fig, axes = plt.subplots(rows_n, cols, figsize=(6.0 * cols, 5.2 * rows_n), dpi=220)
    axes_arr = np.atleast_1d(axes).reshape(rows_n, cols)

    global_max = max(max(r[split]["tn"], r[split]["fp"], r[split]["fn"], r[split]["tp"]) for r in rows)

    for index, row in enumerate(rows):
        ax = axes_arr[index // cols, index % cols]
        metrics = row[split]
        matrix = np.array([[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]], dtype=float)
        row_sums = matrix.sum(axis=1, keepdims=True)
        percents = np.divide(matrix, row_sums, out=np.zeros_like(matrix), where=row_sums > 0) * 100.0
        ax.imshow(matrix, cmap="Blues", vmin=0, vmax=global_max)
        for y in range(2):
            for x in range(2):
                annotate_cell(ax, x, y, int(matrix[y, x]), float(percents[y, x]), global_max)
        ax.set_xticks([0, 1], labels=["Spoof", "Bonafide"], fontsize=10)
        ax.set_yticks([0, 1], labels=["Spoof", "Bonafide"], fontsize=10)
        ax.set_xlabel("Predicted", fontsize=10, fontweight="bold")
        ax.set_ylabel("Actual", fontsize=10, fontweight="bold")
        ax.set_xticks(np.arange(-0.5, 2, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, 2, 1), minor=True)
        ax.grid(which="minor", color="white", linestyle="-", linewidth=2.2)
        ax.tick_params(which="minor", bottom=False, left=False)
        for spine in ax.spines.values():
            spine.set_visible(False)

    for index in range(len(rows), rows_n * cols):
        axes_arr[index // cols, index % cols].axis("off")

    fig.tight_layout()
    fig.savefig(output_path.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/media/uzair/Data/asvspoof5_exp_result/confusion matrix"),
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--include-root", action="append", type=Path, default=[])
    args = parser.parse_args()

    roots = DEFAULT_ROOTS + args.include_root
    rows = [row for path in iter_final_summaries(roots) if (row := load_row(path)) is not None]
    rows = sorted(rows, key=lambda item: item["eval_eer"])[: args.top_k]
    if not rows:
        raise SystemExit("No usable final_summary.json files with confusion counts were found.")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    with (args.output_dir / "best5_confusion_matrix_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "rank",
                "experiment",
                "display",
                "best_epoch",
                "dev_eer",
                "eval_eer",
                "f1",
                "dev_tn",
                "dev_fp",
                "dev_fn",
                "dev_tp",
                "eval_tn",
                "eval_fp",
                "eval_fn",
                "eval_tp",
                "source",
            ],
        )
        writer.writeheader()
        for rank, row in enumerate(rows, start=1):
            writer.writerow(
                {
                    "rank": rank,
                    "experiment": row["experiment"],
                    "display": row["display"],
                    "best_epoch": row["best_epoch"],
                    "dev_eer": f"{row['dev_eer']:.4f}",
                    "eval_eer": f"{row['eval_eer']:.4f}",
                    "f1": f"{row['f1']:.4f}",
                    "dev_tn": row["dev"]["tn"],
                    "dev_fp": row["dev"]["fp"],
                    "dev_fn": row["dev"]["fn"],
                    "dev_tp": row["dev"]["tp"],
                    "eval_tn": row["eval"]["tn"],
                    "eval_fp": row["eval"]["fp"],
                    "eval_fn": row["eval"]["fn"],
                    "eval_tp": row["eval"]["tp"],
                    "source": row["path"],
                }
            )

    for rank, row in enumerate(rows, start=1):
        base = args.output_dir / f"{rank:02d}_{safe_name(row['display'])}"
        draw_matrix(row, "dev", Path(f"{base}_dev_confusion_matrix"))
        draw_matrix(row, "eval", Path(f"{base}_eval_confusion_matrix"))

    draw_combined(rows, "dev", args.output_dir / "top5_dev_confusion_matrices_combined")
    draw_combined(rows, "eval", args.output_dir / "top5_eval_confusion_matrices_combined")

    print(f"Saved confusion matrices to: {args.output_dir}")
    for rank, row in enumerate(rows, start=1):
        print(
            f"{rank}. {row['display']} | Eval EER={row['eval_eer']:.2f}% | "
            f"Dev EER={row['dev_eer']:.2f}% | F1={row['f1']:.2f}% | "
            f"Eval TN={row['eval']['tn']} FP={row['eval']['fp']} FN={row['eval']['fn']} TP={row['eval']['tp']}"
        )


if __name__ == "__main__":
    main()
