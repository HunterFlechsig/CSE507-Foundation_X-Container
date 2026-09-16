#!/usr/bin/env python3
"""Plot focused vs unfocused CANDID-PTX performance across Foundation X+ runs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


TRAINED_TASKS = {
    "candidptx_cls": {"CLS"},
    "candidptx_loc": {"LOC"},
    "candidptx_seg": {"SEG"},
    "candidptx_cls_loc": {"CLS", "LOC"},
    "candidptx_cls_seg": {"CLS", "SEG"},
    "candidptx_loc_seg": {"LOC", "SEG"},
    "candidptx_cls_loc_seg": {"CLS", "LOC", "SEG"},
}

TASK_METRICS = {
    "CLS": "AUC",
    "LOC": "mAP50",
    "SEG": "DICE",
}


def infer_experiment(run_dir: Path) -> str:
    name = run_dir.name
    if "_seed_" in name:
        name = name.rsplit("_seed_", 1)[0]
    return name


def infer_eval_task(row: dict[str, str]) -> str | None:
    text = " ".join(
        str(row.get(key, "")).upper()
        for key in ("Task-Test", "Dataset", "Task-Train")
    )
    if "SEG" in text or "DICE" in text:
        if _has_number(row.get("DICE")):
            return "SEG"
    if "LOC" in text or "MAP" in text:
        if _has_number(row.get("mAP50")) or _has_number(row.get("mAP50_95")):
            return "LOC"
    if "CLS" in text or "AUC" in text:
        if _has_number(row.get("AUC")):
            return "CLS"
    if _has_number(row.get("AUC")):
        return "CLS"
    if _has_number(row.get("mAP50")):
        return "LOC"
    if _has_number(row.get("DICE")):
        return "SEG"
    return None


def _has_number(value: str | None) -> bool:
    if value is None or value.strip() in {"", "-", "None", "nan"}:
        return False
    try:
        float(value)
        return True
    except ValueError:
        return False


def metric_value(row: dict[str, str], task: str) -> float | None:
    column = TASK_METRICS[task]
    value = row.get(column)
    if not _has_number(value):
        if task == "LOC" and _has_number(row.get("mAP50_95")):
            return float(row["mAP50_95"])
        return None
    return float(value)


def load_rows(run_dir: Path) -> list[dict[str, str]]:
    csv_path = run_dir / "export_csvFile.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing {csv_path}")
    with csv_path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def series_for_task(rows: list[dict[str, str]], task: str) -> tuple[list[int], list[float]]:
    points: dict[int, float] = {}
    for row in rows:
        eval_task = infer_eval_task(row)
        if eval_task != task:
            continue
        try:
            epoch = int(float(row["Epoch"]))
        except (KeyError, TypeError, ValueError):
            continue
        value = metric_value(row, task)
        if value is None:
            continue
        points[epoch] = value
    epochs = sorted(points)
    return epochs, [points[epoch] for epoch in epochs]


def plot_runs(run_dirs: list[Path], output: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharex=False)
    tasks = ("CLS", "LOC", "SEG")
    for axis, task in zip(axes, tasks):
        for run_dir in run_dirs:
            experiment = infer_experiment(run_dir)
            trained = TRAINED_TASKS.get(experiment, set())
            rows = load_rows(run_dir)
            epochs, values = series_for_task(rows, task)
            if not epochs:
                continue
            focus = "focused" if task in trained else "unfocused"
            linestyle = "-" if focus == "focused" else "--"
            axis.plot(epochs, values, linestyle=linestyle, marker="o", label=f"{experiment} ({focus})")
        axis.set_title(f"{task} ({TASK_METRICS[task]})")
        axis.set_xlabel("Epoch")
        axis.set_ylabel(TASK_METRICS[task])
        axis.grid(True, alpha=0.3)
        axis.legend(fontsize=7)
    fig.suptitle("CANDID-PTX focused vs unfocused performance")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    print(f"Wrote {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", nargs="+", type=Path, help="Experiment output directories")
    parser.add_argument("--output", type=Path, default=Path("plot/candidptx_synergy.png"))
    args = parser.parse_args()
    plot_runs(args.run_dirs, args.output)


if __name__ == "__main__":
    main()
