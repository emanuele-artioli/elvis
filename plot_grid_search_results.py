#!/usr/bin/env python3
"""Aggregate grid search analysis reports and generate metric scatter plots."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib

# Force a non-interactive backend so the script works on headless machines.
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Ellipse  # noqa: E402

DEFAULT_METRICS: Sequence[str] = (
    "fvmd",
    "psnr_mean",
    "ssim_mean",
    "lpips_mean",
    "vmaf_mean",
)

EXCLUDED_METRIC_KEYWORDS: Sequence[str] = ("execution", "parameter")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Collect every analysis_results.json under the given directory and "
            "generate a foreground/background scatter plot for the main metrics."
        )
    )
    default_root = Path(__file__).resolve().parent / "grid_search_results"
    parser.add_argument(
        "--root",
        nargs="?",
        default=str(default_root),
        help=(
            "Path to the grid search results directory "
            f"(default: {default_root})."
        ),
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=None,
        help="Optional list of metric keys to plot (defaults to a curated set).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output path for the generated figure (default: <root>/grid_search_metric_scatter.png).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="DPI for the saved figure (default: 200).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the plot window in addition to saving the file.",
    )
    return parser.parse_args()


def _find_analysis_reports(root: Path) -> List[Path]:
    return sorted(root.rglob("analysis_results.json"))


def _load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _prepare_data(
    report_paths: Sequence[Path],
    metrics: Sequence[str],
    root: Path,
) -> Tuple[Dict[str, List[Dict[str, float]]], List[str]]:
    metric_points: Dict[str, List[Dict[str, float]]] = {metric: [] for metric in metrics}
    approaches: List[str] = []

    for path in report_paths:
        try:
            payload = _load_json(path)
        except (json.JSONDecodeError, OSError) as exc:
            print(f"Skipping {path}: failed to read JSON ({exc}).")
            continue

        try:
            rel_parent = path.parent.relative_to(root)
        except ValueError:
            rel_parent = path.parent

        for experiment_name, experiment_data in payload.items():
            if not isinstance(experiment_data, dict):
                continue
            foreground = experiment_data.get("foreground", {})
            background = experiment_data.get("background", {})

            added = False

            for metric in metrics:
                fg_value = foreground.get(metric)
                bg_value = background.get(metric)
                if fg_value is None or bg_value is None:
                    continue
                if metric.endswith("_mean"):
                    std_key = metric.replace("_mean", "_std")
                else:
                    std_key = f"{metric}_std"

                fg_std = foreground.get(std_key)
                bg_std = background.get(std_key)

                metric_points.setdefault(metric, []).append(
                    {
                        "foreground": float(fg_value),
                        "background": float(bg_value),
                        "foreground_std": float(fg_std) if fg_std is not None else None,
                        "background_std": float(bg_std) if bg_std is not None else None,
                        "experiment": experiment_name,
                        "source": str(rel_parent),
                    }
                )
                added = True

            if added and experiment_name not in approaches:
                approaches.append(experiment_name)

    return metric_points, approaches


def _plot_metric_scatter(
    metric_points: Dict[str, List[Dict[str, float]]],
    metrics: Sequence[str],
    approaches: Sequence[str],
    output_path: Path,
    dpi: int,
    show_plot: bool,
) -> None:
    metrics_to_plot = [metric for metric in metrics if metric_points.get(metric)]
    if not metrics_to_plot:
        raise RuntimeError("No metrics with foreground/background pairs were found.")

    num_metrics = len(metrics_to_plot)
    cols = min(3, num_metrics)
    rows = math.ceil(num_metrics / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5.5, rows * 5.0))
    axes_iter = list(axes.flat) if hasattr(axes, "flat") else [axes]

    cmap = plt.get_cmap("tab20", max(1, len(approaches)))
    approach_color_map = {approach: cmap(idx % cmap.N) for idx, approach in enumerate(approaches)}

    legend_elements = [
        Line2D([0], [0], marker="o", color="w", label=approach, markerfacecolor=approach_color_map[approach], markersize=8)
        for approach in approaches
    ]

    point_size = 80

    for idx, metric in enumerate(metrics_to_plot):
        ax = axes_iter[idx]
        points = metric_points[metric]
        fg_values = [point["foreground"] for point in points]
        bg_values = [point["background"] for point in points]
        extent_values: List[float] = []

        for point, fg_val, bg_val in zip(points, fg_values, bg_values):
            approach = point["experiment"]
            color = approach_color_map.get(approach)
            ax.scatter(
                [bg_val],
                [fg_val],
                color=color,
                alpha=0.85,
                edgecolor="white",
                linewidth=0.5,
                s=point_size,
            )

            fg_std = point.get("foreground_std")
            bg_std = point.get("background_std")
            if fg_std is not None and bg_std is not None and (fg_std > 0 or bg_std > 0):
                ellipse = Ellipse(
                    xy=(bg_val, fg_val),
                    width=max(1e-6, 2 * bg_std),
                    height=max(1e-6, 2 * fg_std),
                    edgecolor=color,
                    facecolor="none",
                    linewidth=1.0,
                    alpha=0.4,
                )
                ax.add_patch(ellipse)
                extent_values.extend([
                    fg_val + fg_std,
                    fg_val - fg_std,
                    bg_val + bg_std,
                    bg_val - bg_std,
                ])

        combined = fg_values + bg_values + extent_values
        finite_values = [val for val in combined if math.isfinite(val)]
        if finite_values:
            min_val = min(finite_values)
            max_val = max(finite_values)
            span = max_val - min_val
            padding = 1.0 if span < 1e-6 else span * 0.05
            lower = min_val - padding
            upper = max_val + padding
            ax.set_xlim(lower, upper)
            ax.set_ylim(lower, upper)
            ax.plot([lower, upper], [lower, upper], linestyle="--", color="#666666", linewidth=1.0)

        title = metric.replace("_mean", "").replace("_", " ").upper()
        ax.set_title(title)
        ax.set_xlabel("Background")
        ax.set_ylabel("Foreground")
        ax.grid(True, linestyle="--", alpha=0.3)

    for ax in axes_iter[num_metrics:]:
        ax.axis("off")

    if legend_elements:
        fig.legend(legend_elements, [elem.get_label() for elem in legend_elements], loc="lower right", title="Approach")
    fig.suptitle("Foreground vs Background Metric Scatter", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=(0.02, 0.02, 0.92, 0.95))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    print(f"Saved scatter plot to {output_path}")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def main() -> None:
    args = _parse_args()
    root = Path(args.root).expanduser().resolve()

    if not root.exists():
        raise FileNotFoundError(f"Root directory not found: {root}")

    selected_metrics: Sequence[str] = tuple(args.metrics) if args.metrics else DEFAULT_METRICS
    metrics: Sequence[str] = tuple(
        metric
        for metric in selected_metrics
        if not any(keyword in metric.lower() for keyword in EXCLUDED_METRIC_KEYWORDS)
    )
    if not metrics:
        raise ValueError("No valid metrics selected after filtering excluded keywords.")
    report_paths = _find_analysis_reports(root)
    if not report_paths:
        raise RuntimeError(f"No analysis_results.json files found under {root}")

    print(f"Discovered {len(report_paths)} analysis reports under {root}.")

    metric_points, approaches = _prepare_data(report_paths, metrics, root)

    for metric in metrics:
        count = len(metric_points.get(metric, []))
        print(f"  - {metric}: {count} data points")
    print(f"Found {len(approaches)} unique approaches.")

    output_path = Path(args.output) if args.output else root / "grid_search_metric_scatter.png"

    _plot_metric_scatter(metric_points, metrics, approaches, output_path=output_path, dpi=args.dpi, show_plot=args.show)


if __name__ == "__main__":
    main()
