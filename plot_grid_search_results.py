#!/usr/bin/env python3
"""Aggregate grid search analysis reports and generate metric scatter plots."""

from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import argparse
import json
import math
import statistics
from collections import defaultdict

import numpy as np

import matplotlib
matplotlib.use("Agg")  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, Normalize  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Ellipse  # noqa: E402

DEFAULT_METRICS: Sequence[str] = (
    "psnr_mean",
    "ssim_mean",
    "vmaf_mean",
    "lpips_mean",
    "fvmd",
)

EXCLUDED_METRIC_KEYWORDS: Sequence[str] = ("execution", "parameter")

LOG_SCALE_METRICS: Sequence[str] = ("fvmd",)

MIN_LOG_VALUE: float = 1e-6

TIME_COMPLEXITY_FIGURE_NAME = "grid_search_execution_throughput.png"
METRIC_HEXBIN_FIGURE_NAME = "grid_search_metric_hexbin.png"
METRIC_KDE_FIGURE_NAME = "grid_search_metric_kde.png"
METRIC_VIOLIN_FIGURE_NAME = "grid_search_metric_violin_foreground.png"
METRIC_ALPHA_SCATTER_FIGURE_NAME = "grid_search_metric_alpha_scatter.png"
METRIC_STACKED_BAR_FIGURE_NAME = "grid_search_metric_stacked_bar.png"
PARAM_CORRELATION_FIGURE_NAME = "grid_search_parameter_correlation.png"

PARAMETER_KEYS_FOR_CORRELATION: Sequence[str] = (
    "reference_video",
    "experiment_dir",
    "width",
    "height",
    "block_size",
    "shrink_amount",
    "enable_fvmd",
    "quality_factor",
    "removability_alpha",
    "removability_smoothing_beta",
    "downsample_strength_target_bitrate",
    "gaussian_strength_target_bitrate",
    "propainter_resize_ratio",
    "propainter_ref_stride",
    "propainter_neighbor_length",
    "propainter_subvideo_length",
    "propainter_mask_dilation",
    "propainter_raft_iter",
    "propainter_fp16",
    "propainter_parallel_chunk_length",
    "propainter_chunk_overlap",
    "e2fgvi_ref_stride",
    "e2fgvi_neighbor_stride",
    "e2fgvi_num_ref",
    "e2fgvi_mask_dilation",
    "realesrgan_denoise_strength",
    "realesrgan_tile",
    "realesrgan_tile_pad",
    "realesrgan_pre_pad",
    "realesrgan_parallel_chunk_length",
    "realesrgan_per_device_workers",
    "instantir_cfg",
    "instantir_creative_start",
    "instantir_preview_start",
    "instantir_batch_size",
    "analysis_sample_frames",
    "generate_opencv_benchmarks",
    "metric_stride",
    "fvmd_stride",
    "fvmd_max_frames",
    "fvmd_processes",
    "fvmd_early_stop_delta",
    "fvmd_early_stop_window",
    "vmaf_stride",
)

KDE_METRIC_KEY: str = "vmaf_mean"
KDE_GROUP_KEY: str = "video_name"


def _estimate_frame_count(payload: Dict[str, Any]) -> Optional[float]:
    length = payload.get("video_length_seconds")
    framerate = payload.get("video_framerate")
    try:
        length_val = float(length)
        framerate_val = float(framerate)
    except (TypeError, ValueError):
        return None
    if length_val <= 0 or framerate_val <= 0:
        return None
    return length_val * framerate_val


def _collect_execution_samples(
    samples: Dict[str, List[float]],
    times: Dict[str, Any],
    frame_count: Optional[float],
) -> None:
    if not times or frame_count is None or frame_count <= 0:
        return

    for raw_task, seconds in times.items():
        task = raw_task
        try:
            duration = float(seconds)
        except (TypeError, ValueError):
            continue
        if duration <= 0:
            continue
        fps = frame_count / duration
        samples.setdefault(task, []).append(fps)


def _metric_root(metric: str) -> str:
    return metric.split("_", 1)[0].lower()


def _derive_video_name(rel_parent: Path) -> str:
    candidate = rel_parent.parts[0] if rel_parent.parts else str(rel_parent)
    text = str(candidate)
    if "reference_video--" in text:
        text = text.split("reference_video--", 1)[1]
    if "avc_encoded-" in text:
        text = text.split("avc_encoded-", 1)[1]
    text = text.replace("__", "_")
    if text.endswith("pmp4"):
        text = text[: -len("pmp4")]
    elif text.endswith(".mp4"):
        text = text[: -len(".mp4")]
    text = text.strip("-_ ")
    if not text:
        text = str(candidate)
    return text or "unknown"


def _extract_foreground_background_arrays(
    entries: Sequence[Dict[str, Any]]
) -> Tuple[np.ndarray, np.ndarray]:
    background_values: List[float] = []
    foreground_values: List[float] = []
    for entry in entries:
        bg = entry.get("background")
        fg = entry.get("foreground")
        if bg is None or fg is None:
            continue
        if isinstance(bg, (list, tuple)):
            bg = bg[0] if bg else None
        if isinstance(fg, (list, tuple)):
            fg = fg[0] if fg else None
        try:
            bg_val = float(bg)
            fg_val = float(fg)
        except (TypeError, ValueError):
            continue
        background_values.append(bg_val)
        foreground_values.append(fg_val)
    return np.asarray(background_values, dtype=float), np.asarray(foreground_values, dtype=float)


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
        "--time-output",
        default=None,
        help="Optional output path for the execution throughput figure (default: <root>/grid_search_execution_throughput.png).",
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
) -> Tuple[Dict[str, List[Dict[str, Any]]], List[str], Dict[str, List[float]]]:
    metric_points: Dict[str, List[Dict[str, Any]]] = {metric: [] for metric in metrics}
    approaches: List[str] = []
    execution_samples: Dict[str, List[float]] = {}

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

        video_name = _derive_video_name(rel_parent)
        frame_count = _estimate_frame_count(payload)
        times_entry = payload.get("execution_times_seconds")
        if isinstance(times_entry, dict):
            _collect_execution_samples(execution_samples, times_entry, frame_count)

        for experiment_name, experiment_data in payload.items():
            if experiment_name == "execution_times_seconds":
                continue

            if not isinstance(experiment_data, dict):
                continue

            if "foreground" not in experiment_data or "background" not in experiment_data:
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
                        "approach": experiment_name,
                        "source": str(rel_parent),
                        "video_name": video_name,
                    }
                )
                added = True

            if added and experiment_name not in approaches:
                approaches.append(experiment_name)

    return metric_points, approaches, execution_samples


def _plot_metric_scatter(
    metric_points: Dict[str, List[Dict[str, Any]]],
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

    unique_approaches = list(dict.fromkeys(approaches)) if approaches else ["default"]
    cmap = plt.get_cmap("tab20", max(1, len(unique_approaches)))
    marker_cycle = ["o", "s", "^", "D", "v", "P", "*", "X", "<", ">"]
    approach_styles = {
        approach: {
            "color": cmap(idx % cmap.N),
            "marker": marker_cycle[idx % len(marker_cycle)],
        }
        for idx, approach in enumerate(unique_approaches)
    }

    legend_elements = [
        Line2D(
            [0],
            [0],
            marker=style["marker"],
            color="w",
            label=approach,
            markerfacecolor=style["color"],
            markeredgecolor="black",
            markersize=8,
        )
        for approach, style in approach_styles.items()
    ]

    point_size = 80

    for idx, metric in enumerate(metrics_to_plot):
        ax = axes_iter[idx]
        points = metric_points[metric]
        metric_root = _metric_root(metric)
        use_log_scale = metric_root in LOG_SCALE_METRICS
        fg_values: List[float] = []
        bg_values: List[float] = []
        extent_values: List[float] = []

        for point in points:
            fg_val = float(point["foreground"])
            bg_val = float(point["background"])
            fg_plot = fg_val
            bg_plot = bg_val
            if use_log_scale:
                fg_plot = max(fg_plot, MIN_LOG_VALUE)
                bg_plot = max(bg_plot, MIN_LOG_VALUE)

            fg_values.append(fg_plot)
            bg_values.append(bg_plot)

            approach = point.get("approach") or point.get("experiment") or "default"
            style = approach_styles.get(approach, {"color": "#999999", "marker": "o"})
            point_color = style["color"]
            ax.scatter(
                [bg_plot],
                [fg_plot],
                color=point_color,
                alpha=0.85,
                marker=style["marker"],
                edgecolor="white",
                linewidth=0.5,
                s=point_size,
            )

            fg_std = point.get("foreground_std")
            bg_std = point.get("background_std")
            if fg_std is not None and bg_std is not None and (fg_std > 0 or bg_std > 0):
                ellipse = Ellipse(
                    xy=(bg_plot, fg_plot),
                    width=max(1e-6, 2 * bg_std),
                    height=max(1e-6, 2 * fg_std),
                    edgecolor=point_color,
                    facecolor="none",
                    linewidth=1.0,
                    alpha=0.4,
                )
                ax.add_patch(ellipse)
                candidate_values = [
                    fg_plot + fg_std,
                    fg_plot - fg_std,
                    bg_plot + bg_std,
                    bg_plot - bg_std,
                ]
                if use_log_scale:
                    for candidate in candidate_values:
                        if math.isfinite(candidate):
                            extent_values.append(max(candidate, MIN_LOG_VALUE))
                else:
                    extent_values.extend([candidate for candidate in candidate_values if math.isfinite(candidate)])

        combined = fg_values + bg_values + extent_values
        finite_values = [val for val in combined if math.isfinite(val)]
        if use_log_scale:
            finite_values = [val for val in finite_values if val > 0]

        line_lower = None
        line_upper = None
        if finite_values:
            min_val = min(finite_values)
            max_val = max(finite_values)
            if use_log_scale:
                log_min = math.log10(min_val)
                log_max = math.log10(max_val)
                if math.isclose(log_min, log_max):
                    delta = 0.5
                else:
                    delta = (log_max - log_min) * 0.05
                lower = 10 ** (log_min - delta)
                upper = 10 ** (log_max + delta)
                lower = max(lower, MIN_LOG_VALUE)
                ax.set_xscale("log")
                ax.set_yscale("log")
            else:
                span = max_val - min_val
                padding = 1.0 if span < 1e-6 else span * 0.05
                lower = min_val - padding
                upper = max_val + padding
            if upper <= lower:
                if use_log_scale:
                    upper = lower * 1.1
                else:
                    upper = lower + 1.0
            ax.set_xlim(lower, upper)
            ax.set_ylim(lower, upper)
            line_lower, line_upper = lower, upper
        elif use_log_scale:
            ax.set_xscale("log")
            ax.set_yscale("log")

        if line_lower is not None and line_upper is not None:
            ax.plot([line_lower, line_upper], [line_lower, line_upper], linestyle="--", color="#666666", linewidth=1.0)

        title = metric.replace("_mean", "").replace("_", " ").upper()
        ax.set_title(title)
        ax.set_xlabel("Background")
        ax.set_ylabel("Foreground")
        grid_kwargs = {"linestyle": "--", "alpha": 0.3}
        if use_log_scale:
            ax.grid(True, which="both", **grid_kwargs)
        else:
            ax.grid(True, **grid_kwargs)

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


def _plot_execution_throughput(
    execution_throughput: Dict[str, List[float]],
    output_path: Path,
    dpi: int,
    show_plot: bool,
) -> None:
    if not execution_throughput:
        print("No execution throughput data available; skipping throughput plot.")
        return

    aggregated = [
        (
            task,
            statistics.fmean(samples) if samples else 0.0,
            statistics.pstdev(samples) if len(samples) > 1 else 0.0,
        )
        for task, samples in execution_throughput.items()
        if samples
    ]

    if not aggregated:
        print("No valid execution throughput samples collected; skipping throughput plot.")
        return

    aggregated.sort(key=lambda item: item[1], reverse=True)
    tasks = [item[0] for item in aggregated]
    means = [item[1] for item in aggregated]
    stds = [item[2] for item in aggregated]

    samples_by_task = [execution_throughput.get(task, []) for task in tasks]
    positive_samples = all(
        sample > 0 for samples in samples_by_task for sample in samples
    )

    sanitized_samples = [
        [max(sample, MIN_LOG_VALUE) for sample in samples] if positive_samples else list(samples)
        for samples in samples_by_task
    ]

    use_log_scale = all(mean > 0 for mean in means) and positive_samples

    fig_height = max(4.0, 0.45 * len(tasks) + 1.5)
    fig_width = 9.5
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    y_positions = list(range(len(tasks)))

    palette = LinearSegmentedColormap.from_list(
        "execution_fps_palette",
        ["black", "red", "yellow", "green", "blue"],
    )
    if means:
        min_mean = min(means)
        max_mean = max(means)
        if math.isclose(max_mean, min_mean):
            colors = [palette(0.8) for _ in means]
        else:
            norm = Normalize(vmin=min_mean, vmax=max_mean)
            colors = [palette(norm(value)) for value in means]
    else:
        colors = []

    boxplot = ax.boxplot(
        sanitized_samples,
        vert=False,
        patch_artist=True,
        showfliers=True,
        positions=y_positions,
    )

    for patch, color in zip(boxplot["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_edgecolor("black")
        patch.set_alpha(0.6)

    ax.set_yticks(y_positions)
    ax.set_yticklabels(tasks)
    ax.invert_yaxis()

    grid_kwargs = {"linestyle": "--", "alpha": 0.3}
    if use_log_scale:
        ax.set_xscale("log")
        flattened = [value for row in sanitized_samples for value in row if value > 0]
        if not flattened:
            print("No positive samples for log-scale throughput plot; falling back to linear scale.")
            ax.set_xscale("linear")
            use_log_scale = False
        else:
            min_val = min(flattened)
            max_val = max(flattened + [mean + std for mean, std in zip(means, stds)])
            lower = max(min_val / 1.5, MIN_LOG_VALUE)
            upper = max(max_val * 1.2, lower * 1.5)
            ax.set_xlim(lower, upper)
            ax.grid(True, axis="x", which="both", **grid_kwargs)
    if not use_log_scale:
        flattened = [value for row in sanitized_samples for value in row]
        min_val = min(flattened + [mean for mean in means]) if flattened else 0.0
        max_extent = max(flattened + [mean + std for mean, std in zip(means, stds)]) if flattened else 0.0
        span = max_extent - min_val
        padding = 0.1 * span if span > 0 else max_extent * 0.2 if max_extent > 0 else 1.0
        left = min(0.0, min_val - padding * 0.5)
        ax.set_xlim(left, max_extent + padding)
        ax.grid(True, axis="x", **grid_kwargs)

    ax.set_xlabel("Throughput (frames per second)")
    ax.set_title("Execution Throughput Distribution per Task")

    x_min, x_max = ax.get_xlim()
    span = x_max - x_min
    if use_log_scale:
        log_min, log_max = math.log10(x_min), math.log10(x_max)
        log_span = max(log_max - log_min, 1e-6)

    for y_pos, mean, std in zip(y_positions, means, stds):
        label = f"{mean:.2f} fps"
        if std > 0:
            label += f" ± {std:.2f} fps"

        if use_log_scale and mean > 0:
            log_mean = math.log10(mean)
            log_left_room = log_mean - log_min
            log_right_room = log_max - log_mean
            place_right = log_right_room >= log_left_room
            offset = 0.15 * log_span
            if place_right:
                text_log = min(log_max - 0.05 * log_span, log_mean + offset)
                text_x = 10 ** text_log
                ha = "left"
            else:
                text_log = max(log_min + 0.05 * log_span, log_mean - offset)
                text_x = 10 ** text_log
                ha = "right"
        else:
            left_room = mean - x_min
            right_room = x_max - mean
            place_right = right_room >= left_room
            offset = max(0.02 * span, 0.1)
            if place_right:
                text_x = min(x_max - 0.05 * span, mean + offset)
                ha = "left"
            else:
                text_x = max(x_min + 0.05 * span, mean - offset)
                ha = "right"

        ax.text(
            text_x,
            y_pos,
            label,
            va="center",
            ha=ha,
            fontsize=9,
        )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    print(f"Saved execution throughput plot to {output_path}")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def _plot_metric_hexbin(
    metric_points: Dict[str, List[Dict[str, Any]]],
    metrics: Sequence[str],
    output_path: Path,
    dpi: int,
    show_plot: bool,
) -> None:
    metrics_to_plot = [metric for metric in metrics if metric_points.get(metric)]
    if not metrics_to_plot:
        print("No metric data available for hexbin plot; skipping.")
        return

    cols = min(3, len(metrics_to_plot))
    rows = math.ceil(len(metrics_to_plot) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5.5, rows * 5.0))
    axes_iter = np.atleast_1d(axes).ravel()

    for idx, metric in enumerate(metrics_to_plot):
        ax = axes_iter[idx]
        entries = metric_points.get(metric, [])
        backgrounds, foregrounds = _extract_foreground_background_arrays(entries)
        if backgrounds.size == 0 or foregrounds.size == 0:
            ax.set_visible(False)
            continue

        delta = foregrounds - backgrounds
        if delta.size == 0:
            ax.set_visible(False)
            continue
        vmax = float(np.max(np.abs(delta))) if delta.size else 0.0
        if math.isclose(vmax, 0.0):
            vmax = 1.0
        hb = ax.hexbin(
            backgrounds,
            foregrounds,
            C=delta,
            reduce_C_function=np.mean,
            gridsize=40,
            cmap="coolwarm",
            mincnt=1,
            vmin=-vmax,
            vmax=vmax,
        )
        fig.colorbar(hb, ax=ax, label="Mean (Foreground - Background)")
        min_ref = min(backgrounds.min(), foregrounds.min())
        max_ref = max(backgrounds.max(), foregrounds.max())
        ax.plot(
            [min_ref, max_ref],
            [min_ref, max_ref],
            linestyle="--",
            color="#666666",
            linewidth=1.0,
        )
        ax.set_xlabel("Background")
        ax.set_ylabel("Foreground")
        ax.set_title(f"{metric.replace('_', ' ').title()} Hexbin")
        if metric in LOG_SCALE_METRICS:
            ax.set_xscale("log")
            ax.set_yscale("log")
        ax.grid(True, linestyle="--", alpha=0.25)

    for ax in axes_iter[len(metrics_to_plot) :]:
        ax.remove()

    fig.suptitle("Foreground vs Background Δ Hexbin", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=(0.02, 0.02, 0.98, 0.95))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    print(f"Saved hexbin plot to {output_path}")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def _plot_metric_kde(
    metric_points: Dict[str, List[Dict[str, Any]]],
    metrics: Sequence[str],
    output_path: Path,
    dpi: int,
    show_plot: bool,
) -> None:
    target_entries = metric_points.get(KDE_METRIC_KEY, [])
    if not target_entries:
        print(
            f"Metric '{KDE_METRIC_KEY}' not available for grouped hexbin plot; "
            "skipping."
        )
        return

    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for entry in target_entries:
        grouped[str(entry.get(KDE_GROUP_KEY, "unknown"))].append(entry)

    payload = []
    for label, entries in grouped.items():
        backgrounds, foregrounds = _extract_foreground_background_arrays(entries)
        if backgrounds.size == 0 or foregrounds.size == 0:
            continue
        payload.append((label, backgrounds, foregrounds))

    if not payload:
        print("No samples available for grouped hexbin plot; skipping.")
        return

    payload.sort(key=lambda item: item[0])

    all_backgrounds = np.concatenate([backgrounds for _, backgrounds, _ in payload])
    all_foregrounds = np.concatenate([foregrounds for _, _, foregrounds in payload])
    x_min, x_max = float(all_backgrounds.min()), float(all_backgrounds.max())
    y_min, y_max = float(all_foregrounds.min()), float(all_foregrounds.max())
    if math.isclose(x_min, x_max):
        pad = max(abs(x_min) * 0.05, 1e-3)
        x_min -= pad
        x_max += pad
    if math.isclose(y_min, y_max):
        pad = max(abs(y_min) * 0.05, 1e-3)
        y_min -= pad
        y_max += pad

    num_groups = len(payload)
    cols = min(3, num_groups)
    rows = math.ceil(num_groups / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5.5, rows * 5.0), sharex=True, sharey=True)
    axes_iter = np.atleast_1d(axes).ravel()

    mappable = None
    for ax, (label, backgrounds, foregrounds) in zip(axes_iter, payload):
        hb = ax.hexbin(
            backgrounds,
            foregrounds,
            gridsize=30,
            cmap="viridis",
            mincnt=1,
        )
        mappable = hb
        ax.plot([x_min, x_max], [x_min, x_max], linestyle="--", color="#666666", linewidth=1.0)
        ax.set_title(label)
        ax.grid(True, linestyle="--", alpha=0.25)

    for ax in axes_iter[num_groups:]:
        ax.remove()

    used_axes = axes_iter[:num_groups]
    for idx, ax in enumerate(used_axes):
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        if idx // cols == rows - 1 or rows == 1:
            ax.set_xlabel("Background")
        if idx % cols == 0:
            ax.set_ylabel("Foreground")

    if mappable is not None:
        cbar = fig.colorbar(mappable, ax=used_axes, shrink=0.9)
        cbar.set_label("Sample Count")

    fig.suptitle(
        f"Foreground vs Background Hexbins — {KDE_METRIC_KEY} grouped by {KDE_GROUP_KEY}",
        fontsize=16,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0.02, 0.02, 0.95, 0.95))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    print(f"Saved grouped hexbin plot to {output_path}")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def _plot_metric_violin_foreground(
    metric_points: Dict[str, List[Dict[str, Any]]],
    metrics: Sequence[str],
    approaches: Sequence[str],
    output_path: Path,
    dpi: int,
    show_plot: bool,
) -> None:
    metrics_to_plot = [metric for metric in metrics if metric_points.get(metric)]
    if not metrics_to_plot:
        print("No metric data available for foreground violin plot; skipping.")
        return

    cols = min(3, len(metrics_to_plot))
    rows = math.ceil(len(metrics_to_plot) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5.5, rows * 5.0))
    axes_iter = np.atleast_1d(axes).ravel()
    unique_approaches = list(dict.fromkeys(approaches)) if approaches else ["default"]
    cmap = plt.get_cmap("tab20", max(1, len(unique_approaches)))
    approach_colors = {approach: cmap(idx % cmap.N) for idx, approach in enumerate(unique_approaches)}

    for idx, metric in enumerate(metrics_to_plot):
        ax = axes_iter[idx]
        entries = metric_points.get(metric, [])
        distributions: List[np.ndarray] = []
        labels: List[str] = []
        colors: List[Any] = []

        for approach in unique_approaches:
            filtered = [entry for entry in entries if entry.get("approach") == approach]
            if not filtered and approach == "default":
                filtered = entries
            _, foregrounds = _extract_foreground_background_arrays(filtered)
            if foregrounds.size == 0:
                continue
            distributions.append(foregrounds)
            labels.append(approach)
            colors.append(approach_colors.get(approach, "#888888"))

        if not distributions:
            ax.set_visible(False)
            continue

        parts = ax.violinplot(distributions, showmeans=True, showextrema=False)
        for body, color in zip(parts["bodies"], colors):
            body.set_facecolor(color)
            body.set_edgecolor("black")
            body.set_alpha(0.6)

        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_ylabel("Foreground")
        ax.set_title(f"{metric.replace('_', ' ').title()} Foreground Distribution")
        if metric in LOG_SCALE_METRICS:
            ax.set_yscale("log")
        ax.grid(True, axis="y", linestyle="--", alpha=0.25)

    for ax in axes_iter[len(metrics_to_plot) :]:
        ax.remove()

    fig.suptitle("Foreground Metric Violin Plots by Approach", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=(0.02, 0.02, 0.98, 0.95))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    print(f"Saved foreground violin plot to {output_path}")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def _plot_metric_alpha_scatter(
    metric_points: Dict[str, List[Dict[str, Any]]],
    metrics: Sequence[str],
    approaches: Sequence[str],
    output_path: Path,
    dpi: int,
    show_plot: bool,
) -> None:
    metrics_to_plot = [metric for metric in metrics if metric_points.get(metric)]
    if not metrics_to_plot:
        print("No metric data available for alpha scatter plot; skipping.")
        return

    cols = min(3, len(metrics_to_plot))
    rows = math.ceil(len(metrics_to_plot) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5.5, rows * 5.0))
    axes_iter = np.atleast_1d(axes).ravel()
    unique_approaches = list(dict.fromkeys(approaches)) if approaches else ["default"]
    cmap = plt.get_cmap("tab20", max(1, len(unique_approaches)))
    marker_cycle = ["o", "s", "^", "D", "v", "P", "*", "X", "<", ">"]
    approach_styles = {
        approach: {
            "color": cmap(idx % cmap.N),
            "marker": marker_cycle[idx % len(marker_cycle)],
        }
        for idx, approach in enumerate(unique_approaches)
    }

    for idx, metric in enumerate(metrics_to_plot):
        ax = axes_iter[idx]
        entries = metric_points.get(metric, [])
        legend_handles: Dict[str, Line2D] = {}
        metric_root = _metric_root(metric)
        use_log_scale = metric_root in LOG_SCALE_METRICS

        all_backgrounds, all_foregrounds = _extract_foreground_background_arrays(entries)
        if all_backgrounds.size and all_foregrounds.size:
            bg_plot = all_backgrounds.copy()
            fg_plot = all_foregrounds.copy()
            if use_log_scale:
                bg_plot = np.clip(bg_plot, MIN_LOG_VALUE, None)
                fg_plot = np.clip(fg_plot, MIN_LOG_VALUE, None)
            ax.scatter(
                bg_plot,
                fg_plot,
                s=18,
                c="#444444",
                alpha=0.08,
                marker=".",
                linewidths=0,
                edgecolors="none",
                label="_nolegend_",
            )

        for approach in unique_approaches:
            filtered = [entry for entry in entries if entry.get("approach") == approach]
            if not filtered and approach == "default":
                filtered = entries
            backgrounds, foregrounds = _extract_foreground_background_arrays(filtered)
            if backgrounds.size == 0:
                continue
            style = approach_styles.get(approach, {"color": "#888888", "marker": "o"})
            ax.scatter(
                backgrounds,
                foregrounds,
                alpha=0.25,
                s=40,
                color=style["color"],
                marker=style["marker"],
                edgecolors="white",
                linewidth=0.3,
                label=approach,
            )
            legend_handles[approach] = Line2D(
                [0],
                [0],
                marker=style["marker"],
                linestyle="",
                color=style["color"],
                markerfacecolor=style["color"],
                markeredgecolor="white",
                label=approach,
            )

        ax.set_xlabel("Background")
        ax.set_ylabel("Foreground")
        ax.set_title(f"{metric.replace('_', ' ').title()} Alpha Scatter")
        if use_log_scale:
            ax.set_xscale("log")
            ax.set_yscale("log")
        ax.grid(True, linestyle="--", alpha=0.25)

        if legend_handles:
            ax.legend(handles=list(legend_handles.values()), loc="upper left", fontsize=8)

    for ax in axes_iter[len(metrics_to_plot) :]:
        ax.remove()

    fig.suptitle("Foreground vs Background Alpha-Blended Scatter", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=(0.02, 0.02, 0.98, 0.95))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    print(f"Saved alpha scatter plot to {output_path}")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def _plot_metric_stacked_bar(
    metric_points: Dict[str, List[Dict[str, Any]]],
    metrics: Sequence[str],
    output_path: Path,
    dpi: int,
    show_plot: bool,
) -> None:
    metrics_to_plot = [metric for metric in metrics if metric_points.get(metric)]
    if not metrics_to_plot:
        print("No metric data available for stacked bar plot; skipping.")
        return

    cols = min(2, len(metrics_to_plot))
    rows = math.ceil(len(metrics_to_plot) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 7.0, rows * 5.5))
    axes_iter = np.atleast_1d(axes).ravel()

    for idx, metric in enumerate(metrics_to_plot):
        ax = axes_iter[idx]
        entries = metric_points.get(metric, [])
        video_map: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: {"foreground": [], "background": []})

        for entry in entries:
            video_name = entry.get("video_name") or entry.get("label") or "unknown"
            fg = entry.get("foreground")
            bg = entry.get("background")
            try:
                if fg is not None:
                    video_map[video_name]["foreground"].append(float(fg))
                if bg is not None:
                    video_map[video_name]["background"].append(float(bg))
            except (TypeError, ValueError):
                continue

        aggregated = []
        for video, values in video_map.items():
            if values["foreground"]:
                fg_mean = statistics.fmean(values["foreground"])
                bg_mean = statistics.fmean(values["background"]) if values["background"] else 0.0
                aggregated.append((video, fg_mean, bg_mean))

        if not aggregated:
            ax.set_visible(False)
            continue

        aggregated.sort(key=lambda item: item[1], reverse=True)
        videos = [item[0] for item in aggregated]
        foreground = np.array([item[1] for item in aggregated])
        background = np.array([item[2] for item in aggregated])

        indices = np.arange(len(videos))
        ax.bar(indices, foreground, label="Foreground", color="#1f77b4")
        ax.bar(indices, background, bottom=foreground, label="Background", color="#ff7f0e", alpha=0.8)

        ax.set_xticks(indices)
        ax.set_xticklabels(videos, rotation=45, ha="right")
        ax.set_ylabel("Metric Value")
        ax.set_title(f"{metric.replace('_', ' ').title()} Sorted by Foreground")
        if metric in LOG_SCALE_METRICS:
            ax.set_yscale("log")
        ax.grid(True, axis="y", linestyle="--", alpha=0.25)
        ax.legend(loc="upper right")

    for ax in axes_iter[len(metrics_to_plot) :]:
        ax.remove()

    fig.suptitle("Per-Video Foreground/Background Stacked Metrics", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=(0.02, 0.02, 0.98, 0.95))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    print(f"Saved stacked bar plot to {output_path}")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def _plot_parameter_correlation(
    summary_path: Path,
    output_path: Path,
    dpi: int,
    show_plot: bool,
) -> None:
    if not summary_path.exists():
        print(f"No summary file at {summary_path}; skipping parameter correlation plot.")
        return

    with summary_path.open("r", encoding="utf-8") as fp:
        summary_data = json.load(fp)

    if not summary_data:
        print("Summary file is empty; skipping parameter correlation plot.")
        return

    parameter_rows: List[Dict[str, Any]] = []
    for entry in summary_data:
        overrides = entry.get("overrides") or {}
        parameter_rows.append({key: overrides.get(key) for key in PARAMETER_KEYS_FOR_CORRELATION})

    numeric_keys: List[str] = []
    numeric_columns: List[List[float]] = []
    for key in PARAMETER_KEYS_FOR_CORRELATION:
        column: List[float] = []
        valid = True
        for row in parameter_rows:
            value = row.get(key)
            if value is None:
                valid = False
                break
            if isinstance(value, bool):
                value = int(value)
            if isinstance(value, (int, float)):
                column.append(float(value))
            else:
                valid = False
                break
        if valid and len(column) >= 2 and len(set(column)) > 1:
            numeric_keys.append(key)
            numeric_columns.append(column)

    if len(numeric_keys) < 2:
        print("Not enough numeric parameter variation for correlation plot; skipping.")
        return

    data = np.column_stack(numeric_columns)
    corr = np.corrcoef(data, rowvar=False)

    fig_size = max(6.0, 0.6 * len(numeric_keys))
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1.0, vmax=1.0)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Correlation")
    ax.set_xticks(range(len(numeric_keys)))
    ax.set_xticklabels(numeric_keys, rotation=45, ha="right")
    ax.set_yticks(range(len(numeric_keys)))
    ax.set_yticklabels(numeric_keys)

    for i in range(len(numeric_keys)):
        for j in range(len(numeric_keys)):
            ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center", fontsize=8, color="black")

    ax.set_title("Correlation Between Grid Search Parameters")
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    print(f"Saved parameter correlation plot to {output_path}")

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

    metric_points, approaches, execution_throughput = _prepare_data(report_paths, metrics, root)

    for metric in metrics:
        count = len(metric_points.get(metric, []))
        print(f"  - {metric}: {count} data points")
    print(f"Found {len(approaches)} unique approaches.")

    if execution_throughput:
        print("Execution throughput samples per task (fps):")
        for task, samples in sorted(execution_throughput.items(), key=lambda item: statistics.fmean(item[1]) if item[1] else 0.0, reverse=True):
            if not samples:
                continue
            avg = statistics.fmean(samples)
            spread = statistics.pstdev(samples) if len(samples) > 1 else 0.0
            if spread > 0:
                print(f"  - {task}: mean={avg:.2f} fps, std={spread:.2f} fps")
            else:
                print(f"  - {task}: mean={avg:.2f} fps")

    scatter_output = Path(args.output) if args.output else root / "grid_search_metric_scatter.png"
    _plot_metric_scatter(metric_points, metrics, approaches, scatter_output, args.dpi, args.show)

    hexbin_output = scatter_output.with_name(METRIC_HEXBIN_FIGURE_NAME)
    _plot_metric_hexbin(metric_points, metrics, hexbin_output, args.dpi, args.show)

    kde_output = scatter_output.with_name(METRIC_KDE_FIGURE_NAME)
    _plot_metric_kde(metric_points, metrics, kde_output, args.dpi, args.show)

    violin_output = scatter_output.with_name(METRIC_VIOLIN_FIGURE_NAME)
    _plot_metric_violin_foreground(metric_points, metrics, approaches, violin_output, args.dpi, args.show)

    alpha_output = scatter_output.with_name(METRIC_ALPHA_SCATTER_FIGURE_NAME)
    _plot_metric_alpha_scatter(metric_points, metrics, approaches, alpha_output, args.dpi, args.show)

    stacked_output = scatter_output.with_name(METRIC_STACKED_BAR_FIGURE_NAME)
    _plot_metric_stacked_bar(metric_points, metrics, stacked_output, args.dpi, args.show)

    correlation_output = scatter_output.with_name(PARAM_CORRELATION_FIGURE_NAME)
    _plot_parameter_correlation(root / "runs_summary.json", correlation_output, args.dpi, args.show)


if __name__ == "__main__":
    main()
