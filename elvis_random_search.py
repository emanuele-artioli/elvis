#!/usr/bin/env python3
"""Random search runner for the ELVIS pipeline."""

from __future__ import annotations

import argparse
import itertools
import json
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List

from elvis import ElvisConfig, run_elvis
from elvis_grid_search import (
    _ensure_clean_dir,
    _extract_metric_sections,
    _is_valid_overrides,
    _slugify,
)

# Directory that will contain renamed experiment runs
GRID_RESULTS_DIR = Path("random_search_results")

# Parameter grid: each entry lists the candidate values for a configuration field.
# Single-value lists keep the search manageable by default while allowing easy edits.
PARAMETER_GRID: Dict[str, List[Any]] = {
    "reference_video": [
        "/home/itec/emanuele/Datasets/DAVIS/avc_encoded/bear.mp4",
        "/home/itec/emanuele/Datasets/DAVIS/avc_encoded/camel.mp4",
        "/home/itec/emanuele/Datasets/DAVIS/avc_encoded/dance-jump.mp4",
        "/home/itec/emanuele/Datasets/DAVIS/avc_encoded/elephant.mp4",
        "/home/itec/emanuele/Datasets/DAVIS/avc_encoded/goat.mp4",
        "/home/itec/emanuele/Datasets/DAVIS/avc_encoded/hike.mp4",
        "/home/itec/emanuele/Datasets/DAVIS/avc_encoded/schoolgirls.mp4",
        "/home/itec/emanuele/Datasets/DAVIS/avc_encoded/upside-down.mp4",
        "/home/itec/emanuele/Datasets/DAVIS/avc_encoded/varanus-cage.mp4",
        "/home/itec/emanuele/Datasets/DAVIS/avc_encoded/walking.mp4",
        # "/home/itec/emanuele/Datasets/DAVIS/avc_encoded/flamingo.mp4",
        # "/home/itec/emanuele/Datasets/DAVIS/avc_encoded/india.mp4",
        # "/home/itec/emanuele/Datasets/DAVIS/avc_encoded/judo.mp4",
        # "/home/itec/emanuele/Datasets/DAVIS/avc_encoded/kite-surf.mp4",
        # "/home/itec/emanuele/Datasets/DAVIS/avc_encoded/lady-running.mp4",
        # "/home/itec/emanuele/Datasets/DAVIS/avc_encoded/motorbike.mp4",
        # "/home/itec/emanuele/Datasets/DAVIS/avc_encoded/night-race.mp4",
        # "/home/itec/emanuele/Datasets/DAVIS/avc_encoded/paragliding.mp4",
        # "/home/itec/emanuele/Datasets/DAVIS/avc_encoded/rhino.mp4",
        # "/home/itec/emanuele/Datasets/DAVIS/avc_encoded/shooting.mp4",
        # "/home/itec/emanuele/Datasets/DAVIS/avc_encoded/tennis.mp4",
        ],
    "experiment_dir": ["experiment"],
    "width": [640, 1280, 1920],
    "height": [360, 720, 1080],
    "block_size": [8, 16, 32, 64],
    "shrink_amount": [0.1, 0.25, 0.5],
    "enable_fvmd": [False, False, False, False, True], # FVMD is expensive, bias towards False
    "quality_factor": [1.2],
    "removability_alpha": [0, 0.25, 0.5, 0.75, 1],
    "removability_smoothing_beta": [0, 0.25, 0.5, 0.75, 1],
    "strength_maps_target_bitrate": [10000, 25000, 50000],

    "propainter_resize_ratio": [1.0],
    "propainter_ref_stride": [2],
    "propainter_neighbor_length": [4],
    "propainter_subvideo_length": [8],
    "propainter_mask_dilation": [4],
    "propainter_raft_iter": [20],
    "propainter_fp16": [True],
    "propainter_parallel_chunk_length": [None],
    "propainter_chunk_overlap": [2],
    "e2fgvi_ref_stride": [2],
    "e2fgvi_neighbor_stride": [2],
    "e2fgvi_num_ref": [8],
    "e2fgvi_mask_dilation": [4],
    "realesrgan_denoise_strength": [1.0],
    "realesrgan_tile": [0],
    "realesrgan_tile_pad": [8],
    "realesrgan_pre_pad": [0],
    "realesrgan_parallel_chunk_length": [None],
    "realesrgan_per_device_workers": [1],
    "instantir_cfg": [7.0],
    "instantir_creative_start": [1.0],
    "instantir_preview_start": [0.0],
    "instantir_batch_size": [2],
    "analysis_sample_frames": [None],
    "generate_opencv_benchmarks": [False],
    "metric_stride": [1],
    "fvmd_stride": [5],
    "fvmd_max_frames": [None],
    "fvmd_processes": [16],
    "fvmd_early_stop_delta": [0.01],
    "fvmd_early_stop_window": [50],
    "vmaf_stride": [1],
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run random ELVIS configurations")
    parser.add_argument(
        "--runs",
        type=int,
        default=100,
        help="Number of random configurations to evaluate (default: 10)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="RNG seed for reproducible sampling (default: None)",
    )
    return parser.parse_args()


def _sample_unique_configs(rng: random.Random, runs: int) -> List[Dict[str, Any]]:
    grid_keys = list(PARAMETER_GRID.keys())
    grid_values = [PARAMETER_GRID[key] for key in grid_keys]

    valid_overrides: List[Dict[str, Any]] = []
    signature_lookup = {}

    for combo in itertools.product(*grid_values):
        overrides = {key: combo[idx] for idx, key in enumerate(grid_keys)}
        if not _is_valid_overrides(overrides):
            continue
        signature = tuple(overrides[key] for key in grid_keys)
        if signature in signature_lookup:
            continue
        signature_lookup[signature] = overrides
        valid_overrides.append(overrides)

    if not valid_overrides:
        return []

    if runs > len(valid_overrides):
        print(
            f"Requested runs ({runs}) exceed the number of valid configurations ({len(valid_overrides)}). "
            "Running each valid configuration once instead."
        )
        runs = len(valid_overrides)

    if runs == 0:
        return []

    sampled_overrides: List[Dict[str, Any]] = []
    seen_signatures = set()
    max_attempts = runs * 100
    attempts = 0

    while len(sampled_overrides) < runs and attempts < max_attempts:
        attempts += 1
        candidate = {key: rng.choice(PARAMETER_GRID[key]) for key in grid_keys}
        if not _is_valid_overrides(candidate):
            continue
        signature = tuple(candidate[key] for key in grid_keys)
        if signature in seen_signatures:
            continue
        reference = signature_lookup.get(signature)
        if reference is None:
            continue
        seen_signatures.add(signature)
        sampled_overrides.append(candidate)

    if len(sampled_overrides) < runs:
        remaining_signatures = [
            sig for sig in signature_lookup if sig not in seen_signatures
        ]
        rng.shuffle(remaining_signatures)
        for sig in remaining_signatures:
            sampled_overrides.append(signature_lookup[sig])
            seen_signatures.add(sig)
            if len(sampled_overrides) >= runs:
                break

    return sampled_overrides[:runs]


def main() -> None:
    args = _parse_args()
    rng = random.Random(args.seed)

    GRID_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    grid_keys = list(PARAMETER_GRID.keys())
    varying_keys = [key for key, values in PARAMETER_GRID.items() if len(values) > 1]

    sampled_configs = _sample_unique_configs(rng, args.runs)

    if not sampled_configs:
        print("No valid parameter combinations available after applying sanity checks.")
        return

    summary: List[Dict[str, Any]] = []

    for run_index, overrides in enumerate(sampled_configs, start=1):
        slug_source = {key: overrides[key] for key in varying_keys if key in overrides}
        slug = _slugify(slug_source) or f"run_{run_index:03d}"
        final_dir = GRID_RESULTS_DIR / slug

        config_data = asdict(ElvisConfig())
        config_data["minimal_figures"] = True
        config_data.update(overrides)
        config_data["experiment_dir"] = str(final_dir)
        config = ElvisConfig(**config_data)

        experiment_path = Path(config.experiment_dir)
        _ensure_clean_dir(experiment_path)
        experiment_path.mkdir(parents=True, exist_ok=True)

        print(f"\n[{run_index}/{len(sampled_configs)}] Running configuration: {overrides}")
        results = run_elvis(config)

        analysis_path = experiment_path / "analysis_results.json"
        analysis_data: Dict[str, Any] = results
        if analysis_path.exists():
            with analysis_path.open("r") as fp:
                analysis_data = json.load(fp)
            analysis_data.setdefault("parameters", {}).setdefault("derived", {})[
                "experiment_dir"
            ] = str(experiment_path)
            analysis_data["parameters"]["derived"]["analysis_results_path"] = str(
                analysis_path
            )
            analysis_data["experiment_dir"] = str(experiment_path)
            analysis_data["analysis_results_path"] = str(analysis_path)
            analysis_data["grid_search_label"] = slug
            with analysis_path.open("w") as fp:
                json.dump(analysis_data, fp, indent=4)

        metrics_snapshot = _extract_metric_sections(analysis_data)
        timings_snapshot = analysis_data.get("execution_times_seconds", {})
        video_metadata = {
            "video_name": analysis_data.get("video_name"),
            "video_length_seconds": analysis_data.get("video_length_seconds"),
            "video_framerate": analysis_data.get("video_framerate"),
            "video_resolution": analysis_data.get("video_resolution"),
            "block_size": analysis_data.get("block_size"),
            "target_bitrate_bps": analysis_data.get("target_bitrate_bps"),
        }

        summary.append(
            {
                "label": slug,
                "overrides": overrides,
                "analysis_results_path": str(analysis_path),
                "metrics": metrics_snapshot,
                "execution_times_seconds": timings_snapshot,
                "video_metadata": video_metadata,
            }
        )

    summary_path = GRID_RESULTS_DIR / "runs_summary.json"
    with summary_path.open("w") as fp:
        json.dump(summary, fp, indent=4)

    print(
        f"\nCompleted {len(sampled_configs)} random search runs. Summary saved to {summary_path}"
    )


if __name__ == "__main__":
    main()
