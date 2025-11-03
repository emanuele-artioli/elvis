#!/usr/bin/env python3
"""Grid search runner for the ELVIS pipeline."""

from __future__ import annotations

import itertools
import json
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List

from elvis import ElvisConfig, run_elvis

# Directory that will contain renamed experiment runs
GRID_RESULTS_DIR = Path("grid_search_results")

# Parameter grid: each entry lists the candidate values for a configuration field.
# Single-value lists keep the search manageable by default while allowing easy edits.
PARAMETER_GRID: Dict[str, List[Any]] = {
    "reference_video": [
        "/home/itec/emanuele/Datasets/DAVIS/avc_encoded/bear.mp4",
        "/home/itec/emanuele/Datasets/DAVIS/avc_encoded/camel.mp4",
        "/home/itec/emanuele/Datasets/DAVIS/avc_encoded/dance-jump.mp4",
        "/home/itec/emanuele/Datasets/DAVIS/avc_encoded/elephant.mp4",
        "/home/itec/emanuele/Datasets/DAVIS/avc_encoded/flamingo.mp4",
        "/home/itec/emanuele/Datasets/DAVIS/avc_encoded/goat.mp4",
        "/home/itec/emanuele/Datasets/DAVIS/avc_encoded/hike.mp4",
        "/home/itec/emanuele/Datasets/DAVIS/avc_encoded/india.mp4",
        "/home/itec/emanuele/Datasets/DAVIS/avc_encoded/judo.mp4",
        "/home/itec/emanuele/Datasets/DAVIS/avc_encoded/kite-surf.mp4",
        "/home/itec/emanuele/Datasets/DAVIS/avc_encoded/lady-running.mp4",
        "/home/itec/emanuele/Datasets/DAVIS/avc_encoded/motorbike.mp4",
        "/home/itec/emanuele/Datasets/DAVIS/avc_encoded/night-race.mp4",
        "/home/itec/emanuele/Datasets/DAVIS/avc_encoded/paragliding.mp4",
        "/home/itec/emanuele/Datasets/DAVIS/avc_encoded/rhino.mp4",
        "/home/itec/emanuele/Datasets/DAVIS/avc_encoded/schoolgirls.mp4",
        "/home/itec/emanuele/Datasets/DAVIS/avc_encoded/shooting.mp4",
        "/home/itec/emanuele/Datasets/DAVIS/avc_encoded/tennis.mp4",
        "/home/itec/emanuele/Datasets/DAVIS/avc_encoded/upside-down.mp4",
        "/home/itec/emanuele/Datasets/DAVIS/avc_encoded/varanus-cage.mp4",
        "/home/itec/emanuele/Datasets/DAVIS/avc_encoded/walking.mp4",
        ],
    "experiment_dir": ["experiment"],
    "width": [640],
    "height": [360],
    "block_size": [8],
    "shrink_amount": [0.25],
    "quality_factor": [1.2],
    "target_bitrate_override": [None],
    "force_framerate": [None],
    "removability_alpha": [0.5],
    "removability_smoothing_beta": [0.5],
    "removability_working_dir": [None],
    "decode_quality": [1],
    "decode_start_number": [1],
    "baseline_encode_preset": ["medium"],
    "baseline_encode_pix_fmt": ["yuv420p"],
    "adaptive_encode_preset": ["medium"],
    "adaptive_encode_pix_fmt": ["yuv420p"],
    "shrunk_encode_preset": ["medium"],
    "shrunk_encode_pix_fmt": ["yuv420p"],
    "downsample_encode_preset": ["medium"],
    "downsample_encode_pix_fmt": ["yuv420p"],
    "gaussian_encode_preset": ["medium"],
    "gaussian_encode_pix_fmt": ["yuv420p"],
    "roi_save_qp_maps": [True],
    "downsample_strength_target_bitrate": [10000],
    "gaussian_strength_target_bitrate": [10000],
    "propainter_dir": [None],
    "propainter_resize_ratio": [1.0],
    "propainter_ref_stride": [20],
    "propainter_neighbor_length": [4],
    "propainter_subvideo_length": [40],
    "propainter_mask_dilation": [4],
    "propainter_raft_iter": [20],
    "propainter_fp16": [True],
    "propainter_devices": [None],
    "propainter_parallel_chunk_length": [None],
    "propainter_chunk_overlap": [None],
    "e2fgvi_dir": [None],
    "e2fgvi_model": ["e2fgvi_hq"],
    "e2fgvi_ckpt": [None],
    "e2fgvi_ref_stride": [10],
    "e2fgvi_neighbor_stride": [5],
    "e2fgvi_num_ref": [-1],
    "e2fgvi_mask_dilation": [4],
    "realesrgan_dir": [None],
    "realesrgan_model_name": ["RealESRGAN_x4plus"],
    "realesrgan_denoise_strength": [1.0],
    "realesrgan_tile": [0],
    "realesrgan_tile_pad": [10],
    "realesrgan_pre_pad": [0],
    "realesrgan_fp32": [False],
    "realesrgan_devices": [None],
    "realesrgan_parallel_chunk_length": [None],
    "realesrgan_per_device_workers": [1],
    "realesrgan_model_path": [None],
    "instantir_weights_dir": [None],
    "instantir_cfg": [7.0],
    "instantir_creative_start": [1.0],
    "instantir_preview_start": [0.0],
    "instantir_seed": [42],
    "instantir_devices": [None],
    "instantir_batch_size": [4],
    "instantir_parallel_chunk_length": [None],
    "analysis_sample_frames": [None],
    "generate_opencv_benchmarks": [True],
    "metric_stride": [1],
    "fvmd_stride": [1],
    "fvmd_max_frames": [None],
    "fvmd_processes": [None],
    "fvmd_early_stop_delta": [0.002],
    "fvmd_early_stop_window": [50],
    "vmaf_stride": [1],
}


def _slugify(parts: Dict[str, Any]) -> str:
    sanitized_segments: List[str] = []
    for key, value in parts.items():
        text = str(value)
        text = text.replace("/", "-").replace("\\", "-")
        text = text.replace(" ", "-").replace(".", "p")
        text = "".join(ch for ch in text if ch.isalnum() or ch in {"-", "_"})
        sanitized_segments.append(f"{key}-{text}")
    return "_".join(sanitized_segments)


def _ensure_clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)


def _extract_metric_sections(analysis_data: Dict[str, Any]) -> Dict[str, Any]:
    """Return only the nested metric dictionaries for each pipeline variant."""

    metric_sections: Dict[str, Any] = {}
    for key, value in analysis_data.items():
        if isinstance(value, dict) and "foreground" in value and "background" in value:
            metric_sections[key] = value
    return metric_sections


def main() -> None:
    GRID_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    grid_keys = list(PARAMETER_GRID.keys())
    grid_values = [PARAMETER_GRID[key] for key in grid_keys]
    combinations = list(itertools.product(*grid_values))

    varying_keys = [key for key, values in PARAMETER_GRID.items() if len(values) > 1]

    summary: List[Dict[str, Any]] = []

    for run_index, combo in enumerate(combinations, start=1):
        overrides = {key: combo[idx] for idx, key in enumerate(grid_keys)}

        config_data = asdict(ElvisConfig())
        config_data.update(overrides)
        config = ElvisConfig(**config_data)

        experiment_path = Path(config.experiment_dir)
        _ensure_clean_dir(experiment_path)

        print(f"\n[{run_index}/{len(combinations)}] Running configuration: {overrides}")
        results = run_elvis(config)

        slug_source = {key: overrides[key] for key in varying_keys}
        slug = _slugify(slug_source) or f"run_{run_index:03d}"
        final_dir = GRID_RESULTS_DIR / slug
        _ensure_clean_dir(final_dir)

        shutil.move(str(experiment_path), final_dir)

        analysis_path = final_dir / "analysis_results.json"
        analysis_data: Dict[str, Any] = results
        if analysis_path.exists():
            with analysis_path.open("r") as fp:
                analysis_data = json.load(fp)
            analysis_data.setdefault("parameters", {}).setdefault("derived", {})["experiment_dir"] = str(final_dir)
            analysis_data["parameters"]["derived"]["analysis_results_path"] = str(analysis_path)
            analysis_data["experiment_dir"] = str(final_dir)
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

    print(f"\nCompleted {len(combinations)} grid search runs. Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
