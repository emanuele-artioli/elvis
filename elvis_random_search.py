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
    # Path to input video file to be processed. Determines video content characteristics.
    # Performance impact: Different videos have varying complexity, motion, and object detection requirements.
    "reference_video": [
        "/home/itec/emanuele/Datasets/DAVIS/avc_encoded/bear.mp4",
        "/home/itec/emanuele/Datasets/DAVIS/avc_encoded/camel.mp4",
        "/home/itec/emanuele/Datasets/DAVIS/avc_encoded/dog.mp4",
        "/home/itec/emanuele/Datasets/DAVIS/avc_encoded/flamingo.mp4",
        "/home/itec/emanuele/Datasets/DAVIS/avc_encoded/goat.mp4",
        "/home/itec/emanuele/Datasets/DAVIS/avc_encoded/horsejump-low.mp4",
        "/home/itec/emanuele/Datasets/DAVIS/avc_encoded/judo.mp4",
        "/home/itec/emanuele/Datasets/DAVIS/avc_encoded/kite-surf.mp4",
        "/home/itec/emanuele/Datasets/DAVIS/avc_encoded/lady-running.mp4",
        "/home/itec/emanuele/Datasets/DAVIS/avc_encoded/motocross-bumps.mp4",
        "/home/itec/emanuele/Datasets/DAVIS/avc_encoded/parkour.mp4",
        "/home/itec/emanuele/Datasets/DAVIS/avc_encoded/scooter-gray.mp4",
        "/home/itec/emanuele/Datasets/DAVIS/avc_encoded/tuk-tuk.mp4",
        "/home/itec/emanuele/Datasets/DAVIS/avc_encoded/walking.mp4"
        ],
    # Directory where experiment results are saved. No impact on quality or runtime performance.
    "experiment_dir": ["experiment"],
    # Video width in pixels after processing. Affects encoding bitrate calculation and memory usage.
    # Performance impact: Higher values increase memory usage and processing time linearly.
    "width": [640],
    # Video height in pixels after processing. Affects encoding bitrate calculation and memory usage.
    # Performance impact: Higher values increase memory usage and processing time linearly.
    "height": [360],
    # Size of blocks in pixels for selective removal (must divide width and height evenly).
    # Performance impact: Smaller blocks = finer granularity but more computation; larger blocks = coarser control.
    "block_size": [8],
    # Fraction (0-1) or number of blocks to remove per row. Controls compression aggressiveness.
    # Performance impact: Higher values = more compression, smaller file size, but more quality degradation.
    # Quality impact: Higher = more aggressive removal, potentially visible artifacts.
    "shrink_amount": [0.1],
    # Whether to compute FVMD (Fréchet Video Motion Distance) metric during evaluation.
    # Performance impact: FVMD is computationally expensive (requires keypoint tracking), disabling saves significant time.
    # Bias towards False (2:1) since it's expensive.
    "enable_fvmd": [False, False, True],
    # Multiplier for target bitrate calculation (higher = better quality, larger file).
    # Performance impact: Higher values produce larger files but better visual quality; uses formula: pixels_per_second * 0.01 * quality_factor.
    "quality_factor": [3],
    # Weight (0-1) balancing spatial vs temporal complexity in removability scoring.
    # Performance impact: 0 = only temporal complexity, 1 = only spatial complexity, 0.5 = balanced.
    # Quality impact: Affects which blocks are selected for removal; optimal value depends on video content.
    "removability_alpha": [0, 0.25, 0.5, 0.75, 1],
    # Temporal smoothing factor (0-1) for removability scores across frames.
    # Performance impact: Lower values = more smoothing (more dependence on previous frame), 1 = no smoothing.
    # Quality impact: Smoothing reduces temporal flickering in block removal patterns.
    "removability_smoothing_beta": [0, 0.25, 0.5, 0.75, 1],
    # Target bitrate in bps for encoding strength map videos (used in PRESLEY approaches).
    # Performance impact: Higher = larger strength map files, longer encoding time, but preserves more detail in maps.
    "strength_maps_target_bitrate": [10000, 25000, 50000],

    # ProPainter processing scale (1.0 = native resolution, <1.0 = downscale for speed).
    # Performance impact: Lower values significantly speed up inpainting but may reduce quality.
    "propainter_resize_ratio": [1.0],
    # Stride between global reference frames for ProPainter's long-term propagation.
    # Performance impact: Larger stride = faster but may miss temporal context; smaller = slower but more accurate.
    "propainter_ref_stride": [5],
    # Number of neighboring frames ProPainter uses for local temporal context.
    # Performance impact: More neighbors = better temporal consistency but slower processing and higher memory.
    "propainter_neighbor_length": [10],
    # Maximum frames per ProPainter sub-video chunk for long video processing.
    # Performance impact: Larger chunks = better temporal coherence but higher memory usage; smaller = faster, lower memory.
    "propainter_subvideo_length": [30],
    # Number of mask dilation iterations in ProPainter (expands inpainting regions).
    # Performance impact: Higher values enlarge masked areas, increase inpainting workload and improve edge quality.
    "propainter_mask_dilation": [4],
    # RAFT optical flow iterations for ProPainter motion estimation.
    # Performance impact: More iterations = better flow accuracy but slower; fewer = faster but less precise motion.
    "propainter_raft_iter": [20],
    # Use half-precision (fp16) in ProPainter to reduce memory and speed up inference.
    # Performance impact: True = 2x faster, ~50% memory usage, minimal quality loss; False = slower but full precision.
    "propainter_fp16": [True],
    # Frames per parallel chunk when using multiple GPUs for ProPainter (None = auto-calculated).
    # Performance impact: Smaller chunks = better GPU utilization but more overhead; larger = less overhead but may underutilize GPUs.
    "propainter_parallel_chunk_length": [None],
    # Overlapping frames between parallel ProPainter chunks to maintain temporal consistency.
    # Performance impact: More overlap = better temporal continuity but redundant computation; less = faster but potential seams.
    "propainter_chunk_overlap": [2],
    # Stride between reference frames for E2FGVI inpainting model.
    # Performance impact: Larger stride = faster but less temporal accuracy; smaller = more reference frames, slower.
    "e2fgvi_ref_stride": [5],
    # Stride for selecting neighboring frames in E2FGVI's local attention mechanism.
    # Performance impact: Larger = faster but coarser temporal sampling; smaller = finer sampling, slower.
    "e2fgvi_neighbor_stride": [5],
    # Number of reference frames E2FGVI uses (-1 = all available within window).
    # Performance impact: More refs = better quality but higher memory and computation; fewer = faster, lower quality.
    "e2fgvi_num_ref": [10],
    # Mask dilation iterations for E2FGVI (expands regions to inpaint).
    # Performance impact: More dilation = larger inpainting areas, more work, better edge blending.
    "e2fgvi_mask_dilation": [4],
    # Denoising strength for RealESRGAN upscaling (0-1, higher = more aggressive denoising).
    # Performance impact: Higher values remove more noise but may over-smooth details; lower preserves texture but keeps noise.
    "realesrgan_denoise_strength": [1.0],
    # Tile size for RealESRGAN processing (0 = no tiling, process entire image).
    # Performance impact: Non-zero tiling reduces memory usage for large images but adds overhead; 0 = faster for small images.
    "realesrgan_tile": [0],
    # Padding around tiles in RealESRGAN to reduce seam artifacts when tiling.
    # Performance impact: More padding = better tile blending but more computation per tile; less = faster, potential seams.
    "realesrgan_tile_pad": [8],
    # Pre-padding for RealESRGAN to handle image edges (reduces edge artifacts).
    # Performance impact: More padding = slightly more computation but better edge quality.
    "realesrgan_pre_pad": [8],
    # Frames per parallel chunk for RealESRGAN when using multiple GPUs (None = auto).
    # Performance impact: Affects GPU workload distribution; smaller = better load balancing, larger = less overhead.
    "realesrgan_parallel_chunk_length": [None],
    # Number of worker threads per device for RealESRGAN parallel processing.
    # Performance impact: More workers = higher throughput if GPU can handle it, but potential resource contention.
    "realesrgan_per_device_workers": [1],
    # Classifier-free guidance scale for InstantIR diffusion model (higher = more adherence to prompt/condition).
    # Performance impact: Higher values produce sharper, more structured results but may over-sharpen; lower = softer, more natural.
    "instantir_cfg": [7.0],
    # Timestep (0-1) when InstantIR starts creative sampling in diffusion process.
    # Performance impact: 1.0 = no creative phase, faster, stays closer to input; <1.0 = more creative reconstruction, slower.
    "instantir_creative_start": [1.0],
    # Timestep (0-1) when InstantIR starts preview generation (early stopping for speed).
    # Performance impact: Earlier preview = faster but potentially lower quality; 0.0 = full diffusion process.
    "instantir_preview_start": [0.0],
    # Batch size for InstantIR processing (number of frames processed simultaneously).
    # Performance impact: Larger batches = better GPU utilization and faster throughput but higher memory usage.
    "instantir_batch_size": [2],
    # Specific frame indices to sample for detailed analysis (None = sample evenly across video).
    # Performance impact: None = auto-sampling; explicit list allows targeted analysis, no runtime impact.
    "analysis_sample_frames": [None],
    # Whether to generate OpenCV-based restoration benchmarks (Lanczos, Unsharp Mask) for comparison.
    # Performance impact: True = generates additional benchmark videos, increases runtime; False = skips benchmarks, faster.
    "generate_opencv_benchmarks": [False],
    # Stride for sampling frames during quality metric calculation (PSNR, SSIM, LPIPS).
    # Performance impact: Larger stride = fewer frames evaluated, faster but coarser quality assessment; 1 = all frames, thorough but slow.
    "metric_stride": [1],
    # Stride for sampling frames during FVMD calculation (motion feature analysis).
    # Performance impact: Larger stride = faster FVMD computation but less accurate motion representation; smaller = slower, more accurate.
    "fvmd_stride": [5],
    # Maximum frames to use for FVMD calculation (None = no limit).
    # Performance impact: Lower limit = faster FVMD but may miss motion patterns; None = comprehensive but slow for long videos.
    "fvmd_max_frames": [None],
    # Number of parallel processes for FVMD keypoint tracking (CPU-based parallelization).
    # Performance impact: More processes = faster tracking up to CPU core limit; too many causes overhead and contention.
    "fvmd_processes": [64],
    # Early stopping threshold for FVMD: stop if FD change is below this delta.
    # Performance impact: Higher delta = stops earlier, faster but less precise; lower = more accurate but slower.
    "fvmd_early_stop_delta": [0.1],
    # Window size (number of frames) for checking FVMD early stopping condition.
    # Performance impact: Larger window = more conservative stopping, longer runtime; smaller = aggressive stopping, faster.
    "fvmd_early_stop_window": [50],
    # Stride for VMAF calculation (frame sampling for video quality assessment).
    # Performance impact: Larger stride = faster VMAF computation but less granular quality profile; 1 = frame-by-frame, thorough.
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
