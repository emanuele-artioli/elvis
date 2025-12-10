"""PRESLEY: Extended ELVIS for Importance-Based Video Compression

PRESLEY extends ELVIS with multiple quality-reduction strategies that leverage
per-block importance scores to reduce bitrate while preserving foreground quality.
"""

import argparse
import sys
import os
from pathlib import Path
from typing import List, Optional
import numpy as np
import cv2
import torch

from evca import analyze_frames, EVCAConfig
from ufo import segment_frames

# from utils import *

from propainter import ProPainterModel, InpaintingConfig as ProPainterConfig
from e2fgvi import E2FGVIModel, InpaintingConfig as E2FGVIConfig
from realesrgan import create_upsampler
from instantir import InstantIRRuntime, load_runtime, restore_image
from upscale_a_video import UpscaleAVideo

import subprocess
import tempfile
import os
import time
import json
import io
import warnings
import logging
from pathlib import Path
from typing import List, Optional, Tuple, Callable, Dict, Any
from dataclasses import dataclass
from datetime import datetime
from functools import wraps, partial
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stdout, redirect_stderr

import numpy as np
import cv2
import torch
import pytorch_msssim


# =============================================================================
# Encoding Presets, Configuration and Setup
# =============================================================================

# Quality presets: maps quality names to encoder-specific parameters
QUALITY_PRESETS = {
    "lossless": {"kvazaar_qp": 2, "svtav1_crf": 2, "qp_range": 1, "description": "Lossless"},
    "high": {"kvazaar_qp": 30, "svtav1_crf": 50, "qp_range": 12, "description": "~400 kbps"},
    "medium": {"kvazaar_qp": 35, "svtav1_crf": 55, "qp_range": 12, "description": "~220 kbps, best efficiency"},
    "low": {"kvazaar_qp": 38, "svtav1_crf": 60, "qp_range": 12, "description": "~150 kbps"},
    "lowest": {"kvazaar_qp": 42, "svtav1_crf": 63, "qp_range": 10, "description": "~85 kbps, size-reducing"},
}

@dataclass
class PresleyConfig:
    """Configuration for the ELVIS/PRESLEY encoding pipeline."""
    reference_video: str = "/home/itec/emanuele/Datasets/DAVIS/avc_encoded/bear.mp4"
    width: int = 1280  # Reduced for faster testing (was 1280)
    height: int = 720  # Reduced for faster testing (was 720)
    max_frames: int = 5  # Limit frames for testing (None = all frames)
    framerate: float = None  # Auto-detected from reference video
    quality: str = "low"  # Quality preset (see QUALITY_PRESETS) - best efficiency
    qp_range: int = None  # Auto-set from quality preset, or override manually
    block_size: int = 16  # Block size for importance calculation
    alpha: float = 0.5  # Weight for spatial vs temporal complexity
    beta: float = 0.5  # Smoothing factor for importance scores
    shrink_amount: float = 0.25  # Fraction of blocks to remove in shrinking
    # Inpainting parameters (reduced for faster testing)
    propainter_ref_stride: int = 10  # Reduced from 20
    propainter_neighbor_length: int = 3  # Reduced from 4
    propainter_subvideo_length: int = 20  # Reduced from 40
    propainter_mask_dilation: int = 4
    propainter_raft_iter: int = 10  # Reduced from 20
    propainter_fp16: bool = True
    e2fgvi_ref_stride: int = 5  # Reduced from 10
    e2fgvi_neighbor_stride: int = 3  # Reduced from 5
    e2fgvi_num_ref: int = 3  # Changed from -1 to limit refs
    e2fgvi_mask_dilation: int = 4
    # Degradation parameters (recommended: downsample with scale=2 for best quality)
    downsample_max_scale: int = 2  # Reduced from 4 for testing
    blur_max_rounds: int = 3  # Reduced from 5 for testing
    # Restoration parameters - shared across all neural methods
    context_halo: int = 8  # Context halo/tile_pad in pixels (reduces tile edge artifacts)
    neural_tile_size: int = 128  # Reduced from 256 for faster processing
    temporal_blend: float = 0.1  # Blend factor for temporal consistency (0=none, higher=more smoothing)
    # RealESRGAN parameters
    realesrgan_blend_base: float = 0.85  # High blend with original (only 15% SR) for SSIM preservation
    realesrgan_denoise_strength: float = 0.3  # Low denoise to preserve original detail
    realesrgan_pre_pad: int = 0  # Pre-padding for input
    realesrgan_fp32: bool = False  # Use FP32 instead of FP16
    # InstantIR parameters
    instantir_cfg: float = 2.0  # Very low CFG for minimal hallucination
    instantir_creative_start: float = 1.0  # Max fidelity - creative mode disabled
    instantir_preview_start: float = 0.0
    instantir_seed: int = 42
    instantir_steps: int = 10  # Reduced steps for faster/less hallucination
    # Upscale-A-Video parameters (NOTE: very VRAM intensive)
    uav_tile_size: int = 64  # Reduced from 128 for faster testing
    uav_noise_level: int = 50  # Reduced significantly for less hallucination
    uav_guidance_scale: float = 2.0  # Very low for more faithful restoration
    uav_inference_steps: int = 10  # Reduced for faster testing
    uav_chunk_size: int = 4  # Reduced from 8 for faster testing
    uav_chunk_overlap: int = 1  # Reduced from 2


#TODO: create class frames, with numpy frames and metadata, so functions can get it in I/O and decorator can check for that instead of assuming list of np.ndarray.


def parse_and_update_config(config_obj):
    parser = argparse.ArgumentParser(description="PRESLEY Ablation Test")
    
    # Add arguments for config fields
    for field in PresleyConfig.__dataclass_fields__:
        default_val = getattr(config_obj, field)
        field_type = type(default_val) if default_val is not None else str
        # Handle boolean types specifically for argparse
        if field_type == bool:
            parser.add_argument(f"--{field}", action="store_true" if not default_val else "store_false", help=f"Toggle {field} (Default: {default_val})")
        else:
            parser.add_argument(f"--{field}", type=field_type, default=default_val, help=f"Default: {default_val}")
            
    args = parser.parse_args()
    
    # Update config
    for field in PresleyConfig.__dataclass_fields__:
        if hasattr(args, field):
            setattr(config_obj, field, getattr(args, field))
    
    return config_obj


def calculate_importance_scores(frames: List[np.ndarray], block_size: int, alpha: float, beta: float, complexities: np.ndarray, foreground_masks: np.ndarray) -> List[np.ndarray]:
    """Calculate per-block importance scores combining complexity and foreground detection."""
    
    # Combine spatial and temporal complexity
    complexity = np.zeros_like(complexities.SC)
    complexity[:-1] = alpha * complexities.SC[:-1] + (1 - alpha) * complexities.TC[1:]
    complexity[-1] = complexities.SC[-1]  # Last frame has no future TC

    # Temporal smoothing
    importance = np.zeros_like(complexity)
    importance[0] = complexity[0]
    importance[1:] = beta * complexity[1:] + (1 - beta) * complexity[:-1]

    # Invert importance for background (foreground_masks < 0.5)
    fg = foreground_masks.copy()
    fg[fg < 0.5] = -1.0
    importance *= fg
    
    # Normalize to [0, 1]
    min_val = importance.min(axis=(1, 2), keepdims=True)
    max_val = importance.max(axis=(1, 2), keepdims=True)
    importance = (importance - min_val) / (max_val - min_val + 1e-8)

    return [importance[i] for i in range(len(importance))]


# Global config instance
config = PresleyConfig()

# Parse command-line arguments and update configuration parameters
config = parse_and_update_config(config)

# Setup experiment
experiment_name = f"{Path(config.reference_video).stem}_{config.height}p_{config.quality}_ablation"
print(f"Experiment: {experiment_name}")
experiment_folder = Path("/home/itec/emanuele/elvis/experiments") / experiment_name
experiment_folder.mkdir(parents=True, exist_ok=True)

# Load video
cap = cv2.VideoCapture(config.reference_video)
config.framerate = cap.get(cv2.CAP_PROP_FPS)
reference_frames = []
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Resize frame to target resolution
    frame = cv2.resize(frame, (config.width, config.height), interpolation=cv2.INTER_AREA)
    reference_frames.append(frame)
cap.release()

# Limit frames for faster testing
if config.max_frames is not None and len(reference_frames) > config.max_frames:
    reference_frames = reference_frames[:config.max_frames]
    print(f"Limited to {config.max_frames} frames for testing")

# Generate importance scores 
print("Analyzing complexity and foreground...")
evca = analyze_frames(np.array(reference_frames), EVCAConfig(block_size=config.block_size))
ufo_masks = segment_frames(np.array(reference_frames), device="cuda" if torch.cuda.is_available() else "cpu")
ufo_masks = np.array([cv2.resize(m.astype(np.float32), (config.width // config.block_size, config.height // config.block_size), interpolation=cv2.INTER_NEAREST) for m in ufo_masks])
importance_scores = calculate_importance_scores(reference_frames, config.block_size, config.alpha, config.beta, evca, ufo_masks)


# =============================================================================
# Performance Functions 
# =============================================================================

def calculate_block_ssim(frames1: List[np.ndarray], frames2: List[np.ndarray], block_size: int, device: str = "cuda" if torch.cuda.is_available() else "cpu") -> List[np.ndarray]:
    """Calculate per-block SSIM between two frame sequences using GPU."""
    ssim_maps = []
    
    for f1, f2 in zip(frames1, frames2):
        h, w = f1.shape[:2]
        blocks_y, blocks_x = h // block_size, w // block_size
        
        # Crop to block boundaries and convert to tensors
        f1_crop = f1[:blocks_y * block_size, :blocks_x * block_size]
        f2_crop = f2[:blocks_y * block_size, :blocks_x * block_size]
        
        t1 = torch.from_numpy(f1_crop.copy()).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
        t2 = torch.from_numpy(f2_crop.copy()).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
        
        # Extract blocks using unfold
        patches1 = t1.unfold(2, block_size, block_size).unfold(3, block_size, block_size)
        patches2 = t2.unfold(2, block_size, block_size).unfold(3, block_size, block_size)
        
        c = patches1.shape[1]
        patches1 = patches1.permute(0, 2, 3, 1, 4, 5).reshape(-1, c, block_size, block_size)
        patches2 = patches2.permute(0, 2, 3, 1, 4, 5).reshape(-1, c, block_size, block_size)
        
        # Compute SSIM per block in batches
        num_blocks = patches1.shape[0]
        ssim_values = torch.zeros(num_blocks, device=device)
        batch_size = min(256, num_blocks)
        
        for i in range(0, num_blocks, batch_size):
            end = min(i + batch_size, num_blocks)
            ssim_values[i:end] = pytorch_msssim.ssim(patches1[i:end], patches2[i:end], data_range=1.0, size_average=False)
        ssim_maps.append(ssim_values.reshape(blocks_y, blocks_x).cpu().numpy())
    
    return ssim_maps


def compute_fg_bg_ssim(ssim_maps: List[np.ndarray], foreground_masks: np.ndarray, fg_threshold: float = 0.5) -> Tuple[float, float, float]:
    """Compute foreground/background SSIM from pre-calculated block SSIM maps.
    
    Uses output from calculate_block_ssim (GPU-accelerated) to efficiently
    compute separate metrics for foreground and background regions.
    
    ELVIS/PRESLEY's goal is to preserve foreground quality while degrading background
    to save bitrate, then restore background on the client. This metric helps evaluate
    how well this strategy works.
    
    Args:
        ssim_maps: Per-block SSIM maps from calculate_block_ssim (list of 2D arrays)
        foreground_masks: Per-block foreground masks, shape (num_frames, blocks_y, blocks_x)
                         Values in [0, 1] where 1=foreground, 0=background
        fg_threshold: Threshold to classify blocks as foreground (default 0.5)
    
    Returns:
        Tuple of (overall_ssim, foreground_ssim, background_ssim)
    """
    all_ssim = []
    fg_ssim = []
    bg_ssim = []
    
    for i, ssim_map in enumerate(ssim_maps):
        fg_mask = foreground_masks[i] if i < len(foreground_masks) else foreground_masks[0]
        
        # Resize mask to match SSIM map if shapes don't match
        if fg_mask.shape != ssim_map.shape:
            fg_mask = cv2.resize(fg_mask.astype(np.float32), (ssim_map.shape[1], ssim_map.shape[0]), interpolation=cv2.INTER_NEAREST)
        
        # Classify blocks as foreground or background
        is_foreground = fg_mask >= fg_threshold
        is_background = ~is_foreground
        
        # Collect SSIM values by region
        all_ssim.extend(ssim_map.flatten())
        if is_foreground.any():
            fg_ssim.extend(ssim_map[is_foreground])
        if is_background.any():
            bg_ssim.extend(ssim_map[is_background])
    
    overall = float(np.mean(all_ssim)) if all_ssim else 0.0
    fg = float(np.mean(fg_ssim)) if fg_ssim else overall  # Default to overall if no FG
    bg = float(np.mean(bg_ssim)) if bg_ssim else overall  # Default to overall if no BG
    
    return overall, fg, bg


def measure_performance(reference_frames: List[np.ndarray], foreground_masks: Optional[np.ndarray] = None, block_size: int = 16) -> Callable:
    """Decorator to measure performance of functions that return frames.
    
    Measures execution speed (in fps) and computes SSIM metrics by comparing output frames
    with reference frames. The decorated function returns a tuple of
    (original_result, performance_metrics).
    
    Args:
        reference_frames: Ground truth frames for comparison
        foreground_masks: Optional per-block foreground masks for FG/BG SSIM
                         Shape: (num_frames, blocks_y, blocks_x)
        block_size: Block size for SSIM calculation
    
    Returns:
        Decorator that wraps the function to return (result, metrics) tuple
        where metrics contains:
            - execution_speed: Execution speed in frames per second (fps)
            - result_frames: Output frames from the function
            - block_ssim_maps: Per-block SSIM maps
            - overall_ssim: Mean SSIM across all blocks
            - foreground_ssim: Mean SSIM for foreground blocks
            - background_ssim: Mean SSIM for background blocks
            - status: "success" or "error"
            - error: Error message if failed
    
    Example:
        @measure_performance(reference_frames, ufo_masks, block_size=16)
        def restore_frames(frames, **kwargs):
            return restored_frames
        
        result, metrics = restore_frames(degraded_frames, param=value)
        print(f"Speed: {metrics['execution_speed']:.2f} fps, SSIM: {metrics['overall_ssim']:.4f}")
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Tuple[Any, Dict[str, Any]]:
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                status = "success"
                error_msg = None
            except Exception as e:
                status = "error"
                error_msg = str(e)
                elapsed_time = time.time() - start_time
                frame_count = len(reference_frames)
                metrics = {
                    "execution_speed": frame_count / elapsed_time if elapsed_time > 0 else 0.0,
                    "result_frames": None,
                    "block_ssim_maps": None,
                    "overall_ssim": 0.0,
                    "foreground_ssim": 0.0,
                    "background_ssim": 0.0,
                    "status": status,
                    "error": error_msg
                }
                raise  # Re-raise the exception after recording metrics
            
            elapsed_time = time.time() - start_time
            frame_count = len(reference_frames)
            
            # Extract frames from result (handle various return types)
            if isinstance(result, list) and len(result) > 0 and isinstance(result[0], np.ndarray):
                result_frames = result
            elif isinstance(result, tuple):
                # Assume first element is frames if tuple
                result_frames = result[0] if isinstance(result[0], list) else result
            else:
                result_frames = result
            
            # Compute block-wise SSIM
            device = "cuda" if torch.cuda.is_available() else "cpu"
            block_ssim_maps = calculate_block_ssim(reference_frames, result_frames, block_size, device=device)
            
            # Compute overall and FG/BG SSIM
            if foreground_masks is not None:
                overall_ssim, fg_ssim, bg_ssim = compute_fg_bg_ssim(block_ssim_maps, foreground_masks)
            else:
                # Compute only overall SSIM
                all_ssim = [ssim_map.flatten() for ssim_map in block_ssim_maps]
                overall_ssim = float(np.mean(np.concatenate(all_ssim)))
                fg_ssim = overall_ssim
                bg_ssim = overall_ssim
            
            metrics = {
                "execution_speed": frame_count / elapsed_time if elapsed_time > 0 else 0.0,
                "result_frames": result_frames,
                "block_ssim_maps": block_ssim_maps,
                "overall_ssim": overall_ssim,
                "foreground_ssim": fg_ssim,
                "background_ssim": bg_ssim,
                "status": status,
                "error": error_msg
            }
            
            return result, metrics
        
        return wrapper
    return decorator


# =============================================================================
# Video I/O Functions
# =============================================================================

@measure_performance(reference_frames, foreground_masks=ufo_masks, block_size=config.block_size)
def load_frames(video_path: str, width: int, height: int) -> List[np.ndarray]:
    """Load video frames as RGB numpy arrays, scaled to target resolution."""
    command = [
        'ffmpeg', '-hide_banner', '-loglevel', 'warning',
        "-i", video_path,
        "-vf", f"scale={width}:{height}",
        "-f", "image2pipe",
        "-pix_fmt", "rgb24",
        "-vcodec", "rawvideo",
        "-"
    ]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    frames = []
    frame_size = width * height * 3
    while True:
        raw_frame = process.stdout.read(frame_size)
        if len(raw_frame) < frame_size:
            break
        frame = np.frombuffer(raw_frame, np.uint8).reshape((height, width, 3))
        frames.append(frame)
    process.stdout.close()
    process.wait()
    return frames


def save_frames(frames: List[np.ndarray], output_folder: Path) -> None:
    """Save frames as PNG images."""
    output_folder.mkdir(parents=True, exist_ok=True)
    for i, frame in enumerate(frames):
        frame_path = output_folder / f"frame_{i:05d}.png"
        cv2.imwrite(str(frame_path), frame)


def encode_video(frames: List[np.ndarray], output_path: str, quality: str = "medium", qp_range: int = None, importance_scores: Optional[List[np.ndarray]] = None, encoder: str = "kvazaar") -> None:
    """Encode video with specified encoder and quality preset."""
    
    if quality not in QUALITY_PRESETS:
        raise ValueError(f"Unknown quality preset: {quality}. Available: {list(QUALITY_PRESETS.keys())}")
    
    preset = QUALITY_PRESETS[quality]
    
    # Use preset qp_range if not explicitly specified
    effective_qp_range = qp_range if qp_range is not None else preset.get("qp_range", 12)
    
    if encoder == "kvazaar":
        encode_kvazaar(frames, output_path, framerate=config.framerate, qp=preset["kvazaar_qp"],  qp_range=effective_qp_range, importance_scores=importance_scores)
    elif encoder == "svtav1":
        encode_svtav1(frames, output_path, framerate=config.framerate, crf=preset["svtav1_crf"],  qp_range=effective_qp_range, importance_scores=importance_scores)
    else:
        raise ValueError(f"Unknown encoder: {encoder}. Supported: kvazaar, svtav1")


def write_y4m(frames: List[np.ndarray], y4m_path: str, framerate: float) -> None:
    """Write frames to Y4M file with YUV420 color space."""
    height, width = frames[0].shape[:2]
    fps_num = int(round(framerate * 1000))
    with open(y4m_path, 'wb') as f:
        f.write(f"YUV4MPEG2 W{width} H{height} F{fps_num}:1000 Ip A1:1 C420\n".encode())
        for frame in frames:
            f.write(b"FRAME\n")
            yuv = cv2.cvtColor(frame, cv2.COLOR_RGB2YUV_I420)
            f.write(yuv.tobytes())


def encode_kvazaar(frames: List[np.ndarray], output_path: str, framerate: float, qp: int = 48, qp_range: int = 15, importance_scores: Optional[List[np.ndarray]] = None) -> None:
    """Encode video using Kvazaar HEVC encoder with optional ROI-based quality control.
    
    Kvazaar ROI features:
    - Block size: 16x16 CTU (configurable via importance score resolution)
    - QP range: Continuous (-127 to +127)
    - File format: Binary (width, height as int32, then int8 delta QP array per frame)
    """
    height, width = frames[0].shape[:2]
    output_path = Path(output_path)
    
    # Create temporary Y4M file
    with tempfile.NamedTemporaryFile(suffix='.y4m', delete=False) as tmp:
        y4m_path = tmp.name
    write_y4m(frames, y4m_path, framerate)
    
    # Build kvazaar command
    hevc_path = str(output_path).replace('.mp4', '.hevc')
    cmd = ["kvazaar", "-i", y4m_path, "-q", str(qp), "-o", hevc_path, "--preset", "medium"]
    
    # Add ROI file if importance scores provided
    roi_path = None
    if importance_scores is not None:
        roi_path = str(output_path).replace('.mp4', '_roi.bin')
        create_kvazaar_roi_file(importance_scores, roi_path, base_qp=qp, qp_range=qp_range)
        cmd.extend(["--roi", roi_path])
    
    # Run kvazaar (may crash at end due to memory bug, but output is valid)
    subprocess.run(cmd, capture_output=True, text=True)
    
    if not os.path.exists(hevc_path) or os.path.getsize(hevc_path) == 0:
        raise RuntimeError(f"Kvazaar failed to produce output: {hevc_path}")
    
    # Mux HEVC to MP4 using mkvmerge for proper timestamps
    mkv_path = hevc_path.replace('.hevc', '.mkv')
    subprocess.run([
        "mkvmerge", "-o", mkv_path,
        "--default-duration", f"0:{int(framerate)}fps",
        hevc_path
    ], check=True, capture_output=True)
    
    subprocess.run([
        "ffmpeg", "-hide_banner", "-loglevel", "warning", "-y",
        "-i", mkv_path, "-c:v", "copy", "-movflags", "+faststart",
        str(output_path)
    ], check=True, capture_output=True)
    
    # Cleanup temporary files
    os.unlink(y4m_path)
    os.unlink(hevc_path)
    os.unlink(mkv_path)
    if roi_path:
        os.unlink(roi_path)


def encode_svtav1(frames: List[np.ndarray], output_path: str, framerate: float, crf: int = 35, qp_range: int = 15, importance_scores: Optional[List[np.ndarray]] = None) -> None:
    """Encode video using SVT-AV1 encoder with optional ROI-based quality control.
    
    SVT-AV1 ROI features:
    - Block size: 64x64 superblock (FIXED - AV1 architecture constraint)
    - QP levels: 8 distinct levels (AV1 segment limit)
    - File format: Text (frame_num offset1 offset2 ... per line)
    """
    height, width = frames[0].shape[:2]
    output_path = Path(output_path)
    
    # Create temporary Y4M file
    with tempfile.NamedTemporaryFile(suffix='.y4m', delete=False) as tmp:
        y4m_path = tmp.name
    write_y4m(frames, y4m_path, framerate)
    
    # Build SVT-AV1 command
    ivf_path = str(output_path).replace('.mp4', '.ivf')
    cmd = ["SvtAv1EncApp", "-i", y4m_path, "-b", ivf_path, "--preset", "8", "--crf", str(crf)]
    
    # Add ROI file if importance scores provided
    roi_path = None
    if importance_scores is not None:
        roi_path = str(output_path).replace('.mp4', '_roi.txt')
        create_svtav1_roi_file(importance_scores, roi_path, base_crf=crf, qp_range=qp_range, 
                               width=width, height=height)
        cmd.extend(["--roi-map-file", roi_path])
    
    # Run SVT-AV1
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if not os.path.exists(ivf_path) or os.path.getsize(ivf_path) == 0:
        raise RuntimeError(f"SVT-AV1 failed: {result.stderr}")
    
    # Mux IVF to MP4
    subprocess.run([
        "ffmpeg", "-hide_banner", "-loglevel", "warning", "-y",
        "-i", ivf_path, "-c:v", "copy", "-movflags", "+faststart",
        str(output_path)
    ], check=True, capture_output=True)
    
    # Cleanup
    os.unlink(y4m_path)
    os.unlink(ivf_path)
    if roi_path:
        os.unlink(roi_path)


# =============================================================================
# ELVIS: Frame Shrinking and Stretching Functions
# Remove low-importance blocks to reduce frame size. The removed blocks can
# later be reconstructed via video inpainting (e.g., ProPainter, E2FGVI).
# =============================================================================

# Shrinking Method 1: Row-only removal (original ELVIS style)

def shrink_frame_row_only(frame: np.ndarray, importance: np.ndarray, block_size: int, shrink_amount: float) -> Tuple[np.ndarray, np.ndarray]:
    """Remove blocks only from rows (columns shrink uniformly).
    
    Returns:
        shrunken: The shrunken frame with reduced width
        removal_mask: Boolean mask of original block positions (True = removed)
    """
    frame = frame.copy()
    importance = importance.copy()
    
    height, width = frame.shape[:2]
    blocks_y = height // block_size
    blocks_x = width // block_size
    orig_blocks_y, orig_blocks_x = blocks_y, blocks_x
    
    blocked = frame[:blocks_y * block_size, :blocks_x * block_size].reshape(
        blocks_y, block_size, blocks_x, block_size, 3)
    
    removal_mask = np.zeros((orig_blocks_y, orig_blocks_x), dtype=bool)
    target_removals = int(orig_blocks_y * orig_blocks_x * shrink_amount)
    removed = 0
    
    while removed < target_removals and blocks_x > 1:
        # Remove one block from each row
        for by in range(blocks_y):
            if removed >= target_removals:
                break
            least_idx = np.argmin(importance[by, :blocks_x])
            # Mark in removal mask - find which original column this maps to
            # Since we only remove from rows, the column index directly maps
            # We need to track which original columns remain
            remaining_cols = np.where(~removal_mask[by, :])[0]
            if len(remaining_cols) > least_idx:
                removal_mask[by, remaining_cols[least_idx]] = True
            
            # Shift blocks left
            blocked[by, :, least_idx:blocks_x-1] = blocked[by, :, least_idx+1:blocks_x].copy()
            importance[by, least_idx:blocks_x-1] = importance[by, least_idx+1:blocks_x]
            removed += 1
        
        blocks_x -= 1
        importance = importance[:, :blocks_x]
    
    shrunken = blocked[:blocks_y, :, :blocks_x].reshape(blocks_y * block_size, blocks_x * block_size, 3)
    return shrunken, removal_mask

# Shrinking Method 2: Row+Col removal with removal_indices list

def shrink_frame_removal_indices(frame: np.ndarray, importance: np.ndarray, block_size: int, shrink_amount: float) -> Tuple[np.ndarray, list]:
    """Remove blocks from both rows and columns, tracking removed indices per pass.
    
    Returns:
        shrunken: The shrunken frame with reduced dimensions
        removal_indices: List of arrays - alternating row-pass and col-pass removals.
                        Element 0: indices removed from each row (length = orig_blocks_y)
                        Element 1: indices removed from each col (length = orig_blocks_x - 1)
                        Element 2: indices removed from each row (length = orig_blocks_y - 1)
                        etc.
    """
    frame = frame.copy()
    importance = importance.copy()
    
    height, width = frame.shape[:2]
    blocks_y = height // block_size
    blocks_x = width // block_size
    orig_blocks_y, orig_blocks_x = blocks_y, blocks_x
    
    blocked = frame[:blocks_y * block_size, :blocks_x * block_size].reshape(
        blocks_y, block_size, blocks_x, block_size, 3)
    
    removal_indices = []  # List of arrays tracking removed indices per pass
    target_removals = int(orig_blocks_y * orig_blocks_x * shrink_amount)
    removed = 0
    
    while removed < target_removals:
        # Row pass: remove one block from each row
        row_indices = []
        row_pass_complete = True
        for by in range(blocks_y):
            if removed >= target_removals:
                row_pass_complete = False
                break
            least_idx = np.argmin(importance[by, :blocks_x])
            row_indices.append(least_idx)
            
            blocked[by, :, least_idx:blocks_x-1] = blocked[by, :, least_idx+1:blocks_x].copy()
            importance[by, least_idx:blocks_x-1] = importance[by, least_idx+1:blocks_x]
            removed += 1
        
        if row_indices:
            removal_indices.append(np.array(row_indices, dtype=np.int32))
        
        if row_pass_complete:
            blocks_x -= 1
            importance = importance[:, :blocks_x]
        
        if removed >= target_removals:
            break
        
        # Column pass: remove one block from each column
        col_indices = []
        col_pass_complete = True
        for bx in range(blocks_x):
            if removed >= target_removals:
                col_pass_complete = False
                break
            least_idx = np.argmin(importance[:blocks_y, bx])
            col_indices.append(least_idx)
            
            blocked[least_idx:blocks_y-1, :, bx] = blocked[least_idx+1:blocks_y, :, bx].copy()
            importance[least_idx:blocks_y-1, bx] = importance[least_idx+1:blocks_y, bx]
            removed += 1
        
        if col_indices:
            removal_indices.append(np.array(col_indices, dtype=np.int32))
        
        if col_pass_complete:
            blocks_y -= 1
            importance = importance[:blocks_y, :]
    
    shrunken = blocked[:blocks_y, :, :blocks_x].reshape(blocks_y * block_size, blocks_x * block_size, 3)
    return shrunken, removal_indices


def _removal_indices_to_mask(removal_indices: list, orig_blocks_y: int, orig_blocks_x: int) -> np.ndarray:
    """Convert removal indices to boolean mask.
    
    Args:
        removal_indices: List of arrays tracking removed indices per pass
        orig_blocks_y: Original number of blocks vertically
        orig_blocks_x: Original number of blocks horizontally
    
    Returns:
        Boolean mask where True = removed block
    """
    mask = np.zeros((orig_blocks_y, orig_blocks_x), dtype=bool)
    
    # Track current grid dimensions as we replay the removal process
    blocks_y, blocks_x = orig_blocks_y, orig_blocks_x
    
    # Process removal_indices forward to mark removed positions
    for pass_idx, indices in enumerate(removal_indices):
        is_row_pass = (pass_idx % 2 == 0)  # Even indices are row passes
        
        if is_row_pass:
            # Row pass: removed blocks from each row
            for by in range(min(len(indices), blocks_y)):
                remove_idx = indices[by]
                if remove_idx < blocks_x:
                    # Find the actual column position in original grid
                    # Count how many non-removed positions come before this one
                    remaining_cols = np.where(~mask[by, :])[0]
                    if remove_idx < len(remaining_cols):
                        mask[by, remaining_cols[remove_idx]] = True
            blocks_x -= 1
        else:
            # Column pass: removed blocks from each column
            for bx in range(min(len(indices), blocks_x)):
                remove_idx = indices[bx]
                if remove_idx < blocks_y:
                    # Find the actual row position in original grid
                    # Count how many non-removed positions come before this one
                    remaining_rows = np.where(~mask[:, bx])[0]
                    if remove_idx < len(remaining_rows):
                        mask[remaining_rows[remove_idx], bx] = True
            blocks_y -= 1
    
    return mask

# Wrapper functions

def shrink_video_frames(frames: List[np.ndarray], importance_scores: List[np.ndarray], block_size: int, shrink_amount: float, method: Callable) -> Tuple[List[np.ndarray], List[Any]]:
    """Shrink a list of video frames using the specified shrinking method.
    
    Args:
        frames: List of input frames (H, W, 3)
        importance_scores: List of importance score arrays (blocks_y, blocks_x)
        block_size: Size of blocks for shrinking
        shrink_amount: Fraction of blocks to remove
        method: Shrinking method function to use
    
    Returns:
        shrunken_frames: List of shrunken frames
        removal_masks: List of removal masks per frame
    """
    shrunken_frames = []
    removal_masks = []
    aux_data = []
    
    for frame, importance in zip(frames, importance_scores):
        shrunken, removal_mask = method(frame, importance, block_size, shrink_amount)
        shrunken_frames.append(shrunken)
        removal_masks.append(removal_mask)
    
    return shrunken_frames, removal_masks


def stretch_video_frames(shrunken_frames: List[np.ndarray], removal_masks: List[Any], block_size: int) -> List[np.ndarray]:
    """Stretch a list of shrunken video frames using the removal masks.
    
    Args:
        shrunken_frames: List of shrunken frames (H', W', 3)
        removal_masks: List of binary removal masks per frame
        block_size: Size of blocks for stretching
    
    Returns:
        stretched_frames: List of reconstructed frames
    """
    stretched_frames = []
    
    for frame_idx, frame in enumerate(shrunken_frames):
        removal_mask = removal_masks[frame_idx]
        orig_blocks_y, orig_blocks_x = removal_mask.shape
        h, w, c = frame.shape
        shrunk_blocks_y, shrunk_blocks_x = h // block_size, w // block_size
        
        shrunk_blocks = frame.reshape(shrunk_blocks_y, block_size, shrunk_blocks_x, block_size, c).swapaxes(1, 2)
        final_blocks = np.zeros((orig_blocks_y, orig_blocks_x, block_size, block_size, c), dtype=frame.dtype)
        
        # For each row, place shrunk blocks at non-removed positions
        for orig_y in range(orig_blocks_y):
            kept_cols = np.where(~removal_mask[orig_y, :])[0]
            for shrunk_x, orig_x in enumerate(kept_cols):
                if shrunk_x < shrunk_blocks_x:
                    final_blocks[orig_y, orig_x] = shrunk_blocks[orig_y, shrunk_x]
        
        stretched = final_blocks.swapaxes(1, 2).reshape(orig_blocks_y * block_size, orig_blocks_x * block_size, c)
        stretched_frames.append(stretched)
    
    return stretched_frames


# =============================================================================
# ELVIS: Video Inpainting
# Use video inpainting models to restore removed regions in shrunken frames.
# Supported models: OpenCV Telea, ProPainter, E2FGVI
# =============================================================================


@measure_performance(reference_frames, foreground_masks=ufo_masks, block_size=config.block_size)
def inpaint_with_opencv(frames: np.ndarray, masks: np.ndarray) -> np.ndarray:
    """Inpaint frames using OpenCV's Telea method."""
    inpainted = []
    num_frames = frames.shape[0]
    
    for i in range(num_frames):
        frame = frames[i]
        mask = masks[i].astype(np.uint8) * 255  # Convert boolean to uint8 mask
        mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST) # Resize masks to frame size
        inpainted_frame = cv2.inpaint(frame, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
        inpainted.append(inpainted_frame)
    
    return np.array(inpainted)


@measure_performance(reference_frames, foreground_masks=ufo_masks, block_size=config.block_size)
def inpaint_with_propainter(frames: np.ndarray, masks: np.ndarray, model: ProPainterModel, device: str = "cuda") -> np.ndarray:
    """Inpaint frames using ProPainter."""
    pp_config = ProPainterConfig(
        mask_dilation=config.propainter_mask_dilation,
        ref_stride=config.propainter_ref_stride,
        neighbor_length=config.propainter_neighbor_length,
        subvideo_length=config.propainter_subvideo_length,
        raft_iter=config.propainter_raft_iter,
        fp16=config.propainter_fp16,
        device=torch.device(device)
    )
    masks = [cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST) for mask in masks] # Resize masks to frame size
    return model.inpaint(frames, masks, config=pp_config)


@measure_performance(reference_frames, foreground_masks=ufo_masks, block_size=config.block_size)
def inpaint_with_e2fgvi(frames: np.ndarray, masks: np.ndarray, model: E2FGVIModel, device: str = "cuda") -> np.ndarray:
    """Inpaint frames using E2FGVI."""
    e2_config = E2FGVIConfig(
        model="e2fgvi_hq",
        step=config.e2fgvi_ref_stride,
        num_ref=config.e2fgvi_num_ref,
        neighbor_stride=config.e2fgvi_neighbor_stride,
        mask_dilation=config.e2fgvi_mask_dilation,
        device=torch.device(device)
    )
    masks = [cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST) for mask in masks] # Resize masks to frame size
    return model.inpaint(frames, masks, config=e2_config)


# =============================================================================
# PRESLEY: ROI-Based Encoding
# Kvazaar (HEVC) and SVT-AV1 both support external per-block delta QP maps.
# =============================================================================

def create_kvazaar_roi_file(importance_scores: List[np.ndarray], roi_path: str, base_qp: int, qp_range: int = 15) -> None:
    """Create a Kvazaar ROI file from importance scores.
    
    Format: Binary file with per-frame data:
      - width (int32): Number of CTUs horizontally
      - height (int32): Number of CTUs vertically  
      - delta_qp[height][width] (int8): Delta QP for each CTU
    
    CTU size is determined by importance score resolution (typically 16x16).
    Delta QP is clamped to kvazaar's limit (approximately ±14) and HEVC valid range.
    """
    # Kvazaar's internal delta QP limit (empirically determined)
    KVAZAAR_DELTA_LIMIT = 14
    # HEVC valid QP range
    MIN_QP, MAX_QP = 0, 51
    
    with open(roi_path, 'wb') as f:
        for importance in importance_scores:
            h, w = importance.shape
            f.write(np.array([w, h], dtype=np.int32).tobytes())
            # importance=1 (foreground) -> -qp_range (better quality)
            # importance=0 (background) -> +qp_range (lower quality)
            delta_qp = (1.0 - importance) * 2 * qp_range - qp_range
            # Clamp to kvazaar's internal limit
            delta_qp = np.clip(delta_qp, -KVAZAAR_DELTA_LIMIT, KVAZAAR_DELTA_LIMIT)
            # Also ensure final QP stays within HEVC valid range
            delta_qp = np.clip(delta_qp, MIN_QP - base_qp, MAX_QP - base_qp)
            f.write(delta_qp.astype(np.int8).tobytes())


def create_svtav1_roi_file(importance_scores: List[np.ndarray], roi_path: str, base_crf: int, qp_range: int, width: int, height: int) -> None:
    """Create an SVT-AV1 ROI map file from importance scores.
    
    Format: Text file with one line per frame:
      frame_num offset1 offset2 offset3 ... (for each 64x64 block in row order)
    
    IMPORTANT: AV1 spec limits to 8 segments, so importance is quantized to 8 levels.
    Block size is fixed at 64x64 (AV1 superblock architecture).
    Delta QP is clamped to ensure final QP stays within AV1 valid range (0-63).
    """
    BLOCK_SIZE = 64  # Fixed by AV1 superblock architecture
    NUM_SEGMENTS = 8  # AV1 segment limit
    # AV1 valid QP range
    MIN_QP, MAX_QP = 0, 63
    
    blocks_x = (width + BLOCK_SIZE - 1) // BLOCK_SIZE
    blocks_y = (height + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    with open(roi_path, 'w') as f:
        for frame_idx, importance in enumerate(importance_scores):
            # Resize importance to 64x64 block grid
            imp_resized = cv2.resize(importance.astype(np.float32), (blocks_x, blocks_y), 
                                     interpolation=cv2.INTER_AREA)
            
            # Quantize importance to 8 levels (0-7)
            levels = np.clip((imp_resized * NUM_SEGMENTS).astype(np.int32), 0, NUM_SEGMENTS - 1)
            
            # Map levels to QP offsets: higher importance -> lower QP (better quality)
            delta_qp = qp_range - (levels * 2 * qp_range // (NUM_SEGMENTS - 1))
            
            # Clamp delta_qp so final QP stays within valid range
            delta_qp = np.clip(delta_qp, MIN_QP - base_crf, MAX_QP - base_crf)
            
            # Write line: frame_num offset1 offset2 ...
            offsets = delta_qp.flatten().astype(int)
            line = f"{frame_idx} " + " ".join(map(str, offsets))
            f.write(line + "\n")


# =============================================================================
# PRESLEY: Adaptive Block Degradation
# Degrade low-importance blocks via downsampling or blur, encode, then restore.
# Downsampling uses scale factors 0/2/3/4 (2-bit representation per block).
# =============================================================================

def generate_degradation_map(importance: np.ndarray, max_value: int) -> np.ndarray:
    """Generate degradation map from importance scores by mapping importance to discrete levels."""
    # Invert importance
    inv_importance = 1 - importance
    # Map to discrete levels
    degradation_map = np.round(inv_importance * max_value).astype(np.int32)
    degradation_map = np.clip(degradation_map, 0, max_value)
    return degradation_map


def downscale_block(block: np.ndarray, scale: int) -> np.ndarray:
    """Downscale and then upscale a block by the given scale factor."""
    block_size = block.shape[0]
    small_size = max(1, block_size // scale)
    small = cv2.resize(block, (small_size, small_size), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (block_size, block_size), interpolation=cv2.INTER_LINEAR)


def blur_block(block: np.ndarray, rounds: int) -> np.ndarray:
    """Apply Gaussian blur to a block for a given number of rounds."""
    for _ in range(rounds):
        block = cv2.GaussianBlur(block, (5, 5), sigmaX=1.0)
    return block

# TODO: combine downscale and blur so that each block gets either or both based on what is best. Would require a more complex degradation map, that evaluates the best method per block.
def degrade_frame(frame: np.ndarray, degradation_map: np.ndarray, block_size: int, method: Callable) -> np.ndarray:
    """Degrade frame based on degradation map using specified method."""
    h, w = frame.shape[:2]
    blocks_y, blocks_x = h // block_size, w // block_size
    
    blocks = frame[:blocks_y * block_size, :blocks_x * block_size].reshape(blocks_y, block_size, blocks_x, block_size, 3).swapaxes(1, 2)
    
    processed_blocks = blocks.copy()
    for by in range(blocks_y):
        for bx in range(blocks_x):
            level = degradation_map[by, bx]
            if level > 0:
                block = blocks[by, bx]
                processed_blocks[by, bx] = method(block, level)
    
    result = processed_blocks.swapaxes(1, 2).reshape(blocks_y * block_size, blocks_x * block_size, 3)
    
    # Restore original frame size
    output = frame.copy()
    output[:blocks_y * block_size, :blocks_x * block_size] = result
    return output


def degrade_video_adaptive(frames: List[np.ndarray], importance_scores: List[np.ndarray], block_size: int, max_value: int, method: Callable) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Degrade video frames adaptively based on importance scores.
    
    Args:
        frames: List of input frames
        importance_scores: List of per-block importance scores
        block_size: Block size for degradation
        max_value: Maximum degradation level
        method: Degradation method (downscale_block or blur_block)
    
    Returns:
        degraded_frames: List of degraded frames
        degradation_maps: List of per-frame degradation maps
    """
    degraded_frames = []
    degradation_maps = []
    
    for frame, importance in zip(frames, importance_scores):
        degradation_map = generate_degradation_map(importance, max_value)
        degraded = degrade_frame(frame, degradation_map, block_size, method)
        degraded_frames.append(degraded)
        degradation_maps.append(degradation_map)
    
    return degraded_frames, degradation_maps




# =============================================================================
# PRESLEY: Restoration Models
# Restore degraded videos using neural super-resolution and image restoration.
# Includes both "naive" (whole-frame) and "adaptive" (per-block) variants.
# Tiled processing with context halo for large images and block edge smoothing.
# =============================================================================

def _extract_tile_with_halo(frame: np.ndarray, y: int, x: int, tile_h: int, tile_w: int, halo: int) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """Extract a tile with context halo from a frame.
    
    Returns:
        tile: The extracted tile with halo
        crop_bounds: (top, left, bottom, right) bounds to crop halo from result
    """
    h, w = frame.shape[:2]
    
    # Compute halo-extended bounds, clamped to frame
    y0 = max(0, y - halo)
    x0 = max(0, x - halo)
    y1 = min(h, y + tile_h + halo)
    x1 = min(w, x + tile_w + halo)
    
    tile = frame[y0:y1, x0:x1].copy()
    
    # Compute crop bounds to remove halo from processed result
    crop_top = y - y0
    crop_left = x - x0
    crop_bottom = crop_top + tile_h
    crop_right = crop_left + tile_w
    
    return tile, (crop_top, crop_left, crop_bottom, crop_right)


@measure_performance(reference_frames, foreground_masks=ufo_masks, block_size=config.block_size)
def restore_with_opencv_lanczos(frames: List[np.ndarray], degradation_maps: np.ndarray, block_size: int, halo: int = 0, temporal_blend: float = 0.0) -> List[np.ndarray]:
    """Restore downsampled frames using OpenCV sharpening (per-block).
    
    Applies adaptive sharpening based on degradation level (scale factor).
    Higher scale factor = more sharpening to compensate for lost detail.
    
    Args:
        frames: List of RGB frames
        degradation_maps: Per-frame scale maps (0=none, 2/3/4=scale factor)
        block_size: Block size in pixels
        halo: Context halo for block processing (reduces edge artifacts)
        temporal_blend: Blend factor with previous frame (0=none, reduces flicker)
    
    Returns:
        Restored frames list
    """
    restored = []
    prev_output = None  # For temporal blending
    
    for i, frame in enumerate(frames):
        h, w = frame.shape[:2]
        blocks_y, blocks_x = h // block_size, w // block_size
        deg_map = degradation_maps[i] if len(degradation_maps) > i else np.zeros((blocks_y, blocks_x))
        
        if deg_map.shape != (blocks_y, blocks_x):
            deg_map = cv2.resize(deg_map.astype(np.float32), (blocks_x, blocks_y), 
                                interpolation=cv2.INTER_NEAREST).astype(np.int32)
        
        output = frame.copy()
        
        for by in range(blocks_y):
            for bx in range(blocks_x):
                scale_factor = deg_map[by, bx]
                if scale_factor > 0:
                    y, x = by * block_size, bx * block_size
                    
                    # Extract block with halo
                    if halo > 0:
                        tile, crop = _extract_tile_with_halo(frame, y, x, block_size, block_size, halo)
                    else:
                        tile = frame[y:y+block_size, x:x+block_size].copy()
                        crop = (0, 0, block_size, block_size)
                    
                    # Sharpening proportional to scale factor
                    amount = scale_factor * 0.5  # Strength scales with downscale factor
                    radius = max(1, scale_factor)
                    blurred = cv2.GaussianBlur(tile, (0, 0), radius)
                    sharpened = cv2.addWeighted(tile, 1.0 + amount, blurred, -amount, 0)
                    sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
                    
                    # Crop halo and paste back
                    result_block = sharpened[crop[0]:crop[2], crop[1]:crop[3]]
                    output[y:y+block_size, x:x+block_size] = result_block
        
        # Apply temporal blending to reduce flickering between frames
        if temporal_blend > 0 and prev_output is not None:
            output = (temporal_blend * prev_output + (1 - temporal_blend) * output).astype(np.uint8)
        
        prev_output = output.copy()
        restored.append(output)
    
    return restored


@measure_performance(reference_frames, foreground_masks=ufo_masks, block_size=config.block_size)
def restore_with_opencv_unsharp(frames: List[np.ndarray], degradation_maps: np.ndarray, block_size: int, halo: int = 0, temporal_blend: float = 0.0) -> List[np.ndarray]:
    """Restore blurred frames using OpenCV unsharp masking (per-block).
    
    Unsharp masking enhances edges by subtracting a blurred version of the image.
    The formula is: sharpened = original + amount * (original - blurred)
    Which is equivalent to: sharpened = (1 + amount) * original - amount * blurred
    
    The sharpening strength is adaptive based on the blur level in degradation_map:
    - amount = blur_level * 0.5 (more blur -> more sharpening)
    - radius = blur_level (larger blur kernel for more blurred blocks)
    
    Args:
        frames: List of RGB frames to restore
        degradation_maps: Per-frame blur level maps (higher = more blur applied)
        block_size: Block size in pixels (must match degradation grid)
        halo: Context halo in pixels around each block (reduces edge artifacts)
        temporal_blend: Blend factor with previous frame for temporal consistency
                       (0 = no blending, 0.1 = light smoothing to reduce flicker)
    
    Returns:
        Restored frames list with sharpening applied to blurred blocks
    """
    restored = []
    prev_output = None  # For temporal blending
    
    for i, frame in enumerate(frames):
        h, w = frame.shape[:2]
        blocks_y, blocks_x = h // block_size, w // block_size
        
        # Get degradation map for this frame, or use zeros if not available
        blur_map = degradation_maps[i] if len(degradation_maps) > i else np.zeros((blocks_y, blocks_x))
        
        # Resize degradation map to match block grid if dimensions differ
        if blur_map.shape != (blocks_y, blocks_x):
            blur_map = cv2.resize(blur_map.astype(np.float32), (blocks_x, blocks_y), 
                                 interpolation=cv2.INTER_NEAREST).astype(np.int32)
        
        output = frame.copy()
        
        # Process each block that was blurred (blur_level > 0)
        for by in range(blocks_y):
            for bx in range(blocks_x):
                blur_level = blur_map[by, bx]
                if blur_level > 0:
                    # Calculate pixel coordinates of this block
                    y, x = by * block_size, bx * block_size
                    
                    # Extract tile with context halo for better edge handling
                    if halo > 0:
                        tile, crop = _extract_tile_with_halo(frame, y, x, block_size, block_size, halo)
                    else:
                        tile = frame[y:y+block_size, x:x+block_size].copy()
                        crop = (0, 0, block_size, block_size)
                    
                    # Calculate sharpening parameters based on degradation level
                    # Higher blur_level means more aggressive sharpening is needed
                    amount = blur_level * 0.5  # Sharpening strength multiplier
                    radius = max(1, blur_level)  # Gaussian blur sigma for unsharp mask
                    
                    # Apply unsharp mask: sharp = (1+amount)*orig - amount*blurred
                    blurred = cv2.GaussianBlur(tile, (0, 0), radius)
                    sharpened = cv2.addWeighted(tile, 1.0 + amount, blurred, -amount, 0)
                    sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
                    
                    # Crop halo from result and paste into output
                    result_block = sharpened[crop[0]:crop[2], crop[1]:crop[3]]
                    output[y:y+block_size, x:x+block_size] = result_block
        
        # Apply temporal blending to reduce flickering between frames
        if temporal_blend > 0 and prev_output is not None:
            output = (temporal_blend * prev_output + (1 - temporal_blend) * output).astype(np.uint8)
        
        prev_output = output.copy()
        restored.append(output)
    
    return restored


@measure_performance(reference_frames, foreground_masks=ufo_masks, block_size=config.block_size)
def restore_video_adaptively(restore_fn: Callable, frames: List[np.ndarray], degradation_maps: List[np.ndarray], block_size: int = 16, **kwargs) -> List[np.ndarray]:
    """Wrapper for restoration function that runs it multiple times in parallel, each time with different parameters based on degradation levels, then takes blocks restored at their respective levels and combines them."""
    if not frames:
        return []
        
    h, w = frames[0].shape[:2]
    n_frames = len(frames)
    blocks_y = h // block_size
    blocks_x = w // block_size
    
    # Determine unique degradation levels across all frames
    unique_levels = set()
    for dmap in degradation_maps:
        unique_levels.update(np.unique(dmap))
    unique_levels = sorted(unique_levels)
    
    # Store restored versions per level
    restored_versions: Dict[float, List[np.ndarray]] = {}
    
    def _restore_for_level(level: float) -> Tuple[float, List[np.ndarray]]:
        # Modify kwargs to include current degradation level
        level_kwargs = kwargs.copy()
        level_kwargs['degradation_level'] = level
        result = restore_fn(frames=frames, **level_kwargs)
        
        # Handle both decorated (returns tuple) and non-decorated (returns frames) functions
        if isinstance(result, tuple) and len(result) == 2:
            # Decorated function: (frames, metrics) - extract just the frames
            restored = result[0]
        else:
            # Non-decorated function: just frames
            restored = result
        
        return (level, restored)

    # Run parallel restoration for each degradation level
    with ThreadPoolExecutor(max_workers=len(unique_levels)) as executor:
        futures = [executor.submit(_restore_for_level, level) for level in unique_levels]
        for future in futures:
            level, restored = future.result()
            restored_versions[level] = restored

    # Combine blocks from different restored versions based on degradation maps
    final_frames = []
    for i in range(n_frames):
        final_frame = np.zeros((h, w, 3), dtype=np.uint8)
        dmap = degradation_maps[i]
        
        for by in range(blocks_y):
            for bx in range(blocks_x):
                block_level = dmap[by, bx]
                final_frame[by * block_size:(by + 1) * block_size, bx * block_size:(bx + 1) * block_size] = restored_versions[block_level][i][by * block_size:(by + 1) * block_size, bx * block_size:(bx + 1) * block_size]

        final_frames.append(final_frame)
    
    return final_frames


@measure_performance(reference_frames, foreground_masks=ufo_masks, block_size=config.block_size)
def upscale_with_realesrgan(frames: List[np.ndarray], upsampler, **kwargs) -> List[np.ndarray]:
    """Restore frames using RealESRGAN with naive whole-frame upscaling.
    
    Upscales entire frame then downscales to original size.
    
    Args:
        frames: List of RGB frames (H, W, 3)
        upsampler: Pre-initialized RealESRGAN upsampler model
        **kwargs: Optional arguments:
            - degradation_level: If provided, used as outscale value (default 4)
    
    Returns:
        Restored frames list
    """
    # Use degradation_level as outscale if provided
    outscale = int(kwargs.get('degradation_level', 4))
    outscale = 1 if outscale == 0 else outscale  # Handle 0 level (no degradation) as 1x scale
    
    restored = []
    
    for i, frame in enumerate(frames):
        h, w = frame.shape[:2]
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # Upscale with adaptive or default scale
        upscaled, _ = upsampler.enhance(frame_bgr, outscale=outscale)
        result = cv2.resize(upscaled, (w, h), interpolation=cv2.INTER_LANCZOS4)
        result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        
        restored.append(result_rgb)
    
    return restored


@measure_performance(reference_frames, foreground_masks=ufo_masks, block_size=config.block_size)
def restore_with_instantir(frames: List[np.ndarray], runtime, cfg: float = 7.0, creative_start: float = 1.0, preview_start: float = 0.0, seed: int = 42, num_inference_steps: int = 30, **kwargs) -> List[np.ndarray]:
    """Restore frames using InstantIR with naive whole-frame processing.
    
    NOTE: InstantIR is a diffusion model that hallucinates details by design.
    For SSIM-focused evaluation, this will typically reduce SSIM compared to 
    OpenCV methods. InstantIR excels at perceptual quality, not fidelity.
    
    Args:
        frames: List of RGB frames (H, W, 3)
        runtime: Pre-initialized InstantIR runtime model
        cfg: Classifier-free guidance scale (lower = less hallucination)
        creative_start: Control guidance end point (1.0 = max fidelity, lower = more creative)
        preview_start: Preview start point
        seed: Random seed
        num_inference_steps: Number of diffusion steps (lower = faster, less hallucination)
        **kwargs: Optional arguments:
            - degradation_level: If provided, used as cfg value
    
    Returns:
        Restored frames list
    """
    
    # Use degradation_level as cfg if provided by restore_video_adaptively
    if 'degradation_level' in kwargs:
        cfg = float(kwargs['degradation_level'])
    
    restored = []
    num_frames = len(frames)
    
    for i, frame in enumerate(frames):
        print(f"\r      InstantIR Naive: frame {i+1}/{num_frames}", end="", flush=True)
        
        # Process whole frame - send output to null to suppress logs, errors and warnings
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            result = restore_image(
                runtime, frame,
                cfg=cfg, preview_start=preview_start,
                creative_start=creative_start,
                num_inference_steps=num_inference_steps,
                seed=seed + i, output_type="numpy"
                )
        
        restored.append(result)
    
    print()  # Newline after progress
    return restored


@measure_performance(reference_frames, foreground_masks=ufo_masks, block_size=config.block_size)
def upscale_with_uav(frames: List[np.ndarray], upscaler, noise_level: int = 120, guidance_scale: float = 6.0, inference_steps: int = 30, **kwargs) -> List[np.ndarray]:
    """Restore frames using Upscale-A-Video with naive whole-frame processing.
    
    Args:
        frames: List of RGB frames (H, W, 3)
        upscaler: Pre-initialized UpscaleAVideo model
        noise_level: Noise level [0-200]
        guidance_scale: CFG scale
        inference_steps: Denoising steps
        **kwargs: Optional arguments:
            - degradation_level: If provided, used as guidance_scale value
    
    Returns:
        Restored frames list
    """
    
    # Use degradation_level as guidance_scale if provided by restore_video_adaptively
    if 'degradation_level' in kwargs:
        guidance_scale = float(kwargs['degradation_level'])
    
    n_frames = len(frames)
    h, w = frames[0].shape[:2]
    
    # Standard processing
    restored = [None] * n_frames
    
    try:
        upscaled = upscaler.upscale_frames(
            frames, noise_level=noise_level,
            guidance_scale=guidance_scale,
            inference_steps=inference_steps,
            output_format="numpy"
        )
        for j, res in enumerate(upscaled):
            result = cv2.resize(res, (w, h), interpolation=cv2.INTER_LANCZOS4)
            restored[j] = result
    except Exception as e:
        print(f"    UAV error: {e}")
        for j in range(n_frames):
            if restored[j] is None:
                restored[j] = frames[j]
    
    return restored


# =============================================================================
# Main Experiment Script
# =============================================================================

if __name__ == "__main__":
    
    # Dictionary to store all performance metrics
    performance_metrics = {}

    # Encode baseline videos
    print("Encoding baseline videos...")
    encode_video(reference_frames, "kvazaar.mp4", config.quality, config.qp_range, encoder="kvazaar")
    encode_video(reference_frames, "svtav1.mp4", config.quality, config.qp_range, encoder="svtav1")

    # ELVIS: shrink video frames
    print("Shrinking video frames...")
    shrunk_frames_row_only, masks_row_only = shrink_video_frames(reference_frames, importance_scores, config.block_size, config.shrink_amount, method=shrink_frame_row_only)
    shrunk_frames_rem_ind, masks_rem_ind = shrink_video_frames(reference_frames, importance_scores, config.block_size, config.shrink_amount, method=shrink_frame_removal_indices)

    # ELVIS: stretch video frames back to original size
    print("Stretching video frames back to original size...")
    stretched_frames_row_only = stretch_video_frames(shrunk_frames_row_only, masks_row_only, config.block_size)
    stretched_frames_rem_ind = stretch_video_frames(shrunk_frames_rem_ind, masks_rem_ind, config.block_size)

    # ELVIS: inpaint removed regions
    print("Inpainting removed regions with OpenCV Telea...")
    inpainted_frames_row_only_telea, performance_metrics['inpaint_row_only_telea'] = inpaint_with_opencv(np.array(stretched_frames_row_only), np.array(masks_row_only))
    inpainted_frames_rem_ind_telea, performance_metrics['inpaint_rem_ind_telea'] = inpaint_with_opencv(np.array(stretched_frames_rem_ind), np.array(masks_rem_ind))
    print("Inpainting removed regions with ProPainter...")
    inpainted_frames_row_only_propainter, performance_metrics['inpaint_row_only_propainter'] = inpaint_with_propainter(np.array(stretched_frames_row_only), np.array(masks_row_only), model=propainter_model, device=device)
    inpainted_frames_rem_ind_propainter, performance_metrics['inpaint_rem_ind_propainter'] = inpaint_with_propainter(np.array(stretched_frames_rem_ind), np.array(masks_rem_ind), model=propainter_model, device=device)
    print("Inpainting removed regions with E2FGVI...")
    inpainted_frames_row_only_e2fgvi, performance_metrics['inpaint_row_only_e2fgvi'] = inpaint_with_e2fgvi(np.array(stretched_frames_row_only), np.array(masks_row_only), model=e2fgvi_model, device=device)
    inpainted_frames_rem_ind_e2fgvi, performance_metrics['inpaint_rem_ind_e2fgvi'] = inpaint_with_e2fgvi(np.array(stretched_frames_rem_ind), np.array(masks_rem_ind), model=e2fgvi_model, device=device)

    # PRESLEY: create ROI files for Kvazaar and SVT-AV1
    print("Creating Kvazaar ROI file...")
    kvazaar_roi_path = experiment_folder / "kvazaar_roi.bin"
    create_kvazaar_roi_file(importance_scores, str(kvazaar_roi_path), base_qp=QUALITY_PRESETS[config.quality]["kvazaar_qp"], qp_range=config.qp_range or QUALITY_PRESETS[config.quality]["qp_range"])
    print("Creating SVT-AV1 ROI file...")
    svtav1_roi_path = experiment_folder / "svtav1_roi.txt"
    create_svtav1_roi_file(importance_scores, str(svtav1_roi_path), base_crf=QUALITY_PRESETS[config.quality]["svtav1_crf"], qp_range=config.qp_range or QUALITY_PRESETS[config.quality]["qp_range"], width=config.width, height=config.height)

    # PRESLEY: Initialize restoration models
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Loading restoration models...")
    propainter_model = ProPainterModel(device=torch.device(device), fp16=config.propainter_fp16) if ProPainterModel else None
    e2fgvi_model = E2FGVIModel(model="e2fgvi_hq", device=torch.device(device)) if E2FGVIModel else None
    realesrgan_upsampler = create_upsampler(model_name=config.realesrgan_model_name if hasattr(config, 'realesrgan_model_name') else "RealESRGAN_x4plus", device=torch.device(device), tile=config.neural_tile_size, tile_pad=config.context_halo, pre_pad=config.realesrgan_pre_pad, half=not config.realesrgan_fp32, denoise_strength=config.realesrgan_denoise_strength)
    instantir_runtime = load_runtime(instantir_path=Path("~/.cache/instantir").expanduser(), device=device, torch_dtype=torch.float16)
    uav_upscaler = UpscaleAVideo(device=device).load_models()

    # PRESLEY: Degrade video adaptively
    print("Degrading video adaptively...")
    downscaled_frames, downscaled_maps = degrade_video_adaptive(reference_frames, importance_scores, config.block_size, max_value=4, method=downscale_block)
    blurred_frames, blurred_maps = degrade_video_adaptive(reference_frames, importance_scores, config.block_size, max_value=4, method=blur_block)

    # PRESLEY: Comparing restoration methods naively and adaptively
    print("Restoring downscaled video adaptively with OpenCV Lanczos...")
    restored_downscaled_opencv, performance_metrics['restore_opencv_lanczos_adaptive'] = restore_video_adaptively(restore_with_opencv_lanczos, downscaled_frames, downscaled_maps, block_size=config.block_size, halo=config.context_halo, temporal_blend=config.temporal_blend)

    print("Restoring blurred video adaptively with OpenCV Unsharp Masking...")
    restored_blurred_opencv, performance_metrics['restore_opencv_unsharp_adaptive'] = restore_video_adaptively(restore_with_opencv_unsharp, blurred_frames, blurred_maps, block_size=config.block_size, halo=config.context_halo, temporal_blend=config.temporal_blend)
    
    print("Restoring downscaled video naively with RealESRGAN...")
    restored_downscaled_realesrgan_naive, performance_metrics['restore_realesrgan_naive'] = upscale_with_realesrgan(downscaled_frames, realesrgan_upsampler)
    
    print("Restoring downscaled video adaptively with RealESRGAN...")
    restored_downscaled_realesrgan, performance_metrics['restore_realesrgan_adaptive'] = restore_video_adaptively(upscale_with_realesrgan, downscaled_frames, downscaled_maps, block_size=config.block_size, upsampler=realesrgan_upsampler)
    
    print("Restoring blurred video naively with InstantIR...")
    restored_blurred_instantir_naive, performance_metrics['restore_instantir_naive'] = restore_with_instantir(blurred_frames, instantir_runtime, cfg=config.instantir_cfg, creative_start=config.instantir_creative_start, preview_start=config.instantir_preview_start, seed=config.instantir_seed, num_inference_steps=config.instantir_inference_steps)
    
    print("Restoring blurred video adaptively with InstantIR...")
    restored_blurred_instantir, performance_metrics['restore_instantir_adaptive'] = restore_video_adaptively(restore_with_instantir, blurred_frames, blurred_maps, block_size=config.block_size, runtime=instantir_runtime, cfg=config.instantir_cfg, creative_start=config.instantir_creative_start, preview_start=config.instantir_preview_start, seed=config.instantir_seed, num_inference_steps=config.instantir_inference_steps)
    
    print("Restoring downscaled video naively with Upscale-A-Video...")
    restored_downscaled_uav_naive, performance_metrics['restore_uav_naive'] = upscale_with_uav(downscaled_frames, uav_upscaler, noise_level=config.uav_noise_level, guidance_scale=config.uav_guidance_scale, inference_steps=config.uav_inference_steps)
    
    print("Restoring downscaled video adaptively with Upscale-A-Video...")
    restored_downscaled_uav, performance_metrics['restore_uav_adaptive'] = restore_video_adaptively(upscale_with_uav, downscaled_frames, downscaled_maps, block_size=config.block_size, upscaler=uav_upscaler, noise_level=config.uav_noise_level, guidance_scale=config.uav_guidance_scale, inference_steps=config.uav_inference_steps)
    
    # Save performance metrics to JSON
    print("\nSaving performance metrics...")
    metrics_path = experiment_folder / "performance_metrics.json"
    with open(metrics_path, 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        serializable_metrics = {}
        for key, metrics in performance_metrics.items():
            serializable_metrics[key] = {
                'execution_speed': metrics['execution_speed'],
                'overall_ssim': metrics['overall_ssim'],
                'foreground_ssim': metrics['foreground_ssim'],
                'background_ssim': metrics['background_ssim'],
                'status': metrics['status'],
                'error': metrics['error']
            }
        json.dump(serializable_metrics, f, indent=2)
    print(f"Performance metrics saved to {metrics_path}")