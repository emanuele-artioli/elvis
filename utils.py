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

# Try importing optional dependencies, handle if missing
try:
    from propainter import ProPainterModel, InpaintingConfig as ProPainterConfig
except ImportError:
    ProPainterModel = None

try:
    from e2fgvi import E2FGVIModel, InpaintingConfig as E2FGVIConfig
except ImportError:
    E2FGVIModel = None

try:
    from realesrgan import create_upsampler
except ImportError:
    create_upsampler = None

try:
    from instantir import InstantIRRuntime, load_runtime, restore_images_batch, restore_image
except ImportError:
    InstantIRRuntime = None

try:
    from upscale_a_video import UpscaleAVideo
except ImportError:
    UpscaleAVideo = None


# =============================================================================
# Configuration & Presets
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
    width: int = 640  # Reduced for faster testing (was 1280)
    height: int = 360  # Reduced for faster testing (was 720)
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

# Global config instance
config = PresleyConfig()


# =============================================================================
# Timing & Resource Wrappers
# =============================================================================

def measure_time(log_path: str) -> Callable:
    """Decorator to measure execution time of a function and log it to a JSON file.
    
    Args:
        log_path: Path to the JSON log file.
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                status = "success"
                error_msg = None
            except Exception as e:
                status = "error"
                error_msg = str(e)
                raise e
            finally:
                end_time = time.time()
                duration = end_time - start_time
                
                log_entry = {
                    "function": func.__name__,
                    "timestamp": datetime.now().isoformat(),
                    "duration_sec": duration,
                    "status": status,
                    "error": error_msg
                }
                
                # Append to log file (create if not exists)
                try:
                    if os.path.exists(log_path):
                        with open(log_path, 'r') as f:
                            try:
                                logs = json.load(f)
                                if not isinstance(logs, list):
                                    logs = []
                            except json.JSONDecodeError:
                                logs = []
                    else:
                        logs = []
                    
                    logs.append(log_entry)
                    
                    with open(log_path, 'w') as f:
                        json.dump(logs, f, indent=2)
                except Exception as log_err:
                    print(f"Failed to write timing log: {log_err}")
            
            return result
        return wrapper
    return decorator


def resource_aware_restore(restore_fn: Callable, frames: List[np.ndarray], tile_size: int = 512, halo: int = 16, chunk_size: int = 8, chunk_overlap: int = 2, max_workers: int = 1, device: str = "cuda", **kwargs) -> List[np.ndarray]:
    """
    Wrapper for restoration functions that handles spatial tiling and temporal chunking
    to prevent OOM errors and enable parallelization.
    
    Args:
        restore_fn: The restoration function to wrap. Must accept 'frames' and 'device'.
        frames: List of input frames (H, W, 3).
        tile_size: Size of spatial tiles (0 to disable tiling).
        halo: Overlap between tiles/chunks for blending.
        chunk_size: Number of frames per temporal chunk (0 to disable chunking).
        chunk_overlap: Overlap between temporal chunks.
        max_workers: Number of parallel workers for tile processing.
        device: Device to run restoration on.
        **kwargs: Additional arguments passed to restore_fn.
        
    Returns:
        List of restored frames.
    """
    if not frames:
        return []
        
    h, w = frames[0].shape[:2]
    n_frames = len(frames)
    
    # Determine if tiling/chunking is needed
    do_tiling = tile_size > 0 and (h > tile_size or w > tile_size)
    do_chunking = chunk_size > 0 and n_frames > chunk_size
    
    # If no resource management needed, run directly
    if not do_tiling and not do_chunking:
        return restore_fn(frames=frames, device=device, **kwargs)
    
    # Initialize output buffers
    # We use float32 for accumulation to handle blending
    output_frames = [np.zeros((h, w, 3), dtype=np.float32) for _ in range(n_frames)]
    weight_maps = [np.zeros((h, w, 1), dtype=np.float32) for _ in range(n_frames)]
    
    # Define tile coordinates
    if do_tiling:
        y_steps = range(0, h, tile_size - halo)
        x_steps = range(0, w, tile_size - halo)
    else:
        y_steps = [0]
        x_steps = [0]
        tile_size = max(h, w) # effectively no tiling
        
    # Define temporal chunks
    if do_chunking:
        t_steps = range(0, n_frames, chunk_size - chunk_overlap)
    else:
        t_steps = [0]
        chunk_size = n_frames
        
    # Helper to process a single task (chunk of tiles)
    def process_task(task_args):
        t_start, y_start, x_start = task_args
        
        # Calculate bounds
        t_end = min(t_start + chunk_size, n_frames)
        y_end = min(y_start + tile_size, h)
        x_end = min(x_start + tile_size, w)
        
        # Extract input volume
        input_chunk = [f[y_start:y_end, x_start:x_end] for f in frames[t_start:t_end]]
        
        # Run restoration
        try:
            # Pass tile coordinates to allow adaptive logic to know context
            restored_chunk = restore_fn(
                frames=input_chunk, 
                device=device, 
                tile_coords=(t_start, t_end, y_start, y_end, x_start, x_end),
                **kwargs
            )
        except Exception as e:
            print(f"Error processing chunk t={t_start}:{t_end}, y={y_start}:{y_end}, x={x_start}:{x_end}: {e}")
            # Fallback: return input as output
            restored_chunk = input_chunk
            
        return (t_start, t_end, y_start, y_end, x_start, x_end, restored_chunk)

    # Generate tasks
    tasks = []
    for t in t_steps:
        for y in y_steps:
            for x in x_steps:
                tasks.append((t, y, x))
                
    # Execute tasks
    results = []
    if max_workers > 1:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(process_task, tasks))
    else:
        for task in tasks:
            results.append(process_task(task))
            
    # Blend results
    for res in results:
        t_start, t_end, y_start, y_end, x_start, x_end, restored_chunk = res
        
        # Create spatial weight mask (feathered edges)
        chunk_h, chunk_w = restored_chunk[0].shape[:2]
        spatial_weight = np.ones((chunk_h, chunk_w, 1), dtype=np.float32)
        
        if do_tiling:
            feather = halo // 2
            if feather > 0:
                # Feather Y
                if y_start > 0:
                    spatial_weight[:feather, :, :] *= np.linspace(0, 1, feather)[:, None, None]
                if y_end < h:
                    spatial_weight[-feather:, :, :] *= np.linspace(1, 0, feather)[:, None, None]
                # Feather X
                if x_start > 0:
                    spatial_weight[:, :feather, :] *= np.linspace(0, 1, feather)[None, :, None]
                if x_end < w:
                    spatial_weight[:, -feather:, :] *= np.linspace(1, 0, feather)[None, :, None]
                    
        # Accumulate results
        for i, frame in enumerate(restored_chunk):
            global_t = t_start + i
            
            # Temporal weighting (linear blend in overlap regions)
            temporal_weight = 1.0
            if do_chunking:
                # Fade in at start (unless first chunk)
                if t_start > 0 and i < chunk_overlap:
                    temporal_weight *= (i + 1) / (chunk_overlap + 1)
                # Fade out at end (unless last chunk)
                if t_end < n_frames and i >= (len(restored_chunk) - chunk_overlap):
                    dist_from_end = len(restored_chunk) - i
                    temporal_weight *= dist_from_end / (chunk_overlap + 1)
            
            total_weight = spatial_weight * temporal_weight
            
            output_frames[global_t][y_start:y_end, x_start:x_end] += frame.astype(np.float32) * total_weight
            weight_maps[global_t][y_start:y_end, x_start:x_end] += total_weight

    # Normalize and convert back to uint8
    final_frames = []
    for i in range(n_frames):
        # Avoid division by zero
        mask = weight_maps[i] > 0
        safe_weight = weight_maps[i].copy()
        safe_weight[~mask] = 1.0
        output_frames[i] /= safe_weight
        final_frames.append(np.clip(output_frames[i], 0, 255).astype(np.uint8))
        
    return final_frames


def adaptive_restore(restore_fn: Callable, frames: List[np.ndarray], degradation_maps: Optional[np.ndarray] = None, block_size: int = 16, tile_coords: Optional[Tuple[int, int, int, int, int, int]] = None, threshold: float = 0.0, **kwargs) -> List[np.ndarray]:
    """
    Adaptive restoration wrapper.
    Decides whether to apply restore_fn based on degradation_maps.
    
    Args:
        restore_fn: The restoration function to wrap.
        frames: List of input frames (tiles or full frames).
        degradation_maps: Full-size degradation maps (N, H//BS, W//BS).
        block_size: Size of blocks in degradation map.
        tile_coords: (t_start, t_end, y_start, y_end, x_start, x_end) if tiled.
        threshold: Threshold for degradation level to trigger restoration.
        **kwargs: Arguments passed to restore_fn.
    """
    # If no degradation maps, just run restoration (fallback)
    if degradation_maps is None:
        return restore_fn(frames=frames, **kwargs)
        
    # Determine if we should restore based on degradation map
    should_restore = False
    
    if tile_coords:
        t_start, t_end, y_start, y_end, x_start, x_end = tile_coords
        
        # Convert pixel coords to block coords
        # Use floor for start and ceil for end to cover all partial blocks
        by_start = y_start // block_size
        by_end = (y_end + block_size - 1) // block_size + 1
        bx_start = x_start // block_size
        bx_end = (x_end + block_size - 1) // block_size + 1
        
        # Handle map boundaries
        h_blocks, w_blocks = degradation_maps.shape[1:]
        by_end = min(by_end, h_blocks)
        bx_end = min(bx_end, w_blocks)
        
        # Slice maps for this chunk
        # degradation_maps is (N, BlocksY, BlocksX)
        # Ensure t indices are valid
        n_maps = len(degradation_maps)
        t_start_map = min(t_start, n_maps)
        t_end_map = min(t_end, n_maps)
        
        if t_start_map < t_end_map:
            chunk_maps = degradation_maps[t_start_map:t_end_map, by_start:by_end, bx_start:bx_end]
            if chunk_maps.size > 0 and np.max(chunk_maps) > threshold:
                should_restore = True
    else:
        # Full frame processing - check whole map for these frames
        # We assume frames correspond to degradation_maps indices if not tiled?
        # This is ambiguous without t_start. 
        # But if tile_coords is None, we assume we are processing the whole video or passed frames match maps?
        # For safety, if no coords, we assume we must restore if ANY degradation exists in the passed maps?
        # But we don't know which maps correspond to 'frames'.
        # So we rely on 'degradation_maps' being passed correctly (maybe sliced already?).
        # But resource_aware_restore passes FULL maps.
        # So without tile_coords, we can't know which part of map to check.
        # So we default to True if we can't determine.
        should_restore = True

    if should_restore:
        return restore_fn(frames=frames, **kwargs)
    else:
        # Skip restoration (identity)
        # Assumes restore_fn preserves resolution.
        return frames


# =============================================================================
# Video I/O Functions
# =============================================================================

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
# Quality Metrics
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
            ssim_values[i:end] = pytorch_msssim.ssim(
                patches1[i:end], patches2[i:end], data_range=1.0, size_average=False
            )
        
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


# =============================================================================
# ELVIS: Frame Shrinking
# Remove low-importance blocks to reduce frame size. The removed blocks can
# later be reconstructed via video inpainting (e.g., ProPainter, E2FGVI).
# =============================================================================

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


def stretch_frame_row_only(shrunk_frame: np.ndarray, removal_mask: np.ndarray, block_size: int) -> np.ndarray:
    """Reconstruct frame from row-only shrunk version.
    
    For row-only removal, blocks in each row of the shrunken frame map directly
    to non-removed positions in that same row of the original.
    """
    orig_blocks_y, orig_blocks_x = removal_mask.shape
    h, w, c = shrunk_frame.shape
    shrunk_blocks_y, shrunk_blocks_x = h // block_size, w // block_size
    
    shrunk_blocks = shrunk_frame.reshape(shrunk_blocks_y, block_size, shrunk_blocks_x, block_size, c).swapaxes(1, 2)
    final_blocks = np.zeros((orig_blocks_y, orig_blocks_x, block_size, block_size, c), dtype=shrunk_frame.dtype)
    
    # For each row, place shrunk blocks at non-removed positions
    for orig_y in range(orig_blocks_y):
        kept_cols = np.where(~removal_mask[orig_y, :])[0]
        for shrunk_x, orig_x in enumerate(kept_cols):
            if shrunk_x < shrunk_blocks_x:
                final_blocks[orig_y, orig_x] = shrunk_blocks[orig_y, shrunk_x]
    
    return final_blocks.swapaxes(1, 2).reshape(orig_blocks_y * block_size, orig_blocks_x * block_size, c)

# Shrinking Method 2: Row+Col removal with position_map

def shrink_frame_position_map(frame: np.ndarray, importance: np.ndarray, block_size: int, shrink_amount: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Remove blocks from both rows and columns, tracking positions via position_map.
    
    Returns:
        shrunken: The shrunken frame with reduced dimensions
        removal_mask: Boolean mask of original block positions (True = removed)
        position_map: Array mapping (shrunk_y, shrunk_x) -> (orig_y, orig_x) for kept blocks
    """
    frame = frame.copy()
    importance = importance.copy()
    
    height, width = frame.shape[:2]
    blocks_y = height // block_size
    blocks_x = width // block_size
    orig_blocks_y, orig_blocks_x = blocks_y, blocks_x
    
    # Reshape into blocks: (blocks_y, block_size, blocks_x, block_size, channels)
    blocked = frame[:blocks_y * block_size, :blocks_x * block_size].reshape(
        blocks_y, block_size, blocks_x, block_size, 3)
    
    # Track original positions for each current position
    # position_map[by, bx] = (orig_y, orig_x)
    position_map = np.array([[(by, bx) for bx in range(orig_blocks_x)] for by in range(orig_blocks_y)])
    removal_mask = np.zeros((orig_blocks_y, orig_blocks_x), dtype=bool)
    target_removals = int(orig_blocks_y * orig_blocks_x * shrink_amount)
    removed = 0
    
    while removed < target_removals:
        # Remove least important block from each row
        row_pass_complete = True
        for by in range(blocks_y):
            if removed >= target_removals:
                row_pass_complete = False
                break
            least_idx = np.argmin(importance[by, :blocks_x])
            orig_pos = position_map[by, least_idx]
            removal_mask[orig_pos[0], orig_pos[1]] = True
            
            # Shift blocks left
            blocked[by, :, least_idx:blocks_x-1] = blocked[by, :, least_idx+1:blocks_x].copy()
            importance[by, least_idx:blocks_x-1] = importance[by, least_idx+1:blocks_x]
            position_map[by, least_idx:blocks_x-1] = position_map[by, least_idx+1:blocks_x]
            removed += 1
        
        if row_pass_complete:
            blocks_x -= 1
            importance = importance[:, :blocks_x]
        
        if removed >= target_removals:
            break
        
        # Remove least important block from each column
        col_pass_complete = True
        for bx in range(blocks_x):
            if removed >= target_removals:
                col_pass_complete = False
                break
            least_idx = np.argmin(importance[:blocks_y, bx])
            orig_pos = position_map[least_idx, bx]
            removal_mask[orig_pos[0], orig_pos[1]] = True
            
            # Shift blocks up
            blocked[least_idx:blocks_y-1, :, bx] = blocked[least_idx+1:blocks_y, :, bx].copy()
            importance[least_idx:blocks_y-1, bx] = importance[least_idx+1:blocks_y, bx]
            position_map[least_idx:blocks_y-1, bx] = position_map[least_idx+1:blocks_y, bx]
            removed += 1
        
        if col_pass_complete:
            blocks_y -= 1
            importance = importance[:blocks_y, :]
    
    shrunken = blocked[:blocks_y, :, :blocks_x].reshape(blocks_y * block_size, blocks_x * block_size, 3)
    final_position_map = position_map[:blocks_y, :blocks_x].copy()
    return shrunken, removal_mask, final_position_map


def stretch_frame_position_map(shrunk_frame: np.ndarray, removal_mask: np.ndarray, position_map: np.ndarray, block_size: int) -> np.ndarray:
    """Reconstruct frame using explicit position_map."""
    orig_blocks_y, orig_blocks_x = removal_mask.shape
    h, w, c = shrunk_frame.shape
    shrunk_blocks_y, shrunk_blocks_x = h // block_size, w // block_size
    
    # Split shrunk frame into blocks
    shrunk_blocks = shrunk_frame.reshape(shrunk_blocks_y, block_size, shrunk_blocks_x, block_size, c).swapaxes(1, 2)
    
    # Create empty canvas of blocks for full resolution
    final_blocks = np.zeros((orig_blocks_y, orig_blocks_x, block_size, block_size, c), dtype=shrunk_frame.dtype)
    
    # Place each shrunk block at its original position
    for by in range(shrunk_blocks_y):
        for bx in range(shrunk_blocks_x):
            orig_y, orig_x = position_map[by, bx]
            final_blocks[orig_y, orig_x] = shrunk_blocks[by, bx]
    
    # Combine blocks back into image
    return final_blocks.swapaxes(1, 2).reshape(orig_blocks_y * block_size, orig_blocks_x * block_size, c)

# Shrinking Method 3: Row+Col removal with removal_indices list

def shrink_frame_removal_indices(frame: np.ndarray, importance: np.ndarray, block_size: int, shrink_amount: float) -> Tuple[np.ndarray, np.ndarray, list]:
    """Remove blocks from both rows and columns, tracking removed indices per pass.
    
    Returns:
        shrunken: The shrunken frame with reduced dimensions
        removal_mask: Boolean mask of original block positions (True = removed)
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
    
    removal_mask = np.zeros((orig_blocks_y, orig_blocks_x), dtype=bool)
    removal_indices = []  # List of arrays tracking removed indices per pass
    target_removals = int(orig_blocks_y * orig_blocks_x * shrink_amount)
    removed = 0
    
    # Track original positions for removal_mask
    position_map = np.array([[(by, bx) for bx in range(orig_blocks_x)] for by in range(orig_blocks_y)])
    
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
            
            orig_pos = position_map[by, least_idx]
            removal_mask[orig_pos[0], orig_pos[1]] = True
            
            blocked[by, :, least_idx:blocks_x-1] = blocked[by, :, least_idx+1:blocks_x].copy()
            importance[by, least_idx:blocks_x-1] = importance[by, least_idx+1:blocks_x]
            position_map[by, least_idx:blocks_x-1] = position_map[by, least_idx+1:blocks_x]
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
            
            orig_pos = position_map[least_idx, bx]
            removal_mask[orig_pos[0], orig_pos[1]] = True
            
            blocked[least_idx:blocks_y-1, :, bx] = blocked[least_idx+1:blocks_y, :, bx].copy()
            importance[least_idx:blocks_y-1, bx] = importance[least_idx+1:blocks_y, bx]
            position_map[least_idx:blocks_y-1, bx] = position_map[least_idx+1:blocks_y, bx]
            removed += 1
        
        if col_indices:
            removal_indices.append(np.array(col_indices, dtype=np.int32))
        
        if col_pass_complete:
            blocks_y -= 1
            importance = importance[:blocks_y, :]
    
    shrunken = blocked[:blocks_y, :, :blocks_x].reshape(blocks_y * block_size, blocks_x * block_size, 3)
    return shrunken, removal_mask, removal_indices


def stretch_frame_removal_indices(shrunk_frame: np.ndarray, removal_indices: list, orig_blocks_y: int, orig_blocks_x: int, block_size: int) -> np.ndarray:
    """Reconstruct frame by inverting the removal process using removal_indices.
    
    Process removal_indices in reverse order, inserting black blocks at the recorded positions.
    """
    h, w, c = shrunk_frame.shape
    blocks_y, blocks_x = h // block_size, w // block_size
    
    # Start with the shrunken blocks reshaped properly
    blocked = shrunk_frame.reshape(blocks_y, block_size, blocks_x, block_size, c)
    
    # Process removal_indices in reverse to expand back
    for pass_idx in range(len(removal_indices) - 1, -1, -1):
        indices = removal_indices[pass_idx]
        is_row_pass = (pass_idx % 2 == 0)  # Even indices are row passes
        
        if is_row_pass:
            # Row pass: we removed one block per row, so expand width
            new_blocks_x = blocks_x + 1
            new_blocked = np.zeros((blocks_y, block_size, new_blocks_x, block_size, c), dtype=blocked.dtype)
            
            # Process each row that had a removal
            for by in range(min(len(indices), blocks_y)):
                insert_idx = indices[by]
                # Clamp insert_idx to valid range
                insert_idx = min(insert_idx, blocks_x)
                # Copy blocks before insertion point
                if insert_idx > 0:
                    new_blocked[by, :, :insert_idx] = blocked[by, :, :insert_idx]
                # Insert black block at removal position (stays zeros)
                # Copy blocks after insertion point
                if insert_idx < blocks_x:
                    new_blocked[by, :, insert_idx+1:new_blocks_x] = blocked[by, :, insert_idx:blocks_x]
            
            # Copy remaining rows unchanged (these didn't have a removal in this pass)
            for by in range(len(indices), blocks_y):
                new_blocked[by, :, :blocks_x] = blocked[by, :, :blocks_x]
            
            blocked = new_blocked
            blocks_x = new_blocks_x
        else:
            # Column pass: we removed one block per column, so expand height
            new_blocks_y = blocks_y + 1
            new_blocked = np.zeros((new_blocks_y, block_size, blocks_x, block_size, c), dtype=blocked.dtype)
            
            # Process each column that had a removal
            for bx in range(min(len(indices), blocks_x)):
                insert_idx = indices[bx]
                # Clamp insert_idx to valid range
                insert_idx = min(insert_idx, blocks_y)
                # Copy blocks before insertion point
                if insert_idx > 0:
                    new_blocked[:insert_idx, :, bx] = blocked[:insert_idx, :, bx]
                # Insert black block at removal position (stays zeros)
                # Copy blocks after insertion point
                if insert_idx < blocks_y:
                    new_blocked[insert_idx+1:new_blocks_y, :, bx] = blocked[insert_idx:blocks_y, :, bx]
            
            # Copy remaining columns unchanged (these didn't have a removal in this pass)
            for bx in range(len(indices), blocks_x):
                new_blocked[:blocks_y, :, bx] = blocked[:blocks_y, :, bx]
            
            blocked = new_blocked
            blocks_y = new_blocks_y
    
    # Reshape to final image - crop to original size if needed
    result = blocked.reshape(blocks_y * block_size, blocks_x * block_size, c)
    return result[:orig_blocks_y * block_size, :orig_blocks_x * block_size]


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

def degrade_adaptive_downsample(frame: np.ndarray, importance: np.ndarray, block_size: int, max_scale: int = 4) -> Tuple[np.ndarray, np.ndarray]:
    """Adaptively downsample blocks based on importance scores using scale factors.
    
    Low importance → higher downscale factor.
    Scale factors are dynamically determined based on max_scale:
    - Divides importance into max_scale levels (0 to max_scale-1)
    - Level 0 = no degradation, Level max_scale-1 = maximum degradation
    
    Args:
        frame: RGB frame (H, W, 3)
        importance: Per-block importance scores [0, 1] (blocks_y, blocks_x)
        block_size: Block size in pixels
        max_scale: Maximum downscale factor (any int >= 2)
    
    Returns:
        degraded_frame: Frame with adaptively degraded blocks
        degradation_map: Per-block scale factor (0=none, 2 to max_scale)
    """
    h, w = frame.shape[:2]
    blocks_y, blocks_x = h // block_size, w // block_size
    
    # Crop frame to block-aligned size
    frame_cropped = frame[:blocks_y * block_size, :blocks_x * block_size].copy()
    blocks = frame_cropped.reshape(blocks_y, block_size, blocks_x, block_size, 3).swapaxes(1, 2)
    
    # Resize importance to match block dimensions if needed
    if importance.shape != (blocks_y, blocks_x):
        importance = cv2.resize(importance, (blocks_x, blocks_y), interpolation=cv2.INTER_LINEAR)
    
    # Map importance to scale factors dynamically based on max_scale
    # High importance (1.0) -> 0 (no degradation)
    # Low importance (0.0) -> max_scale (maximum degradation)
    # Quantize into max_scale levels: 0, 2, 3, ..., max_scale
    inv_importance = 1 - importance
    
    # Create scale levels: divide [0,1] into max_scale bins
    # Bin 0: inv_importance < 1/max_scale -> scale 0 (no degradation)
    # Bin 1: 1/max_scale <= inv_importance < 2/max_scale -> scale 2
    # Bin 2: 2/max_scale <= inv_importance < 3/max_scale -> scale 3
    # ...
    # Bin max_scale-1: inv_importance >= (max_scale-1)/max_scale -> scale max_scale
    bin_indices = np.floor(inv_importance * max_scale).astype(np.int32)
    bin_indices = np.clip(bin_indices, 0, max_scale - 1)
    
    # Map bin index to scale factor: bin 0 -> 0, bin 1 -> 2, bin 2 -> 3, ...
    # Special case: bin 0 means no degradation (scale 0)
    # For bins >= 1, scale = bin_index + 1 (so bin 1 -> 2, bin 2 -> 3, etc.)
    scale_levels = np.where(bin_indices == 0, 0, bin_indices + 1)
    degradation_map = scale_levels.astype(np.int32)
    
    # Process each block
    processed_blocks = blocks.copy()
    for by in range(blocks_y):
        for bx in range(blocks_x):
            scale = degradation_map[by, bx]
            if scale > 0:
                block = blocks[by, bx]
                # Downscale then upscale
                small_size = max(1, block_size // scale)
                small = cv2.resize(block, (small_size, small_size), interpolation=cv2.INTER_AREA)
                processed_blocks[by, bx] = cv2.resize(small, (block_size, block_size), interpolation=cv2.INTER_LINEAR)
    
    result = processed_blocks.swapaxes(1, 2).reshape(blocks_y * block_size, blocks_x * block_size, 3)
    
    # Restore original frame size (paste result into original)
    output = frame.copy()
    output[:blocks_y * block_size, :blocks_x * block_size] = result
    return output, degradation_map


def degrade_adaptive_blur(frame: np.ndarray, importance: np.ndarray, block_size: int, max_rounds: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """Adaptively blur blocks based on importance scores.
    
    Low importance → more blur rounds.
    Each round applies Gaussian blur (5x5, sigma=1).
    
    Args:
        frame: RGB frame (H, W, 3)
        importance: Per-block importance scores [0, 1] (blocks_y, blocks_x)
        block_size: Block size in pixels
        max_rounds: Maximum blur rounds (0-max_rounds)
    
    Returns:
        degraded_frame: Frame with adaptively blurred blocks
        degradation_map: Per-block blur level (0 = no blur)
    """
    h, w = frame.shape[:2]
    blocks_y, blocks_x = h // block_size, w // block_size
    
    # Crop frame to block-aligned size
    frame_cropped = frame[:blocks_y * block_size, :blocks_x * block_size].copy()
    blocks = frame_cropped.reshape(blocks_y, block_size, blocks_x, block_size, 3).swapaxes(1, 2)
    
    # Resize importance to match block dimensions if needed
    if importance.shape != (blocks_y, blocks_x):
        importance = cv2.resize(importance, (blocks_x, blocks_y), interpolation=cv2.INTER_LINEAR)
    
    # Map (1 - importance) to [0, max_rounds] blur rounds
    degradation_map = np.round((1 - importance) * max_rounds).astype(np.int32)
    degradation_map = np.clip(degradation_map, 0, max_rounds)
    
    processed_blocks = blocks.copy()
    for by in range(blocks_y):
        for bx in range(blocks_x):
            rounds = degradation_map[by, bx]
            if rounds > 0:
                block = blocks[by, bx]
                for _ in range(rounds):
                    block = cv2.GaussianBlur(block, (5, 5), sigmaX=1.0)
                processed_blocks[by, bx] = block
    
    result = processed_blocks.swapaxes(1, 2).reshape(blocks_y * block_size, blocks_x * block_size, 3)
    
    # Restore original frame size
    output = frame.copy()
    output[:blocks_y * block_size, :blocks_x * block_size] = result
    return output, degradation_map


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


def restore_with_opencv_lanczos(frames: List[np.ndarray], degradation_maps: np.ndarray, block_size: int, halo: int = 0, temporal_blend: float = 0.0, **kwargs) -> List[np.ndarray]:
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


def restore_with_opencv_unsharp(frames: List[np.ndarray], degradation_maps: np.ndarray, block_size: int, halo: int = 0, temporal_blend: float = 0.0, **kwargs) -> List[np.ndarray]:
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


def inpaint_propainter(frames: np.ndarray, masks: np.ndarray, device: str = "cuda") -> np.ndarray:
    """Inpaint frames using ProPainter."""
    if ProPainterModel is None:
        raise ImportError("ProPainter not installed")
    model = ProPainterModel(device=torch.device(device), fp16=config.propainter_fp16)
    pp_config = ProPainterConfig(
        mask_dilation=config.propainter_mask_dilation,
        ref_stride=config.propainter_ref_stride,
        neighbor_length=config.propainter_neighbor_length,
        subvideo_length=config.propainter_subvideo_length,
        raft_iter=config.propainter_raft_iter,
        fp16=config.propainter_fp16,
        device=torch.device(device)
    )
    return model.inpaint(frames, masks, config=pp_config)


def inpaint_e2fgvi(frames: np.ndarray, masks: np.ndarray, device: str = "cuda") -> np.ndarray:
    """Inpaint frames using E2FGVI."""
    if E2FGVIModel is None:
        raise ImportError("E2FGVI not installed")
    model = E2FGVIModel(model="e2fgvi_hq", device=torch.device(device))
    e2_config = E2FGVIConfig(
        model="e2fgvi_hq",
        step=config.e2fgvi_ref_stride,
        num_ref=config.e2fgvi_num_ref,
        neighbor_stride=config.e2fgvi_neighbor_stride,
        mask_dilation=config.e2fgvi_mask_dilation,
        device=torch.device(device)
    )
    return model.inpaint(frames, masks, config=e2_config)


def restore_with_realesrgan_naive(frames: List[np.ndarray], device: str = "cuda", model_name: str = "RealESRGAN_x4plus", tile: int = 0, tile_pad: int = 10, pre_pad: int = 0, fp32: bool = False, denoise_strength: float = 1.0, **kwargs) -> List[np.ndarray]:
    """Restore frames using RealESRGAN with naive whole-frame 4x upscaling.
    
    Upscales entire frame 4x then downscales to original size.
    
    Args:
        frames: List of RGB frames (H, W, 3)
        device: CUDA device string
        model_name: RealESRGAN model variant
        tile: Tile size for internal processing (0=no tiling)
        tile_pad: Padding around tiles
        pre_pad: Pre-padding for input
        fp32: Use FP32 instead of FP16
        denoise_strength: Denoise strength [0-1]
        **kwargs: Ignored arguments (e.g. tile_coords from resource wrapper)
    
    Returns:
        Restored frames list
    """
    if create_upsampler is None:
        raise ImportError("RealESRGAN not installed")
    upsampler = create_upsampler(
        model_name=model_name, 
        device=torch.device(device),
        tile=tile,
        tile_pad=tile_pad,
        pre_pad=pre_pad,
        half=not fp32,
        denoise_strength=denoise_strength
    )
    restored = []
    
    for i, frame in enumerate(frames):
        h, w = frame.shape[:2]
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # Upscale 4x
        upscaled, _ = upsampler.enhance(frame_bgr, outscale=4)
        result = cv2.resize(upscaled, (w, h), interpolation=cv2.INTER_LANCZOS4)
        result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        
        restored.append(result_rgb)
    
    del upsampler
    torch.cuda.empty_cache()
    return restored


def restore_with_instantir_naive(frames: List[np.ndarray], device: str = "cuda", weights_dir: str = "~/.cache/instantir", cfg: float = 7.0, creative_start: float = 1.0, preview_start: float = 0.0, seed: int = 42, num_inference_steps: int = 30, **kwargs) -> List[np.ndarray]:
    """Restore frames using InstantIR with naive whole-frame processing.
    
    NOTE: InstantIR is a diffusion model that hallucinates details by design.
    For SSIM-focused evaluation, this will typically reduce SSIM compared to 
    OpenCV methods. InstantIR excels at perceptual quality, not fidelity.
    
    Args:
        frames: List of RGB frames (H, W, 3)
        device: CUDA device string
        weights_dir: Path to InstantIR weights
        cfg: Classifier-free guidance scale (lower = less hallucination)
        creative_start: Control guidance end point (1.0 = max fidelity, lower = more creative)
        preview_start: Preview start point
        seed: Random seed
        num_inference_steps: Number of diffusion steps (lower = faster, less hallucination)
    
    Returns:
        Restored frames list
    """
    if InstantIRRuntime is None:
        raise ImportError("InstantIR not installed")
    
    # Suppress InstantIR output during loading
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        runtime = load_runtime(
            instantir_path=Path(weights_dir).expanduser(),
            device=device,
            torch_dtype=torch.float16
        )
    
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
    del runtime
    torch.cuda.empty_cache()
    return restored


def restore_with_upscale_a_video_naive(frames: List[np.ndarray], device: str = "cuda", noise_level: int = 120, guidance_scale: float = 6.0, inference_steps: int = 30) -> List[np.ndarray]:
    """Restore frames using Upscale-A-Video with naive whole-frame processing.
    
    Args:
        frames: List of RGB frames (H, W, 3)
        device: CUDA device string
        noise_level: Noise level [0-200]
        guidance_scale: CFG scale
        inference_steps: Denoising steps
    
    Returns:
        Restored frames list
    """
    if UpscaleAVideo is None:
        raise ImportError("UpscaleAVideo not installed")
    upscaler = UpscaleAVideo(device=device)
    upscaler.load_models()
    
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
    
    upscaler.unload_models()
    return restored


def blended_restoration(frames, degradation_maps, block_size, alpha=1.0, restore_fn=restore_with_realesrgan_naive, **kwargs):
    # Run restoration
    restored = resource_aware_restore(restore_fn, frames, **kwargs)
    
    # Blend based on degradation map
    final_frames = []
    for i, (orig, rest, dmap) in enumerate(zip(frames, restored, degradation_maps)):
        # Resize dmap to frame size
        h, w = orig.shape[:2]
        # Ensure dmap matches block grid
        blocks_y, blocks_x = h // block_size, w // block_size
        if dmap.shape != (blocks_y, blocks_x):
                dmap = cv2.resize(dmap.astype(np.float32), (blocks_x, blocks_y), interpolation=cv2.INTER_NEAREST)
        
        mask = cv2.resize(dmap.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)
        
        # Create binary mask: 1 where degraded, 0 where original
        mask_binary = (mask > 0).astype(np.float32)[:, :, None]
        
        # Blend: if degraded, use alpha*rest + (1-alpha)*orig
        w_rest = mask_binary * alpha
        w_orig = 1.0 - w_rest
        
        blended = orig.astype(np.float32) * w_orig + rest.astype(np.float32) * w_rest
        final_frames.append(np.clip(blended, 0, 255).astype(np.uint8))
        
    return final_frames

