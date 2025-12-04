"""PRESLEY: Extended ELVIS for Importance-Based Video Compression

PRESLEY extends ELVIS with multiple quality-reduction strategies that leverage
per-block importance scores to reduce bitrate while preserving foreground quality.

Strategies:
1. ELVIS (Frame Shrinking): Remove low-importance blocks, reconstruct via inpainting
2. ROI Encoding: Per-block quality control using external importance maps
3. Block Degradation (planned): Blur/downscale blocks, restore via super-resolution

Supported ROI Encoders:
- Kvazaar (HEVC): 16x16 CTU, continuous QP offsets, binary ROI file
- SVT-AV1 (AV1): 64x64 superblock, 8 QP levels (AV1 segment limit), text ROI file

Key components:
- EVCA: Edge-based Video Complexity Analysis for spatial/temporal complexity
- UFO: Unified Foundation Object segmentation for foreground detection

Note: x265 does NOT support external per-CTU delta QP (only internal AQ modes).
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple
import subprocess
import tempfile
import os
import numpy as np
import cv2
import torch
import pytorch_msssim

from evca import analyze_frames, EVCAConfig
from ufo import segment_frames
from propainter import ProPainterModel, InpaintingConfig as ProPainterConfig
from e2fgvi import E2FGVIModel, InpaintingConfig as E2FGVIConfig

# Restoration model imports (from presley venv)
from realesrgan import create_upsampler
from instantir import InstantIRRuntime, load_runtime, restore_images_batch
from upscale_a_video import UpscaleAVideo


# Quality presets: maps quality names to encoder-specific parameters
# Format: {"preset_name": {"kvazaar_qp": int, "svtav1_crf": int, "description": str}}
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
    # NOTE: Neural models hallucinate details which reduces SSIM but may improve perceptual quality
    # The key insight: SSIM measures pixel-level fidelity, neural models optimize for perceptual quality
    realesrgan_blend_base: float = 0.85  # High blend with original (only 15% SR) for SSIM preservation
    realesrgan_denoise_strength: float = 0.3  # Low denoise to preserve original detail
    realesrgan_pre_pad: int = 0  # Pre-padding for input
    realesrgan_fp32: bool = False  # Use FP32 instead of FP16
    # InstantIR parameters
    # NOTE: InstantIR is a diffusion model that hallucinates by design - expect lower SSIM
    # creative_start: 1.0 = max fidelity (creative mode disabled), smaller = more creative/less faithful
    # For SSIM-focused tasks, use OpenCV restoration instead
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


def encode_kvazaar(frames: List[np.ndarray], output_path: str, qp: int = 48, qp_range: int = 15, importance_scores: Optional[List[np.ndarray]] = None) -> None:
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
    write_y4m(frames, y4m_path, config.framerate)
    
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
        "--default-duration", f"0:{int(config.framerate)}fps",
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


def encode_svtav1(frames: List[np.ndarray], output_path: str, crf: int = 35, qp_range: int = 15, importance_scores: Optional[List[np.ndarray]] = None) -> None:
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
    write_y4m(frames, y4m_path, config.framerate)
    
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


def encode_video(frames: List[np.ndarray], output_path: str, quality: str = "medium", qp_range: int = None, importance_scores: Optional[List[np.ndarray]] = None, encoder: str = "kvazaar") -> None:
    """Encode video with specified encoder and quality preset.
    
    Args:
        frames: List of RGB frames to encode
        output_path: Output MP4 path
        quality: Quality preset name (see QUALITY_PRESETS)
        qp_range: Delta QP range for ROI encoding (None = use preset default)
        importance_scores: Optional per-block importance scores for ROI encoding
        encoder: Encoder to use ("kvazaar" or "svtav1")
    """
    if quality not in QUALITY_PRESETS:
        raise ValueError(f"Unknown quality preset: {quality}. Available: {list(QUALITY_PRESETS.keys())}")
    
    preset = QUALITY_PRESETS[quality]
    
    # Use preset qp_range if not explicitly specified
    effective_qp_range = qp_range if qp_range is not None else preset.get("qp_range", 12)
    
    if encoder == "kvazaar":
        encode_kvazaar(frames, output_path, qp=preset["kvazaar_qp"], 
                       qp_range=effective_qp_range, importance_scores=importance_scores)
    elif encoder == "svtav1":
        encode_svtav1(frames, output_path, crf=preset["svtav1_crf"], 
                      qp_range=effective_qp_range, importance_scores=importance_scores)
    else:
        raise ValueError(f"Unknown encoder: {encoder}. Supported: kvazaar, svtav1")


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


def compute_fg_bg_ssim(ssim_maps: List[np.ndarray], foreground_masks: np.ndarray,
                       fg_threshold: float = 0.5) -> Tuple[float, float, float]:
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
            fg_mask = cv2.resize(fg_mask.astype(np.float32), 
                                (ssim_map.shape[1], ssim_map.shape[0]),
                                interpolation=cv2.INTER_NEAREST)
        
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


def inpaint_propainter(frames: np.ndarray, masks: np.ndarray, device: str = "cuda") -> np.ndarray:
    """Inpaint frames using ProPainter."""
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


def restore_with_opencv_lanczos(frames: List[np.ndarray], degradation_maps: np.ndarray, 
                                block_size: int, halo: int = 0,
                                temporal_blend: float = 0.0, **kwargs) -> List[np.ndarray]:
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


def restore_with_opencv_unsharp(frames: List[np.ndarray], degradation_maps: np.ndarray, 
                                block_size: int, halo: int = 0,
                                temporal_blend: float = 0.0, **kwargs) -> List[np.ndarray]:
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


# =============================================================================
# RealESRGAN Restoration: Naive (whole-frame) and Adaptive (per-block)
# =============================================================================

def restore_with_realesrgan_naive(frames: List[np.ndarray], degradation_maps: np.ndarray, block_size: int, device: str = "cuda", model_name: str = "RealESRGAN_x4plus", tile: int = 256, tile_pad: int = 10, pre_pad: int = 0, fp32: bool = False, denoise_strength: float = 1.0, blend_with_original: float = 0.3, temporal_blend: float = 0.0) -> List[np.ndarray]:
    """Restore frames using RealESRGAN with naive whole-frame 4x upscaling.
    
    Upscales entire frame 4x then downscales to original size.
    Uses tiled processing internally for large images.
    
    Args:
        frames: List of RGB frames (H, W, 3)
        degradation_maps: Per-frame degradation maps (used for blending)
        block_size: Block size in pixels
        device: CUDA device string
        model_name: RealESRGAN model variant
        tile: Tile size for internal processing (0=no tiling)
        tile_pad: Padding around tiles
        pre_pad: Pre-padding for input
        fp32: Use FP32 instead of FP16
        denoise_strength: Denoise strength [0-1]
        blend_with_original: Blend factor (0=all restored, 1=all original)
        temporal_blend: Blend with previous frame (0=none, reduces flicker)
    
    Returns:
        Restored frames list
    """
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
    prev_output = None  # For temporal blending
    
    for i, frame in enumerate(frames):
        h, w = frame.shape[:2]
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # Upscale 4x with internal tiling
        upscaled, _ = upsampler.enhance(frame_bgr, outscale=4)
        result = cv2.resize(upscaled, (w, h), interpolation=cv2.INTER_LANCZOS4)
        result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        
        # Blend with original based on degradation map
        if blend_with_original > 0 and len(degradation_maps) > i:
            deg_map = degradation_maps[i]
            max_deg = deg_map.max() if deg_map.max() > 0 else 1
            weight = deg_map.astype(np.float32) / max_deg
            weight = cv2.resize(weight, (w, h), interpolation=cv2.INTER_LINEAR)
            weight = weight[:, :, np.newaxis]
            
            # Degraded regions get restoration, clean regions keep original
            blend_factor = (1 - blend_with_original) + blend_with_original * (1 - weight)
            result_rgb = (result_rgb * (1 - blend_factor) + frame * blend_factor).astype(np.uint8)
        
        # Apply temporal blending to reduce flickering
        if temporal_blend > 0 and prev_output is not None:
            result_rgb = (temporal_blend * prev_output + (1 - temporal_blend) * result_rgb).astype(np.uint8)
        
        prev_output = result_rgb.copy()
        restored.append(result_rgb)
    
    del upsampler
    torch.cuda.empty_cache()
    return restored


def restore_with_realesrgan_adaptive(frames: List[np.ndarray], degradation_maps: np.ndarray, block_size: int, device: str = "cuda", tile: int = 256, tile_pad: int = 10, pre_pad: int = 0, fp32: bool = False, denoise_strength: float = 1.0, blend_base: float = 0.5, temporal_blend: float = 0.0) -> List[np.ndarray]:
    """Restore frames using RealESRGAN with adaptive spatially-varying blending.
    
    Performs whole-frame 4x upscaling then uses degradation map to blend:
    - scale=0: Keep original (0% restoration)
    - scale=2: Light restoration (25% of blend_base)
    - scale=3: Medium restoration (50% of blend_base)  
    - scale=4: Full restoration (100% of blend_base)
    
    This is much faster than per-block processing while achieving similar results.
    
    Args:
        frames: List of RGB frames (H, W, 3)
        degradation_maps: Per-frame scale maps (0/2/3/4 per block)
        block_size: Block size in pixels
        device: CUDA device string
        tile: Tile size for internal processing (0=no tiling)
        tile_pad: Padding around tiles for internal processing
        pre_pad: Pre-padding for input
        fp32: Use FP32 instead of FP16
        denoise_strength: Denoise strength [0-1]
        blend_base: Base blend factor for maximum degradation (0=all restored, 1=all original)
        temporal_blend: Blend with previous frame (0=none, reduces flicker)
    
    Returns:
        Restored frames list
    """
    # Create single 4x upsampler (most effective for all degradation levels)
    upsampler = create_upsampler(
        model_name="RealESRGAN_x4plus",
        device=torch.device(device),
        tile=tile,
        tile_pad=tile_pad,
        pre_pad=pre_pad,
        half=not fp32,
        denoise_strength=denoise_strength
    )
    
    restored = []
    prev_output = None  # For temporal blending
    
    for i, frame in enumerate(frames):
        h, w = frame.shape[:2]
        blocks_y, blocks_x = h // block_size, w // block_size
        deg_map = degradation_maps[i] if len(degradation_maps) > i else np.zeros((blocks_y, blocks_x), dtype=np.float32)
        
        if deg_map.shape != (blocks_y, blocks_x):
            deg_map = cv2.resize(deg_map.astype(np.float32), (blocks_x, blocks_y),
                                interpolation=cv2.INTER_NEAREST)
        
        # Whole-frame upscaling (fast with internal tiling)
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        upscaled, _ = upsampler.enhance(frame_bgr, outscale=4)
        upscaled_rgb = cv2.cvtColor(upscaled, cv2.COLOR_BGR2RGB)
        
        # Resize back to original size
        restored_frame = cv2.resize(upscaled_rgb, (w, h), interpolation=cv2.INTER_LANCZOS4)
        
        # Create spatially-varying blend weight from degradation map
        # Higher degradation = more restoration (less blending with original)
        max_deg = deg_map.max() if deg_map.max() > 0 else 1
        weight = deg_map.astype(np.float32) / max_deg  # Normalize to [0, 1]
        
        # Upsample weight map to pixel level with smooth interpolation
        weight = cv2.resize(weight, (w, h), interpolation=cv2.INTER_LINEAR)
        weight = cv2.GaussianBlur(weight, (0, 0), block_size / 4)  # Smooth transitions
        weight = weight[:, :, np.newaxis]
        
        # blend_factor determines how much of restored vs original:
        # weight=0 (no degradation): keep original
        # weight=1 (max degradation): use (1-blend_base) restoration + blend_base original
        restoration_strength = weight * (1 - blend_base)
        output = (restoration_strength * restored_frame + (1 - restoration_strength) * frame).astype(np.uint8)
        
        # Apply temporal blending to reduce flickering
        if temporal_blend > 0 and prev_output is not None:
            output = (temporal_blend * prev_output + (1 - temporal_blend) * output).astype(np.uint8)
        
        prev_output = output.copy()
        restored.append(output)
    
    del upsampler
    torch.cuda.empty_cache()
    
    return restored


# =============================================================================
# InstantIR Restoration: Naive (whole-frame) and Adaptive (tiled)
# =============================================================================

def restore_with_instantir_naive(frames: List[np.ndarray], degradation_maps: np.ndarray, block_size: int, device: str = "cuda", weights_dir: str = "~/.cache/instantir", cfg: float = 7.0, creative_start: float = 1.0, preview_start: float = 0.0, seed: int = 42, tile_size: int = 512, num_inference_steps: int = 30, temporal_blend: float = 0.0) -> List[np.ndarray]:
    """Restore frames using InstantIR with naive whole-frame processing.
    
    For large images, processes in tiles with overlap blending.
    
    NOTE: InstantIR is a diffusion model that hallucinates details by design.
    For SSIM-focused evaluation, this will typically reduce SSIM compared to 
    OpenCV methods. InstantIR excels at perceptual quality, not fidelity.
    
    Args:
        frames: List of RGB frames (H, W, 3)
        degradation_maps: Per-frame degradation maps (not used for naive)
        block_size: Block size in pixels
        device: CUDA device string
        weights_dir: Path to InstantIR weights
        cfg: Classifier-free guidance scale (lower = less hallucination)
        creative_start: Control guidance end point (1.0 = max fidelity, lower = more creative)
        preview_start: Preview start point
        seed: Random seed
        tile_size: Tile size for processing large images
        num_inference_steps: Number of diffusion steps (lower = faster, less hallucination)
        temporal_blend: Blend with previous frame (0=none, reduces flicker)
    
    Returns:
        Restored frames list
    """
    import sys
    import io
    from contextlib import redirect_stdout, redirect_stderr
    from instantir import load_runtime, restore_image
    
    # Suppress InstantIR output during loading
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        runtime = load_runtime(
            instantir_path=Path(weights_dir).expanduser(),
            device=device,
            torch_dtype=torch.float16
        )
    
    restored = []
    prev_output = None  # For temporal blending
    halo = tile_size // 8  # Overlap for blending
    num_frames = len(frames)
    
    for i, frame in enumerate(frames):
        print(f"\r      InstantIR Naive: frame {i+1}/{num_frames}", end="", flush=True)
        h, w = frame.shape[:2]
        
        # Check if tiling is needed
        if h <= tile_size and w <= tile_size:
            # Process whole frame - suppress diffusion progress bars
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                result = restore_image(
                    runtime, frame,
                    cfg=cfg, preview_start=preview_start,
                    creative_start=creative_start,
                    num_inference_steps=num_inference_steps,
                    seed=seed + i, output_type="numpy"
            )
        else:
            # Tiled processing
            output = np.zeros_like(frame)
            weight_map = np.zeros((h, w), dtype=np.float32)
            
            for y in range(0, h, tile_size - halo):
                for x in range(0, w, tile_size - halo):
                    # Extract tile
                    y1 = min(y + tile_size, h)
                    x1 = min(x + tile_size, w)
                    tile = frame[y:y1, x:x1]
                    
                    # Process tile - suppress diffusion progress bars
                    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                        tile_result = restore_image(
                            runtime, tile,
                            cfg=cfg, preview_start=preview_start,
                            creative_start=creative_start,
                            num_inference_steps=num_inference_steps,
                            seed=seed + i + y * w + x,
                            output_type="numpy"
                        )
                    
                    # Create weight mask for blending (feather edges)
                    th, tw = tile_result.shape[:2]
                    tile_weight = np.ones((th, tw), dtype=np.float32)
                    # Feather edges
                    feather = min(halo // 2, th // 4, tw // 4)
                    if feather > 0:
                        for f in range(feather):
                            alpha = (f + 1) / feather
                            if y > 0: tile_weight[f, :] *= alpha
                            if x > 0: tile_weight[:, f] *= alpha
                            if y1 < h: tile_weight[th-1-f, :] *= alpha
                            if x1 < w: tile_weight[:, tw-1-f] *= alpha
                    
                    # Accumulate
                    output[y:y1, x:x1] = (
                        output[y:y1, x:x1] + 
                        tile_result * tile_weight[:, :, np.newaxis]
                    ).astype(np.uint8)
                    weight_map[y:y1, x:x1] += tile_weight
            
            # Normalize by weight
            weight_map = np.maximum(weight_map, 1e-8)
            result = (output.astype(np.float32) / weight_map[:, :, np.newaxis]).astype(np.uint8)
        
        # Apply temporal blending to reduce flickering
        if temporal_blend > 0 and prev_output is not None:
            result = (temporal_blend * prev_output + (1 - temporal_blend) * result).astype(np.uint8)
        
        prev_output = result.copy()
        restored.append(result)
    
    print()  # Newline after progress
    del runtime
    torch.cuda.empty_cache()
    return restored


def restore_with_instantir_adaptive(frames: List[np.ndarray], degradation_maps: np.ndarray, block_size: int, device: str = "cuda", weights_dir: str = "~/.cache/instantir", cfg: float = 7.0, creative_start: float = 1.0, preview_start: float = 0.0, seed: int = 42, halo: int = 8, num_inference_steps: int = 30, temporal_blend: float = 0.0) -> List[np.ndarray]:
    """Restore frames using InstantIR with adaptive per-region processing.
    
    Only processes regions that have degradation, preserving clean areas.
    Groups adjacent degraded blocks into regions for efficient processing.
    
    NOTE: InstantIR is a diffusion model that hallucinates details by design.
    For SSIM-focused evaluation, this will typically reduce SSIM compared to 
    OpenCV methods. InstantIR excels at perceptual quality, not fidelity.
    
    Args:
        frames: List of RGB frames (H, W, 3)
        degradation_maps: Per-frame degradation maps
        block_size: Block size in pixels
        device: CUDA device string
        weights_dir: Path to InstantIR weights
        cfg: CFG scale (lower = less hallucination)
        creative_start: Control guidance end point (0 = disable creative mode)
        preview_start: Preview start point
        seed: Random seed
        halo: Context halo in pixels
        num_inference_steps: Number of diffusion steps (lower = faster, less hallucination)
        temporal_blend: Blend with previous frame (0=none, reduces flicker)
    
    Returns:
        Restored frames list
    """
    import sys
    import io
    from contextlib import redirect_stdout, redirect_stderr
    from instantir import load_runtime, restore_image
    
    # Suppress InstantIR output during loading
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        runtime = load_runtime(
            instantir_path=Path(weights_dir).expanduser(),
            device=device,
            torch_dtype=torch.float16
        )
    
    restored = []
    prev_output = None  # For temporal blending
    num_frames = len(frames)
    
    for i, frame in enumerate(frames):
        print(f"\r      InstantIR Adaptive: frame {i+1}/{num_frames}", end="", flush=True)
        h, w = frame.shape[:2]
        blocks_y, blocks_x = h // block_size, w // block_size
        deg_map = degradation_maps[i] if len(degradation_maps) > i else np.zeros((blocks_y, blocks_x))
        
        if deg_map.shape != (blocks_y, blocks_x):
            deg_map = cv2.resize(deg_map.astype(np.float32), (blocks_x, blocks_y),
                                interpolation=cv2.INTER_NEAREST).astype(np.int32)
        
        output = frame.copy()
        
        # Find degraded regions (contiguous blocks)
        degraded_mask = deg_map > 0
        if not degraded_mask.any():
            # Apply temporal blending even to unchanged frames
            if temporal_blend > 0 and prev_output is not None:
                output = (temporal_blend * prev_output + (1 - temporal_blend) * output).astype(np.uint8)
            prev_output = output.copy()
            restored.append(output)
            continue
        
        # Process degraded blocks
        for by in range(blocks_y):
            for bx in range(blocks_x):
                if deg_map[by, bx] > 0:
                    y, x = by * block_size, bx * block_size
                    deg_level = deg_map[by, bx]
                    
                    # Extract block with halo
                    tile, crop = _extract_tile_with_halo(frame, y, x, block_size, block_size, halo)
                    
                    # Adaptive CFG based on degradation level (higher degradation = slightly more restoration)
                    # Keep CFG low to minimize hallucination
                    adaptive_cfg = cfg * (1 + deg_level * 0.05)
                    
                    # Process tile - suppress diffusion progress bars
                    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                        tile_result = restore_image(
                            runtime, tile,
                            cfg=adaptive_cfg, preview_start=preview_start,
                            creative_start=creative_start,
                            num_inference_steps=num_inference_steps,
                            seed=seed + i * blocks_y * blocks_x + by * blocks_x + bx,
                            output_type="numpy"
                        )
                    
                    # Crop and paste
                    result_block = tile_result[crop[0]:crop[2], crop[1]:crop[3]]
                    if result_block.shape[:2] != (block_size, block_size):
                        result_block = cv2.resize(result_block, (block_size, block_size), interpolation=cv2.INTER_LANCZOS4)
                    output[y:y+block_size, x:x+block_size] = result_block
        
        # Apply temporal blending to reduce flickering
        if temporal_blend > 0 and prev_output is not None:
            output = (temporal_blend * prev_output + (1 - temporal_blend) * output).astype(np.uint8)
        
        prev_output = output.copy()
        restored.append(output)
    
    print()  # Newline after progress
    del runtime
    torch.cuda.empty_cache()
    return restored


# =============================================================================
# Upscale-A-Video Restoration: Naive (whole-frame) and Adaptive (per-region)
# =============================================================================

def restore_with_upscale_a_video_naive(frames: List[np.ndarray], degradation_maps: np.ndarray, block_size: int, device: str = "cuda", noise_level: int = 120, guidance_scale: float = 6.0, inference_steps: int = 30, chunk_size: int = 16, chunk_overlap: int = 4, tile_size: int = 256) -> List[np.ndarray]:
    """Restore frames using Upscale-A-Video with naive whole-frame processing.
    
    For large frames, processes in spatial tiles.
    Maintains temporal consistency through chunk processing.
    
    Args:
        frames: List of RGB frames (H, W, 3)
        degradation_maps: Per-frame degradation maps
        block_size: Block size in pixels
        device: CUDA device string
        noise_level: Noise level [0-200]
        guidance_scale: CFG scale
        inference_steps: Denoising steps
        chunk_size: Temporal chunk size
        chunk_overlap: Temporal overlap
        tile_size: Spatial tile size for large images
    
    Returns:
        Restored frames list
    """
    upscaler = UpscaleAVideo(device=device)
    upscaler.load_models()
    
    n_frames = len(frames)
    h, w = frames[0].shape[:2]
    
    # Determine if spatial tiling is needed
    max_input = tile_size
    needs_tiling = h > max_input * 4 or w > max_input * 4
    
    if needs_tiling:
        # Process tiles independently with temporal chunks
        restored = [np.zeros_like(f) for f in frames]
        weight_maps = [np.zeros((h, w), dtype=np.float32) for _ in frames]
        
        tile_halo = tile_size // 4
        for ty in range(0, h, tile_size - tile_halo):
            for tx in range(0, w, tile_size - tile_halo):
                ty1, tx1 = min(ty + tile_size, h), min(tx + tile_size, w)
                
                # Extract tiles from all frames
                tiles = [f[ty:ty1, tx:tx1] for f in frames]
                tile_h, tile_w = tiles[0].shape[:2]
                
                # Downscale tiles for UAV input
                scale = min(max_input / tile_h, max_input / tile_w, 1.0)
                if scale < 1.0:
                    input_tiles = [cv2.resize(t, (int(tile_w * scale), int(tile_h * scale)), interpolation=cv2.INTER_AREA) for t in tiles]
                else:
                    input_tiles = tiles
                
                # Process temporal chunks
                tile_results = [None] * n_frames
                pos = 0
                while pos < n_frames:
                    end = min(pos + chunk_size, n_frames)
                    chunk_tiles = input_tiles[pos:end]
                    
                    try:
                        upscaled = upscaler.upscale_frames(
                            chunk_tiles,
                            noise_level=noise_level,
                            guidance_scale=guidance_scale,
                            inference_steps=inference_steps,
                            output_format="numpy"
                        )
                        for j, res in enumerate(upscaled):
                            idx = pos + j
                            tile_results[idx] = cv2.resize(res, (tile_w, tile_h), interpolation=cv2.INTER_LANCZOS4)
                    except Exception as e:
                        print(f"      UAV tile error at {ty},{tx} chunk {pos}: {e}")
                        for j in range(len(chunk_tiles)):
                            if tile_results[pos + j] is None:
                                tile_results[pos + j] = tiles[pos + j]
                    
                    pos += chunk_size - chunk_overlap
                    if pos + chunk_overlap >= n_frames:
                        break
                
                # Fill remaining
                for j in range(n_frames):
                    if tile_results[j] is None:
                        tile_results[j] = tiles[j]
                
                # Create weight mask
                tile_weight = np.ones((tile_h, tile_w), dtype=np.float32)
                feather = tile_halo // 2
                if feather > 0:
                    for f in range(feather):
                        alpha = (f + 1) / feather
                        if ty > 0: tile_weight[f, :] *= alpha
                        if tx > 0: tile_weight[:, f] *= alpha
                        if ty1 < h: tile_weight[tile_h-1-f, :] *= alpha
                        if tx1 < w: tile_weight[:, tile_w-1-f] *= alpha
                
                # Accumulate
                for j in range(n_frames):
                    restored[j][ty:ty1, tx:tx1] = (
                        restored[j][ty:ty1, tx:tx1].astype(np.float32) + tile_results[j] * tile_weight[:, :, np.newaxis]
                    )
                    weight_maps[j][ty:ty1, tx:tx1] += tile_weight
        
        # Normalize
        for j in range(n_frames):
            weight_maps[j] = np.maximum(weight_maps[j], 1e-8)
            restored[j] = (restored[j].astype(np.float32) / weight_maps[j][:, :, np.newaxis]).astype(np.uint8)
    else:
        # Standard processing with downscale if needed
        scale_factor = min(max_input / h, max_input / w, 1.0)
        if scale_factor < 1.0:
            input_h, input_w = int(h * scale_factor), int(w * scale_factor)
            scaled_frames = [cv2.resize(f, (input_w, input_h), interpolation=cv2.INTER_AREA) for f in frames]
            print(f"    Scaled {w}x{h} -> {input_w}x{input_h} for UAV")
        else:
            scaled_frames = frames
        
        restored = [None] * n_frames
        pos = 0
        while pos < n_frames:
            end = min(pos + chunk_size, n_frames)
            chunk = scaled_frames[pos:end]
            
            try:
                upscaled = upscaler.upscale_frames(
                    chunk, noise_level=noise_level,
                    guidance_scale=guidance_scale,
                    inference_steps=inference_steps,
                    output_format="numpy"
                )
                for j, res in enumerate(upscaled):
                    idx = pos + j
                    result = cv2.resize(res, (w, h), interpolation=cv2.INTER_LANCZOS4)
                    if idx < pos + chunk_overlap and restored[idx] is not None:
                        alpha = j / chunk_overlap
                        restored[idx] = (alpha * result + (1 - alpha) * restored[idx]).astype(np.uint8)
                    else:
                        restored[idx] = result
            except Exception as e:
                print(f"    UAV error at chunk {pos}: {e}")
                for j in range(len(chunk)):
                    if restored[pos + j] is None:
                        restored[pos + j] = frames[pos + j]
            
            pos += chunk_size - chunk_overlap
            if pos + chunk_overlap >= n_frames:
                break
        
        for i in range(n_frames):
            if restored[i] is None:
                restored[i] = frames[i]
    
    upscaler.unload_models()
    return restored


def restore_with_upscale_a_video_adaptive(frames: List[np.ndarray], degradation_maps: np.ndarray, block_size: int, device: str = "cuda", noise_level: int = 120, guidance_scale: float = 6.0, inference_steps: int = 30, halo: int = 8) -> List[np.ndarray]:
    """Restore frames using Upscale-A-Video with adaptive per-region processing.
    
    Only processes degraded regions, adapting noise level to degradation.
    Processes frame-by-frame (no temporal consistency) for per-block adaptation.
    
    Args:
        frames: List of RGB frames
        degradation_maps: Per-frame degradation maps
        block_size: Block size in pixels
        device: CUDA device string
        noise_level: Base noise level
        guidance_scale: CFG scale
        inference_steps: Denoising steps
        halo: Context halo in pixels
    
    Returns:
        Restored frames list
    """
    upscaler = UpscaleAVideo(device=device)
    upscaler.load_models()
    
    restored = []
    
    for i, frame in enumerate(frames):
        h, w = frame.shape[:2]
        blocks_y, blocks_x = h // block_size, w // block_size
        deg_map = degradation_maps[i] if len(degradation_maps) > i else np.zeros((blocks_y, blocks_x))
        
        if deg_map.shape != (blocks_y, blocks_x):
            deg_map = cv2.resize(deg_map.astype(np.float32), (blocks_x, blocks_y),
                                interpolation=cv2.INTER_NEAREST).astype(np.int32)
        
        output = frame.copy()
        
        if not (deg_map > 0).any():
            restored.append(output)
            continue
        
        # Process each degraded block
        for by in range(blocks_y):
            for bx in range(blocks_x):
                deg_level = deg_map[by, bx]
                if deg_level > 0:
                    y, x = by * block_size, bx * block_size
                    
                    # Extract block with halo
                    tile, crop = _extract_tile_with_halo(frame, y, x, block_size, block_size, halo)
                    
                    # Adaptive noise level based on degradation
                    adaptive_noise = int(noise_level * (1 + deg_level * 0.2))
                    
                    try:
                        # Process single-frame "video" through UAV
                        tile_result = upscaler.upscale_frames(
                            [tile],
                            noise_level=adaptive_noise,
                            guidance_scale=guidance_scale,
                            inference_steps=inference_steps,
                            output_format="numpy"
                        )[0]
                        
                        # Resize if needed
                        tile_h, tile_w = tile.shape[:2]
                        if tile_result.shape[:2] != (tile_h, tile_w):
                            tile_result = cv2.resize(tile_result, (tile_w, tile_h),
                                                     interpolation=cv2.INTER_LANCZOS4)
                        
                        # Crop and paste
                        result_block = tile_result[crop[0]:crop[2], crop[1]:crop[3]]
                        if result_block.shape[:2] != (block_size, block_size):
                            result_block = cv2.resize(result_block, (block_size, block_size),
                                                      interpolation=cv2.INTER_LANCZOS4)
                        output[y:y+block_size, x:x+block_size] = result_block
                    except Exception as e:
                        print(f"    UAV adaptive error at block ({by},{bx}): {e}")
                        # Keep original
        
        restored.append(output)
    
    upscaler.unload_models()
    return restored


# =============================================================================
# PRESLEY: ROI-Based Encoding
# Kvazaar (HEVC) and SVT-AV1 both support external per-block delta QP maps.
# =============================================================================

def create_kvazaar_roi_file(importance_scores: List[np.ndarray], roi_path: str, 
                            base_qp: int, qp_range: int = 15) -> None:
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


def create_svtav1_roi_file(importance_scores: List[np.ndarray], roi_path: str, 
                           base_crf: int, qp_range: int, width: int, height: int) -> None:
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
# Main Pipeline - Multi-Encoder Comparison
# =============================================================================

if __name__ == "__main__":
    # Setup experiment
    experiment_name = f"{Path(config.reference_video).stem}_{config.height}p_{config.quality}_bs{config.block_size}_a{config.alpha}_b{config.beta}_sa{config.shrink_amount}"
    print(f"Experiment: {experiment_name}")
    
    experiment_folder = Path("/home/itec/emanuele/elvis/experiments") / experiment_name
    experiment_folder.mkdir(parents=True, exist_ok=True)
    
    config.framerate = cv2.VideoCapture(config.reference_video).get(cv2.CAP_PROP_FPS)
    reference_frames = load_frames(config.reference_video, config.width, config.height)
    
    # Limit frames for faster testing
    if config.max_frames is not None and len(reference_frames) > config.max_frames:
        reference_frames = reference_frames[:config.max_frames]
        print(f"Limited to {config.max_frames} frames for testing")
    
    # EVCA: spatial and temporal complexity analysis
    evca = analyze_frames(np.array(reference_frames), EVCAConfig(block_size=config.block_size))
    
    # UFO: foreground segmentation
    ufo_masks = segment_frames(np.array(reference_frames), device="cuda:0" if torch.cuda.is_available() else "cpu")
    ufo_masks = np.array([cv2.resize(m.astype(np.float32), (config.width // config.block_size, config.height // config.block_size), interpolation=cv2.INTER_NEAREST) for m in ufo_masks])
    
    # Combine complexity and foreground into importance scores
    importance_scores = calculate_importance_scores(reference_frames, config.block_size, config.alpha, config.beta, evca, ufo_masks)
    print(f"Computed importance scores for {len(importance_scores)} frames")
    
    # Save visualizations
    save_frames([(s * 255).astype(np.uint8) for s in importance_scores], experiment_folder / "importance")
    save_frames([(m * 255).astype(np.uint8) for m in ufo_masks], experiment_folder / "foreground_masks")
    
    # ==========================================================================
    # Multi-Encoder Ablation Study
    # ==========================================================================
    encoders = ["kvazaar", "svtav1"]  # Use same quality preset for both
    
    results = {}
    
    for encoder_name in encoders:
        print(f"\n{'='*60}")
        print(f"Encoder: {encoder_name.upper()}")
        print(f"{'='*60}")
        
        encoder_folder = experiment_folder / encoder_name
        encoder_folder.mkdir(parents=True, exist_ok=True)
        
        # Baseline encoding (no ROI)
        baseline_path = encoder_folder / "baseline.mp4"
        encode_video(reference_frames, str(baseline_path), 
                     quality=config.quality, encoder=encoder_name)
        print(f"  Baseline: {baseline_path}")
        
        # ROI encoding (with importance scores)
        roi_path = encoder_folder / "roi_encoded.mp4"
        encode_video(reference_frames, str(roi_path), 
                     quality=config.quality, qp_range=config.qp_range,
                     importance_scores=importance_scores, encoder=encoder_name)
        print(f"  ROI: {roi_path}")
        
        # Load encoded frames
        baseline_frames = load_frames(str(baseline_path), config.width, config.height)
        roi_frames = load_frames(str(roi_path), config.width, config.height)
        
        # Calculate block SSIM
        ssim_baseline = calculate_block_ssim(reference_frames, baseline_frames, config.block_size)
        ssim_roi = calculate_block_ssim(reference_frames, roi_frames, config.block_size)
        
        # Save SSIM visualizations
        save_frames([(s * 255).astype(np.uint8) for s in ssim_baseline], encoder_folder / "ssim_baseline")
        save_frames([(s * 255).astype(np.uint8) for s in ssim_roi], encoder_folder / "ssim_roi")
        
        # Calculate FG/BG SSIM using the new unified function
        overall_baseline, fg_baseline, bg_baseline = compute_fg_bg_ssim(ssim_baseline, ufo_masks)
        overall_roi, fg_roi, bg_roi = compute_fg_bg_ssim(ssim_roi, ufo_masks)
        
        # File sizes
        baseline_size = os.path.getsize(baseline_path)
        roi_size = os.path.getsize(roi_path)
        
        # Store results
        results[encoder_name] = {
            "overall_ssim_baseline": overall_baseline,
            "overall_ssim_roi": overall_roi,
            "fg_ssim_baseline": fg_baseline,
            "fg_ssim_roi": fg_roi,
            "bg_ssim_baseline": bg_baseline,
            "bg_ssim_roi": bg_roi,
            "fg_improvement": (fg_roi - fg_baseline) * 100,
            "bg_improvement": (bg_roi - bg_baseline) * 100,
            "baseline_size": baseline_size,
            "roi_size": roi_size,
            "size_change_pct": (roi_size - baseline_size) / baseline_size * 100,
        }
        
        print(f"\n  Overall SSIM: Baseline={overall_baseline:.4f}, ROI={overall_roi:.4f}")
        print(f"  Foreground SSIM: Baseline={fg_baseline:.4f}, ROI={fg_roi:.4f} ({results[encoder_name]['fg_improvement']:+.2f}%)")
        print(f"  Background SSIM: Baseline={bg_baseline:.4f}, ROI={bg_roi:.4f} ({results[encoder_name]['bg_improvement']:+.2f}%)")
        print(f"  File sizes: Baseline={baseline_size:,}, ROI={roi_size:,} ({results[encoder_name]['size_change_pct']:+.1f}%)")
    
    # ==========================================================================
    # ELVIS: Frame shrinking and inpainting (uses svtav1 for encoding)
    # ==========================================================================

    print(f"\n{'='*60}")
    print("Frame Shrinking Ablation Study")
    print(f"{'='*60}")
    
    shrink_methods = {
        "row_only": {
            "shrink": shrink_frame_row_only,
            "stretch": lambda shrunk, data, bs: stretch_frame_row_only(shrunk, data["mask"], bs),
            "description": "Row-only removal (original ELVIS)"
        },
        "position_map": {
            "shrink": shrink_frame_position_map,
            "stretch": lambda shrunk, data, bs: stretch_frame_position_map(shrunk, data["mask"], data["position_map"], bs),
            "description": "Row+Col with position_map"
        },
        "removal_indices": {
            "shrink": shrink_frame_removal_indices,
            "stretch": lambda shrunk, data, bs: stretch_frame_removal_indices(shrunk, data["removal_indices"], data["orig_blocks_y"], data["orig_blocks_x"], bs),
            "description": "Row+Col with removal_indices"
        }
    }
    
    shrink_results = {}
    
    for method_name, method in shrink_methods.items():
        print(f"\n  Method: {method['description']}")
        print(f"  {'-'*50}")
        
        method_folder = experiment_folder / f"shrink_{method_name}"
        method_folder.mkdir(parents=True, exist_ok=True)
        
        # Shrink frames
        shrunken_frames, shrink_data = [], []
        for i, frame in enumerate(reference_frames):
            result = method["shrink"](frame, importance_scores[i], config.block_size, config.shrink_amount)
            shrunken_frames.append(result[0])
            
            # Store data needed for stretching
            if method_name == "row_only":
                shrink_data.append({"mask": result[1]})
            elif method_name == "position_map":
                shrink_data.append({"mask": result[1], "position_map": result[2]})
            else:  # removal_indices
                orig_blocks_y = config.height // config.block_size
                orig_blocks_x = config.width // config.block_size
                shrink_data.append({
                    "mask": result[1], 
                    "removal_indices": result[2],
                    "orig_blocks_y": orig_blocks_y,
                    "orig_blocks_x": orig_blocks_x
                })
        
        # Encode shrunken video
        shrunken_path = method_folder / "shrunken.mp4"
        encode_video(shrunken_frames, str(shrunken_path), quality=config.quality, encoder="svtav1")
        print(f"    Shrunken: {shrunken_frames[0].shape[1]}x{shrunken_frames[0].shape[0]} -> {shrunken_path}")
        
        # Save removal masks
        removal_masks = [d["mask"] for d in shrink_data]
        save_frames([(m * 255).astype(np.uint8) for m in removal_masks], method_folder / "removal_masks")
        np.savez_compressed(method_folder / "removal_masks.npz", masks=np.array(removal_masks))
        
        # Save method-specific data
        if method_name == "position_map":
            np.savez_compressed(method_folder / "position_maps.npz", 
                              maps=np.array([d["position_map"] for d in shrink_data], dtype=object))
        elif method_name == "removal_indices":
            np.savez_compressed(method_folder / "removal_indices.npz",
                              indices=np.array([d["removal_indices"] for d in shrink_data], dtype=object))
        
        # Stretch frames back
        stretched_frames = []
        for i, shrunk in enumerate(shrunken_frames):
            stretched = method["stretch"](shrunk, shrink_data[i], config.block_size)
            stretched_frames.append(stretched)
        stretched_frames = np.array(stretched_frames)
        
        # Scale removal masks to pixel-level for inpainting
        inpaint_masks = np.array([
            cv2.resize(m.astype(np.float32), (config.width, config.height), interpolation=cv2.INTER_NEAREST) * 255 
            for m in removal_masks
        ]).astype(np.uint8)
        
        # Verify stretched dimensions match original
        if stretched_frames[0].shape[:2] != (config.height, config.width):
            print(f"    WARNING: Stretched size {stretched_frames[0].shape[1]}x{stretched_frames[0].shape[0]} != original {config.width}x{config.height}")
        
        # Inpaint with ProPainter
        print(f"    Inpainting with ProPainter...")
        propainter_frames = inpaint_propainter(stretched_frames, inpaint_masks, device="cuda")
        propainter_path = method_folder / "propainter.mp4"
        encode_video(list(propainter_frames), str(propainter_path), quality="lossless", encoder="svtav1")
        
        # Inpaint with E2FGVI
        print(f"    Inpainting with E2FGVI...")
        e2fgvi_frames = inpaint_e2fgvi(stretched_frames, inpaint_masks, device="cuda")
        e2fgvi_path = method_folder / "e2fgvi.mp4"
        encode_video(list(e2fgvi_frames), str(e2fgvi_path), quality="lossless", encoder="svtav1")
        
        # Calculate metrics using GPU-accelerated block SSIM
        propainter_frames_loaded = load_frames(str(propainter_path), config.width, config.height)
        e2fgvi_frames_loaded = load_frames(str(e2fgvi_path), config.width, config.height)
        
        ssim_maps_propainter = calculate_block_ssim(reference_frames, propainter_frames_loaded, config.block_size)
        ssim_maps_e2fgvi = calculate_block_ssim(reference_frames, e2fgvi_frames_loaded, config.block_size)
        
        pp_overall, pp_fg, pp_bg = compute_fg_bg_ssim(ssim_maps_propainter, ufo_masks)
        e2_overall, e2_fg, e2_bg = compute_fg_bg_ssim(ssim_maps_e2fgvi, ufo_masks)
        
        shrunken_size = os.path.getsize(shrunken_path)
        propainter_size = os.path.getsize(propainter_path)
        e2fgvi_size = os.path.getsize(e2fgvi_path)
        
        shrink_results[method_name] = {
            "shrunken_size": shrunken_size,
            "propainter_ssim": pp_overall,
            "propainter_fg_ssim": pp_fg,
            "propainter_bg_ssim": pp_bg,
            "e2fgvi_ssim": e2_overall,
            "e2fgvi_fg_ssim": e2_fg,
            "e2fgvi_bg_ssim": e2_bg,
            "propainter_size": propainter_size,
            "e2fgvi_size": e2fgvi_size,
        }
        
        print(f"    ProPainter: Overall={pp_overall:.4f}, FG={pp_fg:.4f}, BG={pp_bg:.4f}, size: {propainter_size:,}")
        print(f"    E2FGVI: Overall={e2_overall:.4f}, FG={e2_fg:.4f}, BG={e2_bg:.4f}, size: {e2fgvi_size:,}")
        print(f"    Shrunken size: {shrunken_size:,}")
    
    # Print summary
    print(f"\n{'='*60}")
    print("Shrinking Method Comparison")
    print(f"{'='*60}")
    for method_name, res in shrink_results.items():
        print(f"  {method_name}:")
        print(f"    ProPainter: Overall={res['propainter_ssim']:.4f}, FG={res['propainter_fg_ssim']:.4f}, BG={res['propainter_bg_ssim']:.4f}")
        print(f"    E2FGVI:     Overall={res['e2fgvi_ssim']:.4f}, FG={res['e2fgvi_fg_ssim']:.4f}, BG={res['e2fgvi_bg_ssim']:.4f}")
        print(f"    Shrunken size: {res['shrunken_size']:,} bytes")

    # ==========================================================================
    # PRESLEY: Adaptive Degradation & Restoration Ablation Study
    # ==========================================================================
    
    print(f"\n{'='*60}")
    print("Adaptive Degradation & Restoration Ablation Study")
    print(f"{'='*60}")
    
    # Define degradation methods
    degradation_methods = {
        "downsample": {
            "fn": degrade_adaptive_downsample,
            "kwargs": {"max_scale": config.downsample_max_scale},
            "description": f"Adaptive Downsample (scale 0/2/3/4, max {config.downsample_max_scale}x)",
            "restoration_filter": ["opencv_lanczos", "realesrgan_naive", "realesrgan_adaptive"]
        },
        "blur": {
            "fn": degrade_adaptive_blur,
            "kwargs": {"max_rounds": config.blur_max_rounds},
            "description": f"Adaptive Gaussian Blur (max {config.blur_max_rounds} rounds)",
            "restoration_filter": ["opencv_unsharp", "instantir_naive", "instantir_adaptive"]
        }
    }
    
    # Define restoration methods (naive and adaptive variants)
    restoration_methods = {
        "opencv_lanczos": {
            "fn": restore_with_opencv_lanczos,
            "kwargs": {"halo": config.context_halo, "temporal_blend": config.temporal_blend},
            "description": "OpenCV Sharpening (per-block, with halo)"
        },
        "opencv_unsharp": {
            "fn": restore_with_opencv_unsharp,
            "kwargs": {"halo": config.context_halo, "temporal_blend": config.temporal_blend},
            "description": "OpenCV Unsharp Mask (per-block, with halo)"
        },
        "realesrgan_naive": {
            "fn": restore_with_realesrgan_naive,
            "kwargs": {
                "model_name": "RealESRGAN_x4plus",
                "tile": config.neural_tile_size,
                "tile_pad": config.context_halo,
                "pre_pad": config.realesrgan_pre_pad,
                "fp32": config.realesrgan_fp32,
                "denoise_strength": config.realesrgan_denoise_strength,
                "blend_with_original": config.realesrgan_blend_base,
                "temporal_blend": config.temporal_blend
            },
            "description": f"RealESRGAN Naive (4x, {int(config.realesrgan_blend_base*100)}% orig blend)"
        },
        "realesrgan_adaptive": {
            "fn": restore_with_realesrgan_adaptive,
            "kwargs": {
                "tile": config.neural_tile_size,
                "tile_pad": config.context_halo,
                "pre_pad": config.realesrgan_pre_pad,
                "fp32": config.realesrgan_fp32,
                "denoise_strength": config.realesrgan_denoise_strength,
                "blend_base": config.realesrgan_blend_base,
                "temporal_blend": config.temporal_blend
            },
            "description": "RealESRGAN Adaptive (spatially-varying blend)"
        },
        "instantir_naive": {
            "fn": restore_with_instantir_naive,
            "kwargs": {
                "weights_dir": "~/.cache/instantir",
                "cfg": config.instantir_cfg,
                "creative_start": config.instantir_creative_start,
                "preview_start": config.instantir_preview_start,
                "seed": config.instantir_seed,
                "tile_size": config.neural_tile_size,
                "num_inference_steps": config.instantir_steps,
                "temporal_blend": config.temporal_blend
            },
            "description": f"InstantIR Naive (cfg={config.instantir_cfg}, steps={config.instantir_steps})"
        },
        "instantir_adaptive": {
            "fn": restore_with_instantir_adaptive,
            "kwargs": {
                "weights_dir": "~/.cache/instantir",
                "cfg": config.instantir_cfg,
                "creative_start": config.instantir_creative_start,
                "preview_start": config.instantir_preview_start,
                "seed": config.instantir_seed,
                "halo": config.context_halo,
                "num_inference_steps": config.instantir_steps,
                "temporal_blend": config.temporal_blend
            },
            "description": f"InstantIR Adaptive (cfg={config.instantir_cfg}, steps={config.instantir_steps})"
        },
        "uav_naive": {
            "fn": restore_with_upscale_a_video_naive,
            "kwargs": {
                "noise_level": config.uav_noise_level,
                "guidance_scale": config.uav_guidance_scale,
                "inference_steps": config.uav_inference_steps,
                "tile_size": config.uav_tile_size
            },
            "description": "Upscale-A-Video Naive (temporal, whole-frame)"
        },
        "uav_adaptive": {
            "fn": restore_with_upscale_a_video_adaptive,
            "kwargs": {
                "noise_level": config.uav_noise_level,
                "guidance_scale": config.uav_guidance_scale,
                "inference_steps": config.uav_inference_steps,
                "halo": config.context_halo
            },
            "description": "Upscale-A-Video Adaptive (per-block)"
        }
    }
    
    degradation_results = {}
    
    for deg_name, deg_method in degradation_methods.items():
        print(f"\n  Degradation: {deg_method['description']}")
        print(f"  {'-'*50}")
        
        deg_folder = experiment_folder / f"degrade_{deg_name}"
        deg_folder.mkdir(parents=True, exist_ok=True)
        
        # Apply degradation to all frames
        degraded_frames = []
        degradation_maps = []
        for i, frame in enumerate(reference_frames):
            degraded, deg_map = deg_method["fn"](
                frame, importance_scores[i], config.block_size, **deg_method["kwargs"]
            )
            degraded_frames.append(degraded)
            degradation_maps.append(deg_map)
        degradation_maps = np.array(degradation_maps)
        
        # Encode degraded video
        degraded_path = deg_folder / "degraded.mp4"
        encode_video(degraded_frames, str(degraded_path), quality=config.quality, encoder="svtav1")
        degraded_size = os.path.getsize(degraded_path)
        print(f"    Degraded video: {degraded_path} ({degraded_size:,} bytes)")
        
        # Save degradation maps
        save_frames([(m / m.max() * 255).astype(np.uint8) if m.max() > 0 else np.zeros_like(m, dtype=np.uint8) 
                    for m in degradation_maps], deg_folder / "degradation_maps")
        np.savez_compressed(deg_folder / "degradation_maps.npz", maps=degradation_maps)
        
        # Calculate degraded SSIM using GPU-accelerated block SSIM
        degraded_ssim_maps = calculate_block_ssim(reference_frames, degraded_frames, config.block_size)
        degraded_overall, degraded_fg, degraded_bg = compute_fg_bg_ssim(degraded_ssim_maps, ufo_masks)
        print(f"    Degraded SSIM: Overall={degraded_overall:.4f}, FG={degraded_fg:.4f}, BG={degraded_bg:.4f}")
        
        degradation_results[deg_name] = {
            "degraded_size": degraded_size,
            "degraded_ssim": degraded_overall,
            "degraded_fg_ssim": degraded_fg,
            "degraded_bg_ssim": degraded_bg,
            "restorations": {}
        }
        
        # Filter restoration methods for this degradation type
        compatible_methods = deg_method.get("restoration_filter", list(restoration_methods.keys()))
        
        # Apply each compatible restoration method
        for rest_name in compatible_methods:
            if rest_name not in restoration_methods:
                continue
            rest_method = restoration_methods[rest_name]
            print(f"\n    Restoration: {rest_method['description']}")
            
            rest_folder = deg_folder / f"restored_{rest_name}"
            rest_folder.mkdir(parents=True, exist_ok=True)
            
            try:
                # Restore frames
                restored_frames = rest_method["fn"](
                    degraded_frames, degradation_maps, config.block_size,
                    device="cuda" if torch.cuda.is_available() else "cpu",
                    **rest_method["kwargs"]
                )
                
                # Encode restored video
                restored_path = rest_folder / "restored.mp4"
                encode_video(restored_frames, str(restored_path), quality="lossless", encoder="svtav1")
                restored_size = os.path.getsize(restored_path)
                
                # Calculate restored SSIM using GPU-accelerated block SSIM
                restored_ssim_maps = calculate_block_ssim(reference_frames, restored_frames, config.block_size)
                restored_overall, restored_fg, restored_bg = compute_fg_bg_ssim(restored_ssim_maps, ufo_masks)
                
                ssim_improvement = restored_overall - degraded_overall
                fg_improvement = restored_fg - degraded_fg
                bg_improvement = restored_bg - degraded_bg
                
                degradation_results[deg_name]["restorations"][rest_name] = {
                    "restored_ssim": restored_overall,
                    "restored_fg_ssim": restored_fg,
                    "restored_bg_ssim": restored_bg,
                    "restored_size": restored_size,
                    "ssim_improvement": ssim_improvement,
                    "fg_improvement": fg_improvement,
                    "bg_improvement": bg_improvement
                }
                
                print(f"      Overall: {restored_overall:.4f} ({ssim_improvement:+.4f})")
                print(f"      FG SSIM: {restored_fg:.4f} ({fg_improvement:+.4f})")
                print(f"      BG SSIM: {restored_bg:.4f} ({bg_improvement:+.4f})")
                print(f"      Size: {restored_size:,} bytes")
                
            except Exception as e:
                print(f"      ERROR: {e}")
                degradation_results[deg_name]["restorations"][rest_name] = {
                    "error": str(e)
                }
    
    # ==========================================================================
    # Print Final Summary
    # ==========================================================================
    
    print(f"\n{'='*60}")
    print("DEGRADATION/RESTORATION ABLATION SUMMARY")
    print(f"{'='*60}")
    
    for deg_name, deg_res in degradation_results.items():
        print(f"\n{deg_name.upper()}:")
        print(f"  Degraded: Overall={deg_res['degraded_ssim']:.4f}, FG={deg_res['degraded_fg_ssim']:.4f}, BG={deg_res['degraded_bg_ssim']:.4f}")
        print(f"  Size: {deg_res['degraded_size']:,} bytes")
        for rest_name, rest_res in deg_res["restorations"].items():
            if "error" in rest_res:
                print(f"  -> {rest_name}: ERROR - {rest_res['error']}")
            else:
                print(f"  -> {rest_name}:")
                print(f"       Overall: {rest_res['restored_ssim']:.4f} ({rest_res['ssim_improvement']:+.4f})")
                print(f"       FG:      {rest_res['restored_fg_ssim']:.4f} ({rest_res['fg_improvement']:+.4f})")
                print(f"       BG:      {rest_res['restored_bg_ssim']:.4f} ({rest_res['bg_improvement']:+.4f})")
    
    # Save all results to JSON
    import json
    all_results = {
        "config": {
            "video": config.reference_video,
            "width": config.width,
            "height": config.height,
            "quality": config.quality,
            "block_size": config.block_size,
            "shrink_amount": config.shrink_amount
        },
        "roi_encoding": results,
        "shrinking": shrink_results,
        "degradation": degradation_results
    }
    
    with open(experiment_folder / "results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"\nResults saved to {experiment_folder / 'results.json'}")
