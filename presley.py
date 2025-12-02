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


# Quality presets: maps quality names to encoder-specific parameters
# Format: {"preset_name": {"kvazaar_qp": int, "svtav1_crf": int, "description": str}}
# Optimal qp_range varies by quality level (empirically determined):
#   - QP 30-35: qp_range 12 gives best efficiency
#   - QP 38-45: qp_range 10-12 works well
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
    width: int = 1280
    height: int = 720
    framerate: float = None  # Auto-detected from reference video
    quality: str = "low"  # Quality preset (see QUALITY_PRESETS) - best efficiency
    qp_range: int = None  # Auto-set from quality preset, or override manually
    block_size: int = 16  # Block size for importance calculation
    alpha: float = 0.5  # Weight for spatial vs temporal complexity
    beta: float = 0.5  # Smoothing factor for importance scores
    shrink_amount: float = 0.25  # Fraction of blocks to remove in shrinking
    propainter_ref_stride: int = 20
    propainter_neighbor_length: int = 4
    propainter_subvideo_length: int = 40
    propainter_mask_dilation: int = 4
    propainter_raft_iter: int = 20
    propainter_fp16: bool = True
    e2fgvi_ref_stride: int = 10
    e2fgvi_neighbor_stride: int = 5
    e2fgvi_num_ref: int = -1
    e2fgvi_mask_dilation: int = 4
    # realesrgan_denoise_strength: float = 1.0
    # realesrgan_tile: int = 0
    # realesrgan_tile_pad: int = 10
    # realesrgan_pre_pad: int = 0
    # realesrgan_fp32: bool = False
    # realesrgan_devices: Optional[List[Union[int, str]]] = None
    # realesrgan_parallel_chunk_length: Optional[int] = None
    # realesrgan_per_device_workers: int = 1
    # instantir_cfg: float = 7.0
    # instantir_creative_start: float = 1.0
    # instantir_preview_start: float = 0.0
    # instantir_seed: Optional[int] = 42
    # instantir_devices: Optional[List[Union[int, str]]] = None
    # instantir_batch_size: int = 4
    # instantir_parallel_chunk_length: Optional[int] = None
    # generate_opencv_benchmarks: bool = True
    # metric_stride: int = 1
    # fvmd_stride: int = 1
    # fvmd_max_frames: Optional[int] = None
    # fvmd_processes: Optional[int] = None
    # fvmd_early_stop_delta: float = 0.002
    # fvmd_early_stop_window: int = 50
    # vmaf_stride: int = 1
    # enable_fvmd: bool = True


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


def get_video_framerate(video_path: str) -> float:
    """Extract framerate from video file."""
    cap = cv2.VideoCapture(video_path)
    framerate = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return framerate


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


# =============================================================================
# Shrinking Method 1: Row-only removal (original ELVIS style)
# =============================================================================
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


# =============================================================================
# Shrinking Method 2: Row+Col removal with position_map
# =============================================================================
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


# =============================================================================
# Shrinking Method 3: Row+Col removal with removal_indices list
# =============================================================================
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
    
    config.framerate = get_video_framerate(config.reference_video)
    reference_frames = load_frames(config.reference_video, config.width, config.height)
    
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
        
        # Calculate metrics
        avg_ssim_baseline = np.mean([s.mean() for s in ssim_baseline])
        avg_ssim_roi = np.mean([s.mean() for s in ssim_roi])
        
        # Foreground-only SSIM
        fg_ssim_baseline, fg_ssim_roi = [], []
        for ssim_b, ssim_r, imp in zip(ssim_baseline, ssim_roi, importance_scores):
            fg_mask = imp > 0.5
            if fg_mask.any():
                fg_ssim_baseline.append(ssim_b[fg_mask].mean())
                fg_ssim_roi.append(ssim_r[fg_mask].mean())
        avg_fg_baseline = np.mean(fg_ssim_baseline) if fg_ssim_baseline else 0.0
        avg_fg_roi = np.mean(fg_ssim_roi) if fg_ssim_roi else 0.0
        
        # File sizes
        baseline_size = os.path.getsize(baseline_path)
        roi_size = os.path.getsize(roi_path)
        
        # Store results
        results[encoder_name] = {
            "overall_ssim_baseline": avg_ssim_baseline,
            "overall_ssim_roi": avg_ssim_roi,
            "fg_ssim_baseline": avg_fg_baseline,
            "fg_ssim_roi": avg_fg_roi,
            "fg_improvement": (avg_fg_roi - avg_fg_baseline) * 100,
            "baseline_size": baseline_size,
            "roi_size": roi_size,
            "size_change_pct": (roi_size - baseline_size) / baseline_size * 100,
        }
        
        print(f"\n  Overall SSIM: Baseline={avg_ssim_baseline:.4f}, ROI={avg_ssim_roi:.4f}")
        print(f"  Foreground SSIM: Baseline={avg_fg_baseline:.4f}, ROI={avg_fg_roi:.4f}")
        print(f"  Foreground improvement: {results[encoder_name]['fg_improvement']:.2f}%")
        print(f"  File sizes: Baseline={baseline_size:,}, ROI={roi_size:,} ({results[encoder_name]['size_change_pct']:+.1f}%)")
    
    # Frame shrinking (uses svtav1 for encoding)
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
        
        # Calculate metrics
        propainter_frames_loaded = load_frames(str(propainter_path), config.width, config.height)
        e2fgvi_frames_loaded = load_frames(str(e2fgvi_path), config.width, config.height)
        
        ssim_propainter = np.mean([pytorch_msssim.ssim(
            torch.from_numpy(ref).permute(2, 0, 1).unsqueeze(0).float(),
            torch.from_numpy(out).permute(2, 0, 1).unsqueeze(0).float(),
            data_range=255
        ).item() for ref, out in zip(reference_frames, propainter_frames_loaded)])
        
        ssim_e2fgvi = np.mean([pytorch_msssim.ssim(
            torch.from_numpy(ref).permute(2, 0, 1).unsqueeze(0).float(),
            torch.from_numpy(out).permute(2, 0, 1).unsqueeze(0).float(),
            data_range=255
        ).item() for ref, out in zip(reference_frames, e2fgvi_frames_loaded)])
        
        shrunken_size = os.path.getsize(shrunken_path)
        propainter_size = os.path.getsize(propainter_path)
        e2fgvi_size = os.path.getsize(e2fgvi_path)
        
        shrink_results[method_name] = {
            "shrunken_size": shrunken_size,
            "propainter_ssim": ssim_propainter,
            "e2fgvi_ssim": ssim_e2fgvi,
            "propainter_size": propainter_size,
            "e2fgvi_size": e2fgvi_size,
        }
        
        print(f"    ProPainter SSIM: {ssim_propainter:.4f}, size: {propainter_size:,}")
        print(f"    E2FGVI SSIM: {ssim_e2fgvi:.4f}, size: {e2fgvi_size:,}")
        print(f"    Shrunken size: {shrunken_size:,}")
    
    # Print summary
    print(f"\n{'='*60}")
    print("Shrinking Method Comparison")
    print(f"{'='*60}")
    for method_name, res in shrink_results.items():
        print(f"  {method_name}:")
        print(f"    ProPainter SSIM: {res['propainter_ssim']:.4f}")
        print(f"    E2FGVI SSIM: {res['e2fgvi_ssim']:.4f}")
        print(f"    Shrunken size: {res['shrunken_size']:,} bytes")