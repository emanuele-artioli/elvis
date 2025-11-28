"""
PRESLEY: Extended ELVIS for Importance-Based Video Compression

PRESLEY extends ELVIS with multiple quality-reduction strategies that leverage
per-block importance scores to reduce bitrate while preserving foreground quality.

Strategies:
1. ELVIS (Frame Shrinking): Remove low-importance blocks, reconstruct via inpainting
2. ROI Encoding: Per-CTU quality control using kvazaar's --roi option
3. Block Degradation (planned): Blur/downscale blocks, restore via super-resolution

Key components:
- EVCA: Edge-based Video Complexity Analysis for spatial/temporal complexity
- UFO: Unified Foundation Object segmentation for foreground detection
- Kvazaar: HEVC encoder with per-CTU delta QP support via --roi option
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
import subprocess
import tempfile
import os
import numpy as np
import cv2
import torch
import pytorch_msssim

from evca import analyze_frames, EVCAConfig
from ufo import segment_frames


@dataclass
class PresleyConfig:
    """Configuration for the ELVIS/PRESLEY encoding pipeline."""
    reference_video: str = "/home/itec/emanuele/Datasets/DAVIS/avc_encoded/bear.mp4"
    width: int = 1280
    height: int = 720
    framerate: float = None  # Auto-detected from reference video
    base_qp: int = 45  # Base quality level for kvazaar encoding
    qp_range: int = 15  # Delta QP range for ROI encoding
    block_size: int = 16  # Block size for importance calculation
    alpha: float = 0.5  # Weight for spatial vs temporal complexity
    beta: float = 0.5  # Smoothing factor for importance scores
    shrink_amount: float = 0.25  # Fraction of blocks to remove in shrinking
    # propainter_resize_ratio: float = 1.0
    # propainter_ref_stride: int = 20
    # propainter_neighbor_length: int = 4
    # propainter_subvideo_length: int = 40
    # propainter_mask_dilation: int = 4
    # propainter_raft_iter: int = 20
    # propainter_fp16: bool = True
    # propainter_devices: Optional[List[Union[int, str]]] = None
    # propainter_parallel_chunk_length: Optional[int] = None
    # propainter_chunk_overlap: Optional[int] = None
    # e2fgvi_ref_stride: int = 10
    # e2fgvi_neighbor_stride: int = 5
    # e2fgvi_num_ref: int = -1
    # e2fgvi_mask_dilation: int = 4
    # e2fgvi_devices: Optional[List[Union[int, str]]] = None
    # e2fgvi_parallel_chunk_length: Optional[int] = None
    # e2fgvi_chunk_overlap: Optional[int] = None
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


def encode_video(frames: List[np.ndarray], output_path: str, qp: int = 48, qp_range: int = 15, importance_scores: Optional[List[np.ndarray]] = None) -> None:
    """Encode video using kvazaar HEVC encoder with optional ROI-based quality control."""
    height, width = frames[0].shape[:2]
    output_path = Path(output_path)
    
    # Create temporary Y4M file (kvazaar's preferred input format)
    with tempfile.NamedTemporaryFile(suffix='.y4m', delete=False) as tmp:
        y4m_path = tmp.name
    
    # Write Y4M file with YUV420 color space
    fps_num = int(round(config.framerate * 1000))
    with open(y4m_path, 'wb') as f:
        f.write(f"YUV4MPEG2 W{width} H{height} F{fps_num}:1000 Ip A1:1 C420\n".encode())
        for frame in frames:
            f.write(b"FRAME\n")
            yuv = cv2.cvtColor(frame, cv2.COLOR_RGB2YUV_I420)
            f.write(yuv.tobytes())
    
    # Build kvazaar command
    hevc_path = str(output_path).replace('.mp4', '.hevc')
    cmd = ["kvazaar", "-i", y4m_path, "-q", str(qp), "-o", hevc_path, "--preset", "medium"]
    
    # Add ROI file if importance scores provided
    roi_path = None
    if importance_scores is not None:
        roi_path = str(output_path).replace('.mp4', '_roi.bin')
        create_roi_file(importance_scores, roi_path, qp_range)
        cmd.extend(["--roi", roi_path])
    
    # Run kvazaar (may crash at end due to memory bug, but output is valid)
    subprocess.run(cmd, capture_output=True, text=True)
    
    if not os.path.exists(hevc_path) or os.path.getsize(hevc_path) == 0:
        raise RuntimeError(f"Kvazaar failed to produce output: {hevc_path}")
    
    # Mux HEVC to MP4 using mkvmerge for proper timestamps, then ffmpeg for MP4
    # Raw HEVC has no timestamps; mkvmerge adds them based on framerate
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


def shrink_frame(frame: np.ndarray, importance: np.ndarray, block_size: int, shrink_amount: float):
    """Iteratively removes the least important block from each row/column, producing a smaller frame. Also returns mask of removed blocks."""

    frame = frame.copy()
    importance = importance.copy()
    
    height, width = frame.shape[:2]
    blocks_y = height // block_size
    blocks_x = width // block_size
    orig_blocks_y, orig_blocks_x = blocks_y, blocks_x
    
    # Reshape into blocks
    blocked = frame[:blocks_y * block_size, :blocks_x * block_size].reshape(blocks_y, block_size, blocks_x, block_size, 3)
    
    # Track original positions for removal mask
    origins = np.empty((orig_blocks_y, orig_blocks_x, 2), dtype=np.int32)
    origins = np.array([[(by, bx) for bx in range(orig_blocks_x)] for by in range(orig_blocks_y)])
    removal_mask = np.zeros((orig_blocks_y, orig_blocks_x), dtype=bool)
    target_removals = int(orig_blocks_y * orig_blocks_x * shrink_amount)
    removed = 0
    
    while removed < target_removals:
        # Remove least important block from each row
        for by in range(blocks_y):
            least_idx = np.argmin(importance[by, :blocks_x])
            orig_pos = origins[by, least_idx]
            removal_mask[orig_pos[0], orig_pos[1]] = True
            
            # Shift blocks left
            blocked[by, :, least_idx:blocks_x-1] = blocked[by, :, least_idx+1:blocks_x].copy()
            importance[by, least_idx:blocks_x-1] = importance[by, least_idx+1:blocks_x]
            origins[by, least_idx:blocks_x-1] = origins[by, least_idx+1:blocks_x]
            removed += 1
        
        blocks_x -= 1
        importance = importance[:, :blocks_x]
        if removed >= target_removals:
            break
        
        # Remove least important block from each column
        for bx in range(blocks_x):
            least_idx = np.argmin(importance[:blocks_y, bx])
            orig_pos = origins[least_idx, bx]
            removal_mask[orig_pos[0], orig_pos[1]] = True
            
            # Shift blocks up
            blocked[least_idx:blocks_y-1, :, bx] = blocked[least_idx+1:blocks_y, :, bx].copy()
            importance[least_idx:blocks_y-1, bx] = importance[least_idx+1:blocks_y, bx]
            origins[least_idx:blocks_y-1, bx] = origins[least_idx+1:blocks_y, bx]
            removed += 1
        
        blocks_y -= 1
        importance = importance[:blocks_y, :]
    
    shrunken = blocked[:blocks_y, :, :blocks_x].reshape(blocks_y * block_size, blocks_x * block_size, 3)
    return shrunken, removal_mask


# =============================================================================
# PRESLEY: ROI-Based Encoding
# Use kvazaar's per-CTU delta QP to allocate more bits to important regions.
# This is the only HEVC encoder we found that accepts external importance maps.
# =============================================================================

def create_roi_file(importance_scores: List[np.ndarray], roi_path: str, qp_range: int = 15) -> None:
    """Create a kvazaar ROI file from importance scores."""

    with open(roi_path, 'wb') as f:
        for importance in importance_scores:
            h, w = importance.shape
            f.write(np.array([w, h], dtype=np.int32).tobytes())
            delta_qp = ((1.0 - importance) * 2 * qp_range - qp_range).astype(np.int8)
            f.write(delta_qp.tobytes())


# =============================================================================
# Main Pipeline
# =============================================================================

if __name__ == "__main__":
    # Setup experiment
    experiment_name = f"{Path(config.reference_video).stem}_w{config.width}_h{config.height}_qp{config.base_qp}"
    print(f"Experiment: {experiment_name}")
    
    experiment_folder = Path("presley/experiments") / experiment_name
    experiment_folder.mkdir(parents=True, exist_ok=True)
    
    config.framerate = get_video_framerate(config.reference_video)
    reference_frames = load_frames(config.reference_video, config.width, config.height)
    
    # Baseline encoding
    baseline_path = experiment_folder / "baseline.mp4"
    encode_video(reference_frames, baseline_path, qp=config.base_qp)
    print(f"Baseline encoded to {baseline_path}")
    
    # EVCA: spatial and temporal complexity analysis
    evca = analyze_frames(np.array(reference_frames), EVCAConfig(block_size=config.block_size))
    
    # UFO: foreground segmentation
    ufo_masks = segment_frames(np.array(reference_frames), device="cuda:0" if torch.cuda.is_available() else "cpu")
    ufo_masks = np.array([cv2.resize(m.astype(np.float32), (config.width // config.block_size, config.height // config.block_size), interpolation=cv2.INTER_NEAREST) for m in ufo_masks]) # Resize masks to block level
    
    # Combine complexity and foreground into importance scores
    importance_scores = calculate_importance_scores(reference_frames, config.block_size, config.alpha, config.beta, evca, ufo_masks)
    print(f"Computed importance scores for {len(importance_scores)} frames")
    
    # Save visualizations
    save_frames([(s * 255).astype(np.uint8) for s in importance_scores], experiment_folder / "importance")
    save_frames([(m * 255).astype(np.uint8) for m in ufo_masks], experiment_folder / "foreground_masks")
    
    # ROI-based encoding using importance scores
    roi_path = experiment_folder / "roi_encoded.mp4"
    encode_video(reference_frames, roi_path, qp=config.base_qp, importance_scores=importance_scores, qp_range=15)
    print(f"ROI-encoded to {roi_path}")
    
    # Frame shrinking
    shrunken_frames, removal_masks = [], []
    for i, frame in enumerate(reference_frames):
        shrunken, mask = shrink_frame(frame, importance_scores[i], config.block_size, config.shrink_amount)
        shrunken_frames.append(shrunken)
        removal_masks.append(mask)
    shrunken_path = experiment_folder / "shrunken.mp4"
    encode_video(shrunken_frames, shrunken_path, qp=config.base_qp)
    print(f"Shrunken video ({shrunken_frames[0].shape[1]}x{shrunken_frames[0].shape[0]}) encoded to {shrunken_path}")
    
    # Save removal masks
    save_frames([(m * 255).astype(np.uint8) for m in removal_masks], experiment_folder / "removal_masks")
    np.savez_compressed(experiment_folder / "removal_masks.npz", masks=np.array(removal_masks))
    
    # Quality evaluation
    print("\n=== Quality Metrics ===")
    
    # Load encoded videos
    baseline_frames = load_frames(str(baseline_path), config.width, config.height)
    roi_frames = load_frames(str(roi_path), config.width, config.height)
    
    # Calculate and save block SSIM
    ssim_baseline = calculate_block_ssim(reference_frames, baseline_frames, config.block_size)
    ssim_roi = calculate_block_ssim(reference_frames, roi_frames, config.block_size)
    save_frames([(s * 255).astype(np.uint8) for s in ssim_baseline], experiment_folder / "metrics/ssim_baseline")
    save_frames([(s * 255).astype(np.uint8) for s in ssim_roi], experiment_folder / "metrics/ssim_roi")
    
    # Overall SSIM
    avg_ssim_baseline = np.mean([s.mean() for s in ssim_baseline])
    avg_ssim_roi = np.mean([s.mean() for s in ssim_roi])
    print(f"Overall SSIM \n Baseline: {avg_ssim_baseline:.4f}, \n ROI: {avg_ssim_roi:.4f}")
    
    # Foreground-only SSIM
    fg_ssim_baseline, fg_ssim_roi = [], []
    for ssim_b, ssim_r, imp in zip(ssim_baseline, ssim_roi, importance_scores):
        fg_mask = imp > 0.5
        if fg_mask.any():
            fg_ssim_baseline.append(ssim_b[fg_mask].mean())
            fg_ssim_roi.append(ssim_r[fg_mask].mean())
    avg_fg_baseline = np.mean(fg_ssim_baseline) if fg_ssim_baseline else 0.0
    avg_fg_roi = np.mean(fg_ssim_roi) if fg_ssim_roi else 0.0
    improvement = (avg_fg_roi - avg_fg_baseline) * 100
    print(f"Foreground SSIM \n Baseline: {avg_fg_baseline:.4f}, \n ROI: {avg_fg_roi:.4f}")
    print(f"Foreground SSIM improvement: {improvement:.2f}%")
    
    # File sizes
    baseline_size = os.path.getsize(baseline_path)
    roi_size = os.path.getsize(roi_path)
    print(f"\nFile sizes \n Baseline: {baseline_size:,} bytes, \n ROI: {roi_size:,} bytes")
