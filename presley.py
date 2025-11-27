from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union
import subprocess
import math
import tempfile
import numpy as np
import cv2
import torch

from evca import analyze_frames, EVCAConfig
from ufo import segment_frames

@dataclass
class ElvisConfig:
    reference_video: str = "Datasets/DAVIS/avc_encoded/bear.mp4"
    width: int = 1280
    height: int = 720
    framerate: float = None
    quality_factor: float = 0.02
    crf: Optional[int] = 30  # If set, use CRF mode instead of ABR
    vbv_bufsize: int = 50  # VBV buffer size in kbits
    encode_preset: str = "medium"
    block_size: int = 16
    gop_size: int = 24  # GOP size for QP-based encoding (frames per ROI map)
    alpha: float = 0.5
    beta: float = 0.5
    shrink_amount: float = 0.25
    propainter_resize_ratio: float = 1.0
    propainter_ref_stride: int = 20
    propainter_neighbor_length: int = 4
    propainter_subvideo_length: int = 40
    propainter_mask_dilation: int = 4
    propainter_raft_iter: int = 20
    propainter_fp16: bool = True
    propainter_devices: Optional[List[Union[int, str]]] = None
    propainter_parallel_chunk_length: Optional[int] = None
    propainter_chunk_overlap: Optional[int] = None
    e2fgvi_ref_stride: int = 10
    e2fgvi_neighbor_stride: int = 5
    e2fgvi_num_ref: int = -1
    e2fgvi_mask_dilation: int = 4
    e2fgvi_devices: Optional[List[Union[int, str]]] = None
    e2fgvi_parallel_chunk_length: Optional[int] = None
    e2fgvi_chunk_overlap: Optional[int] = None
    realesrgan_denoise_strength: float = 1.0
    realesrgan_tile: int = 0
    realesrgan_tile_pad: int = 10
    realesrgan_pre_pad: int = 0
    realesrgan_fp32: bool = False
    realesrgan_devices: Optional[List[Union[int, str]]] = None
    realesrgan_parallel_chunk_length: Optional[int] = None
    realesrgan_per_device_workers: int = 1
    instantir_cfg: float = 7.0
    instantir_creative_start: float = 1.0
    instantir_preview_start: float = 0.0
    instantir_seed: Optional[int] = 42
    instantir_devices: Optional[List[Union[int, str]]] = None
    instantir_batch_size: int = 4
    instantir_parallel_chunk_length: Optional[int] = None
    generate_opencv_benchmarks: bool = True
    metric_stride: int = 1
    fvmd_stride: int = 1
    fvmd_max_frames: Optional[int] = None
    fvmd_processes: Optional[int] = None
    fvmd_early_stop_delta: float = 0.002
    fvmd_early_stop_window: int = 50
    vmaf_stride: int = 1
    enable_fvmd: bool = True

global config
config = ElvisConfig()


def load_frames(video_path: str, width: int, height: int) -> List[np.ndarray]:
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
    output_folder.mkdir(parents=True, exist_ok=True)
    for i, frame in enumerate(frames):
        frame_path = output_folder / f"frame_{i:05d}.png"
        cv2.imwrite(str(frame_path), frame)


def get_video_framerate(video_path: str) -> float:
    cap = cv2.VideoCapture(video_path)
    framerate = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return framerate


def frames_to_blocks(frames: np.ndarray, block_size: int) -> np.ndarray:
    """Convert frames (N, H, W, C) to blocks (N, blocks_y, block_size, blocks_x, block_size, C)."""
    if frames.ndim == 3:  # Single frame (H, W, C)
        frames = frames[np.newaxis, ...]
    n, h, w, c = frames.shape
    blocks_y, blocks_x = h // block_size, w // block_size
    cropped = frames[:, :blocks_y * block_size, :blocks_x * block_size, :]
    return cropped.reshape(n, blocks_y, block_size, blocks_x, block_size, c)


def blocks_to_frames(blocks: np.ndarray) -> np.ndarray:
    """Convert blocks (N, blocks_y, block_size, blocks_x, block_size, C) back to frames (N, H, W, C)."""
    n, blocks_y, block_size_y, blocks_x, block_size_x, c = blocks.shape
    return blocks.reshape(n, blocks_y * block_size_y, blocks_x * block_size_x, c)


def encode_video(frames: List[np.ndarray], output_path: str, crf: Optional[int] = None, target_bitrate: Optional[int] = None, x265_params: Optional[str] = None, vf: Optional[str] = None) -> None:
    height, width = frames[0].shape[:2]
    # Detect input pixel format from frame shape
    input_pix_fmt = "gray" if frames[0].ndim == 2 else "rgb24"
    output_pix_fmt = "gray" if frames[0].ndim == 2 else "yuv444p"
    x265_params = [x265_params] if x265_params else []
    x265_params.append("log-level=1")
    
    # Determine rate control mode
    if crf:
        x265_params.append(f"crf={crf}:vbv-bufsize={config.vbv_bufsize}:vbv-maxrate={config.vbv_bufsize}")
        rate_args = []
    elif target_bitrate:
        x265_params.append(f"vbv-bufsize={config.vbv_bufsize}:vbv-maxrate={target_bitrate}")
        rate_args = ["-b:v", f"{target_bitrate}k"]
    else:
        x265_params.append("lossless=1")
        rate_args = []
    
    x265_params = ":".join(x265_params)
    
    # Video filter arguments
    vf_args = ["-vf", vf] if vf else []
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "warning", "-y",
        "-f", "rawvideo",
        "-pix_fmt", input_pix_fmt,
        "-s:v", f"{width}x{height}",
        "-r", str(config.framerate),
        "-i", "-",
        *vf_args,
        "-c:v", "libx265",
        *rate_args,
        "-x265-params", x265_params,
        "-preset", config.encode_preset,
        "-pix_fmt", output_pix_fmt,
        str(output_path)
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    for frame in frames:
        process.stdin.write(frame.tobytes())
    process.stdin.close()
    process.wait()


def calculate_importance_scores(frames: List[np.ndarray], block_size: int, alpha: float, beta: float, complexities: np.ndarray, foreground_masks: np.ndarray) -> List[np.ndarray]:
    frames = np.array(frames)

    # Combine spatial and temporal complexity scores
    complexity_scores = np.zeros_like(complexities.SC)
    complexity_scores[:-1] = alpha * complexities.SC[:-1] + (1 - alpha) * complexities.TC[1:]
    # For the last frame, there is no successive temporal complexity, so we rely only on spatial
    complexity_scores[-1] = complexities.SC[-1]

    # Smooth importance scores with prior frame's score to balance sudden changes
    importance_scores = np.zeros_like(complexity_scores)
    # The first frame has no prior frame, so its scores remain unchanged.
    importance_scores[0] = complexity_scores[0]
    importance_scores[1:] = (beta * complexity_scores[1:] + (1 - beta) * complexity_scores[:-1])

    # Multiply importance scores of background regions by -1 so that background is less important than foreground, and its complex regions have lowest scores
    foreground_masks[foreground_masks < 0.5] = -1.0
    importance_scores *= foreground_masks
    # Normalize importance scores to [0, 1]
    min_score = importance_scores.min(axis=(1, 2), keepdims=True)
    max_score = importance_scores.max(axis=(1, 2), keepdims=True)
    importance_scores = (importance_scores - min_score) / (max_score - min_score + 1e-8)

    return [importance_scores[i] for i in range(importance_scores.shape[0])]


def shrink_frame(frame: np.ndarray, importance_score: np.ndarray, block_size: int, shrink_amount: float) -> tuple:
    # Make writable copies
    frame = frame.copy()
    importance_score = importance_score.copy()
    
    height, width, _ = frame.shape
    orig_blocks_y = height // block_size
    orig_blocks_x = width // block_size
    blocks_y = orig_blocks_y
    blocks_x = orig_blocks_x
    
    # Reshape frame into blocks: (blocks_y, block_size, blocks_x, block_size, 3)
    blocked_frame = frame[:blocks_y * block_size, :blocks_x * block_size].reshape(blocks_y, block_size, blocks_x, block_size, 3)
    
    # Track original positions of blocks: (orig_y, orig_x) for each current position
    block_origins = np.empty((orig_blocks_y, orig_blocks_x, 2), dtype=np.int32)
    for by in range(orig_blocks_y):
        for bx in range(orig_blocks_x):
            block_origins[by, bx] = [by, bx]
    
    # Track which original blocks have been removed
    removal_mask = np.zeros((orig_blocks_y, orig_blocks_x), dtype=bool)
    
    target_blocks_to_remove = int(orig_blocks_y * orig_blocks_x * shrink_amount)
    blocks_removed = 0
    while blocks_removed < target_blocks_to_remove:
        # For each row, remove the least important block by shifting all blocks after it
        for by in range(blocks_y):
            row_importance = importance_score[by, :blocks_x]
            least_important_idx = np.argmin(row_importance)
            # Record the original position of the removed block
            orig_pos = block_origins[by, least_important_idx]
            removal_mask[orig_pos[0], orig_pos[1]] = True
            # Shift all blocks after the least important one to the left
            blocked_frame[by, :, least_important_idx:blocks_x-1, :, :] = blocked_frame[by, :, least_important_idx+1:blocks_x, :, :].copy()
            importance_score[by, least_important_idx:blocks_x-1] = importance_score[by, least_important_idx+1:blocks_x]
            block_origins[by, least_important_idx:blocks_x-1] = block_origins[by, least_important_idx+1:blocks_x]
            blocks_removed += 1
        # Remove the last column of blocks
        blocks_x -= 1
        importance_score = importance_score[:, :blocks_x]
        if blocks_removed >= target_blocks_to_remove:
            break
            
        # For each column, remove the least important block by shifting all blocks after it
        for bx in range(blocks_x):
            col_importance = importance_score[:blocks_y, bx]
            least_important_idx = np.argmin(col_importance)
            # Record the original position of the removed block
            orig_pos = block_origins[least_important_idx, bx]
            removal_mask[orig_pos[0], orig_pos[1]] = True
            # Shift all blocks after the least important one up
            blocked_frame[least_important_idx:blocks_y-1, :, bx, :, :] = blocked_frame[least_important_idx+1:blocks_y, :, bx, :, :].copy()
            importance_score[least_important_idx:blocks_y-1, bx] = importance_score[least_important_idx+1:blocks_y, bx]
            block_origins[least_important_idx:blocks_y-1, bx] = block_origins[least_important_idx+1:blocks_y, bx]
            blocks_removed += 1
        # Remove the last row of blocks
        blocks_y -= 1
        importance_score = importance_score[:blocks_y, :]
        
    # Reconstruct the shrunken frame
    shrunken_frame = blocked_frame[:blocks_y, :, :blocks_x, :, :].reshape(
        blocks_y * block_size, blocks_x * block_size, 3
    )
    return shrunken_frame, removal_mask


def build_roi_filter(importance_map: np.ndarray, width: int, height: int, threshold: float = 0.2) -> Optional[str]:
    """Build addroi filter string from a 2D importance map.
    
    Args:
        importance_map: 2D array (blocks_y, blocks_x) with values in [0, 1]
        width: Frame width in pixels
        height: Frame height in pixels
        threshold: Only add ROI for blocks with |qoffset| > threshold
    
    Returns:
        Filter string or None if no significant ROIs
    """
    score_h, score_w = importance_map.shape
    roi_block_h = height // score_h
    roi_block_w = width // score_w
    
    # Map importance [0,1] to QP offset [-0.5,0.5]: high importance = negative offset = better quality
    qp_offsets = 0.5 - importance_map
    
    roi_filters = []
    for by in range(score_h):
        for bx in range(score_w):
            offset = float(qp_offsets[by, bx])
            if abs(offset) > threshold:
                offset = max(-1.0, min(1.0, offset))
                x = bx * roi_block_w
                y = by * roi_block_h
                roi_filters.append(f"addroi=x={x}:y={y}:w={roi_block_w}:h={roi_block_h}:qoffset={offset:.4f}")
    
    return ",".join(roi_filters) if roi_filters else None


def encode_video_with_qp(frames: List[np.ndarray], output_path: str, importance_scores: List[np.ndarray], crf: Optional[int] = None, target_bitrate: Optional[int] = None, gop_size: Optional[int] = None) -> None:
    """Encode video with per-GOP, per-block QP offsets using addroi filters.
    
    Splits the video into GOPs of gop_size frames, computes average importance
    scores within each GOP, and encodes each GOP with its own ROI map. This
    provides a tradeoff between:
    - GOP=1: Per-frame ROI but loses inter-frame compression
    - GOP=all: Single ROI map, maximum compression but loses temporal adaptivity
    
    Args:
        frames: List of frames to encode
        output_path: Output video path
        importance_scores: List of per-frame importance score maps
        crf: Constant Rate Factor for quality control
        target_bitrate: Target bitrate in kbps (alternative to CRF)
        gop_size: Number of frames per GOP (default from config.gop_size)
    """
    if gop_size is None:
        gop_size = config.gop_size
    
    height, width = frames[0].shape[:2]
    scores_array = np.stack(importance_scores, axis=0)
    num_frames = len(frames)
    
    # If gop_size >= num_frames, use single GOP (original behavior)
    if gop_size >= num_frames:
        avg_importance = scores_array.mean(axis=0)
        vf = build_roi_filter(avg_importance, width, height)
        if vf:
            print(f"  Using single GOP with ROI-based encoding")
        else:
            print(f"  No significant QP variations, using standard encoding")
        encode_video(frames, output_path, crf, target_bitrate, vf=vf)
        return
    
    # Split into GOPs and encode each separately
    num_gops = math.ceil(num_frames / gop_size)
    print(f"  Encoding {num_frames} frames in {num_gops} GOPs of up to {gop_size} frames each")
    
    output_dir = Path(output_path).parent
    gop_files = []
    
    with tempfile.TemporaryDirectory(dir=output_dir) as temp_dir:
        temp_path = Path(temp_dir).resolve()  # Use absolute path
        
        for gop_idx in range(num_gops):
            start_frame = gop_idx * gop_size
            end_frame = min(start_frame + gop_size, num_frames)
            gop_frames = frames[start_frame:end_frame]
            gop_scores = scores_array[start_frame:end_frame]
            
            # Average importance within this GOP
            gop_avg_importance = gop_scores.mean(axis=0)
            vf = build_roi_filter(gop_avg_importance, width, height)
            
            # Encode this GOP
            gop_file = temp_path / f"gop_{gop_idx:04d}.mp4"
            gop_files.append(gop_file)
            
            # Force keyframe at start of each GOP segment
            x265_params = f"keyint={gop_size}:min-keyint={gop_size}"
            encode_video(gop_frames, str(gop_file), crf, target_bitrate, x265_params=x265_params, vf=vf)
        
        # Concatenate all GOP files
        concat_list = temp_path / "concat.txt"
        with open(concat_list, 'w') as f:
            for gop_file in gop_files:
                # Use absolute path in concat file
                f.write(f"file '{gop_file.resolve()}'\n")
        
        # Use ffmpeg concat demuxer to join segments
        concat_cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "warning", "-y",
            "-f", "concat", "-safe", "0",
            "-i", str(concat_list),
            "-c", "copy",
            str(output_path)
        ]
        subprocess.run(concat_cmd, check=True)
    
    print(f"  Encoded {num_gops} GOPs with per-GOP ROI maps")


# Setup
experiment_name = f"elvis_{Path(config.reference_video).stem}_w{config.width}_h{config.height}_q{config.quality_factor}"
print(f"Experiment Name: {experiment_name}")
experiment_folder = Path("elvis/experiments") / experiment_name
experiment_folder.mkdir(parents=True, exist_ok=True)
config.framerate = get_video_framerate(config.reference_video)
target_bitrate = (config.width * config.height * config.framerate * config.quality_factor) / 1000  # in kbps
reference_frames = load_frames(config.reference_video, config.width, config.height)

# Reference encoding
encode_video(reference_frames, experiment_folder / "reference.mp4")
print(f"Reference video encoded losslessly to {experiment_folder / 'reference.mp4'}.")

# Baseline encoding
encode_video(reference_frames, experiment_folder / "baseline.mp4", config.crf)
print(f"Baseline video encoded to {experiment_folder / 'baseline.mp4'}.")

# Complexity score calculation via EVCA
evca = analyze_frames(np.array(reference_frames), EVCAConfig(block_size=config.block_size))
evca_SC_folder = experiment_folder / "evca_SC"
save_frames(evca.SC, evca_SC_folder)
evca_TC_folder = experiment_folder / "evca_TC"
save_frames(evca.TC, evca_TC_folder)
ufo_masks = segment_frames(np.array(reference_frames), device="cuda:0" if torch.cuda.is_available() else "cpu")

# UFO masks are 1 for foreground pixels and 0 for background pixels, resize to block level
block_height = reference_frames[0].shape[0] // config.block_size
block_width = reference_frames[0].shape[1] // config.block_size
ufo_masks = np.array([cv2.resize(mask.astype(np.float32), (block_width, block_height), interpolation=cv2.INTER_NEAREST) for mask in ufo_masks])
ufo_masks_folder = experiment_folder / "ufo_masks"
save_frames([ (mask * 255).astype(np.uint8) for mask in ufo_masks ], ufo_masks_folder)

# Importance score calculation
importance_scores = calculate_importance_scores(reference_frames, config.block_size, config.alpha, config.beta, evca, ufo_masks)
print(f"Calculated importance scores for {len(importance_scores)} frames.")
importance_scores_folder = experiment_folder / "importance_scores"
save_frames([ (score * 255).astype(np.uint8) for score in importance_scores ], importance_scores_folder)

# QP-based encoding using importance scores
qp_video_path = experiment_folder / "qp_encoded.mp4"
encode_video_with_qp(reference_frames, qp_video_path, importance_scores, config.crf)
print(f"QP-encoded video encoded to {qp_video_path}.")

# Frame shrinking based on importance scores
shrunken_frames = []
removal_masks = []
for i, frame in enumerate(reference_frames):
    shrunken_frame, removal_mask = shrink_frame(frame, importance_scores[i], config.block_size, config.shrink_amount)
    shrunken_frames.append(shrunken_frame)
    removal_masks.append(removal_mask)
encode_video(shrunken_frames, experiment_folder / "shrunken_video.mp4", config.crf)
print(f"Shrunken video encoded to {experiment_folder / 'shrunken_video.mp4'} with resolution {shrunken_frame.shape[1]}x{shrunken_frame.shape[0]}.")

# Removal masks encoding
removal_masks_folder = experiment_folder / "removal_masks"
save_frames([ (mask.astype(np.uint8) * 255) for mask in removal_masks ], removal_masks_folder)
print(f"Removal masks saved to {removal_masks_folder}.")
# Encode as video (masks are 2D arrays, encode_video will auto-detect gray format)
encode_video([ (mask.astype(np.uint8) * 255) for mask in removal_masks ], experiment_folder / "removal_masks.mp4", config.crf)
# Encode as npz
np.savez_compressed(experiment_folder / "removal_masks.npz", removal_masks=np.array(removal_masks))
print(f"Removal masks video and npz saved to {experiment_folder / 'removal_masks.mp4'} and {experiment_folder / 'removal_masks.npz'} for ablation tests.")

