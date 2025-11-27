import argparse
import builtins
import contextlib
import functools
import gc
import glob
import json
import math
import multiprocessing
import os
os.environ.setdefault("LOGURU_LEVEL", "WARNING")
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid
import warnings
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, NamedTuple, Optional, Sequence, Tuple, Union

import cv2
import lpips
import numpy as np
import torch
from PIL import Image
from skimage.metrics import structural_similarity as ssim

from fvmd.datasets.video_datasets import VideoDataset
from fvmd.keypoint_tracking import track_keypoints
from fvmd.extract_motion_features import calc_hist
from fvmd.frechet_distance import calculate_fd_given_vectors
from instantir import InstantIRRuntime, load_runtime, restore_images_batch

try:
    from diffusers.utils import logging as _diffusers_logging
    _diffusers_logging.disable_progress_bar()
    _diffusers_logging.set_verbosity_error()
except ImportError:
    _diffusers_logging = None

@dataclass
class ElvisConfig:
    reference_video: str = "davis_test/bear.mp4"
    width: int = 640
    height: int = 360
    block_size: int = 8
    shrink_amount: float = 0.25
    quality_factor: float = 1.2
    target_bitrate_override: Optional[int] = None
    removability_alpha: float = 0.5
    removability_smoothing_beta: float = 0.5
    encode_preset: str = "medium"
    encode_pix_fmt: str = "yuv420p"
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
    

# ---------------------------------------------------------------------------
# Pipeline and reporting label constants
# ---------------------------------------------------------------------------
APPROACH_BASELINE = "Baseline"
APPROACH_PRESLEY_QP = "PRESLEY QP"
APPROACH_ELVIS = "ELVIS"
APPROACH_ELVIS_CV2 = f"ELVIS CV2"
APPROACH_ELVIS_PROP = f"ELVIS ProPainter"
APPROACH_ELVIS_E2FGVI = f"ELVIS E2FGVI"
APPROACH_PRESLEY_REALESRGAN = "PRESLEY RealESRGAN"
APPROACH_PRESLEY_INSTANTIR = "PRESLEY InstantIR"
APPROACH_PRESLEY_LANCZOS = "PRESLEY Lanczos"
APPROACH_PRESLEY_UNSHARP = "PRESLEY Unsharp"


# ---------------------------------------------------------------------------
# Layer 1: IO Utilities - Unified frame/video/mask IO
# ---------------------------------------------------------------------------

def load_frame(path: str) -> np.ndarray:
    """Load a single frame from disk as BGR numpy array."""
    frame = cv2.imread(path, cv2.IMREAD_COLOR)
    if frame is None:
        raise IOError(f"Failed to load frame: {path}")
    return frame


def save_frame(frame: np.ndarray, path: str) -> None:
    """Save a single frame to disk."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if not cv2.imwrite(path, frame):
        raise IOError(f"Failed to save frame: {path}")


def load_frames(directory: str, pattern: str = "*.png") -> List[np.ndarray]:
    """Load all frames from a directory matching the pattern, sorted by name."""
    dir_path = Path(directory)
    if not dir_path.is_dir():
        raise ValueError(f"Directory does not exist: {directory}")
    
    paths = sorted(dir_path.glob(pattern))
    if not paths:
        # Try common image extensions
        for ext in ("*.png", "*.jpg", "*.jpeg"):
            paths = sorted(dir_path.glob(ext))
            if paths:
                break
    
    frames = []
    for path in paths:
        frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if frame is not None:
            frames.append(frame)
    return frames


def save_frames(
    frames: Sequence[np.ndarray],
    directory: str,
    pattern: str = "%05d.png",
    start_index: int = 1,
) -> List[str]:
    """Save frames to directory with numbered filenames. Returns list of saved paths."""
    os.makedirs(directory, exist_ok=True)
    saved_paths = []
    for idx, frame in enumerate(frames):
        filename = pattern % (start_index + idx)
        path = os.path.join(directory, filename)
        if not cv2.imwrite(path, frame):
            raise IOError(f"Failed to save frame: {path}")
        saved_paths.append(path)
    return saved_paths


def load_masks(
    directory: str,
    width: int,
    height: int,
    expected_count: int,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Load and resize masks, returning (fg_masks, bg_masks) as boolean arrays."""
    if expected_count <= 0:
        return [], []

    dir_path = Path(directory)
    mask_files = []
    if dir_path.is_dir():
        mask_files = sorted([
            f for f in os.listdir(directory)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])

    fg_masks: List[np.ndarray] = []
    bg_masks: List[np.ndarray] = []
    last_mask: Optional[np.ndarray] = None

    for frame_idx in range(expected_count):
        if frame_idx < len(mask_files):
            mask_path = os.path.join(directory, mask_files[frame_idx])
            mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask_img is not None:
                if mask_img.shape[:2] != (height, width):
                    mask_img = cv2.resize(mask_img, (width, height), interpolation=cv2.INTER_NEAREST)
                last_mask = mask_img
        
        if last_mask is not None:
            fg_mask = last_mask > 127
            bg_mask = ~fg_mask
        else:
            fg_mask = np.ones((height, width), dtype=bool)
            bg_mask = np.zeros((height, width), dtype=bool)
        
        fg_masks.append(fg_mask)
        bg_masks.append(bg_mask)

    return fg_masks, bg_masks


def clear_directory(directory: str, patterns: Sequence[str] = ("*.png", "*.jpg", "*.jpeg")) -> None:
    """Remove files matching patterns from directory."""
    dir_path = Path(directory)
    if not dir_path.is_dir():
        return
    for pattern in patterns:
        for file_path in dir_path.glob(pattern):
            if file_path.is_file():
                file_path.unlink()


def get_frame_paths(directory: str) -> List[Path]:
    """Get sorted list of frame file paths in a directory."""
    dir_path = Path(directory)
    if not dir_path.is_dir():
        return []
    valid_suffixes = (".png", ".jpg", ".jpeg")
    return sorted([p for p in dir_path.iterdir() if p.suffix.lower() in valid_suffixes])


# ---------------------------------------------------------------------------
# Layer 2: Concurrency Utilities - Unified parallelization
# ---------------------------------------------------------------------------

@dataclass
class ChunkSpec:
    """Specification for a processing chunk."""
    start: int
    end: int
    device: torch.device
    chunk_id: int = 0


def chunk_for_devices(
    total: int,
    devices: List[torch.device],
    min_chunk_size: int = 1,
) -> List[ChunkSpec]:
    """Split total items into chunks, one per device."""
    if not devices or total <= 0:
        return []
    
    num_devices = len(devices)
    base_size = total // num_devices
    remainder = total % num_devices
    
    chunks = []
    start = 0
    for idx, device in enumerate(devices):
        size = base_size + (1 if idx < remainder else 0)
        if size < min_chunk_size and idx > 0:
            # Merge small chunk with previous
            continue
        end = start + size
        if end > start:
            chunks.append(ChunkSpec(start=start, end=end, device=device, chunk_id=idx))
        start = end
    
    return chunks


def parallel_process_frames(
    process_fn: Callable[[List[np.ndarray], torch.device], List[np.ndarray]],
    frames: List[np.ndarray],
    devices: List[torch.device],
    chunk_size: Optional[int] = None,
    max_workers: Optional[int] = None,
) -> List[np.ndarray]:
    """
    Process frames in parallel across devices using ThreadPoolExecutor.
    
    Args:
        process_fn: Function that takes (frames_chunk, device) and returns processed frames
        frames: List of frames to process
        devices: List of devices to use
        chunk_size: Optional fixed chunk size (otherwise auto-calculated)
        max_workers: Maximum number of parallel workers
    
    Returns:
        List of processed frames in original order
    """
    if not frames:
        return []
    
    if not devices:
        devices = [torch.device("cpu")]
    
    num_frames = len(frames)
    
    # Calculate chunks
    if chunk_size is None:
        chunks = chunk_for_devices(num_frames, devices)
    else:
        chunks = []
        cursor = 0
        chunk_id = 0
        while cursor < num_frames:
            end = min(cursor + chunk_size, num_frames)
            device = devices[chunk_id % len(devices)]
            chunks.append(ChunkSpec(start=cursor, end=end, device=device, chunk_id=chunk_id))
            cursor = end
            chunk_id += 1
    
    if not chunks:
        return []
    
    # Single chunk - process directly
    if len(chunks) == 1:
        chunk = chunks[0]
        return process_fn(frames[chunk.start:chunk.end], chunk.device)
    
    # Multiple chunks - parallel processing
    results: Dict[int, List[np.ndarray]] = {}
    workers = max_workers or min(len(chunks), len(devices))
    
    def _process_chunk(chunk: ChunkSpec) -> Tuple[int, List[np.ndarray]]:
        chunk_frames = frames[chunk.start:chunk.end]
        processed = process_fn(chunk_frames, chunk.device)
        return chunk.chunk_id, processed
    
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_process_chunk, chunk): chunk for chunk in chunks}
        for future in as_completed(futures):
            chunk_id, processed_frames = future.result()
            results[chunk_id] = processed_frames
    
    # Reassemble in order
    output = []
    for chunk in sorted(chunks, key=lambda c: c.chunk_id):
        output.extend(results[chunk.chunk_id])
    
    return output


class _NullStream:
    """Lightweight write-only stream that safely discards all data."""

    def write(self, text: str) -> int:  # type: ignore[override]
        return len(text)

    def flush(self) -> None:
        pass

    def writelines(self, lines: Sequence[str]) -> None:  # pragma: no cover - trivial
        for line in lines:
            self.write(line)

    def close(self) -> None:  # pragma: no cover - noop
        pass

    @property
    def closed(self) -> bool:  # pragma: no cover - constant property
        return False

    def isatty(self) -> bool:  # pragma: no cover - interface compatibility
        return False


@contextlib.contextmanager
def _silence_console_output() -> Iterator[None]:
    """Redirect stdout/stderr to a resilient null stream for noisy calls."""

    null_stream = _NullStream()
    original_stdout = sys.stdout
    original_stderr = sys.stderr

    try:
        sys.stdout = null_stream  # type: ignore[assignment]
        sys.stderr = null_stream  # type: ignore[assignment]
        yield
    finally:
        sys.stdout = original_stdout  # type: ignore[assignment]
        sys.stderr = original_stderr  # type: ignore[assignment]


def _safe_print(*args: Any, **kwargs: Any) -> None:
    """Print helper resilient to closed stdout/stderr streams."""

    target_stream = kwargs.get("file", sys.stdout)
    try:
        builtins.print(*args, **kwargs)
    except (ValueError, OSError):
        fallback = getattr(sys, "__stdout__", None)
        if fallback is None or fallback is target_stream:
            return
        kwargs["file"] = fallback
        try:
            builtins.print(*args, **kwargs)
        except (ValueError, OSError):
            pass

_LPIPS_MODEL_CACHE: Dict[str, lpips.LPIPS] = {}


def _configure_fvmd_logging() -> None:
    """Restrict FVMD's internal logger to warnings/errors to reduce noise."""

    try:
        from loguru import logger as _loguru_logger
    except ImportError:
        return

    try:
        _loguru_logger.disable("fvmd")
    except Exception:
        try:
            _loguru_logger.remove()
            _loguru_logger.add(sys.stderr, level="WARNING")
        except Exception:
            pass


_configure_fvmd_logging()


def _get_lpips_model(device: str = 'cuda' if torch.cuda.is_available() else 'cpu') -> lpips.LPIPS:
    """Lazy-load and cache LPIPS models per device."""

    normalized_device = device
    if normalized_device.startswith('cuda') and not torch.cuda.is_available():
        normalized_device = 'cpu'

    cached = _LPIPS_MODEL_CACHE.get(normalized_device)
    if cached is None:
        cached = lpips.LPIPS(net='alex').to(normalized_device)
        _LPIPS_MODEL_CACHE[normalized_device] = cached
    return cached


def _resolve_device_list(
    devices: Optional[Sequence[Union[int, str, torch.device]]],
    *,
    prefer_cuda: bool = True,
    allow_cpu_fallback: bool = True,
) -> List[torch.device]:
    """Normalize user-provided device specifiers into unique torch.device entries."""

    available_gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0

    def _normalize_device(spec: Union[int, str, torch.device]) -> torch.device:
        if isinstance(spec, torch.device):
            device_obj = spec
        elif isinstance(spec, int):
            if not torch.cuda.is_available():
                raise ValueError(
                    "CUDA device indices were provided but no CUDA devices are available."
                )
            if spec < 0 or spec >= available_gpu_count:
                raise ValueError(f"Requested CUDA device index {spec} is out of range.")
            device_obj = torch.device(f"cuda:{spec}")
        else:
            spec_str = str(spec)
            if spec_str.startswith("cuda"):
                if not torch.cuda.is_available():
                    raise ValueError("CUDA devices were requested but CUDA is not available.")
                if spec_str == "cuda" or spec_str == "cuda:":
                    device_obj = torch.device("cuda")
                else:
                    try:
                        idx_part = spec_str.split(":", 1)[1]
                        idx_val = int(idx_part)
                    except (IndexError, ValueError):
                        raise ValueError(f"Invalid CUDA device string '{spec_str}'.") from None
                    if idx_val < 0 or idx_val >= available_gpu_count:
                        raise ValueError(
                            f"Requested CUDA device {spec_str} exceeds detected count {available_gpu_count}."
                        )
                    device_obj = torch.device(f"cuda:{idx_val}")
            else:
                device_obj = torch.device(spec_str)

        if device_obj.type == "cuda":
            idx = device_obj.index if device_obj.index is not None else 0
            if idx < 0 or idx >= available_gpu_count:
                raise ValueError(
                    f"Requested CUDA device {idx} is not available. Detected {available_gpu_count} device(s)."
                )
        return device_obj

    if not devices:
        if prefer_cuda and available_gpu_count > 0:
            device_specs: Sequence[Union[int, str, torch.device]] = [f"cuda:{idx}" for idx in range(available_gpu_count)]
        elif allow_cpu_fallback:
            device_specs = ["cpu"]
        else:
            raise ValueError("No CUDA devices available and CPU fallback disabled.")
    else:
        device_specs = devices

    resolved_devices: List[torch.device] = []
    seen_keys = set()
    for spec in device_specs:
        device_obj = _normalize_device(spec)
        key = str(device_obj)
        if device_obj.type == "cuda":
            idx = device_obj.index if device_obj.index is not None else 0
            key = f"cuda:{idx}"
        if key in seen_keys:
            continue
        seen_keys.add(key)
        resolved_devices.append(device_obj)

    if not resolved_devices:
        if allow_cpu_fallback:
            resolved_devices.append(torch.device("cpu"))
        else:
            raise ValueError("No valid compute devices resolved from the provided specification.")

    return resolved_devices

# Helper function to get package installation directory



def _load_resized_masks(
    masks_dir: str,
    width: int,
    height: int,
    expected_frames: int
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Load, resize, and cache foreground/background masks as boolean arrays."""

    if expected_frames <= 0:
        return [], []

    mask_files = sorted([
        f for f in os.listdir(masks_dir)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ]) if os.path.isdir(masks_dir) else []

    fg_masks: List[np.ndarray] = []
    bg_masks: List[np.ndarray] = []

    last_mask: Optional[np.ndarray] = None

    for frame_idx in range(expected_frames):
        if frame_idx < len(mask_files):
            mask_path = os.path.join(masks_dir, mask_files[frame_idx])
            mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask_img is None:
                mask_bool = np.zeros((height, width), dtype=bool)
            else:
                mask_resized = cv2.resize(mask_img, (width, height), interpolation=cv2.INTER_NEAREST)
                mask_bool = mask_resized > 128
        elif last_mask is not None:
            mask_bool = last_mask
        else:
            mask_bool = np.zeros((height, width), dtype=bool)

        fg_masks.append(mask_bool)
        bg_masks.append(~mask_bool)
        last_mask = mask_bool

    return fg_masks, bg_masks


def _compute_mask_union_bbox(
    masks: Sequence[np.ndarray],
    width: int,
    height: int,
    padding_ratio: float = 0.05
) -> Tuple[int, int, int, int]:
    """Compute a padded bounding box over the union of provided masks."""

    if not masks:
        return (0, 0, width, height)

    union_mask = np.zeros((height, width), dtype=bool)
    for mask in masks:
        if mask is not None:
            union_mask |= mask

    if not np.any(union_mask):
        return (0, 0, width, height)

    ys, xs = np.where(union_mask)
    min_y, max_y = int(ys.min()), int(ys.max())
    min_x, max_x = int(xs.min()), int(xs.max())

    bbox_height = max_y - min_y + 1
    bbox_width = max_x - min_x + 1

    pad_y = max(1, int(bbox_height * padding_ratio))
    pad_x = max(1, int(bbox_width * padding_ratio))

    y = max(0, min_y - pad_y)
    x = max(0, min_x - pad_x)
    h = min(height - y, bbox_height + 2 * pad_y)
    w = min(width - x, bbox_width + 2 * pad_x)

    return (x, y, w, h)


def _apply_binary_mask(frame: np.ndarray, mask: np.ndarray, invert: bool = False) -> np.ndarray:
    """Return a copy of frame with pixels outside mask zeroed."""

    if frame is None or mask is None:
        return frame

    mask_bool = mask if not invert else ~mask
    masked_frame = np.zeros_like(frame)
    masked_frame[mask_bool] = frame[mask_bool]
    return masked_frame


def _masked_psnr(ref: np.ndarray, dec: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    """Compute PSNR restricted to masked pixels."""

    if ref is None or dec is None:
        return 0.0

    ref_f = ref.astype(np.float32)
    dec_f = dec.astype(np.float32)

    if mask is not None:
        valid = mask.astype(bool)
        if not np.any(valid):
            return 100.0
        diff = ref_f[valid] - dec_f[valid]
    else:
        diff = ref_f - dec_f

    mse = float(np.mean(diff ** 2)) if diff.size else 0.0
    if mse < 1e-10:
        return 100.0

    max_pixel_value = 255.0
    psnr_val = 20 * math.log10(max_pixel_value / math.sqrt(mse))
    return float(min(psnr_val, 100.0))


def _masked_mse(ref: np.ndarray, dec: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    """Compute MSE (Mean Squared Error) restricted to masked pixels."""

    if ref is None or dec is None:
        return 0.0

    ref_f = ref.astype(np.float32)
    dec_f = dec.astype(np.float32)

    if mask is not None:
        valid = mask.astype(bool)
        if not np.any(valid):
            return 0.0
        diff = ref_f[valid] - dec_f[valid]
    else:
        diff = ref_f - dec_f

    mse = float(np.mean(diff ** 2)) if diff.size else 0.0
    return mse


def _masked_ssim(ref: np.ndarray, dec: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    """Compute SSIM on luminance channel within the mask."""

    if ref is None or dec is None:
        return 0.0

    ref_y = cv2.cvtColor(ref, cv2.COLOR_BGR2YCrCb)[:, :, 0]
    dec_y = cv2.cvtColor(dec, cv2.COLOR_BGR2YCrCb)[:, :, 0]

    if mask is not None:
        mask_bool = mask.astype(bool)
        if not np.any(mask_bool):
            return 1.0

        ys, xs = np.where(mask_bool)
        y1, y2 = ys.min(), ys.max() + 1
        x1, x2 = xs.min(), xs.max() + 1

        ref_y = ref_y[y1:y2, x1:x2].copy()
        dec_y = dec_y[y1:y2, x1:x2].copy()
        mask_crop = mask_bool[y1:y2, x1:x2]

        ref_y[~mask_crop] = 0
        dec_y[~mask_crop] = 0

    # skimage.metrics.structural_similarity uses a default win_size of 7 which
    # fails for small crops. Compute a safe odd win_size not larger than the
    # smallest image side, and at least 3.
    h, w = ref_y.shape[:2]
    smallest_dim = min(h, w)
    if smallest_dim < 3:
        # Too small to compute a meaningful SSIM window; treat as perfect.
        return 1.0

    if smallest_dim < 7:
        win_size = smallest_dim if (smallest_dim % 2 == 1) else max(3, smallest_dim - 1)
    else:
        win_size = 7

    return float(
        ssim(
            ref_y,
            dec_y,
            data_range=255,
            gaussian_weights=True,
            win_size=win_size,
        )
    )


def _decode_video_to_frames(video_path: str, max_frames: Optional[int] = None) -> List[np.ndarray]:
    """Decode a video into a list of BGR frames using OpenCV."""

    frames: List[np.ndarray] = []
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Warning: Unable to open video for decoding: {video_path}")
        return frames

    total = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        total += 1
        if max_frames is not None and total >= max_frames:
            break

    cap.release()
    return frames


def _encode_frames_to_video(
    frames: Sequence[np.ndarray],
    output_path: str,
    framerate: float,
    filter_chain: Optional[str] = None,
    codec: str = 'libx264',
    preset: str = 'ultrafast',
    pix_fmt: str = 'yuv420p',
    extra_codec_args: Optional[Sequence[str]] = None,
) -> None:
    """Encode a sequence of frames to video via FFmpeg piping."""

    if not frames:
        raise ValueError("No frames provided for video encoding.")

    height, width = frames[0].shape[:2]

    # Some pixel formats (e.g., YUV 4:2:0) require even dimensions. When we
    # generate cropped ROI clips it's easy to end up with odd sizes, so we pad a
    # single row/column on the fly to keep the encoder happy without changing
    # the visible content.
    pix_fmt_lower = (pix_fmt or '').lower()
    requires_even_dims = False
    if pix_fmt_lower:
        chroma_tokens = ('420', 'nv12', 'nv21', 'p010', 'p016')
        if any(token in pix_fmt_lower for token in chroma_tokens):
            requires_even_dims = True
    else:
        # FFmpeg defaults to yuv420p when no pixel format is provided.
        requires_even_dims = True

    pad_right = 1 if requires_even_dims and (width % 2 != 0) else 0
    pad_bottom = 1 if requires_even_dims and (height % 2 != 0) else 0

    even_pad_filter: Optional[str] = None
    if requires_even_dims:
        even_pad_filter = 'pad=ceil(iw/2)*2:ceil(ih/2)*2:x=0:y=0:color=black'
        if filter_chain:
            filter_chain = f"{filter_chain},{even_pad_filter}"
        else:
            filter_chain = even_pad_filter

    adjusted_width = width + pad_right
    adjusted_height = height + pad_bottom

    if pad_right or pad_bottom:
        print(
            f"  - Adjusting frame dimensions from {width}x{height} to {adjusted_width}x{adjusted_height} "
            f"for {pix_fmt or 'default'} encoding"
        )

    cmd = [
        'ffmpeg', '-hide_banner', '-loglevel', 'error', '-y',
        '-f', 'rawvideo',
        '-pix_fmt', 'bgr24',
        '-s', f'{adjusted_width}x{adjusted_height}',
        '-r', f'{framerate}',
        '-i', '-'
    ]

    if filter_chain:
        cmd.extend(['-vf', filter_chain])

    cmd.extend(['-c:v', codec, '-preset', preset])

    if codec == 'libx264' and '-crf' not in (extra_codec_args or []):
        cmd.extend(['-crf', '0'])

    if pix_fmt:
        cmd.extend(['-pix_fmt', pix_fmt])

    if extra_codec_args:
        cmd.extend(extra_codec_args)

    cmd.append(output_path)

    process = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    try:
        for frame in frames:
            if frame is None:
                continue
            frame_to_write = frame
            if pad_right or pad_bottom:
                frame_to_write = cv2.copyMakeBorder(
                    frame_to_write,
                    0,
                    pad_bottom,
                    0,
                    pad_right,
                    borderType=cv2.BORDER_REPLICATE,
                )
            frame_bytes = np.ascontiguousarray(frame_to_write.astype(np.uint8)).tobytes()
            process.stdin.write(frame_bytes)
    finally:
        if process.stdin:
            process.stdin.close()
    ret_code = process.wait()
    if ret_code != 0:
        raise RuntimeError(f"FFmpeg failed with exit code {ret_code} while encoding {output_path}")


def _slugify_name(name: str) -> str:
    """Generate filesystem-friendly identifier from a video name."""
    import re
    slug = re.sub(r'[^\w\-]', '_', name.strip())
    return slug.strip('_') or 'video'

# Core functions

def calculate_target_bitrate(width: int, height: int, framerate: float, quality_factor: float = 1.0) -> int:
    """Calculate target bitrate based on video characteristics. Returns bitrate in bps."""
    pixels_per_second = width * height * framerate
    bits_per_pixel = 0.01 * quality_factor
    target_bps = int(pixels_per_second * bits_per_pixel)
    return target_bps

def normalize_array(arr: np.ndarray) -> np.ndarray:
    """Normalizes a NumPy array to the range [0, 1]."""
    min_val, max_val = arr.min(), arr.max()
    return (arr - min_val) / (max_val - min_val) if max_val > min_val else arr


def generate_opencv_benchmarks(
    reference_frames: Sequence[np.ndarray],
    strength_maps: Optional[Dict[str, np.ndarray]],
    block_size: int,
    framerate: float,
    width: int,
    height: int,
    temp_dir: str,
    video_bitrates: Dict[str, float],
) -> Tuple[Dict[str, str], Dict[str, float]]:
    """Generate OpenCV restoration benchmark videos for downstream evaluation."""

    if not strength_maps:
        return {}, {}

    print("\n" + "=" * 80)
    print("GENERATING OPENCV RESTORATION BENCHMARKS")
    print("=" * 80)

    benchmarks_dir = os.path.join(temp_dir, "opencv_benchmarks")
    os.makedirs(benchmarks_dir, exist_ok=True)

    opencv_benchmarks: Dict[str, str] = {}
    updated_bitrates = dict(video_bitrates)

    for method_name, maps in strength_maps.items():
        if maps is None:
            continue

        print(f"\nProcessing benchmarks for: {method_name}")
        target_bitrate = video_bitrates.get(method_name, 1_000_000)

        normalized_name = method_name.lower()

        if "downsample" in normalized_name or "realesrgan" in normalized_name:
            print("  - Generating Lanczos restoration benchmark...")
            benchmark_frames_lanczos: List[np.ndarray] = []
            for frame_idx, frame in enumerate(reference_frames):
                map_frame = maps[frame_idx]
                normalizer = np.max(map_frame)
                normalized_map = map_frame / normalizer if normalizer > 0 else map_frame
                downsampled_frame, _ = filter_frame_downsample(frame, normalized_map, block_size)
                restored_frame = restore_downsample_opencv_lanczos(downsampled_frame, map_frame, block_size)
                benchmark_frames_lanczos.append(restored_frame)

            lanczos_frames_dir = os.path.join(benchmarks_dir, f"{method_name}_lanczos_frames")
            os.makedirs(lanczos_frames_dir, exist_ok=True)
            for i, frame in enumerate(benchmark_frames_lanczos):
                cv2.imwrite(os.path.join(lanczos_frames_dir, f"{i+1:05d}.png"), frame)

            lanczos_video = os.path.join(benchmarks_dir, f"{method_name}_lanczos.mp4")
            encode_video(
                lanczos_frames_dir,
                lanczos_video,
                framerate,
                width,
                height,
                target_bitrate=target_bitrate,
            )
            key = APPROACH_PRESLEY_LANCZOS if method_name == APPROACH_PRESLEY_REALESRGAN else f"{method_name} Lanczos"
            opencv_benchmarks[key] = lanczos_video
            updated_bitrates[key] = video_bitrates.get(method_name, 0.0)

        elif "gaussian" in normalized_name or "blur" in normalized_name or "instantir" in normalized_name:
            print("  - Generating unsharp mask restoration benchmark...")
            benchmark_frames_unsharp: List[np.ndarray] = []
            for frame_idx, frame in enumerate(reference_frames):
                map_frame = maps[frame_idx]
                normalizer = np.max(map_frame)
                normalized_map = map_frame / normalizer if normalizer > 0 else map_frame
                blurred_frame, _ = filter_frame_gaussian(frame, normalized_map, block_size)
                restored_frame = restore_blur_opencv_unsharp_mask(blurred_frame, map_frame, block_size)
                benchmark_frames_unsharp.append(restored_frame)

            unsharp_frames_dir = os.path.join(benchmarks_dir, f"{method_name}_unsharp_frames")
            os.makedirs(unsharp_frames_dir, exist_ok=True)
            for i, frame in enumerate(benchmark_frames_unsharp):
                cv2.imwrite(os.path.join(unsharp_frames_dir, f"{i+1:05d}.png"), frame)

            unsharp_video = os.path.join(benchmarks_dir, f"{method_name}_unsharp.mp4")
            encode_video(
                unsharp_frames_dir,
                unsharp_video,
                framerate,
                width,
                height,
                target_bitrate=target_bitrate,
            )
            key = APPROACH_PRESLEY_UNSHARP if method_name == APPROACH_PRESLEY_INSTANTIR else f"{method_name} Unsharp"
            opencv_benchmarks[key] = unsharp_video
            updated_bitrates[key] = video_bitrates.get(method_name, 0.0)

    print(f"\nGenerated {len(opencv_benchmarks)} OpenCV restoration benchmarks.")
    print("=" * 80 + "\n")

    return opencv_benchmarks, updated_bitrates


def calculate_removability_scores(raw_video_file: str, reference_frames_folder: str, width: int, height: int, block_size: int, alpha: float = 0.5, working_dir: str = ".", smoothing_beta: float = 1) -> np.ndarray:
    """Compute removability scores via EVCA and UFO. Returns 3D array (frames, blocks_y, blocks_x) in [0,1]."""
    # Create temporary directories for outputs
    working_dir_abs = os.path.abspath(working_dir)
    maps_dir = os.path.join(working_dir_abs, "maps")
    ufo_masks_dir = os.path.join(maps_dir, "ufo_masks")
    os.makedirs(maps_dir, exist_ok=True)
    os.makedirs(ufo_masks_dir, exist_ok=True)
    
    # Get frame count
    frame_count = len(os.listdir(reference_frames_folder))
    
    # Save current directory and get absolute paths
    raw_video_abs = os.path.abspath(raw_video_file)
    reference_frames_abs = os.path.abspath(reference_frames_folder)
    ufo_masks_abs = os.path.abspath(ufo_masks_dir)
    evca_csv_dest = Path(maps_dir) / "evca.csv"
    evca_tc_dest = Path(maps_dir) / "evca_TC_blocks.csv"
    evca_sc_dest = Path(maps_dir) / "evca_SC_blocks.csv"
    
    try:
        # Calculate scene complexity with EVCA
        print("Running EVCA for complexity analysis...")
        try:
            import importlib

            evca_pkg = importlib.import_module("evca")
        except ImportError as exc:
            raise RuntimeError(
                "The 'evca' package is not installed in the current environment. "
                "Install it via 'pip install evca' (or pip install -e . inside the repo) before running Elvis."
            ) from exc

        evca_root = Path(evca_pkg.__file__).resolve().parent
        package_csv = evca_root / "evca.csv"
        package_tc = evca_root / "evca_TC_blocks.csv"
        package_sc = evca_root / "evca_SC_blocks.csv"

        # Remove stale outputs inside the package to avoid reading old data
        for path in (package_csv, package_tc, package_sc):
            if path.exists():
                try:
                    path.unlink()
                except Exception:
                    pass

        evca_cmd = [
            sys.executable,
            "-m",
            "evca.main",
            "-i",
            raw_video_abs,
            "-r",
            f"{width}x{height}",
            "-b",
            str(block_size),
            "-f",
            str(frame_count),
            "-c",
            "./evca.csv",
            "-bi",
            "1",
        ]
        result = subprocess.run(evca_cmd, capture_output=True, text=True, cwd=working_dir_abs)
        if result.returncode != 0:
            print(f"EVCA command failed: {result.stderr}")
            print(f"EVCA stdout: {result.stdout}")
            raise RuntimeError(f"EVCA execution failed: {result.stderr}")
        print("EVCA completed successfully")

        # Copy the generated CSVs from the installed package into the experiment maps directory
        for dest in (evca_csv_dest, evca_tc_dest, evca_sc_dest):
            if dest.exists():
                try:
                    dest.unlink()
                except Exception:
                    pass

        try:
            shutil.copy2(package_tc, evca_tc_dest)
            shutil.copy2(package_sc, evca_sc_dest)
            if package_csv.exists():
                shutil.copy2(package_csv, evca_csv_dest)
        except FileNotFoundError as exc:
            raise RuntimeError(
                "EVCA finished without producing the expected CSV outputs inside the package. "
                "Verify the installation and rerun the analysis."
            ) from exc
        
        # Calculate ROI with UFO (use installed `ufo` package when possible)
        print("Running UFO for object detection (using installed package if available)...")

        # Prepare a small temporary dataset structure for UFO under working_dir
        ufo_dataset_root = os.path.join(working_dir_abs, "ufo_dataset")
        ufo_image_dir = os.path.join(ufo_dataset_root, "image")
        ufo_class_dir = os.path.join(ufo_image_dir, "ref")
        # Ensure clean dataset dir
        shutil.rmtree(ufo_dataset_root, ignore_errors=True)
        os.makedirs(ufo_class_dir, exist_ok=True)

        # Copy reference frames into the temporary UFO class folder
        for fname in sorted(os.listdir(reference_frames_abs)):
            src = os.path.join(reference_frames_abs, fname)
            dst = os.path.join(ufo_class_dir, fname)
            shutil.copy(src, dst)

        # Helper to locate or download UFO weights in the installed package
        def _find_ufo_weights():
            candidate_names = [
                'model_best.pth', 'video_best.pth', 'ufo_weights.pth',
                'video_weights.pth', 'weights.pth'
            ]
            try:
                import importlib
                ufo_pkg = importlib.import_module('ufo')
                pkg_weights_dir = Path(ufo_pkg.__file__).parent / 'weights'
            except Exception:
                pkg_weights_dir = Path(working_dir_abs) / 'weights'

            for n in candidate_names:
                p = pkg_weights_dir / n
                if p.exists():
                    return str(p)

            # Try the bundled downloader if available
            try:
                downloader = importlib.import_module('ufo.download_ufo_weights')
                if hasattr(downloader, 'main'):
                    downloaded = downloader.main()
                    if downloaded:
                        return str(downloaded)
            except Exception:
                pass

            return None

        model_path_for_ufo = _find_ufo_weights() or 'weights/video_best.pth'

        # Try running the installed package programmatically first
        device_str = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        ran_ufo = False
        try:
            import importlib
            ufo_test = importlib.import_module('ufo.test')
            # debug_test expects: gpu_id, model_path, datapath(list), save_root_path(list), group_size, img_size, img_dir_name
            ufo_test.debug_test(device_str, model_path_for_ufo, [ufo_dataset_root], [ufo_masks_abs], 5, 224, 'image')
            ran_ufo = True
        except Exception as e:
            print(f"Programmatic UFO run failed or package not available: {e}")

        # Fallback: try invoking the module as a subprocess (python -m ufo.test)
        if not ran_ufo:
            try:
                ufo_cmd = f"python -m ufo.test --model='{model_path_for_ufo}' --data_path='{ufo_dataset_root}' --output_dir='{ufo_masks_abs}' --task='VSOD' --gpu_id='{device_str}'"
                result = subprocess.run(ufo_cmd, shell=True, capture_output=True, text=True)
                if result.returncode != 0:
                    print(f"UFO subprocess failed: {result.stderr}")
                    raise RuntimeError(f"UFO execution failed: {result.stderr}")
            except Exception as e:
                print(f"UFO execution (programmatic and subprocess) both failed: {e}")
                raise

        # UFO typically writes outputs into class subdirectories under the output dir
        # Flatten any nested class folders so mask files are directly inside ufo_masks_abs
        for root, dirs, files in os.walk(ufo_masks_abs):
            for fname in files:
                fpath = os.path.join(root, fname)
                target = os.path.join(ufo_masks_abs, fname)
                # If file already directly under ufo_masks_abs, skip
                if os.path.dirname(fpath) == ufo_masks_abs:
                    continue
                try:
                    shutil.move(fpath, target)
                except Exception:
                    try:
                        shutil.copy(fpath, target)
                    except Exception:
                        pass

        # Remove any leftover empty directories under ufo_masks_abs
        for dirpath, dirnames, filenames in os.walk(ufo_masks_abs, topdown=False):
            if dirpath == ufo_masks_abs:
                continue
            try:
                os.rmdir(dirpath)
            except Exception:
                pass

        # Cleanup temporary dataset
        shutil.rmtree(ufo_dataset_root, ignore_errors=True)
        
        # Load the CSV files from EVCA (copied into experiment/maps)
        temporal_array = np.loadtxt(evca_tc_dest, delimiter=',', skiprows=1)
        spatial_array = np.loadtxt(evca_sc_dest, delimiter=',', skiprows=1)

        num_blocks_x = width // block_size
        num_blocks_y = height // block_size
        # Ensure num_frames is consistent, handle potential empty CSV rows
        num_frames = min(temporal_array.shape[1], spatial_array.shape[1])

        # Trim arrays to the minimum number of frames and reshape
        temporal_3d = temporal_array[:, :num_frames].T.reshape(num_frames, num_blocks_y, num_blocks_x)
        spatial_3d = spatial_array[:, :num_frames].T.reshape(num_frames, num_blocks_y, num_blocks_x)

        # Normalize arrays to the range [0, 1]
        temporal_3d = normalize_array(temporal_3d)
        spatial_3d = normalize_array(spatial_3d)

        # Initialize the removability array.
        removability_scores = np.zeros_like(spatial_3d)

        # Calculate removability for all frames except the last one
        removability_scores[:-1] = alpha * spatial_3d[:-1] + (1 - alpha) * temporal_3d[1:]

        # For the last frame, there is no successive temporal complexity, so we rely only on spatial
        removability_scores[-1] = spatial_3d[-1]

        # Load masks and adjust removability scores
        for i in range(num_frames):
            mask_path = os.path.join(ufo_masks_abs, f"{i+1:05d}.png")
            if os.path.exists(mask_path):
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                # Resize mask to match the block grid dimensions
                resized_mask = cv2.resize(mask, (num_blocks_x, num_blocks_y), interpolation=cv2.INTER_NEAREST)

                # --- Vectorized Mask Application ---
                # Find background blocks (mask value is 0)
                background_blocks = (resized_mask == 0)
                # Significantly increase the score for background blocks, making them prime candidates for removal.
                removability_scores[i][background_blocks] *= 10.0
            else:
                print(f"Warning: Mask file not found for frame {i}: {mask_path}")

        # Apply temporal smoothing if requested
        if smoothing_beta < 1 and removability_scores.shape[0] >= 2:
            print("Applying temporal smoothing to removability scores...")
            smoothed_scores = np.zeros_like(removability_scores)

            # The first frame has no prior frame, so its scores remain unchanged.
            smoothed_scores[0] = removability_scores[0]

            # For all subsequent frames, apply the smoothing formula (vectorized for speed).
            smoothed_scores[1:] = (
                smoothing_beta * removability_scores[1:]
                + (1 - smoothing_beta) * removability_scores[:-1]
            )

            removability_scores = smoothed_scores

        # Final normalization to [0, 1]
        removability_scores = normalize_array(removability_scores)

        return removability_scores

    except Exception as e:
        print(f"Error in calculate_removability_scores: {e}")
        raise

def encode_video(input_frames_dir: str, output_video: str, framerate: float, width: int, height: int, target_bitrate: int = None, preset: str = "medium", pix_fmt: str = "yuv420p", **extra_params) -> None:
    """Encode video using two-pass libx265. Supports lossy (with target_bitrate) or lossless mode."""
    temp_dir = os.path.dirname(output_video) or '.'
    os.makedirs(temp_dir, exist_ok=True)
    
    passlog_file = os.path.join(temp_dir, f"ffmpeg_2pass_log_{os.path.basename(output_video)}")
    null_device = "NUL" if platform.system() == "Windows" else "/dev/null"
    
    try:
        extra_params = {key: value for key, value in extra_params.items() if value is not None}
        pass1_extra_params = {key: value for key, value in extra_params.items() if key != "qpfile"}

        def _extend_x265_params(base: str, params: Dict[str, Any]) -> str:
            if not params:
                return base
            suffix = "".join(f":{key}={value}" for key, value in params.items())
            return f"{base}{suffix}"

        # Base command shared by both passes

        base_cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-framerate", str(framerate),
            "-i", f"{input_frames_dir}/%05d.png",
            "-vf", f"scale={width}:{height}:flags=lanczos,format={pix_fmt}",
        ]

        if target_bitrate is None:
            # Lossless encoding with two passes - use faster preset for lossless
            preset = "fast"
            x265_base_params = "lossless=1"
            
            # Pass 1
            pass1_params = _extend_x265_params(
                f"{x265_base_params}:pass=1:stats={passlog_file}", pass1_extra_params
            )
            pass1_cmd = base_cmd + [
                "-c:v", "libx265",
                "-preset", preset,
                "-x265-params", pass1_params,
                "-f", "mp4", "-y", null_device
            ]
            subprocess.run(pass1_cmd, check=True, capture_output=True, text=True)
            
            # Pass 2
            pass2_params = _extend_x265_params(
                f"{x265_base_params}:pass=2:stats={passlog_file}", extra_params
            )
            
            pass2_cmd = base_cmd + [
                "-c:v", "libx265",
                "-preset", preset,
                "-x265-params", pass2_params,
                "-y", output_video
            ]
            result = subprocess.run(pass2_cmd, check=True, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"Error in pass 2 encoding: {result.stderr}")
        else:
            # Lossy encoding with bitrate control and two passes
            
            # Pass 1
            pass1_params = _extend_x265_params(
                f"pass=1:stats={passlog_file}", pass1_extra_params
            )
            pass1_cmd = base_cmd + [
                "-c:v", "libx265",
                "-b:v", str(target_bitrate),
                "-minrate", str(int(target_bitrate * 0.9)),
                "-maxrate", str(int(target_bitrate * 1.1)),
                "-bufsize", str(target_bitrate),
                "-preset", preset,
                "-g", str(framerate),  # Set GOP size to framerate for approx 1-second keyframes
                "-x265-params", pass1_params,
                "-f", "mp4", "-y", null_device
            ]
            subprocess.run(pass1_cmd, check=True, capture_output=True, text=True)
            
            # Pass 2
            pass2_params = _extend_x265_params(
                f"pass=2:stats={passlog_file}", extra_params
            )
            
            pass2_cmd = base_cmd + [
                "-c:v", "libx265",
                "-b:v", str(target_bitrate),
                "-minrate", str(int(target_bitrate * 0.9)),
                "-maxrate", str(int(target_bitrate * 1.1)),
                "-bufsize", str(target_bitrate),
                "-preset", preset,
                "-g", str(framerate),  # Set GOP size to framerate for approx 1-second keyframes
                "-x265-params", pass2_params,
                "-y", output_video
            ]
            result = subprocess.run(pass2_cmd, check=True, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"Error in pass 2 encoding: {result.stderr}")

    except subprocess.CalledProcessError as e:
        print("--- FFMPEG COMMAND FAILED ---")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        raise RuntimeError(f"FFmpeg command failed with exit code {e.returncode}") from e
    finally:
        # Clean up pass log files - use glob for faster matching
        import glob
        log_pattern = os.path.join(temp_dir, f"ffmpeg_2pass_log_{os.path.basename(output_video)}*")
        for f in glob.glob(log_pattern):
            try:
                os.remove(f)
            except:
                pass

def decode_video(video_path: str, output_dir: str, framerate: float = None, start_number: int = 1, quality: int = 1) -> bool:
    """Decode video to PNG frames. Returns True on success."""
    os.makedirs(output_dir, exist_ok=True)
    
    decode_cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-i", video_path,
        "-pix_fmt", "rgb24",
        "-q:v", str(quality),
    ]
    
    # Add framerate if specified
    if framerate is not None:
        decode_cmd.extend(["-r", str(framerate)])
    
    # Add output parameters
    decode_cmd.extend([
        "-f", "image2",
        "-start_number", str(start_number),
        "-y", os.path.join(output_dir, "%05d.png")
    ])
    
    result = subprocess.run(decode_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error decoding {video_path}: {result.stderr}")
        return False
    return True

# Elvis v1 functions

def split_image_into_blocks(image: np.ndarray, block_size: int) -> np.ndarray:
    """
    Splits an image into a 5D array of blocks.
    Shape: (num_blocks_y, num_blocks_x, block_size, block_size, channels)
    """
    h, w, c = image.shape
    # Ensure the image dimensions are divisible by the block size
    if h % block_size != 0 or w % block_size != 0:
        raise ValueError("Image dimensions must be divisible by block_size.")
        
    num_blocks_y = h // block_size
    num_blocks_x = w // block_size
    
    # The crucial reshape and transpose operations
    blocks = image.reshape(num_blocks_y, block_size, num_blocks_x, block_size, c)
    blocks = blocks.swapaxes(1, 2)
    return blocks

def apply_selective_removal(image: np.ndarray, frame_scores: np.ndarray, block_size: int, shrink_amount: float) -> Tuple[np.ndarray, np.ndarray, List[List[int]]]:
    """Remove blocks based on scores. Returns (new_image, removal_mask, removed_coords)."""
    num_blocks_y, num_blocks_x = frame_scores.shape
    
    # --- 1. Selection Step ---
    if shrink_amount < 1.0:
        num_blocks_to_remove = int(shrink_amount * num_blocks_x)
    else:
        num_blocks_to_remove = int(shrink_amount)
    num_blocks_to_remove = min(num_blocks_to_remove, num_blocks_x)

    block_coords_to_remove = []
    for j in range(num_blocks_y):
        row_scores = frame_scores[j, :]
        indices_to_remove = np.argsort(-row_scores)[:num_blocks_to_remove]
        indices_to_remove.sort()
        block_coords_to_remove.append(indices_to_remove.tolist())

    # --- 2. Action Step ---
    
    # Split image into blocks
    blocks = split_image_into_blocks(image, block_size)
    
    # Create a 2D mask of which blocks to remove
    removal_mask = np.zeros((num_blocks_y, num_blocks_x), dtype=np.int8)
    rows_indices = np.arange(num_blocks_y).repeat([len(cols) for cols in block_coords_to_remove])
    if len(rows_indices) > 0:
        cols_indices = np.concatenate(block_coords_to_remove)
        removal_mask[rows_indices, cols_indices] = 1

    # Filter out the blocks marked for removal
    kept_blocks_list = [
        blocks[i, np.where(removal_mask[i] == 0)[0]]
        for i in range(num_blocks_y)
    ]
    kept_blocks = np.stack(kept_blocks_list, axis=0)

    # Reconstruct the final image from the remaining blocks
    new_image = combine_blocks_into_image(kept_blocks)
    
    return new_image, removal_mask, block_coords_to_remove

def combine_blocks_into_image(blocks: np.ndarray) -> np.ndarray:
    """Combine 5D array of blocks back into single image. Inverse of split_image_into_blocks."""
    num_blocks_y, num_blocks_x, block_size, _, c = blocks.shape
    image = blocks.swapaxes(1, 2)
    image = image.reshape(num_blocks_y * block_size, num_blocks_x * block_size, c)
    return image

def stretch_frame(shrunk_frame: np.ndarray, binary_mask: np.ndarray, block_size: int) -> np.ndarray:
    """Reconstruct full-resolution frame from shrunk version using removal mask."""
    num_blocks_y, num_blocks_x = binary_mask.shape
    channels = shrunk_frame.shape[2]
    final_blocks = np.zeros((num_blocks_y, num_blocks_x, block_size, block_size, channels), dtype=shrunk_frame.dtype)

    # 3. Get the blocks from the shrunk frame
    shrunk_blocks = split_image_into_blocks(shrunk_frame, block_size)

    # 4. Use boolean indexing to place the shrunk blocks onto the canvas.
    #    This is the key step. It finds all positions where mask is 0 (i.e., "keep")
    #    and places the shrunk_blocks there in a single, fast operation.
    #    Note: The number of `True` elements in `(binary_mask == 0)` must equal the number of blocks
    #    in `shrunk_blocks`.
    final_blocks[binary_mask == 0] = shrunk_blocks.reshape(-1, block_size, block_size, channels)

    # 5. Combine the completed set of blocks back into a single image
    reconstructed_image = combine_blocks_into_image(final_blocks)
    
    return reconstructed_image


def inpaint_with_propainter(
    stretched_frames_dir: str,
    removal_masks_dir: str,
    output_frames_dir: str,
    width: int,
    height: int,
    framerate: float,
    resize_ratio: float = 1.0,
    ref_stride: int = 20,
    neighbor_length: int = 4,
    subvideo_length: int = 40,
    mask_dilation: int = 4,
    raft_iter: int = 20,
    fp16: bool = True,
    devices: Optional[Sequence[Union[int, str, torch.device]]] = None,
    parallel_chunk_length: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
) -> None:
    """Use ProPainter to inpaint stretched frames with removed blocks."""
    original_dir = os.getcwd()

    stretched_frames_abs = os.path.abspath(stretched_frames_dir)
    removal_masks_abs = os.path.abspath(removal_masks_dir)
    output_frames_abs = os.path.abspath(output_frames_dir)
    output_frames_path = Path(output_frames_abs)
    os.makedirs(output_frames_abs, exist_ok=True)

    for stale_frame in output_frames_path.glob("*.png"):
        if stale_frame.is_file():
            stale_frame.unlink()

    try:
        import propainter as _propainter  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("propainter package is not installed. Install it with `pip install propainter`.") from exc

    try:
        frame_files = sorted([f for f in os.listdir(stretched_frames_abs) if f.lower().endswith(('.jpg', '.png'))])
        mask_files = sorted([f for f in os.listdir(removal_masks_abs) if f.lower().endswith(('.jpg', '.png'))])
        if len(frame_files) == 0 or len(frame_files) != len(mask_files):
            raise ValueError("Frame and mask counts must match and be non-zero for ProPainter input.")

        total_frames = len(frame_files)
        frame_paths = [Path(stretched_frames_abs) / f for f in frame_files]
        mask_paths = [Path(removal_masks_abs) / f for f in mask_files]

        resolved_devices = _resolve_device_list(devices, prefer_cuda=True, allow_cpu_fallback=True)

        effective_chunk = parallel_chunk_length if parallel_chunk_length is not None else subvideo_length
        if effective_chunk is None or effective_chunk <= 0:
            effective_chunk = total_frames
        effective_chunk = max(1, min(effective_chunk, total_frames))

        effective_overlap = chunk_overlap if chunk_overlap is not None else neighbor_length
        effective_overlap = max(0, effective_overlap)
        if effective_chunk == 1:
            effective_overlap = 0
        else:
            effective_overlap = min(effective_overlap, effective_chunk // 2)

        total_chunks = max(1, math.ceil(total_frames / effective_chunk))

        propainter_entry = [
            sys.executable,
            "-m",
            "propainter.inference_propainter",
        ]
        run_cwd = original_dir

        base_flags = [
            "--width", str(width),
            "--height", str(height),
            "--resize_ratio", str(resize_ratio),
            "--ref_stride", str(ref_stride),
            "--neighbor_length", str(neighbor_length),
            "--mask_dilation", str(mask_dilation),
            "--raft_iter", str(raft_iter),
            "--save_fps", str(int(framerate)),
        ]
        if fp16:
            base_flags.append("--fp16")

        class _PropainterChunk(NamedTuple):
            job_id: int
            start: int
            end: int
            expanded_start: int
            expanded_end: int

        chunks: List[_PropainterChunk] = []
        cursor = 0
        job_id = 0
        while cursor < total_frames:
            end = min(total_frames, cursor + effective_chunk)
            expanded_start = max(0, cursor - effective_overlap)
            expanded_end = min(total_frames, end + effective_overlap)
            if expanded_end <= expanded_start:
                expanded_end = min(total_frames, expanded_start + effective_chunk)
            chunks.append(
                _PropainterChunk(
                    job_id=job_id,
                    start=cursor,
                    end=end,
                    expanded_start=expanded_start,
                    expanded_end=expanded_end,
                )
            )
            job_id += 1
            cursor = end

        device_summary = ", ".join(str(dev) for dev in resolved_devices)
        print(
            f"Using ProPainter on devices: {device_summary} | chunk length: {effective_chunk} | overlap: {effective_overlap}"
        )
        print(f"Total frames: {total_frames} | parallel chunks: {len(chunks)}")

        def _visible_device_token(device: torch.device) -> Optional[str]:
            if device.type == "cuda":
                return str(device.index if device.index is not None else 0)
            return None

        def _run_chunk(chunk: _PropainterChunk, device: torch.device) -> None:
            visible_token = _visible_device_token(device)
            env = os.environ.copy()
            if visible_token is not None:
                env["CUDA_VISIBLE_DEVICES"] = visible_token
            else:
                env.pop("CUDA_VISIBLE_DEVICES", None)

            with tempfile.TemporaryDirectory() as temp_dir:
                temp_dir_path = Path(temp_dir)
                video_input_dir = temp_dir_path / f"propainter_job_{chunk.job_id:04d}"
                mask_input_dir = temp_dir_path / f"propainter_masks_{chunk.job_id:04d}"
                output_root_dir = temp_dir_path / f"propainter_output_{chunk.job_id:04d}"
                video_input_dir.mkdir(parents=True, exist_ok=True)
                mask_input_dir.mkdir(parents=True, exist_ok=True)
                output_root_dir.mkdir(parents=True, exist_ok=True)

                expanded_indices = list(range(chunk.expanded_start, chunk.expanded_end))
                for local_idx, frame_idx in enumerate(expanded_indices):
                    shutil.copy(frame_paths[frame_idx], video_input_dir / f"{local_idx:04d}.png")
                    shutil.copy(mask_paths[frame_idx], mask_input_dir / f"{local_idx:04d}.png")

                chunk_length = max(1, chunk.expanded_end - chunk.expanded_start)
                sub_len = subvideo_length if subvideo_length and subvideo_length > 0 else chunk_length
                chunk_specific_flags = base_flags + [
                    "--subvideo_length",
                    str(max(1, min(sub_len, chunk_length))),
                ]

                cmd = (
                    propainter_entry
                    + [
                        "--video",
                        str(video_input_dir),
                        "--mask",
                        str(mask_input_dir),
                        "--output",
                        str(output_root_dir),
                    ]
                    + chunk_specific_flags
                )

                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    cwd=run_cwd,
                    env=env,
                )
                if result.returncode != 0:
                    print(f"ProPainter stdout (chunk {chunk.job_id}): {result.stdout}")
                    print(f"ProPainter stderr (chunk {chunk.job_id}): {result.stderr}")
                    raise RuntimeError(
                        f"ProPainter inference failed for chunk {chunk.job_id}. See logs above for details."
                    )

                video_name = video_input_dir.name
                generated_frames_dir = output_root_dir / video_name / "frames"
                if not generated_frames_dir.exists():
                    raise RuntimeError(
                        f"ProPainter did not emit frames for chunk {chunk.job_id} at {generated_frames_dir}"
                    )

                generated_files = sorted(
                    [p for p in generated_frames_dir.iterdir() if p.suffix.lower() == ".png"]
                )
                if not generated_files:
                    raise RuntimeError(
                        f"No frames produced by ProPainter for chunk {chunk.job_id} in {generated_frames_dir}"
                    )

                skip_prefix = chunk.start - chunk.expanded_start
                keep_count = chunk.end - chunk.start
                # Trim overlapping context so only the target frame range is persisted
                selected_files = generated_files[skip_prefix : skip_prefix + keep_count]
                if len(selected_files) != keep_count:
                    raise RuntimeError(
                        f"Unexpected frame count for chunk {chunk.job_id}: expected {keep_count}, got {len(selected_files)}"
                    )

                for offset, frame_path in enumerate(selected_files):
                    dst_frame = output_frames_path / f"{chunk.start + offset + 1:05d}.png"
                    shutil.copy(frame_path, dst_frame)

                print(
                    f"  ✓ ProPainter chunk {chunk.job_id + 1}/{len(chunks)} "
                    f"frames {chunk.start + 1}-{chunk.end} on {device}"
                )

        max_workers = min(len(resolved_devices), len(chunks))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            chunk_iter = iter(chunks)
            while True:
                tasks = []
                for device in resolved_devices:
                    chunk = next(chunk_iter, None)
                    if chunk is None:
                        break
                    tasks.append(executor.submit(_run_chunk, chunk, device))

                if not tasks:
                    break

                for future in tasks:
                    future.result()

        print(f"Inpainted frames saved to {output_frames_abs}")

    except Exception as exc:
        print(f"Error in inpaint_with_propainter: {exc}")
        raise
    finally:
        os.chdir(original_dir)

def inpaint_with_e2fgvi(
    stretched_frames_dir: str,
    removal_masks_dir: str,
    output_frames_dir: str,
    width: int,
    height: int,
    framerate: float,
    ref_stride: int = 10,
    neighbor_stride: int = 5,
    num_ref: int = -1,
    mask_dilation: int = 4,
    devices: Optional[Sequence[Union[int, str, torch.device]]] = None,
    parallel_chunk_length: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
) -> None:
    """Use E2FGVI to inpaint stretched frames with removed blocks. Supports multi-GPU parallelism."""
    stretched_frames_abs = os.path.abspath(stretched_frames_dir)
    removal_masks_abs = os.path.abspath(removal_masks_dir)
    output_frames_abs = os.path.abspath(output_frames_dir)

    frames_path = Path(stretched_frames_abs)
    masks_path = Path(removal_masks_abs)
    output_path = Path(output_frames_abs)

    if not frames_path.is_dir():
        raise ValueError(f"Stretched frames directory does not exist: {stretched_frames_abs}")
    if not masks_path.is_dir():
        raise ValueError(f"Removal masks directory does not exist: {removal_masks_abs}")

    output_path.mkdir(parents=True, exist_ok=True)

    valid_suffixes = (".png", ".jpg", ".jpeg")

    frame_paths = sorted([p for p in frames_path.iterdir() if p.suffix.lower() in valid_suffixes])
    if not frame_paths:
        raise ValueError(f"No frames found in {stretched_frames_abs}")
    mask_paths = sorted([p for p in masks_path.iterdir() if p.suffix.lower() in valid_suffixes])
    if len(frame_paths) != len(mask_paths):
        raise ValueError(
            f"Frame count ({len(frame_paths)}) does not match mask count ({len(mask_paths)})."
        )

    for frame_file, mask_file in zip(frame_paths, mask_paths):
        if frame_file.stem != mask_file.stem:
            raise ValueError(
                f"Frame/mask mismatch: {frame_file.name} vs {mask_file.name}"
            )

    for stale_file in output_path.iterdir():
        if stale_file.is_file() and stale_file.suffix.lower() in valid_suffixes:
            stale_file.unlink()

    try:
        import e2fgvi as e2fgvi_pkg  # type: ignore[import-untyped]
    except ImportError as exc:
        raise RuntimeError(
            "E2FGVI package is not installed. Install it with `pip install e2fgvi`."
        ) from exc
    package_dir = Path(e2fgvi_pkg.__file__).resolve().parent
    base_cmd_prefix = [
        sys.executable,
        "-m",
        "e2fgvi",
    ]

    ckpt_path = package_dir / "release_model" / "E2FGVI-HQ-CVPR22.pth"

    if framerate is None:
        raise ValueError("`framerate` must be provided for E2FGVI inference.")

    def _build_command(video_dir: str, mask_dir: str, save_dir: str) -> List[str]:
        cmd = list(base_cmd_prefix)
        cmd.extend(
            [
                "--model",
                "e2fgvi_hq",
                "--video",
                video_dir,
                "--mask",
                mask_dir,
                "--ckpt",
                str(ckpt_path),
                "--step",
                str(ref_stride),
                "--num_ref",
                str(num_ref),
                "--neighbor_stride",
                str(neighbor_stride),
                "--set_size",
                "--width",
                str(width),
                "--height",
                str(height),
                "--savefps",
                str(int(framerate)),
                "--save_frames",
                save_dir,
            ]
        )
        return cmd

    total_frames = len(frame_paths)
    resolved_devices = _resolve_device_list(devices, prefer_cuda=True, allow_cpu_fallback=True)
    cuda_devices = [dev for dev in resolved_devices if dev.type == "cuda"]

    preferred_device: Optional[torch.device] = None
    if cuda_devices:
        preferred_device = cuda_devices[0]
    elif resolved_devices:
        preferred_device = resolved_devices[0]

    def _device_label(device_obj: Optional[torch.device]) -> str:
        if device_obj is None:
            return "default"
        if device_obj.type == "cuda":
            idx = device_obj.index if device_obj.index is not None else 0
            return f"cuda:{idx}"
        return str(device_obj)

    def _run_single(device_override: Optional[torch.device]) -> None:
        print("Running E2FGVI inference...")
        cmd = _build_command(stretched_frames_abs, removal_masks_abs, output_frames_abs)
        env = os.environ.copy()
        if device_override is not None:
            if device_override.type == "cuda":
                cuda_idx = device_override.index if device_override.index is not None else 0
                env["CUDA_VISIBLE_DEVICES"] = str(cuda_idx)
            else:
                env["CUDA_VISIBLE_DEVICES"] = ""
        result = subprocess.run(cmd, capture_output=True, text=True, env=env)
        if result.returncode != 0:
            print(f"E2FGVI stdout: {result.stdout}")
            print(f"E2FGVI stderr: {result.stderr}")
            raise RuntimeError(f"E2FGVI inference failed: {result.stderr}")
        if result.stdout:
            print(f"E2FGVI output: {result.stdout}")

        generated_frames = list(output_path.glob("*.png"))
        if generated_frames:
            print(f"E2FGVI inpainted frames saved to {output_frames_abs}")
            return

        results_dir = package_dir / "results"
        if not results_dir.exists():
            raise RuntimeError(f"E2FGVI results directory not found at {results_dir}")

        result_videos = [f for f in results_dir.iterdir() if f.suffix == ".mp4"]
        if not result_videos:
            raise RuntimeError(f"No result video found in {results_dir}")

        result_videos.sort(key=lambda path: path.stat().st_mtime, reverse=True)
        result_video_path = result_videos[0]

        print("Decoding E2FGVI result video to frames...")

        if not decode_video(
            str(result_video_path),
            output_frames_abs,
            framerate=framerate,
            start_number=1,
            quality=1,
        ):
            raise RuntimeError(f"Failed to decode E2FGVI result video: {result_video_path}")

        print(f"E2FGVI inpainted frames saved to {output_frames_abs}")

        try:
            result_video_path.unlink()
        except OSError:
            pass

    def _run_parallel(device_list: Sequence[torch.device]) -> None:
        default_overlap = max(neighbor_stride, 1) * 2
        overlap = int(chunk_overlap) if chunk_overlap is not None else default_overlap
        overlap = max(0, overlap)
        if total_frames == 1:
            overlap = 0
        else:
            overlap = min(overlap, total_frames - 1)

        if parallel_chunk_length is None or parallel_chunk_length <= 0:
            chunk_len = math.ceil(total_frames / len(device_list))
        else:
            chunk_len = int(parallel_chunk_length)

        chunk_len = max(1, min(chunk_len, total_frames))
        if chunk_len <= overlap:
            chunk_len = min(total_frames, overlap + 1)

        class _E2FGVIChunk(NamedTuple):
            index: int
            chunk_start: int
            chunk_end: int
            core_start: int
            core_end: int

        chunks: List[_E2FGVIChunk] = []
        cursor = 0
        idx = 0
        while cursor < total_frames:
            core_end = min(total_frames, cursor + chunk_len)
            chunk_start = max(0, cursor - (overlap if cursor > 0 else 0))
            chunk_end = min(total_frames, core_end + (overlap if core_end < total_frames else 0))
            chunks.append(
                _E2FGVIChunk(
                    index=idx,
                    chunk_start=chunk_start,
                    chunk_end=chunk_end,
                    core_start=cursor,
                    core_end=core_end,
                )
            )
            cursor = core_end
            idx += 1

        device_labels = [_device_label(dev) for dev in device_list]
        print("Running E2FGVI inference across multiple devices...")
        print(
            f"  Devices: {', '.join(device_labels)} | chunk length: {chunk_len} | overlap: {overlap} | total chunks: {len(chunks)}"
        )

        def _link_or_copy(src: Path, dst: Path) -> None:
            try:
                os.symlink(src, dst)
            except OSError:
                shutil.copy2(src, dst)

        def _process_chunk(chunk: _E2FGVIChunk, device_obj: torch.device) -> None:
            device_label = _device_label(device_obj)
            print(
                f"    -> E2FGVI chunk {chunk.index + 1}/{len(chunks)} frames {chunk.core_start + 1}-{chunk.core_end} on {device_label}"
            )
            with tempfile.TemporaryDirectory(prefix=f"e2fgvi_chunk_{chunk.index:03d}_") as tmp_root:
                tmp_root_path = Path(tmp_root)
                chunk_frames_path = tmp_root_path / "frames"
                chunk_masks_path = tmp_root_path / "masks"
                chunk_output_path = tmp_root_path / "output"
                chunk_frames_path.mkdir(parents=True, exist_ok=True)
                chunk_masks_path.mkdir(parents=True, exist_ok=True)
                chunk_output_path.mkdir(parents=True, exist_ok=True)

                chunk_indices = list(range(chunk.chunk_start, chunk.chunk_end))

                for seq_idx, original_idx in enumerate(chunk_indices, start=1):
                    frame_src = frame_paths[original_idx]
                    mask_src = mask_paths[original_idx]

                    frame_dest = chunk_frames_path / f"{seq_idx:05d}{frame_src.suffix}"
                    mask_dest = chunk_masks_path / f"{seq_idx:05d}{mask_src.suffix}"

                    _link_or_copy(frame_src, frame_dest)
                    _link_or_copy(mask_src, mask_dest)

                cmd = _build_command(str(chunk_frames_path), str(chunk_masks_path), str(chunk_output_path))

                env = os.environ.copy()
                if device_obj.type == "cuda":
                    cuda_idx = device_obj.index if device_obj.index is not None else 0
                    env["CUDA_VISIBLE_DEVICES"] = str(cuda_idx)
                else:
                    env["CUDA_VISIBLE_DEVICES"] = ""

                result = subprocess.run(cmd, capture_output=True, text=True, env=env)
                if result.returncode != 0:
                    print(f"E2FGVI stdout (chunk {chunk.index}): {result.stdout}")
                    print(f"E2FGVI stderr (chunk {chunk.index}): {result.stderr}")
                    raise RuntimeError(f"E2FGVI inference failed for chunk {chunk.index}: {result.stderr}")

                output_files = sorted(
                    [p for p in chunk_output_path.iterdir() if p.suffix.lower() in valid_suffixes],
                    key=lambda p: p.name,
                )
                if len(output_files) != len(chunk_indices):
                    raise RuntimeError(
                        f"Mismatch between produced frames ({len(output_files)}) and expected count ({len(chunk_indices)}) "
                        f"for E2FGVI chunk {chunk.index}."
                    )

                for rel_idx, original_idx in enumerate(chunk_indices):
                    if original_idx < chunk.core_start or original_idx >= chunk.core_end:
                        continue
                    output_file = output_files[rel_idx]
                    final_path = output_path / frame_paths[original_idx].name
                    if final_path.exists():
                        final_path.unlink()
                    shutil.copy2(output_file, final_path)

        max_workers = len(device_list)
        if max_workers <= 0:
            raise RuntimeError("No devices available for E2FGVI parallel execution.")

        chunk_iter = iter(chunks)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            while True:
                futures = []
                for device_obj in device_list:
                    chunk = next(chunk_iter, None)
                    if chunk is None:
                        break
                    futures.append(executor.submit(_process_chunk, chunk, device_obj))
                if not futures:
                    break
                for future in futures:
                    future.result()

        missing = [path.name for path in frame_paths if not (output_path / path.name).exists()]
        if missing:
            raise RuntimeError(
                f"E2FGVI parallel execution missing {len(missing)} frame(s); examples: {', '.join(missing[:5])}"
            )

        print(f"E2FGVI inpainted frames saved to {output_frames_abs}")

    if len(cuda_devices) >= 2 and total_frames > 1:
        _run_parallel(cuda_devices)
    else:
        _run_single(preferred_device)

# Elvis v2 functions

def encode_with_roi(input_frames_dir: str, output_video: str, removability_scores: np.ndarray, block_size: int, framerate: float, width: int, height: int, target_bitrate: int = 1000000, save_qp_maps: bool = False, qp_maps_dir: str = None) -> None:
    """Encode video with per-block QP control based on removability scores, using two-pass encoding."""
    num_frames, num_blocks_y, num_blocks_x = removability_scores.shape
    temp_dir = os.path.dirname(output_video) or '.'
    os.makedirs(temp_dir, exist_ok=True)
    
    qpfile_path = os.path.join(temp_dir, "qpfile_per_block.txt")
    passlog_file = os.path.join(temp_dir, "ffmpeg_2pass_log")
    
    # Use platform-specific null device for the first pass output
    null_device = "NUL" if platform.system() == "Windows" else "/dev/null"

    try:
        # --- Part 1: Generate the detailed per-block qpfile ---
        print("Generating detailed qpfile for per-block quality control...")

        # Map normalized scores [0, 1] to QP offsets [-1, 1]
        qp_maps = np.clip((removability_scores * 2.0) - 1.0, -1.0, 1.0).astype(np.float32)

        valid_ctu_sizes = [16, 32, 64]
        largest_dimension = max(width, height)
        min_ctu_by_resolution = 16
        if largest_dimension >= 4320:
            min_ctu_by_resolution = 64
        elif largest_dimension >= 2160:
            min_ctu_by_resolution = 32

        nearest_ctu = min(valid_ctu_sizes, key=lambda size: abs(size - block_size))
        if nearest_ctu < block_size:
            larger_sizes = [size for size in valid_ctu_sizes if size >= block_size]
            ctu_size = larger_sizes[0] if larger_sizes else valid_ctu_sizes[-1]
        else:
            ctu_size = nearest_ctu

        if ctu_size < min_ctu_by_resolution:
            compliant_sizes = [size for size in valid_ctu_sizes if size >= min_ctu_by_resolution]
            if compliant_sizes:
                ctu_size = compliant_sizes[0]
            else:
                ctu_size = valid_ctu_sizes[-1]

        ctu_cols = math.ceil(width / ctu_size)
        ctu_rows = math.ceil(height / ctu_size)

        qp_maps_aligned = np.empty((num_frames, ctu_rows, ctu_cols), dtype=np.float32)
        if (ctu_rows, ctu_cols) != (num_blocks_y, num_blocks_x):
            print(
                f"Resizing per-block QP maps from {num_blocks_y}x{num_blocks_x} blocks to CTU grid {ctu_rows}x{ctu_cols} (CTU={ctu_size})."
            )

        for frame_idx in range(num_frames):
            frame_map = qp_maps[frame_idx]
            if frame_map.shape == (ctu_rows, ctu_cols):
                qp_maps_aligned[frame_idx] = frame_map
                continue

            interpolation = cv2.INTER_AREA if ctu_size >= block_size else cv2.INTER_LINEAR
            qp_maps_aligned[frame_idx] = cv2.resize(
                frame_map,
                (ctu_cols, ctu_rows),
                interpolation=interpolation,
            )

        # Generate qpfile with optimized string building
        with open(qpfile_path, 'w') as f:
            for frame_idx in range(num_frames):
                # Start the line with frame index and a generic frame type ('P')
                # The '-1' for QP indicates we are providing per-block QPs
                line_parts = [f"{frame_idx} P -1"]
                
                # Append QP offsets for each CTU in raster order
                qp_frame = qp_maps_aligned[frame_idx]
                block_qps = [
                    f"{bx},{by},{qp_frame[by, bx]:.4f}"
                    for by in range(ctu_rows)
                    for bx in range(ctu_cols)
                ]
                line_parts.extend(block_qps)

                f.write(" ".join(line_parts) + "\n")
        print(f"qpfile generated at {qpfile_path}")
        
        # --- Save QP maps as images if requested ---
        if save_qp_maps:
            if qp_maps_dir is None:
                qp_maps_dir = os.path.join(temp_dir, "qp_maps")
            os.makedirs(qp_maps_dir, exist_ok=True)
            for frame_idx in range(num_frames):
                # Save at block resolution, not CTU resolution
                qp_map_block_res = qp_maps[frame_idx]  # Original block resolution
                qp_map_image = np.clip((qp_map_block_res + 1.0) * 127.5, 0, 255).astype(np.uint8)
                cv2.imwrite(os.path.join(qp_maps_dir, f"qp_map_{frame_idx:05d}.png"), qp_map_image)
            print(f"QP maps saved to {qp_maps_dir} at block resolution ({num_blocks_y}x{num_blocks_x})")

        # --- Part 2: Run Global Two-Pass Encode with QP file ---
        print(f"Starting two-pass encoding with per-block QP control (CTU {ctu_size})...")
        encode_video(
            input_frames_dir=input_frames_dir,
            output_video=output_video,
            framerate=framerate,
            width=width,
            height=height,
            target_bitrate=target_bitrate,
            ctu=ctu_size,
            qpfile=qpfile_path
        )

        print(f"\nTwo-pass per-block encoding complete. Output saved to {output_video}")

    except subprocess.CalledProcessError as e:
        print("--- FFMPEG COMMAND FAILED ---")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        raise RuntimeError(f"FFmpeg command failed with exit code {e.returncode}") from e
    finally:
        # --- Part 3: Clean up temporary files ---
        print("Cleaning up temporary files...")
        if os.path.exists(qpfile_path):
            os.remove(qpfile_path)
        # FFmpeg might create multiple log files (e.g., .log, .log.mbtree) - use glob
        import glob
        log_pattern = os.path.join(temp_dir, f"{os.path.basename(passlog_file)}*")
        for f in glob.glob(log_pattern):
            try:
                os.remove(f)
            except:
                pass

def filter_frame_downsample(image: np.ndarray, frame_scores: np.ndarray, block_size: int) -> Tuple[np.ndarray, np.ndarray]:
    """Adaptively downsample each block based on removability scores. Returns (image, downsample_maps)."""
    blocks = split_image_into_blocks(image, block_size)

    # Map removability scores to downsampling factors (powers of 2, from 1 to block_size)
    downsample_maps = np.round(frame_scores * (int(np.log2(block_size)))).astype(np.int32)
    downsample_strengths = np.power(2.0, downsample_maps).astype(np.float32)

    # Downsample each block based on its strength - optimized with vectorization where possible
    processed_blocks = blocks.copy()  # Start with copy of original blocks
    num_blocks_y, num_blocks_x = blocks.shape[0], blocks.shape[1]
    
    # Only process blocks that need downsampling (strength > 1)
    for by in range(num_blocks_y):
        for bx in range(num_blocks_x):
            strength = downsample_strengths[by, bx]
            if strength > 1:
                block = blocks[by, bx]
                # Downscale
                small_size = max(1, int(block_size / strength))
                small_block = cv2.resize(block, (small_size, small_size), interpolation=cv2.INTER_AREA)
                # Upscale back to original block size
                upsampled_block = cv2.resize(small_block, (block_size, block_size), interpolation=cv2.INTER_LINEAR)
                processed_blocks[by, bx] = upsampled_block

    # Reconstruct the final image from the processed blocks
    new_image = combine_blocks_into_image(processed_blocks)

    return new_image, downsample_maps

def filter_frame_gaussian(image: np.ndarray, frame_scores: np.ndarray, block_size: int) -> Tuple[np.ndarray, np.ndarray]:
    """Apply adaptive Gaussian blur per block based on scores. Returns (image, blur_strengths)."""
    blocks = split_image_into_blocks(image, block_size)

    # Map removability scores to blurring rounds (0 to 10)
    blur_strengths = np.round(frame_scores * 10).astype(np.int32)

    # Apply multiple rounds of Gaussian blur to each block based on its strength - optimized
    processed_blocks = blocks.copy()  # Start with copy of original blocks
    num_blocks_y, num_blocks_x = blocks.shape[0], blocks.shape[1]
    
    # Only process blocks that need blurring (strength > 0)
    for by in range(num_blocks_y):
        for bx in range(num_blocks_x):
            strength = blur_strengths[by, bx]
            if strength > 0:
                block = blocks[by, bx]
                blurred_block = block.copy()
                for _ in range(strength):
                    blurred_block = cv2.GaussianBlur(blurred_block, (5, 5), sigmaX=1.0)
                processed_blocks[by, bx] = blurred_block

    # Reconstruct the final image from the processed blocks
    new_image = combine_blocks_into_image(processed_blocks)

    return new_image, blur_strengths

def encode_strength_maps(strength_maps: np.ndarray, output_video: str, framerate: float, target_bitrate: int = 50000) -> None:
    """Encode strength maps as grayscale video. Maps normalized to 0-255 for encoding."""
    min_val = np.min(strength_maps)
    max_val = np.max(strength_maps)
    normalized_maps = ((strength_maps - min_val) / (max_val - min_val) * 255.0).astype(np.uint8)

    # Save normalized maps as PNG frames in folder with same name as output video
    frames_dir = os.path.splitext(output_video)[0]
    os.makedirs(frames_dir, exist_ok=True)
    for i in range(normalized_maps.shape[0]):
        map_img = normalized_maps[i]
        cv2.imwrite(os.path.join(frames_dir, f"{i+1:05d}.png"), map_img)

    # Encode frames to video
    encode_video(
        input_frames_dir=frames_dir,
        output_video=output_video,
        framerate=framerate,
        width=normalized_maps.shape[2],
        height=normalized_maps.shape[1],
        target_bitrate=target_bitrate,
        pix_fmt="gray"
    )

def decode_strength_maps(video_path: str, block_size: int, frames_dir: str) -> np.ndarray:
    """Decode strength maps from compressed video. Returns 3D array (frames, blocks_y, blocks_x)."""
    if "gaussian" in video_path:
        min_val, max_val = 0.0, 10.0
    elif "downsample" in video_path:
        min_val, max_val = 0.0, int(np.log2(block_size))

    # Decode video to frames in the specified directory
    os.makedirs(frames_dir, exist_ok=True)
    decode_video(video_path, frames_dir, quality=1)

    # Load frames and reconstruct strength maps
    frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith(('.png', '.jpg'))])
    strength_maps = []
    for frame_file in frame_files:
        img = cv2.imread(os.path.join(frames_dir, frame_file), cv2.IMREAD_GRAYSCALE)
        # Normalize back to original strength range
        strength_map = img.astype(np.float32) / 255.0 * (max_val - min_val) + min_val
        # Round to nearest possible value
        strength_map = np.round(strength_map).astype(np.uint8)
        strength_maps.append(strength_map)

    # Stack all strength maps into a 3D array
    return np.stack(strength_maps, axis=0)

def encode_strength_maps_to_npz(strength_maps: np.ndarray, output_path: str) -> None:
    """Encode strength maps as compressed .npz file. Stored as uint8 for minimal size."""
    if isinstance(strength_maps, list):
        strength_maps = np.stack(strength_maps, axis=0)
    
    # Convert to uint8 if not already (strength maps are typically integers in range 0-10)
    if strength_maps.dtype != np.uint8:
        strength_maps = strength_maps.astype(np.uint8)
    
    # Save with compression
    np.savez_compressed(output_path, strength_maps=strength_maps)
    
    print(f"  Strength maps saved to {output_path} ({os.path.getsize(output_path) / 1024:.2f} KB)")

def decode_strength_maps_from_npz(npz_path: str) -> np.ndarray:
    """Decode strength maps from .npz file. Returns 3D array (frames, blocks_y, blocks_x)."""
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"Strength maps file not found: {npz_path}")
    
    # Load the compressed array
    data = np.load(npz_path)
    strength_maps = data['strength_maps']
    
    print(f"  Strength maps loaded from {npz_path} ({os.path.getsize(npz_path) / 1024:.2f} KB)")
    
    return strength_maps

def upscale_realesrgan_2x(image: np.ndarray, realesrgan_dir: str = None, temp_dir: str = None) -> np.ndarray:
    """Apply Real-ESRGAN 2x upscaling. Returns upscaled image (2*H, 2*W, C) in BGR format."""
    cleanup_temp = False
    if temp_dir is None:
        temp_dir = tempfile.mkdtemp()
        cleanup_temp = True
    
    # Save current directory so we can restore it if we cd into a local repo
    original_dir = os.getcwd()

    try:
        temp_dir_abs = os.path.abspath(temp_dir)

        # Create input/output directories
        input_dir = os.path.join(temp_dir_abs, "input")
        output_dir = os.path.join(temp_dir_abs, "output")
        os.makedirs(input_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)
        
        # Save input image
        input_path = os.path.join(input_dir, "input.png")
        cv2.imwrite(input_path, image)
        
        run_args = None
        use_package = False

        # Prefer the installed package entry point when available
        if realesrgan_dir is None:
            try:
                import importlib

                importlib.import_module("realesrgan.entrypoints")
                use_package = True
            except ImportError:
                use_package = False

        if use_package:
            run_args = [
                sys.executable,
                "-m",
                "realesrgan.entrypoints",
                "-n",
                "RealESRGAN_x4plus",
                "-i",
                input_dir,
                "-o",
                output_dir,
                "-s",
                "2",
                "--suffix",
                "out",
                "--ext",
                "png",
            ]
        else:
            if realesrgan_dir is None:
                raise RuntimeError(
                    "The 'realesrgan' package is not installed and no local realesrgan_dir was provided."
                )
            realesrgan_dir_abs = os.path.abspath(realesrgan_dir)
            inference_script = os.path.join(realesrgan_dir_abs, "inference_realesrgan.py")

            if not os.path.exists(inference_script):
                raise FileNotFoundError(f"Real-ESRGAN inference script not found at: {inference_script}")

            os.chdir(realesrgan_dir_abs)
            run_args = [
                sys.executable,
                inference_script,
                "-n",
                "RealESRGAN_x4plus",
                "-i",
                input_dir,
                "-o",
                output_dir,
                "-s",
                "2",
                "--suffix",
                "out",
                "--ext",
                "png",
            ]

        # Run the inference entry point
        result = subprocess.run(run_args, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise RuntimeError(f"Real-ESRGAN failed: {result.stderr}\nStdout: {result.stdout}")
        
        # Load upscaled image (output will be named input_out.png)
        output_path = os.path.join(output_dir, "input_out.png")
        
        if not os.path.exists(output_path):
            raise RuntimeError(f"Failed to find upscaled image at {output_path}")
        
        upscaled = cv2.imread(output_path)
        
        if upscaled is None:
            raise RuntimeError(f"Failed to read upscaled image from {output_path}")
        
        return upscaled
        
    finally:
        # Return to original directory
        os.chdir(original_dir)
        
        # Cleanup temp directory if we created it
        if cleanup_temp:
            shutil.rmtree(temp_dir, ignore_errors=True)

def _instantiate_realesrgan_upsampler(
    model_name: str,
    device: torch.device,
    *,
    denoise_strength: float = 1.0,
    tile: int = 0,
    tile_pad: int = 10,
    pre_pad: int = 0,
    fp32: bool = False,
) -> "RealESRGANer":
    """Create and warm a Real-ESRGAN upsampler on the specified device."""

    try:
        import realesrgan  # type: ignore
        from realesrgan.utils import RealESRGANer  # type: ignore
        from realesrgan.archs.srvgg_arch import SRVGGNetCompact  # type: ignore
        from realesrgan.inference import (  # type: ignore
            DEFAULT_RELEASE_SUBDIR,
            DEFAULT_WEIGHTS_SUBDIR,
            _resolve_existing_model_path,
        )
        from basicsr.archs.rrdbnet_arch import RRDBNet  # type: ignore
        from basicsr.utils.download_util import load_file_from_url  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "Real-ESRGAN python package with its dependencies is required. "
            "Install it with `pip install realesrgan basicsr`."
        ) from exc

    model_name = model_name.split('.')[0]

    if model_name == 'RealESRGAN_x4plus':
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        netscale = 4
        file_urls = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth']
    elif model_name == 'RealESRNet_x4plus':
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        netscale = 4
        file_urls = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth']
    elif model_name == 'RealESRGAN_x4plus_anime_6B':
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
        netscale = 4
        file_urls = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth']
    elif model_name == 'RealESRGAN_x2plus':
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
        netscale = 2
        file_urls = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth']
    elif model_name == 'realesr-animevideov3':
        model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4, act_type='prelu')
        netscale = 4
        file_urls = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth']
    elif model_name == 'realesr-general-x4v3':
        model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu')
        netscale = 4
        file_urls = [
            'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-wdn-x4v3.pth',
            'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth',
        ]
    else:
        raise ValueError(f"Unsupported Real-ESRGAN model '{model_name}'.")

    package_dir = Path(realesrgan.__file__).resolve().parent
    release_dir = package_dir / DEFAULT_RELEASE_SUBDIR
    weights_dir = package_dir / DEFAULT_WEIGHTS_SUBDIR
    cwd_weights_dir = Path.cwd() / DEFAULT_WEIGHTS_SUBDIR
    search_dirs: List[Path] = [release_dir, weights_dir, cwd_weights_dir]

    existing = _resolve_existing_model_path(model_name, search_dirs)
    if existing is None:
        weights_dir.mkdir(parents=True, exist_ok=True)
        root_dir = package_dir
        for url in file_urls:
            load_file_from_url(
                url=url,
                model_dir=os.path.join(root_dir, DEFAULT_WEIGHTS_SUBDIR),
                progress=True,
                file_name=None,
            )
        existing = _resolve_existing_model_path(model_name, search_dirs)
        if existing is None:
            raise RuntimeError(f"Unable to locate Real-ESRGAN weights for model '{model_name}'.")
        if model_name == 'realesr-general-x4v3':
            # Ensure the WDN variant is also available for DNI support
            wdn_existing = _resolve_existing_model_path('realesr-general-wdn-x4v3', search_dirs)
            if wdn_existing is None:
                raise RuntimeError("Missing realesr-general-wdn-x4v3 weights required for DNI mode.")
    resolved_model_path: Union[str, List[str]] = str(existing)

    dni_weight: Optional[List[float]] = None
    if model_name == 'realesr-general-x4v3' and not math.isclose(denoise_strength, 1.0):
        wdn_path = _resolve_existing_model_path('realesr-general-wdn-x4v3', search_dirs)
        if wdn_path is None:
            raise RuntimeError("Unable to locate realesr-general-wdn-x4v3 weights for DNI upsampling.")
        resolved_model_path = [resolved_model_path, str(wdn_path)]
        dni_weight = [denoise_strength, 1 - denoise_strength]

    half_precision = device.type == 'cuda' and not fp32

    upsampler = RealESRGANer(
        scale=netscale,
        model_path=resolved_model_path,
        dni_weight=dni_weight,
        model=model,
        tile=tile,
        tile_pad=tile_pad,
        pre_pad=pre_pad,
        half=half_precision,
        device=device,
    )
    return upsampler


def _device_slot_key(device_obj: torch.device, slot_id: int) -> str:
    idx = device_obj.index if device_obj.type == 'cuda' and device_obj.index is not None else None
    base = f"cuda:{idx}" if idx is not None else str(device_obj)
    return base if slot_id <= 0 else f"{base}#{slot_id}"


def _format_device_slot(device_obj: torch.device, slot_id: int) -> str:
    key = _device_slot_key(device_obj, slot_id)
    return key


def _upsample_with_realesrgan(
    upsampler: "RealESRGANer",
    image: np.ndarray,
    *,
    device_obj: Optional[torch.device] = None,
    outscale: float = 2.0,
) -> np.ndarray:
    try:
        output, _ = upsampler.enhance(image, outscale=outscale)
        return output
    except RuntimeError as exc:
        device_label = str(device_obj) if device_obj is not None else 'unknown device'
        raise RuntimeError(f"Real-ESRGAN failed on {device_label}: {exc}") from exc


def upscale_realesrgan_adaptive(
    downsampled_image: np.ndarray,
    downscale_maps: np.ndarray,
    block_size: int,
    realesrgan_dir: str = None,
    *,
    upsample_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
) -> np.ndarray:
    """
    Applies adaptive Real-ESRGAN upscaling to restore an image where different blocks
    were downsampled by different factors (powers of 2).

    The algorithm works in multiple stages:
    1. Find max downscaling factor and downscale image to that resolution.
    2. Apply Real-ESRGAN 2x to entire frame and update maps.
    3. Restore blocks that were originally downsampled by a factor smaller or equal to the current stage.
    4. Repeat for next stage until full resolution is reached and all blocks are restored.
    
    This allows blocks to see their neighbors during upscaling for proper context,
    while avoiding applying unnecessary upscaling artifacts to higher-quality blocks.
    
    Args:
        downsampled_image: The downsampled image (non-uniform block sizes) in BGR format
        downscale_maps: 2D array (num_blocks_y, num_blocks_x) indicating the downscale factor applied to each block.
        block_size: The side length of each block in the original resolution
        realesrgan_dir: Path to Real-ESRGAN directory
        upsample_fn: Optional callable performing a single 2x upscale. Defaults to calling upscale_realesrgan_2x.
    
    Returns:
        The adaptively upscaled image at original resolution
    """

    if upsample_fn is None:
        upsample_fn = functools.partial(upscale_realesrgan_2x, realesrgan_dir=realesrgan_dir)

    # Convert upscale factors into upscale values (1, 2, 4, 8, ...)
    downscale_maps = np.power(2, downscale_maps).astype(np.int32)
    
    # Find max downscaling factor
    max_factor = int(downscale_maps.max())

    # Downscale the image to the lowest resolution
    height, width, _ = downsampled_image.shape
    current_image = cv2.resize(downsampled_image, (width // max_factor, height // max_factor), interpolation=cv2.INTER_AREA)

    # Process in stages, doubling resolution each time
    num_blocks_y, num_blocks_x = downscale_maps.shape
    current_factor = max_factor / 2
    while current_factor >= 1:

        current_block_size = block_size // int(current_factor)

        # 1. Apply Real-ESRGAN 2x to the entire current image
        current_image = upsample_fn(current_image)

        # 2. Restore blocks that were downsampled by <= current_factor
        blocks = split_image_into_blocks(current_image, current_block_size)

        # Downscale original image and split into blocks for restoration
        downscaled_image = cv2.resize(downsampled_image, (current_image.shape[1], current_image.shape[0]), interpolation=cv2.INTER_AREA)
        downsampled_blocks = split_image_into_blocks(downscaled_image, current_block_size) # Original blocks at current resolution

        for i in range(num_blocks_y):
            for j in range(num_blocks_x):
                block_factor = downscale_maps[i, j]
                if block_factor <= current_factor:
                    # This block was downsampled by <= current_factor, so restore it to original
                    blocks[i, j] = downsampled_blocks[i, j]
                else:
                    # Update downscale map for next iteration
                    downscale_maps[i, j] = current_factor

        # Combine blocks back into the current image
        current_image = combine_blocks_into_image(blocks)
        
        # Update current factor (halve it)
        current_factor /= 2

    return current_image


# ---------------------------------------------------------------------------
# RealESRGAN Upsampler Cache - Thread-safe singleton per device
# ---------------------------------------------------------------------------

_REALESRGAN_UPSAMPLER_CACHE: Dict[str, "RealESRGANer"] = {}
_REALESRGAN_UPSAMPLER_LOCK = threading.Lock()


def get_realesrgan_upsampler(
    device: torch.device,
    *,
    model_name: str = 'RealESRGAN_x4plus',
    denoise_strength: float = 1.0,
    tile: int = 0,
    tile_pad: int = 10,
    pre_pad: int = 0,
    fp32: bool = False,
) -> "RealESRGANer":
    """Get or create a cached RealESRGAN upsampler for the given device."""
    key = f"{device}_{model_name}_{denoise_strength}_{tile}_{tile_pad}_{pre_pad}_{fp32}"
    with _REALESRGAN_UPSAMPLER_LOCK:
        upsampler = _REALESRGAN_UPSAMPLER_CACHE.get(key)
        if upsampler is None:
            _safe_print(f"    -> Warming Real-ESRGAN runtime on {device}...")
            upsampler = _instantiate_realesrgan_upsampler(
                model_name=model_name,
                device=device,
                denoise_strength=denoise_strength,
                tile=tile,
                tile_pad=tile_pad,
                pre_pad=pre_pad,
                fp32=fp32,
            )
            _REALESRGAN_UPSAMPLER_CACHE[key] = upsampler
    return upsampler


def restore_frames_realesrgan(
    frames: List[np.ndarray],
    downscale_maps: np.ndarray,
    block_size: int,
    device: torch.device,
    *,
    model_name: str = 'RealESRGAN_x4plus',
    denoise_strength: float = 1.0,
    tile: int = 0,
    tile_pad: int = 10,
    pre_pad: int = 0,
    fp32: bool = False,
) -> List[np.ndarray]:
    """
    Pure restoration function: restore frames using RealESRGAN.
    
    Takes frames and downscale maps as numpy arrays, returns restored frames.
    No file IO, no parallelization - just core restoration logic.
    """
    upsampler = get_realesrgan_upsampler(
        device,
        model_name=model_name,
        denoise_strength=denoise_strength,
        tile=tile,
        tile_pad=tile_pad,
        pre_pad=pre_pad,
        fp32=fp32,
    )
    
    def _enhance_once(img: np.ndarray) -> np.ndarray:
        return _upsample_with_realesrgan(upsampler, img, device_obj=device, outscale=2.0)
    
    restored_frames = []
    for idx, frame in enumerate(frames):
        restored = upscale_realesrgan_adaptive(
            frame,
            downscale_maps[idx],
            block_size,
            upsample_fn=_enhance_once,
        )
        restored_frames.append(restored)
    
    return restored_frames


def restore_downsampled_with_realesrgan(
    input_frames_dir: str,
    output_frames_dir: str,
    downscale_maps: np.ndarray,
    block_size: int,
    *,
    model_name: str = 'RealESRGAN_x4plus',
    denoise_strength: float = 1.0,
    tile: int = 0,
    tile_pad: int = 10,
    pre_pad: int = 0,
    fp32: bool = False,
    devices: Optional[Sequence[Union[int, str, torch.device]]] = None,
    parallel_chunk_length: Optional[int] = None,
    per_device_workers: int = 1,
) -> None:
    """Parallel adaptive Real-ESRGAN restoration over a directory of frames."""
    
    # --- IO: Load inputs ---
    frame_paths = get_frame_paths(input_frames_dir)
    if not frame_paths:
        raise ValueError(f"No frames found in {input_frames_dir}")
    
    downscale_maps = np.asarray(downscale_maps)
    num_frames = len(frame_paths)
    if downscale_maps.shape[0] != num_frames:
        raise ValueError(
            f"Downscale maps length ({downscale_maps.shape[0]}) does not match frame count ({num_frames})."
        )
    
    # --- Prepare output directory ---
    clear_directory(output_frames_dir)
    os.makedirs(output_frames_dir, exist_ok=True)
    
    # --- Resolve devices ---
    resolved_devices = _resolve_device_list(devices, prefer_cuda=True, allow_cpu_fallback=True)
    
    device_summary = ", ".join(str(dev) for dev in resolved_devices)
    tile_desc = str(tile) if tile and tile > 0 else 'full-frame'
    _safe_print(f"  Using Real-ESRGAN on devices: {device_summary} | tile: {tile_desc}")
    _safe_print(f"  Total frames: {num_frames}")
    
    # --- Create processing function for parallel_process_frames ---
    def process_chunk(chunk_frames: List[np.ndarray], device: torch.device) -> List[np.ndarray]:
        return restore_frames_realesrgan(
            chunk_frames,
            downscale_maps[:len(chunk_frames)],  # Slice will be adjusted below
            block_size,
            device,
            model_name=model_name,
            denoise_strength=denoise_strength,
            tile=tile,
            tile_pad=tile_pad,
            pre_pad=pre_pad,
            fp32=fp32,
        )
    
    # --- Chunk and process in parallel using our utility ---
    chunks = chunk_for_devices(num_frames, resolved_devices)
    
    all_restored: List[np.ndarray] = []
    for chunk in chunks:
        chunk_frames = [load_frame(str(frame_paths[i])) for i in range(chunk.start, chunk.end)]
        chunk_maps = downscale_maps[chunk.start:chunk.end]
        
        _safe_print(f"    -> Real-ESRGAN frames {chunk.start + 1}-{chunk.end} on {chunk.device}")
        
        restored = restore_frames_realesrgan(
            chunk_frames,
            chunk_maps,
            block_size,
            chunk.device,
            model_name=model_name,
            denoise_strength=denoise_strength,
            tile=tile,
            tile_pad=tile_pad,
            pre_pad=pre_pad,
            fp32=fp32,
        )
        all_restored.extend(restored)
    
    # --- IO: Save outputs ---
    for idx, restored_frame in enumerate(all_restored):
        output_path = os.path.join(output_frames_dir, frame_paths[idx].name)
        save_frame(restored_frame, output_path)

# OpenCV-based restoration benchmarks for Elvis v2

def restore_downsample_opencv_lanczos(downsampled_image: np.ndarray, downscale_maps: np.ndarray, block_size: int) -> np.ndarray:
    """
    Restores a downsampled image using OpenCV's Lanczos interpolation.
    This is a simple client-side restoration benchmark that doesn't require any ML models.
    Lanczos generally provides better quality than bilinear/bicubic for upscaling.
    
    Args:
        downsampled_image: The downsampled image (non-uniform block sizes) in BGR format
        downscale_maps: 2D array (num_blocks_y, num_blocks_x) indicating the downscale factor applied to each block
        block_size: The side length of each block in the original resolution
    
    Returns:
        The restored image at original resolution using Lanczos upscaling
    """
    # Convert downscale maps to actual downscale factors (powers of 2)
    downscale_factors = np.power(2, downscale_maps).astype(np.int32)
    
    # Find max downscaling factor
    max_factor = int(downscale_factors.max())
    
    if max_factor == 1:
        # No downsampling was applied
        return downsampled_image
    
    # Split image into blocks at current resolution
    height, width, _ = downsampled_image.shape
    num_blocks_y, num_blocks_x = downscale_maps.shape
    blocks = split_image_into_blocks(downsampled_image, block_size)
    
    # Upscale each block individually using Lanczos interpolation
    restored_blocks = np.zeros_like(blocks)
    for i in range(num_blocks_y):
        for j in range(num_blocks_x):
            factor = downscale_factors[i, j]
            if factor > 1:
                # Block was downsampled, upscale it
                block = blocks[i, j]
                # Downscale to simulate the degraded version
                small_size = max(1, block_size // factor)
                small_block = cv2.resize(block, (small_size, small_size), interpolation=cv2.INTER_AREA)
                # Upscale back using Lanczos interpolation
                restored_block = cv2.resize(small_block, (block_size, block_size), interpolation=cv2.INTER_LANCZOS4)
                restored_blocks[i, j] = restored_block
            else:
                # Block was not downsampled
                restored_blocks[i, j] = blocks[i, j]
    
    return combine_blocks_into_image(restored_blocks)

def restore_blur_opencv_unsharp_mask(blurred_image: np.ndarray, blur_maps: np.ndarray, block_size: int) -> np.ndarray:
    """
    Restores a blurred image using OpenCV's unsharp masking technique.
    This is a simple client-side restoration benchmark that doesn't require any ML models.
    
    Unsharp masking works by:
    1. Blurring the image
    2. Subtracting the blurred version from the original to get high-frequency details
    3. Adding these details back to enhance sharpness
    
    Args:
        blurred_image: The blurred image in BGR format
        blur_maps: 2D array (num_blocks_y, num_blocks_x) indicating blur rounds applied to each block
        block_size: The side length of each block
    
    Returns:
        The restored image with adaptive unsharp masking applied
    """
    # Split image into blocks
    num_blocks_y, num_blocks_x = blur_maps.shape
    blocks = split_image_into_blocks(blurred_image, block_size)
    
    # Apply unsharp mask to each block based on its blur strength
    restored_blocks = np.zeros_like(blocks)
    for i in range(num_blocks_y):
        for j in range(num_blocks_x):
            block = blocks[i, j]
            blur_strength = int(blur_maps[i, j])
            
            if blur_strength > 0:
                # Apply unsharp mask with strength proportional to blur
                # Parameters: amount controls strength, radius controls blur size
                amount = blur_strength * 0.5  # Sharpening strength
                radius = max(1, blur_strength)  # Blur radius
                
                # Create blurred version
                blurred = cv2.GaussianBlur(block, (0, 0), radius)
                # Calculate high-frequency details
                sharpened = cv2.addWeighted(block, 1.0 + amount, blurred, -amount, 0)
                # Clip values to valid range
                restored_blocks[i, j] = np.clip(sharpened, 0, 255).astype(np.uint8)
            else:
                # No blurring was applied
                restored_blocks[i, j] = block
    
    return combine_blocks_into_image(restored_blocks)

def _instantir_chunk_worker(
    frames_dir: str,
    frame_names: Sequence[str],
    blur_maps: np.ndarray,
    block_size: int,
    weights_dir: str,
    cfg: float,
    creative_start: float,
    preview_start: float,
    batch_size: int,
    device_str: str,
    seed: Optional[int],
    chunk_index: int,
    total_chunks: int,
    global_start: int,
    global_end: int,
) -> None:
    """Worker entry point that restores a contiguous frame chunk on a single device."""

    device = torch.device(device_str)

    if device.type == "cuda":
        torch.cuda.set_device(device)

    if seed is not None:
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)

    weights_path = Path(weights_dir).expanduser()
    frame_paths = [os.path.join(frames_dir, name) for name in frame_names]

    chunk_frames: List[np.ndarray] = []
    original_blocks: List[np.ndarray] = []
    for path in frame_paths:
        frame = cv2.imread(path, cv2.IMREAD_COLOR)
        if frame is None:
            raise RuntimeError(f"Failed to load frame for InstantIR restoration: {path}")
        chunk_frames.append(frame)
        original_blocks.append(split_image_into_blocks(frame, block_size))

    if not chunk_frames:
        return

    chunk_original_blocks = np.stack(original_blocks, axis=0)
    chunk_blur_maps = np.asarray(blur_maps, dtype=np.int32).copy()
    local_length = len(chunk_frames)
    max_rounds = int(chunk_blur_maps.max()) if chunk_blur_maps.size else 0

    device_label = device_str
    _safe_print(
        f"    -> InstantIR chunk {chunk_index + 1}/{total_chunks} "
        f"frames {global_start + 1}-{global_end} on {device_label} (max rounds: {max_rounds})"
    )

    runtime: Optional[InstantIRRuntime] = None
    try:
        dtype = torch.float16 if device.type == "cuda" else torch.float32
        with _silence_console_output():
            runtime = load_runtime(
                instantir_path=weights_path,
                device=device,
                torch_dtype=dtype,
                map_location="cpu",
            )
        if hasattr(runtime, "pipe") and hasattr(runtime.pipe, "set_progress_bar_config"):
            runtime.pipe.set_progress_bar_config(disable=True)

        if max_rounds <= 0:
            _safe_print(
                f"       No blur detected for chunk {chunk_index + 1}; skipping restoration."
            )
        else:
            def _iter_batches(indices: Sequence[int], batch_len: int) -> Iterator[List[int]]:
                step = max(1, batch_len)
                for offset in range(0, len(indices), step):
                    yield list(indices[offset:offset + step])

            for round_idx in range(max_rounds):
                active_indices = [idx for idx in range(local_length) if np.any(chunk_blur_maps[idx] > 0)]
                if not active_indices:
                    break

                _safe_print(
                    f"       Round {round_idx + 1}/{max_rounds}: processing {len(active_indices)} frame(s)"
                )

                for batch_indices in _iter_batches(active_indices, batch_size):
                    pil_batch = []
                    for local_idx in batch_indices:
                        frame_rgb = cv2.cvtColor(chunk_frames[local_idx], cv2.COLOR_BGR2RGB)
                        pil_batch.append(Image.fromarray(frame_rgb))

                    with _silence_console_output():
                        restored_pils = restore_images_batch(
                            runtime,
                            pil_batch,
                            num_inference_steps=1,
                            cfg=cfg,
                            preview_start=preview_start,
                            creative_start=creative_start,
                        )

                    for local_idx, restored_pil in zip(batch_indices, restored_pils):
                        restored_bgr = cv2.cvtColor(np.array(restored_pil), cv2.COLOR_RGB2BGR)
                        restored_blocks = split_image_into_blocks(restored_bgr, block_size)
                        completed_mask = chunk_blur_maps[local_idx] <= 0
                        if np.any(completed_mask):
                            restored_blocks[completed_mask] = chunk_original_blocks[local_idx][completed_mask]
                        chunk_frames[local_idx] = combine_blocks_into_image(restored_blocks)

                positive_mask = chunk_blur_maps > 0
                chunk_blur_maps[positive_mask] -= 1

        for path, frame in zip(frame_paths, chunk_frames):
            if not cv2.imwrite(path, frame):
                raise RuntimeError(f"Failed to write restored frame: {path}")

    finally:
        if runtime is not None:
            del runtime
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass
        gc.collect()


def restore_with_instantir_adaptive(
    input_frames_dir: str,
    blur_maps: np.ndarray,
    block_size: int,
    cfg: float = 7.0,
    creative_start: float = 1.0,
    preview_start: float = 0.0,
    seed: Optional[int] = 42,
    devices: Optional[Sequence[Union[int, str, torch.device]]] = None,
    batch_size: int = 4,
    parallel_chunk_length: Optional[int] = None,
) -> None:
    """Apply adaptive InstantIR blind restoration with simple per-device chunking."""

    _ = parallel_chunk_length  # Retained for backwards compatibility; no longer used.

    if batch_size < 1:
        raise ValueError("`batch_size` must be at least 1.")

    _safe_print("  Preparing InstantIR workers...")

    weights_dir = Path("./InstantIR/models").expanduser()
    weights_dir.mkdir(parents=True, exist_ok=True)

    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    resolved_devices = _resolve_device_list(devices, prefer_cuda=True, allow_cpu_fallback=True)
    cuda_devices = [dev for dev in resolved_devices if dev.type == "cuda"]
    worker_devices = cuda_devices if cuda_devices else [resolved_devices[0]]

    frames_files = sorted([f for f in os.listdir(input_frames_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))])
    num_frames = len(frames_files)
    if num_frames == 0:
        raise ValueError(f"No frames found in {input_frames_dir}")
    if num_frames != blur_maps.shape[0]:
        raise ValueError(
            f"Number of frames ({num_frames}) doesn't match blur_maps shape ({blur_maps.shape[0]})"
        )

    if np.max(blur_maps) == 0:
        _safe_print("  No blurring detected, skipping restoration.")
        return

    def _split_ranges(total: int, parts: int) -> List[Tuple[int, int]]:
        if parts <= 0:
            return [(0, total)]
        base = total // parts
        remainder = total % parts
        ranges: List[Tuple[int, int]] = []
        start = 0
        for idx in range(parts):
            length = base + (1 if idx < remainder else 0)
            end = start + length
            ranges.append((start, end))
            start = end
        return ranges

    initial_ranges = _split_ranges(num_frames, len(worker_devices))

    jobs: List[Dict[str, Any]] = []
    for device, (start, end) in zip(worker_devices, initial_ranges):
        if start >= end:
            continue
        frames_subset = frames_files[start:end]
        jobs.append(
            {
                "device": device,
                "start": start,
                "end": end,
                "frames": frames_subset,
                "blur": np.array(blur_maps[start:end], copy=True),
            }
        )

    if not jobs:
        _safe_print("  No frame chunks were assigned; skipping InstantIR restoration.")
        return

    for idx, job in enumerate(jobs):
        device = job["device"]
        if device.type == "cuda":
            dev_idx = device.index if device.index is not None else 0
            job["device_str"] = f"cuda:{dev_idx}"
        else:
            job["device_str"] = str(device)
        job["chunk_index"] = idx

    device_summary = ", ".join(job["device_str"] for job in jobs)
    _safe_print(
        f"  Using InstantIR on devices: {device_summary} | batch size per device: {batch_size}"
    )

    total_chunks = len(jobs)
    chunk_shapes = ", ".join(
        f"{job['start'] + 1}-{job['end']} ({job['end'] - job['start']} frames)" for job in jobs
    )
    _safe_print(
        f"  Assigned frame spans per worker: {chunk_shapes}"
    )

    if total_chunks == 1:
        job = jobs[0]
        worker_seed = seed
        _instantir_chunk_worker(
            input_frames_dir,
            job["frames"],
            job["blur"],
            block_size,
            str(weights_dir),
            cfg,
            creative_start,
            preview_start,
            batch_size,
            job["device_str"],
            worker_seed,
            job["chunk_index"],
            total_chunks,
            job["start"],
            job["end"],
        )
    else:
        ctx = multiprocessing.get_context("spawn")
        processes: List[multiprocessing.Process] = []
        for job in jobs:
            worker_seed = (seed + job["chunk_index"]) if (seed is not None) else None
            proc = ctx.Process(
                target=_instantir_chunk_worker,
                args=(
                    input_frames_dir,
                    job["frames"],
                    job["blur"],
                    block_size,
                    str(weights_dir),
                    cfg,
                    creative_start,
                    preview_start,
                    batch_size,
                    job["device_str"],
                    worker_seed,
                    job["chunk_index"],
                    total_chunks,
                    job["start"],
                    job["end"],
                ),
            )
            proc.start()
            processes.append(proc)

        errors: List[int] = []
        for proc in processes:
            proc.join()
            if proc.exitcode not in (0, None):
                errors.append(proc.exitcode)

        if errors:
            raise RuntimeError(f"InstantIR worker(s) exited with non-zero code(s): {errors}")

    _safe_print(f"  Adaptive InstantIR restoration complete. Frames saved to {input_frames_dir}")


def calculate_lpips_per_frame(
    reference_frames: List[np.ndarray],
    decoded_frames: List[np.ndarray],
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> List[float]:
    """Calculate LPIPS over an aligned list of frame pairs."""

    if not reference_frames or not decoded_frames:
        return []

    lpips_model = _get_lpips_model(device)
    model_device = next(lpips_model.parameters()).device

    lpips_scores: List[float] = []

    with torch.no_grad():
        for ref_frame, dec_frame in zip(reference_frames, decoded_frames):
            if ref_frame is None or dec_frame is None:
                continue

            ref_rgb = cv2.cvtColor(ref_frame, cv2.COLOR_BGR2RGB)
            dec_rgb = cv2.cvtColor(dec_frame, cv2.COLOR_BGR2RGB)

            ref_tensor = torch.from_numpy(np.ascontiguousarray(ref_rgb)).permute(2, 0, 1).unsqueeze(0).float()
            dec_tensor = torch.from_numpy(np.ascontiguousarray(dec_rgb)).permute(2, 0, 1).unsqueeze(0).float()

            ref_tensor = ref_tensor.to(model_device) / 127.5 - 1.0
            dec_tensor = dec_tensor.to(model_device) / 127.5 - 1.0

            lpips_score = lpips_model(ref_tensor, dec_tensor).item()
            lpips_scores.append(lpips_score)

    return lpips_scores

def calculate_vmaf(
    reference_video: str,
    distorted_video: str,
    width: int,
    height: int,
    framerate: float,
    model_path: str = None,
    frame_stride: int = 1
) -> Dict[str, float]:
    """
    Calculate VMAF (Video Multimethod Assessment Fusion) using the standalone vmaf command-line tool.
    Automatically converts videos to YUV format if needed.
    
    Args:
        reference_video: Path to the reference video file
        distorted_video: Path to the distorted/encoded video file
        width: Video width
        height: Video height
        framerate: Video framerate
        model_path: Optional path to VMAF model file
    
    Returns:
        Dictionary containing VMAF statistics (mean, min, max, etc.)
    """
    
    # Helper function to convert video to YUV
    def _convert_to_yuv(video_path: str, output_yuv: str, width: int, height: int, stride: int = 1) -> bool:
        """Convert a video to YUV420p format."""
        try:
            vf_filters: List[str] = []
            if stride > 1:
                vf_filters.append(f"select='not(mod(n,{stride}))'")
                vf_filters.append('setpts=N/(FRAME_RATE*TB)')
            vf_filters.append(f'scale={width}:{height}')

            filter_arg = ','.join(vf_filters)
            convert_cmd = [
                'ffmpeg', '-hide_banner', '-loglevel', 'error',
                '-i', video_path,
                '-vf', filter_arg,
                '-pix_fmt', 'yuv420p',
                '-y', output_yuv
            ]
            result = subprocess.run(convert_cmd, capture_output=True, text=True, check=True)
            return True
        except subprocess.CalledProcessError as e:
            print(f"Error converting video to YUV: {e.stderr}")
            return False
    
    try:
        # Check if videos are already in YUV format
        ref_yuv = reference_video
        dist_yuv = distorted_video
        temp_ref_yuv = None
        temp_dist_yuv = None
        
        # Convert reference video if needed
        if not reference_video.endswith('.yuv'):
            temp_ref_yuv = tempfile.NamedTemporaryFile(suffix='.yuv', delete=False)
            temp_ref_yuv.close()
            ref_yuv = temp_ref_yuv.name
            print(f"  - Converting reference video to YUV format...")
            if not _convert_to_yuv(reference_video, ref_yuv, width, height, frame_stride):
                return {'mean': 0, 'min': 0, 'max': 0, 'std': 0, 'harmonic_mean': 0}
        
        # Convert distorted video if needed
        if not distorted_video.endswith('.yuv'):
            temp_dist_yuv = tempfile.NamedTemporaryFile(suffix='.yuv', delete=False)
            temp_dist_yuv.close()
            dist_yuv = temp_dist_yuv.name
            print(f"  - Converting distorted video to YUV format...")
            if not _convert_to_yuv(distorted_video, dist_yuv, width, height, frame_stride):
                return {'mean': 0, 'min': 0, 'max': 0, 'std': 0, 'harmonic_mean': 0}
        
        # Create a temporary file for the JSON output
        with tempfile.NamedTemporaryFile(mode='w+', suffix='.json', delete=False) as temp_file:
            output_json = temp_file.name
        
        # Build the vmaf command
        vmaf_cmd = [
            '/opt/local/bin/vmaf',
            '-r', ref_yuv,
            '-d', dist_yuv,
            '-w', str(width),
            '-h', str(height),
            '-p', '420',
            '-b', '8',
            '--json',
            '-o', output_json
        ]
        
        # Add model path if provided
        if model_path:
            vmaf_cmd.extend(['--model', model_path])
        
        # Run the vmaf command
        result = subprocess.run(vmaf_cmd, capture_output=True, text=True, check=True)
        
        # Read and parse the JSON output
        with open(output_json, 'r') as f:
            vmaf_data = json.load(f)
        
        # Extract VMAF scores from the JSON structure
        if 'frames' in vmaf_data:
            vmaf_scores = [frame['metrics']['vmaf'] for frame in vmaf_data['frames']]
        elif 'pooled_metrics' in vmaf_data:
            pooled = vmaf_data['pooled_metrics']['vmaf']
            return {
                'mean': pooled.get('mean', 0),
                'min': pooled.get('min', 0),
                'max': pooled.get('max', 0),
                'std': pooled.get('stddev', 0),
                'harmonic_mean': pooled.get('harmonic_mean', 0)
            }
        else:
            print(f"Warning: Unexpected VMAF output format for {distorted_video}")
            return {'mean': 0, 'min': 0, 'max': 0, 'std': 0, 'harmonic_mean': 0}
        
        # Calculate statistics
        vmaf_array = np.array(vmaf_scores)
        harmonic_mean = len(vmaf_scores) / np.sum([1.0/max(score, 0.001) for score in vmaf_scores])
        
        # Clean up temp files
        os.unlink(output_json)
        if temp_ref_yuv:
            os.unlink(temp_ref_yuv.name)
        if temp_dist_yuv:
            os.unlink(temp_dist_yuv.name)
        
        return {
            'mean': float(np.mean(vmaf_array)),
            'min': float(np.min(vmaf_array)),
            'max': float(np.max(vmaf_array)),
            'std': float(np.std(vmaf_array)),
            'harmonic_mean': float(harmonic_mean)
        }
        
    except subprocess.CalledProcessError as e:
        print(f"Error running VMAF command: {e.stderr}")
        return {'mean': 0, 'min': 0, 'max': 0, 'std': 0, 'harmonic_mean': 0}
    except Exception as e:
        print(f"Error calculating VMAF: {str(e)}")
        return {'mean': 0, 'min': 0, 'max': 0, 'std': 0, 'harmonic_mean': 0}
    finally:
        # Ensure temp files are cleaned up even if there's an error
        if 'output_json' in locals() and os.path.exists(output_json):
            try:
                os.unlink(output_json)
            except:
                pass
        if 'temp_ref_yuv' in locals() and temp_ref_yuv and os.path.exists(temp_ref_yuv.name):
            try:
                os.unlink(temp_ref_yuv.name)
            except:
                pass
        if 'temp_dist_yuv' in locals() and temp_dist_yuv and os.path.exists(temp_dist_yuv.name):
            try:
                os.unlink(temp_dist_yuv.name)
            except:
                pass

def calculate_fvmd(
    reference_frames: List[np.ndarray],
    decoded_frames: List[np.ndarray],
    log_root: Optional[str] = None,
    stride: int = 1,
    max_frames: Optional[int] = None,
    early_stop_delta: float = 0.002,
    early_stop_window: int = 50,
    device: Optional[int] = None,
    verbose: bool = True,
) -> Tuple[float, float]:
    """Calculate FVMD statistics. Returns tuple of (fvmd_value, std_dev)."""

    printer = _safe_print if verbose else (lambda *args, **kwargs: None)

    if not reference_frames or not decoded_frames:
        raise ValueError("Both reference_frames and decoded_frames must contain at least one frame.")

    total_frames = min(len(reference_frames), len(decoded_frames))
    if total_frames < 2:
        raise ValueError("FVMD requires at least two frames in both reference and decoded sequences.")

    base_stride = max(1, stride)

    def _build_indices(stride_value: int) -> List[int]:
        idxs = list(range(0, total_frames, stride_value))
        if len(idxs) < 2 and total_frames >= 2:
            idxs = [0, total_frames - 1]
        if max_frames is not None and max_frames > 0:
            idxs = idxs[:max_frames]
        unique: List[int] = []
        seen: set[int] = set()
        for idx in idxs:
            if idx not in seen:
                unique.append(idx)
                seen.add(idx)
        return unique

    class _FvmdNoTrajectories(RuntimeError):
        pass

    def _render_frames(frame_indices: Sequence[int], gt_clip: Path, gen_clip: Path) -> None:
        for idx, frame_idx in enumerate(frame_indices, start=1):
            ref_frame = reference_frames[frame_idx]
            dec_frame = decoded_frames[frame_idx]

            if ref_frame is None or dec_frame is None:
                raise ValueError("Frames must not be None when computing FVMD.")

            ref_frame_contig = np.ascontiguousarray(ref_frame)
            dec_frame_contig = np.ascontiguousarray(dec_frame)

            if ref_frame_contig.dtype != np.uint8:
                ref_frame_contig = np.clip(ref_frame_contig, 0, 255).astype(np.uint8)
            if dec_frame_contig.dtype != np.uint8:
                dec_frame_contig = np.clip(dec_frame_contig, 0, 255).astype(np.uint8)

            ref_path = gt_clip / f"{idx:05d}.png"
            dec_path = gen_clip / f"{idx:05d}.png"

            if not cv2.imwrite(str(ref_path), ref_frame_contig):
                raise RuntimeError(f"Failed to write reference frame for FVMD: {ref_path}")
            if not cv2.imwrite(str(dec_path), dec_frame_contig):
                raise RuntimeError(f"Failed to write decoded frame for FVMD: {dec_path}")

    min_required_frames = 10

    def _run_fvmd_once(frame_indices: Sequence[int]) -> float:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            gt_root = tmp_path / "gt"
            gen_root = tmp_path / "gen"
            clip_name = "clip_0001"
            gt_clip = gt_root / clip_name
            gen_clip = gen_root / clip_name
            gt_clip.mkdir(parents=True, exist_ok=True)
            gen_clip.mkdir(parents=True, exist_ok=True)

            _render_frames(frame_indices, gt_clip, gen_clip)

            if log_root is None:
                logs_root_path = tmp_path / "fvmd_logs"
            else:
                logs_root_path = Path(log_root)

            logs_root_path.mkdir(parents=True, exist_ok=True)
            run_log_dir = logs_root_path / f"run_{uuid.uuid4().hex}"
            run_log_dir.mkdir(parents=True, exist_ok=True)

            if len(frame_indices) < min_required_frames:
                raise _FvmdNoTrajectories(
                    f"Only {len(frame_indices)} frame(s) sampled; FVMD requires at least {min_required_frames}."
                )

            clip_seq_len = max(min_required_frames, min(16, len(frame_indices)))

            gen_dataset = VideoDataset(str(gen_root), seq_len=clip_seq_len, stride=1)
            gt_dataset = VideoDataset(str(gt_root), seq_len=clip_seq_len, stride=1)
            if len(gen_dataset) == 0 or len(gt_dataset) == 0:
                raise _FvmdNoTrajectories("Insufficient frames after sampling for FVMD evaluation.")

            if not torch.cuda.is_available():
                raise RuntimeError("FVMD evaluation requires a CUDA-capable GPU, but none were detected.")

            available_gpus = torch.cuda.device_count()
            if device is not None:
                if device < 0 or device >= available_gpus:
                    raise ValueError(f"Requested FVMD device index {device} is out of range (found {available_gpus}).")
                device_ids = [int(device)]
            else:
                device_ids = [0]

            torch.cuda.set_device(device_ids[0])
            device_label = f"cuda:{device_ids[0]}"
            printer(f"    FVMD evaluating {len(frame_indices)} frame(s) on {device_label}")

            try:
                with _silence_console_output():
                    velo_gen, velo_gt, acc_gen, acc_gt = track_keypoints(
                        log_dir=str(run_log_dir),
                        gen_dataset=gen_dataset,
                        gt_dataset=gt_dataset,
                        v_stride=1,
                        S=clip_seq_len,
                        device_ids=device_ids,
                    )
            except RuntimeError as exc:
                raise _FvmdNoTrajectories(str(exc)) from exc

            if any(arr.size == 0 for arr in (velo_gen, velo_gt, acc_gen, acc_gt)):
                raise _FvmdNoTrajectories("FVMD keypoint tracking returned empty trajectories.")

            B = velo_gen.shape[0]
            if B == 0:
                raise _FvmdNoTrajectories("FVMD keypoint tracking produced zero batches.")

            try:
                gt_v_hist = calc_hist(velo_gt).reshape(B, -1)
                gen_v_hist = calc_hist(velo_gen).reshape(B, -1)
                gt_a_hist = calc_hist(acc_gt).reshape(B, -1)
                gen_a_hist = calc_hist(acc_gen).reshape(B, -1)
            except ValueError as exc:
                raise _FvmdNoTrajectories(f"Histogram computation failed: {exc}") from exc

            gt_hist = np.concatenate((gt_v_hist, gt_a_hist), axis=1)
            gen_hist = np.concatenate((gen_v_hist, gen_a_hist), axis=1)

            fvmd_value = calculate_fd_given_vectors(gt_hist, gen_hist)
            if not np.isfinite(fvmd_value):
                raise RuntimeError("FVMD produced a non-finite score.")

        return float(fvmd_value)

    def _compute_std(selected_indices: Sequence[int]) -> float:
        if len(selected_indices) < 2:
            return 0.0

        scores: List[float] = []
        warned = False
        window_span = min(len(selected_indices), max(min_required_frames, 8))
        for start in range(0, len(selected_indices) - window_span + 1):
            window_indices = selected_indices[start : start + window_span]
            try:
                scores.append(_run_fvmd_once(window_indices))
            except _FvmdNoTrajectories as exc:
                if not warned:
                    printer(
                        "  Warning: FVMD could not compute variability for one or more windows; "
                        f"first failure involving frames {window_indices}: {exc}"
                    )
                    warned = True

        if len(scores) <= 1:
            return 0.0

        return float(np.std(scores, ddof=1))

    window = max(1, early_stop_window)
    attempt_stride = base_stride

    while True:
        indices = _build_indices(attempt_stride)
        if len(indices) < min_required_frames:
            if attempt_stride == 1:
                raise ValueError(
                    f"FVMD requires at least {min_required_frames} sampled frames; provide more frames or disable stride."
                )
            next_stride = max(1, attempt_stride // 2)
            if next_stride == attempt_stride:
                next_stride = attempt_stride - 1
            printer(
                f"  Warning: FVMD sampling with stride {attempt_stride} yielded only {len(indices)} frame(s); retrying with stride {next_stride}."
            )
            attempt_stride = next_stride
            continue

        processed = 0
        last_score: Optional[float] = None
        used_indices: Sequence[int] = []

        try:
            while processed < len(indices):
                next_count = min(len(indices), processed + window)
                current_indices = indices[:next_count]
                current_score = _run_fvmd_once(current_indices)
                used_indices = list(current_indices)

                if last_score is not None:
                    baseline = max(abs(last_score), 1e-6)
                    delta = abs(current_score - last_score) / baseline
                    if delta < early_stop_delta:
                        std_value = _compute_std(used_indices)
                        if attempt_stride != base_stride:
                            printer(
                                f"  Info: FVMD used effective stride {attempt_stride} instead of requested {base_stride}."
                            )
                        return current_score, std_value

                last_score = current_score
                processed = next_count

            assert last_score is not None
            std_value = _compute_std(used_indices if used_indices else indices)
            if attempt_stride != base_stride:
                printer(f"  Info: FVMD used effective stride {attempt_stride} instead of requested {base_stride}.")
            return last_score, std_value

        except _FvmdNoTrajectories as exc:
            if attempt_stride == 1:
                raise RuntimeError(
                    "FVMD failed to track keypoints even with stride=1. Consider reviewing the input frames or masks."
                ) from exc

            next_stride = max(1, attempt_stride // 2)
            if next_stride == attempt_stride:
                next_stride = attempt_stride - 1
            printer(
                f"  Warning: FVMD tracking failed with stride {attempt_stride}; retrying with stride {next_stride}."
            )
            attempt_stride = next_stride

def analyze_encoding_performance(
    reference_frames: List[np.ndarray],
    encoded_videos: Dict[str, str],
    block_size: int,
    width: int,
    height: int,
    temp_dir: str,
    masks_dir: str,
    video_bitrates: Dict[str, float] = {},
    framerate: float = 30.0,
    metric_stride: int = 1,
    fvmd_stride: int = 1,
    fvmd_max_frames: Optional[int] = None,
    fvmd_early_stop_delta: float = 0.002,
    fvmd_early_stop_window: int = 50,
    vmaf_stride: int = 1,
    enable_fvmd: bool = True,
) -> Dict:
    """Analyze encoded videos with mask-aware metrics."""

    metric_stride = max(1, metric_stride)
    fvmd_stride = max(1, fvmd_stride)
    vmaf_stride = max(1, vmaf_stride)

    os.makedirs(temp_dir, exist_ok=True)
    masked_videos_dir = os.path.join(temp_dir, "masked_videos")
    fvmd_log_root = os.path.join(temp_dir, "fvmd_logs")
    os.makedirs(masked_videos_dir, exist_ok=True)
    os.makedirs(fvmd_log_root, exist_ok=True)

    if not os.path.isdir(masks_dir):
        print(f"Warning: Masks directory not found at '{masks_dir}'. Cannot perform FG/BG analysis.")
        return {}

    total_reference_frames = len(reference_frames)
    if total_reference_frames == 0:
        print("Warning: No reference frames provided. Skipping analysis.")
        return {}

    fg_masks, bg_masks = _load_resized_masks(masks_dir, width, height, total_reference_frames)
    fg_bbox = _compute_mask_union_bbox(fg_masks, width, height)
    bbox_x, bbox_y, bbox_w, bbox_h = fg_bbox

    y_start = bbox_y
    y_stop = min(height, bbox_y + max(1, bbox_h))
    x_start = bbox_x
    x_stop = min(width, bbox_x + max(1, bbox_w))
    roi_slice = (slice(y_start, y_stop), slice(x_start, x_stop))
    crop_width = max(1, x_stop - x_start)
    crop_height = max(1, y_stop - y_start)
    crop_filter = f"crop={crop_width}:{crop_height}:{x_start}:{y_start}"

    masked_reference_fg_frames = [
        _apply_binary_mask(reference_frames[idx], fg_masks[idx])
        for idx in range(total_reference_frames)
    ]
    masked_reference_bg_frames = [
        _apply_binary_mask(reference_frames[idx], bg_masks[idx])
        for idx in range(total_reference_frames)
    ]

    try:
        mp_ctx = multiprocessing.get_context("spawn")
    except ValueError:
        mp_ctx = multiprocessing.get_context()

    gpu_device_ids: List[Optional[int]] = []
    if enable_fvmd and torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        if gpu_count > 0:
            gpu_device_ids = list(range(gpu_count))
    if not gpu_device_ids:
        gpu_device_ids = [None]

    unique_device_ids: List[Optional[int]] = []
    for device_id in gpu_device_ids:
        if device_id not in unique_device_ids:
            unique_device_ids.append(device_id)

    fvmd_device_locks: Dict[Optional[int], Any] = {}
    for device_id in unique_device_ids:
        fvmd_device_locks[device_id] = mp_ctx.Semaphore(1)

    global _EVALUATION_CONTEXT
    evaluation_context = _EvaluationContext(
        reference_frames=reference_frames,
        masked_reference_fg_frames=masked_reference_fg_frames,
        masked_reference_bg_frames=masked_reference_bg_frames,
        fg_masks=fg_masks,
        bg_masks=bg_masks,
        roi_slice=roi_slice,
        crop_width=crop_width,
        crop_height=crop_height,
        crop_filter=crop_filter,
        framerate=framerate,
        block_size=block_size,
        masked_videos_dir=masked_videos_dir,
        fvmd_log_root=fvmd_log_root,
        enable_fvmd=enable_fvmd,
        fvmd_device_locks=fvmd_device_locks,
    )
    _EVALUATION_CONTEXT = evaluation_context

    analysis_results: Dict[str, Dict[str, Dict[str, float]]] = {}

    videos_to_process: List[Tuple[str, str]] = []
    for video_name, video_path in encoded_videos.items():
        if not os.path.exists(video_path):
            print(f"\nProcessing '{video_name}'...")
            print("  - Video not found, skipping.")
            continue
        videos_to_process.append((video_name, video_path))

    if not videos_to_process:
        print("No encoded videos available for analysis.")
        _EVALUATION_CONTEXT = None
        _FVMD_DEVICE_LOCKS = {}
        return analysis_results

    cpu_count = multiprocessing.cpu_count() if hasattr(multiprocessing, "cpu_count") else os.cpu_count()
    max_workers = min(len(videos_to_process), cpu_count or len(videos_to_process) or 1)

    effective_gpu_workers = len([device for device in gpu_device_ids if device is not None])
    if enable_fvmd and effective_gpu_workers > 0:
        max_workers = min(max_workers, effective_gpu_workers)

    max_workers = max(1, max_workers)

    futures = {}
    with ProcessPoolExecutor(
        max_workers=max_workers,
        mp_context=mp_ctx,
        initializer=_initialise_evaluation_worker,
        initargs=(evaluation_context,),
    ) as executor:
        for idx, (video_name, video_path) in enumerate(videos_to_process):
            fvmd_device = gpu_device_ids[idx % len(gpu_device_ids)] if gpu_device_ids else None
            bitrate = video_bitrates.get(video_name, 0.0)
            future = executor.submit(
                _evaluate_single_video_metrics,
                video_name,
                video_path,
                metric_stride,
                fvmd_stride,
                fvmd_max_frames,
                fvmd_early_stop_delta,
                fvmd_early_stop_window,
                vmaf_stride,
                bitrate,
                fvmd_device,
            )
            futures[future] = video_name

        for future in as_completed(futures):
            video_name = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                raise RuntimeError(f"Error during analysis of '{video_name}'") from exc
            if result is not None:
                analysis_results[video_name] = result

    _EVALUATION_CONTEXT = None
    _FVMD_DEVICE_LOCKS = {}

    if analysis_results:
        _print_summary_report(analysis_results)
    else:
        print("No results to display.")

    print(f"\nAnalysis complete. Masked videos saved to: {masked_videos_dir}")
    return analysis_results

def _evaluate_single_video_metrics(
    video_name: str,
    video_path: str,
    metric_stride: int,
    fvmd_stride: int,
    fvmd_max_frames: Optional[int],
    fvmd_early_stop_delta: float,
    fvmd_early_stop_window: int,
    vmaf_stride: int,
    bitrate_bps: float,
    fvmd_device: Optional[int],
) -> Optional[Dict[str, Dict[str, float]]]:
    """Evaluate quality metrics for a single encoded approach in an isolated process."""

    if _EVALUATION_CONTEXT is None:
        raise RuntimeError("Evaluation context was not initialised before spawning workers.")

    ctx = _EVALUATION_CONTEXT

    print(f"\nProcessing '{video_name}'...")
    if not os.path.exists(video_path):
        print("  - Video not found, skipping.")
        return None

    decoded_frames = _decode_video_to_frames(video_path)
    reference_frames = ctx.reference_frames
    total_reference_frames = len(reference_frames)
    frame_count = min(total_reference_frames, len(decoded_frames))
    if frame_count == 0:
        print("  - No decoded frames available, skipping.")
        return None

    frame_indices = list(range(0, frame_count, metric_stride))
    if not frame_indices:
        frame_indices = [0]
    if frame_indices[-1] != frame_count - 1:
        frame_indices.append(frame_count - 1)
    frame_indices = sorted(set(frame_indices))

    slug = _slugify_name(video_name)
    block_size = ctx.block_size
    fg_masks = ctx.fg_masks
    bg_masks = ctx.bg_masks
    roi_slice = ctx.roi_slice

    masked_reference_fg_frames = ctx.masked_reference_fg_frames[:frame_count]
    masked_reference_bg_frames = ctx.masked_reference_bg_frames[:frame_count]

    masked_decoded_fg_frames = [
        _apply_binary_mask(decoded_frames[idx], fg_masks[idx])
        for idx in range(frame_count)
    ]
    masked_decoded_bg_frames = [
        _apply_binary_mask(decoded_frames[idx], bg_masks[idx])
        for idx in range(frame_count)
    ]

    fg_psnr_vals: List[float] = []
    fg_ssim_vals: List[float] = []
    fg_mse_vals: List[float] = []
    fg_ref_lpips_frames: List[np.ndarray] = []
    fg_dec_lpips_frames: List[np.ndarray] = []
    bg_psnr_vals: List[float] = []
    bg_ssim_vals: List[float] = []
    bg_mse_vals: List[float] = []
    bg_ref_lpips_frames: List[np.ndarray] = []
    bg_dec_lpips_frames: List[np.ndarray] = []

    for idx in frame_indices:
        ref_frame = reference_frames[idx]
        dec_frame = decoded_frames[idx]
        fg_mask = fg_masks[idx]
        bg_mask = bg_masks[idx]

        ref_roi = ref_frame[roi_slice]
        dec_roi = dec_frame[roi_slice]
        fg_mask_roi = fg_mask[roi_slice]

        fg_psnr_vals.append(_masked_psnr(ref_roi, dec_roi, fg_mask_roi))
        fg_ssim_vals.append(_masked_ssim(ref_roi, dec_roi, fg_mask_roi))
        fg_mse_vals.append(_masked_mse(ref_roi, dec_roi, fg_mask_roi))
        fg_ref_lpips_frames.append(masked_reference_fg_frames[idx][roi_slice])
        fg_dec_lpips_frames.append(masked_decoded_fg_frames[idx][roi_slice])

        bg_psnr_vals.append(_masked_psnr(ref_frame, dec_frame, bg_mask))
        bg_ssim_vals.append(_masked_ssim(ref_frame, dec_frame, bg_mask))
        bg_mse_vals.append(_masked_mse(ref_frame, dec_frame, bg_mask))
        bg_ref_lpips_frames.append(masked_reference_bg_frames[idx])
        bg_dec_lpips_frames.append(masked_decoded_bg_frames[idx])

    result: Dict[str, Dict[str, float]] = {
        'foreground': {
            'psnr_mean': float(np.mean(fg_psnr_vals)) if fg_psnr_vals else 0.0,
            'psnr_std': float(np.std(fg_psnr_vals)) if fg_psnr_vals else 0.0,
            'ssim_mean': float(np.mean(fg_ssim_vals)) if fg_ssim_vals else 0.0,
            'ssim_std': float(np.std(fg_ssim_vals)) if fg_ssim_vals else 0.0,
            'mse_mean': float(np.mean(fg_mse_vals)) if fg_mse_vals else 0.0,
            'mse_std': float(np.std(fg_mse_vals)) if fg_mse_vals else 0.0,
        },
        'background': {
            'psnr_mean': float(np.mean(bg_psnr_vals)) if bg_psnr_vals else 0.0,
            'psnr_std': float(np.std(bg_psnr_vals)) if bg_psnr_vals else 0.0,
            'ssim_mean': float(np.mean(bg_ssim_vals)) if bg_ssim_vals else 0.0,
            'ssim_std': float(np.std(bg_ssim_vals)) if bg_ssim_vals else 0.0,
            'mse_mean': float(np.mean(bg_mse_vals)) if bg_mse_vals else 0.0,
            'mse_std': float(np.std(bg_mse_vals)) if bg_mse_vals else 0.0,
        },
        'bitrate_mbps': bitrate_bps / 1_000_000,
    }

    result['foreground']['fvmd'] = float('nan')
    result['foreground']['fvmd_std'] = float('nan')
    result['background']['fvmd'] = float('nan')
    result['background']['fvmd_std'] = float('nan')

    fg_lpips_scores = calculate_lpips_per_frame(fg_ref_lpips_frames, fg_dec_lpips_frames)
    bg_lpips_scores = calculate_lpips_per_frame(bg_ref_lpips_frames, bg_dec_lpips_frames)

    result['foreground']['lpips_mean'] = float(np.mean(fg_lpips_scores)) if fg_lpips_scores else 0.0
    result['foreground']['lpips_std'] = float(np.std(fg_lpips_scores)) if fg_lpips_scores else 0.0
    result['background']['lpips_mean'] = float(np.mean(bg_lpips_scores)) if bg_lpips_scores else 0.0
    result['background']['lpips_std'] = float(np.std(bg_lpips_scores)) if bg_lpips_scores else 0.0

    ref_fg_video_path = os.path.join(ctx.masked_videos_dir, f"{slug}_reference_fg_{frame_count:05d}.mp4")
    if not os.path.exists(ref_fg_video_path):
        _encode_frames_to_video(
            masked_reference_fg_frames,
            ref_fg_video_path,
            ctx.framerate,
            filter_chain=ctx.crop_filter,
            extra_codec_args=['-g', '1'],
        )

    ref_bg_video_path = os.path.join(ctx.masked_videos_dir, f"{slug}_reference_bg_{frame_count:05d}.mp4")
    if not os.path.exists(ref_bg_video_path):
        _encode_frames_to_video(
            masked_reference_bg_frames,
            ref_bg_video_path,
            ctx.framerate,
            extra_codec_args=['-g', '1'],
        )

    enc_fg_video_path = os.path.join(ctx.masked_videos_dir, f"{slug}_fg_{frame_count:05d}.mp4")
    _encode_frames_to_video(
        masked_decoded_fg_frames,
        enc_fg_video_path,
        ctx.framerate,
        filter_chain=ctx.crop_filter,
        extra_codec_args=['-g', '1'],
    )

    enc_bg_video_path = os.path.join(ctx.masked_videos_dir, f"{slug}_bg_{frame_count:05d}.mp4")
    _encode_frames_to_video(
        masked_decoded_bg_frames,
        enc_bg_video_path,
        ctx.framerate,
        extra_codec_args=['-g', '1'],
    )

    frame_height, frame_width = reference_frames[0].shape[:2]
    vmaf_fg = calculate_vmaf(
        ref_fg_video_path,
        enc_fg_video_path,
        ctx.crop_width,
        ctx.crop_height,
        ctx.framerate,
        frame_stride=vmaf_stride,
    )
    vmaf_bg = calculate_vmaf(
        ref_bg_video_path,
        enc_bg_video_path,
        frame_width,
        frame_height,
        ctx.framerate,
        frame_stride=vmaf_stride,
    )

    result['foreground']['vmaf_mean'] = float(vmaf_fg.get('mean', 0))
    result['foreground']['vmaf_std'] = float(vmaf_fg.get('std', 0))
    result['background']['vmaf_mean'] = float(vmaf_bg.get('mean', 0))
    result['background']['vmaf_std'] = float(vmaf_bg.get('std', 0))

    if ctx.enable_fvmd:
        min_fvmd_samples = 10
        total_available_frames = frame_count
        if fvmd_max_frames is not None and fvmd_max_frames > 0:
            total_available_frames = min(total_available_frames, fvmd_max_frames)

        effective_stride = max(1, fvmd_stride)
        if total_available_frames >= min_fvmd_samples:
            max_stride_for_min_samples = max(1, total_available_frames // min_fvmd_samples)
            effective_stride = min(effective_stride, max_stride_for_min_samples)
        else:
            effective_stride = 1

        if effective_stride != fvmd_stride:
            _safe_print(
                f"    Adjusted FVMD stride from {fvmd_stride} to {effective_stride} for '{video_name}' to sample enough frames."
            )

        fvmd_indices = list(range(0, frame_count, effective_stride))
        if not fvmd_indices:
            fvmd_indices = [0]
        if fvmd_max_frames is not None and fvmd_max_frames > 0:
            fvmd_indices = fvmd_indices[:fvmd_max_frames]

        if len(fvmd_indices) < min_fvmd_samples:
            _safe_print(
                f"    Skipping FVMD for '{video_name}': only {len(fvmd_indices)} sampled frame(s); need at least {min_fvmd_samples}."
            )
        else:
            ref_fg_fvmd_frames = [masked_reference_fg_frames[i] for i in fvmd_indices]
            dec_fg_fvmd_frames = [masked_decoded_fg_frames[i] for i in fvmd_indices]
            ref_bg_fvmd_frames = [masked_reference_bg_frames[i] for i in fvmd_indices]
            dec_bg_fvmd_frames = [masked_decoded_bg_frames[i] for i in fvmd_indices]

            fvmd_log_dir = os.path.join(ctx.fvmd_log_root, slug)
            os.makedirs(fvmd_log_dir, exist_ok=True)

            fvmd_lock = None
            lock_acquired = False
            if _FVMD_DEVICE_LOCKS:
                if fvmd_device in _FVMD_DEVICE_LOCKS:
                    fvmd_lock = _FVMD_DEVICE_LOCKS[fvmd_device]
                elif None in _FVMD_DEVICE_LOCKS:
                    fvmd_lock = _FVMD_DEVICE_LOCKS[None]

            try:
                if fvmd_lock is not None:
                    if fvmd_lock.acquire(block=False):
                        lock_acquired = True
                    else:
                        device_label = f"cuda:{fvmd_device}" if fvmd_device is not None else "cpu"
                        _safe_print(f"    Waiting for FVMD device {device_label} to become available...")
                        fvmd_lock.acquire()
                        lock_acquired = True

                fg_fvmd_mean, fg_fvmd_std = calculate_fvmd(
                    ref_fg_fvmd_frames,
                    dec_fg_fvmd_frames,
                    log_root=fvmd_log_dir,
                    stride=1,
                    max_frames=None,
                    early_stop_delta=fvmd_early_stop_delta,
                    early_stop_window=fvmd_early_stop_window,
                    device=fvmd_device,
                    verbose=False,
                )
                bg_fvmd_mean, bg_fvmd_std = calculate_fvmd(
                    ref_bg_fvmd_frames,
                    dec_bg_fvmd_frames,
                    log_root=fvmd_log_dir,
                    stride=1,
                    max_frames=None,
                    early_stop_delta=fvmd_early_stop_delta,
                    early_stop_window=fvmd_early_stop_window,
                    device=fvmd_device,
                    verbose=False,
                )
            finally:
                if lock_acquired and fvmd_lock is not None:
                    fvmd_lock.release()

            result['foreground']['fvmd'] = fg_fvmd_mean
            result['foreground']['fvmd_std'] = fg_fvmd_std
            result['background']['fvmd'] = bg_fvmd_mean
            result['background']['fvmd_std'] = bg_fvmd_std

    print(f"  ✓ Completed evaluation for '{video_name}'.")
    return result


def _print_summary_report(results: Dict) -> None:
    """Prints a unified summary report with all metrics in one table."""
    print(f"\n{'='*180}")
    print(f"{'COMPREHENSIVE ANALYSIS SUMMARY':^180}")
    print(f"{'='*180}")

    if not results:
        print("No results to display.")
        return

    def _fmt(value: Optional[float], precision: int = 2) -> str:
        return 'N/A' if value is None or not math.isfinite(value) else f"{value:.{precision}f}"
    
    def _format_pair(fg: Optional[float], bg: Optional[float], prec: int = 2) -> str:
        return f"{_fmt(fg, prec)} / {_fmt(bg, prec)}"
    
    def _format_change(value: Optional[float]) -> str:
        return 'N/A' if value is None or not math.isfinite(value) else f"{value:+.2f}%"

    # Unified metrics table
    print(f"\n{'QUALITY METRICS (Foreground / Background)':^200}")
    print(f"{'Method':<20} {'PSNR (dB)':<25} {'SSIM':<25} {'MSE':<25} {'LPIPS':<25} {'FVMD':<25} {'VMAF':<25} {'Bitrate (Mbps)':<15}")
    print(f"{'-'*200}")

    for video_name, data in results.items():
        fg_data = data['foreground']
        bg_data = data['background']

        # Format metric strings (FG / BG)
        psnr_str = _format_pair(fg_data.get('psnr_mean'), bg_data.get('psnr_mean'), precision_fg=2)
        ssim_str = _format_pair(fg_data.get('ssim_mean'), bg_data.get('ssim_mean'), precision_fg=4, precision_bg=4)
        mse_str = _format_pair(fg_data.get('mse_mean'), bg_data.get('mse_mean'), precision_fg=2, precision_bg=2)
        lpips_str = _format_pair(fg_data.get('lpips_mean'), bg_data.get('lpips_mean'), precision_fg=4, precision_bg=4)
        fvmd_str = _format_pair(fg_data.get('fvmd'), bg_data.get('fvmd'), precision_fg=2)
        vmaf_str = _format_pair(fg_data.get('vmaf_mean'), bg_data.get('vmaf_mean'), precision_fg=2)
        bitrate_str = _fmt(data.get('bitrate_mbps'), precision=2)

        print(f"{video_name:<20} {psnr_str:<25} {ssim_str:<25} {mse_str:<25} {lpips_str:<25} {fvmd_str:<25} {vmaf_str:<25} {bitrate_str:<15}")

    print(f"{'-'*200}")

    # Trade-off analysis against the first video as baseline
    if len(results) > 1:
        baseline_name = list(results.keys())[0]
        print(f"\n{'TRADE-OFF ANALYSIS (vs. ' + baseline_name + ')':^200}")
        print(f"{'Method':<20} {'PSNR FG %':<15} {'PSNR BG %':<15} {'SSIM FG %':<15} {'SSIM BG %':<15} {'MSE FG %':<15} {'MSE BG %':<15} {'LPIPS FG %':<15} {'LPIPS BG %':<15} {'FVMD FG %':<15} {'FVMD BG %':<15} {'VMAF FG %':<15} {'VMAF BG %':<15}")
        print(f"{'-'*200}")

        for video_name in list(results.keys())[1:]:
            # Calculate changes for all metrics
            psnr_fg_change = math.nan
            psnr_bg_change = math.nan
            ssim_fg_change = math.nan
            ssim_bg_change = math.nan
            mse_fg_change = math.nan
            mse_bg_change = math.nan
            lpips_fg_change = math.nan
            lpips_bg_change = math.nan
            fvmd_fg_change = math.nan
            fvmd_bg_change = math.nan
            vmaf_fg_change = math.nan
            vmaf_bg_change = math.nan

            for metric in ['psnr', 'ssim', 'mse', 'lpips', 'vmaf']:
                for region in ['foreground', 'background']:
                    baseline_val = results[baseline_name][region].get(f'{metric}_mean')
                    current_val = results[video_name][region].get(f'{metric}_mean')

                    change = math.nan
                    if (
                        isinstance(baseline_val, (int, float))
                        and isinstance(current_val, (int, float))
                        and math.isfinite(baseline_val)
                        and math.isfinite(current_val)
                        and baseline_val != 0
                    ):
                        if metric == 'lpips':
                            if current_val > 0:
                                change = ((baseline_val / current_val) - 1) * 100
                        else:
                            change = ((current_val / baseline_val) - 1) * 100

                    if metric == 'psnr' and region == 'foreground':
                        psnr_fg_change = change
                    elif metric == 'psnr' and region == 'background':
                        psnr_bg_change = change
                    elif metric == 'ssim' and region == 'foreground':
                        ssim_fg_change = change
                    elif metric == 'ssim' and region == 'background':
                        ssim_bg_change = change
                    elif metric == 'mse' and region == 'foreground':
                        mse_fg_change = change
                    elif metric == 'mse' and region == 'background':
                        mse_bg_change = change
                    elif metric == 'lpips' and region == 'foreground':
                        lpips_fg_change = change
                    elif metric == 'lpips' and region == 'background':
                        lpips_bg_change = change
                    elif metric == 'vmaf' and region == 'foreground':
                        vmaf_fg_change = change
                    elif metric == 'vmaf' and region == 'background':
                        vmaf_bg_change = change

            # Calculate FVMD changes (lower is better, so invert like LPIPS)
            for region in ['foreground', 'background']:
                baseline_fvmd = results[baseline_name][region].get('fvmd')
                current_fvmd = results[video_name][region].get('fvmd')

                change = math.nan
                if (
                    isinstance(baseline_fvmd, (int, float))
                    and isinstance(current_fvmd, (int, float))
                    and math.isfinite(baseline_fvmd)
                    and math.isfinite(current_fvmd)
                    and baseline_fvmd > 0
                    and current_fvmd > 0
                ):
                    change = ((baseline_fvmd / current_fvmd) - 1) * 100

                if region == 'foreground':
                    fvmd_fg_change = change
                else:
                    fvmd_bg_change = change

            psnr_fg_change_str = _format_change(psnr_fg_change)
            psnr_bg_change_str = _format_change(psnr_bg_change)
            ssim_fg_change_str = _format_change(ssim_fg_change)
            ssim_bg_change_str = _format_change(ssim_bg_change)
            mse_fg_change_str = _format_change(mse_fg_change)
            mse_bg_change_str = _format_change(mse_bg_change)
            lpips_fg_change_str = _format_change(lpips_fg_change)
            lpips_bg_change_str = _format_change(lpips_bg_change)
            fvmd_fg_change_str = _format_change(fvmd_fg_change)
            fvmd_bg_change_str = _format_change(fvmd_bg_change)
            vmaf_fg_change_str = _format_change(vmaf_fg_change)
            vmaf_bg_change_str = _format_change(vmaf_bg_change)

            print(
                f"{video_name:<20} "
                f"{psnr_fg_change_str:<15} {psnr_bg_change_str:<15} "
                f"{ssim_fg_change_str:<15} {ssim_bg_change_str:<15} "
                f"{mse_fg_change_str:<15} {mse_bg_change_str:<15} "
                f"{lpips_fg_change_str:<15} {lpips_bg_change_str:<15} "
                f"{fvmd_fg_change_str:<15} {fvmd_bg_change_str:<15} "
                f"{vmaf_fg_change_str:<15} {vmaf_bg_change_str:<15}"
            )

        print(f"{'-'*180}")


def run_elvis(config: ElvisConfig) -> Dict[str, Any]:
    """Execute the full Elvis pipeline with the supplied configuration."""

    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

    script_dir = Path(__file__).resolve().parent
    os.chdir(str(script_dir))

    reference_video = config.reference_video
    width, height = config.width, config.height
    block_size = config.block_size
    shrink_amount = config.shrink_amount

    # Generate experiment directory name from key parameters
    video_name = Path(reference_video).stem
    experiment_dir = os.path.abspath(
        f"experiment_{video_name}_w{width}_h{height}_bs{block_size}_shrink{shrink_amount}"
    )
    os.makedirs(experiment_dir, exist_ok=True)

    execution_times: Dict[str, float] = {}
    approach_times = defaultdict(float)

    # Always use framerate from the reference video
    cap = cv2.VideoCapture(reference_video)
    framerate = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    if not framerate or framerate <= 0:
        framerate = 30.0

    target_bitrate = config.target_bitrate_override
    if target_bitrate is None:
        target_bitrate = calculate_target_bitrate(width, height, framerate, quality_factor=config.quality_factor)

    config_dict = asdict(config)
    pipeline_params: Dict[str, Any] = {
        "config": config_dict,
        "derived": {
            "framerate": framerate,
            "target_bitrate": target_bitrate,
            "experiment_dir": experiment_dir,
            "quality_factor": config.quality_factor,
        },
        "functions": {
            "calculate_removability_scores": {
                "alpha": config.removability_alpha,
                "smoothing_beta": config.removability_smoothing_beta,
                "block_size": block_size,
            },
            "apply_selective_removal": {
                "shrink_amount": shrink_amount,
            },
            "inpaint_with_propainter": {
                "resize_ratio": config.propainter_resize_ratio,
                "ref_stride": config.propainter_ref_stride,
                "neighbor_length": config.propainter_neighbor_length,
                "subvideo_length": config.propainter_subvideo_length,
                "mask_dilation": config.propainter_mask_dilation,
                "raft_iter": config.propainter_raft_iter,
                "fp16": config.propainter_fp16,
                "devices": list(config.propainter_devices) if config.propainter_devices else None,
                "parallel_chunk_length": config.propainter_parallel_chunk_length,
                "chunk_overlap": config.propainter_chunk_overlap,
            },
            "inpaint_with_e2fgvi": {
                "ref_stride": config.e2fgvi_ref_stride,
                "neighbor_stride": config.e2fgvi_neighbor_stride,
                "num_ref": config.e2fgvi_num_ref,
                "mask_dilation": config.e2fgvi_mask_dilation,
                "devices": list(config.e2fgvi_devices) if config.e2fgvi_devices else None,
                "parallel_chunk_length": config.e2fgvi_parallel_chunk_length,
                "chunk_overlap": config.e2fgvi_chunk_overlap,
            },
            "restore_downsampled_with_realesrgan": {
                "denoise_strength": config.realesrgan_denoise_strength,
                "tile": config.realesrgan_tile,
                "tile_pad": config.realesrgan_tile_pad,
                "pre_pad": config.realesrgan_pre_pad,
                "fp32": config.realesrgan_fp32,
                "devices": list(config.realesrgan_devices) if config.realesrgan_devices else None,
                "parallel_chunk_length": config.realesrgan_parallel_chunk_length,
                "per_device_workers": config.realesrgan_per_device_workers,
            },
            "restore_with_instantir_adaptive": {
                "cfg": config.instantir_cfg,
                "creative_start": config.instantir_creative_start,
                "preview_start": config.instantir_preview_start,
                "seed": config.instantir_seed,
                "devices": list(config.instantir_devices) if config.instantir_devices else None,
                "batch_size": config.instantir_batch_size,
                "parallel_chunk_length": config.instantir_parallel_chunk_length,
            },
            "analyze_encoding_performance": {
                "generate_opencv_benchmarks": config.generate_opencv_benchmarks,
                "metric_stride": config.metric_stride,
                "fvmd_stride": config.fvmd_stride,
                "fvmd_max_frames": config.fvmd_max_frames,
                "fvmd_processes": config.fvmd_processes,
                "fvmd_early_stop_delta": config.fvmd_early_stop_delta,
                "fvmd_early_stop_window": config.fvmd_early_stop_window,
                "vmaf_stride": config.vmaf_stride,
                "enable_fvmd": config.enable_fvmd,
            },
        },
    }

    encode_function_params = {
        "preset": config.encode_preset,
        "pix_fmt": config.encode_pix_fmt,
        "target_bitrate": target_bitrate,
    }
    pipeline_params["functions"]["encode_video"] = encode_function_params

    print(f"Processing video: {reference_video}")
    print(f"Target resolution: {width}x{height}")
    print(
        f"Calculated target bitrate: {target_bitrate} bps "
        f"({target_bitrate/1_000_000:.1f} Mbps) for {width}x{height}@{framerate:.1f}fps"
    )

    # Preprocessing
    start = time.time()

    frames_dir = os.path.join(experiment_dir, "frames")
    reference_frames_dir = os.path.join(frames_dir, "reference")
    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(reference_frames_dir, exist_ok=True)

    print("Converting video to raw YUV format...")
    raw_video_path = os.path.join(experiment_dir, "reference_raw.yuv")
    subprocess.run(
        f"ffmpeg -hide_banner -loglevel error -y -i {reference_video} "
        f"-vf scale={width}:{height} -c:v rawvideo -pix_fmt yuv420p {raw_video_path}",
        shell=True,
        check=False,
    )

    print("Extracting reference frames...")
    subprocess.run(
        f"ffmpeg -hide_banner -loglevel error -y -video_size {width}x{height} "
        f"-r {framerate} -pixel_format yuv420p -i {raw_video_path} -q:v 2 {reference_frames_dir}/%05d.png",
        shell=True,
        check=False,
    )

    frame_files = sorted([f for f in os.listdir(reference_frames_dir) if f.endswith('.png')])
    reference_frames = [cv2.imread(os.path.join(reference_frames_dir, f)) for f in frame_files]

    end = time.time()
    execution_times["Preprocessing"] = end - start
    print(f"Video preprocessing completed in {end - start:.2f} seconds.\n")

    # Removability scores
    start = time.time()
    print(f"Calculating removability scores with block size: {block_size}x{block_size}")
    removability_scores = calculate_removability_scores(
        raw_video_file=raw_video_path,
        reference_frames_folder=reference_frames_dir,
        width=width,
        height=height,
        block_size=block_size,
        alpha=config.removability_alpha,
        working_dir=experiment_dir,
        smoothing_beta=config.removability_smoothing_beta,
    )
    end = time.time()
    execution_times["Removability Calculation"] = end - start
    print(f"Removability scores calculation completed in {end - start:.2f} seconds.\n")

    # Baseline encoding
    start = time.time()
    print("Encoding reference frames with two-pass for baseline comparison...")
    baseline_video = os.path.join(experiment_dir, "baseline.mp4")
    encode_video(
        input_frames_dir=reference_frames_dir,
        output_video=baseline_video,
        framerate=framerate,
        width=width,
        height=height,
        target_bitrate=target_bitrate,
        preset=config.encode_preset,
        pix_fmt=config.encode_pix_fmt,
    )
    end = time.time()
    duration = end - start
    approach_times[APPROACH_BASELINE] += duration
    print(f"Baseline encoding completed in {duration:.2f} seconds.\n")

    # ELVIS shrinking
    start = time.time()
    print(f"Shrinking and encoding frames with {APPROACH_ELVIS}...")
    shrunk_frames_dir = os.path.join(experiment_dir, "frames", "shrunk")
    os.makedirs(shrunk_frames_dir, exist_ok=True)

    shrunk_frames, removal_masks, block_coords_to_remove = zip(
        *[
            apply_selective_removal(img, scores, block_size, shrink_amount=shrink_amount)
            for img, scores in zip(reference_frames, removability_scores)
        ]
    )

    for i, frame in enumerate(shrunk_frames):
        cv2.imwrite(os.path.join(shrunk_frames_dir, f"{i+1:05d}.png"), frame)

    shrunk_video = os.path.join(experiment_dir, "shrunk.mp4")
    shrunk_width = shrunk_frames[0].shape[1]
    encode_video(
        input_frames_dir=shrunk_frames_dir,
        output_video=shrunk_video,
        framerate=framerate,
        width=shrunk_width,
        height=height,
        target_bitrate=target_bitrate,
        preset=config.encode_preset,
        pix_fmt=config.encode_pix_fmt,
    )

    removal_masks_np = np.array(removal_masks, dtype=np.uint8)
    masks_packed = np.packbits(removal_masks_np)
    np.savez(
        os.path.join(experiment_dir, f"shrink_masks_{block_size}.npz"),
        packed=masks_packed,
        shape=removal_masks_np.shape,
    )
    end = time.time()
    duration = end - start
    approach_times[APPROACH_ELVIS] += duration
    print(f"{APPROACH_ELVIS} shrinking completed in {duration:.2f} seconds.\n")

    # Adaptive ROI encoding
    start = time.time()
    print("Encoding frames with ROI-based adaptive quantization...")
    adaptive_video = os.path.join(experiment_dir, "adaptive.mp4")
    maps_dir = os.path.join(experiment_dir, "maps")
    qp_maps_dir = os.path.join(maps_dir, "qp_maps")
    os.makedirs(maps_dir, exist_ok=True)

    valid_ctu_sizes = [16, 32, 64]
    roi_ctu_size = min(valid_ctu_sizes, key=lambda x: abs(x - block_size))
    pipeline_params["functions"]["encode_with_roi"] = {
        "target_bitrate": target_bitrate,
        "ctu_size": roi_ctu_size,
    }

    encode_with_roi(
        input_frames_dir=reference_frames_dir,
        output_video=adaptive_video,
        removability_scores=removability_scores,
        block_size=block_size,
        framerate=framerate,
        width=width,
        height=height,
        target_bitrate=target_bitrate,
        save_qp_maps=True,
        qp_maps_dir=qp_maps_dir,
    )
    end = time.time()
    duration = end - start
    approach_times[APPROACH_PRESLEY_QP] += duration
    print(f"{APPROACH_PRESLEY_QP} encoding completed in {duration:.2f} seconds.\n")

    # PRESLEY RealESRGAN
    start = time.time()
    print(f"Applying {APPROACH_PRESLEY_REALESRGAN} adaptive filtering and encoding...")
    downsampled_frames_dir = os.path.join(experiment_dir, "frames", "downsampled")
    os.makedirs(downsampled_frames_dir, exist_ok=True)

    downsampled_frames, downsample_maps = zip(
        *[
            filter_frame_downsample(img, scores, block_size)
            for img, scores in zip(reference_frames, removability_scores)
        ]
    )

    for i, frame in enumerate(downsampled_frames):
        cv2.imwrite(os.path.join(downsampled_frames_dir, f"{i+1:05d}.png"), frame)

    downsampled_video = os.path.join(experiment_dir, "downsampled_encoded.mp4")
    encode_video(
        input_frames_dir=downsampled_frames_dir,
        output_video=downsampled_video,
        framerate=framerate,
        width=width,
        height=height,
        target_bitrate=target_bitrate,
        preset=config.encode_preset,
        pix_fmt=config.encode_pix_fmt,
    )

    # Encode strength maps using NPZ compression
    downsample_maps_file = os.path.join(maps_dir, "downsample_maps.npz")
    encode_strength_maps_to_npz(
        strength_maps=list(downsample_maps),
        output_path=downsample_maps_file,
    )
    end = time.time()
    duration = end - start
    approach_times[APPROACH_PRESLEY_REALESRGAN] += duration
    print(f"{APPROACH_PRESLEY_REALESRGAN} filtering and encoding completed in {duration:.2f} seconds.\n")

    # PRESLEY InstantIR
    start = time.time()
    print(f"Applying {APPROACH_PRESLEY_INSTANTIR} adaptive filtering and encoding...")
    gaussian_frames_dir = os.path.join(experiment_dir, "frames", "gaussian")
    os.makedirs(gaussian_frames_dir, exist_ok=True)

    gaussian_frames, gaussian_maps = zip(
        *[
            filter_frame_gaussian(img, scores, block_size)
            for img, scores in zip(reference_frames, removability_scores)
        ]
    )

    for i, frame in enumerate(gaussian_frames):
        cv2.imwrite(os.path.join(gaussian_frames_dir, f"{i+1:05d}.png"), frame)

    gaussian_video = os.path.join(experiment_dir, "gaussian_encoded.mp4")
    encode_video(
        input_frames_dir=gaussian_frames_dir,
        output_video=gaussian_video,
        framerate=framerate,
        width=width,
        height=height,
        target_bitrate=target_bitrate,
        preset=config.encode_preset,
        pix_fmt=config.encode_pix_fmt,
    )

    # Encode strength maps using NPZ compression
    gaussian_maps_file = os.path.join(maps_dir, "gaussian_maps.npz")
    encode_strength_maps_to_npz(
        strength_maps=list(gaussian_maps),
        output_path=gaussian_maps_file,
    )
    end = time.time()
    duration = end - start
    approach_times[APPROACH_PRESLEY_INSTANTIR] += duration
    print(f"{APPROACH_PRESLEY_INSTANTIR} filtering and encoding completed in {duration:.2f} seconds.\n")

    # Client-side stretching
    start = time.time()
    print(f"Decoding and stretching {APPROACH_ELVIS} video...")
    removal_masks_file = np.load(os.path.join(experiment_dir, f"shrink_masks_{block_size}.npz"))
    removal_masks_loaded = np.unpackbits(removal_masks_file['packed'])
    removal_masks = removal_masks_loaded[:np.prod(removal_masks_file['shape'])].reshape(removal_masks_file['shape'])

    stretched_frames_dir = os.path.join(experiment_dir, "frames", "stretched")
    if not decode_video(
        shrunk_video,
        stretched_frames_dir,
        framerate=framerate,
    ):
        raise RuntimeError(f"Failed to decode shrunk video: {shrunk_video}")

    num_masks = len(removal_masks)
    stretched_frames = [
        stretch_frame(
            cv2.imread(os.path.join(stretched_frames_dir, f"{i+1:05d}.png")),
            removal_masks[i],
            block_size,
        )
        for i in range(num_masks)
    ]

    for i, frame in enumerate(stretched_frames):
        cv2.imwrite(os.path.join(stretched_frames_dir, f"{i+1:05d}.png"), frame)

    # Save removal masks at block resolution (not scaled to video resolution)
    removal_masks_dir = os.path.join(maps_dir, "removal_masks")
    os.makedirs(removal_masks_dir, exist_ok=True)
    for i, mask in enumerate(removal_masks):
        mask_img = (mask * 255).astype(np.uint8)
        # Keep at block resolution - no resizing
        cv2.imwrite(os.path.join(removal_masks_dir, f"{i+1:05d}.png"), mask_img)
    num_blocks_y, num_blocks_x = removal_masks[0].shape
    print(f"Removal masks saved at block resolution ({num_blocks_y}x{num_blocks_x})")
    
    # Create full-resolution masks for inpainting (separate from metadata masks)
    removal_masks_fullres_dir = os.path.join(experiment_dir, "frames", "removal_masks_fullres")
    os.makedirs(removal_masks_fullres_dir, exist_ok=True)
    for i, mask in enumerate(removal_masks):
        mask_img = (mask * 255).astype(np.uint8)
        # Upscale to full video resolution for inpainting algorithms
        mask_fullres = cv2.resize(mask_img, (width, height), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(os.path.join(removal_masks_fullres_dir, f"{i+1:05d}.png"), mask_fullres)
    print(f"Full-resolution masks for inpainting saved to {removal_masks_fullres_dir}")
    
    end = time.time()
    duration = end - start
    approach_times[APPROACH_ELVIS] += duration
    print(f"{APPROACH_ELVIS} stretching completed in {duration:.2f} seconds.\n")

    stretched_video = os.path.join(experiment_dir, "stretched.mp4")
    encode_video(
        input_frames_dir=stretched_frames_dir,
        output_video=stretched_video,
        framerate=framerate,
        width=width,
        height=height,
        target_bitrate=None,
    )

    start = time.time()
    print("Inpainting stretched frames with CV2...")
    inpainted_cv2_frames_dir = os.path.join(experiment_dir, "frames", "inpainted_cv2")
    os.makedirs(inpainted_cv2_frames_dir, exist_ok=True)
    for i in range(len(removal_masks)):
        stretched_frame = cv2.imread(os.path.join(stretched_frames_dir, f"{i+1:05d}.png"))
        # Use full-resolution masks for inpainting
        mask_img = cv2.imread(os.path.join(removal_masks_fullres_dir, f"{i+1:05d}.png"), cv2.IMREAD_GRAYSCALE)
        inpainted_frame = cv2.inpaint(stretched_frame, mask_img, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
        cv2.imwrite(os.path.join(inpainted_cv2_frames_dir, f"{i+1:05d}.png"), inpainted_frame)
    end = time.time()
    duration = end - start
    approach_times[APPROACH_ELVIS_CV2] += duration
    print(f"{APPROACH_ELVIS_CV2} inpainting completed in {duration:.2f} seconds.\n")

    inpainted_cv2_video = os.path.join(experiment_dir, "inpainted_cv2.mp4")
    encode_video(
        input_frames_dir=inpainted_cv2_frames_dir,
        output_video=inpainted_cv2_video,
        framerate=framerate,
        width=width,
        height=height,
        target_bitrate=None,
    )

    start = time.time()
    print("Inpainting stretched frames with ProPainter...")
    inpainted_frames_dir = os.path.join(experiment_dir, "frames", "inpainted")
    inpaint_with_propainter(
        stretched_frames_dir=stretched_frames_dir,
        removal_masks_dir=removal_masks_fullres_dir,
        output_frames_dir=inpainted_frames_dir,
        width=width,
        height=height,
        framerate=framerate,
        resize_ratio=config.propainter_resize_ratio,
        ref_stride=config.propainter_ref_stride,
        neighbor_length=config.propainter_neighbor_length,
        subvideo_length=config.propainter_subvideo_length,
        mask_dilation=config.propainter_mask_dilation,
        raft_iter=config.propainter_raft_iter,
        fp16=config.propainter_fp16,
        devices=list(config.propainter_devices) if config.propainter_devices else None,
        parallel_chunk_length=config.propainter_parallel_chunk_length,
        chunk_overlap=config.propainter_chunk_overlap,
    )
    end = time.time()
    duration = end - start
    approach_times[APPROACH_ELVIS_PROP] += duration
    print(f"{APPROACH_ELVIS_PROP} inpainting completed in {duration:.2f} seconds.\n")

    inpainted_video = os.path.join(experiment_dir, "inpainted_propainter.mp4")
    encode_video(
        input_frames_dir=inpainted_frames_dir,
        output_video=inpainted_video,
        framerate=framerate,
        width=width,
        height=height,
        target_bitrate=None,
    )

    start = time.time()
    print("Inpainting stretched frames with E2FGVI...")
    inpainted_e2fgvi_frames_dir = os.path.join(experiment_dir, "frames", "inpainted_e2fgvi")
    inpaint_with_e2fgvi(
        stretched_frames_dir=stretched_frames_dir,
        removal_masks_dir=removal_masks_fullres_dir,
        output_frames_dir=inpainted_e2fgvi_frames_dir,
        width=width,
        height=height,
        framerate=framerate,
        ref_stride=config.e2fgvi_ref_stride,
        neighbor_stride=config.e2fgvi_neighbor_stride,
        num_ref=config.e2fgvi_num_ref,
        mask_dilation=config.e2fgvi_mask_dilation,
        devices=config.e2fgvi_devices,
        parallel_chunk_length=config.e2fgvi_parallel_chunk_length,
        chunk_overlap=config.e2fgvi_chunk_overlap,
    )
    end = time.time()
    duration = end - start
    approach_times[APPROACH_ELVIS_E2FGVI] += duration
    print(f"{APPROACH_ELVIS_E2FGVI} inpainting completed in {duration:.2f} seconds.\n")

    inpainted_e2fgvi_video = os.path.join(experiment_dir, "inpainted_e2fgvi.mp4")
    encode_video(
        input_frames_dir=inpainted_e2fgvi_frames_dir,
        output_video=inpainted_e2fgvi_video,
        framerate=framerate,
        width=width,
        height=height,
        target_bitrate=None,
    )

    # PRESLEY RealESRGAN restoration
    start = time.time()
    print(f"Decoding {APPROACH_PRESLEY_REALESRGAN} video and strength maps...")
    downsampled_frames_decoded_dir = os.path.join(experiment_dir, "frames", "downsampled_decoded")
    if not decode_video(
        downsampled_video,
        downsampled_frames_decoded_dir,
        framerate=framerate,
    ):
        raise RuntimeError(f"Failed to decode downsampled video: {downsampled_video}")

    # Decode strength maps from NPZ
    downsample_maps_file = os.path.join(maps_dir, "downsample_maps.npz")
    strength_maps = decode_strength_maps_from_npz(downsample_maps_file)
    # Save decoded maps as PNG for debugging
    downsampled_maps_decoded_dir = os.path.join(experiment_dir, "maps", "downsampled_maps_decoded")
    os.makedirs(downsampled_maps_decoded_dir, exist_ok=True)
    for i, map_frame in enumerate(strength_maps):
        # Normalize to 0-255 for visualization
        map_img = np.clip(map_frame.astype(np.float32) * 25.5, 0, 255).astype(np.uint8)
        cv2.imwrite(os.path.join(downsampled_maps_decoded_dir, f"{i+1:05d}.png"), map_img)
    print(f"  Decoded downsample maps saved to {downsampled_maps_decoded_dir} at block resolution ({strength_maps.shape[1]}x{strength_maps.shape[2]})")
    end = time.time()
    duration = end - start
    approach_times[APPROACH_PRESLEY_REALESRGAN] += duration
    print(f"Decoding completed in {duration:.2f} seconds.\n")

    start = time.time()
    print(f"Applying adaptive upsampling restoration for {APPROACH_PRESLEY_REALESRGAN}...")
    downsampled_restored_frames_dir = os.path.join(experiment_dir, "frames", "downsampled_restored")
    os.makedirs(downsampled_restored_frames_dir, exist_ok=True)
    restore_downsampled_with_realesrgan(
        input_frames_dir=downsampled_frames_decoded_dir,
        output_frames_dir=downsampled_restored_frames_dir,
        downscale_maps=strength_maps,
        block_size=block_size,
        denoise_strength=config.realesrgan_denoise_strength,
        tile=config.realesrgan_tile,
        tile_pad=config.realesrgan_tile_pad,
        pre_pad=config.realesrgan_pre_pad,
        fp32=config.realesrgan_fp32,
        devices=list(config.realesrgan_devices) if config.realesrgan_devices else None,
        parallel_chunk_length=config.realesrgan_parallel_chunk_length,
        per_device_workers=config.realesrgan_per_device_workers,
    )
    end = time.time()
    duration = end - start
    approach_times[APPROACH_PRESLEY_REALESRGAN] += duration
    print(f"{APPROACH_PRESLEY_REALESRGAN} restoration completed in {duration:.2f} seconds.\n")

    downsampled_restored_video = os.path.join(experiment_dir, "downsampled_restored.mp4")
    encode_video(
        input_frames_dir=downsampled_restored_frames_dir,
        output_video=downsampled_restored_video,
        framerate=framerate,
        width=width,
        height=height,
        target_bitrate=None,
    )

    # Gaussian restoration
    start = time.time()
    print(f"Decoding {APPROACH_PRESLEY_INSTANTIR} video and strength maps...")
    gaussian_frames_decoded_dir = os.path.join(experiment_dir, "frames", "gaussian_decoded")
    if not decode_video(
        gaussian_video,
        gaussian_frames_decoded_dir,
        framerate=framerate,
    ):
        raise RuntimeError(f"Failed to decode Gaussian video: {gaussian_video}")

    # Decode strength maps from NPZ
    gaussian_maps_file = os.path.join(maps_dir, "gaussian_maps.npz")
    strength_maps_gaussian = decode_strength_maps_from_npz(gaussian_maps_file)
    # Save decoded maps as PNG for debugging
    gaussian_maps_decoded_dir = os.path.join(experiment_dir, "maps", "gaussian_maps_decoded")
    os.makedirs(gaussian_maps_decoded_dir, exist_ok=True)
    for i, map_frame in enumerate(strength_maps_gaussian):
        # Normalize to 0-255 for visualization (gaussian maps are 0-10)
        map_img = np.clip(map_frame.astype(np.float32) * 25.5, 0, 255).astype(np.uint8)
        cv2.imwrite(os.path.join(gaussian_maps_decoded_dir, f"{i+1:05d}.png"), map_img)
    print(f"  Decoded gaussian maps saved to {gaussian_maps_decoded_dir} at block resolution ({strength_maps_gaussian.shape[1]}x{strength_maps_gaussian.shape[2]})")
    end = time.time()
    duration = end - start
    approach_times[APPROACH_PRESLEY_INSTANTIR] += duration
    print(f"Decoding completed in {duration:.2f} seconds.\n")

    start = time.time()
    print(f"Applying adaptive deblurring restoration for {APPROACH_PRESLEY_INSTANTIR}...")
    instantir_work_dir = os.path.join(experiment_dir, "instantir_work")
    gaussian_instantir_input_dir = os.path.join(instantir_work_dir, "gaussian_decoded")
    os.makedirs(gaussian_instantir_input_dir, exist_ok=True)

    decoded_gaussian_frames = [
        f for f in os.listdir(gaussian_frames_decoded_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ]
    print(f"  Copying {len(decoded_gaussian_frames)} frames to InstantIR input directory...")
    for frame_file in decoded_gaussian_frames:
        shutil.copy2(
            os.path.join(gaussian_frames_decoded_dir, frame_file),
            os.path.join(gaussian_instantir_input_dir, frame_file),
        )

    restore_with_instantir_adaptive(
        input_frames_dir=gaussian_instantir_input_dir,
        blur_maps=strength_maps_gaussian,
        block_size=block_size,
        cfg=config.instantir_cfg,
        creative_start=config.instantir_creative_start,
        preview_start=config.instantir_preview_start,
        seed=config.instantir_seed,
        devices=list(config.instantir_devices) if config.instantir_devices else None,
        batch_size=config.instantir_batch_size,
        parallel_chunk_length=config.instantir_parallel_chunk_length,
    )

    gaussian_restored_frames_dir = os.path.join(experiment_dir, "frames", "gaussian_restored")
    os.makedirs(gaussian_restored_frames_dir, exist_ok=True)
    print("  Copying restored frames to output directory...")
    for frame_file in os.listdir(gaussian_instantir_input_dir):
        if frame_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            shutil.copy2(
                os.path.join(gaussian_instantir_input_dir, frame_file),
                os.path.join(gaussian_restored_frames_dir, frame_file),
            )

    end = time.time()
    duration = end - start
    approach_times[APPROACH_PRESLEY_INSTANTIR] += duration
    print(f"{APPROACH_PRESLEY_INSTANTIR} restoration completed in {duration:.2f} seconds.\n")

    gaussian_restored_video = os.path.join(experiment_dir, "gaussian_restored.mp4")
    encode_video(
        input_frames_dir=gaussian_restored_frames_dir,
        output_video=gaussian_restored_video,
        framerate=framerate,
        width=width,
        height=height,
        target_bitrate=None,
    )

    # Performance evaluation
    print("Evaluating and comparing encoding performance...")
    start = time.time()

    # Calculate video sizes including metadata
    if config.strength_maps_use_npz:
        downsample_maps_path = os.path.join(maps_dir, "downsample_maps.npz")
        gaussian_maps_path = os.path.join(maps_dir, "gaussian_maps.npz")
    else:
        downsample_maps_path = os.path.join(maps_dir, "downsample_encoded.mp4")
        gaussian_maps_path = os.path.join(maps_dir, "gaussian_encoded.mp4")
    
    video_sizes = {
        APPROACH_BASELINE: os.path.getsize(baseline_video),
        APPROACH_ELVIS: os.path.getsize(shrunk_video) + os.path.getsize(os.path.join(experiment_dir, f"shrink_masks_{block_size}.npz")),
        APPROACH_PRESLEY_QP: os.path.getsize(adaptive_video),
        APPROACH_PRESLEY_REALESRGAN: os.path.getsize(downsampled_video) + os.path.getsize(downsample_maps_path),
        APPROACH_PRESLEY_INSTANTIR: os.path.getsize(gaussian_video) + os.path.getsize(gaussian_maps_path),
    }

    frame_count = len(frame_files)
    duration = frame_count / framerate if framerate else frame_count
    bitrates = {key: (size * 8) / duration for key, size in video_sizes.items()}

    print(
        f"\nEncoding Results (Target Bitrate: {target_bitrate} bps / {target_bitrate/1_000_000:.1f} Mbps):"
    )
    for key, bitrate in bitrates.items():
        print(f"{key} bitrate: {bitrate / 1_000_000:.2f} Mbps")

    encoded_videos = {
        APPROACH_BASELINE: baseline_video,
        APPROACH_PRESLEY_QP: adaptive_video,
        APPROACH_ELVIS_CV2: inpainted_cv2_video,
        APPROACH_ELVIS_PROP: inpainted_video,
        APPROACH_ELVIS_E2FGVI: inpainted_e2fgvi_video,
        APPROACH_PRESLEY_REALESRGAN: downsampled_restored_video,
        APPROACH_PRESLEY_INSTANTIR: gaussian_restored_video,
    }

    ufo_masks_dir = os.path.join(experiment_dir, "maps", "ufo_masks")

    strength_maps_dict = {
        APPROACH_PRESLEY_REALESRGAN: strength_maps,
        APPROACH_PRESLEY_INSTANTIR: strength_maps_gaussian,
    }

    if config.generate_opencv_benchmarks:
        opencv_benchmarks, opencv_bitrates = generate_opencv_benchmarks(
            reference_frames=reference_frames,
            strength_maps=strength_maps_dict,
            block_size=block_size,
            framerate=framerate,
            width=width,
            height=height,
            temp_dir=experiment_dir,
            video_bitrates=bitrates,
        )
        encoded_videos.update(opencv_benchmarks)
        bitrates.update(opencv_bitrates)

    analysis_results = analyze_encoding_performance(
        reference_frames=reference_frames,
        encoded_videos=encoded_videos,
        block_size=block_size,
        width=width,
        height=height,
        temp_dir=experiment_dir,
        masks_dir=ufo_masks_dir,
        video_bitrates=bitrates,
        framerate=framerate,
        metric_stride=config.metric_stride,
        fvmd_stride=config.fvmd_stride,
        fvmd_max_frames=config.fvmd_max_frames,
        fvmd_early_stop_delta=config.fvmd_early_stop_delta,
        fvmd_early_stop_window=config.fvmd_early_stop_window,
        vmaf_stride=config.vmaf_stride,
        enable_fvmd=config.enable_fvmd,
    )

    end = time.time()
    execution_times["Performance Evaluation"] = end - start

    for approach, total in approach_times.items():
        execution_times[approach] = total

    analysis_results["execution_times_seconds"] = execution_times
    analysis_results["video_name"] = reference_video
    analysis_results["video_length_seconds"] = duration
    analysis_results["video_framerate"] = framerate
    analysis_results["video_resolution"] = f"{width}x{height}"
    analysis_results["block_size"] = block_size
    analysis_results["target_bitrate_bps"] = target_bitrate

    results_json_path = os.path.join(experiment_dir, "analysis_results.json")
    pipeline_params["derived"]["analysis_results_path"] = results_json_path
    analysis_results["parameters"] = pipeline_params
    analysis_results["experiment_dir"] = experiment_dir
    analysis_results["analysis_results_path"] = results_json_path

    with open(results_json_path, "w") as f:
        json.dump(analysis_results, f, indent=4)

    print(f"Analysis results saved to: {results_json_path}")

    return analysis_results


def _load_config_from_cli() -> ElvisConfig:
    parser = argparse.ArgumentParser(description="Run the ELVIS pipeline with configurable parameters.")
    parser.add_argument("--config", type=str, help="Path to a JSON file containing ElvisConfig fields.")
    parser.add_argument("--reference-video", type=str, help="Path to the input reference video.")
    parser.add_argument("--width", type=int, help="Target frame width.")
    parser.add_argument("--height", type=int, help="Target frame height.")
    parser.add_argument("--block-size", type=int, help="Processing block size.")
    parser.add_argument("--shrink-amount", type=float, help="Shrink amount for ELVIS.")
    parser.add_argument("--quality-factor", type=float, help="Quality factor for target bitrate calculation.")
    parser.add_argument("--target-bitrate", type=int, help="Override target bitrate in bits per second")
    parser.add_argument("--removability-alpha", type=float, help="Alpha parameter for removability scoring.")
    parser.add_argument("--removability-smoothing-beta", type=float, help="Smoothing beta for removability scoring.")
    parser.add_argument("--encode-preset", type=str, help="FFmpeg preset for encoding (e.g., medium, fast, slow).")
    parser.add_argument("--encode-pix-fmt", type=str, help="Pixel format for encoding (e.g., yuv420p).")
    parser.add_argument("--generate-opencv-benchmarks", dest="generate_opencv_benchmarks", action="store_true", help="Enable OpenCV baseline generation.")
    parser.add_argument("--disable-opencv-benchmarks", dest="generate_opencv_benchmarks", action="store_false", help="Disable OpenCV baseline generation.")
    parser.set_defaults(generate_opencv_benchmarks=None)
    parser.add_argument("--metric-stride", type=int, help="Stride for PSNR/SSIM/LPIPS metrics.")
    parser.add_argument("--fvmd-stride", type=int, help="Stride for FVMD computation.")
    parser.add_argument("--fvmd-max-frames", type=int, help="Maximum frames for FVMD computation.")
    parser.add_argument("--fvmd-processes", type=int, help="Number of FVMD worker processes.")
    parser.add_argument("--fvmd-early-stop-delta", type=float, help="Early stop delta for FVMD.")
    parser.add_argument("--fvmd-early-stop-window", type=int, help="Early stop window for FVMD.")
    parser.add_argument("--vmaf-stride", type=int, help="Stride for VMAF computation.")

    args = parser.parse_args()

    config_data: Dict[str, Any] = asdict(ElvisConfig())

    if args.config:
        with open(args.config, "r") as f:
            file_config = json.load(f)
        config_data.update(file_config)

    overrides = {
        "reference_video": args.reference_video,
        "width": args.width,
        "height": args.height,
        "block_size": args.block_size,
        "shrink_amount": args.shrink_amount,
        "quality_factor": args.quality_factor,
        "target_bitrate_override": args.target_bitrate,
        "removability_alpha": args.removability_alpha,
        "removability_smoothing_beta": args.removability_smoothing_beta,
        "encode_preset": args.encode_preset,
        "encode_pix_fmt": args.encode_pix_fmt,
        "metric_stride": args.metric_stride,
        "fvmd_stride": args.fvmd_stride,
        "fvmd_max_frames": args.fvmd_max_frames,
        "fvmd_processes": args.fvmd_processes,
        "fvmd_early_stop_delta": args.fvmd_early_stop_delta,
        "fvmd_early_stop_window": args.fvmd_early_stop_window,
        "vmaf_stride": args.vmaf_stride,
    }

    for key, value in overrides.items():
        if value is not None:
            config_data[key] = value

    if args.generate_opencv_benchmarks is not None:
        config_data["generate_opencv_benchmarks"] = args.generate_opencv_benchmarks

    return ElvisConfig(**config_data)


def main() -> None:
    config = _load_config_from_cli()
    results = run_elvis(config)
    path = results.get("analysis_results_path")
    if path:
        print(f"\nFinal analysis JSON: {path}")


if __name__ == "__main__":
    main()