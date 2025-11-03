import argparse
import time
import shutil
import os
import subprocess
import math
import threading
import contextlib
import warnings
from dataclasses import dataclass, asdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from itertools import islice
from pathlib import Path
import cv2
import numpy as np
from typing import Any, List, Callable, Tuple, Dict, Optional, Sequence, Union, NamedTuple, Iterator
from skimage.metrics import structural_similarity as ssim
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import json
import functools
import multiprocessing
import torch
import lpips
import platform
import tempfile
import uuid
from fvmd import fvmd
import sys
from instantir import InstantIRRuntime, load_runtime, restore_images_batch

# Cache platform-specific constants at module level for performance
NULL_DEVICE = "NUL" if platform.system() == "Windows" else "/dev/null"
IS_WINDOWS = platform.system() == "Windows"

try:
    from diffusers.utils import logging as _diffusers_logging  # type: ignore

    _diffusers_logging.disable_progress_bar()
    _diffusers_logging.set_verbosity_error()
except ImportError:  # pragma: no cover - optional dependency
    _diffusers_logging = None


@dataclass
class ElvisConfig:
    reference_video: str = "davis_test/bear.mp4"
    experiment_dir: str = "experiment"
    width: int = 640
    height: int = 360
    block_size: int = 8
    shrink_amount: float = 0.25
    quality_factor: float = 1.2
    target_bitrate_override: Optional[int] = None
    force_framerate: Optional[float] = None
    removability_alpha: float = 0.5
    removability_smoothing_beta: float = 0.5
    removability_working_dir: Optional[str] = None
    decode_quality: int = 1
    decode_start_number: int = 1
    baseline_encode_preset: str = "medium"
    baseline_encode_pix_fmt: str = "yuv420p"
    adaptive_encode_preset: str = "medium"
    adaptive_encode_pix_fmt: str = "yuv420p"
    shrunk_encode_preset: str = "medium"
    shrunk_encode_pix_fmt: str = "yuv420p"
    downsample_encode_preset: str = "medium"
    downsample_encode_pix_fmt: str = "yuv420p"
    gaussian_encode_preset: str = "medium"
    gaussian_encode_pix_fmt: str = "yuv420p"
    roi_save_qp_maps: bool = True
    downsample_strength_target_bitrate: int = 50000
    gaussian_strength_target_bitrate: int = 50000
    propainter_dir: Optional[str] = None
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
    e2fgvi_dir: Optional[str] = None
    e2fgvi_model: str = "e2fgvi_hq"
    e2fgvi_ckpt: Optional[str] = None
    e2fgvi_ref_stride: int = 10
    e2fgvi_neighbor_stride: int = 5
    e2fgvi_num_ref: int = -1
    e2fgvi_mask_dilation: int = 4
    realesrgan_dir: Optional[str] = None
    realesrgan_model_name: str = "RealESRGAN_x4plus"
    realesrgan_denoise_strength: float = 1.0
    realesrgan_tile: int = 0
    realesrgan_tile_pad: int = 10
    realesrgan_pre_pad: int = 0
    realesrgan_fp32: bool = False
    realesrgan_devices: Optional[List[Union[int, str]]] = None
    realesrgan_parallel_chunk_length: Optional[int] = None
    realesrgan_per_device_workers: int = 1
    realesrgan_model_path: Optional[str] = None
    instantir_weights_dir: Optional[str] = None
    instantir_cfg: float = 7.0
    instantir_creative_start: float = 1.0
    instantir_preview_start: float = 0.0
    instantir_seed: Optional[int] = 42
    instantir_devices: Optional[List[Union[int, str]]] = None
    instantir_batch_size: int = 4
    instantir_parallel_chunk_length: Optional[int] = None
    analysis_sample_frames: Optional[List[int]] = None
    generate_opencv_benchmarks: bool = True
    metric_stride: int = 1
    fvmd_stride: int = 1
    fvmd_max_frames: Optional[int] = None
    fvmd_processes: Optional[int] = None
    fvmd_early_stop_delta: float = 0.002
    fvmd_early_stop_window: int = 50
    vmaf_stride: int = 1


def _list_or_none(value: Optional[Sequence[Any]]) -> Optional[List[Any]]:
    if value is None:
        return None
    return list(value)


@contextlib.contextmanager
def _silence_console_output() -> Iterator[None]:
    """Redirect stdout/stderr to null device for noisy third-party calls."""

    try:
        with open(NULL_DEVICE, "w") as devnull, contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            yield
    except OSError:
        # If the null device cannot be opened, fall back to regular execution
        yield

_LPIPS_MODEL_CACHE: Dict[str, lpips.LPIPS] = {}


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
def get_package_dir(package_name: str) -> str:
    """
    Get the installation directory of a package.
    Falls back to local directory if package is not installed.
    """
    try:
        import importlib.util
        spec = importlib.util.find_spec(package_name)
        if spec and spec.origin:
            return os.path.dirname(spec.origin)
    except (ImportError, AttributeError):
        pass
    
    # Fallback to local directory (for backwards compatibility)
    return package_name


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

    return float(ssim(ref_y, dec_y, data_range=255, gaussian_weights=True))


def _block_ssim(ref_block: np.ndarray, dec_block: np.ndarray) -> float:
    """SSIM helper for block-wise comparisons."""

    height, width = ref_block.shape[:2]
    smallest_dim = min(height, width)

    if smallest_dim < 3:
        return 1.0

    if smallest_dim < 7:
        win_size = smallest_dim if smallest_dim % 2 == 1 else max(3, smallest_dim - 1)
    else:
        win_size = 7

    return float(
        ssim(
            ref_block,
            dec_block,
            data_range=255,
            gaussian_weights=True,
            channel_axis=-1,
            win_size=win_size,
        )
    )


def _generate_quality_visualizations(results: Dict, heatmaps_dir: str) -> None:
    """Generate comprehensive quality visualization charts."""

    if not results:
        return

    print("\nGenerating quality visualization charts...")

    video_names = list(results.keys())

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Quality Metrics Comparison (Foreground vs Background)', fontsize=16, fontweight='bold')

    metrics = [
        ('psnr_mean', 'PSNR (dB)', [20, 50]),
        ('ssim_mean', 'SSIM', [0, 1]),
        ('lpips_mean', 'LPIPS (lower is better)', [0, 0.5]),
        ('vmaf_mean', 'VMAF', [0, 100]),
    ]

    for idx, (metric_key, metric_label, ylim) in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]

        fg_values = [results[name]['foreground'].get(metric_key, 0) for name in video_names]
        bg_values = [results[name]['background'].get(metric_key, 0) for name in video_names]

        x = np.arange(len(video_names))
        width = 0.35

        bars1 = ax.bar(x - width / 2, fg_values, width, label='Foreground', alpha=0.8, color='#2E86AB')
        bars2 = ax.bar(x + width / 2, bg_values, width, label='Background', alpha=0.8, color='#A23B72')

        if metric_key in ['psnr_mean', 'vmaf_mean']:
            ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))
        else:
            ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.3f'))

        ax.set_ylabel(metric_label, fontsize=11, fontweight='bold')
        ax.set_title(metric_label, fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(video_names, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_ylim(ylim)

        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f'{height:.2f}',
                    ha='center',
                    va='bottom',
                    fontsize=8,
                )

    plt.tight_layout()
    output_path = os.path.join(heatmaps_dir, '1_overall_quality_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Generated quality comparison visualization in {heatmaps_dir}")


def _generate_patch_comparison(
    reference_frames: List[np.ndarray],
    decoded_frames_cache: Dict[str, List[np.ndarray]],
    results: Dict,
    fg_masks: Sequence[np.ndarray],
    heatmaps_dir: str,
    sample_frame_idx: Optional[int] = None,
    patch_size: int = 128,
) -> None:
    """Generate visual comparison of FG/BG patches for a representative frame."""

    if not results or not reference_frames:
        return

    if sample_frame_idx is None:
        sample_frame_idx = len(reference_frames) // 2

    print(f"  - Generating patch comparison visualization for frame {sample_frame_idx}...")

    ref_frame = reference_frames[sample_frame_idx]
    mask = fg_masks[sample_frame_idx].astype(np.uint8) * 255

    mask_binary = (mask > 127).astype(np.uint8)
    moments = cv2.moments(mask_binary)
    if moments['m00'] > 0:
        fg_center_x = int(moments['m10'] / moments['m00'])
        fg_center_y = int(moments['m01'] / moments['m00'])
    else:
        fg_center_x = ref_frame.shape[1] // 2
        fg_center_y = ref_frame.shape[0] // 2

    mask_inverted = 1 - mask_binary
    moments_bg = cv2.moments(mask_inverted)
    if moments_bg['m00'] > 0:
        bg1_center_x = int(moments_bg['m10'] / moments_bg['m00'])
        bg1_center_y = int(moments_bg['m01'] / moments_bg['m00'])
    else:
        bg1_center_x = ref_frame.shape[1] // 4
        bg1_center_y = ref_frame.shape[0] // 4

    bg2_center_x = ref_frame.shape[1] * 3 // 4
    bg2_center_y = ref_frame.shape[0] // 4
    if (
        abs(bg2_center_x - bg1_center_x) < patch_size
        and abs(bg2_center_y - bg1_center_y) < patch_size
    ):
        bg2_center_x = ref_frame.shape[1] // 4
        bg2_center_y = ref_frame.shape[0] * 3 // 4

    def extract_patch(frame: np.ndarray, center_x: int, center_y: int, size: int) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        h, w = frame.shape[:2]
        x1 = max(0, center_x - size // 2)
        y1 = max(0, center_y - size // 2)
        x2 = min(w, x1 + size)
        y2 = min(h, y1 + size)
        x1 = max(0, x2 - size)
        y1 = max(0, y2 - size)
        return frame[y1:y2, x1:x2], (x1, y1, x2, y2)

    ref_fg_patch, _ = extract_patch(ref_frame, fg_center_x, fg_center_y, patch_size)
    ref_bg1_patch, _ = extract_patch(ref_frame, bg1_center_x, bg1_center_y, patch_size)
    ref_bg2_patch, _ = extract_patch(ref_frame, bg2_center_x, bg2_center_y, patch_size)

    video_names = list(decoded_frames_cache.keys())
    num_methods = len(video_names)

    fig, axes = plt.subplots(3, num_methods + 1, figsize=(4 * (num_methods + 1), 12))
    fig.suptitle(
        f'Visual Patch Comparison (Frame {sample_frame_idx + 1})',
        fontsize=16,
        fontweight='bold',
    )

    axes[0, 0].imshow(cv2.cvtColor(ref_fg_patch, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('Reference\nForeground', fontsize=10, fontweight='bold')
    axes[0, 0].axis('off')

    axes[1, 0].imshow(cv2.cvtColor(ref_bg1_patch, cv2.COLOR_BGR2RGB))
    axes[1, 0].set_title('Reference\nBackground 1', fontsize=10, fontweight='bold')
    axes[1, 0].axis('off')

    axes[2, 0].imshow(cv2.cvtColor(ref_bg2_patch, cv2.COLOR_BGR2RGB))
    axes[2, 0].set_title('Reference\nBackground 2', fontsize=10, fontweight='bold')
    axes[2, 0].axis('off')

    for idx, video_name in enumerate(video_names):
        decoded_frames = decoded_frames_cache.get(video_name)
        if not decoded_frames or sample_frame_idx >= len(decoded_frames):
            continue

        decoded_frame = decoded_frames[sample_frame_idx]
        dec_fg_patch, _ = extract_patch(decoded_frame, fg_center_x, fg_center_y, patch_size)
        dec_bg1_patch, _ = extract_patch(decoded_frame, bg1_center_x, bg1_center_y, patch_size)
        dec_bg2_patch, _ = extract_patch(decoded_frame, bg2_center_x, bg2_center_y, patch_size)

        fg_vmaf = results[video_name]['foreground'].get('vmaf_mean', 0)
        bg_vmaf = results[video_name]['background'].get('vmaf_mean', 0)

        axes[0, idx + 1].imshow(cv2.cvtColor(dec_fg_patch, cv2.COLOR_BGR2RGB))
        axes[0, idx + 1].set_title(
            f'{video_name}\nFG VMAF: {fg_vmaf:.1f}',
            fontsize=10,
            fontweight='bold',
        )
        axes[0, idx + 1].axis('off')

        axes[1, idx + 1].imshow(cv2.cvtColor(dec_bg1_patch, cv2.COLOR_BGR2RGB))
        axes[1, idx + 1].set_title(
            f'{video_name}\nBG VMAF: {bg_vmaf:.1f}',
            fontsize=10,
            fontweight='bold',
        )
        axes[1, idx + 1].axis('off')

        axes[2, idx + 1].imshow(cv2.cvtColor(dec_bg2_patch, cv2.COLOR_BGR2RGB))
        axes[2, idx + 1].set_title(
            f'{video_name}\nBG VMAF: {bg_vmaf:.1f}',
            fontsize=10,
            fontweight='bold',
        )
        axes[2, idx + 1].axis('off')

    plt.tight_layout()
    output_path = os.path.join(
        heatmaps_dir,
        f'2_patch_comparison_frame_{sample_frame_idx + 1}.png',
    )
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print("  ✓ Generated patch comparison visualization")


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

    allowed = []
    for char in name.strip():
        if char.isalnum() or char in ('-', '_'):
            allowed.append(char)
        elif char.isspace() or char in ('/', '\\'):
            allowed.append('_')
        else:
            allowed.append('_')
    slug = ''.join(allowed).strip('_')
    return slug or 'video'

# Core functions

def calculate_target_bitrate(width: int, height: int, framerate: float, quality_factor: float = 1.0) -> int:
    """
    Calculate an appropriate target bitrate based on video characteristics.
    
    Args:
        width: Video width
        height: Video height  
        framerate: Video framerate
        quality_factor: Quality multiplier (1.0 = standard, higher = better quality)
        
    Returns:
        Bitrate in bits per second (integer)
    """
    # Basic bitrate calculation: pixels per second * bits per pixel
    pixels_per_second = width * height * framerate
    
    # Base bits per pixel for  (typically 0.1-0.2 for good quality)
    bits_per_pixel = 0.01 * quality_factor
    
    # Calculate target bitrate in bps
    target_bps = int(pixels_per_second * bits_per_pixel)
    
    return target_bps

def normalize_array(arr: np.ndarray) -> np.ndarray:
    """Normalizes a NumPy array to the range [0, 1]."""
    min_val = arr.min()
    max_val = arr.max()
    if max_val - min_val > 0:
        return (arr - min_val) / (max_val - min_val)
    return arr

def calculate_removability_scores(raw_video_file: str, reference_frames_folder: str, width: int, height: int, block_size: int, alpha: float = 0.5, working_dir: str = ".", smoothing_beta: float = 1) -> np.ndarray:
    """
    This function computes a "removability score" by running EVCA for complexity analysis
    and UFO for object detection, then combining the results. Higher scores mean the block is a
    better candidate for removal (e.g., background, low complexity).

    Args:
        raw_video_file: Path to the raw YUV video file.
        reference_frames_folder: Path to the folder containing reference frames.
        width: The width of the original video frame.
        height: The height of the original video frame.
        block_size: The size of each block in pixels.
        alpha: The weight for combining spatial and temporal scores.
        working_dir: Working directory for temporary files (default: current directory).
        smoothing_beta: Smoothing factor for temporal smoothing. 1.0 = no smoothing, 0.0 = only previous frame (default: 1).

    Returns:
        A 3D NumPy array of shape (num_frames, num_blocks_y, num_blocks_x)
        containing the final removability scores normalized to [0, 1].
    """
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
    """
    Encodes a video using a two-pass process with libx265.
    
    Args:
        input_frames_dir: Directory containing input frames (e.g., '%05d.png').
        output_video: The path for the final encoded video file.
        framerate: The framerate of the output video. If None, encode losslessly.
        width: The width of the video.
        height: The height of the video.
        target_bitrate: The target bitrate for lossy encoding in bits per second.
        preset: Encoding preset (default: "medium" for good speed/quality balance)
        **extra_params: Additional x265 parameters to append (e.g., qpfile path).
    """
    temp_dir = os.path.dirname(output_video) or '.'
    os.makedirs(temp_dir, exist_ok=True)
    
    passlog_file = os.path.join(temp_dir, f"ffmpeg_2pass_log_{os.path.basename(output_video)}")
    
    # Use cached platform-specific null device
    null_device = NULL_DEVICE
    
    try:
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
            pass1_cmd = base_cmd + [
                "-c:v", "libx265",
                "-preset", preset,
                "-x265-params", f"{x265_base_params}:pass=1:stats={passlog_file}",
                "-f", "mp4", "-y", null_device
            ]
            subprocess.run(pass1_cmd, check=True, capture_output=True, text=True)
            
            # Pass 2
            pass2_params = f"{x265_base_params}:pass=2:stats={passlog_file}"
            if extra_params:
                for key, value in extra_params.items():
                    pass2_params += f":{key}={value}"
            
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
            pass1_cmd = base_cmd + [
                "-c:v", "libx265",
                "-b:v", str(target_bitrate),
                "-minrate", str(int(target_bitrate * 0.9)),
                "-maxrate", str(int(target_bitrate * 1.1)),
                "-bufsize", str(target_bitrate),
                "-preset", preset,
                "-g", str(framerate),  # Set GOP size to framerate for approx 1-second keyframes
                "-x265-params", f"pass=1:stats={passlog_file}",
                "-f", "mp4", "-y", null_device
            ]
            subprocess.run(pass1_cmd, check=True, capture_output=True, text=True)
            
            # Pass 2
            pass2_params = f"pass=2:stats={passlog_file}"
            if extra_params:
                for key, value in extra_params.items():
                    pass2_params += f":{key}={value}"
            
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
    """
    Decodes a video file to PNG frames with proper color space handling.
    
    Args:
        video_path: Path to the input video file
        output_dir: Directory where decoded frames will be saved
        framerate: Optional framerate to decode at (if None, uses video's native framerate)
        start_number: Starting number for output frames (default: 1)
        quality: JPEG quality for PNG encoding, 1-31 where lower is better (default: 1)
    
    Returns:
        True if successful, False otherwise
    """
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
    """
    Selects and removes blocks from a single image based on removability scores.

    This function combines the logic of:
    1. Selecting the N blocks with the highest scores in each row.
    2. Splitting the image into blocks.
    3. Removing the selected blocks.
    4. Reconstructing the image from the remaining blocks.

    Args:
        image: The original image for the frame (H, W, C).
        frame_scores: The 2D array of final scores for this frame (num_blocks_y, num_blocks_x).
        block_size: The side length (l) of each block block.
        shrink_amount: The number of blocks to remove per row. If < 1, it's a percentage.

    Returns:
        A tuple containing:
        - new_image: The reconstructed image with blocks removed.
        - removal_mask: A 2D array indicating removed (1) vs. kept (0) blocks.
        - block_coords_to_remove: The list of lists of column indices that were removed.
    """
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
    """
    Combines a 5D array of blocks back into a single image.
    This is the exact inverse of split_image_into_blocks.
    """
    num_blocks_y, num_blocks_x, block_size, _, c = blocks.shape
    
    # The crucial inverse transpose and reshape operations
    image = blocks.swapaxes(1, 2)
    image = image.reshape(num_blocks_y * block_size, num_blocks_x * block_size, c)
    return image

def stretch_frame(shrunk_frame: np.ndarray, binary_mask: np.ndarray, block_size: int) -> np.ndarray:
    """
    Recreates a full-resolution video frame from a shrunk version and a removal mask.

    Args:
        shrunk_frame: The shrunk image, containing only the "kept" blocks stitched together.
        binary_mask: A 2D array of 0s and 1s where 1 indicates a removed block.
        block_size: The side length (l) of each block.

    Returns:
        The reconstructed full-resolution frame.
    """
    # 1. Get the dimensions of the final frame from the mask
    num_blocks_y, num_blocks_x = binary_mask.shape
    # Assuming 3 color channels (e.g., RGB)
    channels = shrunk_frame.shape[2] 

    # 2. Create a "canvas" of final blocks, initialized to all black
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
    propainter_dir: str = None,
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
    """
    Uses ProPainter to inpaint stretched frames with removed blocks.
    
    This function:
    1. Creates a temporary video from stretched frames
    2. Creates a mask video from removal masks
    3. Runs ProPainter inference
    4. Extracts inpainted frames back to the output directory
    
    Args:
        stretched_frames_dir: Directory containing stretched frames with black regions
        removal_masks_dir: Directory containing binary mask images (white = regions to inpaint)
        output_frames_dir: Directory where inpainted frames will be saved
        width: Video width
        height: Video height
        framerate: Video framerate
        propainter_dir: Path to ProPainter directory (default: "ProPainter")
        resize_ratio: Resize scale for processing (default: 1.0)
        ref_stride: Stride of global reference frames (default: 10)
        neighbor_length: Length of local neighboring frames (default: 10)
        subvideo_length: Length of sub-video for long video inference (default: 80)
        mask_dilation: Mask dilation for video and flow masking (default: 4)
    raft_iter: Iterations for RAFT inference (default: 20)
    fp16: Use fp16 (half precision) during inference (default: True)
    devices: Optional sequence of compute devices for parallel jobs. Defaults to all available GPUs (CPU fallback).
    parallel_chunk_length: Optional frames per ProPainter chunk when running in parallel. Defaults to subvideo_length.
    chunk_overlap: Optional overlapping frames between parallel chunks. Defaults to neighbor_length.
    """
    
    # Save current directory
    original_dir = os.getcwd()
    
    # Get absolute paths
    stretched_frames_abs = os.path.abspath(stretched_frames_dir)
    removal_masks_abs = os.path.abspath(removal_masks_dir)
    output_frames_abs = os.path.abspath(output_frames_dir)
    output_frames_path = Path(output_frames_abs)
    os.makedirs(output_frames_abs, exist_ok=True)

    # Clear out stale frames so we can detect fresh output from E2FGVI
    for stale_frame in output_frames_path.glob("*.png"):
        if stale_frame.is_file():
            stale_frame.unlink()

    use_installed_package = propainter_dir is None
    if use_installed_package:
        try:
            import propainter as _propainter  # noqa: F401
        except ImportError as exc:
            raise RuntimeError("propainter package is not available. Install it or provide propainter_dir.") from exc
    else:
        propainter_dir = os.path.abspath(propainter_dir)

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

        if use_installed_package:
            propainter_entry = [
                sys.executable,
                "-m",
                "propainter.inference_propainter",
            ]
            run_cwd = original_dir
        else:
            propainter_entry = [
                sys.executable,
                os.path.join(propainter_dir, "inference_propainter.py"),
            ]
            run_cwd = propainter_dir

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

def inpaint_with_e2fgvi(stretched_frames_dir: str, removal_masks_dir: str, output_frames_dir: str, width: int, height: int, framerate: float, e2fgvi_dir: str = None, model: str = "e2fgvi_hq", ckpt: str = None, ref_stride: int = 10, neighbor_stride: int = 5, num_ref: int = -1, mask_dilation: int = 4) -> None:
    """
    Uses E2FGVI to inpaint stretched frames with removed blocks.
    
    This function:
    1. Runs E2FGVI test.py directly on stretched frames and masks
    2. Extracts inpainted frames to the output directory
    
    Args:
        stretched_frames_dir: Directory containing stretched frames with black regions
        removal_masks_dir: Directory containing binary mask images (white = regions to inpaint)
        output_frames_dir: Directory where inpainted frames will be saved
        width: Video width
        height: Video height
        framerate: Video framerate
        e2fgvi_dir: Path to E2FGVI directory (default: "E2FGVI")
        model: Model type ('e2fgvi' or 'e2fgvi_hq', default: 'e2fgvi_hq')
        ckpt: Path to model checkpoint (default: None, uses default path)
        ref_stride: Stride of reference frames (default: 10)
        neighbor_stride: Stride of neighboring frames (default: 5)
        num_ref: Number of reference frames (default: -1 for all)
        mask_dilation: Mask dilation iterations (default: 4)
    """
    
    stretched_frames_abs = os.path.abspath(stretched_frames_dir)
    removal_masks_abs = os.path.abspath(removal_masks_dir)
    output_frames_abs = os.path.abspath(output_frames_dir)

    use_installed_package = e2fgvi_dir is None

    if use_installed_package:
        try:
            import e2fgvi as e2fgvi_pkg  # type: ignore[import-untyped]
        except ImportError as exc:
            raise RuntimeError(
                "E2FGVI package is not available. Install it with `pip install e2fgvi`."
            ) from exc
        package_dir = Path(e2fgvi_pkg.__file__).resolve().parent
        e2fgvi_cmd = [
            sys.executable,
            "-m",
            "e2fgvi",
        ]
    else:
        repo_candidate = Path(e2fgvi_dir).resolve()
        if repo_candidate.is_file():
            repo_candidate = repo_candidate.parent
        if (repo_candidate / "e2fgvi" / "__init__.py").exists():
            package_dir = repo_candidate / "e2fgvi"
            repo_root = repo_candidate
        elif (repo_candidate / "__init__.py").exists():
            package_dir = repo_candidate
            repo_root = repo_candidate.parent
        else:
            raise RuntimeError(f"Invalid E2FGVI directory: {e2fgvi_dir}")
        test_script = (repo_root / "test.py").resolve()
        if not test_script.exists():
            raise RuntimeError(f"E2FGVI test entry point not found at {test_script}")
        e2fgvi_cmd = [
            sys.executable,
            str(test_script),
        ]

    if ckpt is None:
        ckpt_path = package_dir / "release_model" / "E2FGVI-HQ-CVPR22.pth"
    else:
        ckpt_path = Path(ckpt).resolve()

    e2fgvi_cmd.extend([
        "--model",
        model,
        "--video",
        stretched_frames_abs,
        "--mask",
        removal_masks_abs,
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
        output_frames_abs,
    ])

    print("Running E2FGVI inference...")
    result = subprocess.run(e2fgvi_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"E2FGVI stdout: {result.stdout}")
        print(f"E2FGVI stderr: {result.stderr}")
        raise RuntimeError(f"E2FGVI inference failed: {result.stderr}")

    print(f"E2FGVI output: {result.stdout}")

    generated_frames = list(Path(output_frames_abs).glob("*.png"))
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

    if not decode_video(str(result_video_path), output_frames_abs, framerate=framerate, start_number=1, quality=1):
        raise RuntimeError(f"Failed to decode E2FGVI result video: {result_video_path}")

    print(f"E2FGVI inpainted frames saved to {output_frames_abs}")

    # Remove the mp4 artefact so subsequent runs start cleanly
    try:
        result_video_path.unlink()
    except OSError:
        pass

# Elvis v2 functions

def encode_with_roi(input_frames_dir: str, output_video: str, removability_scores: np.ndarray, block_size: int, framerate: float, width: int, height: int, target_bitrate: int = 1000000, save_qp_maps: bool = False, qp_maps_dir: str = None) -> None:
    """
    Encodes a video using a two-pass process with a detailed qpfile for per-block QP control.

    This method provides the ultimate control:
    1.  **Per-Block Quality:** Assigns a specific Quantization Parameter (QP) to each
        individual block in every frame based on its removability score.
    2.  **Bitrate Adherence:** Uses a standard two-pass encode to ensure the final
        video file accurately meets the target bitrate.

    Args:
        input_frames_dir: Directory containing input frames (e.g., '%05d.png').
        output_video: The path for the final encoded video file.
        removability_scores: A 3D numpy array of shape (num_frames, num_blocks_y, num_blocks_x)
                             containing the importance score for each block.
        block_size: The side length of the square blocks used for scoring (e.g., 64).
                    This must match the block size used to generate the scores.
        framerate: The framerate of the output video.
        width: The width of the video.
        height: The height of the video.
        target_bitrate: The target bitrate for the video in bits per second.
        save_qp_maps: Whether to save QP maps as grayscale images for debugging.
        qp_maps_dir: Directory to save QP maps. If None, uses temp_dir/qp_maps.
    """
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

        # Map normalized scores [0, 1] to QP values [1, 50]
        qp_maps = np.round(removability_scores * 49 + 1).astype(int)

        # Generate qpfile with optimized string building
        with open(qpfile_path, 'w') as f:
            for frame_idx in range(num_frames):
                # Start the line with frame index and a generic frame type ('P')
                # The '-1' for QP indicates we are providing per-block QPs
                line_parts = [f"{frame_idx} P -1"]
                
                # Append QP for each block in raster-scan order - build list then join once
                qp_frame = qp_maps[frame_idx]
                block_qps = [f"{bx},{by},{qp_frame[by, bx]}" 
                            for by in range(num_blocks_y) 
                            for bx in range(num_blocks_x)]
                line_parts.extend(block_qps)

                f.write(" ".join(line_parts) + "\n")
        print(f"qpfile generated at {qpfile_path}")
        
        # --- Save QP maps as images if requested ---
        if save_qp_maps:
            if qp_maps_dir is None:
                qp_maps_dir = os.path.join(temp_dir, "qp_maps")
            os.makedirs(qp_maps_dir, exist_ok=True)
            for frame_idx in range(num_frames):
                qp_map_image = (qp_maps[frame_idx] / 50.0 * 255).astype(np.uint8)  # Scale to [0, 255]
                cv2.imwrite(os.path.join(qp_maps_dir, f"qp_map_{frame_idx:05d}.png"), qp_map_image)
            print(f"QP maps saved to {qp_maps_dir}")

        # --- Part 2: Run Global Two-Pass Encode with QP file ---
        # x265 requires CTU size to be 16, 32, or 64. Map block_size to nearest valid CTU.
        # The qpfile will still use the block_size grid, but CTU sets the encoding block size.
        valid_ctu_sizes = [16, 32, 64]
        ctu_size = min(valid_ctu_sizes, key=lambda x: abs(x - block_size))
        
        print("Starting two-pass encoding with per-block QP control...")
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
    """
    Simplifies a frame by adaptively downsampling each block.
    Removability scores control the downsampling factor.
    Args:
        image: The original image for the frame (H, W, C).
        frame_scores: The 2D array of final scores for this frame (num_blocks_y, num_blocks_x).
        block_size: The side length (l) of each block.
    Returns:
        The reconstructed image with blocks adaptively downsampled.
        The downsample maps (for encoding).
    """
    # Split image into blocks
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
# TODO: not in use currently, either implement or remove
def filter_frame_bilateral(image: np.ndarray, frame_scores: np.ndarray, block_size: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Applies adaptive bilateral filtering. Higher scores indicate more aggressive filtering.
    Args:
        image: The original image for the frame (H, W, C).
        frame_scores: The 2D array of final scores for this frame (num_blocks_y, num_blocks_x).
        block_size: The side length (l) of each block.
    Returns:
        The reconstructed image with blocks adaptively filtered.
        The filtering strengths (for encoding).
    """
    # Split image into blocks
    blocks = split_image_into_blocks(image, block_size)

    # Map removability scores to filtering strengths (0 to 15)
    filter_strengths = np.round(frame_scores * 15).astype(np.int32)

    # Apply bilateral filter to each block based on its strength
    processed_blocks = np.zeros_like(blocks)
    for by in range(blocks.shape[0]):
        for bx in range(blocks.shape[1]):
            block = blocks[by, bx]
            strength = filter_strengths[by, bx]
            if strength > 0:
                # Apply bilateral filter
                filtered_block = cv2.bilateralFilter(block, d=5, sigmaColor=strength*10, sigmaSpace=strength*10)
                processed_blocks[by, bx] = filtered_block
            else:
                # No filtering
                processed_blocks[by, bx] = block

    # Reconstruct the final image from the processed blocks
    new_image = combine_blocks_into_image(processed_blocks)

    return new_image, filter_strengths

def filter_frame_gaussian(image: np.ndarray, frame_scores: np.ndarray, block_size: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Applies adaptive Gaussian blurring. Higher scores indicate more rounds of blurring.
    Args:
        image: The original image for the frame (H, W, C).
        frame_scores: The 2D array of scores for this frame (num_blocks_y, num_blocks_x).
        block_size: The side length (l) of each block.
    Returns:
        The reconstructed image with blocks adaptively blurred.
        The blurring strengths (for encoding).
    """
    # Split image into blocks
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
# TODO: check how quality is impacted by different bitrates
def encode_strength_maps(strength_maps: np.ndarray, output_video: str, framerate: float, target_bitrate: int = 50000) -> None:
    """
    Encodes strength maps as a grayscale video for compression.
    
    This allows leveraging video codecs to compress the strength maps efficiently.
    The maps are kept at block resolution (one pixel per block) and encoded as grayscale.
    
    Args:
        strength_maps: 3D array of shape (num_frames, num_blocks_y, num_blocks_x)
        output_video: Path to output video file (should be in maps directory)
        framerate: Video framerate
        target_bitrate: Target bitrate for lossy compression (default: 50kbps, sufficient for small maps)
    """
    # Normalize strength maps to 0-255 for encoding
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
# TODO: apply slight median or gaussian filter before normalization to reduce compression artifacts?
def decode_strength_maps(video_path: str, block_size: int, frames_dir: str) -> np.ndarray:
    """
    Decodes strength maps from a compressed video and attempts to revert encoding artifacts.
    Min/max values are set:
    For gaussian filtering: min=0, max=10
    For downsampling: min=0, max=int(log2(block_size))
    
    Args:
        video_path: Path to encoded strength maps video
        block_size: The side length of each block (used to determine max downsampling)
        frames_dir: Directory to save decoded frames

    Returns:
        3D array of strength values with shape (num_frames, num_blocks_y, num_blocks_x)
    """
    # Figure out whether it's gaussian or downsampling based on filename
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

def upscale_realesrgan_2x(image: np.ndarray, realesrgan_dir: str = None, temp_dir: str = None) -> np.ndarray:
    """
    Applies Real-ESRGAN 2x upscaling to an image via the installed package entry point.

    Falls back to a local repository if the package is unavailable.

    Args:
        image: Input image (H, W, C) in BGR format
        realesrgan_dir: Optional path to a local Real-ESRGAN checkout (legacy fallback)
        temp_dir: Temporary directory for intermediate files (if None, creates one)

    Returns:
        Upscaled image (2*H, 2*W, C) in BGR format
    """
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
    model_path: Optional[str] = None,
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

    resolved_model_path: Union[str, List[str]]
    if model_path is not None:
        resolved_model_path = model_path
    else:
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
        resolved_model_path = str(existing)

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
    model_path: Optional[str] = None,
) -> None:
    """Parallel adaptive Real-ESRGAN restoration over a directory of frames."""

    input_path = Path(input_frames_dir)
    output_path = Path(output_frames_dir)
    if not input_path.is_dir():
        raise ValueError(f"Input frames directory does not exist: {input_frames_dir}")
    output_path.mkdir(parents=True, exist_ok=True)

    # Clear old outputs to avoid mixing stale data
    for pattern in ('*.png', '*.jpg', '*.jpeg'):
        for stale_file in output_path.glob(pattern):
            if stale_file.is_file():
                stale_file.unlink()

    frame_files = sorted(
        [path for path in input_path.iterdir() if path.suffix.lower() in ('.png', '.jpg', '.jpeg')]
    )
    if not frame_files:
        raise ValueError(f"No frames found in {input_frames_dir}")

    downscale_maps = np.asarray(downscale_maps)
    num_frames = len(frame_files)
    if downscale_maps.shape[0] != num_frames:
        raise ValueError(
            f"Downscale maps length ({downscale_maps.shape[0]}) does not match frame count ({num_frames})."
        )

    resolved_devices = _resolve_device_list(devices, prefer_cuda=True, allow_cpu_fallback=True)
    per_device_workers = max(1, int(per_device_workers))

    device_slots: List[Tuple[torch.device, int]] = []
    for device_obj in resolved_devices:
        for slot_id in range(per_device_workers):
            device_slots.append((device_obj, slot_id))

    if not device_slots:
        raise RuntimeError("No compute devices available for Real-ESRGAN restoration.")

    if parallel_chunk_length is None or parallel_chunk_length <= 0:
        parallel_chunk_length = max(1, math.ceil(num_frames / len(device_slots)))

    effective_chunk = max(1, min(parallel_chunk_length, num_frames))

    class _RealesrganChunk(NamedTuple):
        job_id: int
        start: int
        end: int

    chunks: List[_RealesrganChunk] = []
    cursor = 0
    job_id = 0
    while cursor < num_frames:
        end = min(num_frames, cursor + effective_chunk)
        chunks.append(_RealesrganChunk(job_id=job_id, start=cursor, end=end))
        job_id += 1
        cursor = end

    device_summary = ", ".join(str(dev) for dev in resolved_devices)
    tile_desc = str(tile) if tile and tile > 0 else 'full-frame'
    print(
        f"  Using Real-ESRGAN on devices: {device_summary} | tile: {tile_desc} | per-device workers: {per_device_workers}"
    )
    print(f"  Chunk length: {effective_chunk} | total frames: {num_frames} | total chunks: {len(chunks)}")

    output_paths = [output_path / path.name for path in frame_files]

    upsampler_cache: Dict[str, "RealESRGANer"] = {}
    upsampler_lock = threading.Lock()

    def _get_upsampler(device_obj: torch.device, slot_id: int) -> "RealESRGANer":
        key = _device_slot_key(device_obj, slot_id)
        with upsampler_lock:
            upsampler = upsampler_cache.get(key)
            if upsampler is None:
                print(f"    -> Warming Real-ESRGAN runtime on {key}...")
                upsampler = _instantiate_realesrgan_upsampler(
                    model_name=model_name,
                    device=device_obj,
                    denoise_strength=denoise_strength,
                    tile=tile,
                    tile_pad=tile_pad,
                    pre_pad=pre_pad,
                    fp32=fp32,
                    model_path=model_path,
                )
                upsampler_cache[key] = upsampler
        return upsampler

    def _process_chunk(chunk: _RealesrganChunk, device_obj: torch.device, slot_id: int) -> None:
        upsampler = _get_upsampler(device_obj, slot_id)
        slot_label = _format_device_slot(device_obj, slot_id)
        print(
            f"    -> Real-ESRGAN chunk {chunk.job_id + 1}/{len(chunks)} frames {chunk.start + 1}-{chunk.end} on {slot_label}"
        )

        for frame_index in range(chunk.start, chunk.end):
            frame_path = frame_files[frame_index]
            frame = cv2.imread(str(frame_path))
            if frame is None:
                raise RuntimeError(f"Failed to load frame: {frame_path}")

            def _enhance_once(img: np.ndarray) -> np.ndarray:
                return _upsample_with_realesrgan(upsampler, img, device_obj=device_obj, outscale=2.0)

            restored = upscale_realesrgan_adaptive(
                frame,
                downscale_maps[frame_index],
                block_size,
                upsample_fn=_enhance_once,
            )

            output_path_frame = output_paths[frame_index]
            if not cv2.imwrite(str(output_path_frame), restored):
                raise RuntimeError(f"Failed to write restored frame: {output_path_frame}")

    max_workers = min(len(device_slots), len(chunks))
    if max_workers <= 0:
        return

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        chunk_iter = iter(chunks)
        while True:
            tasks = []
            for device_obj, slot_id in device_slots:
                chunk = next(chunk_iter, None)
                if chunk is None:
                    break
                tasks.append(executor.submit(_process_chunk, chunk, device_obj, slot_id))

            if not tasks:
                break

            for future in tasks:
                future.result()

# OpenCV-based restoration benchmarks for Elvis v2

def restore_downsample_opencv_bilinear(downsampled_image: np.ndarray, downscale_maps: np.ndarray, block_size: int) -> np.ndarray:
    """
    Restores a downsampled image using OpenCV's bilinear interpolation.
    This is a simple client-side restoration benchmark that doesn't require any ML models.
    
    Args:
        downsampled_image: The downsampled image (non-uniform block sizes) in BGR format
        downscale_maps: 2D array (num_blocks_y, num_blocks_x) indicating the downscale factor applied to each block
        block_size: The side length of each block in the original resolution
    
    Returns:
        The restored image at original resolution using bilinear upscaling
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
    
    # Upscale each block individually using bilinear interpolation
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
                # Upscale back using bilinear interpolation
                restored_block = cv2.resize(small_block, (block_size, block_size), interpolation=cv2.INTER_LINEAR)
                restored_blocks[i, j] = restored_block
            else:
                # Block was not downsampled
                restored_blocks[i, j] = blocks[i, j]
    
    return combine_blocks_into_image(restored_blocks)

def restore_downsample_opencv_bicubic(downsampled_image: np.ndarray, downscale_maps: np.ndarray, block_size: int) -> np.ndarray:
    """
    Restores a downsampled image using OpenCV's bicubic interpolation.
    This is a simple client-side restoration benchmark that doesn't require any ML models.
    
    Args:
        downsampled_image: The downsampled image (non-uniform block sizes) in BGR format
        downscale_maps: 2D array (num_blocks_y, num_blocks_x) indicating the downscale factor applied to each block
        block_size: The side length of each block in the original resolution
    
    Returns:
        The restored image at original resolution using bicubic upscaling
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
    
    # Upscale each block individually using bicubic interpolation
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
                # Upscale back using bicubic interpolation
                restored_block = cv2.resize(small_block, (block_size, block_size), interpolation=cv2.INTER_CUBIC)
                restored_blocks[i, j] = restored_block
            else:
                # Block was not downsampled
                restored_blocks[i, j] = blocks[i, j]
    
    return combine_blocks_into_image(restored_blocks)

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

def restore_blur_opencv_wiener_approximation(blurred_image: np.ndarray, blur_maps: np.ndarray, block_size: int) -> np.ndarray:
    """
    Restores a blurred image using a Wiener filter approximation with OpenCV.
    This uses a combination of deconvolution and denoising to restore sharpness.
    
    Since OpenCV doesn't have a direct Wiener filter, we approximate it using:
    1. High-pass filtering to enhance edges
    2. Adaptive histogram equalization to improve contrast
    3. Bilateral filtering to reduce noise while preserving edges
    
    Args:
        blurred_image: The blurred image in BGR format
        blur_maps: 2D array (num_blocks_y, num_blocks_x) indicating blur rounds applied to each block
        block_size: The side length of each block
    
    Returns:
        The restored image with Wiener-like deconvolution applied
    """
    # Split image into blocks
    num_blocks_y, num_blocks_x = blur_maps.shape
    blocks = split_image_into_blocks(blurred_image, block_size)
    
    # Apply restoration to each block based on its blur strength
    restored_blocks = np.zeros_like(blocks)
    for i in range(num_blocks_y):
        for j in range(num_blocks_x):
            block = blocks[i, j]
            blur_strength = int(blur_maps[i, j])
            
            if blur_strength > 0:
                # Apply adaptive sharpening
                # 1. Convert to LAB color space for better results
                lab = cv2.cvtColor(block, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                
                # 2. Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to L channel
                clahe = cv2.createCLAHE(clipLimit=2.0 * blur_strength, tileGridSize=(8, 8))
                l = clahe.apply(l)
                
                # 3. Merge channels and convert back to BGR
                enhanced = cv2.merge([l, a, b])
                enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
                
                # 4. Apply bilateral filter to reduce noise while preserving edges
                sigma = max(5, blur_strength * 5)
                denoised = cv2.bilateralFilter(enhanced, d=5, sigmaColor=sigma, sigmaSpace=sigma)
                
                # 5. Sharpen using unsharp mask
                amount = blur_strength * 0.3
                blurred_temp = cv2.GaussianBlur(denoised, (0, 0), max(1, blur_strength * 0.5))
                sharpened = cv2.addWeighted(denoised, 1.0 + amount, blurred_temp, -amount, 0)
                
                restored_blocks[i, j] = np.clip(sharpened, 0, 255).astype(np.uint8)
            else:
                # No blurring was applied
                restored_blocks[i, j] = block
    
    return combine_blocks_into_image(restored_blocks)

def restore_with_instantir_adaptive(
    input_frames_dir: str,
    blur_maps: np.ndarray,
    block_size: int,
    instantir_weights_dir: Optional[str] = None,
    cfg: float = 7.0,
    creative_start: float = 1.0,
    preview_start: float = 0.0,
    seed: Optional[int] = 42,
    devices: Optional[Sequence[Union[int, str, torch.device]]] = None,
    batch_size: int = 4,
    parallel_chunk_length: Optional[int] = None,
) -> None:
    """Apply adaptive InstantIR blind restoration with multi-device chunked execution."""

    if batch_size < 1:
        raise ValueError("`batch_size` must be at least 1.")

    print("  Loading InstantIR runtime(s)...")

    weights_dir = Path(instantir_weights_dir or "./InstantIR/models").expanduser()
    weights_dir.mkdir(parents=True, exist_ok=True)

    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    resolved_devices = _resolve_device_list(devices, prefer_cuda=True, allow_cpu_fallback=True)

    runtime_cache: Dict[str, InstantIRRuntime] = {}
    runtime_lock = threading.Lock()

    def _device_key(device_obj: torch.device) -> str:
        if device_obj.type == "cuda":
            idx = device_obj.index if device_obj.index is not None else 0
            return f"cuda:{idx}"
        return str(device_obj)

    def _get_runtime(device_obj: torch.device) -> InstantIRRuntime:
        key = _device_key(device_obj)
        with runtime_lock:
            runtime = runtime_cache.get(key)
            if runtime is None:
                print(f"    -> Warming InstantIR runtime on {key}...")
                dtype = torch.float16 if device_obj.type == "cuda" else torch.float32
                with _silence_console_output():
                    runtime = load_runtime(
                        instantir_path=weights_dir,
                        device=device_obj,
                        torch_dtype=dtype,
                        map_location="cpu",
                    )
                if hasattr(runtime, "pipe") and hasattr(runtime.pipe, "set_progress_bar_config"):
                    runtime.pipe.set_progress_bar_config(disable=True)
                runtime_cache[key] = runtime
        return runtime

    devices_summary = ", ".join(str(dev) for dev in resolved_devices)
    print(
        f"  Using InstantIR on devices: {devices_summary} | batch size per device: {batch_size}"
    )

    frames_files = sorted([f for f in os.listdir(input_frames_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
    num_frames = len(frames_files)
    if num_frames == 0:
        raise ValueError(f"No frames found in {input_frames_dir}")
    if num_frames != blur_maps.shape[0]:
        raise ValueError(
            f"Number of frames ({num_frames}) doesn't match blur_maps shape ({blur_maps.shape[0]})"
        )

    frames_images = [cv2.imread(os.path.join(input_frames_dir, f)) for f in frames_files]
    frames_blocks = np.stack([split_image_into_blocks(frame, block_size) for frame in frames_images], axis=0)

    if np.max(blur_maps) == 0:
        print("  No blurring detected, skipping restoration.")
        return

    effective_chunk_length = parallel_chunk_length or num_frames
    effective_chunk_length = max(1, min(effective_chunk_length, num_frames))
    effective_batch = max(1, batch_size)

    class _InstantIRChunk(NamedTuple):
        job_id: int
        start: int
        end: int

    chunks: List[_InstantIRChunk] = []
    cursor = 0
    job_id = 0
    while cursor < num_frames:
        end = min(num_frames, cursor + effective_chunk_length)
        chunks.append(_InstantIRChunk(job_id=job_id, start=cursor, end=end))
        job_id += 1
        cursor = end

    print(
        f"  Chunk length: {effective_chunk_length} | total frames: {num_frames} | total chunks: {len(chunks)}"
    )

    def _iter_batches(indices: Sequence[int], batch_len: int):
        iterator = iter(indices)
        while True:
            batch = list(islice(iterator, batch_len))
            if not batch:
                break
            yield batch

    def _process_chunk(chunk: _InstantIRChunk, device_obj: torch.device) -> List[Tuple[int, np.ndarray]]:
        runtime = _get_runtime(device_obj)
        local_length = chunk.end - chunk.start
        chunk_frames = [frames_images[idx].copy() for idx in range(chunk.start, chunk.end)]
        chunk_blur_maps = blur_maps[chunk.start:chunk.end].astype(np.int32).copy()
        chunk_original_blocks = frames_blocks[chunk.start:chunk.end]
        max_rounds = int(chunk_blur_maps.max())

        print(
            f"    -> InstantIR chunk {chunk.job_id + 1}/{len(chunks)} "
            f"frames {chunk.start + 1}-{chunk.end} on {device_obj} (rounds: {max_rounds})"
        )

        if max_rounds <= 0:
            return [(chunk.start + idx, chunk_frames[idx]) for idx in range(local_length)]

        for round_idx in range(max_rounds):
            active_indices = [idx for idx in range(local_length) if np.any(chunk_blur_maps[idx] > 0)]
            if not active_indices:
                break

            print(
                f"       Round {round_idx + 1}/{max_rounds}: processing {len(active_indices)} frame(s)"
            )

            for batch_indices in _iter_batches(active_indices, effective_batch):
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

        return [(chunk.start + idx, chunk_frames[idx]) for idx in range(local_length)]

    updated_frames: Dict[int, np.ndarray] = {}

    max_workers = min(len(resolved_devices), len(chunks))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        chunk_iter = iter(chunks)
        while True:
            tasks = []
            for device_obj in resolved_devices:
                chunk = next(chunk_iter, None)
                if chunk is None:
                    break
                tasks.append(executor.submit(_process_chunk, chunk, device_obj))

            if not tasks:
                break

            for future in tasks:
                for frame_idx, updated_frame in future.result():
                    updated_frames[frame_idx] = updated_frame

    for idx in range(num_frames):
        frames_images[idx] = updated_frames.get(idx, frames_images[idx])

    for i, frame in enumerate(frames_images):
        cv2.imwrite(os.path.join(input_frames_dir, frames_files[i]), frame)

    print(f"  Adaptive InstantIR restoration complete. Frames saved to {input_frames_dir}")
        
# TODO: not used anymore, but we should bring back the blockwise figures at some point
def calculate_blockwise_metric(ref_blocks: np.ndarray, test_blocks: np.ndarray, metric_func: Callable[..., float]) -> np.ndarray:
    """
    Applies a given metric function to each pair of blocks in parallel.
    """
    num_blocks_y, num_blocks_x, block_h, block_w, channels = ref_blocks.shape
    
    # Reshape for easy iteration: (total_num_blocks, h, w, c)
    ref_blocks_list = ref_blocks.reshape(-1, block_h, block_w, channels)
    test_blocks_list = test_blocks.reshape(-1, block_h, block_w, channels)
    
    # Create pairs of (ref_block, test_block) for starmap
    block_pairs = zip(ref_blocks_list, test_blocks_list)
    
    # Use a multiprocessing pool to parallelize the metric calculation
    # By default, this uses all available CPU cores.
    spawn_context = None
    if hasattr(multiprocessing, "get_context"):
        try:
            spawn_context = multiprocessing.get_context("spawn")
        except ValueError:
            spawn_context = None
    pool_cls = spawn_context.Pool if spawn_context is not None else multiprocessing.Pool

    with pool_cls() as pool:
        scores_flat = pool.starmap(metric_func, block_pairs)
        
    # Reshape the flat list of scores back into a 2D map
    metric_map = np.array(scores_flat).reshape(num_blocks_y, num_blocks_x)
    
    return metric_map

def psnr(ref_block: np.ndarray, test_block: np.ndarray) -> float:
    """
    Calculates the PSNR for a single pair of image blocks.
    """
    ref_block_f = ref_block.astype(np.float64)
    test_block_f = test_block.astype(np.float64)
    
    mse = np.mean((ref_block_f - test_block_f) ** 2)
    
    if mse < 1e-10:
        return 100.0
        
    max_pixel_value = 255.0
    psnr_val = 20 * np.log10(max_pixel_value / np.sqrt(mse))
    
    return min(psnr_val, 100.0)

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
) -> float:
    """Calculate FVMD (Fréchet Video Mask Distance) with optional subsampling and early exit."""

    if not reference_frames or not decoded_frames:
        raise ValueError("Both reference_frames and decoded_frames must contain at least one frame.")

    total_frames = min(len(reference_frames), len(decoded_frames))
    indices = list(range(0, total_frames, max(1, stride)))

    if max_frames is not None and max_frames > 0:
        indices = indices[:max_frames]

    if not indices:
        raise ValueError("FVMD sampling produced no frames to evaluate.")

    # Ensure early_stop_window is sensible
    window = max(1, early_stop_window)

    def _compute_once(frame_indices: Sequence[int]) -> float:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            gt_root = tmp_path / "gt"
            gen_root = tmp_path / "gen"
            clip_name = "clip_0001"
            gt_clip = gt_root / clip_name
            gen_clip = gen_root / clip_name
            gt_clip.mkdir(parents=True, exist_ok=True)
            gen_clip.mkdir(parents=True, exist_ok=True)

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

            if log_root is None:
                logs_root_path = tmp_path / "fvmd_logs"
            else:
                logs_root_path = Path(log_root)

            logs_root_path.mkdir(parents=True, exist_ok=True)
            run_log_dir = logs_root_path / f"run_{uuid.uuid4().hex}"
            run_log_dir.mkdir(parents=True, exist_ok=True)

            score = fvmd(
                log_dir=str(run_log_dir),
                gen_path=str(gen_root),
                gt_path=str(gt_root),
            )

        return float(score)

    processed = 0
    last_score: Optional[float] = None

    while processed < len(indices):
        next_count = min(len(indices), processed + window)
        current_indices = indices[:next_count]
        current_score = _compute_once(current_indices)

        if last_score is not None:
            baseline = max(abs(last_score), 1e-6)
            delta = abs(current_score - last_score) / baseline
            if delta < early_stop_delta:
                return current_score

        last_score = current_score
        processed = next_count

    assert last_score is not None
    return last_score

def analyze_encoding_performance(
    reference_frames: List[np.ndarray],
    encoded_videos: Dict[str, str],
    block_size: int,
    width: int,
    height: int,
    temp_dir: str,
    masks_dir: str,
    sample_frames: List[int] = [0, 20, 40],
    video_bitrates: Dict[str, float] = {},
    reference_video_path: str = None,
    framerate: float = 30.0,
    strength_maps: Dict[str, np.ndarray] = None,
    generate_opencv_benchmarks: bool = True,
    metric_stride: int = 1,
    fvmd_stride: int = 1,
    fvmd_max_frames: Optional[int] = None,
    fvmd_processes: Optional[int] = None,
    fvmd_early_stop_delta: float = 0.002,
    fvmd_early_stop_window: int = 50,
    vmaf_stride: int = 1,
) -> Dict:
    """Analyze encoded videos with mask-aware metrics and sampling controls.

    Args:
        reference_frames: Source frames used for quality comparisons.
        encoded_videos: Mapping from method name to encoded video path.
        block_size: Block size used for spatial visualizations.
        width: Frame width in pixels.
        height: Frame height in pixels.
        temp_dir: Working directory for intermediate artifacts.
        masks_dir: Directory containing per-frame UFO masks.
        sample_frames: Frame indices for heatmap export.
        video_bitrates: Optional bitrate lookup (bps) per method name.
        reference_video_path: Unused placeholder retained for compatibility.
        framerate: Video framerate.
        strength_maps: Optional strength maps for OpenCV baselines.
        generate_opencv_benchmarks: Toggle for generating OpenCV reference clips.
        metric_stride: Sampling stride for PSNR/SSIM/LPIPS computation.
        fvmd_stride: Sampling stride for FVMD computation.
        fvmd_max_frames: Maximum sampled frames passed to FVMD after stride.
        fvmd_processes: Number of parallel FVMD workers (0 disables parallelism).
        fvmd_early_stop_delta: Relative delta threshold for FVMD early termination.
        fvmd_early_stop_window: Batch size of sampled frames added before FVMD convergence check.
        vmaf_stride: Sampling stride injected into the VMAF evaluation pipeline.

    Returns:
        Aggregated analysis results keyed by encoded video name.
    """

    metric_stride = max(1, metric_stride)
    fvmd_stride = max(1, fvmd_stride)
    vmaf_stride = max(1, vmaf_stride)

    os.makedirs(temp_dir, exist_ok=True)
    masked_videos_dir = os.path.join(temp_dir, "masked_videos")
    heatmaps_dir = os.path.join(temp_dir, "performance_figures")
    fvmd_log_root = os.path.join(temp_dir, "fvmd_logs")
    os.makedirs(masked_videos_dir, exist_ok=True)
    os.makedirs(heatmaps_dir, exist_ok=True)
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

    reference_vmaf_cache: Dict[Tuple[str, int], str] = {}
    analysis_results: Dict[str, Dict[str, Dict[str, float]]] = {}
    decoded_frames_cache: Dict[str, List[np.ndarray]] = {}

    fvmd_tasks: List[Tuple[str, str, "Future"]] = []
    fvmd_executor: Optional[ProcessPoolExecutor] = None
    if fvmd_processes != 0:
        cpu_count = multiprocessing.cpu_count() if hasattr(multiprocessing, "cpu_count") else 1
        max_workers = fvmd_processes if fvmd_processes is not None else max(1, min(4, cpu_count))
        if max_workers > 0:
            spawn_context = None
            if hasattr(multiprocessing, "get_context"):
                try:
                    spawn_context = multiprocessing.get_context("spawn")
                except ValueError:
                    spawn_context = None
            if spawn_context is not None:
                fvmd_executor = ProcessPoolExecutor(max_workers=max_workers, mp_context=spawn_context)
            else:
                fvmd_executor = ProcessPoolExecutor(max_workers=max_workers)
            print(f"  Using spawn-based FVMD worker pool with {max_workers} process(es)")

    # --- Generate OpenCV Restoration Benchmarks ---
    opencv_benchmarks: Dict[str, str] = {}
    if generate_opencv_benchmarks and strength_maps is not None:
        print("\n" + "=" * 80)
        print("GENERATING OPENCV RESTORATION BENCHMARKS")
        print("=" * 80)

        benchmarks_dir = os.path.join(temp_dir, "opencv_benchmarks")
        os.makedirs(benchmarks_dir, exist_ok=True)

        for method_name, maps in strength_maps.items():
            if maps is None:
                continue

            print(f"\nProcessing benchmarks for: {method_name}")

            if "downsample" in method_name.lower():
                print("  - Generating bilinear restoration benchmark...")
                benchmark_frames_bilinear = []
                for frame_idx, frame in enumerate(reference_frames):
                    downsampled_frame, _ = filter_frame_downsample(frame, maps[frame_idx] / np.max(maps[frame_idx]) if np.max(maps[frame_idx]) > 0 else maps[frame_idx], block_size)
                    restored_frame = restore_downsample_opencv_bilinear(downsampled_frame, maps[frame_idx], block_size)
                    benchmark_frames_bilinear.append(restored_frame)

                bilinear_frames_dir = os.path.join(benchmarks_dir, f"{method_name}_bilinear_frames")
                os.makedirs(bilinear_frames_dir, exist_ok=True)
                for i, frame in enumerate(benchmark_frames_bilinear):
                    cv2.imwrite(os.path.join(bilinear_frames_dir, f"{i+1:05d}.png"), frame)

                bilinear_video = os.path.join(benchmarks_dir, f"{method_name}_bilinear.mp4")
                encode_video(bilinear_frames_dir, bilinear_video, framerate, width, height, target_bitrate=video_bitrates.get(method_name, 1000000))
                opencv_benchmarks[f"{method_name} (OpenCV Bilinear)"] = bilinear_video
                video_bitrates[f"{method_name} (OpenCV Bilinear)"] = video_bitrates.get(method_name, 0)

                print("  - Generating bicubic restoration benchmark...")
                benchmark_frames_bicubic = []
                for frame_idx, frame in enumerate(reference_frames):
                    downsampled_frame, _ = filter_frame_downsample(frame, maps[frame_idx] / np.max(maps[frame_idx]) if np.max(maps[frame_idx]) > 0 else maps[frame_idx], block_size)
                    restored_frame = restore_downsample_opencv_bicubic(downsampled_frame, maps[frame_idx], block_size)
                    benchmark_frames_bicubic.append(restored_frame)

                bicubic_frames_dir = os.path.join(benchmarks_dir, f"{method_name}_bicubic_frames")
                os.makedirs(bicubic_frames_dir, exist_ok=True)
                for i, frame in enumerate(benchmark_frames_bicubic):
                    cv2.imwrite(os.path.join(bicubic_frames_dir, f"{i+1:05d}.png"), frame)

                bicubic_video = os.path.join(benchmarks_dir, f"{method_name}_bicubic.mp4")
                encode_video(bicubic_frames_dir, bicubic_video, framerate, width, height, target_bitrate=video_bitrates.get(method_name, 1000000))
                opencv_benchmarks[f"{method_name} (OpenCV Bicubic)"] = bicubic_video
                video_bitrates[f"{method_name} (OpenCV Bicubic)"] = video_bitrates.get(method_name, 0)

                print("  - Generating Lanczos restoration benchmark...")
                benchmark_frames_lanczos = []
                for frame_idx, frame in enumerate(reference_frames):
                    downsampled_frame, _ = filter_frame_downsample(frame, maps[frame_idx] / np.max(maps[frame_idx]) if np.max(maps[frame_idx]) > 0 else maps[frame_idx], block_size)
                    restored_frame = restore_downsample_opencv_lanczos(downsampled_frame, maps[frame_idx], block_size)
                    benchmark_frames_lanczos.append(restored_frame)

                lanczos_frames_dir = os.path.join(benchmarks_dir, f"{method_name}_lanczos_frames")
                os.makedirs(lanczos_frames_dir, exist_ok=True)
                for i, frame in enumerate(benchmark_frames_lanczos):
                    cv2.imwrite(os.path.join(lanczos_frames_dir, f"{i+1:05d}.png"), frame)

                lanczos_video = os.path.join(benchmarks_dir, f"{method_name}_lanczos.mp4")
                encode_video(lanczos_frames_dir, lanczos_video, framerate, width, height, target_bitrate=video_bitrates.get(method_name, 1000000))
                opencv_benchmarks[f"{method_name} (OpenCV Lanczos)"] = lanczos_video
                video_bitrates[f"{method_name} (OpenCV Lanczos)"] = video_bitrates.get(method_name, 0)

            elif "gaussian" in method_name.lower() or "blur" in method_name.lower():
                print("  - Generating unsharp mask restoration benchmark...")
                benchmark_frames_unsharp = []
                for frame_idx, frame in enumerate(reference_frames):
                    blurred_frame, _ = filter_frame_gaussian(frame, maps[frame_idx] / np.max(maps[frame_idx]) if np.max(maps[frame_idx]) > 0 else maps[frame_idx], block_size)
                    restored_frame = restore_blur_opencv_unsharp_mask(blurred_frame, maps[frame_idx], block_size)
                    benchmark_frames_unsharp.append(restored_frame)

                unsharp_frames_dir = os.path.join(benchmarks_dir, f"{method_name}_unsharp_frames")
                os.makedirs(unsharp_frames_dir, exist_ok=True)
                for i, frame in enumerate(benchmark_frames_unsharp):
                    cv2.imwrite(os.path.join(unsharp_frames_dir, f"{i+1:05d}.png"), frame)

                unsharp_video = os.path.join(benchmarks_dir, f"{method_name}_unsharp.mp4")
                encode_video(unsharp_frames_dir, unsharp_video, framerate, width, height, target_bitrate=video_bitrates.get(method_name, 1000000))
                opencv_benchmarks[f"{method_name} (OpenCV Unsharp Mask)"] = unsharp_video
                video_bitrates[f"{method_name} (OpenCV Unsharp Mask)"] = video_bitrates.get(method_name, 0)

                print("  - Generating Wiener approximation restoration benchmark...")
                benchmark_frames_wiener = []
                for frame_idx, frame in enumerate(reference_frames):
                    blurred_frame, _ = filter_frame_gaussian(frame, maps[frame_idx] / np.max(maps[frame_idx]) if np.max(maps[frame_idx]) > 0 else maps[frame_idx], block_size)
                    restored_frame = restore_blur_opencv_wiener_approximation(blurred_frame, maps[frame_idx], block_size)
                    benchmark_frames_wiener.append(restored_frame)

                wiener_frames_dir = os.path.join(benchmarks_dir, f"{method_name}_wiener_frames")
                os.makedirs(wiener_frames_dir, exist_ok=True)
                for i, frame in enumerate(benchmark_frames_wiener):
                    cv2.imwrite(os.path.join(wiener_frames_dir, f"{i+1:05d}.png"), frame)

                wiener_video = os.path.join(benchmarks_dir, f"{method_name}_wiener.mp4")
                encode_video(wiener_frames_dir, wiener_video, framerate, width, height, target_bitrate=video_bitrates.get(method_name, 1000000))
                opencv_benchmarks[f"{method_name} (OpenCV Wiener Approx)"] = wiener_video
                video_bitrates[f"{method_name} (OpenCV Wiener Approx)"] = video_bitrates.get(method_name, 0)

        encoded_videos.update(opencv_benchmarks)

        print(f"\nGenerated {len(opencv_benchmarks)} OpenCV restoration benchmarks.")
        print("=" * 80 + "\n")

    sample_frames_unique = sorted({idx for idx in sample_frames if 0 <= idx < total_reference_frames})

    for video_name, video_path in encoded_videos.items():
        print(f"\nProcessing '{video_name}'...")
        if not os.path.exists(video_path):
            print("  - Video not found, skipping.")
            continue

        decoded_frames = _decode_video_to_frames(video_path)
        decoded_frames_cache[video_name] = decoded_frames

        frame_count = min(total_reference_frames, len(decoded_frames))
        if frame_count == 0:
            print("  - No decoded frames available, skipping.")
            continue

        frame_indices = list(range(0, frame_count, metric_stride))
        if not frame_indices:
            frame_indices = [0]
        if frame_indices[-1] != frame_count - 1:
            frame_indices.append(frame_count - 1)
        frame_indices = sorted(set(frame_indices))

        slug = _slugify_name(video_name)

        analysis_results[video_name] = {
            'foreground': {'fvmd': 0.0},
            'background': {'fvmd': 0.0},
            'bitrate_mbps': video_bitrates.get(video_name, 0) / 1_000_000,
        }

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
        fg_ref_lpips_frames: List[np.ndarray] = []
        fg_dec_lpips_frames: List[np.ndarray] = []
        bg_psnr_vals: List[float] = []
        bg_ssim_vals: List[float] = []
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
            fg_ref_lpips_frames.append(masked_reference_fg_frames[idx][roi_slice])
            fg_dec_lpips_frames.append(masked_decoded_fg_frames[idx][roi_slice])

            bg_psnr_vals.append(_masked_psnr(ref_frame, dec_frame, bg_mask))
            bg_ssim_vals.append(_masked_ssim(ref_frame, dec_frame, bg_mask))
            bg_ref_lpips_frames.append(masked_reference_bg_frames[idx])
            bg_dec_lpips_frames.append(masked_decoded_bg_frames[idx])

        analysis_results[video_name]['foreground']['psnr_mean'] = float(np.mean(fg_psnr_vals)) if fg_psnr_vals else 0.0
        analysis_results[video_name]['foreground']['psnr_std'] = float(np.std(fg_psnr_vals)) if fg_psnr_vals else 0.0
        analysis_results[video_name]['foreground']['ssim_mean'] = float(np.mean(fg_ssim_vals)) if fg_ssim_vals else 0.0
        analysis_results[video_name]['foreground']['ssim_std'] = float(np.std(fg_ssim_vals)) if fg_ssim_vals else 0.0

        analysis_results[video_name]['background']['psnr_mean'] = float(np.mean(bg_psnr_vals)) if bg_psnr_vals else 0.0
        analysis_results[video_name]['background']['psnr_std'] = float(np.std(bg_psnr_vals)) if bg_psnr_vals else 0.0
        analysis_results[video_name]['background']['ssim_mean'] = float(np.mean(bg_ssim_vals)) if bg_ssim_vals else 0.0
        analysis_results[video_name]['background']['ssim_std'] = float(np.std(bg_ssim_vals)) if bg_ssim_vals else 0.0

        fg_lpips_scores = calculate_lpips_per_frame(fg_ref_lpips_frames, fg_dec_lpips_frames)
        bg_lpips_scores = calculate_lpips_per_frame(bg_ref_lpips_frames, bg_dec_lpips_frames)

        analysis_results[video_name]['foreground']['lpips_mean'] = float(np.mean(fg_lpips_scores)) if fg_lpips_scores else 0.0
        analysis_results[video_name]['foreground']['lpips_std'] = float(np.std(fg_lpips_scores)) if fg_lpips_scores else 0.0
        analysis_results[video_name]['background']['lpips_mean'] = float(np.mean(bg_lpips_scores)) if bg_lpips_scores else 0.0
        analysis_results[video_name]['background']['lpips_std'] = float(np.std(bg_lpips_scores)) if bg_lpips_scores else 0.0

        ref_fg_key = ("fg", frame_count)
        if ref_fg_key in reference_vmaf_cache:
            ref_fg_video_path = reference_vmaf_cache[ref_fg_key]
        else:
            ref_fg_video_path = os.path.join(masked_videos_dir, f"reference_fg_{frame_count:05d}.mp4")
            _encode_frames_to_video(
                masked_reference_fg_frames[:frame_count],
                ref_fg_video_path,
                framerate,
                filter_chain=crop_filter,
                extra_codec_args=['-g', '1'],
            )
            reference_vmaf_cache[ref_fg_key] = ref_fg_video_path

        ref_bg_key = ("bg", frame_count)
        if ref_bg_key in reference_vmaf_cache:
            ref_bg_video_path = reference_vmaf_cache[ref_bg_key]
        else:
            ref_bg_video_path = os.path.join(masked_videos_dir, f"reference_bg_{frame_count:05d}.mp4")
            _encode_frames_to_video(
                masked_reference_bg_frames[:frame_count],
                ref_bg_video_path,
                framerate,
                extra_codec_args=['-g', '1'],
            )
            reference_vmaf_cache[ref_bg_key] = ref_bg_video_path

        enc_fg_video_path = os.path.join(masked_videos_dir, f"{slug}_fg.mp4")
        _encode_frames_to_video(
            masked_decoded_fg_frames,
            enc_fg_video_path,
            framerate,
            filter_chain=crop_filter,
            extra_codec_args=['-g', '1'],
        )

        enc_bg_video_path = os.path.join(masked_videos_dir, f"{slug}_bg.mp4")
        _encode_frames_to_video(
            masked_decoded_bg_frames,
            enc_bg_video_path,
            framerate,
            extra_codec_args=['-g', '1'],
        )

        vmaf_fg = calculate_vmaf(ref_fg_video_path, enc_fg_video_path, crop_width, crop_height, framerate, frame_stride=vmaf_stride)
        vmaf_bg = calculate_vmaf(ref_bg_video_path, enc_bg_video_path, width, height, framerate, frame_stride=vmaf_stride)

        analysis_results[video_name]['foreground']['vmaf_mean'] = float(vmaf_fg.get('mean', 0))
        analysis_results[video_name]['foreground']['vmaf_std'] = float(vmaf_fg.get('std', 0))
        analysis_results[video_name]['background']['vmaf_mean'] = float(vmaf_bg.get('mean', 0))
        analysis_results[video_name]['background']['vmaf_std'] = float(vmaf_bg.get('std', 0))

        fvmd_indices = list(range(0, frame_count, fvmd_stride))
        if not fvmd_indices:
            fvmd_indices = [0]
        if fvmd_max_frames is not None and fvmd_max_frames > 0:
            fvmd_indices = fvmd_indices[:fvmd_max_frames]

        ref_fg_fvmd_frames = [masked_reference_fg_frames[i] for i in fvmd_indices]
        dec_fg_fvmd_frames = [masked_decoded_fg_frames[i] for i in fvmd_indices]
        ref_bg_fvmd_frames = [masked_reference_bg_frames[i] for i in fvmd_indices]
        dec_bg_fvmd_frames = [masked_decoded_bg_frames[i] for i in fvmd_indices]

        if fvmd_executor is not None:
            fvmd_tasks.append((video_name, 'foreground', fvmd_executor.submit(
                calculate_fvmd,
                ref_fg_fvmd_frames,
                dec_fg_fvmd_frames,
                log_root=fvmd_log_root,
                stride=1,
                max_frames=None,
                early_stop_delta=fvmd_early_stop_delta,
                early_stop_window=fvmd_early_stop_window,
            )))
            fvmd_tasks.append((video_name, 'background', fvmd_executor.submit(
                calculate_fvmd,
                ref_bg_fvmd_frames,
                dec_bg_fvmd_frames,
                log_root=fvmd_log_root,
                stride=1,
                max_frames=None,
                early_stop_delta=fvmd_early_stop_delta,
                early_stop_window=fvmd_early_stop_window,
            )))
        else:
            analysis_results[video_name]['foreground']['fvmd'] = calculate_fvmd(
                ref_fg_fvmd_frames,
                dec_fg_fvmd_frames,
                log_root=fvmd_log_root,
                stride=1,
                max_frames=None,
                early_stop_delta=fvmd_early_stop_delta,
                early_stop_window=fvmd_early_stop_window,
            )
            analysis_results[video_name]['background']['fvmd'] = calculate_fvmd(
                ref_bg_fvmd_frames,
                dec_bg_fvmd_frames,
                log_root=fvmd_log_root,
                stride=1,
                max_frames=None,
                early_stop_delta=fvmd_early_stop_delta,
                early_stop_window=fvmd_early_stop_window,
            )

        for frame_idx in sample_frames_unique:
            if frame_idx >= frame_count:
                continue
            ref_frame = reference_frames[frame_idx]
            dec_frame = decoded_frames[frame_idx]
            try:
                ref_blocks = split_image_into_blocks(ref_frame, block_size)
                dec_blocks = split_image_into_blocks(dec_frame, block_size)
            except ValueError:
                continue

            psnr_map = calculate_blockwise_metric(ref_blocks, dec_blocks, psnr)
            ssim_map = calculate_blockwise_metric(ref_blocks, dec_blocks, _block_ssim)
            mask_img = fg_masks[frame_idx].astype(np.uint8) * 255
            heatmap_path = os.path.join(
                heatmaps_dir,
                f"{slug}_frame_{frame_idx + 1:05d}.png",
            )
            _generate_and_save_heatmap(
                ref_frame,
                dec_frame,
                mask_img,
                {'PSNR': psnr_map, 'SSIM': ssim_map},
                video_name,
                frame_idx,
                heatmap_path,
                block_size,
            )

    if fvmd_executor is not None:
        for video_name, region, future in fvmd_tasks:
            try:
                score = future.result()
            except Exception as exc:
                print(f"  Warning: FVMD failed for '{video_name}' ({region}): {exc}")
                score = 0.0
            analysis_results.setdefault(video_name, {}).setdefault(region, {})['fvmd'] = score
        fvmd_executor.shutdown(wait=True)

    if analysis_results:
        _generate_quality_visualizations(analysis_results, heatmaps_dir)
        _generate_patch_comparison(
            reference_frames=reference_frames,
            decoded_frames_cache=decoded_frames_cache,
            results=analysis_results,
            fg_masks=fg_masks,
            heatmaps_dir=heatmaps_dir,
            sample_frame_idx=len(reference_frames) // 2,
            patch_size=128,
        )
        _print_summary_report(analysis_results)
    else:
        print("No results to display.")

    print(f"\nAnalysis complete. Masked videos saved to: {masked_videos_dir}")
    print(f"Quality visualizations saved to: {heatmaps_dir}")

    return analysis_results


def _generate_and_save_heatmap(
    ref_frame: np.ndarray,
    decoded_frame: np.ndarray,
    mask: np.ndarray,
    metric_maps: Dict[str, np.ndarray],
    video_name: str,
    frame_idx: int,
    output_path: str,
    block_size: int
) -> None:
    """Generates and saves a 2x3 visualization grid for a single frame."""
    try:
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'{video_name} - Frame {frame_idx+1}', fontsize=16)

        height, width, _ = ref_frame.shape

        # Row 1: Images
        axes[0, 0].imshow(cv2.cvtColor(ref_frame, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title('Reference Frame')
        axes[0, 0].axis('off')

        axes[0, 1].imshow(cv2.cvtColor(decoded_frame, cv2.COLOR_BGR2RGB))
        axes[0, 1].set_title('Encoded Frame')
        axes[0, 1].axis('off')

        mask_colored = np.zeros_like(ref_frame)
        mask_colored[mask > 0] = [255, 0, 0]
        overlay = cv2.addWeighted(ref_frame, 0.7, mask_colored, 0.3, 0)
        axes[0, 2].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        axes[0, 2].set_title('UFO Mask (Red=FG)')
        axes[0, 2].axis('off')

        # Row 2: Heatmaps
        ssim_map = metric_maps.get('SSIM', np.zeros(list(metric_maps.values())[0].shape))
        psnr_map = metric_maps.get('PSNR', np.zeros(list(metric_maps.values())[0].shape))

        im1 = axes[1, 0].imshow(ssim_map, cmap='viridis', vmin=0, vmax=1)
        axes[1, 0].set_title(f'SSIM (Mean: {np.mean(ssim_map):.3f})')
        plt.colorbar(im1, ax=axes[1, 0])

        psnr_capped = np.clip(psnr_map, 0, 50)
        im2 = axes[1, 1].imshow(psnr_capped, cmap='viridis', vmin=0, vmax=50)
        finite_mask = np.isfinite(psnr_map)
        psnr_mean = np.mean(psnr_map[finite_mask]) if np.any(finite_mask) else 0.0
        axes[1, 1].set_title(f'PSNR (Mean: {psnr_mean:.1f} dB)')
        plt.colorbar(im2, ax=axes[1, 1])

        # Quality overlay (vectorised block colouring)
        num_blocks_y, num_blocks_x = ssim_map.shape
        block_overlay = np.zeros((num_blocks_y, num_blocks_x, 3), dtype=np.uint8)
        severe_mask = ssim_map < 0.5
        moderate_mask = (ssim_map >= 0.5) & (ssim_map < 0.7)
        block_overlay[moderate_mask] = [255, 255, 0]
        block_overlay[severe_mask] = [255, 0, 0]

        overlay_pixels = np.kron(block_overlay, np.ones((block_size, block_size, 1), dtype=np.uint8))
        overlay_pixels = overlay_pixels[:height, :width]
        blended = cv2.addWeighted(ref_frame, 0.65, overlay_pixels, 0.35, 0)
        axes[1, 2].imshow(cv2.cvtColor(blended, cv2.COLOR_BGR2RGB))
        axes[1, 2].set_title('Low Quality Overlay')
        axes[1, 2].axis('off')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

    except Exception as e:
        print(f"Error generating heatmap for {video_name} frame {frame_idx+1}: {e}")


def _print_summary_report(results: Dict) -> None:
    """Prints a unified summary report with all metrics in one table."""
    print(f"\n{'='*180}")
    print(f"{'COMPREHENSIVE ANALYSIS SUMMARY':^180}")
    print(f"{'='*180}")

    if not results:
        print("No results to display.")
        return

    # Unified metrics table
    print(f"\n{'QUALITY METRICS (Foreground / Background)':^180}")
    print(f"{'Method':<20} {'PSNR (dB)':<25} {'SSIM':<25} {'LPIPS':<25} {'FVMD':<25} {'VMAF':<25} {'Bitrate (Mbps)':<15}")
    print(f"{'-'*180}")

    for video_name, data in results.items():
        fg_data = data['foreground']
        bg_data = data['background']

        # Format metric strings (FG / BG)
        psnr_str = f"{fg_data.get('psnr_mean', 0):.2f} / {bg_data.get('psnr_mean', 0):.2f}"
        ssim_str = f"{fg_data.get('ssim_mean', 0):.4f} / {bg_data.get('ssim_mean', 0):.4f}"
        lpips_str = f"{fg_data.get('lpips_mean', 0):.4f} / {bg_data.get('lpips_mean', 0):.4f}"
        fvmd_str = f"{fg_data.get('fvmd', 0):.2f} / {bg_data.get('fvmd', 0):.2f}"
        vmaf_str = f"{fg_data.get('vmaf_mean', 0):.2f} / {bg_data.get('vmaf_mean', 0):.2f}"
        bitrate_str = f"{data['bitrate_mbps']:.2f}"

        print(f"{video_name:<20} {psnr_str:<25} {ssim_str:<25} {lpips_str:<25} {fvmd_str:<25} {vmaf_str:<25} {bitrate_str:<15}")

    print(f"{'-'*180}")

    # Trade-off analysis against the first video as baseline
    if len(results) > 1:
        baseline_name = list(results.keys())[0]
        print(f"\n{'TRADE-OFF ANALYSIS (vs. ' + baseline_name + ')':^180}")
        print(f"{'Method':<20} {'PSNR FG %':<15} {'PSNR BG %':<15} {'SSIM FG %':<15} {'SSIM BG %':<15} {'LPIPS FG %':<15} {'LPIPS BG %':<15} {'FVMD FG %':<15} {'FVMD BG %':<15} {'VMAF FG %':<15} {'VMAF BG %':<15}")
        print(f"{'-'*180}")

        for video_name in list(results.keys())[1:]:
            # Calculate changes for all metrics
            psnr_fg_change = 0.0
            psnr_bg_change = 0.0
            ssim_fg_change = 0.0
            ssim_bg_change = 0.0
            lpips_fg_change = 0.0
            lpips_bg_change = 0.0
            fvmd_fg_change = 0.0
            fvmd_bg_change = 0.0
            vmaf_fg_change = 0.0
            vmaf_bg_change = 0.0

            for metric in ['psnr', 'ssim', 'lpips', 'vmaf']:
                for region in ['foreground', 'background']:
                    baseline_val = results[baseline_name][region].get(f'{metric}_mean', 0)
                    current_val = results[video_name][region].get(f'{metric}_mean', 0)

                    if baseline_val > 0:
                        # For LPIPS, lower is better, so invert the change
                        if metric == 'lpips':
                            change = ((baseline_val / current_val) - 1) * 100 if current_val > 0 else 0.0
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
                baseline_fvmd = results[baseline_name][region].get('fvmd', 0)
                current_fvmd = results[video_name][region].get('fvmd', 0)

                if baseline_fvmd > 0 and current_fvmd > 0:
                    change = ((baseline_fvmd / current_fvmd) - 1) * 100
                    if region == 'foreground':
                        fvmd_fg_change = change
                    else:
                        fvmd_bg_change = change

            print(
                f"{video_name:<20} {psnr_fg_change:+.2f}%{' '*8} {psnr_bg_change:+.2f}%{' '*8} "
                f"{ssim_fg_change:+.2f}%{' '*8} {ssim_bg_change:+.2f}%{' '*8} "
                f"{lpips_fg_change:+.2f}%{' '*8} {lpips_bg_change:+.2f}%{' '*8} "
                f"{fvmd_fg_change:+.2f}%{' '*8} {fvmd_bg_change:+.2f}%{' '*8} "
                f"{vmaf_fg_change:+.2f}%{' '*8} {vmaf_bg_change:+.2f}%{' '*8}"
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

    experiment_dir = os.path.abspath(config.experiment_dir)
    os.makedirs(experiment_dir, exist_ok=True)

    execution_times: Dict[str, float] = {}

    if config.force_framerate is not None:
        framerate = float(config.force_framerate)
    else:
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
                "working_dir": config.removability_working_dir or experiment_dir,
            },
            "apply_selective_removal": {
                "shrink_amount": shrink_amount,
            },
            "decode_video": {
                "quality": config.decode_quality,
                "start_number": config.decode_start_number,
            },
            "inpaint_with_propainter": {
                "propainter_dir": config.propainter_dir,
                "resize_ratio": config.propainter_resize_ratio,
                "ref_stride": config.propainter_ref_stride,
                "neighbor_length": config.propainter_neighbor_length,
                "subvideo_length": config.propainter_subvideo_length,
                "mask_dilation": config.propainter_mask_dilation,
                "raft_iter": config.propainter_raft_iter,
                "fp16": config.propainter_fp16,
                "devices": _list_or_none(config.propainter_devices),
                "parallel_chunk_length": config.propainter_parallel_chunk_length,
                "chunk_overlap": config.propainter_chunk_overlap,
            },
            "inpaint_with_e2fgvi": {
                "e2fgvi_dir": config.e2fgvi_dir,
                "model": config.e2fgvi_model,
                "ckpt": config.e2fgvi_ckpt,
                "ref_stride": config.e2fgvi_ref_stride,
                "neighbor_stride": config.e2fgvi_neighbor_stride,
                "num_ref": config.e2fgvi_num_ref,
                "mask_dilation": config.e2fgvi_mask_dilation,
            },
            "restore_downsampled_with_realesrgan": {
                "realesrgan_dir": config.realesrgan_dir,
                "model_name": config.realesrgan_model_name,
                "denoise_strength": config.realesrgan_denoise_strength,
                "tile": config.realesrgan_tile,
                "tile_pad": config.realesrgan_tile_pad,
                "pre_pad": config.realesrgan_pre_pad,
                "fp32": config.realesrgan_fp32,
                "devices": _list_or_none(config.realesrgan_devices),
                "parallel_chunk_length": config.realesrgan_parallel_chunk_length,
                "per_device_workers": config.realesrgan_per_device_workers,
                "model_path": config.realesrgan_model_path,
            },
            "restore_with_instantir_adaptive": {
                "instantir_weights_dir": config.instantir_weights_dir,
                "cfg": config.instantir_cfg,
                "creative_start": config.instantir_creative_start,
                "preview_start": config.instantir_preview_start,
                "seed": config.instantir_seed,
                "devices": _list_or_none(config.instantir_devices),
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
            },
        },
    }

    encode_function_params = {
        "baseline": {
            "preset": config.baseline_encode_preset,
            "pix_fmt": config.baseline_encode_pix_fmt,
            "target_bitrate": target_bitrate,
        },
        "adaptive": {
            "preset": config.adaptive_encode_preset,
            "pix_fmt": config.adaptive_encode_pix_fmt,
            "target_bitrate": target_bitrate,
        },
        "elvis_v1_shrunk": {
            "preset": config.shrunk_encode_preset,
            "pix_fmt": config.shrunk_encode_pix_fmt,
            "target_bitrate": target_bitrate,
        },
        "downsample": {
            "preset": config.downsample_encode_preset,
            "pix_fmt": config.downsample_encode_pix_fmt,
            "target_bitrate": target_bitrate,
        },
        "gaussian": {
            "preset": config.gaussian_encode_preset,
            "pix_fmt": config.gaussian_encode_pix_fmt,
            "target_bitrate": target_bitrate,
        },
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
    execution_times["preprocessing"] = end - start
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
        working_dir=config.removability_working_dir or experiment_dir,
        smoothing_beta=config.removability_smoothing_beta,
    )
    end = time.time()
    execution_times["removability_score_calculation"] = end - start
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
        preset=config.baseline_encode_preset,
        pix_fmt=config.baseline_encode_pix_fmt,
    )
    end = time.time()
    execution_times["baseline_encoding"] = end - start
    print(f"Baseline encoding completed in {end - start:.2f} seconds.\n")

    # ELVIS v1 shrinking
    start = time.time()
    print("Shrinking and encoding frames with ELVIS v1...")
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
        preset=config.shrunk_encode_preset,
        pix_fmt=config.shrunk_encode_pix_fmt,
    )

    removal_masks_np = np.array(removal_masks, dtype=np.uint8)
    masks_packed = np.packbits(removal_masks_np)
    np.savez(
        os.path.join(experiment_dir, f"shrink_masks_{block_size}.npz"),
        packed=masks_packed,
        shape=removal_masks_np.shape,
    )
    end = time.time()
    execution_times["elvis_v1_shrinking"] = end - start
    print(f"ELVIS v1 shrinking completed in {end - start:.2f} seconds.\n")

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
        "save_qp_maps": config.roi_save_qp_maps,
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
        save_qp_maps=config.roi_save_qp_maps,
        qp_maps_dir=qp_maps_dir,
    )
    end = time.time()
    execution_times["adaptive_encoding"] = end - start
    print(f"Adaptive encoding completed in {end - start:.2f} seconds.\n")

    # Downsampling-based ELVIS v2
    start = time.time()
    print("Applying downsampling-based ELVIS v2 adaptive filtering and encoding...")
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
        preset=config.downsample_encode_preset,
        pix_fmt=config.downsample_encode_pix_fmt,
    )

    downsample_maps_video = os.path.join(maps_dir, "downsample_encoded.mp4")
    encode_strength_maps(
        strength_maps=list(downsample_maps),
        output_video=downsample_maps_video,
        framerate=framerate,
        target_bitrate=config.downsample_strength_target_bitrate,
    )
    end = time.time()
    execution_times["elvis_v2_downsampling"] = end - start
    print(f"Downsampling-based ELVIS v2 filtering and encoding completed in {end - start:.2f} seconds.\n")

    # Gaussian blur-based ELVIS v2
    start = time.time()
    print("Applying Gaussian blur Filtering-based ELVIS v2 adaptive filtering and encoding...")
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
        preset=config.gaussian_encode_preset,
        pix_fmt=config.gaussian_encode_pix_fmt,
    )

    gaussian_maps_video = os.path.join(maps_dir, "gaussian_encoded.mp4")
    encode_strength_maps(
        strength_maps=list(gaussian_maps),
        output_video=gaussian_maps_video,
        framerate=framerate,
        target_bitrate=config.gaussian_strength_target_bitrate,
    )
    end = time.time()
    execution_times["elvis_v2_gaussian"] = end - start
    print(f"Gaussian blur filtering-based ELVIS v2 filtering and encoding completed in {end - start:.2f} seconds.\n")

    # Client-side stretching
    start = time.time()
    print("Decoding and stretching ELVIS v1 video...")
    removal_masks_file = np.load(os.path.join(experiment_dir, f"shrink_masks_{block_size}.npz"))
    removal_masks_loaded = np.unpackbits(removal_masks_file['packed'])
    removal_masks = removal_masks_loaded[:np.prod(removal_masks_file['shape'])].reshape(removal_masks_file['shape'])

    stretched_frames_dir = os.path.join(experiment_dir, "frames", "stretched")
    if not decode_video(
        shrunk_video,
        stretched_frames_dir,
        framerate=framerate,
        start_number=config.decode_start_number,
        quality=config.decode_quality,
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

    removal_masks_dir = os.path.join(maps_dir, "removal_masks")
    os.makedirs(removal_masks_dir, exist_ok=True)
    for i, mask in enumerate(removal_masks):
        mask_img = (mask * 255).astype(np.uint8)
        mask_img = cv2.resize(mask_img, (width, height), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(os.path.join(removal_masks_dir, f"{i+1:05d}.png"), mask_img)
    end = time.time()
    execution_times["elvis_v1_stretching"] = end - start
    print(f"ELVIS v1 stretching completed in {end - start:.2f} seconds.\n")

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
        mask_img = cv2.imread(os.path.join(removal_masks_dir, f"{i+1:05d}.png"), cv2.IMREAD_GRAYSCALE)
        inpainted_frame = cv2.inpaint(stretched_frame, mask_img, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
        cv2.imwrite(os.path.join(inpainted_cv2_frames_dir, f"{i+1:05d}.png"), inpainted_frame)
    end = time.time()
    execution_times["elvis_v1_inpainting_cv2"] = end - start
    print(f"ELVIS v1 CV2 inpainting completed in {end - start:.2f} seconds.\n")

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
        removal_masks_dir=removal_masks_dir,
        output_frames_dir=inpainted_frames_dir,
        width=width,
        height=height,
        framerate=framerate,
        propainter_dir=config.propainter_dir,
        resize_ratio=config.propainter_resize_ratio,
        ref_stride=config.propainter_ref_stride,
        neighbor_length=config.propainter_neighbor_length,
        subvideo_length=config.propainter_subvideo_length,
        mask_dilation=config.propainter_mask_dilation,
        raft_iter=config.propainter_raft_iter,
        fp16=config.propainter_fp16,
        devices=_list_or_none(config.propainter_devices),
        parallel_chunk_length=config.propainter_parallel_chunk_length,
        chunk_overlap=config.propainter_chunk_overlap,
    )
    end = time.time()
    execution_times["elvis_v1_inpainting_propainter"] = end - start
    print(f"ELVIS v1 ProPainter inpainting completed in {end - start:.2f} seconds.\n")

    inpainted_video = os.path.join(experiment_dir, "inpainted.mp4")
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
        removal_masks_dir=removal_masks_dir,
        output_frames_dir=inpainted_e2fgvi_frames_dir,
        width=width,
        height=height,
        framerate=framerate,
        e2fgvi_dir=config.e2fgvi_dir,
        model=config.e2fgvi_model,
        ckpt=config.e2fgvi_ckpt,
        ref_stride=config.e2fgvi_ref_stride,
        neighbor_stride=config.e2fgvi_neighbor_stride,
        num_ref=config.e2fgvi_num_ref,
        mask_dilation=config.e2fgvi_mask_dilation,
    )
    end = time.time()
    execution_times["elvis_v1_inpainting_e2fgvi"] = end - start
    print(f"ELVIS v1 E2FGVI inpainting completed in {end - start:.2f} seconds.\n")

    inpainted_e2fgvi_video = os.path.join(experiment_dir, "inpainted_e2fgvi.mp4")
    encode_video(
        input_frames_dir=inpainted_e2fgvi_frames_dir,
        output_video=inpainted_e2fgvi_video,
        framerate=framerate,
        width=width,
        height=height,
        target_bitrate=None,
    )

    # Downsampling restoration
    start = time.time()
    print("Decoding downsampled ELVIS v2 video and strength maps...")
    downsampled_frames_decoded_dir = os.path.join(experiment_dir, "frames", "downsampled_decoded")
    if not decode_video(
        downsampled_video,
        downsampled_frames_decoded_dir,
        framerate=framerate,
        start_number=config.decode_start_number,
        quality=config.decode_quality,
    ):
        raise RuntimeError(f"Failed to decode downsampled video: {downsampled_video}")

    downsampled_maps_decoded_dir = os.path.join(experiment_dir, "maps", "downsampled_maps_decoded")
    strength_maps = decode_strength_maps(downsample_maps_video, block_size, downsampled_maps_decoded_dir)
    end = time.time()
    execution_times["elvis_v2_downsampling_decoding"] = end - start
    print(f"Decoding completed in {end - start:.2f} seconds.\n")

    start = time.time()
    print("Applying adaptive upsampling restoration for downsampling-based ELVIS v2...")
    downsampled_restored_frames_dir = os.path.join(experiment_dir, "frames", "downsampled_restored")
    os.makedirs(downsampled_restored_frames_dir, exist_ok=True)
    restore_downsampled_with_realesrgan(
        input_frames_dir=downsampled_frames_decoded_dir,
        output_frames_dir=downsampled_restored_frames_dir,
        downscale_maps=strength_maps,
        block_size=block_size,
        model_name=config.realesrgan_model_name,
        denoise_strength=config.realesrgan_denoise_strength,
        tile=config.realesrgan_tile,
        tile_pad=config.realesrgan_tile_pad,
        pre_pad=config.realesrgan_pre_pad,
        fp32=config.realesrgan_fp32,
        devices=_list_or_none(config.realesrgan_devices),
        parallel_chunk_length=config.realesrgan_parallel_chunk_length,
        per_device_workers=config.realesrgan_per_device_workers,
        model_path=config.realesrgan_model_path,
    )
    end = time.time()
    execution_times["elvis_v2_downsampling_restoration"] = end - start
    print(f"Adaptive upsampling restoration completed in {end - start:.2f} seconds.\n")

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
    print("Decoding Gaussian blur filtered ELVIS v2 video and strength maps...")
    gaussian_frames_decoded_dir = os.path.join(experiment_dir, "frames", "gaussian_decoded")
    if not decode_video(
        gaussian_video,
        gaussian_frames_decoded_dir,
        framerate=framerate,
        start_number=config.decode_start_number,
        quality=config.decode_quality,
    ):
        raise RuntimeError(f"Failed to decode Gaussian video: {gaussian_video}")

    gaussian_maps_decoded_dir = os.path.join(experiment_dir, "maps", "gaussian_maps_decoded")
    strength_maps_gaussian = decode_strength_maps(gaussian_maps_video, block_size, gaussian_maps_decoded_dir)
    end = time.time()
    execution_times["elvis_v2_gaussian_decoding"] = end - start
    print(f"Decoding completed in {end - start:.2f} seconds.\n")

    start = time.time()
    print("Applying adaptive deblurring restoration for Gaussian blur filtering-based ELVIS v2...")
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

    instantir_weights_dir = (
        config.instantir_weights_dir
        if config.instantir_weights_dir is not None
        else str(script_dir / "InstantIR" / "models")
    )

    restore_with_instantir_adaptive(
        input_frames_dir=gaussian_instantir_input_dir,
        blur_maps=strength_maps_gaussian,
        block_size=block_size,
        instantir_weights_dir=instantir_weights_dir,
        cfg=config.instantir_cfg,
        creative_start=config.instantir_creative_start,
        preview_start=config.instantir_preview_start,
        seed=config.instantir_seed,
        devices=_list_or_none(config.instantir_devices),
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
    execution_times["elvis_v2_gaussian_restoration"] = end - start
    print(f"Adaptive deblurring restoration completed in {end - start:.2f} seconds.\n")

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

    video_sizes = {
        "Baseline": os.path.getsize(baseline_video),
        "ELVIS v1": os.path.getsize(shrunk_video) + os.path.getsize(os.path.join(experiment_dir, f"shrink_masks_{block_size}.npz")),
        "Adaptive": os.path.getsize(adaptive_video),
        "ELVIS v2 Downsample": os.path.getsize(downsampled_video) + os.path.getsize(downsample_maps_video),
        "ELVIS v2 Gaussian": os.path.getsize(gaussian_video) + os.path.getsize(gaussian_maps_video),
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
        "Baseline": baseline_video,
        "Adaptive": adaptive_video,
        "ELVIS v1 CV2": inpainted_cv2_video,
        "ELVIS v1 ProPainter": inpainted_video,
        "ELVIS v1 E2FGVI": inpainted_e2fgvi_video,
        "ELVIS v2 Downsample": downsampled_restored_video,
        "ELVIS v2 Gaussian": gaussian_restored_video,
    }

    ufo_masks_dir = os.path.join(experiment_dir, "maps", "ufo_masks")
    if config.analysis_sample_frames is not None:
        sample_frames = sorted({int(idx) for idx in config.analysis_sample_frames if idx >= 0})
    else:
        sample_frames = [frame_count // 2]
    sample_frames = [idx for idx in sample_frames if idx < frame_count]
    pipeline_params["derived"]["analysis_sample_frames"] = sample_frames

    strength_maps_dict = {
        "ELVIS v2 Downsample": strength_maps,
        "ELVIS v2 Gaussian": strength_maps_gaussian,
    }

    analysis_results = analyze_encoding_performance(
        reference_frames=reference_frames,
        encoded_videos=encoded_videos,
        block_size=block_size,
        width=width,
        height=height,
        temp_dir=experiment_dir,
        masks_dir=ufo_masks_dir,
        sample_frames=sample_frames,
        video_bitrates=bitrates,
        reference_video_path=reference_video,
        framerate=framerate,
        strength_maps=strength_maps_dict,
        generate_opencv_benchmarks=config.generate_opencv_benchmarks,
        metric_stride=config.metric_stride,
        fvmd_stride=config.fvmd_stride,
        fvmd_max_frames=config.fvmd_max_frames,
        fvmd_processes=config.fvmd_processes,
        fvmd_early_stop_delta=config.fvmd_early_stop_delta,
        fvmd_early_stop_window=config.fvmd_early_stop_window,
        vmaf_stride=config.vmaf_stride,
    )

    end = time.time()
    execution_times["performance_evaluation"] = end - start

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


def _parse_analysis_frames_arg(value: Optional[str]) -> Optional[List[int]]:
    if value is None:
        return None
    frames: List[int] = []
    for part in value.split(','):
        part = part.strip()
        if not part:
            continue
        try:
            frames.append(int(part))
        except ValueError:
            raise ValueError(f"Invalid frame index '{part}' in --analysis-sample-frames") from None
    return frames


def _load_config_from_cli() -> ElvisConfig:
    parser = argparse.ArgumentParser(description="Run the ELVIS pipeline with configurable parameters.")
    parser.add_argument("--config", type=str, help="Path to a JSON file containing ElvisConfig fields.")
    parser.add_argument("--reference-video", type=str, help="Path to the input reference video.")
    parser.add_argument("--experiment-dir", type=str, help="Directory to store experiment artefacts.")
    parser.add_argument("--width", type=int, help="Target frame width.")
    parser.add_argument("--height", type=int, help="Target frame height.")
    parser.add_argument("--block-size", type=int, help="Processing block size.")
    parser.add_argument("--shrink-amount", type=float, help="Shrink amount for ELVIS v1.")
    parser.add_argument("--quality-factor", type=float, help="Quality factor for target bitrate calculation.")
    parser.add_argument("--target-bitrate", type=int, help="Override target bitrate in bits per second")
    parser.add_argument("--framerate", type=float, help="Override source framerate.")
    parser.add_argument("--removability-alpha", type=float, help="Alpha parameter for removability scoring.")
    parser.add_argument("--removability-smoothing-beta", type=float, help="Smoothing beta for removability scoring.")
    parser.add_argument("--analysis-sample-frames", type=str, help="Comma-separated frame indices for analysis.")
    parser.add_argument("--roi-save-qp-maps", dest="roi_save_qp_maps", action="store_true", help="Enable saving QP maps.")
    parser.add_argument("--no-roi-save-qp-maps", dest="roi_save_qp_maps", action="store_false", help="Disable saving QP maps.")
    parser.set_defaults(roi_save_qp_maps=None)
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
        "experiment_dir": args.experiment_dir,
        "width": args.width,
        "height": args.height,
        "block_size": args.block_size,
        "shrink_amount": args.shrink_amount,
        "quality_factor": args.quality_factor,
        "target_bitrate_override": args.target_bitrate,
        "force_framerate": args.framerate,
        "removability_alpha": args.removability_alpha,
        "removability_smoothing_beta": args.removability_smoothing_beta,
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

    if args.roi_save_qp_maps is not None:
        config_data["roi_save_qp_maps"] = args.roi_save_qp_maps

    if args.generate_opencv_benchmarks is not None:
        config_data["generate_opencv_benchmarks"] = args.generate_opencv_benchmarks

    analysis_frames = _parse_analysis_frames_arg(args.analysis_sample_frames)
    if analysis_frames is not None:
        config_data["analysis_sample_frames"] = analysis_frames

    return ElvisConfig(**config_data)


def main() -> None:
    config = _load_config_from_cli()
    results = run_elvis(config)
    path = results.get("analysis_results_path")
    if path:
        print(f"\nFinal analysis JSON: {path}")


if __name__ == "__main__":
    main()