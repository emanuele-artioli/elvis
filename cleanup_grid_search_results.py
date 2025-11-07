#!/usr/bin/env python3
"""Utility for pruning grid-search artifacts and renaming result folders.

The script expects a grid-search output directory such as
```
/home/user/elvis/grid_search_results
```
containing one subfolder per experiment. Inside each subfolder there must be
an ``analysis_results.json`` containing the configuration used for that run.
For every subfolder the tool:

1. Deletes every file that is *not* a JSON document or an image figure.
2. Removes now-empty directories that do not contain any retained files.
3. Renames the subfolder using a concise slug composed of the requested
   parameter overrides.

Only the parameters explicitly listed in the requirements are included in the
slug. Missing parameters are skipped gracefully, and name collisions are
resolved by appending a numerical suffix.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Tuple

ALLOWED_FIGURE_EXTENSIONS = {
    ".json",
    ".png"
}
ANALYSIS_FILENAME = "analysis_results.json"
PERFORMANCE_DIRNAME = "performance_figures"

PARAM_KEYS: Tuple[str, ...] = (
    "reference_video",
    "width",
    "height",
    "block_size",
    "shrink_amount",
    "removability_alpha",
    "removability_smoothing_beta",
    "downsample_strength_target_bitrate",
    "propainter_ref_stride",
    "propainter_neighbor_length",
    "propainter_subvideo_length",
    "propainter_mask_dilation",
    "propainter_raft_iter",
    "e2fgvi_ref_stride",
    "e2fgvi_neighbor_stride",
    "e2fgvi_num_ref",
    "e2fgvi_mask_dilation",
    "realesrgan_denoise_strength",
    "realesrgan_tile",
    "realesrgan_tile_pad",
    "realesrgan_pre_pad",
    "instantir_cfg",
    "instantir_creative_start",
    "instantir_preview_start",
)

SLUG_LABELS: Dict[str, str] = {
    "reference_video": "vid",
    "width": "w",
    "height": "h",
    "block_size": "blk",
    "shrink_amount": "shr",
    "removability_alpha": "ra",
    "removability_smoothing_beta": "rb",
    "downsample_strength_target_bitrate": "dstb",
    "propainter_ref_stride": "pref",
    "propainter_neighbor_length": "pnei",
    "propainter_subvideo_length": "psub",
    "propainter_mask_dilation": "pdil",
    "propainter_raft_iter": "piter",
    "e2fgvi_ref_stride": "eref",
    "e2fgvi_neighbor_stride": "enei",
    "e2fgvi_num_ref": "enum",
    "e2fgvi_mask_dilation": "edil",
    "realesrgan_denoise_strength": "rd",
    "realesrgan_tile": "rtile",
    "realesrgan_tile_pad": "rtpad",
    "realesrgan_pre_pad": "rppad",
    "instantir_cfg": "icfg",
    "instantir_creative_start": "icreat",
    "instantir_preview_start": "iprev",
}

MAX_SLUG_LENGTH = 160

FUNCTION_KEY_MAP: Dict[str, Tuple[str, str]] = {
    "propainter_ref_stride": ("inpaint_with_propainter", "ref_stride"),
    "propainter_neighbor_length": ("inpaint_with_propainter", "neighbor_length"),
    "propainter_subvideo_length": ("inpaint_with_propainter", "subvideo_length"),
    "propainter_mask_dilation": ("inpaint_with_propainter", "mask_dilation"),
    "propainter_raft_iter": ("inpaint_with_propainter", "raft_iter"),
    "e2fgvi_ref_stride": ("inpaint_with_e2fgvi", "ref_stride"),
    "e2fgvi_neighbor_stride": ("inpaint_with_e2fgvi", "neighbor_stride"),
    "e2fgvi_num_ref": ("inpaint_with_e2fgvi", "num_ref"),
    "e2fgvi_mask_dilation": ("inpaint_with_e2fgvi", "mask_dilation"),
    "realesrgan_denoise_strength": ("restore_downsampled_with_realesrgan", "denoise_strength"),
    "realesrgan_tile": ("restore_downsampled_with_realesrgan", "tile"),
    "realesrgan_tile_pad": ("restore_downsampled_with_realesrgan", "tile_pad"),
    "realesrgan_pre_pad": ("restore_downsampled_with_realesrgan", "pre_pad"),
    "instantir_cfg": ("restore_with_instantir_adaptive", "cfg"),
    "instantir_creative_start": ("restore_with_instantir_adaptive", "creative_start"),
    "instantir_preview_start": ("restore_with_instantir_adaptive", "preview_start"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean and rename grid-search result folders.")
    parser.add_argument("root", type=Path, help="Path to the grid search results directory.")
    return parser.parse_args()


def clean_run_folder(run_dir: Path) -> None:
    """Remove everything except the analysis JSON and performance figures."""

    for item in list(run_dir.iterdir()):
        if item.is_file():
            if item.name != ANALYSIS_FILENAME:
                item.unlink()
            continue
        if item.is_dir() and item.name == PERFORMANCE_DIRNAME:
            continue
        if item.is_dir():
            shutil.rmtree(item)


def value_from_payload(payload: Dict[str, Any], key: str) -> Any:
    parameters = payload.get("parameters", {})
    config = parameters.get("config", {})
    if key in config:
        return config[key]

    func_info = FUNCTION_KEY_MAP.get(key)
    if func_info:
        func_name, inner_key = func_info
        return parameters.get("functions", {}).get(func_name, {}).get(inner_key)

    if key == "reference_video":
        return payload.get("video_name") or payload.get(key)

    return payload.get(key)


def sanitise_token(key: str, value: Any) -> str:
    if value is None:
        return ""
    if key == "reference_video":
        value = Path(str(value)).name
    elif isinstance(value, bool):
        value = int(value)

    if isinstance(value, float) and value.is_integer():
        value = int(value)

    token = str(value)
    safe_chars: List[str] = []
    for char in token:
        if char.isalnum() or char in {"-", "_", "."}:
            safe_chars.append(char)
        else:
            safe_chars.append("-")
    cleaned = "".join(safe_chars).strip("-_.")
    return cleaned.lower()


def build_slug(payload: Dict[str, Any]) -> str:
    parts: List[str] = []
    for key in PARAM_KEYS:
        token = sanitise_token(key, value_from_payload(payload, key))
        if token:
            label = SLUG_LABELS.get(key, key)
            parts.append(f"{label}-{token}")
    slug = "__".join(parts) if parts else "run"
    if len(slug) <= MAX_SLUG_LENGTH:
        return slug
    digest = hashlib.sha1(slug.encode("utf-8")).hexdigest()[:10]
    truncated = slug[: MAX_SLUG_LENGTH - 12]
    return f"{truncated}__{digest}"


def ensure_unique_path(root: Path, slug: str) -> Path:
    candidate = root / slug
    if not candidate.exists():
        return candidate
    counter = 1
    while True:
        candidate = root / f"{slug}__{counter}"
        if not candidate.exists():
            return candidate
        counter += 1


def main() -> None:
    args = parse_args()
    root = args.root.expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"Invalid grid search directory: {root}")

    run_dirs = [path for path in root.iterdir() if path.is_dir()]

    for run_dir in run_dirs:
        clean_run_folder(run_dir)

    for run_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        analysis_path = run_dir / ANALYSIS_FILENAME
        if not analysis_path.exists():
            print(f"Skipping {run_dir}: missing {ANALYSIS_FILENAME}")
            continue

        with analysis_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        slug = build_slug(payload)
        target_path = ensure_unique_path(root, slug)
        if target_path == run_dir:
            continue
        run_dir.rename(target_path)


if __name__ == "__main__":
    main()
