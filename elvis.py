import time
import shutil
import os
import subprocess
from pathlib import Path
import cv2
import numpy as np
from typing import List, Callable, Tuple, Dict, Optional
from skimage.metrics import structural_similarity as ssim
from PIL import Image
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
from instantir import load_runtime, restore_image

# Cache platform-specific constants at module level for performance
NULL_DEVICE = "NUL" if platform.system() == "Windows" else "/dev/null"
IS_WINDOWS = platform.system() == "Windows"

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

def inpaint_with_propainter(stretched_frames_dir: str, removal_masks_dir: str, output_frames_dir: str, width: int, height: int, framerate: float, propainter_dir: str = None, resize_ratio: float = 1.0, ref_stride: int = 20, neighbor_length: int = 4, subvideo_length: int = 40, mask_dilation: int = 4, raft_iter: int = 20, fp16: bool = True) -> None:
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
        fp16: Use fp16 (half precision) during inference (default: False)
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
        with tempfile.TemporaryDirectory() as temp_dir:
            # Prepare input and output directories within the temp directory
            temp_dir_path = Path(temp_dir)
            video_input_dir = temp_dir_path / "propainter_job"
            mask_input_dir = temp_dir_path / "propainter_masks"
            output_root_dir = temp_dir_path / "propainter_output"
            video_input_dir.mkdir(parents=True, exist_ok=True)
            mask_input_dir.mkdir(parents=True, exist_ok=True)
            output_root_dir.mkdir(parents=True, exist_ok=True)
            # Copy stretched frames and masks into the ProPainter input directories
            frame_files = sorted([f for f in os.listdir(stretched_frames_abs) if f.lower().endswith(('.jpg', '.png'))])
            mask_files = sorted([f for f in os.listdir(removal_masks_abs) if f.lower().endswith(('.jpg', '.png'))])
            if len(frame_files) == 0 or len(frame_files) != len(mask_files):
                raise ValueError("Frame and mask counts must match and be non-zero for ProPainter input.")
            # Copy frames to video input directory
            for idx, frame_file in enumerate(frame_files):
                src_frame = Path(stretched_frames_abs) / frame_file
                dst_frame = video_input_dir / f"{idx:04d}.png"
                shutil.copy(src_frame, dst_frame)
            # Copy masks to mask input directory
            for idx, mask_file in enumerate(mask_files):
                src_mask = Path(removal_masks_abs) / mask_file
                dst_mask = mask_input_dir / f"{idx:04d}.png"
                shutil.copy(src_mask, dst_mask)

            # Build ProPainter inference command
            if use_installed_package:
                propainter_cmd = [
                    sys.executable,
                    "-m",
                    "propainter.inference_propainter",
                ]
                run_cwd = original_dir
            else:
                propainter_cmd = [
                    sys.executable,
                    os.path.join(propainter_dir, "inference_propainter.py"),
                ]
                run_cwd = propainter_dir
            propainter_cmd.extend([
                "--video", str(video_input_dir),
                "--mask", str(mask_input_dir),
                "--output", str(output_root_dir),
                "--width", str(width),
                "--height", str(height),
                "--resize_ratio", str(resize_ratio),
                "--ref_stride", str(ref_stride),
                "--neighbor_length", str(neighbor_length),
                "--subvideo_length", str(subvideo_length),
                "--mask_dilation", str(mask_dilation),
                "--raft_iter", str(raft_iter),
                "--save_fps", str(int(framerate)),
            ])

            if fp16:
                propainter_cmd.append("--fp16")

            result = subprocess.run(propainter_cmd, capture_output=True, text=True, cwd=run_cwd)
            if result.returncode != 0:
                print(f"ProPainter stdout: {result.stdout}")
                print(f"ProPainter stderr: {result.stderr}")
                raise RuntimeError("ProPainter inference failed. See logs above for details.")

            video_name = video_input_dir.name
            generated_frames_dir = output_root_dir / video_name / "frames"
            if not generated_frames_dir.exists():
                raise RuntimeError(f"ProPainter did not emit frames at {generated_frames_dir}")

            # Copy generated frames to output directory
            os.makedirs(output_frames_abs, exist_ok=True)
            generated_files = sorted([p for p in generated_frames_dir.iterdir() if p.suffix.lower() == ".png"])
            if not generated_files:
                raise RuntimeError(f"No frames produced in {generated_frames_dir}")

            for idx, frame_path in enumerate(generated_files, start=1):
                dst_frame = Path(output_frames_abs) / f"{idx:05d}.png"
                shutil.copy(frame_path, dst_frame)

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

def upscale_realesrgan_adaptive(downsampled_image: np.ndarray, downscale_maps: np.ndarray, block_size: int, realesrgan_dir: str = None) -> np.ndarray:
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
        temp_dir: Temporary directory for intermediate files
    
    Returns:
        The adaptively upscaled image at original resolution
    """

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
        current_image = upscale_realesrgan_2x(current_image, realesrgan_dir)

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
    seed: Optional[int] = 42,) -> None:
    """
    Applies adaptive InstantIR blind image restoration to frames based on per-block blur maps.
    
    This function uses the packaged InstantIR runtime for efficient in-memory restoration.
    Frames are processed iteratively, applying restoration one round at a time. After each round,
    blocks that have completed their required deblurring are restored from saved originals,
    preventing over-restoration.
    
    Algorithm:
    1. Load InstantIR runtime once (keeps models warm)
    2. Calculate max blur rounds across all frames
    3. Load all frames and save blocks that don't need restoration (blur_rounds <= 0)
    4. For each restoration round (max_blur_rounds iterations):
       a. Apply InstantIR to each frame (1 inference step per call)
       b. Split restored frames into blocks
       c. Restore completed blocks (blur_maps <= 0) from saved originals
       d. Decrement all blur_maps by 1
       e. Save frames back to input folder
    5. Input folder now contains fully restored frames
    
    Args:
        input_frames_dir: Directory containing input frames (e.g., '00001.png', '00002.png', ...)
        blur_maps: 3D array (num_frames, num_blocks_y, num_blocks_x) indicating blur rounds per block
        block_size: The side length of each block
        instantir_weights_dir: Path to directory containing InstantIR weights (adapter.pt, aggregator.pt)
            If None, defaults to './InstantIR/models'
        cfg: Classifier-Free Guidance scale (default: 7.0)
        creative_start: Proportion of timesteps for creative restoration (default: 1.0)
        preview_start: Proportion to stop previewing at beginning (default: 0.0)
        seed: Random seed for reproducibility (default: 42)
    
    Returns:
        None (frames are modified in-place in input_frames_dir)
    """

    print("  Loading InstantIR runtime...")

    weights_dir = Path(instantir_weights_dir or "./InstantIR/models").expanduser()
    if not weights_dir.exists():
        raise FileNotFoundError(f"InstantIR weights directory not found: {weights_dir}")

    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    runtime = load_runtime(
        instantir_path=weights_dir,
        device=device,
        torch_dtype=dtype,
        map_location="cpu",
    )
    
    print(f"  Starting adaptive InstantIR restoration on frames in {input_frames_dir}...")

    # Load frames
    frames_files = sorted([f for f in os.listdir(input_frames_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
    num_frames = len(frames_files)
    if num_frames == 0:
        raise ValueError(f"No frames found in {input_frames_dir}")
    if num_frames != blur_maps.shape[0]:
        raise ValueError(f"Number of frames ({num_frames}) doesn't match blur_maps shape ({blur_maps.shape[0]})")
    frames_images = [cv2.imread(os.path.join(input_frames_dir, f)) for f in frames_files]
    # Split frames into blocks
    frames_blocks = np.array([split_image_into_blocks(frame, block_size) for frame in frames_images])

    num_blocks_y, num_blocks_x = blur_maps.shape[1], blur_maps.shape[2]
    # Find maximum blur rounds across all frames
    max_blur_rounds = int(np.max(blur_maps))
    if max_blur_rounds == 0:
        print("  No blurring detected, skipping restoration.")
        return
    
    working_blur_maps = blur_maps.copy().astype(np.int32)
    
    for round_num in range(max_blur_rounds):
        print(f"    Restoration round {round_num + 1}/{max_blur_rounds}...")
        
        # Process each frame with InstantIR
        for i, frame in enumerate(frames_images):
            # Apply InstantIR with 1 inference step using warm runtime
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_pil = Image.fromarray(frame_rgb)
            frame_seed = None
            if seed is not None:
                frame_seed = seed + round_num * num_frames + i

            restored_pil = restore_image(
                runtime,
                frame_pil,
                num_inference_steps=1,
                cfg=cfg,
                preview_start=preview_start,
                creative_start=creative_start,
                seed=frame_seed,
            )
            restored = cv2.cvtColor(np.array(restored_pil), cv2.COLOR_RGB2BGR)
            
            # Split into blocks
            restored_blocks = split_image_into_blocks(restored, block_size)

            # Restore completed blocks from original if their blur rounds are <= 0
            for by in range(num_blocks_y):
                for bx in range(num_blocks_x):
                    if working_blur_maps[i, by, bx] <= 0:
                        # Restore from original
                        restored_blocks[by, bx] = frames_blocks[i, by, bx]

            # Combine blocks back into image
            frames_images[i] = combine_blocks_into_image(restored_blocks)

        # Decrement blur maps for all frames after processing all frames in this round
        working_blur_maps -= 1
        
    # Save restored frames back to input directory
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
    with multiprocessing.Pool() as pool:
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

def calculate_lpips_per_frame(reference_frames: List[np.ndarray], decoded_frames: List[np.ndarray], 
                               device: str = 'cuda' if torch.cuda.is_available() else 'cpu') -> List[float]:
    """
    Calculate LPIPS (Learned Perceptual Image Patch Similarity) for each frame pair.
    
    Args:
        reference_frames: List of reference frames (BGR format from OpenCV)
        decoded_frames: List of decoded frames (BGR format from OpenCV)
        device: Device to run LPIPS on ('cuda' or 'cpu')
    
    Returns:
        List of LPIPS scores (lower is better, range typically 0-1)
    """
    # Initialize LPIPS model (using Alex network by default, can also use 'vgg' or 'squeeze')
    lpips_model = lpips.LPIPS(net='alex').to(device)
    lpips_scores = []
    
    with torch.no_grad():
        for ref_frame, dec_frame in zip(reference_frames, decoded_frames):
            # Convert BGR to RGB
            ref_rgb = cv2.cvtColor(ref_frame, cv2.COLOR_BGR2RGB)
            dec_rgb = cv2.cvtColor(dec_frame, cv2.COLOR_BGR2RGB)
            
            # Convert to tensor and normalize to [-1, 1]
            ref_tensor = torch.from_numpy(ref_rgb).permute(2, 0, 1).unsqueeze(0).float() / 127.5 - 1.0
            dec_tensor = torch.from_numpy(dec_rgb).permute(2, 0, 1).unsqueeze(0).float() / 127.5 - 1.0
            
            # Move to device
            ref_tensor = ref_tensor.to(device)
            dec_tensor = dec_tensor.to(device)
            
            # Calculate LPIPS
            lpips_score = lpips_model(ref_tensor, dec_tensor).item()
            lpips_scores.append(lpips_score)
    
    return lpips_scores

def calculate_vmaf(reference_video: str, distorted_video: str, width: int, height: int, 
                   framerate: float, model_path: str = None) -> Dict[str, float]:
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
    def _convert_to_yuv(video_path: str, output_yuv: str, width: int, height: int) -> bool:
        """Convert a video to YUV420p format."""
        try:
            convert_cmd = [
                'ffmpeg', '-hide_banner', '-loglevel', 'error',
                '-i', video_path,
                '-pix_fmt', 'yuv420p',
                '-s', f'{width}x{height}',
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
            if not _convert_to_yuv(reference_video, ref_yuv, width, height):
                return {'mean': 0, 'min': 0, 'max': 0, 'std': 0, 'harmonic_mean': 0}
        
        # Convert distorted video if needed
        if not distorted_video.endswith('.yuv'):
            temp_dist_yuv = tempfile.NamedTemporaryFile(suffix='.yuv', delete=False)
            temp_dist_yuv.close()
            dist_yuv = temp_dist_yuv.name
            print(f"  - Converting distorted video to YUV format...")
            if not _convert_to_yuv(distorted_video, dist_yuv, width, height):
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
    log_root: Optional[str] = None) -> float:
    """Calculate FVMD (Fréchet Video Mask Distance) between two frame sequences.

    Args:
        reference_frames: List of ground-truth frames in BGR, one per timestep.
        decoded_frames: List of generated frames in BGR, aligned with reference frames.
        log_root: Optional directory where FVMD can persist logs. If omitted, a
            temporary directory is used and cleaned up automatically.

    Returns:
        The FVMD score as a floating point value (lower is better).
    """

    if not reference_frames or not decoded_frames:
        raise ValueError("Both reference_frames and decoded_frames must contain at least one frame.")

    if len(reference_frames) != len(decoded_frames):
        raise ValueError("Reference and decoded frame lists must have the same length for FVMD computation.")

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        gt_root = tmp_path / "gt"
        gen_root = tmp_path / "gen"
        # FVMD expects dataset_root/clip_name/frame.png structure, so create a single clip.
        clip_name = "clip_0001"
        gt_clip = gt_root / clip_name
        gen_clip = gen_root / clip_name
        gt_clip.mkdir(parents=True, exist_ok=True)
        gen_clip.mkdir(parents=True, exist_ok=True)

        for idx, (ref_frame, dec_frame) in enumerate(zip(reference_frames, decoded_frames), start=1):
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

def analyze_encoding_performance(reference_frames: List[np.ndarray], encoded_videos: Dict[str, str], block_size: int, width: int, height: int, temp_dir: str, masks_dir: str, sample_frames: List[int] = [0, 20, 40], video_bitrates: Dict[str, float] = {}, reference_video_path: str = None, framerate: float = 30.0, strength_maps: Dict[str, np.ndarray] = None, generate_opencv_benchmarks: bool = True) -> Dict:
    """
    A comprehensive function to analyze and compare video encoding performance using masked videos.

    This function:
    1. Creates masked versions (foreground/background) of all videos.
    2. Optionally generates OpenCV-based restoration benchmarks for Elvis v2 methods.
    3. Calculates all metrics (PSNR, SSIM, LPIPS, VMAF) separately for foreground and background.
    4. Generates and saves quality heatmaps for sample frames.
    5. Prints a unified summary report comparing all encoding methods.

    Args:
        reference_frames: List of original reference frames.
        encoded_videos: Dictionary mapping a method name to its video file path.
        block_size: The size of blocks for visualization heatmaps.
        width: The width of the video frames.
        height: The height of the video frames.
        temp_dir: A directory for temporary files (masked videos, heatmaps).
        masks_dir: Path to the directory with UFO masks.
        sample_frames: A list of frame indices to generate heatmaps for.
        video_bitrates: Dictionary mapping video name to bitrate in bps.
        reference_video_path: Path to the reference video file (required for VMAF).
        framerate: Video framerate.
        strength_maps: Optional dictionary mapping method names to their strength maps (numpy arrays).
                      Used to generate OpenCV restoration benchmarks. Keys should match video names.
        generate_opencv_benchmarks: If True, generates OpenCV-based restoration benchmarks for
                                   Elvis v2 methods (bilinear, bicubic, Lanczos for downsampling;
                                   unsharp mask and Wiener approximation for blur).

    Returns:
        A dictionary containing the aggregated analysis results for each video.
    """
    
    # Helper functions
    
    def _create_masked_video(frames_dir: str, masks_dir: str, output_video: str, width: int, height: int, 
                            framerate: float, mask_mode: str = 'foreground') -> bool:
        """
        Creates a video with either foreground or background masked out (set to black).
        
        Args:
            frames_dir: Directory containing input frames
            masks_dir: Directory containing mask frames
            output_video: Output video path
            width: Video width
            height: Video height
            framerate: Video framerate
            mask_mode: Either 'foreground' (keep FG, black BG) or 'background' (keep BG, black FG)
        """
        try:
            masked_frames_dir = output_video.replace('.mp4', '_frames')
            os.makedirs(masked_frames_dir, exist_ok=True)
            
            frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith('.png')])
            
            for frame_idx, frame_file in enumerate(frame_files, start=1):
                frame_path = os.path.join(frames_dir, frame_file)
                # UFO masks are numbered sequentially (00001.png, 00002.png, etc.)
                mask_path = os.path.join(masks_dir, f"{frame_idx:05d}.png")
                
                frame = cv2.imread(frame_path)
                if not os.path.exists(mask_path):
                    print(f"Warning: Mask not found for frame {frame_idx} ({frame_file}), using original frame")
                    cv2.imwrite(os.path.join(masked_frames_dir, frame_file), frame)
                    continue
                
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
                
                # Create masked frame
                masked_frame = frame.copy()
                if mask_mode == 'foreground':
                    # Keep foreground (mask > 128), black out background
                    masked_frame[mask <= 128] = 0
                else:  # background
                    # Keep background (mask <= 128), black out foreground
                    masked_frame[mask > 128] = 0
                
                cv2.imwrite(os.path.join(masked_frames_dir, frame_file), masked_frame)
            
            # Encode masked frames to video
            encode_cmd = [
                'ffmpeg', '-hide_banner', '-loglevel', 'error',
                '-framerate', str(framerate),
                '-i', os.path.join(masked_frames_dir, '%05d.png'),
                '-vf', f'scale={width}:{height}:flags=lanczos,format=yuv420p',
                '-c:v', 'libx265',
                '-preset', 'veryslow',
                '-x265-params', 'lossless=1',
                '-y', output_video
            ]
            result = subprocess.run(encode_cmd, capture_output=True, text=True, check=True)
            
            # Clean up frames
            shutil.rmtree(masked_frames_dir, ignore_errors=True)
            return True
            
        except Exception as e:
            print(f"Error creating masked video: {e}")
            return False
    
    def _decode_video_to_frames(video_path: str, output_dir: str) -> bool:
        """Decodes a video into frames using FFmpeg. Returns True on success."""
        return decode_video(video_path, output_dir, quality=2)

    def _get_mask_bounding_box(masks_dir: str, width: int, height: int) -> Tuple[int, int, int, int]:
        """
        Calculate the bounding box that encompasses all foreground regions across all masks.
        
        Args:
            masks_dir: Directory containing mask images
            width: Frame width
            height: Frame height
            
        Returns:
            Tuple of (x, y, w, h) representing the bounding box
        """
        mask_files = sorted([f for f in os.listdir(masks_dir) if f.endswith('.png')])
        if not mask_files:
            return (0, 0, width, height)
        
        # Initialize bounds to find the union of all masks
        min_x, min_y = width, height
        max_x, max_y = 0, 0
        
        for mask_file in mask_files:
            mask_path = os.path.join(masks_dir, mask_file)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
            
            # Find foreground pixels (mask > 128)
            fg_pixels = np.where(mask > 128)
            
            if len(fg_pixels[0]) > 0:
                min_y = min(min_y, fg_pixels[0].min())
                max_y = max(max_y, fg_pixels[0].max())
                min_x = min(min_x, fg_pixels[1].min())
                max_x = max(max_x, fg_pixels[1].max())
        
        # If no foreground found, return full frame
        if min_x >= max_x or min_y >= max_y:
            return (0, 0, width, height)
        
        # Add a small padding (5% on each side)
        padding_x = max(1, int((max_x - min_x) * 0.05))
        padding_y = max(1, int((max_y - min_y) * 0.05))
        
        x = max(0, min_x - padding_x)
        y = max(0, min_y - padding_y)
        w = min(width - x, max_x - min_x + 2 * padding_x)
        h = min(height - y, max_y - min_y + 2 * padding_y)
        
        return (x, y, w, h)

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
            mask_colored[mask > 128] = [255, 0, 0] # Red for FG
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
            axes[1, 1].set_title(f'PSNR (Mean: {np.mean(psnr_map[np.isfinite(psnr_map)]):.1f} dB)')
            plt.colorbar(im2, ax=axes[1, 1])

            # Quality overlay
            axes[1, 2].imshow(cv2.cvtColor(ref_frame, cv2.COLOR_BGR2RGB))
            num_blocks_y, num_blocks_x = ssim_map.shape
            for i in range(num_blocks_y):
                for j in range(num_blocks_x):
                    if ssim_map[i, j] < 0.7:
                        color = 'red' if ssim_map[i, j] < 0.5 else 'yellow'
                        alpha = 0.6 if ssim_map[i, j] < 0.5 else 0.4
                        rect = patches.Rectangle((j * block_size, i * block_size), block_size, block_size, 
                                            linewidth=0, facecolor=color, alpha=alpha)
                        axes[1, 2].add_patch(rect)
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
                psnr_fg_change = 0
                psnr_bg_change = 0
                ssim_fg_change = 0
                ssim_bg_change = 0
                lpips_fg_change = 0
                lpips_bg_change = 0
                fvmd_fg_change = 0
                fvmd_bg_change = 0
                vmaf_fg_change = 0
                vmaf_bg_change = 0
                
                for metric in ['psnr', 'ssim', 'lpips', 'vmaf']:
                    for region in ['foreground', 'background']:
                        baseline_val = results[baseline_name][region].get(f'{metric}_mean', 0)
                        current_val = results[video_name][region].get(f'{metric}_mean', 0)
                        
                        if baseline_val > 0:
                            # For LPIPS, lower is better, so invert the change
                            if metric == 'lpips':
                                change = ((baseline_val / current_val) - 1) * 100 if current_val > 0 else 0
                            else:
                                change = ((current_val / baseline_val) - 1) * 100
                            
                            # Store changes
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
                
                # Print table row
                print(f"{video_name:<20} {psnr_fg_change:+.2f}%{' '*8} {psnr_bg_change:+.2f}%{' '*8} {ssim_fg_change:+.2f}%{' '*8} {ssim_bg_change:+.2f}%{' '*8} {lpips_fg_change:+.2f}%{' '*8} {lpips_bg_change:+.2f}%{' '*8} {fvmd_fg_change:+.2f}%{' '*8} {fvmd_bg_change:+.2f}%{' '*8} {vmaf_fg_change:+.2f}%{' '*8} {vmaf_bg_change:+.2f}%{' '*8}")
            
            print(f"{'-'*180}")
    
    # --- Setup ---
    os.makedirs(temp_dir, exist_ok=True)
    masked_videos_dir = os.path.join(temp_dir, "masked_videos")
    frames_root = os.path.join(temp_dir, "frames")
    heatmaps_dir = os.path.join(temp_dir, "performance_figures")
    os.makedirs(masked_videos_dir, exist_ok=True)
    os.makedirs(frames_root, exist_ok=True)
    os.makedirs(heatmaps_dir, exist_ok=True)
    fvmd_log_root = os.path.join(temp_dir, "fvmd_logs")
    
    if not os.path.isdir(masks_dir):
        print(f"Warning: Masks directory not found at '{masks_dir}'. Cannot perform FG/BG analysis.")
        return {}
    
    # Create masked reference video
    print("Creating masked reference videos...")
    reference_frames_dir = os.path.join(temp_dir, "reference_frames_temp")
    os.makedirs(reference_frames_dir, exist_ok=True)
    for i, frame in enumerate(reference_frames):
        cv2.imwrite(os.path.join(reference_frames_dir, f"{i+1:05d}.png"), frame)
    
    ref_fg_video = os.path.join(masked_videos_dir, "reference_fg.mp4")
    ref_bg_video = os.path.join(masked_videos_dir, "reference_bg.mp4")
    
    _create_masked_video(reference_frames_dir, masks_dir, ref_fg_video, width, height, framerate, 'foreground')
    _create_masked_video(reference_frames_dir, masks_dir, ref_bg_video, width, height, framerate, 'background')
    
    # --- Generate OpenCV Restoration Benchmarks ---
    opencv_benchmarks = {}
    if generate_opencv_benchmarks and strength_maps is not None:
        print("\n" + "="*80)
        print("GENERATING OPENCV RESTORATION BENCHMARKS")
        print("="*80)
        
        benchmarks_dir = os.path.join(temp_dir, "opencv_benchmarks")
        os.makedirs(benchmarks_dir, exist_ok=True)
        
        # Process each Elvis v2 method that has strength maps
        for method_name, maps in strength_maps.items():
            if maps is None:
                continue
                
            print(f"\nProcessing benchmarks for: {method_name}")
            
            # Determine the type of restoration needed based on method name
            if "downsample" in method_name.lower():
                # Generate downsampling restoration benchmarks
                # First, apply downsampling to reference frames, then restore
                print("  - Generating bilinear restoration benchmark...")
                benchmark_frames_bilinear = []
                for frame_idx, frame in enumerate(reference_frames):
                    # Apply downsampling first
                    downsampled_frame, _ = filter_frame_downsample(frame, maps[frame_idx] / np.max(maps[frame_idx]) if np.max(maps[frame_idx]) > 0 else maps[frame_idx], block_size)
                    # Restore using bilinear
                    restored_frame = restore_downsample_opencv_bilinear(downsampled_frame, maps[frame_idx], block_size)
                    benchmark_frames_bilinear.append(restored_frame)
                
                # Encode to video
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
                    # Apply downsampling first
                    downsampled_frame, _ = filter_frame_downsample(frame, maps[frame_idx] / np.max(maps[frame_idx]) if np.max(maps[frame_idx]) > 0 else maps[frame_idx], block_size)
                    # Restore using bicubic
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
                    # Apply downsampling first
                    downsampled_frame, _ = filter_frame_downsample(frame, maps[frame_idx] / np.max(maps[frame_idx]) if np.max(maps[frame_idx]) > 0 else maps[frame_idx], block_size)
                    # Restore using Lanczos
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
                # Generate blur restoration benchmarks
                print("  - Generating unsharp mask restoration benchmark...")
                benchmark_frames_unsharp = []
                for frame_idx, frame in enumerate(reference_frames):
                    # Apply blurring first
                    blurred_frame, _ = filter_frame_gaussian(frame, maps[frame_idx] / np.max(maps[frame_idx]) if np.max(maps[frame_idx]) > 0 else maps[frame_idx], block_size)
                    # Restore using unsharp mask
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
                    # Apply blurring first
                    blurred_frame, _ = filter_frame_gaussian(frame, maps[frame_idx] / np.max(maps[frame_idx]) if np.max(maps[frame_idx]) > 0 else maps[frame_idx], block_size)
                    # Restore using Wiener approximation
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
        
        # Add OpenCV benchmarks to the encoded_videos dictionary
        encoded_videos.update(opencv_benchmarks)
        
        print(f"\nGenerated {len(opencv_benchmarks)} OpenCV restoration benchmarks.")
        print("="*80 + "\n")
    
    analysis_results = {}
    
    # Calculate the bounding box for foreground region once
    print("Calculating foreground bounding box from masks...")
    fg_bbox = _get_mask_bounding_box(masks_dir, width, height)
    print(f"  - Foreground bounding box: x={fg_bbox[0]}, y={fg_bbox[1]}, w={fg_bbox[2]}, h={fg_bbox[3]}")

    # --- Main Loop: Process each video ---
    for video_name, video_path in encoded_videos.items():
        print(f"\nProcessing '{video_name}'...")
        if not os.path.exists(video_path):
            print(f"  - Video not found, skipping.")
            continue

        # 1. Create masked versions of the encoded video
        video_decoded_dir = os.path.join(frames_root, f"{video_name.replace(' ', '_')}_decoded")
        if not _decode_video_to_frames(video_path, video_decoded_dir):
            continue
        
        print(f"  - Creating masked versions of '{video_name}'...")
        enc_fg_video = os.path.join(masked_videos_dir, f"{video_name.replace(' ', '_')}_fg.mp4")
        enc_bg_video = os.path.join(masked_videos_dir, f"{video_name.replace(' ', '_')}_bg.mp4")
        
        _create_masked_video(video_decoded_dir, masks_dir, enc_fg_video, width, height, framerate, 'foreground')
        _create_masked_video(video_decoded_dir, masks_dir, enc_bg_video, width, height, framerate, 'background')
        
        decoded_frame_files = sorted([f for f in os.listdir(video_decoded_dir) if f.endswith('.png')])
        num_frames = min(len(reference_frames), len(decoded_frame_files))
        
        # 2. Calculate all metrics using masked videos for foreground and background
        analysis_results[video_name] = {
            'foreground': {},
            'background': {},
            'bitrate_mbps': video_bitrates.get(video_name, 0) / 1000000
        }
        
        # Calculate metrics for FOREGROUND
        print(f"  - Calculating foreground metrics...")
        
        # Decode masked videos for frame-by-frame metrics
        ref_fg_frames_dir = os.path.join(temp_dir, "ref_fg_frames")
        enc_fg_frames_dir = os.path.join(temp_dir, f"{video_name.replace(' ', '_')}_fg_frames")
        os.makedirs(ref_fg_frames_dir, exist_ok=True)
        os.makedirs(enc_fg_frames_dir, exist_ok=True)
        
        _decode_video_to_frames(ref_fg_video, ref_fg_frames_dir)
        _decode_video_to_frames(enc_fg_video, enc_fg_frames_dir)
        
        # Load frames
        ref_fg_frame_files = sorted([f for f in os.listdir(ref_fg_frames_dir) if f.endswith('.png')])
        enc_fg_frame_files = sorted([f for f in os.listdir(enc_fg_frames_dir) if f.endswith('.png')])
        num_fg_frames = min(len(ref_fg_frame_files), len(enc_fg_frame_files))
        
        # Calculate PSNR and SSIM frame-by-frame
        psnr_scores_fg = []
        ssim_scores_fg = []
        for i in range(num_fg_frames):
            ref_frame = cv2.imread(os.path.join(ref_fg_frames_dir, ref_fg_frame_files[i]))
            enc_frame = cv2.imread(os.path.join(enc_fg_frames_dir, enc_fg_frame_files[i]))
            
            if ref_frame is not None and enc_frame is not None:
                # Crop frames to foreground bounding box to exclude black background
                x, y, w, h = fg_bbox
                ref_frame_cropped = ref_frame[y:y+h, x:x+w]
                enc_frame_cropped = enc_frame[y:y+h, x:x+w]
                
                # Calculate PSNR for the cropped frame
                psnr_val = psnr(ref_frame_cropped, enc_frame_cropped)
                if np.isfinite(psnr_val):
                    psnr_scores_fg.append(psnr_val)
                
                # Calculate SSIM for the cropped frame
                ssim_val = ssim(ref_frame_cropped, enc_frame_cropped, gaussian_weights=True, data_range=255, channel_axis=-1)
                if np.isfinite(ssim_val):
                    ssim_scores_fg.append(ssim_val)
        
        analysis_results[video_name]['foreground']['psnr_mean'] = np.mean(psnr_scores_fg) if psnr_scores_fg else 0
        analysis_results[video_name]['foreground']['psnr_std'] = np.std(psnr_scores_fg) if psnr_scores_fg else 0
        analysis_results[video_name]['foreground']['ssim_mean'] = np.mean(ssim_scores_fg) if ssim_scores_fg else 0
        analysis_results[video_name]['foreground']['ssim_std'] = np.std(ssim_scores_fg) if ssim_scores_fg else 0
        
        # Calculate LPIPS for foreground (crop to bounding box)
        x, y, w, h = fg_bbox
        ref_fg_frames_list = [cv2.imread(os.path.join(ref_fg_frames_dir, f))[y:y+h, x:x+w] for f in ref_fg_frame_files[:num_fg_frames]]
        enc_fg_frames_list = [cv2.imread(os.path.join(enc_fg_frames_dir, f))[y:y+h, x:x+w] for f in enc_fg_frame_files[:num_fg_frames]]
        lpips_scores_fg = calculate_lpips_per_frame(ref_fg_frames_list, enc_fg_frames_list)
        analysis_results[video_name]['foreground']['lpips_mean'] = np.mean(lpips_scores_fg) if lpips_scores_fg else 0
        analysis_results[video_name]['foreground']['lpips_std'] = np.std(lpips_scores_fg) if lpips_scores_fg else 0
        
        # Calculate FVMD for foreground
        print(f"  - Calculating FVMD for foreground...")
        fvmd_fg = calculate_fvmd(ref_fg_frames_list, enc_fg_frames_list, log_root=fvmd_log_root)
        analysis_results[video_name]['foreground']['fvmd'] = fvmd_fg
        
        # Calculate VMAF for foreground (crop videos to bounding box)
        x, y, w, h = fg_bbox
        ref_fg_cropped_video = os.path.join(masked_videos_dir, "reference_fg_cropped.mp4")
        enc_fg_cropped_video = os.path.join(masked_videos_dir, f"{video_name.replace(' ', '_')}_fg_cropped.mp4")
        
        # Create cropped versions of the videos using FFmpeg
        crop_filter = f"crop={w}:{h}:{x}:{y}"
        for input_video, output_video in [(ref_fg_video, ref_fg_cropped_video), (enc_fg_video, enc_fg_cropped_video)]:
            crop_cmd = [
                'ffmpeg', '-hide_banner', '-loglevel', 'error', '-y',
                '-i', input_video,
                '-vf', crop_filter,
                '-c:v', 'libx265', '-preset', 'veryslow', '-x265-params', 'lossless=1',
                output_video
            ]
            subprocess.run(crop_cmd, capture_output=True, text=True, check=True)
        
        vmaf_fg = calculate_vmaf(ref_fg_cropped_video, enc_fg_cropped_video, w, h, framerate)
        analysis_results[video_name]['foreground']['vmaf_mean'] = vmaf_fg.get('mean', 0)
        analysis_results[video_name]['foreground']['vmaf_std'] = vmaf_fg.get('std', 0)
        
        # Clean up FG temp directories
        shutil.rmtree(ref_fg_frames_dir, ignore_errors=True)
        shutil.rmtree(enc_fg_frames_dir, ignore_errors=True)
        
        # Calculate metrics for BACKGROUND
        print(f"  - Calculating background metrics...")
        
        ref_bg_frames_dir = os.path.join(temp_dir, "ref_bg_frames")
        enc_bg_frames_dir = os.path.join(temp_dir, f"{video_name.replace(' ', '_')}_bg_frames")
        os.makedirs(ref_bg_frames_dir, exist_ok=True)
        os.makedirs(enc_bg_frames_dir, exist_ok=True)
        
        _decode_video_to_frames(ref_bg_video, ref_bg_frames_dir)
        _decode_video_to_frames(enc_bg_video, enc_bg_frames_dir)
        
        ref_bg_frame_files = sorted([f for f in os.listdir(ref_bg_frames_dir) if f.endswith('.png')])
        enc_bg_frame_files = sorted([f for f in os.listdir(enc_bg_frames_dir) if f.endswith('.png')])
        num_bg_frames = min(len(ref_bg_frame_files), len(enc_bg_frame_files))
        
        # Calculate PSNR and SSIM frame-by-frame
        psnr_scores_bg = []
        ssim_scores_bg = []
        for i in range(num_bg_frames):
            ref_frame = cv2.imread(os.path.join(ref_bg_frames_dir, ref_bg_frame_files[i]))
            enc_frame = cv2.imread(os.path.join(enc_bg_frames_dir, enc_bg_frame_files[i]))
            
            if ref_frame is not None and enc_frame is not None:
                psnr_val = psnr(ref_frame, enc_frame)
                if np.isfinite(psnr_val):
                    psnr_scores_bg.append(psnr_val)
                
                ssim_val = ssim(ref_frame, enc_frame, gaussian_weights=True, data_range=255, channel_axis=-1)
                if np.isfinite(ssim_val):
                    ssim_scores_bg.append(ssim_val)
        
        analysis_results[video_name]['background']['psnr_mean'] = np.mean(psnr_scores_bg) if psnr_scores_bg else 0
        analysis_results[video_name]['background']['psnr_std'] = np.std(psnr_scores_bg) if psnr_scores_bg else 0
        analysis_results[video_name]['background']['ssim_mean'] = np.mean(ssim_scores_bg) if ssim_scores_bg else 0
        analysis_results[video_name]['background']['ssim_std'] = np.std(ssim_scores_bg) if ssim_scores_bg else 0
        
        # Calculate LPIPS for background
        ref_bg_frames_list = [cv2.imread(os.path.join(ref_bg_frames_dir, f)) for f in ref_bg_frame_files[:num_bg_frames]]
        enc_bg_frames_list = [cv2.imread(os.path.join(enc_bg_frames_dir, f)) for f in enc_bg_frame_files[:num_bg_frames]]
        lpips_scores_bg = calculate_lpips_per_frame(ref_bg_frames_list, enc_bg_frames_list)
        analysis_results[video_name]['background']['lpips_mean'] = np.mean(lpips_scores_bg) if lpips_scores_bg else 0
        analysis_results[video_name]['background']['lpips_std'] = np.std(lpips_scores_bg) if lpips_scores_bg else 0
        
        # Calculate FVMD for background
        print(f"  - Calculating FVMD for background...")
        fvmd_bg = calculate_fvmd(ref_bg_frames_list, enc_bg_frames_list, log_root=fvmd_log_root)
        analysis_results[video_name]['background']['fvmd'] = fvmd_bg
        
        # Calculate VMAF for background
        vmaf_bg = calculate_vmaf(ref_bg_video, enc_bg_video, width, height, framerate)
        analysis_results[video_name]['background']['vmaf_mean'] = vmaf_bg.get('mean', 0)
        analysis_results[video_name]['background']['vmaf_std'] = vmaf_bg.get('std', 0)
        
        # Clean up BG temp directories
        shutil.rmtree(ref_bg_frames_dir, ignore_errors=True)
        shutil.rmtree(enc_bg_frames_dir, ignore_errors=True)

    # --- Generate Visualizations ---
    print("\nGenerating quality visualization charts...")
    
    def _generate_quality_visualizations(results: Dict, heatmaps_dir: str) -> None:
        """Generate comprehensive quality visualization charts."""
        if not results:
            return
        
        video_names = list(results.keys())
        
        # 1. Overall Quality Comparison Bar Chart
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Quality Metrics Comparison (Foreground vs Background)', fontsize=16, fontweight='bold')
        
        metrics = [
            ('psnr_mean', 'PSNR (dB)', [20, 50]),
            ('ssim_mean', 'SSIM', [0, 1]),
            ('lpips_mean', 'LPIPS (lower is better)', [0, 0.5]),
            ('vmaf_mean', 'VMAF', [0, 100])
        ]
        
        for idx, (metric_key, metric_label, ylim) in enumerate(metrics):
            ax = axes[idx // 2, idx % 2]
            
            fg_values = [results[name]['foreground'].get(metric_key, 0) for name in video_names]
            bg_values = [results[name]['background'].get(metric_key, 0) for name in video_names]
            
            x = np.arange(len(video_names))
            width = 0.35
            
            bars1 = ax.bar(x - width/2, fg_values, width, label='Foreground', alpha=0.8, color='#2E86AB')
            bars2 = ax.bar(x + width/2, bg_values, width, label='Background', alpha=0.8, color='#A23B72')
            
            # Set precision for y-axis ticks
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
            
            # Add value labels on bars
            for bars in [bars1, bars2]:
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.2f}',
                           ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(os.path.join(heatmaps_dir, '1_overall_quality_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Generated quality comparison visualization in {heatmaps_dir}")
    
    _generate_quality_visualizations(analysis_results, heatmaps_dir)

    # --- Generate Visual Patch Comparison ---
    def _generate_patch_comparison(
        reference_frames: List[np.ndarray],
        encoded_videos: Dict[str, str],
        results: Dict,
        masks_dir: str,
        heatmaps_dir: str,
        decoded_frames_root: str,
        sample_frame_idx: int = None,
        patch_size: int = 128
    ) -> None:
        """Generate visual comparison of FG/BG patches from each method with VMAF scores."""
        if not results or not reference_frames:
            return
        
        # Use middle frame if not specified
        if sample_frame_idx is None:
            sample_frame_idx = len(reference_frames) // 2
        
        print(f"  - Generating patch comparison visualization for frame {sample_frame_idx}...")
        
        # Load reference frame
        ref_frame = reference_frames[sample_frame_idx]
        
        # Load mask for this frame
        mask_files = sorted([f for f in os.listdir(masks_dir) if f.endswith('.png')])
        if sample_frame_idx >= len(mask_files):
            print(f"    Warning: Mask for frame {sample_frame_idx} not found. Skipping patch comparison.")
            return
        
        mask_path = os.path.join(masks_dir, mask_files[sample_frame_idx])
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"    Warning: Could not load mask from {mask_path}. Skipping patch comparison.")
            return
        
        # Find a good foreground patch (center of mass of mask)
        mask_binary = (mask > 127).astype(np.uint8)
        moments = cv2.moments(mask_binary)
        if moments['m00'] > 0:
            fg_center_x = int(moments['m10'] / moments['m00'])
            fg_center_y = int(moments['m01'] / moments['m00'])
        else:
            # Fallback to center of frame
            fg_center_x = ref_frame.shape[1] // 2
            fg_center_y = ref_frame.shape[0] // 2
        
        # Find good background patches (two different areas)
        mask_inverted = 1 - mask_binary
        
        # First background patch: center of mass of inverted mask
        moments_bg = cv2.moments(mask_inverted)
        if moments_bg['m00'] > 0:
            bg1_center_x = int(moments_bg['m10'] / moments_bg['m00'])
            bg1_center_y = int(moments_bg['m01'] / moments_bg['m00'])
        else:
            # Fallback to top-left corner
            bg1_center_x = ref_frame.shape[1] // 4
            bg1_center_y = ref_frame.shape[0] // 4
        
        # Second background patch: find another area away from first patch
        # Try top-right corner area first
        bg2_center_x = ref_frame.shape[1] * 3 // 4
        bg2_center_y = ref_frame.shape[0] // 4
        
        # If that's too close to the first patch, try bottom-left
        if abs(bg2_center_x - bg1_center_x) < patch_size and abs(bg2_center_y - bg1_center_y) < patch_size:
            bg2_center_x = ref_frame.shape[1] // 4
            bg2_center_y = ref_frame.shape[0] * 3 // 4
        
        # Extract patches from reference frame
        def extract_patch(frame, center_x, center_y, size):
            h, w = frame.shape[:2]
            x1 = max(0, center_x - size // 2)
            y1 = max(0, center_y - size // 2)
            x2 = min(w, x1 + size)
            y2 = min(h, y1 + size)
            x1 = max(0, x2 - size)
            y1 = max(0, y2 - size)
            return frame[y1:y2, x1:x2], (x1, y1, x2, y2)
        
        ref_fg_patch, fg_coords = extract_patch(ref_frame, fg_center_x, fg_center_y, patch_size)
        ref_bg1_patch, bg1_coords = extract_patch(ref_frame, bg1_center_x, bg1_center_y, patch_size)
        ref_bg2_patch, bg2_coords = extract_patch(ref_frame, bg2_center_x, bg2_center_y, patch_size)
        
        # Load decoded frames for each method and extract patches
        video_names = list(encoded_videos.keys())
        num_methods = len(video_names)
        
        # Create figure: 3 rows (FG/BG1/BG2) x (num_methods + 1) columns (reference + methods)
        fig, axes = plt.subplots(3, num_methods + 1, figsize=(4 * (num_methods + 1), 12))
        fig.suptitle(f'Visual Patch Comparison (Frame {sample_frame_idx + 1})', fontsize=16, fontweight='bold')
        
        # Plot reference patches
        axes[0, 0].imshow(cv2.cvtColor(ref_fg_patch, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title('Reference\nForeground', fontsize=10, fontweight='bold')
        axes[0, 0].axis('off')
        
        axes[1, 0].imshow(cv2.cvtColor(ref_bg1_patch, cv2.COLOR_BGR2RGB))
        axes[1, 0].set_title('Reference\nBackground 1', fontsize=10, fontweight='bold')
        axes[1, 0].axis('off')
        
        axes[2, 0].imshow(cv2.cvtColor(ref_bg2_patch, cv2.COLOR_BGR2RGB))
        axes[2, 0].set_title('Reference\nBackground 2', fontsize=10, fontweight='bold')
        axes[2, 0].axis('off')
        
        # Plot patches from each method
        for idx, video_name in enumerate(video_names):
            # Load decoded frame
            video_decoded_dir = os.path.join(decoded_frames_root, f"{video_name.replace(' ', '_')}_decoded")
            frame_files = sorted([f for f in os.listdir(video_decoded_dir) if f.endswith('.png')])
            
            if sample_frame_idx >= len(frame_files):
                continue
            
            decoded_frame = cv2.imread(os.path.join(video_decoded_dir, frame_files[sample_frame_idx]))
            if decoded_frame is None:
                continue
            
            # Extract patches
            dec_fg_patch, _ = extract_patch(decoded_frame, fg_center_x, fg_center_y, patch_size)
            dec_bg1_patch, _ = extract_patch(decoded_frame, bg1_center_x, bg1_center_y, patch_size)
            dec_bg2_patch, _ = extract_patch(decoded_frame, bg2_center_x, bg2_center_y, patch_size)
            
            # Get VMAF scores from results
            fg_vmaf = results[video_name]['foreground'].get('vmaf_mean', 0)
            bg_vmaf = results[video_name]['background'].get('vmaf_mean', 0)
            
            # Plot foreground patch
            axes[0, idx + 1].imshow(cv2.cvtColor(dec_fg_patch, cv2.COLOR_BGR2RGB))
            axes[0, idx + 1].set_title(f'{video_name}\nFG VMAF: {fg_vmaf:.1f}', 
                                       fontsize=10, fontweight='bold')
            axes[0, idx + 1].axis('off')
            
            # Plot background patch 1
            axes[1, idx + 1].imshow(cv2.cvtColor(dec_bg1_patch, cv2.COLOR_BGR2RGB))
            axes[1, idx + 1].set_title(f'{video_name}\nBG VMAF: {bg_vmaf:.1f}', 
                                       fontsize=10, fontweight='bold')
            axes[1, idx + 1].axis('off')
            
            # Plot background patch 2
            axes[2, idx + 1].imshow(cv2.cvtColor(dec_bg2_patch, cv2.COLOR_BGR2RGB))
            axes[2, idx + 1].set_title(f'{video_name}\nBG VMAF: {bg_vmaf:.1f}', 
                                       fontsize=10, fontweight='bold')
            axes[2, idx + 1].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(heatmaps_dir, f'2_patch_comparison_frame_{sample_frame_idx + 1}.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Generated patch comparison visualization")
    
    # Generate patch comparison for middle frame
    _generate_patch_comparison(
        reference_frames=reference_frames,
        encoded_videos=encoded_videos,
        results=analysis_results,
        masks_dir=masks_dir,
        heatmaps_dir=heatmaps_dir,
        decoded_frames_root=frames_root,
        sample_frame_idx=len(reference_frames) // 2,
        patch_size=128
    )

    # --- Finalization ---
    _print_summary_report(analysis_results)
    
    # Cleanup - clean up decoded frame directories
    for video_name in encoded_videos.keys():
        video_decoded_dir = os.path.join(frames_root, f"{video_name.replace(' ', '_')}_decoded")
        shutil.rmtree(video_decoded_dir, ignore_errors=True)
    shutil.rmtree(reference_frames_dir, ignore_errors=True)
    print(f"\nAnalysis complete. Masked videos saved to: {masked_videos_dir}")
    print(f"Quality visualizations saved to: {heatmaps_dir}")
    
    return analysis_results



if __name__ == "__main__":
    # Suppress warnings for cleaner output
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

    # Move to script directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # Example usage parameters
    reference_video = "davis_test/bear.mp4"
    width, height = 1280, 720  # Target resolution
    block_size = 16
    shrink_amount = 0.25      # Number of blocks to remove per row in ELVIS v1

    # Dictionary to store execution times
    execution_times = {}

    # Calculate appropriate target bitrate based on video characteristics
    cap = cv2.VideoCapture(reference_video)
    framerate = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    target_bitrate = calculate_target_bitrate(width, height, framerate, quality_factor=1.2)



    #########################################################################################################
    ############################################## SERVER SIDE ##############################################
    #########################################################################################################



    # --- Video Preprocessing ---
    start = time.time()

    # Create experiment directory for all experiment-related files
    experiment_dir = "experiment"
    os.makedirs(experiment_dir, exist_ok=True)

    print(f"Processing video: {reference_video}")
    print(f"Target resolution: {width}x{height}")
    print(f"Calculated target bitrate: {target_bitrate} bps ({target_bitrate/1000000:.1f} Mbps) for {width}x{height}@{framerate:.1f}fps")

    frames_dir = os.path.join(experiment_dir, "frames")
    reference_frames_dir = os.path.join(frames_dir, "reference")
    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(reference_frames_dir, exist_ok=True)

    print("Converting video to raw YUV format...")
    raw_video_path = os.path.join(experiment_dir, "reference_raw.yuv")
    subprocess.run(f"ffmpeg -hide_banner -loglevel error -y -i {reference_video} -vf scale={width}:{height} -c:v rawvideo -pix_fmt yuv420p {raw_video_path}", shell=True, check=False)

    print("Extracting reference frames...")
    subprocess.run(f"ffmpeg -hide_banner -loglevel error -y -video_size {width}x{height} -r {framerate} -pixel_format yuv420p -i {raw_video_path} -q:v 2 {reference_frames_dir}/%05d.png", shell=True, check=False)
    
    # Cache sorted frame list to avoid multiple listdir calls
    frame_files = sorted([f for f in os.listdir(reference_frames_dir) if f.endswith('.png')])
    reference_frames = [cv2.imread(os.path.join(reference_frames_dir, f)) for f in frame_files]

    end = time.time()
    execution_times["preprocessing"] = end - start
    print(f"Video preprocessing completed in {end - start:.2f} seconds.\n")



    # --- Removability Score Calculation ---
    start = time.time()

    print(f"Calculating removability scores with block size: {block_size}x{block_size}")
    removability_scores = calculate_removability_scores(
        raw_video_file=raw_video_path,
        reference_frames_folder=reference_frames_dir,
        width=width,
        height=height,
        block_size=block_size,
        alpha=0.5,
        working_dir=experiment_dir,
        smoothing_beta=0.5
    )
    
    end = time.time()
    execution_times["removability_score_calculation"] = end - start
    print(f"Removability scores calculation completed in {end - start:.2f} seconds.\n")



    # --- Baseline Encoding ---
    start = time.time()
    
    print(f"Encoding reference frames with two-pass for baseline comparison...")
    baseline_video = os.path.join(experiment_dir, "baseline.mp4")
    # Encode the baseline video
    encode_video(
        input_frames_dir=reference_frames_dir,
        output_video=baseline_video,
        framerate=framerate,
        width=width,
        height=height,
        target_bitrate=target_bitrate
    )
    end = time.time()
    execution_times["baseline_encoding"] = end - start
    print(f"Baseline encoding completed in {end - start:.2f} seconds.\n")



    # --- ELVIS v1 (shrinking and extracting metadata) ---
    start = time.time()
    print(f"Shrinking and encoding frames with ELVIS v1...")
    # Shrink frames based on removability scores - use list comprehension for speed
    shrunk_frames_dir = os.path.join(experiment_dir, "frames", "shrunk")
    os.makedirs(shrunk_frames_dir, exist_ok=True)
    
    # Process frames in parallel for significant speedup
    shrunk_frames, removal_masks, block_coords_to_remove = zip(*(apply_selective_removal(img, scores, block_size, shrink_amount=shrink_amount) for img, scores in zip(reference_frames, removability_scores)))
    
    # Serial version (kept for compatibility, comment out parallel version above to use this)
    shrunk_frames, removal_masks, block_coords_to_remove = zip(*[
        apply_selective_removal(img, scores, block_size, shrink_amount=shrink_amount) 
        for img, scores in zip(reference_frames, removability_scores)
    ])
    for i, frame in enumerate(shrunk_frames):
        cv2.imwrite(os.path.join(shrunk_frames_dir, f"{i+1:05d}.png"), frame)
    # Encode the shrunk frames (use actual shrunk frame dimensions, not original)
    shrunk_video = os.path.join(experiment_dir, "shrunk.mp4")
    shrunk_width = shrunk_frames[0].shape[1]  # Width is reduced due to removed blocks
    encode_video(
        input_frames_dir=shrunk_frames_dir,
        output_video=shrunk_video,
        framerate=framerate,
        width=shrunk_width,
        height=height,
        target_bitrate=target_bitrate
    )
    # Save compressed metadata about removed blocks
    removal_masks = np.array(removal_masks, dtype=np.uint8)
    masks_packed = np.packbits(removal_masks)
    np.savez(
        os.path.join(experiment_dir, 
        f"shrink_masks_{block_size}.npz"), 
        packed=masks_packed, shape=removal_masks.shape
    )
    end = time.time()
    execution_times["elvis_v1_shrinking"] = end - start
    print(f"ELVIS v1 shrinking completed in {end - start:.2f} seconds.\n")



    # --- Adaptive ROI Encoding ---
    start = time.time()
    print(f"Encoding frames with ROI-based adaptive quantization...")
    adaptive_video = os.path.join(experiment_dir, "adaptive.mp4")
    maps_dir = os.path.join(experiment_dir, "maps")
    qp_maps_dir = os.path.join(maps_dir, "qp_maps")
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
        qp_maps_dir=qp_maps_dir
    )
    end = time.time()
    execution_times["adaptive_encoding"] = end - start
    print(f"Adaptive encoding completed in {end - start:.2f} seconds.\n")



    # --- Downsampling-based ELVIS v2 (adaptive filtering and encoding) ---
    start = time.time()
    print(f"Applying downsampling-based ELVIS v2 adaptive filtering and encoding...")
    downsampled_frames_dir = os.path.join(experiment_dir, "frames", "downsampled")
    os.makedirs(downsampled_frames_dir, exist_ok=True)
    # Downsample frames based on removability scores - optimized list comprehension
    downsampled_frames, downsample_maps = zip(*[
        filter_frame_downsample(img, scores, block_size) 
        for img, scores in zip(reference_frames, removability_scores)
    ])
    for i, frame in enumerate(downsampled_frames):
        cv2.imwrite(os.path.join(downsampled_frames_dir, f"{i+1:05d}.png"), frame)
    downsampled_video = os.path.join(experiment_dir, "downsampled_encoded.mp4")
    encode_video(
        input_frames_dir=downsampled_frames_dir,
        output_video=downsampled_video,
        framerate=framerate,
        width=width,
        height=height,
        target_bitrate=target_bitrate
    )
    # Encode downsampling maps as video for client-side adaptive restoration
    downsample_maps_video = os.path.join(maps_dir, "downsample_encoded.mp4")
    encode_strength_maps(
        strength_maps=list(downsample_maps),
        output_video=downsample_maps_video,
        framerate=framerate
    )
    end = time.time()
    execution_times["elvis_v2_downsampling"] = end - start
    print(f"Downsampling-based ELVIS v2 filtering and encoding completed in {end - start:.2f} seconds.\n")



    # --- Gaussian blur Filtering-based ELVIS v2 (adaptive filtering and encoding) ---
    start = time.time()
    print(f"Applying Gaussian blur Filtering-based ELVIS v2 adaptive filtering and encoding...")
    gaussian_frames_dir = os.path.join(experiment_dir, "frames", "gaussian")
    os.makedirs(gaussian_frames_dir, exist_ok=True)
    # Apply Gaussian blur filtering to frames based on removability scores - optimized list comprehension
    gaussian_frames, gaussian_maps = zip(*[
        filter_frame_gaussian(img, scores, block_size) 
        for img, scores in zip(reference_frames, removability_scores)
    ])
    for i, frame in enumerate(gaussian_frames):
        cv2.imwrite(os.path.join(gaussian_frames_dir, f"{i+1:05d}.png"), frame)
    gaussian_video = os.path.join(experiment_dir, "gaussian_encoded.mp4")
    encode_video(
        input_frames_dir=gaussian_frames_dir,
        output_video=gaussian_video,
        framerate=framerate,
        width=width,
        height=height,
        target_bitrate=target_bitrate
    )
    # Encode gaussian strength maps as video for client-side adaptive restoration
    gaussian_maps_video = os.path.join(maps_dir, "gaussian_encoded.mp4")
    encode_strength_maps(
        strength_maps=list(gaussian_maps),
        output_video=gaussian_maps_video,
        framerate=framerate
    )
    end = time.time()
    execution_times["elvis_v2_gaussian"] = end - start
    print(f"Gaussian blur filtering-based ELVIS v2 filtering and encoding completed in {end - start:.2f} seconds.\n")



    #########################################################################################################
    ############################################## CLIENT SIDE ##############################################
    #########################################################################################################



    # --- ELVIS v1 (stretching and inpainting) ---
    start = time.time()
    print(f"Decoding and stretching ELVIS v1 video...")
    # Decode the shrunk video to frames
    removal_masks = np.load(os.path.join(experiment_dir, f"shrink_masks_{block_size}.npz"))
    removal_masks = np.unpackbits(removal_masks['packed'])[:np.prod(removal_masks['shape'])].reshape(removal_masks['shape'])
    stretched_frames_dir = os.path.join(experiment_dir, "frames", "stretched")
    if not decode_video(shrunk_video, stretched_frames_dir, framerate=framerate, start_number=1, quality=1):
        raise RuntimeError(f"Failed to decode shrunk video: {shrunk_video}")
    # Stretch each frame using the removal masks - read and process in single pass
    num_masks = len(removal_masks)
    stretched_frames = [
        stretch_frame(cv2.imread(os.path.join(stretched_frames_dir, f"{i+1:05d}.png")), removal_masks[i], block_size)
        for i in range(num_masks)
    ]
    for i, frame in enumerate(stretched_frames):
        cv2.imwrite(os.path.join(stretched_frames_dir, f"{i+1:05d}.png"), frame)
    # Convert removal_masks to mask images for inpainting (considering block size)
    removal_masks_dir = os.path.join(maps_dir, "removal_masks")
    os.makedirs(removal_masks_dir, exist_ok=True)
    for i, mask in enumerate(removal_masks):
        mask_img = (mask * 255).astype(np.uint8)
        mask_img = cv2.resize(mask_img, (width, height), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(os.path.join(removal_masks_dir, f"{i+1:05d}.png"), mask_img)
    end = time.time()
    execution_times["elvis_v1_stretching"] = end - start
    print(f"ELVIS v1 stretching completed in {end - start:.2f} seconds.\n")
    # Encode the stretched frames losslessly TODO: this is not needed in production
    stretched_video = os.path.join(experiment_dir, "stretched.mp4")
    encode_video(
        input_frames_dir=stretched_frames_dir,
        output_video=stretched_video,
        framerate=framerate,
        width=width,
        height=height,
        target_bitrate=None
    )

    # --- Inpainting with CV2 ---
    start = time.time()
    print(f"Inpainting stretched frames with CV2...")
    # Inpaint the stretched frames to fill in removed blocks using CV2
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
    # Encode the CV2 inpainted frames losslessly TODO: this is not needed in production
    inpainted_cv2_video = os.path.join(experiment_dir, "inpainted_cv2.mp4")
    encode_video(
        input_frames_dir=inpainted_cv2_frames_dir,
        output_video=inpainted_cv2_video,
        framerate=framerate,
        width=width,
        height=height,
        target_bitrate=None
    )

    # --- Inpainting with ProPainter ---
    start = time.time()
    print(f"Inpainting stretched frames with ProPainter...")
    # Inpaint the stretched frames to fill in removed blocks using ProPainter
    inpainted_frames_dir = os.path.join(experiment_dir, "frames", "inpainted")
    inpaint_with_propainter(
        stretched_frames_dir=stretched_frames_dir,
        removal_masks_dir=removal_masks_dir,
        output_frames_dir=inpainted_frames_dir,
        width=width,
        height=height,
        framerate=framerate
    )
    end = time.time()
    execution_times["elvis_v1_inpainting_propainter"] = end - start
    print(f"ELVIS v1 ProPainter inpainting completed in {end - start:.2f} seconds.\n")
    # Encode the ProPainter inpainted frames losslessly TODO: this is not needed in production
    inpainted_video = os.path.join(experiment_dir, "inpainted.mp4")
    encode_video(
        input_frames_dir=inpainted_frames_dir,
        output_video=inpainted_video,
        framerate=framerate,
        width=width,
        height=height,
        target_bitrate=None
    )

    # --- Inpainting with E2FGVI ---
    start = time.time()
    print(f"Inpainting stretched frames with E2FGVI...")
    # Inpaint the stretched frames to fill in removed blocks using E2FGVI
    inpainted_e2fgvi_frames_dir = os.path.join(experiment_dir, "frames", "inpainted_e2fgvi")
    inpaint_with_e2fgvi(
        stretched_frames_dir=stretched_frames_dir,
        removal_masks_dir=removal_masks_dir,
        output_frames_dir=inpainted_e2fgvi_frames_dir,
        width=width,
        height=height,
        framerate=framerate
    )
    end = time.time()
    execution_times["elvis_v1_inpainting_e2fgvi"] = end - start
    print(f"ELVIS v1 E2FGVI inpainting completed in {end - start:.2f} seconds.\n")
    # Encode the E2FGVI inpainted frames losslessly TODO: this is not needed in production
    inpainted_e2fgvi_video = os.path.join(experiment_dir, "inpainted_e2fgvi.mp4")
    encode_video(
        input_frames_dir=inpainted_e2fgvi_frames_dir,
        output_video=inpainted_e2fgvi_video,
        framerate=framerate,
        width=width,
        height=height,
        target_bitrate=None
    )



    # --- Downsampling-based ELVIS v2: Decode video and apply adaptive restoration ---
    start = time.time()
    print(f"Decoding downsampled ELVIS v2 video and strength maps...")
    # Decode the downsampled video to frames
    downsampled_frames_decoded_dir = os.path.join(experiment_dir, "frames", "downsampled_decoded")
    if not decode_video(downsampled_video, downsampled_frames_decoded_dir, framerate=framerate, start_number=1, quality=1):
        raise RuntimeError(f"Failed to decode downsampled video: {downsampled_video}")
    # Decode the downsampling strength maps video to frames
    downsampled_maps_decoded_dir = os.path.join(experiment_dir, "maps", "downsampled_maps_decoded")
    strength_maps = decode_strength_maps(downsample_maps_video, block_size, downsampled_maps_decoded_dir)
    end = time.time()
    execution_times["elvis_v2_downsampling_decoding"] = end - start
    print(f"Decoding completed in {end - start:.2f} seconds.\n")
    start = time.time()
    print(f"Applying adaptive upsampling restoration for downsampling-based ELVIS v2...")
    # Apply adaptive upsampling restoration via Real-ESRGAN
    downsampled_restored_frames_dir = os.path.join(experiment_dir, "frames", "downsampled_restored")
    os.makedirs(downsampled_restored_frames_dir, exist_ok=True)
    for i in range(len(reference_frames)):
        decoded_frame = cv2.imread(os.path.join(downsampled_frames_decoded_dir, f"{i+1:05d}.png"))
        downscale_map = strength_maps[i]
        restored_frame = upscale_realesrgan_adaptive(decoded_frame, downscale_map, block_size)
        cv2.imwrite(os.path.join(downsampled_restored_frames_dir, f"{i+1:05d}.png"), restored_frame)
    end = time.time()
    execution_times["elvis_v2_downsampling_restoration"] = end - start
    print(f"Adaptive upsampling restoration completed in {end - start:.2f} seconds.\n")
    # Encode the restored frames losslessly TODO: this is not needed in production
    downsampled_restored_video = os.path.join(experiment_dir, "downsampled_restored.mp4")
    encode_video(
        input_frames_dir=downsampled_restored_frames_dir,
        output_video=downsampled_restored_video,
        framerate=framerate,
        width=width,
        height=height,
        target_bitrate=None
    )



    # --- Gaussian blur Filtering-based ELVIS v2: Decode video and apply adaptive restoration ---
    start = time.time()
    print(f"Decoding Gaussian blur filtered ELVIS v2 video and strength maps...")
    # Decode the Gaussian video to frames
    gaussian_frames_decoded_dir = os.path.join(experiment_dir, "frames", "gaussian_decoded")
    if not decode_video(gaussian_video, gaussian_frames_decoded_dir, framerate=framerate, start_number=1, quality=1):
        raise RuntimeError(f"Failed to decode Gaussian video: {gaussian_video}")
    # Decode the Gaussian strength maps video to frames
    gaussian_maps_decoded_dir = os.path.join(experiment_dir, "maps", "gaussian_maps_decoded")
    strength_maps_gaussian = decode_strength_maps(gaussian_maps_video, block_size, gaussian_maps_decoded_dir)
    end = time.time()
    execution_times["elvis_v2_gaussian_decoding"] = end - start
    print(f"Decoding completed in {end - start:.2f} seconds.\n")
    start = time.time()
    print(f"Applying adaptive deblurring restoration for Gaussian blur filtering-based ELVIS v2...")
    
    # Copy decoded frames to a subfolder structure for InstantIR
    # InstantIR expects: parent_dir/subfolder/images.png
    instantir_work_dir = os.path.join(experiment_dir, "instantir_work")
    gaussian_instantir_input_dir = os.path.join(instantir_work_dir, "gaussian_decoded")
    os.makedirs(gaussian_instantir_input_dir, exist_ok=True)
    
    # Copy all decoded frames to the InstantIR input subfolder
    print(f"  Copying {len(reference_frames)} frames to InstantIR input directory...")
    for frame_file in os.listdir(gaussian_frames_decoded_dir):
        if frame_file.endswith(('.png', '.jpg', '.jpeg')):
            shutil.copy2(
                os.path.join(gaussian_frames_decoded_dir, frame_file),
                os.path.join(gaussian_instantir_input_dir, frame_file)
            )
    
    # Apply adaptive restoration on the entire folder
    # This will modify frames in-place in gaussian_instantir_input_dir
    restore_with_instantir_adaptive(
        input_frames_dir=gaussian_instantir_input_dir,
        blur_maps=strength_maps_gaussian,
        block_size=block_size,
        instantir_weights_dir=str(Path(__file__).resolve().parent / "InstantIR" / "models"),
        cfg=7.0,
        creative_start=1.0,  # No creative restoration, preserve fidelity
        preview_start=0.0
    )
    
    # Copy restored frames to final output directory
    gaussian_restored_frames_dir = os.path.join(experiment_dir, "frames", "gaussian_restored")
    os.makedirs(gaussian_restored_frames_dir, exist_ok=True)
    print(f"  Copying restored frames to output directory...")
    for frame_file in os.listdir(gaussian_instantir_input_dir):
        if frame_file.endswith(('.png', '.jpg', '.jpeg')):
            shutil.copy2(
                os.path.join(gaussian_instantir_input_dir, frame_file),
                os.path.join(gaussian_restored_frames_dir, frame_file)
            )
    
    end = time.time()
    execution_times["elvis_v2_gaussian_restoration"] = end - start
    print(f"Adaptive deblurring restoration completed in {end - start:.2f} seconds.\n")
    # Encode the restored frames losslessly TODO: this is not needed in production
    gaussian_restored_video = os.path.join(experiment_dir, "gaussian_restored.mp4")
    encode_video(
        input_frames_dir=gaussian_restored_frames_dir,
        output_video=gaussian_restored_video,
        framerate=framerate,
        width=width,
        height=height,
        target_bitrate=None
    )



    #########################################################################################################
    ######################################### PERFORMANCE EVALUATION ########################################
    #########################################################################################################

    print("Evaluating and comparing encoding performance...")
    start = time.time()

    # Compare file sizes and quality metrics (include metadata)
    video_sizes = {
        "Baseline": os.path.getsize(baseline_video),
        "ELVIS v1": os.path.getsize(shrunk_video) + os.path.getsize(os.path.join(experiment_dir, f"shrink_masks_{block_size}.npz")),
        "Adaptive": os.path.getsize(adaptive_video),
        "ELVIS v2 Downsample": os.path.getsize(downsampled_video) + os.path.getsize(downsample_maps_video),
        "ELVIS v2 Gaussian": os.path.getsize(gaussian_video) + os.path.getsize(gaussian_maps_video)
    }

    # Use cached frame_files instead of re-listing directory
    frame_count = len(frame_files)
    duration = frame_count / framerate
    bitrates = {key: (size * 8) / duration for key, size in video_sizes.items()}

    print(f"\nEncoding Results (Target Bitrate: {target_bitrate} bps / {target_bitrate/1000000:.1f} Mbps):")
    for key, bitrate in bitrates.items():
        print(f"{key} bitrate: {bitrate / 1000000:.2f} Mbps")

    encoded_videos = {
        "Baseline": baseline_video,
        "Adaptive": adaptive_video,
        "ELVIS v1 CV2": inpainted_cv2_video,
        "ELVIS v1 ProPainter": inpainted_video,
        "ELVIS v1 E2FGVI": inpainted_e2fgvi_video,
        "ELVIS v2 Downsample": downsampled_restored_video,
        "ELVIS v2 Gaussian": gaussian_restored_video
    }

    ufo_masks_dir = os.path.join(experiment_dir, "maps", "ufo_masks")
    sample_frames = [frame_count // 2] # [0, frame_count // 4, frame_count // 2, 3 * frame_count // 4, frame_count - 1]
    sample_frames = [f for f in sample_frames if f < frame_count]

    # Prepare strength maps for OpenCV benchmarks
    strength_maps_dict = {
        "ELVIS v2 Downsample": strength_maps,
        "ELVIS v2 Gaussian": strength_maps_gaussian
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
        generate_opencv_benchmarks=True
    )
    
    # Add collected times to the results dictionary
    analysis_results["execution_times_seconds"] = execution_times

    # Add video parameters to the results dictionary
    analysis_results["video_name"] = reference_video
    analysis_results["video_length_seconds"] = frame_count / framerate
    analysis_results["video_framerate"] = framerate
    analysis_results["video_resolution"] = f"{width}x{height}"
    analysis_results["block_size"] = block_size
    analysis_results["target_bitrate_bps"] = target_bitrate
    
    # Save analysis results to a JSON file
    results_json_path = os.path.join(experiment_dir, "analysis_results.json")
    with open(results_json_path, 'w') as f:
        json.dump(analysis_results, f, indent=4)
    print(f"Analysis results saved to: {results_json_path}")