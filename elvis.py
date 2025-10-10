import time
import shutil
import os
import subprocess
from pathlib import Path
import cv2
import numpy as np
from typing import List, Callable, Tuple, Dict
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import json
import functools
import multiprocessing
import torch
import lpips
import platform
import tempfile



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
    bits_per_pixel = 0.02 * quality_factor
    
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
    evca_dir = os.path.join(working_dir, "evca")
    maps_dir = os.path.join(working_dir, "maps")
    ufo_masks_dir = os.path.join(maps_dir, "ufo_masks")
    os.makedirs(evca_dir, exist_ok=True)
    os.makedirs(ufo_masks_dir, exist_ok=True)
    
    # Get frame count
    frame_count = len(os.listdir(reference_frames_folder))
    
    # Save current directory and get absolute paths
    original_dir = os.getcwd()
    raw_video_abs = os.path.abspath(raw_video_file)
    reference_frames_abs = os.path.abspath(reference_frames_folder)
    evca_csv_path = os.path.abspath(os.path.join(evca_dir, "evca.csv"))
    ufo_masks_abs = os.path.abspath(ufo_masks_dir)
    
    try:
        # Calculate scene complexity with EVCA
        print("Running EVCA for complexity analysis...")
        # Use the internal EVCA repository within elvis
        evca_path = os.path.join(original_dir, "EVCA", "main.py")
        evca_cmd = f"python {evca_path} -i {raw_video_abs} -r {width}x{height} -b {block_size} -f {frame_count} -c {evca_csv_path} -bi 1"
        result = subprocess.run(evca_cmd, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"EVCA command failed: {result.stderr}")
            print(f"EVCA stdout: {result.stdout}")
            raise RuntimeError(f"EVCA execution failed: {result.stderr}")
        else:
            print("EVCA completed successfully")
        
        # Calculate ROI with UFO
        print("Running UFO for object detection...")
        # Use the internal UFO repository within elvis
        # UFO expects a class subdirectory structure
        UFO_class_folder = os.path.join(original_dir, "UFO", "datasets", "elvis", "image", "reference_frames")
        if os.path.exists(UFO_class_folder):
            shutil.rmtree(UFO_class_folder)
        os.makedirs(os.path.dirname(UFO_class_folder), exist_ok=True)
        shutil.copytree(reference_frames_abs, UFO_class_folder)
        
        # Change to the internal UFO directory
        ufo_dir = os.path.join(original_dir, "UFO")
        os.chdir(ufo_dir)
        ufo_cmd = "python custom_ufo_test.py --model='weights/video_best.pth' --data_path='datasets/elvis' --output_dir='VSOD_results/wo_optical_flow/elvis' --task='VSOD'"
        result = subprocess.run(ufo_cmd, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"UFO command failed: {result.stderr}")
            raise RuntimeError(f"UFO execution failed: {result.stderr}")
        
        # Move UFO masks to working directory
        mask_source = os.path.join(ufo_dir, "VSOD_results", "wo_optical_flow", "elvis", "reference_frames")
        if os.path.exists(mask_source):
            for mask_file in os.listdir(mask_source):
                src_path = os.path.join(mask_source, mask_file)
                dst_path = os.path.join(ufo_masks_abs, mask_file)
                shutil.move(src_path, dst_path)
        
        # Return to original directory
        os.chdir(original_dir)
        
        # Load the CSV files from EVCA
        temporal_csv_path = os.path.join(evca_dir, "evca_TC_blocks.csv")
        spatial_csv_path = os.path.join(evca_dir, "evca_SC_blocks.csv")
        
        temporal_array = np.loadtxt(temporal_csv_path, delimiter=',', skiprows=1)
        spatial_array = np.loadtxt(spatial_csv_path, delimiter=',', skiprows=1)

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

            # For all subsequent frames, apply the smoothing formula.
            # This is a vectorized operation, which is very fast.
            smoothed_scores[1:] = smoothing_beta * removability_scores[1:] + (1 - smoothing_beta) * removability_scores[:-1]
            
            removability_scores = smoothed_scores

        # Final normalization to [0, 1]
        removability_scores = normalize_array(removability_scores)

        return removability_scores
    
    except Exception as e:
        print(f"Error in calculate_removability_scores: {e}")
        raise
    finally:
        # Always return to original directory
        os.chdir(original_dir)

def map_score_to_qp(score: float, min_qp: int = 1, max_qp: int = 50) -> int:
    """Maps a normalized score [0, 1] to a QP value.
    
    A low score (important region) maps to a low QP (high quality).
    A high score (removable region) maps to a high QP (low quality).
    The QP range is chosen to provide a good quality spread.
    """
    return int(min_qp + (score * (max_qp - min_qp)))

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
        **extra_params: Additional x265 parameters to append (e.g., qpfile path).
    """
    temp_dir = os.path.dirname(output_video) or '.'
    os.makedirs(temp_dir, exist_ok=True)
    
    passlog_file = os.path.join(temp_dir, f"ffmpeg_2pass_log_{os.path.basename(output_video)}")
    
    # Use platform-specific null device for the first pass output
    null_device = "NUL" if platform.system() == "Windows" else "/dev/null"
    
    try:
        # Base command shared by both passes

        base_cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-framerate", str(framerate),
            "-i", f"{input_frames_dir}/%05d.png",
            "-vf", f"scale={width}:{height}:flags=lanczos,format={pix_fmt}",
        ]

        if target_bitrate is None:
            # Lossless encoding with two passes
            preset = "veryslow"
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
        # Clean up pass log files
        for f in os.listdir(temp_dir):
            if f.startswith(f"ffmpeg_2pass_log_{os.path.basename(output_video)}"):
                try:
                    os.remove(os.path.join(temp_dir, f))
                except:
                    pass

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

        # Create QP map array for visualization
        qp_maps = np.zeros((num_frames, num_blocks_y, num_blocks_x), dtype=np.uint8)
        
        with open(qpfile_path, 'w') as f:
            for frame_idx in range(num_frames):
                # Start the line with frame index and a generic frame type ('P')
                # The '-1' for QP indicates we are providing per-block QPs
                line_parts = [f"{frame_idx} P -1"]
                
                # Append QP for each block in raster-scan order (left-to-right, top-to-bottom)
                for y in range(num_blocks_y):
                    for x in range(num_blocks_x):
                        score = removability_scores[frame_idx, y, x]
                        qp = map_score_to_qp(score)
                        line_parts.append(str(qp))
                        qp_maps[frame_idx, y, x] = qp
                
                f.write(" ".join(line_parts) + "\n")
        print(f"qpfile generated at {qpfile_path}")
        
        # --- Save QP maps as images if requested ---
        if save_qp_maps:
            if qp_maps_dir is None:
                qp_maps_dir = os.path.join(temp_dir, "qp_maps")
            os.makedirs(qp_maps_dir, exist_ok=True)
            
            print(f"Saving QP maps to {qp_maps_dir}...")
            for frame_idx in range(num_frames):
                # Create a full-resolution image by upscaling the block-level QP map
                qp_map_blocks = qp_maps[frame_idx]
                
                # Scale QP values to grayscale range [0, 255] for visualization
                # Lower QP (high quality) = darker, Higher QP (low quality) = brighter
                qp_map_normalized = (qp_map_blocks / 50.0 * 255).astype(np.uint8)
                
                # Upscale each block to full resolution using nearest neighbor
                qp_map_fullres = np.kron(qp_map_normalized, np.ones((block_size, block_size), dtype=np.uint8))
                
                # Ensure the dimensions match exactly
                qp_map_fullres = qp_map_fullres[:height, :width]
                
                # Save as grayscale image
                qp_map_path = os.path.join(qp_maps_dir, f"qp_map_{frame_idx+1:05d}.png")
                cv2.imwrite(qp_map_path, qp_map_fullres)
            
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
        # FFmpeg might create multiple log files (e.g., .log, .log.mbtree)
        for f in os.listdir(temp_dir):
            if f.startswith(os.path.basename(passlog_file)):
                 os.remove(os.path.join(temp_dir, f))

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

def inpaint_with_propainter(stretched_frames_dir: str, removal_masks_dir: str, output_frames_dir: str, width: int, height: int, framerate: float, propainter_dir: str = "ProPainter", resize_ratio: float = 1.0, ref_stride: int = 10, neighbor_length: int = 10, subvideo_length: int = 80, mask_dilation: int = 4, raft_iter: int = 20, fp16: bool = True) -> None:
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
    propainter_abs = os.path.abspath(propainter_dir)
    
    try:
        # Create temporary directory for ProPainter inputs/outputs
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create ProPainter input directories
            propainter_input_dir = os.path.join(propainter_abs, "inputs", "elvis_temp")
            propainter_output_dir = os.path.join(propainter_abs, "results", "elvis_temp")
            os.makedirs(propainter_input_dir, exist_ok=True)

            # Copy stretched frames to ProPainter input directory
            frame_files = sorted([f for f in os.listdir(stretched_frames_abs) if f.endswith(('.jpg', '.png'))])
            temp_frame_dir = os.path.join(propainter_input_dir, "frames")
            os.makedirs(temp_frame_dir, exist_ok=True)
            for i, frame_file in enumerate(frame_files):
                src_frame = os.path.join(stretched_frames_abs, frame_file)
                dst_frame = os.path.join(temp_frame_dir, f"{i+1:05d}.png")
                shutil.copy(src_frame, dst_frame)
            
            # Copy masks to ProPainter input directory
            temp_mask_dir = os.path.join(propainter_input_dir, "masks")
            os.makedirs(temp_mask_dir, exist_ok=True)
            mask_files = sorted([f for f in os.listdir(removal_masks_abs) if f.endswith(('.jpg', '.png'))])
            for i, mask_file in enumerate(mask_files):
                src_mask = os.path.join(removal_masks_abs, mask_file)
                dst_mask = os.path.join(temp_mask_dir, f"{i+1:05d}.png")
                shutil.copy(src_mask, dst_mask)
            
            # Change to ProPainter directory
            os.chdir(propainter_abs)
            
            # Build ProPainter command
            propainter_cmd = [
                "python", "inference_propainter.py",
                "--video", temp_frame_dir,
                "--mask", temp_mask_dir,
                "--output", propainter_output_dir,
                "--width", str(width),
                "--height", str(height),
                "--resize_ratio", str(resize_ratio),
                "--ref_stride", str(ref_stride),
                "--neighbor_length", str(neighbor_length),
                "--subvideo_length", str(subvideo_length),
                "--mask_dilation", str(mask_dilation),
                "--raft_iter", str(raft_iter),
                "--save_fps", str(int(framerate)),
                "--save_frames"  # Save individual frames
            ]
            
            if fp16:
                propainter_cmd.append("--fp16")
            
            result = subprocess.run(propainter_cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"ProPainter stdout: {result.stdout}")
                print(f"ProPainter stderr: {result.stderr}")
                raise RuntimeError(f"ProPainter inference failed: {result.stderr}")
            
            print(f"ProPainter output: {result.stdout}")
            
            # Extract inpainted frames from ProPainter output
            propainter_frames_dir = os.path.join(propainter_output_dir, "frames/frames")
            if not os.path.exists(propainter_frames_dir):
                raise RuntimeError(f"ProPainter did not generate frames at {propainter_frames_dir}")
            
            # Copy inpainted frames to output directory
            os.makedirs(output_frames_abs, exist_ok=True)
            inpainted_files = sorted([f for f in os.listdir(propainter_frames_dir) if f.endswith('.png')])
            for i, inpainted_file in enumerate(inpainted_files):
                src_inpainted = os.path.join(propainter_frames_dir, inpainted_file)
                dst_inpainted = os.path.join(output_frames_abs, f"{i+1:05d}.png")
                shutil.copy(src_inpainted, dst_inpainted)
            
            print(f"Inpainted frames saved to {output_frames_abs}")
            
            # Clean up ProPainter temp directories
            if os.path.exists(propainter_input_dir):
                shutil.rmtree(propainter_input_dir)
            if os.path.exists(propainter_output_dir):
                shutil.rmtree(propainter_output_dir)
    
    except Exception as e:
        print(f"Error in inpaint_with_propainter: {e}")
        raise
    finally:
        # Always return to original directory
        os.chdir(original_dir)

def inpaint_with_e2fgvi(stretched_frames_dir: str, removal_masks_dir: str, output_frames_dir: str, width: int, height: int, framerate: float, e2fgvi_dir: str = "E2FGVI", model: str = "e2fgvi_hq", ckpt: str = None, ref_stride: int = 10, neighbor_stride: int = 5, num_ref: int = -1, mask_dilation: int = 4) -> None:
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
    
    # Save current directory
    original_dir = os.getcwd()
    
    # Get absolute paths
    stretched_frames_abs = os.path.abspath(stretched_frames_dir)
    removal_masks_abs = os.path.abspath(removal_masks_dir)
    output_frames_abs = os.path.abspath(output_frames_dir)
    e2fgvi_abs = os.path.abspath(e2fgvi_dir)
    
    # Set default checkpoint if not provided
    if ckpt is None:
        ckpt = os.path.join(e2fgvi_abs, "release_model", "E2FGVI-HQ-CVPR22.pth")
    else:
        ckpt = os.path.abspath(ckpt)
    
    try:
        # Change to E2FGVI directory (required for imports)
        os.chdir(e2fgvi_abs)
        
        # Build E2FGVI test.py command
        e2fgvi_cmd = [
            "python", "test.py",
            "--model", model,
            "--video", stretched_frames_abs,
            "--mask", removal_masks_abs,
            "--ckpt", ckpt,
            "--step", str(ref_stride),
            "--num_ref", str(num_ref),
            "--neighbor_stride", str(neighbor_stride),
            "--set_size",
            "--width", str(width),
            "--height", str(height),
            "--savefps", str(int(framerate))
        ]
        
        print(f"Running E2FGVI inference...")
        result = subprocess.run(e2fgvi_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"E2FGVI stdout: {result.stdout}")
            print(f"E2FGVI stderr: {result.stderr}")
            raise RuntimeError(f"E2FGVI inference failed: {result.stderr}")
        
        print(f"E2FGVI output: {result.stdout}")
        
        # E2FGVI saves results in its results/ directory
        # Find the output video (it will be named based on the input)
        results_dir = os.path.join(e2fgvi_abs, "results")
        if not os.path.exists(results_dir):
            raise RuntimeError(f"E2FGVI results directory not found at {results_dir}")
        
        # Find the most recent result video
        result_videos = [f for f in os.listdir(results_dir) if f.endswith('.mp4')]
        if not result_videos:
            raise RuntimeError(f"No result video found in {results_dir}")
        
        # Use the most recently modified video
        result_videos.sort(key=lambda x: os.path.getmtime(os.path.join(results_dir, x)), reverse=True)
        result_video_path = os.path.join(results_dir, result_videos[0])
        
        # Decode the result video to frames
        print(f"Decoding E2FGVI result video to frames...")
        os.makedirs(output_frames_abs, exist_ok=True)
        
        if not decode_video(result_video_path, output_frames_abs, framerate=framerate, start_number=1, quality=1):
            raise RuntimeError(f"Failed to decode E2FGVI result video: {result_video_path}")
        
        print(f"E2FGVI inpainted frames saved to {output_frames_abs}")
        
        # Clean up the result video
        os.remove(result_video_path)
    
    except Exception as e:
        print(f"Error in inpaint_with_e2fgvi: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        # Always return to original directory
        os.chdir(original_dir)

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

def map_scores_to_strengths(normalized_scores: np.ndarray, max_value: float, mapping_type: str = 'linear', dtype: type = np.float32) -> np.ndarray:
    """
    Maps normalized scores [0, 1] to strength values using different mapping strategies.
    
    This unified function supports multiple mapping types:
    - 'linear': Simple linear mapping to [0, max_value]
    - 'power_of_2': Maps to discrete power-of-2 values [1, 2, 4, ..., max_value]
    
    Args:
        normalized_scores: 3D array of shape (num_frames, num_blocks_y, num_blocks_x)
        max_value: Maximum strength/factor value
        mapping_type: Type of mapping ('linear' or 'power_of_2')
        dtype: Data type for the output array (e.g., np.float32, np.int32)
    
    Returns:
        3D array of strength values with the specified dtype
    """
    if mapping_type == 'linear':
        # Simple linear mapping: score * max_value
        strength_maps = (normalized_scores * max_value).astype(dtype)
        
    elif mapping_type == 'power_of_2':
        # Validate that max_value is a power of 2
        if max_value < 1 or (max_value & (max_value - 1) != 0):
            raise ValueError(f"max_value must be a power of 2 for 'power_of_2' mapping, got {max_value}")
        
        # Define the possible power-of-2 factors (e.g., [1, 2, 4, 8])
        levels = int(np.log2(max_value)) + 1
        pow2_factors = np.logspace(0, levels - 1, levels, base=2, dtype=np.int32)
        
        # Map normalized scores to indices of these factors
        indices = np.floor(normalized_scores * (levels - 1)).astype(np.int32)
        
        # Use the indices to look up the corresponding factor
        strength_maps = pow2_factors[indices].astype(dtype)
        
    else:
        raise ValueError(f"Unknown mapping_type: {mapping_type}. Use 'linear' or 'power_of_2'.")
    
    return strength_maps

def encode_strength_maps_to_video(strength_maps: np.ndarray, output_video: str, framerate: float, target_bitrate: int = 50000) -> None:
    """
    Encodes strength maps as a grayscale video for compression.
    
    This allows leveraging video codecs to compress the strength maps efficiently.
    The maps are kept at block resolution (one pixel per block) and encoded as grayscale.
    Min/max values are encoded in the filename for later denormalization.
    
    Args:
        strength_maps: 3D array of shape (num_frames, num_blocks_y, num_blocks_x)
        output_video: Path to output video file (should be in maps directory)
        framerate: Video framerate
        target_bitrate: Target bitrate for lossy compression (default: 50kbps, sufficient for small maps)
    """
    temp_maps_dir = tempfile.mkdtemp()
    try:
        # Find min/max for normalization
        min_val = float(strength_maps.min())
        max_val = float(strength_maps.max())
        
        # Encode min/max in filename: base_name_min{min_val}_max{max_val}.mp4
        base_name = output_video.replace('.mp4', '')
        output_video_with_params = f"{base_name}_min{min_val:.6f}_max{max_val:.6f}.mp4"
        
        # Normalize to 0-255 and save as grayscale images at block resolution
        for i, strengths in enumerate(strength_maps):
            # Normalize to 0-255
            if max_val > min_val:
                normalized = ((strengths - min_val) / (max_val - min_val) * 255).astype(np.uint8)
            else:
                normalized = np.zeros_like(strengths, dtype=np.uint8)
            
            # Save at block resolution (one pixel per block)
            cv2.imwrite(os.path.join(temp_maps_dir, f"{i+1:05d}.png"), normalized)
        
        # Encode as grayscale video with Y-only format
        encode_video(
            input_frames_dir=temp_maps_dir,
            output_video=output_video_with_params,
            framerate=framerate,
            width=strength_maps.shape[2],
            height=strength_maps.shape[1],
            target_bitrate=target_bitrate,
            pix_fmt='gray'  # Grayscale pixel format
        )
        
        print(f"  - Encoded strength maps to {output_video_with_params} (min={min_val:.6f}, max={max_val:.6f})")
        
    finally:
        shutil.rmtree(temp_maps_dir, ignore_errors=True)

def decode_strength_maps_from_video(video_path: str, dtype: type = np.float32) -> np.ndarray:
    """
    Decodes strength maps from a compressed video.
    Min/max values are decoded from the filename.
    
    Args:
        video_path: Path to encoded strength maps video (with min/max in filename)
                   Format: base_name_min{min_val}_max{max_val}.mp4
        dtype: Data type to restore (e.g., np.float32, np.int32)
    
    Returns:
        3D array of strength values with shape (num_frames, num_blocks_y, num_blocks_x)
    """
    # Find the actual video file with min/max parameters in filename
    # The video_path passed in may not have the params, so we need to find it
    base_path = video_path.replace('.mp4', '')
    import glob
    matching_files = glob.glob(f"{base_path}_min*_max*.mp4")
    
    if not matching_files:
        raise FileNotFoundError(f"No strength maps video found matching pattern: {base_path}_min*_max*.mp4")
    
    actual_video_path = matching_files[0]
    
    # Extract min/max from filename
    import re
    match = re.search(r'_min([-+]?[0-9]*\.?[0-9]+)_max([-+]?[0-9]*\.?[0-9]+)\.mp4$', actual_video_path)
    if not match:
        raise ValueError(f"Could not extract min/max values from filename: {actual_video_path}")
    
    min_val = float(match.group(1))
    max_val = float(match.group(2))
    
    # Decode video to frames (already at block resolution)
    temp_frames_dir = tempfile.mkdtemp()
    try:
        if not decode_video(actual_video_path, temp_frames_dir, quality=1):
            raise RuntimeError(f"Failed to decode strength maps video: {actual_video_path}")
        
        # Load frames and denormalize
        frame_files = sorted([f for f in os.listdir(temp_frames_dir) if f.endswith('.png')])
        strength_maps = []
        
        for frame_file in frame_files:
            # Load grayscale map (already at block resolution - one pixel per block)
            map_img = cv2.imread(os.path.join(temp_frames_dir, frame_file), cv2.IMREAD_GRAYSCALE)
            
            # Denormalize from 0-255 back to original range
            if max_val > min_val:
                denormalized = (map_img.astype(np.float32) / 255.0) * (max_val - min_val) + min_val
            else:
                denormalized = np.full_like(map_img, min_val, dtype=np.float32)
            
            strength_maps.append(denormalized.astype(dtype))
        
        strength_maps = np.array(strength_maps)
        print(f"  - Decoded strength maps from {actual_video_path} (shape: {strength_maps.shape}, min={min_val:.6f}, max={max_val:.6f})")
        
        return strength_maps
        
    finally:
        shutil.rmtree(temp_frames_dir, ignore_errors=True)

def apply_adaptive_deblocking(image: np.ndarray, strengths: np.ndarray, block_size: int, filter_func: Callable[[np.ndarray, float], np.ndarray], min_strength_threshold: float = 0.1) -> np.ndarray:
    """
    Applies adaptive restoration/deblocking to an image based on strength maps.
    
    This is similar to apply_adaptive_filtering but designed for client-side restoration.
    It applies restoration filters with varying strength based on the decoded strength maps.
    
    Args:
        image: The degraded image to restore (H, W, C)
        strengths: 2D array of restoration strengths for each block (num_blocks_y, num_blocks_x)
        block_size: The side length of each block
        filter_func: A restoration function that takes a block and strength and returns restored block
        min_strength_threshold: Minimum strength threshold below which no restoration is applied
    
    Returns:
        The restored image
    """
    # Use the same logic as apply_adaptive_filtering
    return apply_adaptive_filtering(image, strengths, block_size, filter_func, min_strength_threshold)

def apply_bilateral_deblocking(block: np.ndarray, strength: float) -> np.ndarray:
    """
    Applies bilateral filtering for deblocking. Strength controls the filter parameters.
    Higher strength means more aggressive filtering.
    """
    if strength < 0.1:
        return block
    
    # Map strength to bilateral filter parameters
    d = max(5, min(15, int(strength * 3)))  # Diameter: 5-15
    sigma_color = max(10, min(150, strength * 30))  # Color sigma: 10-150
    sigma_space = max(10, min(150, strength * 30))  # Space sigma: 10-150
    
    return cv2.bilateralFilter(block, d=d, sigmaColor=sigma_color, sigmaSpace=sigma_space)

def apply_unsharp_mask(block: np.ndarray, strength: float) -> np.ndarray:
    """
    Applies unsharp masking for deblurring. Strength controls the sharpening amount.
    """
    if strength < 0.1:
        return block
    
    # Map strength to unsharp mask parameters
    sigma = max(1.0, min(10.0, strength * 2))
    amount = max(0.5, min(2.0, strength * 0.5))
    
    gaussian = cv2.GaussianBlur(block, (0, 0), sigma)
    sharpened = cv2.addWeighted(block, 1.0 + amount, gaussian, -amount, 0)
    
    return sharpened

def apply_super_resolution(block: np.ndarray, strength: float) -> np.ndarray:
    """
    Applies simple super-resolution (Lanczos upscaling). Strength controls the method.
    For now, this is a placeholder that uses Lanczos interpolation.
    In practice, you'd use a SOTA SR model with strength controlling the model complexity.
    """
    # For downsampling restoration, we don't need per-block SR since the image is already full size
    # This is a no-op placeholder that could be replaced with edge enhancement or sharpening
    if strength < 1.0:
        return block
    
    # Apply slight sharpening based on strength
    kernel = np.array([[-1,-1,-1], [-1, 9+strength,-1], [-1,-1,-1]]) / (strength + 1)
    sharpened = cv2.filter2D(block, -1, kernel)
    
    return sharpened

def apply_gaussian_blur(block: np.ndarray, strength: float) -> np.ndarray:
    """
    Applies a Gaussian blur. Strength controls the sigma and kernel size.
    """
    # Ensure strength is at least a small positive number
    sigma = max(strength, 0.1)
    
    # Kernel size should be an odd number and proportional to sigma
    ksize = int(sigma * 4) # Rule of thumb: kernel size around 4-6x sigma
    ksize = ksize + 1 if ksize % 2 == 0 else ksize
    ksize = max(ksize, 1) # Must be at least 1
    
    return cv2.GaussianBlur(block, (ksize, ksize), sigma)

def apply_downsampling(block: np.ndarray, strength: float) -> np.ndarray:
    """
    Simplifies a block by downsampling and then upsampling it.
    Strength controls the downsampling factor.
    """
    height, width, _ = block.shape
    
    # Map strength [0-inf] to a downscale factor [1x - 8x]
    # Using a non-linear mapping like log can give better perceptual results
    scale_factor = 1 + int(strength)
    
    if scale_factor <= 1:
        return block
        
    # Calculate new dimensions, ensuring they are at least 1 pixel
    new_height, new_width = max(1, height // scale_factor), max(1, width // scale_factor)
    
    # Downsample using an averaging interpolator
    downsampled = cv2.resize(block, (new_width, new_height), interpolation=cv2.INTER_AREA)
    
    # Upsample back to the original size
    upsampled = cv2.resize(downsampled, (width, height), interpolation=cv2.INTER_LINEAR)
    
    return upsampled

def apply_dct_damping(block: np.ndarray, strength: float) -> np.ndarray:
    """
    Simplifies a block by damping its high-frequency DCT coefficients.
    A higher strength zeros out more coefficients, simplifying the block more.
    
    Args:
        block: The input block (l, l, C), likely in uint8 format.
        strength: A float, typically in a range like [0, 64], indicating how many
                  high-frequency components to cut.
    """
    if strength < 1:
        return block

    block_size = block.shape[0]
    
    # DCT works on float32, single-channel images. We process each channel.
    block_float = block.astype(np.float32)
    processed_channels = []
    
    for channel in cv2.split(block_float):
        # 1. Apply forward DCT
        dct_channel = cv2.dct(channel)
        
        # 2. Determine the cutoff
        # A higher strength means a smaller cutoff, keeping fewer coefficients.
        # We ensure we always keep at least the DC component (top-left).
        cutoff = max(1, int(block_size - strength))
        
        # 3. Create a mask and apply it
        # This zeros out a bottom-right block of high-frequency coefficients
        mask = np.zeros((block_size, block_size), dtype=np.float32)
        mask[:cutoff, :cutoff] = 1.0
        dct_channel *= mask
        
        # 4. Apply inverse DCT
        idct_channel = cv2.idct(dct_channel)
        processed_channels.append(idct_channel)
        
    # Merge channels, clip to valid range, and convert back to uint8
    merged = cv2.merge(processed_channels)
    final_block = np.clip(merged, 0, 255).astype(np.uint8)
    
    return final_block

def apply_adaptive_filtering(image: np.ndarray, strengths: np.ndarray, block_size: int, filter_func: Callable[[np.ndarray, float], np.ndarray], min_strength_threshold: float = 0.1) -> np.ndarray:
    """
    Applies a variable-strength filter to each block of an image based on 
    a pre-calculated 2D array of filter strengths.

    Args:
        image: The original image for the frame (H, W, C).
        strengths: The 2D array of filter strengths for each block 
                   (num_blocks_y, num_blocks_x).
        block_size: The side length (l) of each block.
        filter_func: A function that takes a block (l, l, C) and a strength 
                     (float or int) and returns a filtered block.
        min_strength_threshold: A value below which the filter is not applied.

    Returns:
        The new image with adaptive filtering applied to its blocks.
    """
    
    # Split the image into an array of blocks
    blocks = split_image_into_blocks(image, block_size)
    # Create a copy to store the filtered results
    filtered_blocks = blocks.copy()
    
    num_blocks_y, num_blocks_x, _, _, _ = blocks.shape

    # Iterate over each block to apply the corresponding filter
    for i in range(num_blocks_y):
        for j in range(num_blocks_x):
            strength = strengths[i, j]
            
            # Apply the filter function if strength is above the threshold
            if strength > min_strength_threshold:
                block = blocks[i, j]
                # Filter function is called with the pre-calculated strength
                filtered_block = filter_func(block, strength)
                filtered_blocks[i, j] = filtered_block

    # Reconstruct the image from the (partially) filtered blocks
    new_image = combine_blocks_into_image(filtered_blocks)
    
    return new_image

def analyze_encoding_performance(reference_frames: List[np.ndarray], encoded_videos: Dict[str, str], block_size: int, width: int, height: int, temp_dir: str, masks_dir: str, sample_frames: List[int] = [0, 20, 40], video_bitrates: Dict[str, float] = {}, reference_video_path: str = None, framerate: float = 30.0) -> Dict:
    """
    A comprehensive function to analyze and compare video encoding performance using masked videos.

    This function:
    1. Creates masked versions (foreground/background) of all videos.
    2. Calculates all metrics (PSNR, SSIM, LPIPS, VMAF) separately for foreground and background.
    3. Generates and saves quality heatmaps for sample frames.
    4. Prints a unified summary report comparing all encoding methods.

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
        print(f"\n{'='*160}")
        print(f"{'COMPREHENSIVE ANALYSIS SUMMARY':^160}")
        print(f"{'='*160}")

        if not results:
            print("No results to display.")
            return

        # Unified metrics table
        print(f"\n{'QUALITY METRICS (Foreground / Background)':^160}")
        print(f"{'Method':<20} {'PSNR (dB)':<25} {'SSIM':<25} {'LPIPS':<25} {'VMAF':<25} {'Bitrate (Mbps)':<15}")
        print(f"{'-'*160}")

        for video_name, data in results.items():
            fg_data = data['foreground']
            bg_data = data['background']
            
            # Format metric strings (FG / BG)
            psnr_str = f"{fg_data.get('psnr_mean', 0):.2f} / {bg_data.get('psnr_mean', 0):.2f}"
            ssim_str = f"{fg_data.get('ssim_mean', 0):.4f} / {bg_data.get('ssim_mean', 0):.4f}"
            lpips_str = f"{fg_data.get('lpips_mean', 0):.4f} / {bg_data.get('lpips_mean', 0):.4f}"
            vmaf_str = f"{fg_data.get('vmaf_mean', 0):.2f} / {bg_data.get('vmaf_mean', 0):.2f}"
            bitrate_str = f"{data['bitrate_mbps']:.2f}"
            
            print(f"{video_name:<20} {psnr_str:<25} {ssim_str:<25} {lpips_str:<25} {vmaf_str:<25} {bitrate_str:<15}")
        
        print(f"{'-'*160}")

        # Trade-off analysis against the first video as baseline
        if len(results) > 1:
            baseline_name = list(results.keys())[0]
            print(f"\n{'TRADE-OFF ANALYSIS (vs. ' + baseline_name + ')':^160}")
            print(f"{'Method':<20} {'PSNR FG %':<15} {'PSNR BG %':<15} {'SSIM FG %':<15} {'SSIM BG %':<15} {'LPIPS FG %':<15} {'LPIPS BG %':<15} {'VMAF FG %':<15} {'VMAF BG %':<15}")
            print(f"{'-'*160}")
            
            for video_name in list(results.keys())[1:]:
                # Calculate changes for all metrics
                psnr_fg_change = 0
                psnr_bg_change = 0
                ssim_fg_change = 0
                ssim_bg_change = 0
                lpips_fg_change = 0
                lpips_bg_change = 0
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
                
                # Print table row
                print(f"{video_name:<20} {psnr_fg_change:+.2f}%{' '*8} {psnr_bg_change:+.2f}%{' '*8} {ssim_fg_change:+.2f}%{' '*8} {ssim_bg_change:+.2f}%{' '*8} {lpips_fg_change:+.2f}%{' '*8} {lpips_bg_change:+.2f}%{' '*8} {vmaf_fg_change:+.2f}%{' '*8} {vmaf_bg_change:+.2f}%{' '*8}")
            
            print(f"{'-'*160}")
    
    # --- Setup ---
    os.makedirs(temp_dir, exist_ok=True)
    masked_videos_dir = os.path.join(temp_dir, "masked_videos")
    frames_root = os.path.join(temp_dir, "frames")
    heatmaps_dir = os.path.join(temp_dir, "performance_figures")
    os.makedirs(masked_videos_dir, exist_ok=True)
    os.makedirs(frames_root, exist_ok=True)
    os.makedirs(heatmaps_dir, exist_ok=True)
    
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

    # Example usage parameters
    reference_video = "davis_test/bear.mp4"
    width, height = 640, 360
    block_size = 8
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
    os.system(f"ffmpeg -hide_banner -loglevel error -y -i {reference_video} -vf scale={width}:{height} -c:v rawvideo -pix_fmt yuv420p {raw_video_path}")

    print("Extracting reference frames...")
    os.system(f"ffmpeg -hide_banner -loglevel error -y -video_size {width}x{height} -r {framerate} -pixel_format yuv420p -i {raw_video_path} -q:v 2 {reference_frames_dir}/%05d.png")
    
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



    # --- ELVIS v1 (shrinking and extracting metadata) ---
    start = time.time()
    print(f"Shrinking and encoding frames with ELVIS v1...")

    # Shrink frames based on removability scores
    shrunk_frames_dir = os.path.join(experiment_dir, "frames", "shrunk")
    os.makedirs(shrunk_frames_dir, exist_ok=True)
    reference_frames = [cv2.imread(os.path.join(reference_frames_dir, f)) for f in sorted(os.listdir(reference_frames_dir)) if f.endswith('.png')]
    shrunk_frames, removal_masks, block_coords_to_remove = zip(*(apply_selective_removal(img, scores, block_size, shrink_amount=shrink_amount) for img, scores in zip(reference_frames, removability_scores)))
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
    np.savez(os.path.join(experiment_dir, f"shrink_masks_{block_size}.npz"), packed=masks_packed, shape=removal_masks.shape)

    end = time.time()
    execution_times["elvis_v1_shrinking"] = end - start
    print(f"ELVIS v1 shrinking completed in {end - start:.2f} seconds.\n")



    # --- DCT Damping-based ELVIS v2 (adaptive filtering and encoding) ---
    start = time.time()
    print(f"Applying DCT damping-based ELVIS v2 adaptive filtering and encoding...")
    dct_filtered_frames_dir = os.path.join(experiment_dir, "frames", "dct_filtered")
    os.makedirs(dct_filtered_frames_dir, exist_ok=True)
    reference_frames = [cv2.imread(os.path.join(reference_frames_dir, f)) for f
                        in sorted(os.listdir(reference_frames_dir)) if f.endswith('.png')]
    dct_strengths = map_scores_to_strengths(removability_scores, 
                                            max_value=block_size * 0.99,
                                            mapping_type='linear',
                                            dtype=np.int32)
    dct_filtered_frames = [apply_adaptive_filtering(img, strengths, block_size,
                                                   filter_func=apply_dct_damping,
                                                   min_strength_threshold=1.0)
                           for img, strengths in zip(reference_frames, dct_strengths)]
    for i, frame in enumerate(dct_filtered_frames):
        cv2.imwrite(os.path.join(dct_filtered_frames_dir, f"{i+1:05d}.png"), frame)

    dct_filtered_video = os.path.join(experiment_dir, "dct_filtered.mp4")
    encode_video(
        input_frames_dir=dct_filtered_frames_dir,
        output_video=dct_filtered_video,
        framerate=framerate,
        width=width,
        height=height,
        target_bitrate=target_bitrate
    )
    
    # Encode DCT strength maps as video for client-side adaptive restoration
    dct_strengths_video = os.path.join(maps_dir, "dct_strengths.mp4")
    encode_strength_maps_to_video(
        strength_maps=dct_strengths,
        output_video=dct_strengths_video,
        framerate=framerate
    )

    end = time.time()
    execution_times["elvis_v2_dct_filtering"] = end - start
    print(f"DCT damping-based ELVIS v2 filtering and encoding completed in {end - start:.2f} seconds.\n")



    # --- Downsampling-based ELVIS v2 (adaptive filtering and encoding) ---
    start = time.time()
    print(f"Applying downsampling-based ELVIS v2 adaptive filtering and encoding...")
    downsampled_frames_dir = os.path.join(experiment_dir, "frames", "downsampled")
    os.makedirs(downsampled_frames_dir, exist_ok=True)
    reference_frames = [cv2.imread(os.path.join(reference_frames_dir, f)) for f
                        in sorted(os.listdir(reference_frames_dir)) if f.endswith('.png')]
    downsample_strengths = map_scores_to_strengths(removability_scores, 
                                                   max_value=8,
                                                   mapping_type='power_of_2',
                                                   dtype=np.int32)
    downsampled_frames = [apply_adaptive_filtering(img, strengths, block_size,
                                                  filter_func=apply_downsampling,
                                                  min_strength_threshold=1.0)
                         for img, strengths in zip(reference_frames, downsample_strengths)]
    for i, frame in enumerate(downsampled_frames):
        cv2.imwrite(os.path.join(downsampled_frames_dir, f"{i+1:05d}.png"), frame)

    downsampled_video = os.path.join(experiment_dir, "downsampled.mp4")
    encode_video(
        input_frames_dir=downsampled_frames_dir,
        output_video=downsampled_video,
        framerate=framerate,
        width=width,
        height=height,
        target_bitrate=target_bitrate
    )
    
    # Encode downsampling strength maps as video for client-side adaptive restoration
    downsample_strengths_video = os.path.join(maps_dir, "downsample_strengths.mp4")
    encode_strength_maps_to_video(
        strength_maps=downsample_strengths,
        output_video=downsample_strengths_video,
        framerate=framerate
    )

    end = time.time()
    execution_times["elvis_v2_downsampling"] = end - start
    print(f"Downsampling-based ELVIS v2 filtering and encoding completed in {end - start:.2f} seconds.\n")



    # --- Gaussian Blur-based ELVIS v2 (adaptive filtering and encoding) ---
    start = time.time()
    print(f"Applying Gaussian blur-based ELVIS v2 adaptive filtering and encoding...")
    blurred_frames_dir = os.path.join(experiment_dir, "frames", "blurred")
    os.makedirs(blurred_frames_dir, exist_ok=True)
    reference_frames = [cv2.imread(os.path.join(reference_frames_dir, f)) for f
                        in sorted(os.listdir(reference_frames_dir)) if f.endswith('.png')]
    blur_strengths = map_scores_to_strengths(removability_scores, 
                                            max_value=3.0,
                                            mapping_type='linear',
                                            dtype=np.float32)
    blurred_frames = [apply_adaptive_filtering(img, strengths, block_size,
                                             filter_func=apply_gaussian_blur,
                                             min_strength_threshold=0.1)
                      for img, strengths in zip(reference_frames, blur_strengths)]
    for i, frame in enumerate(blurred_frames):
        cv2.imwrite(os.path.join(blurred_frames_dir, f"{i+1:05d}.png"), frame)
    
    blurred_video = os.path.join(experiment_dir, "blurred.mp4")
    encode_video(
        input_frames_dir=blurred_frames_dir,
        output_video=blurred_video,
        framerate=framerate,
        width=width,
        height=height,
        target_bitrate=target_bitrate
    )
    
    # Encode blur strength maps as video for client-side adaptive restoration
    blur_strengths_video = os.path.join(maps_dir, "blur_strengths.mp4")
    encode_strength_maps_to_video(
        strength_maps=blur_strengths,
        output_video=blur_strengths_video,
        framerate=framerate
    )

    end = time.time()
    execution_times["elvis_v2_gaussian_blur"] = end - start
    print(f"Gaussian blur-based ELVIS v2 filtering and encoding completed in {end - start:.2f} seconds.\n")



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

    # Stretch each frame using the removal masks
    stretched_frames = [cv2.imread(os.path.join(stretched_frames_dir, f"{i+1:05d}.png")) for i in range(len(removal_masks))]
    stretched_frames = [stretch_frame(img, mask, block_size) for img, mask in zip(stretched_frames, removal_masks)]
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

    # Encode the stretched frames losslessly TODO: this is not needed in practice
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

    # Encode the CV2 inpainted frames losslessly TODO: this is not needed in practice
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

    # Encode the ProPainter inpainted frames losslessly TODO: this is not needed in practice
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

    # Encode the E2FGVI inpainted frames losslessly TODO: this is not needed in practice
    inpainted_e2fgvi_video = os.path.join(experiment_dir, "inpainted_e2fgvi.mp4")
    encode_video(
        input_frames_dir=inpainted_e2fgvi_frames_dir,
        output_video=inpainted_e2fgvi_video,
        framerate=framerate,
        width=width,
        height=height,
        target_bitrate=None
    )



    # --- DCT Damping-based ELVIS v2: Decode video and apply adaptive restoration ---
    start = time.time()
    print(f"Decoding DCT filtered ELVIS v2 video and strength maps...")
    dct_decoded_frames_dir = os.path.join(experiment_dir, "frames", "dct_decoded")
    if not decode_video(dct_filtered_video, dct_decoded_frames_dir, framerate=framerate, start_number=1, quality=1):
        raise RuntimeError(f"Failed to decode DCT filtered video: {dct_filtered_video}")
    
    # Decode strength maps from compressed video
    dct_strengths_decoded = decode_strength_maps_from_video(
        video_path=dct_strengths_video,
        dtype=np.float32
    )

    # Adaptive restoration: Apply bilateral filtering with varying strength based on decoded maps
    print("Restoring DCT filtered frames using adaptive bilateral filtering...")
    dct_restored_frames_dir = os.path.join(experiment_dir, "frames", "dct_restored")
    os.makedirs(dct_restored_frames_dir, exist_ok=True)
    
    dct_decoded_frames = [cv2.imread(os.path.join(dct_decoded_frames_dir, f"{i+1:05d}.png")) 
                          for i in range(len(dct_strengths_decoded))]
    
    dct_restored_frames = [apply_adaptive_deblocking(img, strengths, block_size,
                                                      filter_func=apply_bilateral_deblocking,
                                                      min_strength_threshold=0.1)
                           for img, strengths in zip(dct_decoded_frames, dct_strengths_decoded)]
    
    for i, frame in enumerate(dct_restored_frames):
        cv2.imwrite(os.path.join(dct_restored_frames_dir, f"{i+1:05d}.png"), frame)

    # Encode the restored frames
    dct_restored_video = os.path.join(experiment_dir, "dct_restored.mp4")
    encode_video(
        input_frames_dir=dct_restored_frames_dir,
        output_video=dct_restored_video,
        framerate=framerate,
        width=width,
        height=height,
        target_bitrate=None
    )

    end = time.time()
    execution_times["elvis_v2_dct_restoration"] = end - start
    print(f"DCT damping-based ELVIS v2 adaptive restoration completed in {end - start:.2f} seconds.\n")



    # --- Downsampling-based ELVIS v2: Decode video and apply adaptive restoration ---
    start = time.time()
    print(f"Decoding downsampled ELVIS v2 video and strength maps...")
    downsampled_decoded_frames_dir = os.path.join(experiment_dir, "frames", "downsampled_decoded")
    if not decode_video(downsampled_video, downsampled_decoded_frames_dir, framerate=framerate, start_number=1, quality=1):
        raise RuntimeError(f"Failed to decode downsampled video: {downsampled_video}")
    
    # Decode strength maps from compressed video
    downsample_strengths_decoded = decode_strength_maps_from_video(
        video_path=downsample_strengths_video,
        dtype=np.int32
    )

    # Adaptive restoration: Apply super-resolution/sharpening with varying strength
    print("Restoring downsampled frames using adaptive super-resolution...")
    downsampled_restored_frames_dir = os.path.join(experiment_dir, "frames", "downsampled_restored")
    os.makedirs(downsampled_restored_frames_dir, exist_ok=True)
    
    downsampled_decoded_frames = [cv2.imread(os.path.join(downsampled_decoded_frames_dir, f"{i+1:05d}.png")) 
                                  for i in range(len(downsample_strengths_decoded))]
    
    downsampled_restored_frames = [apply_adaptive_deblocking(img, strengths, block_size,
                                                             filter_func=apply_super_resolution,
                                                             min_strength_threshold=1.0)
                                   for img, strengths in zip(downsampled_decoded_frames, downsample_strengths_decoded)]
    
    for i, frame in enumerate(downsampled_restored_frames):
        cv2.imwrite(os.path.join(downsampled_restored_frames_dir, f"{i+1:05d}.png"), frame)

    # Encode the restored frames
    downsampled_restored_video = os.path.join(experiment_dir, "downsampled_restored.mp4")
    encode_video(
        input_frames_dir=downsampled_restored_frames_dir,
        output_video=downsampled_restored_video,
        framerate=framerate,
        width=width,
        height=height,
        target_bitrate=None
    )

    end = time.time()
    execution_times["elvis_v2_downsampling_restoration"] = end - start
    print(f"Downsampling-based ELVIS v2 adaptive restoration completed in {end - start:.2f} seconds.\n")



    # --- Gaussian Blur-based ELVIS v2: Decode video and apply adaptive restoration ---
    start = time.time()
    print(f"Decoding blurred ELVIS v2 video and strength maps...")
    blurred_decoded_frames_dir = os.path.join(experiment_dir, "frames", "blurred_decoded")
    if not decode_video(blurred_video, blurred_decoded_frames_dir, framerate=framerate, start_number=1, quality=1):
        raise RuntimeError(f"Failed to decode blurred video: {blurred_video}")
    
    # Decode strength maps from compressed video
    blur_strengths_decoded = decode_strength_maps_from_video(
        video_path=blur_strengths_video,
        dtype=np.float32
    )

    # Adaptive restoration: Apply unsharp masking with varying strength based on decoded maps
    print("Restoring blurred frames using adaptive unsharp masking...")
    blurred_restored_frames_dir = os.path.join(experiment_dir, "frames", "blurred_restored")
    os.makedirs(blurred_restored_frames_dir, exist_ok=True)
    
    blurred_decoded_frames = [cv2.imread(os.path.join(blurred_decoded_frames_dir, f"{i+1:05d}.png")) 
                              for i in range(len(blur_strengths_decoded))]
    
    blurred_restored_frames = [apply_adaptive_deblocking(img, strengths, block_size,
                                                         filter_func=apply_unsharp_mask,
                                                         min_strength_threshold=0.1)
                               for img, strengths in zip(blurred_decoded_frames, blur_strengths_decoded)]
    
    for i, frame in enumerate(blurred_restored_frames):
        cv2.imwrite(os.path.join(blurred_restored_frames_dir, f"{i+1:05d}.png"), frame)

    # Encode the restored frames
    blurred_restored_video = os.path.join(experiment_dir, "blurred_restored.mp4")
    encode_video(
        input_frames_dir=blurred_restored_frames_dir,
        output_video=blurred_restored_video,
        framerate=framerate,
        width=width,
        height=height,
        target_bitrate=None
    )

    end = time.time()
    execution_times["elvis_v2_gaussian_blur_restoration"] = end - start
    print(f"Gaussian blur-based ELVIS v2 adaptive restoration completed in {end - start:.2f} seconds.\n")



    #########################################################################################################
    ######################################### PERFORMANCE EVALUATION ########################################
    #########################################################################################################

    print("Evaluating and comparing encoding performance...")
    start = time.time()

    # Compare file sizes and quality metrics
    video_sizes = {
        "Baseline": os.path.getsize(baseline_video),
        "Adaptive": os.path.getsize(adaptive_video),
        "ELVIS v1": os.path.getsize(shrunk_video),
        "ELVIS v2 DCT": os.path.getsize(dct_filtered_video),
        "ELVIS v2 Downsample": os.path.getsize(downsampled_video),
        "ELVIS v2 Blur": os.path.getsize(blurred_video)
    }

    duration = len(os.listdir(reference_frames_dir)) / framerate
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
        "ELVIS v2 DCT": dct_restored_video,
        "ELVIS v2 Downsample": downsampled_restored_video,
        "ELVIS v2 Blur": blurred_restored_video
    }

    ufo_masks_dir = os.path.join(experiment_dir, "maps", "ufo_masks")
    
    frame_count = len(os.listdir(reference_frames_dir))
    sample_frames = [frame_count // 2] # [0, frame_count // 4, frame_count // 2, 3 * frame_count // 4, frame_count - 1]
    sample_frames = [f for f in sample_frames if f < frame_count]

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
        framerate=framerate
    )
    
    # Add collected times to the results dictionary
    analysis_results["execution_times_seconds"] = execution_times

    # Add video parameters to the results dictionary
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