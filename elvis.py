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

def encode_video(input_frames_dir: str, output_video: str, framerate: float, width: int, height: int, lossless: bool = False, target_bitrate: int = None, **extra_params) -> None:
    """
    Encodes a video using a two-pass process with libx265.
    
    This function provides a unified interface for both lossless and lossy encoding:
    - Lossless: Uses veryslow preset with lossless=1 parameter
    - Lossy: Uses medium preset with bitrate control
    
    Both modes use two-pass encoding for optimal quality/size trade-off.
    
    Args:
        input_frames_dir: Directory containing input frames (e.g., '%05d.jpg').
        output_video: The path for the final encoded video file.
        framerate: The framerate of the output video.
        width: The width of the video.
        height: The height of the video.
        lossless: If True, encode losslessly. If False, use target_bitrate.
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
            "ffmpeg", "-hide_banner", "-loglevel", "warning",
            "-framerate", str(framerate),
            "-i", f"{input_frames_dir}/%05d.jpg",
            "-s", f"{width}x{height}",
        ]
        
        if lossless:
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
            subprocess.run(pass2_cmd, check=True, capture_output=True, text=True)
        else:
            # Lossy encoding with bitrate control and two passes
            if target_bitrate is None:
                raise ValueError("target_bitrate must be specified for lossy encoding")
            
            preset = "medium"
            
            # Pass 1
            pass1_cmd = base_cmd + [
                "-c:v", "libx265",
                "-b:v", str(target_bitrate),
                "-minrate", str(int(target_bitrate * 0.9)),
                "-maxrate", str(int(target_bitrate * 1.1)),
                "-bufsize", str(target_bitrate),
                "-preset", preset,
                "-g", "10",
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
                "-g", "10",
                "-x265-params", pass2_params,
                "-y", output_video
            ]
            subprocess.run(pass2_cmd, check=True, capture_output=True, text=True)
        
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
        input_frames_dir: Directory containing input frames (e.g., '%05d.jpg').
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
            lossless=False,
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
    import subprocess
    import json
    import tempfile
    
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
    Efficiently splits an image into non-overlapping blocks using vectorized operations.

    Args:
        image: The image to split, with shape (H, W, C).
        block_size: The side length (l) of each square block.

    Returns:
        A 5D array of blocks with shape (num_rows, num_cols, l, l, C).
    """
    H, W, C = image.shape
    num_rows = H // block_size
    num_cols = W // block_size
    
    # Reshape and transpose to create a "view" of the blocks without copying data
    blocks = image.reshape(num_rows, block_size, num_cols, block_size, C).transpose(0, 2, 1, 3, 4)
    return blocks

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

    import tempfile
    
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
            
            frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith('.jpg')])
            
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
                '-i', os.path.join(masked_frames_dir, '%05d.jpg'),
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
        os.makedirs(output_dir, exist_ok=True)
        decode_cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-i", video_path,
            "-q:v", "2",
            "-y", os.path.join(output_dir, "%05d.jpg")
        ]
        result = subprocess.run(decode_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Error decoding {video_path}: {result.stderr}")
            return False
        return True

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
        cv2.imwrite(os.path.join(reference_frames_dir, f"{i+1:05d}.jpg"), frame)
    
    ref_fg_video = os.path.join(masked_videos_dir, "reference_fg.mp4")
    ref_bg_video = os.path.join(masked_videos_dir, "reference_bg.mp4")
    
    _create_masked_video(reference_frames_dir, masks_dir, ref_fg_video, width, height, framerate, 'foreground')
    _create_masked_video(reference_frames_dir, masks_dir, ref_bg_video, width, height, framerate, 'background')
    
    analysis_results = {}

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
        
        decoded_frame_files = sorted([f for f in os.listdir(video_decoded_dir) if f.endswith('.jpg')])
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
        ref_fg_frame_files = sorted([f for f in os.listdir(ref_fg_frames_dir) if f.endswith('.jpg')])
        enc_fg_frame_files = sorted([f for f in os.listdir(enc_fg_frames_dir) if f.endswith('.jpg')])
        num_fg_frames = min(len(ref_fg_frame_files), len(enc_fg_frame_files))
        
        # Calculate PSNR and SSIM frame-by-frame
        psnr_scores_fg = []
        ssim_scores_fg = []
        for i in range(num_fg_frames):
            ref_frame = cv2.imread(os.path.join(ref_fg_frames_dir, ref_fg_frame_files[i]))
            enc_frame = cv2.imread(os.path.join(enc_fg_frames_dir, enc_fg_frame_files[i]))
            
            if ref_frame is not None and enc_frame is not None:
                # Calculate PSNR for the whole frame
                psnr_val = psnr(ref_frame, enc_frame)
                if np.isfinite(psnr_val):
                    psnr_scores_fg.append(psnr_val)
                
                # Calculate SSIM for the whole frame
                ssim_val = ssim(ref_frame, enc_frame, gaussian_weights=True, data_range=255, channel_axis=-1)
                if np.isfinite(ssim_val):
                    ssim_scores_fg.append(ssim_val)
        
        analysis_results[video_name]['foreground']['psnr_mean'] = np.mean(psnr_scores_fg) if psnr_scores_fg else 0
        analysis_results[video_name]['foreground']['psnr_std'] = np.std(psnr_scores_fg) if psnr_scores_fg else 0
        analysis_results[video_name]['foreground']['ssim_mean'] = np.mean(ssim_scores_fg) if ssim_scores_fg else 0
        analysis_results[video_name]['foreground']['ssim_std'] = np.std(ssim_scores_fg) if ssim_scores_fg else 0
        
        # Calculate LPIPS for foreground
        ref_fg_frames_list = [cv2.imread(os.path.join(ref_fg_frames_dir, f)) for f in ref_fg_frame_files[:num_fg_frames]]
        enc_fg_frames_list = [cv2.imread(os.path.join(enc_fg_frames_dir, f)) for f in enc_fg_frame_files[:num_fg_frames]]
        lpips_scores_fg = calculate_lpips_per_frame(ref_fg_frames_list, enc_fg_frames_list)
        analysis_results[video_name]['foreground']['lpips_mean'] = np.mean(lpips_scores_fg) if lpips_scores_fg else 0
        analysis_results[video_name]['foreground']['lpips_std'] = np.std(lpips_scores_fg) if lpips_scores_fg else 0
        
        # Calculate VMAF for foreground
        vmaf_fg = calculate_vmaf(ref_fg_video, enc_fg_video, width, height, framerate)
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
        
        ref_bg_frame_files = sorted([f for f in os.listdir(ref_bg_frames_dir) if f.endswith('.jpg')])
        enc_bg_frame_files = sorted([f for f in os.listdir(enc_bg_frames_dir) if f.endswith('.jpg')])
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
            frame_files = sorted([f for f in os.listdir(video_decoded_dir) if f.endswith('.jpg')])
            
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

def combine_blocks_into_image(blocks: np.ndarray) -> np.ndarray:
    """
    Efficiently combines an array of blocks back into a single image.

    Args:
        blocks: A 5D array of blocks with shape (num_rows, num_cols, l, l, C).

    Returns:
        The reconstructed image with shape (num_rows*l, num_cols*l, C).
    """
    num_rows, num_cols, l_h, l_w, C = blocks.shape
    
    # Transpose and reshape back to the original image dimensions
    image = blocks.transpose(0, 2, 1, 3, 4).reshape(num_rows * l_h, num_cols * l_w, C)
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

def map_strength_linear(normalized_scores: np.ndarray, max_strength: float = 1.0,
                       save_maps: bool = False, maps_dir: str = None, width: int = None, height: int = None) -> np.ndarray:
    """
    Linearly maps normalized scores [0, 1] to a strength range [0, max_strength].
    This is suitable for filters where strength is a simple multiplier (e.g., Gaussian blur's sigma).
    
    Args:
        normalized_scores: 3D array of shape (num_frames, num_blocks_y, num_blocks_x)
        max_strength: Maximum strength value
        save_maps: Whether to save visualization maps
        maps_dir: Directory to save maps to
        width: Frame width (required if save_maps is True)
        height: Frame height (required if save_maps is True)
    """
    strength_maps = normalized_scores * max_strength
    
    # Save visualization maps if requested
    if save_maps and maps_dir is not None and width is not None and height is not None:
        blur_maps_dir = os.path.join(maps_dir, "blur_strength_maps")
        os.makedirs(blur_maps_dir, exist_ok=True)
        print(f"Saving blur strength maps to {blur_maps_dir}...")
        for i, strengths in enumerate(strength_maps):
            # Normalize strengths to 0-255 range for visualization
            strength_normalized = (strengths / strengths.max() * 255).astype(np.uint8) if strengths.max() > 0 else strengths.astype(np.uint8)
            # Resize to match frame dimensions for easier visualization
            strength_img = cv2.resize(strength_normalized, (width, height), interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(os.path.join(blur_maps_dir, f"frame_{i+1:05d}.png"), strength_img)

    return strength_maps

def map_strength_dct_cutoff(normalized_scores: np.ndarray, block_size: int, max_cutoff_reduction: float = 0.8, 
                            save_maps: bool = False, maps_dir: str = None, width: int = None, height: int = None) -> np.ndarray:
    """
    Maps normalized scores [0, 1] to a DCT high-frequency cutoff strength.
    
    The strength represents how many high-frequency components to cut.
    A score of 1.0 means cutting max_cutoff_reduction * block_size coefficients.
    e.g., max_cutoff_reduction=0.8 means up to 80% of the block size is cut.
    
    Args:
        normalized_scores: 3D array of shape (num_frames, num_blocks_y, num_blocks_x)
        block_size: Size of blocks
        max_cutoff_reduction: Maximum cutoff reduction factor
        save_maps: Whether to save visualization maps
        maps_dir: Directory to save maps to
        width: Frame width (required if save_maps is True)
        height: Frame height (required if save_maps is True)
    """
    # Max number of coefficients to cut (must be < block_size)
    max_cut = block_size * max_cutoff_reduction
    
    # Scale scores to the cutoff range
    strength = normalized_scores * max_cut
    
    # Ensure strength is at least 1 for any significant score
    strength_maps = np.maximum(strength, 0.0).astype(np.int32)
    
    # Save visualization maps if requested
    if save_maps and maps_dir is not None and width is not None and height is not None:
        dct_maps_dir = os.path.join(maps_dir, "dct_strength_maps")
        os.makedirs(dct_maps_dir, exist_ok=True)
        print(f"Saving DCT strength maps to {dct_maps_dir}...")
        for i, strengths in enumerate(strength_maps):
            # Normalize strengths to 0-255 range for visualization
            strength_normalized = (strengths / strengths.max() * 255).astype(np.uint8) if strengths.max() > 0 else strengths.astype(np.uint8)
            # Resize to match frame dimensions for easier visualization
            strength_img = cv2.resize(strength_normalized, (width, height), interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(os.path.join(dct_maps_dir, f"frame_{i+1:05d}.png"), strength_img)
    
    return strength_maps

def map_strength_downsampling_factor(normalized_scores: np.ndarray, max_factor: int = 8,
                                    save_maps: bool = False, maps_dir: str = None, width: int = None, height: int = None) -> np.ndarray:
    """
    Maps normalized scores [0, 1] to a downsampling factor that is a power of 2: 
    [1, 2, 4, ..., max_factor].

    Args:
        normalized_scores: 3D array of shape (num_frames, num_blocks_y, num_blocks_x)
        max_factor: The maximum allowed downsampling factor (must be a power of 2).
        save_maps: Whether to save visualization maps
        maps_dir: Directory to save maps to
        width: Frame width (required if save_maps is True)
        height: Frame height (required if save_maps is True)

    Returns:
        The 3D array of integer downsampling factors (powers of 2).
    """
    if max_factor < 1 or (max_factor & (max_factor - 1) != 0):
        raise ValueError("max_factor must be a positive power of 2 (e.g., 1, 2, 4, 8).")

    # 1. Define the possible power-of-2 factors (e.g., [1, 2, 4, 8])
    # The number of levels is log2(max_factor) + 1
    levels = int(np.log2(max_factor)) + 1
    pow2_factors = np.logspace(0, levels - 1, levels, base=2, dtype=np.int32) 
    
    # 2. Map normalized scores [0, 1] to the indices of these factors [0, 1, 2, ..., levels-1]
    # We use (levels - 1) to map 1.0 to the highest index.
    
    # Ensure scores slightly less than 1.0 map correctly to the max index
    # Use np.floor to get the index.
    indices = np.floor(normalized_scores * (levels - 1)).astype(np.int32)
    
    # 3. Use the indices to look up the corresponding factor
    factors = pow2_factors[indices]
    
    # Save visualization maps if requested
    if save_maps and maps_dir is not None and width is not None and height is not None:
        downsample_maps_dir = os.path.join(maps_dir, "downsample_strength_maps")
        os.makedirs(downsample_maps_dir, exist_ok=True)
        print(f"Saving downsampling strength maps to {downsample_maps_dir}...")
        for i, strengths in enumerate(factors):
            # Normalize strengths to 0-255 range for visualization
            strength_normalized = (strengths / strengths.max() * 255).astype(np.uint8) if strengths.max() > 0 else strengths.astype(np.uint8)
            # Resize to match frame dimensions for easier visualization
            strength_img = cv2.resize(strength_normalized, (width, height), interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(os.path.join(downsample_maps_dir, f"frame_{i+1:05d}.png"), strength_img)

    return factors

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
    os.system(f"ffmpeg -hide_banner -loglevel error -y -video_size {width}x{height} -r {framerate} -pixel_format yuv420p -i {raw_video_path} -q:v 2 {reference_frames_dir}/%05d.jpg")
    
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
        lossless=False,
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
    reference_frames = [cv2.imread(os.path.join(reference_frames_dir, f)) for f in sorted(os.listdir(reference_frames_dir)) if f.endswith('.jpg')]
    shrunk_frames, removal_masks, block_coords_to_remove = zip(*(apply_selective_removal(img, scores, block_size, shrink_amount=shrink_amount) for img, scores in zip(reference_frames, removability_scores)))
    for i, frame in enumerate(shrunk_frames):
        cv2.imwrite(os.path.join(shrunk_frames_dir, f"{i+1:05d}.jpg"), frame)

    # Encode the shrunk frames (use actual shrunk frame dimensions, not original)
    shrunk_video = os.path.join(experiment_dir, "shrunk.mp4")
    shrunk_width = shrunk_frames[0].shape[1]  # Width is reduced due to removed blocks
    encode_video(
        input_frames_dir=shrunk_frames_dir,
        output_video=shrunk_video,
        framerate=framerate,
        width=shrunk_width,
        height=height,
        lossless=False,
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
                        in sorted(os.listdir(reference_frames_dir)) if f.endswith('.jpg')]
    dct_strengths = map_strength_dct_cutoff(removability_scores, block_size, max_cutoff_reduction=0.99,
                                            save_maps=True, maps_dir=maps_dir, width=width, height=height)
    dct_filtered_frames = [apply_adaptive_filtering(img, strengths, block_size,
                                                   filter_func=apply_dct_damping,
                                                   min_strength_threshold=1.0)
                           for img, strengths in zip(reference_frames, dct_strengths)]
    for i, frame in enumerate(dct_filtered_frames):
        cv2.imwrite(os.path.join(dct_filtered_frames_dir, f"{i+1:05d}.jpg"), frame)

    dct_filtered_video = os.path.join(experiment_dir, "dct_filtered.mp4")
    encode_video(
        input_frames_dir=dct_filtered_frames_dir,
        output_video=dct_filtered_video,
        framerate=framerate,
        width=width,
        height=height,
        lossless=False,
        target_bitrate=target_bitrate
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
                        in sorted(os.listdir(reference_frames_dir)) if f.endswith('.jpg')]
    downsample_strengths = map_strength_downsampling_factor(removability_scores, max_factor=8,
                                                            save_maps=True, maps_dir=maps_dir, width=width, height=height)
    downsampled_frames = [apply_adaptive_filtering(img, strengths, block_size,
                                                  filter_func=apply_downsampling,
                                                  min_strength_threshold=1.0)
                         for img, strengths in zip(reference_frames, downsample_strengths)]
    for i, frame in enumerate(downsampled_frames):
        cv2.imwrite(os.path.join(downsampled_frames_dir, f"{i+1:05d}.jpg"), frame)

    downsampled_video = os.path.join(experiment_dir, "downsampled.mp4")
    encode_video(
        input_frames_dir=downsampled_frames_dir,
        output_video=downsampled_video,
        framerate=framerate,
        width=width,
        height=height,
        lossless=False,
        target_bitrate=target_bitrate
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
                        in sorted(os.listdir(reference_frames_dir)) if f.endswith('.jpg')]
    blur_strengths = map_strength_linear(removability_scores, max_strength=3.0,
                                        save_maps=True, maps_dir=maps_dir, width=width, height=height)
    blurred_frames = [apply_adaptive_filtering(img, strengths, block_size,
                                             filter_func=apply_gaussian_blur,
                                             min_strength_threshold=0.1)
                      for img, strengths in zip(reference_frames, blur_strengths)]
    for i, frame in enumerate(blurred_frames):
        cv2.imwrite(os.path.join(blurred_frames_dir, f"{i+1:05d}.jpg"), frame)
    
    blurred_video = os.path.join(experiment_dir, "blurred.mp4")
    encode_video(
        input_frames_dir=blurred_frames_dir,
        output_video=blurred_video,
        framerate=framerate,
        width=width,
        height=height,
        lossless=False,
        target_bitrate=target_bitrate
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
    os.makedirs(stretched_frames_dir, exist_ok=True)
    decode_cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "warning",
        "-i", shrunk_video,
        "-q:v", "1",
        "-r", str(framerate),
        "-f", "image2",
        "-start_number", "1",
        "-y", f"{stretched_frames_dir}/%05d.jpg"
    ]
    result = subprocess.run(decode_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Stretched decoding failed: {result.stderr}")
        raise RuntimeError(f"Stretched decoding failed: {result.stderr}")

    # Stretch each frame using the removal masks
    stretched_frames = [cv2.imread(os.path.join(stretched_frames_dir, f"{i+1:05d}.jpg")) for i in range(len(removal_masks))]
    stretched_frames = [stretch_frame(img, mask, block_size) for img, mask in zip(stretched_frames, removal_masks)]
    for i, frame in enumerate(stretched_frames):
        cv2.imwrite(os.path.join(stretched_frames_dir, f"{i+1:05d}.jpg"), frame)

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
        lossless=True
    )

    # --- Inpainting ---
    start = time.time()
    print(f"Inpainting stretched frames to fill in removed blocks...")

    # Inpaint the stretched frames to fill in removed blocks TODO: replace with ProPainter or E2FGVI
    inpainted_frames_dir = os.path.join(experiment_dir, "frames", "inpainted")
    os.makedirs(inpainted_frames_dir, exist_ok=True)
    for i in range(len(removal_masks)):
        stretched_frame = cv2.imread(os.path.join(stretched_frames_dir, f"{i+1:05d}.jpg"))
        mask_img = cv2.imread(os.path.join(removal_masks_dir, f"{i+1:05d}.png"), cv2.IMREAD_GRAYSCALE)
        inpainted_frame = cv2.inpaint(stretched_frame, mask_img, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
        cv2.imwrite(os.path.join(inpainted_frames_dir, f"{i+1:05d}.jpg"), inpainted_frame)

    end = time.time()
    execution_times["elvis_v1_inpainting"] = end - start
    print(f"ELVIS v1 inpainting completed in {end - start:.2f} seconds.\n")

    # Encode the inpainted frames losslessly TODO: this is not needed in practice
    inpainted_video = os.path.join(experiment_dir, "inpainted.mp4")
    encode_video(
        input_frames_dir=inpainted_frames_dir,
        output_video=inpainted_video,
        framerate=framerate,
        width=width,
        height=height,
        lossless=True
    )



    # --- DCT Damping-based ELVIS v2: Decode the DCT filtered video to frames and restore it with OpenCV (TODO: use SOTA model) ---
    start = time.time()
    print(f"Decoding DCT filtered ELVIS v2 video...")
    dct_decoded_frames_dir = os.path.join(experiment_dir, "frames", "dct_decoded")
    os.makedirs(dct_decoded_frames_dir, exist_ok=True)
    decode_cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "warning",
        "-i", dct_filtered_video,
        "-q:v", "1",
        "-r", str(framerate),
        "-f", "image2",
        "-start_number", "1",
        "-y", f"{dct_decoded_frames_dir}/%05d.jpg"
    ]
    result = subprocess.run(decode_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"DCT filtered decoding failed: {result.stderr}")
        raise RuntimeError(f"DCT filtered decoding failed: {result.stderr}")

    # Restoration: Deblock using a simple bilateral filter (placeholder for a SOTA model)
    print("Restoring DCT filtered frames using bilateral filter...")
    dct_restored_frames_dir = os.path.join(experiment_dir, "frames", "dct_restored")
    os.makedirs(dct_restored_frames_dir, exist_ok=True)
    for i in range(len(removability_scores)):
        dct_frame = cv2.imread(os.path.join(dct_decoded_frames_dir, f"{i+1:05d}.jpg"))
        restored_frame = cv2.bilateralFilter(dct_frame, d=9, sigmaColor=75, sigmaSpace=75)
        cv2.imwrite(os.path.join(dct_restored_frames_dir, f"{i+1:05d}.jpg"), restored_frame)

    # Encode the restored frames
    dct_restored_video = os.path.join(experiment_dir, "dct_restored.mp4")
    encode_video(
        input_frames_dir=dct_restored_frames_dir,
        output_video=dct_restored_video,
        framerate=framerate,
        width=width,
        height=height,
        lossless=True
    )

    end = time.time()
    execution_times["elvis_v2_dct_restoration"] = end - start
    print(f"DCT damping-based ELVIS v2 restoration completed in {end - start:.2f} seconds.\n")



    # --- Downsampling-based ELVIS v2: Decode the downsampled video to frames and restore it with OpenCV (TODO: use SOTA model) ---
    start = time.time()
    print(f"Decoding downsampled ELVIS v2 video...")
    downsampled_decoded_frames_dir = os.path.join(experiment_dir, "frames", "downsampled_decoded")
    os.makedirs(downsampled_decoded_frames_dir, exist_ok=True)
    decode_cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "warning",
        "-i", downsampled_video,
        "-q:v", "1",
        "-r", str(framerate),
        "-f", "image2",
        "-start_number", "1",
        "-y", f"{downsampled_decoded_frames_dir}/%05d.jpg"
    ]
    result = subprocess.run(decode_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Downsampled decoding failed: {result.stderr}")
        raise RuntimeError(f"Downsampled decoding failed: {result.stderr}")

    # Restoration: Upscale using a simple Lanczos interpolation (placeholder for a SOTA model)
    print("Restoring downsampled frames using Lanczos interpolation...")
    downsampled_restored_frames_dir = os.path.join(experiment_dir, "frames", "downsampled_restored")
    os.makedirs(downsampled_restored_frames_dir, exist_ok=True)
    for i in range(len(removability_scores)):
        downsampled_frame = cv2.imread(os.path.join(downsampled_decoded_frames_dir, f"{i+1:05d}.jpg"))
        restored_frame = cv2.resize(downsampled_frame, (width, height), interpolation=cv2.INTER_LANCZOS4)
        cv2.imwrite(os.path.join(downsampled_restored_frames_dir, f"{i+1:05d}.jpg"), restored_frame)

    # Encode the restored frames
    downsampled_restored_video = os.path.join(experiment_dir, "downsampled_restored.mp4")
    encode_video(
        input_frames_dir=downsampled_restored_frames_dir,
        output_video=downsampled_restored_video,
        framerate=framerate,
        width=width,
        height=height,
        lossless=True
    )

    end = time.time()
    execution_times["elvis_v2_downsampling_restoration"] = end - start
    print(f"Downsampling-based ELVIS v2 restoration completed in {end - start:.2f} seconds.\n")



    # --- Gaussian Blur-based ELVIS v2: Decode the blurred video to frames and restore it with OpenCV (TODO: use SOTA model) ---
    start = time.time()
    print(f"Decoding blurred ELVIS v2 video...")
    blurred_decoded_frames_dir = os.path.join(experiment_dir, "frames", "blurred_decoded")
    os.makedirs(blurred_decoded_frames_dir, exist_ok=True)
    decode_cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "warning",
        "-i", blurred_video,
        "-q:v", "1",
        "-r", str(framerate),
        "-f", "image2",
        "-start_number", "1",
        "-y", f"{blurred_decoded_frames_dir}/%05d.jpg"
    ]
    result = subprocess.run(decode_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Blurred decoding failed: {result.stderr}")
        raise RuntimeError(f"Blurred decoding failed: {result.stderr}")

    # Restoration: Deblur using a simple unsharp mask (placeholder for a SOTA model)
    print("Restoring blurred frames using unsharp masking...")
    blurred_restored_frames_dir = os.path.join(experiment_dir, "frames", "blurred_restored")
    os.makedirs(blurred_restored_frames_dir, exist_ok=True)
    for i in range(len(removability_scores)):
        blurred_frame = cv2.imread(os.path.join(blurred_decoded_frames_dir, f"{i+1:05d}.jpg"))
        gaussian = cv2.GaussianBlur(blurred_frame, (9, 9), 10.0)
        restored_frame = cv2.addWeighted(blurred_frame, 1.5, gaussian, -0.5, 0)
        cv2.imwrite(os.path.join(blurred_restored_frames_dir, f"{i+1:05d}.jpg"), restored_frame)

    # Encode the restored frames
    blurred_restored_video = os.path.join(experiment_dir, "blurred_restored.mp4")
    encode_video(
        input_frames_dir=blurred_restored_frames_dir,
        output_video=blurred_restored_video,
        framerate=framerate,
        width=width,
        height=height,
        lossless=True
    )

    end = time.time()
    execution_times["elvis_v2_gaussian_blur_restoration"] = end - start
    print(f"Gaussian blur-based ELVIS v2 restoration completed in {end - start:.2f} seconds.\n")



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
        "ELVIS v1": inpainted_video,
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