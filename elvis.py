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
    
    # Base bits per pixel for H.265 (typically 0.1-0.2 for good quality)
    bits_per_pixel = 0.3 * quality_factor
    
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
        containing the final removability scores.
    """
    # Create temporary directories for outputs
    evca_dir = os.path.join(working_dir, "evca")
    ufo_masks_dir = os.path.join(working_dir, "UFO_masks")
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
        
        # Load the actual EVCA output files
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
                removability_scores[i][background_blocks] *= 100.0
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

        return removability_scores
    
    except Exception as e:
        print(f"Error in calculate_removability_scores: {e}")
        raise
    finally:
        # Always return to original directory
        os.chdir(original_dir)

# def select_frames_to_drop(removability_scores: np.ndarray, drop_ratio: float = 0.1, validate_with_optical_flow: bool = True, video_path: str = None, motion_threshold: float = 2.0) -> list[int]:
#     """
#     Selects frames to drop using a "uniformity score" (mean/variance) and an
#     optional optical flow validation step.

#     Args:
#         removability_scores: Your 3D array of per-block scores.
#         drop_ratio: The target fraction of frames to drop.
#         validate_with_optical_flow: If True, uses optical flow as a final check.
#         video_path: Path to the video file (required if validation is enabled).
#         motion_threshold: Max allowed motion magnitude for a drop to be confirmed.

#     Returns:
#         A final list of frame indices to be dropped.
#     """
#     if validate_with_optical_flow and not video_path:
#         raise ValueError("video_path must be provided if validate_with_optical_flow is True.")

#     # 1. Calculate frame uniformity scores (mean/variance) - integrated from calculate_frame_uniformity_scores
#     print("Calculating frame uniformity scores (mean/variance)...")
    
#     if removability_scores.ndim != 3:
#         raise ValueError("Input must be a 3D array of shape (frames, height, width)")

#     # Calculate mean and variance across all blocks for each frame
#     mean_scores = np.mean(removability_scores, axis=(1, 2))
#     variance_scores = np.var(removability_scores, axis=(1, 2))

#     # Normalize both to prevent scale issues
#     normalized_mean = normalize_array(mean_scores)
#     normalized_variance = normalize_array(variance_scores)

#     # Combine them. We want high mean and low variance.
#     # Add a small epsilon to the denominator to avoid division by zero.
#     epsilon = 1e-6
#     frame_scores = normalized_mean / (normalized_variance + epsilon)
    
#     # Final normalization of the combined score
#     frame_scores = normalize_array(frame_scores)
    
#     # We assume scene changes are already handled, so we don't mark any frames as forbidden.

#     # 2. Get initial candidates based on the uniformity score
#     num_frames = len(frame_scores)
#     num_to_drop_target = int(num_frames * drop_ratio)
    
#     # Get indices sorted by score (high score = good candidate)
#     sorted_indices = np.argsort(frame_scores)[::-1]
    
#     # 3. Select and (optionally) validate candidates
#     final_dropped_indices = []
#     dropped_flags = np.zeros(num_frames, dtype=bool)

#     print("Selecting best candidate frames...")
#     for idx in sorted_indices:
#         if len(final_dropped_indices) >= num_to_drop_target:
#             break

#         # Boundary and consecutive-frame checks are essential
#         if idx == 0 or idx == num_frames - 1 or dropped_flags[idx-1] or dropped_flags[idx+1]:
#             continue
        
#         # --- Validation Step ---
#         is_candidate_valid = True
#         if validate_with_optical_flow:
#             motion_magnitude = calculate_motion_coherence(video_path, idx - 1, idx + 1)
#             if motion_magnitude >= motion_threshold:
#                 is_candidate_valid = False # Motion is too complex, reject candidate
        
#         if is_candidate_valid:
#             final_dropped_indices.append(idx)
#             dropped_flags[idx] = True
    
#     print(f"Targeted {num_to_drop_target} frames to drop.")
#     if validate_with_optical_flow:
#         print(f"Confirmed {len(final_dropped_indices)} frames after motion validation.")
#     else:
#         print(f"Selected {len(final_dropped_indices)} frames based on uniformity score alone.")
          
#     return sorted(final_dropped_indices)

def encode_with_roi(input_frames_dir: str, output_video: str, removability_scores: np.ndarray, block_size: int, framerate: float, width: int, height: int, segment_size: int = 30, target_bitrate: int = 1000000) -> None:
    """
    Encode video using ROI-based quantization by processing in segments and then concatenating them.
    
    Args:
        input_frames_dir: Directory containing input frames
        output_video: Output video file path
        removability_scores: 3D array of removability scores
        block_size: Size of blocks used for scoring
        framerate: Video framerate
        width: Video width
        height: Video height
        segment_size: Number of frames per segment for processing
        target_bitrate: Target bitrate in bits per second (default: 1000000 = 1 Mbps)
    """
    
    num_frames = removability_scores.shape[0]
    temp_dir = os.path.dirname(output_video)
    segments_dir = os.path.join(temp_dir, "segments")
    os.makedirs(segments_dir, exist_ok=True)
    
    segment_files = []
    filelist_path = os.path.join(temp_dir, "segments_list.txt")

    try:
        # --- Part 1: Encode video segments ---
        for segment_start in range(0, num_frames, segment_size):
            segment_end = min(segment_start + segment_size, num_frames)
            segment_idx = segment_start // segment_size
            
            print(f"Processing segment {segment_idx + 1} (frames {segment_start}-{segment_end-1})")
            
            # Calculate average ROI for this segment
            segment_scores = removability_scores[segment_start:segment_end]
            avg_segment_scores = np.mean(segment_scores, axis=0)
            
            # Generate ROI filter for this segment
            num_blocks_y, num_blocks_x = avg_segment_scores.shape
            
            # Normalize scores to [0, 1] range
            normalized_scores = normalize_array(avg_segment_scores)
            
            # Create ROI filter string
            roi_filters = []
            for block_y in range(num_blocks_y):
                for block_x in range(num_blocks_x):
                    score = normalized_scores[block_y, block_x]
                    x1 = block_x * block_size
                    y1 = block_y * block_size
                    x2 = min(x1 + block_size, width)
                    y2 = min(y1 + block_size, height)
                    
                    # Convert removability score to quantization offset
                    qoffset = (score - 0.5) * 2.0  # Maps [0,1] to [-1,1]
                    
                    if abs(qoffset) > 0.1:
                        roi_filter = f"addroi=x={x1}:y={y1}:w={x2-x1}:h={y2-y1}:qoffset={qoffset:.2f}"
                        roi_filters.append(roi_filter)
            
            roi_filter_string = ",".join(roi_filters) if roi_filters else ""
            
            # Create segment video file
            segment_file = os.path.join(segments_dir, f"segment_{segment_idx:03d}.mp4")
            
            # Build FFmpeg command for this segment
            cmd = [
                "ffmpeg", "-hide_banner", "-loglevel", "warning",
                "-start_number", str(segment_start + 1),
                "-framerate", str(framerate),
                "-i", f"{input_frames_dir}/%05d.jpg",
                "-frames:v", str(segment_end - segment_start),
            ]
            
            if roi_filter_string:
                cmd.extend(["-vf", roi_filter_string])
            
            cmd.extend([
                "-c:v", "libx265",
                "-b:v", str(target_bitrate),
                "-minrate", str(int(target_bitrate * 0.9)),
                "-maxrate", str(int(target_bitrate * 1.1)),
                "-bufsize", str(target_bitrate),
                "-preset", "medium",
                "-g", "1",
                "-y", segment_file
            ])
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"Encoding failed for segment: {result.stderr}")
                raise RuntimeError(f"Encoding failed for segment: {result.stderr}")
            
            segment_files.append(segment_file)
        
        # --- Part 2: Concatenate all segments into final video (inlined functionality) ---
        if not segment_files:
            print("No segments were generated.")
            return

        print(f"Concatenating {len(segment_files)} segments into final video...")

        # Create a temporary file list for ffmpeg concat
        with open(filelist_path, 'w') as f:
            for segment_file in segment_files:
                f.write(f"file '{os.path.abspath(segment_file)}'\n")
        
        # Concatenate using ffmpeg's concat demuxer
        concat_cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "warning",
            "-f", "concat",
            "-safe", "0",
            "-i", filelist_path,
            "-c", "copy",
            "-y", output_video
        ]
        
        result = subprocess.run(concat_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Concatenation failed: {result.stderr}")
            
    finally:
        # --- Part 3: Clean up all temporary files and directories ---
        if os.path.exists(segments_dir):
            shutil.rmtree(segments_dir, ignore_errors=True)
        if os.path.exists(filelist_path):
            os.remove(filelist_path)


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

def msssim(ref_block: np.ndarray, test_block: np.ndarray) -> float:
    """
    Calculates the MS-SSIM for a single pair of image blocks.
    """
    return ssim(ref_block, test_block, gaussian_weights=True, data_range=255, channel_axis=-1)

def calculate_blockwise_metric(ref_blocks: np.ndarray, test_blocks: np.ndarray, metric_func: Callable[..., float]) -> np.ndarray:
    """
    Applies a given metric function to each pair of blocks.

    Args:
        ref_blocks: 5D array of reference blocks.
        test_blocks: 5D array of test blocks.
        metric_func: The metric function to apply (e.g., psnr, ssim).
                     It should accept two blocks (np.ndarray) as its first arguments.

    Returns:
        A 2D NumPy array (map) of the calculated metric scores.
    """
    num_blocks_y, num_blocks_x, block_h, block_w, channels = ref_blocks.shape
    
    # Reshape for easy iteration: (total_num_blocks, h, w, c)
    ref_blocks_list = ref_blocks.reshape(-1, block_h, block_w, channels)
    test_blocks_list = test_blocks.reshape(-1, block_h, block_w, channels)
    
    # Use a list comprehension to apply the metric function to each block pair
    scores_flat = [metric_func(r, t) for r, t in zip(ref_blocks_list, test_blocks_list)]
    
    # Reshape the flat list of scores back into a 2D map
    metric_map = np.array(scores_flat).reshape(num_blocks_y, num_blocks_x)
    
    return metric_map

def split_image_into_blocks(image: np.ndarray, l: int) -> np.ndarray:
    """
    Splits an image into blocks of a specific size using vectorization.
    """
    n, m, c = image.shape
    num_rows = n // l
    num_cols = m // l
    return image.reshape(num_rows, l, num_cols, l, c).transpose(0, 2, 1, 3, 4)

def analyze_encoding_performance(reference_frames_dir: str, encoded_videos: Dict[str, str], metrics: Dict[str, Callable], block_size: int, width: int, height: int, temp_dir: str, masks_dir: str, sample_frames: List[int] = [0, 20, 40]) -> Dict:
    """
    A comprehensive function to analyze and compare video encoding performance.

    This function:
    1. Decodes videos to frames.
    2. For each frame, calculates multiple block-wise quality metrics.
    3. Separates metrics for foreground and background regions using masks.
    4. Generates and saves quality heatmaps for sample frames.
    5. Prints a detailed summary report comparing all encoding methods.

    Args:
        reference_frames_dir: Path to the directory with original frames.
        encoded_videos: Dictionary mapping a method name to its video file path.
        metrics: Dictionary mapping a metric name (e.g., "PSNR") to its function.
                 The function must take two np.ndarray blocks as input.
        block_size: The size of blocks for analysis.
        width: The width of the video frames.
        height: The height of the video frames.
        temp_dir: A directory for temporary files (decoded frames, heatmaps).
        masks_dir: Path to the directory with UFO masks.
        sample_frames: A list of frame indices to generate heatmaps for.

    Returns:
        A dictionary containing the aggregated analysis results for each video.
    """

    # helper functions

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
        """Prints a comprehensive summary report from the analysis results."""
        print(f"\n{'='*100}")
        print(f"{'COMPREHENSIVE ANALYSIS SUMMARY':^100}")
        print(f"{'='*100}")

        if not results:
            print("No results to display.")
            return

        # Header
        print(f"{'Method':<25} {'Region':<12} {'SSIM':<12} {'PSNR (dB)':<12} {'Blocks':<12} {'File Size (MB)':<18}")
        print(f"{'-'*100}")

        for video_name, data in results.items():
            fg_data = data['foreground']
            bg_data = data['background']
            
            # Format metric strings
            fg_ssim_str = f"{fg_data['ssim_mean']:.4f} ± {fg_data['ssim_std']:.4f}"
            fg_psnr_str = f"{fg_data['psnr_mean']:.2f} ± {fg_data['psnr_std']:.2f}"
            bg_ssim_str = f"{bg_data['ssim_mean']:.4f} ± {bg_data['ssim_std']:.4f}"
            bg_psnr_str = f"{bg_data['psnr_mean']:.2f} ± {bg_data['psnr_std']:.2f}"
            
            print(f"{video_name:<25} {'Foreground':<12} {fg_ssim_str:<12} {fg_psnr_str:<12} {fg_data['block_count']:<12} {data['file_size_mb']:<18.2f}")
            print(f"{'':<25} {'Background':<12} {bg_ssim_str:<12} {bg_psnr_str:<12} {bg_data['block_count']:<12}")
            print(f"{'-'*100}")

        # Trade-off analysis against the first video as baseline
        if len(results) > 1:
            baseline_name = list(results.keys())[0]
            print(f"\nTRADE-OFF ANALYSIS (vs. {baseline_name}):\n")
            
            for video_name in list(results.keys())[1:]:
                baseline_fg = results[baseline_name]['foreground']['ssim_mean']
                adaptive_fg = results[video_name]['foreground']['ssim_mean']
                baseline_bg = results[baseline_name]['background']['ssim_mean']
                adaptive_bg = results[video_name]['background']['ssim_mean']

                fg_change = ((adaptive_fg / baseline_fg) - 1) * 100 if baseline_fg > 0 else 0
                bg_change = ((adaptive_bg / baseline_bg) - 1) * 100 if baseline_bg > 0 else 0

                print(f"Analysis for '{video_name}':")
                print(f"  - Foreground SSIM change: {fg_change:+.2f}%")
                print(f"  - Background SSIM change: {bg_change:+.2f}%")
                if fg_change > 0 and bg_change < 0:
                    print("  - Verdict: Successful quality trade-off achieved.")
                elif fg_change > 0 and bg_change >= 0:
                    print("  - Verdict: Overall quality improvement.")
                else:
                    print("  - Verdict: Unfavorable or mixed results.")
                print("-" * 40)
    
    # --- Setup ---
    os.makedirs(temp_dir, exist_ok=True)
    decoded_frames_root = os.path.join(temp_dir, "decoded_frames")
    heatmaps_dir = os.path.join(temp_dir, "quality_heatmaps")
    os.makedirs(heatmaps_dir, exist_ok=True)
    
    if not os.path.isdir(masks_dir):
        print(f"Warning: Masks directory not found at '{masks_dir}'. Skipping FG/BG analysis.")
        masks_dir = None
        
    analysis_results = {}

    # --- Main Loop: Process each video ---
    for video_name, video_path in encoded_videos.items():
        print(f"\nProcessing '{video_name}'...")
        if not os.path.exists(video_path):
            print(f"  - Video not found, skipping.")
            continue

        # 1. Decode video
        video_decoded_dir = os.path.join(decoded_frames_root, video_name.replace(" ", "_"))
        if not _decode_video_to_frames(video_path, video_decoded_dir):
            continue
            
        ref_frame_files = sorted([f for f in os.listdir(reference_frames_dir) if f.endswith('.jpg')])
        decoded_frame_files = sorted([f for f in os.listdir(video_decoded_dir) if f.endswith('.jpg')])
        num_frames = min(len(ref_frame_files), len(decoded_frame_files))

        # Storage for all metric values for this video
        # e.g., fg_scores['SSIM'] = [0.9, 0.95, ...], fg_scores['PSNR'] = [34.5, 35.1, ...]
        fg_scores = {name: [] for name in metrics}
        bg_scores = {name: [] for name in metrics}

        # 2. Process each frame
        for i in range(num_frames):
            ref_frame = cv2.imread(os.path.join(reference_frames_dir, ref_frame_files[i]))
            decoded_frame = cv2.imread(os.path.join(video_decoded_dir, decoded_frame_files[i]))
            
            if ref_frame is None or decoded_frame is None: continue

            ref_frame = cv2.resize(ref_frame, (width, height))
            decoded_frame = cv2.resize(decoded_frame, (width, height))
            
            # Split into blocks ONCE per frame
            ref_blocks = split_image_into_blocks(ref_frame, block_size)
            test_blocks = split_image_into_blocks(decoded_frame, block_size)

            # 3. Calculate all requested metrics
            frame_metric_maps = {
                name: calculate_blockwise_metric(ref_blocks, test_blocks, func)
                for name, func in metrics.items()
            }

            # 4. Separate FG/BG scores if masks are available
            if masks_dir:
                mask_path = os.path.join(masks_dir, f"{i+1:05d}.png")
                if os.path.exists(mask_path):
                    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                    mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
                    
                    # Downsample mask to block resolution
                    num_blocks_y, num_blocks_x = list(frame_metric_maps.values())[0].shape
                    block_mask = cv2.resize(mask, (num_blocks_x, num_blocks_y), interpolation=cv2.INTER_NEAREST)
                    
                    fg_mask_bool = block_mask > 128
                    bg_mask_bool = ~fg_mask_bool

                    for name, metric_map in frame_metric_maps.items():
                        if np.any(fg_mask_bool):
                            fg_scores[name].extend(metric_map[fg_mask_bool])
                        if np.any(bg_mask_bool):
                            bg_scores[name].extend(metric_map[bg_mask_bool])
            
            # 5. Generate heatmap for sample frames
            if i in sample_frames:
                print(f"  - Generating heatmap for frame {i+1}...")
                safe_name = video_name.replace(" ", "_").replace("/", "_")
                output_path = os.path.join(heatmaps_dir, f"{safe_name}_frame_{i+1:03d}.png")
                _generate_and_save_heatmap(
                    ref_frame, decoded_frame, mask, frame_metric_maps,
                    video_name, i, output_path, block_size
                )
        
        # 6. Aggregate results for this video
        analysis_results[video_name] = {
            'foreground': {},
            'background': {},
            'file_size_mb': os.path.getsize(video_path) / (1024 * 1024)
        }
        for name in metrics:
            fg_vals = [v for v in fg_scores[name] if np.isfinite(v)]
            bg_vals = [v for v in bg_scores[name] if np.isfinite(v)]
            analysis_results[video_name]['foreground'][f'{name.lower()}_mean'] = np.mean(fg_vals) if fg_vals else 0
            analysis_results[video_name]['foreground'][f'{name.lower()}_std'] = np.std(fg_vals) if fg_vals else 0
            analysis_results[video_name]['background'][f'{name.lower()}_mean'] = np.mean(bg_vals) if bg_vals else 0
            analysis_results[video_name]['background'][f'{name.lower()}_std'] = np.std(bg_vals) if bg_vals else 0
        analysis_results[video_name]['foreground']['block_count'] = len(fg_scores[list(metrics.keys())[0]])
        analysis_results[video_name]['background']['block_count'] = len(bg_scores[list(metrics.keys())[0]])

    # --- Finalization ---
    _print_summary_report(analysis_results)
    shutil.rmtree(decoded_frames_root, ignore_errors=True)
    print(f"\nAnalysis complete. Heatmaps saved to: {heatmaps_dir}")
    
    return analysis_results


def flatten_image_from_blocks(blocks: np.ndarray) -> np.ndarray:
    """
    Reconstructs an image from an array of blocks using vectorization.
    """
    n, m, l, _, c = blocks.shape
    return blocks.transpose(0, 2, 1, 3, 4).reshape(n * l, m * l, c)

def apply_selective_removal(image: np.ndarray, frame_scores: np.ndarray, block_size: int, to_remove: float) -> Tuple[np.ndarray, np.ndarray, List[List[int]]]:
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
        to_remove: The number of blocks to remove per row. If < 1, it's a percentage.

    Returns:
        A tuple containing:
        - new_image: The reconstructed image with blocks removed.
        - removal_mask: A 2D array indicating removed (1) vs. kept (0) blocks.
        - block_coords_to_remove: The list of lists of column indices that were removed.
    """
    num_blocks_y, num_blocks_x = frame_scores.shape
    
    # --- 1. Selection Step ---
    if to_remove < 1.0:
        num_blocks_to_remove = int(to_remove * num_blocks_x)
    else:
        num_blocks_to_remove = int(to_remove)
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
    new_image = flatten_image_from_blocks(kept_blocks)
    
    return new_image, removal_mask, block_coords_to_remove

def apply_adaptive_filtering(image: np.ndarray, frame_scores: np.ndarray, block_size: int, filter_func: Callable[[np.ndarray, float], np.ndarray], max_filter_strength: float = 1.0) -> np.ndarray:
    """
    Applies a variable-strength filter to each block of an image based on scores.

    A high score for a block results in a stronger filter application, simplifying
    the block to save bits during video encoding.

    Args:
        image: The original image for the frame (H, W, C).
        frame_scores: The 2D array of scores for this frame (num_blocks_y, num_blocks_x).
                      Scores are expected to be in the [0, 1] range for best results.
        block_size: The side length (l) of each block block.
        filter_func: A function that takes a block (l, l, C) and a strength 
                     (float) and returns a filtered block.
        max_filter_strength: A multiplier to scale the normalized scores to a
                             range appropriate for the chosen filter_func.

    Returns:
        The new image with adaptive filtering applied to its blocks.
    """
    # Normalize scores to a [0, 1] range to ensure predictable behavior
    normalized_scores = normalize_array(frame_scores)
    
    # Split the image into an array of blocks
    blocks = split_image_into_blocks(image, block_size)
    # Create a copy to store the filtered results
    filtered_blocks = blocks.copy()
    
    num_blocks_y, num_blocks_x, _, _, _ = blocks.shape

    # Iterate over each block to apply the corresponding filter
    for i in range(num_blocks_y):
        for j in range(num_blocks_x):
            score = normalized_scores[i, j]
            
            # Linearly map the normalized score to the filter's strength range
            strength = score * max_filter_strength
            
            # Apply the user-provided filter function if strength is significant
            if strength > 0.1: # Small threshold to avoid unnecessary computation
                block = blocks[i, j]
                filtered_block = filter_func(block, strength)
                filtered_blocks[i, j] = filtered_block

    # Reconstruct the image from the (partially) filtered blocks
    new_image = flatten_image_from_blocks(filtered_blocks)
    
    return new_image

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


if __name__ == "__main__":
    # Example usage parameters
    reference_video = "davis_test/bear.mp4"
    width, height = 640, 368
    block_size = 16
    segment_size = 30  # Number of frames per segment for processing

    # Dictionary to store execution times
    execution_times = {}

    # Calculate appropriate target bitrate based on video characteristics
    cap = cv2.VideoCapture(reference_video)
    framerate = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    target_bitrate = calculate_target_bitrate(width, height, framerate, quality_factor=1.2)

    # SERVER SIDE

    # --- Video Preprocessing ---
    start = time.time()

    # Create experiment directory for all experiment-related files
    experiment_dir = "experiment"
    os.makedirs(experiment_dir, exist_ok=True)

    print(f"Processing video: {reference_video}")
    print(f"Target resolution: {width}x{height}")
    print(f"Calculated target bitrate: {target_bitrate} bps ({target_bitrate/1000000:.1f} Mbps) for {width}x{height}@{framerate:.1f}fps")

    reference_frames_dir = os.path.join(experiment_dir, "reference_frames")
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

    # --- Baseline H.265 Encoding ---
    start = time.time()
    
    print(f"Encoding reference frames with H.265 for baseline comparison...")
    baseline_video_h265 = os.path.join(experiment_dir, "baseline.mp4")

    baseline_cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "warning",
        "-framerate", str(framerate),
        "-i", f"{reference_frames_dir}/%05d.jpg",
        "-c:v", "libx265",
        "-b:v", str(target_bitrate),
        "-minrate", str(int(target_bitrate * 0.9)),
        "-maxrate", str(int(target_bitrate * 1.1)),
        "-bufsize", str(target_bitrate),
        "-preset", "medium",
        "-g", "1",
        "-y", baseline_video_h265
    ]

    result = subprocess.run(baseline_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Baseline encoding failed: {result.stderr}")
        raise RuntimeError(f"Baseline encoding failed: {result.stderr}")

    end = time.time()
    execution_times["baseline_encoding"] = end - start
    print(f"Baseline encoding completed in {end - start:.2f} seconds.\n")

    # --- Adaptive ROI H.265 Encoding ---
    start = time.time()

    print(f"Encoding frames with ROI-based adaptive quantization...")
    adaptive_video_h265 = os.path.join(experiment_dir, "adaptive.mp4")

    encode_with_roi(
        input_frames_dir=reference_frames_dir,
        output_video=adaptive_video_h265,
        removability_scores=removability_scores,
        block_size=block_size,
        framerate=framerate,
        width=width,
        height=height,
        segment_size=segment_size,
        target_bitrate=target_bitrate
    )

    end = time.time()
    execution_times["adaptive_encoding"] = end - start
    print(f"Adaptive encoding completed in {end - start:.2f} seconds.\n")

    # --- ELVIS v1 (zero information removal and inpainting) ---


    # --- Performance Analysis ---
    start = time.time()
    
    # Compare file sizes and quality metrics
    baseline_size = os.path.getsize(baseline_video_h265)
    adaptive_size = os.path.getsize(adaptive_video_h265)

    print(f"\nEncoding Results (Target Bitrate: {target_bitrate} bps / {target_bitrate/1000000:.1f} Mbps):")
    print(f"Baseline H.265 video size: {baseline_size / 1024 / 1024:.2f} MB")
    print(f"Adaptive ROI video size: {adaptive_size / 1024 / 1024:.2f} MB")
    print(f"Size ratio: {adaptive_size / baseline_size:.2f}x")
    print(f"Size difference: {(adaptive_size - baseline_size) / baseline_size * 100:+.1f}%")

    encoded_videos = {
        "Baseline H.265": baseline_video_h265,
        "Adaptive ROI": adaptive_video_h265
    }

    quality_metrics = {
        "PSNR": psnr,
        "SSIM": msssim
    }

    ufo_masks_dir = os.path.join(experiment_dir, "UFO_masks")
    
    frame_count = len(os.listdir(reference_frames_dir))
    sample_frames = [0, frame_count // 4, frame_count // 2, 3 * frame_count // 4, frame_count - 1]
    sample_frames = [f for f in sample_frames if f < frame_count]

    analysis_results = analyze_encoding_performance(
        reference_frames_dir=reference_frames_dir,
        encoded_videos=encoded_videos,
        metrics=quality_metrics,
        block_size=block_size,
        width=width,
        height=height,
        temp_dir=experiment_dir,
        masks_dir=ufo_masks_dir,
        sample_frames=sample_frames
    )
    
    # Add collected times to the results dictionary
    analysis_results["execution_times_seconds"] = execution_times

    # Add video parameters to the results dictionary
    analysis_results["video_length_seconds"] = frame_count / framerate
    analysis_results["video_framerate"] = framerate
    analysis_results["video_resolution"] = f"{width}x{height}"
    analysis_results["block_size"] = block_size
    analysis_results["segment_size_frames"] = segment_size
    analysis_results["target_bitrate_bps"] = target_bitrate
    
    # Save analysis results to a JSON file
    results_json_path = os.path.join(experiment_dir, "analysis_results.json")
    with open(results_json_path, 'w') as f:
        json.dump(analysis_results, f, indent=4)
    print(f"Analysis results saved to: {results_json_path}")