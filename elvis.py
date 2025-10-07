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
        containing the final removability scores normalized to [0, 1].
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

        # Final normalization to [0, 1]
        removability_scores = normalize_array(removability_scores)

        return removability_scores
    
    except Exception as e:
        print(f"Error in calculate_removability_scores: {e}")
        raise
    finally:
        # Always return to original directory
        os.chdir(original_dir)

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
            
            # Create a temporary file for the filter script
            roi_script_path = os.path.join(segments_dir, f"roi_script_{segment_idx:03d}.txt")
            if roi_filter_string:
                with open(roi_script_path, 'w') as f:
                    f.write(roi_filter_string)
            
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
            
            # Use -filter_complex_script to read the filter graph from the file.
            if roi_filter_string:
                cmd.extend(["-filter_complex_script", roi_script_path])
            
            cmd.extend([
                "-c:v", "libx265",
                "-b:v", str(target_bitrate),
                "-minrate", str(int(target_bitrate * 0.9)),
                "-maxrate", str(int(target_bitrate * 1.1)),
                "-bufsize", str(target_bitrate),
                "-preset", "medium",
                "-g", "10",
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
        
        try:
            subprocess.run(concat_cmd, capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Concatenation failed: {e.stderr}") from e
            
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

def analyze_encoding_performance(reference_frames: List[np.ndarray], encoded_videos: Dict[str, str], metrics: Dict[str, Callable], block_size: int, width: int, height: int, temp_dir: str, masks_dir: str, sample_frames: List[int] = [0, 20, 40], video_bitrates: Dict[str, float] = {}) -> Dict:
    """
    A comprehensive function to analyze and compare video encoding performance.

    This function:
    1. Decodes videos to frames.
    2. For each frame, calculates multiple block-wise quality metrics.
    3. Separates metrics for foreground and background regions using masks.
    4. Generates and saves quality heatmaps for sample frames.
    5. Prints a detailed summary report comparing all encoding methods.

    Args:
        reference_frames: List of original reference frames.
        encoded_videos: Dictionary mapping a method name to its video file path.
        metrics: Dictionary mapping a metric name (e.g., "PSNR") to its function.
                 The function must take two np.ndarray blocks as input.
        block_size: The size of blocks for analysis.
        width: The width of the video frames.
        height: The height of the video frames.
        temp_dir: A directory for temporary files (decoded frames, heatmaps).
        masks_dir: Path to the directory with UFO masks.
        sample_frames: A list of frame indices to generate heatmaps for.
        video_bitrates: Dictionary mapping video name to bitrate in bps.

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
        print(f"{'Method':<25} {'Region':<12} {'SSIM':<12} {'PSNR (dB)':<12} {'Blocks':<12} {'Bitrate (Mbps)':<18}")
        print(f"{'-'*100}")

        for video_name, data in results.items():
            fg_data = data['foreground']
            bg_data = data['background']
            
            # Format metric strings
            fg_ssim_str = f"{fg_data['ssim_mean']:.4f} ± {fg_data['ssim_std']:.4f}"
            fg_psnr_str = f"{fg_data['psnr_mean']:.2f} ± {fg_data['psnr_std']:.2f}"
            bg_ssim_str = f"{bg_data['ssim_mean']:.4f} ± {bg_data['ssim_std']:.4f}"
            bg_psnr_str = f"{bg_data['psnr_mean']:.2f} ± {bg_data['psnr_std']:.2f}"
            
            print(f"{video_name:<25} {'Foreground':<12} {fg_ssim_str:<12} {fg_psnr_str:<12} {fg_data['block_count']:<12} {data['bitrate_mbps']:<18.2f}")
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
            
        decoded_frame_files = sorted([f for f in os.listdir(video_decoded_dir) if f.endswith('.jpg')])
        num_frames = min(len(reference_frames), len(decoded_frame_files))

        # Storage for all metric values for this video
        # e.g., fg_scores['SSIM'] = [0.9, 0.95, ...], fg_scores['PSNR'] = [34.5, 35.1, ...]
        fg_scores = {name: [] for name in metrics}
        bg_scores = {name: [] for name in metrics}

        # 2. Process each frame
        for i in range(num_frames):
            ref_frame = reference_frames[i]
            decoded_frame = cv2.imread(os.path.join(video_decoded_dir, decoded_frame_files[i]))
            
            if ref_frame is None or decoded_frame is None: continue

            ref_frame = cv2.resize(ref_frame, (width, height))
            decoded_frame = cv2.resize(decoded_frame, (width, height))
            
            # Split into blocks ONCE per frame
            ref_blocks = split_image_into_blocks(ref_frame, block_size)
            test_blocks = split_image_into_blocks(decoded_frame, block_size)

            # Calculate win_size ONCE before processing all blocks for the frame.
            min_dim = block_size
            win_size = min(7, min_dim)
            if win_size % 2 == 0:
                win_size -= 1
            win_size = max(3, win_size)

            metrics_to_run = {}
            if 'PSNR' in metrics:
                metrics_to_run['PSNR'] = psnr # psnr has no extra params, so it's fine
            if 'SSIM' in metrics:
                # Create a new function that is ssim() with win_size already set
                ssim_with_win_size = functools.partial(
                    ssim, 
                    gaussian_weights=True, 
                    data_range=255, 
                    channel_axis=-1, 
                    win_size=win_size
                )
                metrics_to_run['SSIM'] = ssim_with_win_size

            # 3. Calculate all requested metrics
            frame_metric_maps = {
                name: calculate_blockwise_metric(ref_blocks, test_blocks, func)
                for name, func in metrics_to_run.items()
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
            'bitrate_mbps': video_bitrates[video_name] / 1000000 if video_name in video_bitrates else 0
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

def map_strength_linear(normalized_scores: np.ndarray, max_strength: float = 1.0) -> np.ndarray:
    """
    Linearly maps normalized scores [0, 1] to a strength range [0, max_strength].
    This is suitable for filters where strength is a simple multiplier (e.g., Gaussian blur's sigma).
    """
    return normalized_scores * max_strength

def map_strength_dct_cutoff(normalized_scores: np.ndarray, block_size: int, max_cutoff_reduction: float = 0.8) -> np.ndarray:
    """
    Maps normalized scores [0, 1] to a DCT high-frequency cutoff strength.
    
    The strength represents how many high-frequency components to cut.
    A score of 1.0 means cutting max_cutoff_reduction * block_size coefficients.
    e.g., max_cutoff_reduction=0.8 means up to 80% of the block size is cut.
    """
    # Max number of coefficients to cut (must be < block_size)
    max_cut = block_size * max_cutoff_reduction
    
    # Scale scores to the cutoff range
    strength = normalized_scores * max_cut
    
    # Ensure strength is at least 1 for any significant score
    return np.maximum(strength, 0.0).astype(np.int32)

def map_strength_downsampling_factor(normalized_scores: np.ndarray, max_factor: int = 8) -> np.ndarray:
    """
    Maps normalized scores [0, 1] to a downsampling factor that is a power of 2: 
    [1, 2, 4, ..., max_factor].

    Args:
        normalized_scores: The 2D array of normalized scores [0, 1].
        max_factor: The maximum allowed downsampling factor (must be a power of 2).

    Returns:
        The 2D array of integer downsampling factors (powers of 2).
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
    # Example usage parameters
    reference_video = "davis_test/bear.mp4"
    width, height = 640, 360
    block_size = 8
    segment_size = 10  # Number of frames per segment for processing
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



    # --- Baseline  Encoding ---
    start = time.time()
    
    print(f"Encoding reference frames with  for baseline comparison...")
    baseline_video = os.path.join(experiment_dir, "baseline.mp4")

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
        "-g", "10",
        "-y", baseline_video
    ]
    result = subprocess.run(baseline_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Baseline encoding failed: {result.stderr}")
        raise RuntimeError(f"Baseline encoding failed: {result.stderr}")

    end = time.time()
    execution_times["baseline_encoding"] = end - start
    print(f"Baseline encoding completed in {end - start:.2f} seconds.\n")



    # --- Adaptive ROI Encoding ---
    start = time.time()
    print(f"Encoding frames with ROI-based adaptive quantization...")
    adaptive_video = os.path.join(experiment_dir, "adaptive.mp4")
    encode_with_roi(
        input_frames_dir=reference_frames_dir,
        output_video=adaptive_video,
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

    # Encode the shrunk frames
    shrunk_video = os.path.join(experiment_dir, "shrunk.mp4")
    shrunk_cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "warning",
        "-framerate", str(framerate),
        "-i", f"{shrunk_frames_dir}/%05d.jpg",
        "-c:v", "libx265",
        "-b:v", str(target_bitrate),
        "-minrate", str(int(target_bitrate * 0.9)),
        "-maxrate", str(int(target_bitrate * 1.1)),
        "-bufsize", str(target_bitrate),
        "-preset", "medium",
        "-g", "10",
        "-y", shrunk_video
    ]
    result = subprocess.run(shrunk_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Shrunk encoding failed: {result.stderr}")
        raise RuntimeError(f"Shrunk encoding failed: {result.stderr}")

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
    dct_strengths = map_strength_dct_cutoff(removability_scores, block_size, max_cutoff_reduction=0.99)
    dct_filtered_frames = [apply_adaptive_filtering(img, strengths, block_size,
                                                   filter_func=apply_dct_damping,
                                                   min_strength_threshold=1.0)
                           for img, strengths in zip(reference_frames, dct_strengths)]
    for i, frame in enumerate(dct_filtered_frames):
        cv2.imwrite(os.path.join(dct_filtered_frames_dir, f"{i+1:05d}.jpg"), frame)

    dct_filtered_video = os.path.join(experiment_dir, "dct_filtered.mp4")
    dct_cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "warning",
        "-framerate", str(framerate),
        "-i", f"{dct_filtered_frames_dir}/%05d.jpg",
        "-c:v", "libx265",
        "-b:v", str(target_bitrate),
        "-minrate", str(int(target_bitrate * 0.9)),
        "-maxrate", str(int(target_bitrate * 1.1)),
        "-bufsize", str(target_bitrate),
        "-preset", "medium",
        "-g", "10",
        "-y", dct_filtered_video
    ]
    result = subprocess.run(dct_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"DCT filtered encoding failed: {result.stderr}")
        raise RuntimeError(f"DCT filtered encoding failed: {result.stderr}")

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
    downsample_strengths = map_strength_downsampling_factor(removability_scores, max_factor=8)
    downsampled_frames = [apply_adaptive_filtering(img, strengths, block_size,
                                                  filter_func=apply_downsampling,
                                                  min_strength_threshold=1.0)
                         for img, strengths in zip(reference_frames, downsample_strengths)]
    for i, frame in enumerate(downsampled_frames):
        cv2.imwrite(os.path.join(downsampled_frames_dir, f"{i+1:05d}.jpg"), frame)

    downsampled_video = os.path.join(experiment_dir, "downsampled.mp4")
    downsampled_cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "warning",
        "-framerate", str(framerate),
        "-i", f"{downsampled_frames_dir}/%05d.jpg",
        "-c:v", "libx265",
        "-b:v", str(target_bitrate),
        "-minrate", str(int(target_bitrate * 0.9)),
        "-maxrate", str(int(target_bitrate * 1.1)),
        "-bufsize", str(target_bitrate),
        "-preset", "medium",
        "-g", "10",
        "-y", downsampled_video
    ]
    result = subprocess.run(downsampled_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Downsampling encoding failed: {result.stderr}")
        raise RuntimeError(f"Downsampling encoding failed: {result.stderr}")

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
    blur_strengths = map_strength_linear(removability_scores, max_strength=3.0)
    blurred_frames = [apply_adaptive_filtering(img, strengths, block_size,
                                             filter_func=apply_gaussian_blur,
                                             min_strength_threshold=0.1)
                      for img, strengths in zip(reference_frames, blur_strengths)]
    for i, frame in enumerate(blurred_frames):
        cv2.imwrite(os.path.join(blurred_frames_dir, f"{i+1:05d}.jpg"), frame)
    blurred_video = os.path.join(experiment_dir, "blurred.mp4")
    blur_cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "warning",
        "-framerate", str(framerate),
        "-i", f"{blurred_frames_dir}/%05d.jpg",
        "-c:v", "libx265",
        "-b:v", str(target_bitrate),
        "-minrate", str(int(target_bitrate * 0.9)),
        "-maxrate", str(int(target_bitrate * 1.1)),
        "-bufsize", str(target_bitrate),
        "-preset", "medium",
        "-g", "10",
        "-y", blurred_video
    ]
    result = subprocess.run(blur_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Gaussian blur encoding failed: {result.stderr}")
        raise RuntimeError(f"Gaussian blur encoding failed: {result.stderr}")

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
    removal_masks_dir = os.path.join(experiment_dir, "removal_masks")
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
    stretch_cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "warning",
        "-framerate", str(framerate),
        "-i", f"{stretched_frames_dir}/%05d.jpg",
        "-c:v", "libx265",
        "-preset", "veryslow",
        "-x265-params", "lossless=1",
        "-y", stretched_video
    ]
    result = subprocess.run(stretch_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ELVIS v1 stretching failed: {result.stderr}")
        raise RuntimeError(f"ELVIS v1 stretching failed: {result.stderr}")

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
    inpaint_cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "warning",
        "-framerate", str(framerate),
        "-i", f"{inpainted_frames_dir}/%05d.jpg",
        "-c:v", "libx265",
        "-preset", "veryslow",
        "-x265-params", "lossless=1",
        "-y", inpainted_video
    ]
    result = subprocess.run(inpaint_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Inpainted encoding failed: {result.stderr}")
        raise RuntimeError(f"Inpainted encoding failed: {result.stderr}")



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
    dct_restore_cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "warning",
        "-framerate", str(framerate),
        "-i", f"{dct_restored_frames_dir}/%05d.jpg",
        "-c:v", "libx265",
        "-preset", "veryslow",
        "-x265-params", "lossless=1",
        "-y", dct_restored_video
    ]
    result = subprocess.run(dct_restore_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"DCT restored encoding failed: {result.stderr}")
        raise RuntimeError(f"DCT restored encoding failed: {result.stderr}")

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
    downsample_restore_cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "warning",
        "-framerate", str(framerate),
        "-i", f"{downsampled_restored_frames_dir}/%05d.jpg",
        "-c:v", "libx265",
        "-preset", "veryslow",
        "-x265-params", "lossless=1",
        "-y", downsampled_restored_video
    ]
    result = subprocess.run(downsample_restore_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Downsampled restored encoding failed: {result.stderr}")
        raise RuntimeError(f"Downsampled restored encoding failed: {result.stderr}")

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
    blur_restore_cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "warning",
        "-framerate", str(framerate),
        "-i", f"{blurred_restored_frames_dir}/%05d.jpg",
        "-c:v", "libx265",
        "-preset", "veryslow",
        "-x265-params", "lossless=1",
        "-y", blurred_restored_video
    ]
    result = subprocess.run(blur_restore_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Blurred restored encoding failed: {result.stderr}")
        raise RuntimeError(f"Blurred restored encoding failed: {result.stderr}")

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

    quality_metrics = {
        "PSNR": psnr,
        "SSIM": ssim
    }

    ufo_masks_dir = os.path.join(experiment_dir, "UFO_masks")
    
    frame_count = len(os.listdir(reference_frames_dir))
    sample_frames = [frame_count // 2] # [0, frame_count // 4, frame_count // 2, 3 * frame_count // 4, frame_count - 1]
    sample_frames = [f for f in sample_frames if f < frame_count]

    analysis_results = analyze_encoding_performance(
        reference_frames=reference_frames,
        encoded_videos=encoded_videos,
        metrics=quality_metrics,
        block_size=block_size,
        width=width,
        height=height,
        temp_dir=experiment_dir,
        masks_dir=ufo_masks_dir,
        sample_frames=sample_frames,
        video_bitrates=bitrates
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