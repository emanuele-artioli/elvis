import time
import shutil
import os
import subprocess
from pathlib import Path
import cv2
import numpy as np
from typing import List, Callable, Tuple

# Utility functions

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

def calculate_removability_scores(raw_video_file: str, reference_frames_folder: str, width: int, height: int, square_size: int, alpha: float = 0.5, working_dir: str = ".", smoothing_beta: float = 1) -> np.ndarray:
    """
    This function computes a "removability score" by running EVCA for complexity analysis
    and UFO for object detection, then combining the results. Higher scores mean the block is a
    better candidate for removal (e.g., background, low complexity).

    Args:
        raw_video_file: Path to the raw YUV video file.
        reference_frames_folder: Path to the folder containing reference frames.
        width: The width of the original video frame.
        height: The height of the original video frame.
        square_size: The size of each block in pixels.
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
        os.chdir(os.path.dirname(original_dir))  # Go to parent directory where EVCA is located
        evca_cmd = f"python EVCA/main.py -i {raw_video_abs} -r {width}x{height} -b {square_size} -f {frame_count} -c {evca_csv_path} -bi 1"
        result = subprocess.run(evca_cmd, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"EVCA command failed: {result.stderr}")
            print(f"EVCA stdout: {result.stdout}")
            raise RuntimeError(f"EVCA execution failed: {result.stderr}")
        else:
            print("EVCA completed successfully")
        
        # Calculate ROI with UFO
        print("Running UFO for object detection...")
        UFO_folder = "UFO/datasets/elvis/image/reference_frames"
        if os.path.exists(UFO_folder):
            shutil.rmtree(UFO_folder)
        shutil.copytree(reference_frames_abs, UFO_folder)
        
        os.chdir("UFO")
        ufo_cmd = "python test.py --model='weights/video_best.pth' --data_path='datasets/elvis' --output_dir='VSOD_results/wo_optical_flow/elvis' --task='VSOD'"
        result = subprocess.run(ufo_cmd, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"UFO command failed: {result.stderr}")
            raise RuntimeError(f"UFO execution failed: {result.stderr}")
        
        # Move UFO masks to working directory
        mask_source = "VSOD_results/wo_optical_flow/elvis/reference_frames"
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

        num_blocks_x = width // square_size
        num_blocks_y = height // square_size
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

def calculate_frame_uniformity_scores(removability_scores: np.ndarray) -> np.ndarray:
    """
    Aggregates block-level removability scores to a single "uniformity score" per frame.
    This score is higher for frames that have a high average removability score AND
    low variance, indicating the frame is uniformly simple.

    Args:
        removability_scores: 3D NumPy array of scores from your script.

    Returns:
        A 1D NumPy array where each element is the uniformity score for that frame.
    """
    if removability_scores.ndim != 3:
        raise ValueError("Input must be a 3D array of shape (frames, height, width)")

    # Calculate mean and variance across all blocks for each frame
    mean_scores = np.mean(removability_scores, axis=(1, 2))
    variance_scores = np.var(removability_scores, axis=(1, 2))

    # Normalize both to prevent scale issues
    normalized_mean = normalize_array(mean_scores)
    normalized_variance = normalize_array(variance_scores)

    # Combine them. We want high mean and low variance.
    # Add a small epsilon to the denominator to avoid division by zero.
    epsilon = 1e-6
    uniformity_scores = normalized_mean / (normalized_variance + epsilon)
    
    # Final normalization of the combined score
    return normalize_array(uniformity_scores)

def select_frames_to_drop(removability_scores: np.ndarray, drop_ratio: float = 0.1, validate_with_optical_flow: bool = True, video_path: str = None, motion_threshold: float = 2.0) -> list[int]:
    """
    Selects frames to drop using a "uniformity score" (mean/variance) and an
    optional optical flow validation step.

    Args:
        removability_scores: Your 3D array of per-block scores.
        drop_ratio: The target fraction of frames to drop.
        validate_with_optical_flow: If True, uses optical flow as a final check.
        video_path: Path to the video file (required if validation is enabled).
        motion_threshold: Max allowed motion magnitude for a drop to be confirmed.

    Returns:
        A final list of frame indices to be dropped.
    """
    if validate_with_optical_flow and not video_path:
        raise ValueError("video_path must be provided if validate_with_optical_flow is True.")

    # 1. Calculate our new uniformity score for each frame
    print("Calculating frame uniformity scores (mean/variance)...")
    frame_scores = calculate_frame_uniformity_scores(removability_scores)
    
    # We assume scene changes are already handled, so we don't mark any frames as forbidden.

    # 2. Get initial candidates based on the uniformity score
    num_frames = len(frame_scores)
    num_to_drop_target = int(num_frames * drop_ratio)
    
    # Get indices sorted by score (high score = good candidate)
    sorted_indices = np.argsort(frame_scores)[::-1]
    
    # 3. Select and (optionally) validate candidates
    final_dropped_indices = []
    dropped_flags = np.zeros(num_frames, dtype=bool)

    print("Selecting best candidate frames...")
    for idx in sorted_indices:
        if len(final_dropped_indices) >= num_to_drop_target:
            break

        # Boundary and consecutive-frame checks are essential
        if idx == 0 or idx == num_frames - 1 or dropped_flags[idx-1] or dropped_flags[idx+1]:
            continue
        
        # --- Validation Step ---
        is_candidate_valid = True
        if validate_with_optical_flow:
            motion_magnitude = calculate_motion_coherence(video_path, idx - 1, idx + 1)
            if motion_magnitude >= motion_threshold:
                is_candidate_valid = False # Motion is too complex, reject candidate
        
        if is_candidate_valid:
            final_dropped_indices.append(idx)
            dropped_flags[idx] = True
    
    print(f"Targeted {num_to_drop_target} frames to drop.")
    if validate_with_optical_flow:
        print(f"Confirmed {len(final_dropped_indices)} frames after motion validation.")
    else:
        print(f"Selected {len(final_dropped_indices)} frames based on uniformity score alone.")
          
    return sorted(final_dropped_indices)

def split_image_into_squares(image: np.ndarray, l: int) -> np.ndarray:
    """
    Splits an image into squares of a specific size using vectorization.
    """
    n, m, c = image.shape
    num_rows = n // l
    num_cols = m // l
    return image.reshape(num_rows, l, num_cols, l, c).transpose(0, 2, 1, 3, 4)

def flatten_image_from_squares(squares: np.ndarray) -> np.ndarray:
    """
    Reconstructs an image from an array of squares using vectorization.
    """
    n, m, l, _, c = squares.shape
    return squares.transpose(0, 2, 1, 3, 4).reshape(n * l, m * l, c)

def apply_selective_removal(image: np.ndarray, frame_scores: np.ndarray, square_size: int, to_remove: float) -> Tuple[np.ndarray, np.ndarray, List[List[int]]]:
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
        square_size: The side length (l) of each square block.
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
    squares = split_image_into_squares(image, square_size)
    
    # Create a 2D mask of which blocks to remove
    removal_mask = np.zeros((num_blocks_y, num_blocks_x), dtype=np.int8)
    rows_indices = np.arange(num_blocks_y).repeat([len(cols) for cols in block_coords_to_remove])
    if len(rows_indices) > 0:
        cols_indices = np.concatenate(block_coords_to_remove)
        removal_mask[rows_indices, cols_indices] = 1

    # Filter out the blocks marked for removal
    kept_squares_list = [
        squares[i, np.where(removal_mask[i] == 0)[0]]
        for i in range(num_blocks_y)
    ]
    kept_squares = np.stack(kept_squares_list, axis=0)

    # Reconstruct the final image from the remaining blocks
    new_image = flatten_image_from_squares(kept_squares)
    
    return new_image, removal_mask, block_coords_to_remove

def apply_adaptive_filtering(image: np.ndarray, frame_scores: np.ndarray, square_size: int, filter_func: Callable[[np.ndarray, float], np.ndarray], max_filter_strength: float = 1.0) -> np.ndarray:
    """
    Applies a variable-strength filter to each block of an image based on scores.

    A high score for a block results in a stronger filter application, simplifying
    the block to save bits during video encoding.

    Args:
        image: The original image for the frame (H, W, C).
        frame_scores: The 2D array of scores for this frame (num_blocks_y, num_blocks_x).
                      Scores are expected to be in the [0, 1] range for best results.
        square_size: The side length (l) of each square block.
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
    squares = split_image_into_squares(image, square_size)
    # Create a copy to store the filtered results
    filtered_squares = squares.copy()
    
    num_blocks_y, num_blocks_x, _, _, _ = squares.shape

    # Iterate over each block to apply the corresponding filter
    for i in range(num_blocks_y):
        for j in range(num_blocks_x):
            score = normalized_scores[i, j]
            
            # Linearly map the normalized score to the filter's strength range
            strength = score * max_filter_strength
            
            # Apply the user-provided filter function if strength is significant
            if strength > 0.1: # Small threshold to avoid unnecessary computation
                block = squares[i, j]
                filtered_block = filter_func(block, strength)
                filtered_squares[i, j] = filtered_block

    # Reconstruct the image from the (partially) filtered blocks
    new_image = flatten_image_from_squares(filtered_squares)
    
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
        # This zeros out a bottom-right square of high-frequency coefficients
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

def generate_qp_map_from_scores(removability_scores: np.ndarray, square_size: int, width: int, height: int, base_qp: int = 23, qp_range: int = 20) -> np.ndarray:
    """
    Generate QP maps for x264/x265 encoding based on removability scores.
    
    Args:
        removability_scores: 3D array of shape (num_frames, num_blocks_y, num_blocks_x)
        square_size: Size of each block in pixels
        width: Video width
        height: Video height
        base_qp: Base QP value (default: 23)
        qp_range: Range of QP adjustment (default: 20)
        
    Returns:
        4D numpy array of shape (num_frames, height, width) containing QP values for each pixel
    """
    num_frames, num_blocks_y, num_blocks_x = removability_scores.shape
    
    # Create QP maps at full resolution
    qp_maps = np.zeros((num_frames, height, width), dtype=np.float32)
    
    for frame_idx in range(num_frames):
        frame_scores = removability_scores[frame_idx]
        
        # Normalize scores to [0, 1] range
        normalized_scores = normalize_array(frame_scores)
        
        # Convert scores to QP values
        # Higher removability score = higher QP (lower quality)
        # Lower removability score = lower QP (higher quality)
        qp_values = base_qp + (normalized_scores * qp_range)
        
        # Clip QP values to valid range [0, 51] for x264/x265
        qp_values = np.clip(qp_values, 0, 51)
        
        # Map block-level QP values to pixel-level
        for block_y in range(num_blocks_y):
            for block_x in range(num_blocks_x):
                y1 = block_y * square_size
                x1 = block_x * square_size
                y2 = min(y1 + square_size, height)
                x2 = min(x1 + square_size, width)
                
                qp_maps[frame_idx, y1:y2, x1:x2] = qp_values[block_y, block_x]
    
    return qp_maps

def save_qp_maps_as_files(qp_maps: np.ndarray, output_dir: str) -> List[str]:
    """
    Save QP maps as binary files that can be read by x264/x265.
    
    Args:
        qp_maps: 3D array of shape (num_frames, height, width)
        output_dir: Directory to save QP map files
        
    Returns:
        List of QP map file paths
    """
    os.makedirs(output_dir, exist_ok=True)
    
    qp_map_files = []
    num_frames, height, width = qp_maps.shape
    
    for frame_idx in range(num_frames):
        # Save as binary file (float32 format)
        qp_file = os.path.join(output_dir, f"qp_map_{frame_idx:05d}.qpmap")
        qp_maps[frame_idx].astype(np.float32).tofile(qp_file)
        qp_map_files.append(qp_file)
    
    return qp_map_files

def generate_roi_filter_string_from_scores(frame_scores: np.ndarray, square_size: int, width: int, height: int) -> str:
    """
    Generate addroi filter string from 2D frame scores.
    
    Args:
        frame_scores: 2D array of scores for a single frame
        square_size: Size of each block in pixels
        width: Video width
        height: Video height
        
    Returns:
        String containing chained addroi filters
    """
    num_blocks_y, num_blocks_x = frame_scores.shape
    
    # Normalize scores to [0, 1] range
    normalized_scores = normalize_array(frame_scores)
    
    # Create ROI filter string
    roi_filters = []
    
    for block_y in range(num_blocks_y):
        for block_x in range(num_blocks_x):
            score = normalized_scores[block_y, block_x]
            
            # Calculate block boundaries
            x1 = block_x * square_size
            y1 = block_y * square_size
            x2 = min(x1 + square_size, width)
            y2 = min(y1 + square_size, height)
            
            # Convert removability score to quantization offset
            # High removability (background) -> positive offset (lower quality)
            # Low removability (important) -> negative offset (higher quality)
            qoffset = (score - 0.5) * 2.0  # Maps [0,1] to [-1,1]
            
            # Only add ROI if the offset is significant
            if abs(qoffset) > 0.1:
                roi_filter = f"addroi=x={x1}:y={y1}:w={x2-x1}:h={y2-y1}:qoffset={qoffset:.2f}"
                roi_filters.append(roi_filter)
    
    # Chain multiple addroi filters
    if roi_filters:
        return ",".join(roi_filters)
    else:
        return ""

def encode_with_h265_roi(input_frames_dir: str, output_video: str, removability_scores: np.ndarray, square_size: int, framerate: float, width: int, height: int, segment_size: int = 30, target_bitrate: int = 1000000) -> None:
    """
    Encode video with H.265 using per-frame ROI-based quantization.
    
    Args:
        input_frames_dir: Directory containing input frames
        output_video: Output video file path
        removability_scores: 3D array of removability scores
        square_size: Size of blocks used for scoring
        framerate: Video framerate
        width: Video width
        height: Video height
        segment_size: Number of frames per segment for processing
        target_bitrate: Target bitrate in bits per second (default: 1000000 = 1 Mbps)
    """
    print("Encoding with H.265 using per-frame ROI-based quantization...")
    
    num_frames = removability_scores.shape[0]
    temp_dir = os.path.dirname(output_video)
    segments_dir = os.path.join(temp_dir, "segments")
    os.makedirs(segments_dir, exist_ok=True)
    
    # Process frames in segments for better efficiency
    segment_files = []
    
    try:
        for segment_start in range(0, num_frames, segment_size):
            segment_end = min(segment_start + segment_size, num_frames)
            segment_idx = segment_start // segment_size
            
            print(f"Processing segment {segment_idx + 1} (frames {segment_start}-{segment_end-1})")
            
            # Calculate average ROI for this segment
            segment_scores = removability_scores[segment_start:segment_end]
            avg_segment_scores = np.mean(segment_scores, axis=0)
            
            # Generate ROI filter for this segment
            roi_filter = generate_roi_filter_string_from_scores(avg_segment_scores, square_size, width, height)
            
            # Create segment video file
            segment_file = os.path.join(segments_dir, f"segment_{segment_idx:03d}.mp4")
            
            # Build FFmpeg command for this segment
            cmd = [
                "ffmpeg", "-hide_banner", "-loglevel", "warning",
                "-start_number", str(segment_start + 1),  # ffmpeg frames are 1-indexed
                "-framerate", str(framerate),
                "-i", f"{input_frames_dir}/%05d.jpg",
                "-frames:v", str(segment_end - segment_start),
            ]
            
            # Add ROI filter if available
            if roi_filter:
                cmd.extend(["-vf", roi_filter])
            
            cmd.extend([
                "-c:v", "libx265",
                "-b:v", str(target_bitrate),
                "-minrate", str(int(target_bitrate * 0.9)),
                "-maxrate", str(int(target_bitrate * 1.1)),
                "-bufsize", str(target_bitrate),  # Small buffer for strict bitrate control
                "-preset", "medium",
                "-g", "1",  # Shorter GOP for segments
                "-y", segment_file
            ])
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"libx265 failed for segment: {result.stderr}")
                raise RuntimeError(f"libx265 failed for segment: {result.stderr}")
            
            segment_files.append(segment_file)
        
        # Concatenate all segments into final video
        print(f"Concatenating {len(segment_files)} segments into final video...")
        concat_segments(segment_files, output_video)
        
        print("Per-frame ROI encoding completed successfully")
        
    finally:
        # Clean up segment files
        shutil.rmtree(segments_dir, ignore_errors=True)

def calculate_video_quality_metrics(reference_frames_dir: str, encoded_video: str, framerate: float, width: int, height: int, temp_dir: str) -> Tuple[float, float]:
    """
    Calculate PSNR and SSIM between reference frames and encoded video using FFmpeg.
    
    Args:
        reference_frames_dir: Directory containing reference frames
        encoded_video: Path to encoded video file
        framerate: Video framerate
        width: Video width
        height: Video height
        temp_dir: Temporary directory for intermediate files
        
    Returns:
        Tuple of (average_psnr, average_ssim)
    """
    print(f"Calculating quality metrics for {os.path.basename(encoded_video)}...")
    
    # Create lossless reference video from frames for comparison
    reference_video_lossless = os.path.join(temp_dir, "reference_lossless.mp4")
    
    try:
        # Create lossless reference video if it doesn't exist
        if not os.path.exists(reference_video_lossless):
            print("Creating lossless reference video for quality comparison...")
            ref_cmd = [
                "ffmpeg", "-hide_banner", "-loglevel", "warning",
                "-framerate", str(framerate),
                "-i", f"{reference_frames_dir}/%05d.jpg",
                "-c:v", "libx265",
                "-crf", "0",        # Lossless
                "-pix_fmt", "yuv420p",
                "-y", reference_video_lossless
            ]
            
            result = subprocess.run(ref_cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"Failed to create lossless reference video: {result.stderr}")
                return 0.0, 0.0
        
        # Calculate PSNR using FFmpeg with more verbose output
        psnr_cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "info",
            "-i", reference_video_lossless,
            "-i", encoded_video,
            "-lavfi", "psnr",
            "-f", "null", "-"
        ]
        
        psnr_result = subprocess.run(psnr_cmd, capture_output=True, text=True)
        if psnr_result.returncode != 0:
            print(f"PSNR calculation failed: {psnr_result.stderr}")
            psnr_avg = 0.0
        else:
            # Parse PSNR from stderr output
            psnr_avg = parse_psnr_from_output(psnr_result.stderr)

        # Calculate SSIM using FFmpeg with more verbose output
        ssim_cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "info",
            "-i", reference_video_lossless,
            "-i", encoded_video,
            "-lavfi", "ssim",
            "-f", "null", "-"
        ]
        
        ssim_result = subprocess.run(ssim_cmd, capture_output=True, text=True)
        if ssim_result.returncode != 0:
            print(f"SSIM calculation failed: {ssim_result.stderr}")
            ssim_avg = 0.0
        else:
            # Parse SSIM from stderr output
            ssim_avg = parse_ssim_from_output(ssim_result.stderr)
        
        print(f"Quality metrics - PSNR: {psnr_avg:.2f} dB, SSIM: {ssim_avg:.4f}")
        return psnr_avg, ssim_avg
        
    except Exception as e:
        print(f"Error calculating quality metrics: {e}")
        return 0.0, 0.0

def parse_psnr_from_output(output: str) -> float:
    """
    Parse average PSNR value from FFmpeg psnr filter output.
    
    Args:
        output: FFmpeg stderr output containing PSNR stats
        
    Returns:
        Average PSNR value in dB
    """
    try:
        # Look for the PSNR summary line
        lines = output.strip().split('\n')
        for line in lines:
            # Look for different possible PSNR output formats
            if 'PSNR' in line and 'average:' in line:
                # Extract PSNR value from line like: "[Parsed_psnr_0 @ 0x...] PSNR y:... u:... v:... average:28.884716 min:... max:..."
                parts = line.split('average:')
                if len(parts) > 1:
                    avg_part = parts[1].split()[0]
                    return float(avg_part)
            elif 'PSNR' in line and 'avg:' in line:
                # Alternative format: "PSNR ... avg:28.88"
                parts = line.split('avg:')
                if len(parts) > 1:
                    avg_part = parts[1].split()[0]
                    return float(avg_part)
                    
        return 0.0
    except (ValueError, IndexError):
        return 0.0

def parse_ssim_from_output(output: str) -> float:
    """
    Parse average SSIM value from FFmpeg ssim filter output.
    
    Args:
        output: FFmpeg stderr output containing SSIM stats
        
    Returns:
        Average SSIM value (0-1)
    """
    try:
        # Look for the SSIM summary line
        lines = output.strip().split('\n')
        for line in lines:
            # Look for different possible SSIM output formats
            if 'SSIM' in line and 'All:' in line:
                # Extract SSIM value from line like: "[Parsed_ssim_0 @ 0x...] SSIM Y:... U:... V:... All:0.873911 (...)"
                parts = line.split('All:')
                if len(parts) > 1:
                    all_part = parts[1].split()[0]
                    return float(all_part)
            elif 'SSIM' in line and 'avg:' in line:
                # Alternative format: "SSIM ... avg:0.8739"
                parts = line.split('avg:')
                if len(parts) > 1:
                    avg_part = parts[1].split()[0]
                    return float(avg_part)
                    
        return 0.0
    except (ValueError, IndexError):
        return 0.0

def calculate_blockwise_ssim(ref_frame: np.ndarray, test_frame: np.ndarray, block_size: int) -> np.ndarray:
    """
    Calculate SSIM for each block in a frame.
    
    Args:
        ref_frame: Reference frame (H, W, C)
        test_frame: Test frame (H, W, C)
        block_size: Size of each block
        
    Returns:
        2D array of SSIM values for each block
    """
    from skimage.metrics import structural_similarity as ssim
    
    height, width = ref_frame.shape[:2]
    num_blocks_y = height // block_size
    num_blocks_x = width // block_size
    
    ssim_blocks = np.zeros((num_blocks_y, num_blocks_x))
    
    for i in range(num_blocks_y):
        for j in range(num_blocks_x):
            y1 = i * block_size
            x1 = j * block_size
            y2 = min(y1 + block_size, height)
            x2 = min(x1 + block_size, width)
            
            ref_block = ref_frame[y1:y2, x1:x2]
            test_block = test_frame[y1:y2, x1:x2]
            
            # Convert to grayscale if needed
            if len(ref_block.shape) == 3:
                ref_block = cv2.cvtColor(ref_block, cv2.COLOR_BGR2GRAY)
                test_block = cv2.cvtColor(test_block, cv2.COLOR_BGR2GRAY)
            
            # Calculate SSIM for this block
            try:
                ssim_value = ssim(ref_block, test_block, data_range=255)
                ssim_blocks[i, j] = ssim_value
            except:
                ssim_blocks[i, j] = 0.0
    
    return ssim_blocks

def calculate_blockwise_psnr(ref_frame: np.ndarray, test_frame: np.ndarray, block_size: int) -> np.ndarray:
    """
    Calculate PSNR for each block in a frame.
    
    Args:
        ref_frame: Reference frame (H, W, C)
        test_frame: Test frame (H, W, C)
        block_size: Size of each block
        
    Returns:
        2D array of PSNR values for each block
    """
    height, width = ref_frame.shape[:2]
    num_blocks_y = height // block_size
    num_blocks_x = width // block_size
    
    psnr_blocks = np.zeros((num_blocks_y, num_blocks_x))
    
    for i in range(num_blocks_y):
        for j in range(num_blocks_x):
            y1 = i * block_size
            x1 = j * block_size
            y2 = min(y1 + block_size, height)
            x2 = min(x1 + block_size, width)
            
            ref_block = ref_frame[y1:y2, x1:x2].astype(np.float64)
            test_block = test_frame[y1:y2, x1:x2].astype(np.float64)
            
            # Calculate MSE
            mse = np.mean((ref_block - test_block) ** 2)
            
            if mse == 0:
                psnr_blocks[i, j] = 100.0  # Cap at 100 dB instead of infinity
            elif mse < 1e-10:  # Very small MSE
                psnr_blocks[i, j] = 100.0
            else:
                psnr_value = 20 * np.log10(255.0 / np.sqrt(mse))
                psnr_blocks[i, j] = min(psnr_value, 100.0)  # Cap at 100 dB
    
    return psnr_blocks

def create_quality_heatmaps(reference_frames_dir: str, encoded_videos: dict, masks_dir: str, 
                           square_size: int, width: int, height: int, temp_dir: str, 
                           sample_frames: List[int] = [0, 20, 40]) -> None:
    """
    Create heatmaps showing quality differences across frames for visualization.
    
    Args:
        reference_frames_dir: Directory containing reference frames
        encoded_videos: Dictionary mapping video names to file paths
        masks_dir: Directory containing UFO masks
        square_size: Size of blocks used for analysis
        width: Video width
        height: Video height
        temp_dir: Temporary directory for intermediate files
        sample_frames: List of frame indices to analyze
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        
        print(f"\nCreating quality heatmaps for sample frames: {sample_frames}")
        
        # Create heatmaps directory
        heatmaps_dir = os.path.join(temp_dir, "quality_heatmaps")
        os.makedirs(heatmaps_dir, exist_ok=True)
        
        # Create temporary directories for decoded frames
        decoded_frames_dir = os.path.join(temp_dir, "decoded_frames")
        
        for video_name, video_path in encoded_videos.items():
            if not os.path.exists(video_path):
                continue
                
            # Create directory for this video's decoded frames
            video_decoded_dir = os.path.join(decoded_frames_dir, video_name.replace(" ", "_"))
            os.makedirs(video_decoded_dir, exist_ok=True)
            
            # Decode video to frames
            decode_cmd = [
                "ffmpeg", "-hide_banner", "-loglevel", "warning",
                "-i", video_path,
                "-q:v", "2",
                "-y", f"{video_decoded_dir}/%05d.jpg"
            ]
            
            subprocess.run(decode_cmd, capture_output=True, text=True)
            
            # Process sample frames
            for frame_idx in sample_frames:
                try:
                    # Load reference and decoded frames
                    ref_frame_path = os.path.join(reference_frames_dir, f"{frame_idx+1:05d}.jpg")
                    decoded_frame_path = os.path.join(video_decoded_dir, f"{frame_idx+1:05d}.jpg")
                    
                    if not os.path.exists(ref_frame_path) or not os.path.exists(decoded_frame_path):
                        continue
                        
                    ref_frame = cv2.imread(ref_frame_path)
                    decoded_frame = cv2.imread(decoded_frame_path)
                    ref_frame = cv2.resize(ref_frame, (width, height))
                    decoded_frame = cv2.resize(decoded_frame, (width, height))
                    
                    # Load UFO mask
                    mask_path = os.path.join(masks_dir, f"{frame_idx+1:05d}.png")
                    if not os.path.exists(mask_path):
                        continue
                        
                    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                    mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
                    
                    # Calculate block-wise quality metrics
                    ssim_blocks = calculate_blockwise_ssim(ref_frame, decoded_frame, square_size)
                    psnr_blocks = calculate_blockwise_psnr(ref_frame, decoded_frame, square_size)
                    
                    # Create visualization
                    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
                    fig.suptitle(f'{video_name} - Frame {frame_idx+1}', fontsize=16)
                    
                    # Original frame
                    axes[0, 0].imshow(cv2.cvtColor(ref_frame, cv2.COLOR_BGR2RGB))
                    axes[0, 0].set_title('Reference Frame')
                    axes[0, 0].axis('off')
                    
                    # Decoded frame
                    axes[0, 1].imshow(cv2.cvtColor(decoded_frame, cv2.COLOR_BGR2RGB))
                    axes[0, 1].set_title('Encoded Frame')
                    axes[0, 1].axis('off')
                    
                    # UFO mask overlay
                    mask_colored = np.zeros((height, width, 3), dtype=np.uint8)
                    mask_colored[mask > 128] = [255, 0, 0]  # Red for foreground
                    mask_colored[mask <= 128] = [0, 0, 255]  # Blue for background
                    
                    overlay = cv2.addWeighted(ref_frame, 0.7, mask_colored, 0.3, 0)
                    axes[0, 2].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
                    axes[0, 2].set_title('UFO Mask (Red=FG, Blue=BG)')
                    axes[0, 2].axis('off')
                    
                    # SSIM heatmap
                    im1 = axes[1, 0].imshow(ssim_blocks, cmap='viridis', vmin=0, vmax=1)
                    axes[1, 0].set_title(f'SSIM Blocks (Mean: {np.mean(ssim_blocks):.3f})')
                    plt.colorbar(im1, ax=axes[1, 0])
                    
                    # PSNR heatmap (cap very high values for visualization)
                    psnr_capped = np.clip(psnr_blocks, 0, 50)
                    im2 = axes[1, 1].imshow(psnr_capped, cmap='viridis', vmin=0, vmax=50)
                    axes[1, 1].set_title(f'PSNR Blocks (Mean: {np.mean(psnr_blocks[np.isfinite(psnr_blocks)]):.1f} dB)')
                    plt.colorbar(im2, ax=axes[1, 1])
                    
                    # Quality difference overlay on original
                    num_blocks_y, num_blocks_x = ssim_blocks.shape
                    axes[1, 2].imshow(cv2.cvtColor(ref_frame, cv2.COLOR_BGR2RGB))
                    
                    # Overlay quality blocks with transparency
                    for i in range(num_blocks_y):
                        for j in range(num_blocks_x):
                            y1 = i * square_size
                            x1 = j * square_size
                            y2 = min(y1 + square_size, height)
                            x2 = min(x1 + square_size, width)
                            
                            ssim_val = ssim_blocks[i, j]
                            if ssim_val < 0.5:  # Poor quality
                                color = 'red'
                                alpha = 0.6
                            elif ssim_val < 0.7:  # Medium quality
                                color = 'yellow'
                                alpha = 0.4
                            else:  # Good quality
                                continue  # Don't highlight good quality blocks
                                
                            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, 
                                                   linewidth=0, edgecolor='none', 
                                                   facecolor=color, alpha=alpha)
                            axes[1, 2].add_patch(rect)
                    
                    axes[1, 2].set_title('Quality Overlay (Red=Poor, Yellow=Medium)')
                    axes[1, 2].axis('off')
                    
                    # Save the figure
                    safe_video_name = video_name.replace(" ", "_").replace("/", "_")
                    output_path = os.path.join(heatmaps_dir, f'{safe_video_name}_frame_{frame_idx+1:03d}.png')
                    plt.savefig(output_path, dpi=1500, bbox_inches='tight')
                    plt.close()
                    
                except Exception as e:
                    print(f"Error creating heatmap for {video_name} frame {frame_idx+1}: {e}")
                    continue
        
        print(f"Quality heatmaps saved to: {heatmaps_dir}")
        
        # Clean up decoded frames
        shutil.rmtree(decoded_frames_dir, ignore_errors=True)
        
    except ImportError:
        print("Matplotlib not available - skipping quality heatmap generation")
    except Exception as e:
        print(f"Error creating quality heatmaps: {e}")

def analyze_foreground_background_quality(reference_frames_dir: str, encoded_videos: dict, masks_dir: str, 
                                         square_size: int, width: int, height: int, temp_dir: str) -> dict:
    """
    Analyze video quality separately for foreground and background regions using UFO masks.
    
    Args:
        reference_frames_dir: Directory containing reference frames
        encoded_videos: Dictionary mapping video names to file paths
        masks_dir: Directory containing UFO masks
        square_size: Size of blocks used for analysis
        width: Video width
        height: Video height
        temp_dir: Temporary directory for intermediate files
        
    Returns:
        Dictionary containing detailed quality analysis results
    """
    print(f"\nAnalyzing foreground/background quality separately...")
    
    results = {}
    
    # Create temporary directories for decoded frames
    decoded_frames_dir = os.path.join(temp_dir, "decoded_frames")
    
    for video_name, video_path in encoded_videos.items():
        if not os.path.exists(video_path):
            print(f"Warning: Video file not found: {video_path}")
            continue
            
        print(f"Processing {video_name}...")
        
        # Create directory for this video's decoded frames
        video_decoded_dir = os.path.join(decoded_frames_dir, video_name.replace(" ", "_"))
        os.makedirs(video_decoded_dir, exist_ok=True)
        
        # Decode video to frames
        decode_cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "warning",
            "-i", video_path,
            "-q:v", "2",
            "-y", f"{video_decoded_dir}/%05d.jpg"
        ]
        
        result = subprocess.run(decode_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Failed to decode {video_name}: {result.stderr}")
            continue
        
        # Get list of reference and decoded frames
        ref_frames = sorted([f for f in os.listdir(reference_frames_dir) if f.endswith('.jpg')])
        decoded_frames = sorted([f for f in os.listdir(video_decoded_dir) if f.endswith('.jpg')])
        
        num_frames = min(len(ref_frames), len(decoded_frames))
        
        # Initialize quality metrics storage
        fg_ssim_values = []
        bg_ssim_values = []
        fg_psnr_values = []
        bg_psnr_values = []
        
        num_blocks_y = height // square_size
        num_blocks_x = width // square_size
        
        for frame_idx in range(num_frames):
            # Load reference and decoded frames
            ref_frame_path = os.path.join(reference_frames_dir, ref_frames[frame_idx])
            decoded_frame_path = os.path.join(video_decoded_dir, decoded_frames[frame_idx])
            
            ref_frame = cv2.imread(ref_frame_path)
            decoded_frame = cv2.imread(decoded_frame_path)
            
            if ref_frame is None or decoded_frame is None:
                continue
                
            # Resize frames to ensure consistent dimensions
            ref_frame = cv2.resize(ref_frame, (width, height))
            decoded_frame = cv2.resize(decoded_frame, (width, height))
            
            # Load corresponding UFO mask
            mask_path = os.path.join(masks_dir, f"{frame_idx+1:05d}.png")
            if not os.path.exists(mask_path):
                print(f"Warning: Mask not found for frame {frame_idx+1}")
                continue
                
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
            
            # Downsample mask to block resolution
            block_mask = cv2.resize(mask, (num_blocks_x, num_blocks_y), interpolation=cv2.INTER_NEAREST)
            
            # Calculate block-wise quality metrics
            ssim_blocks = calculate_blockwise_ssim(ref_frame, decoded_frame, square_size)
            psnr_blocks = calculate_blockwise_psnr(ref_frame, decoded_frame, square_size)
            
            # Separate foreground and background blocks
            # UFO mask: 255 = foreground, 0 = background
            fg_mask = block_mask > 128  # Foreground blocks
            bg_mask = block_mask <= 128  # Background blocks
            
            # Extract quality values for foreground and background
            if np.any(fg_mask):
                fg_ssim_values.extend(ssim_blocks[fg_mask].flatten())
                fg_psnr_values.extend(psnr_blocks[fg_mask].flatten())
            
            if np.any(bg_mask):
                bg_ssim_values.extend(ssim_blocks[bg_mask].flatten())
                bg_psnr_values.extend(psnr_blocks[bg_mask].flatten())
        
        # Calculate statistics
        results[video_name] = {
            'foreground': {
                'ssim_mean': np.mean(fg_ssim_values) if fg_ssim_values else 0.0,
                'ssim_std': np.std(fg_ssim_values) if fg_ssim_values else 0.0,
                'psnr_mean': np.mean([p for p in fg_psnr_values if np.isfinite(p)]) if fg_psnr_values else 0.0,
                'psnr_std': np.std([p for p in fg_psnr_values if np.isfinite(p)]) if fg_psnr_values else 0.0,
                'block_count': len(fg_ssim_values)
            },
            'background': {
                'ssim_mean': np.mean(bg_ssim_values) if bg_ssim_values else 0.0,
                'ssim_std': np.std(bg_ssim_values) if bg_ssim_values else 0.0,
                'psnr_mean': np.mean([p for p in bg_psnr_values if np.isfinite(p)]) if bg_psnr_values else 0.0,
                'psnr_std': np.std([p for p in bg_psnr_values if np.isfinite(p)]) if bg_psnr_values else 0.0,
                'block_count': len(bg_ssim_values)
            },
            'file_size_mb': os.path.getsize(video_path) / (1024 * 1024)
        }
    
    # Clean up decoded frames
    shutil.rmtree(decoded_frames_dir, ignore_errors=True)
    
    return results

def compare_encoding_quality(reference_frames_dir: str, encoded_videos: dict, framerate: float, width: int, height: int, temp_dir: str, square_size: int = 16) -> None:
    """
    Compare quality metrics between multiple encoded videos and display results.
    Now includes foreground/background analysis.
    
    Args:
        reference_frames_dir: Directory containing reference frames
        encoded_videos: Dictionary mapping video names to file paths
        framerate: Video framerate
        width: Video width
        height: Video height
        temp_dir: Temporary directory for intermediate files
        square_size: Size of blocks for foreground/background analysis
    """
    print(f"\nCalculating quality metrics for all encoding approaches...")
    
    # Calculate overall quality metrics for all videos
    quality_results = {}
    file_sizes = {}
    
    for video_name, video_path in encoded_videos.items():
        if os.path.exists(video_path):
            psnr, ssim = calculate_video_quality_metrics(
                reference_frames_dir=reference_frames_dir,
                encoded_video=video_path,
                framerate=framerate,
                width=width,
                height=height,
                temp_dir=temp_dir
            )
            quality_results[video_name] = {"psnr": psnr, "ssim": ssim}
            file_sizes[video_name] = os.path.getsize(video_path) / (1024 * 1024)  # Size in MB
        else:
            print(f"Warning: Video file not found: {video_path}")
            quality_results[video_name] = {"psnr": 0.0, "ssim": 0.0}
            file_sizes[video_name] = 0.0

    # Display overall comparison results
    print(f"\n{'='*80}")
    print(f"{'OVERALL ENCODING QUALITY COMPARISON':^80}")
    print(f"{'='*80}")
    
    # Header
    print(f"{'Method':<20} {'Size (MB)':<12} {'PSNR (dB)':<12} {'SSIM':<8} {'PSNR/MB':<12} {'Efficiency':<12}")
    print(f"{'-'*80}")
    
    # Get reference for comparison (first video in the list)
    reference_name = list(encoded_videos.keys())[0]
    ref_psnr_per_mb = quality_results[reference_name]["psnr"] / file_sizes[reference_name] if file_sizes[reference_name] > 0 else 0
    
    # Display results for each video
    for video_name, video_path in encoded_videos.items():
        psnr = quality_results[video_name]["psnr"]
        ssim = quality_results[video_name]["ssim"]
        size_mb = file_sizes[video_name]
        psnr_per_mb = psnr / size_mb if size_mb > 0 else 0
        
        # Calculate efficiency improvement relative to reference
        if ref_psnr_per_mb > 0 and video_name != reference_name:
            efficiency_improvement = ((psnr_per_mb / ref_psnr_per_mb) - 1) * 100
            efficiency_str = f"{efficiency_improvement:+.1f}%"
        else:
            efficiency_str = "baseline"
        
        print(f"{video_name:<20} {size_mb:<12.2f} {psnr:<12.2f} {ssim:<8.4f} {psnr_per_mb:<12.2f} {efficiency_str:<12}")
    
    print(f"{'-'*80}")
    
    # Foreground/Background Analysis
    masks_dir = os.path.join(temp_dir, "UFO_masks")
    if os.path.exists(masks_dir) and os.listdir(masks_dir):
        print(f"\nPerforming foreground/background quality analysis...")
        
        # Get number of frames for heatmap generation
        ref_frames = [f for f in os.listdir(reference_frames_dir) if f.endswith('.jpg')]
        num_frames = len(ref_frames)
        
        try:
            # Check if scikit-image is available for SSIM calculation
            import_result = subprocess.run(["python", "-c", "from skimage.metrics import structural_similarity"], 
                                         capture_output=True, text=True)
            if import_result.returncode != 0:
                print("Installing scikit-image for block-wise SSIM calculation...")
                subprocess.run(["pip", "install", "scikit-image"], check=True)
            
            fg_bg_results = analyze_foreground_background_quality(
                reference_frames_dir=reference_frames_dir,
                encoded_videos=encoded_videos,
                masks_dir=masks_dir,
                square_size=square_size,
                width=width,
                height=height,
                temp_dir=temp_dir
            )
            
            # Create quality heatmaps for visual analysis
            create_quality_heatmaps(
                reference_frames_dir=reference_frames_dir,
                encoded_videos=encoded_videos,
                masks_dir=masks_dir,
                square_size=square_size,
                width=width,
                height=height,
                temp_dir=temp_dir,
                sample_frames=[0, num_frames//4, num_frames//2, 3*num_frames//4, num_frames-1] if num_frames > 5 else [0, num_frames//2, num_frames-1]
            )
            
            # Display foreground/background results
            print(f"\n{'='*100}")
            print(f"{'FOREGROUND / BACKGROUND QUALITY ANALYSIS':^100}")
            print(f"{'='*100}")
            
            print(f"{'Method':<20} {'Region':<12} {'SSIM':<12} {'PSNR (dB)':<12} {'Blocks':<10} {'Quality/MB':<12}")
            print(f"{'-'*100}")
            
            for video_name, results in fg_bg_results.items():
                fg_data = results['foreground']
                bg_data = results['background']
                file_size = results['file_size_mb']
                
                # Foreground metrics
                fg_quality_per_mb = fg_data['ssim_mean'] / file_size if file_size > 0 else 0
                print(f"{video_name:<20} {'Foreground':<12} {fg_data['ssim_mean']:<12.4f} {fg_data['psnr_mean']:<12.2f} {fg_data['block_count']:<10} {fg_quality_per_mb:<12.4f}")
                
                # Background metrics
                bg_quality_per_mb = bg_data['ssim_mean'] / file_size if file_size > 0 else 0
                print(f"{'':<20} {'Background':<12} {bg_data['ssim_mean']:<12.4f} {bg_data['psnr_mean']:<12.2f} {bg_data['block_count']:<10} {bg_quality_per_mb:<12.4f}")
                print(f"{'-'*100}")
            
            # Calculate and display relative improvements
            if len(fg_bg_results) > 1:
                baseline_name = list(fg_bg_results.keys())[0]
                baseline_fg = fg_bg_results[baseline_name]['foreground']
                baseline_bg = fg_bg_results[baseline_name]['background']
                
                print(f"\nFOREGROUND / BACKGROUND TRADE-OFF ANALYSIS:")
                print(f"{'='*80}")
                
                for video_name in list(fg_bg_results.keys())[1:]:
                    results = fg_bg_results[video_name]
                    fg_data = results['foreground']
                    bg_data = results['background']
                    
                    # Calculate relative changes
                    fg_ssim_change = ((fg_data['ssim_mean'] / baseline_fg['ssim_mean']) - 1) * 100 if baseline_fg['ssim_mean'] > 0 else 0
                    bg_ssim_change = ((bg_data['ssim_mean'] / baseline_bg['ssim_mean']) - 1) * 100 if baseline_bg['ssim_mean'] > 0 else 0
                    fg_psnr_change = fg_data['psnr_mean'] - baseline_fg['psnr_mean']
                    bg_psnr_change = bg_data['psnr_mean'] - baseline_bg['psnr_mean']
                    
                    print(f"\n{video_name} vs {baseline_name}:")
                    print(f"  Foreground SSIM change: {fg_ssim_change:+.2f}%")
                    print(f"  Background SSIM change: {bg_ssim_change:+.2f}%")
                    print(f"  Foreground PSNR change: {fg_psnr_change:+.2f} dB")
                    print(f"  Background PSNR change: {bg_psnr_change:+.2f} dB")
                    
                    # Determine if trade-off is favorable
                    if fg_ssim_change > 0 and bg_ssim_change < 0:
                        trade_off_ratio = abs(fg_ssim_change) / abs(bg_ssim_change) if bg_ssim_change != 0 else float('inf')
                        print(f"  Trade-off ratio (FG gain / BG loss): {trade_off_ratio:.2f}")
                        if trade_off_ratio > 1.0:
                            print(f"  ✓ Favorable trade-off: Foreground improvement outweighs background degradation")
                        else:
                            print(f"  ✗ Unfavorable trade-off: Background degradation outweighs foreground improvement")
                    elif fg_ssim_change > 0 and bg_ssim_change >= 0:
                        print(f"  ✓ Ideal result: Both foreground and background improved")
                    elif fg_ssim_change <= 0 and bg_ssim_change < 0:
                        print(f"  ✗ Poor result: Both foreground and background degraded")
                    else:
                        print(f"  ? Mixed result: Foreground degraded but background improved")
        
        except Exception as e:
            print(f"Error in foreground/background analysis: {e}")
            print("Continuing with overall quality analysis only...")
    else:
        print("UFO masks not found - skipping foreground/background analysis")

def print_comprehensive_summary(fg_bg_results: dict, overall_results: dict, file_sizes: dict) -> None:
    """
    Print a comprehensive summary and recommendations based on the analysis.
    
    Args:
        fg_bg_results: Results from foreground/background analysis
        overall_results: Overall quality results
        file_sizes: File sizes in MB
    """
    print(f"\n{'='*100}")
    print(f"{'COMPREHENSIVE ANALYSIS SUMMARY AND RECOMMENDATIONS':^100}")
    print(f"{'='*100}")
    
    if len(fg_bg_results) < 2:
        print("Insufficient data for comparison.")
        return
        
    baseline_name = list(fg_bg_results.keys())[0]
    adaptive_name = list(fg_bg_results.keys())[1]
    
    baseline_fg = fg_bg_results[baseline_name]['foreground']
    baseline_bg = fg_bg_results[baseline_name]['background']
    adaptive_fg = fg_bg_results[adaptive_name]['foreground']
    adaptive_bg = fg_bg_results[adaptive_name]['background']
    
    baseline_size = file_sizes[baseline_name]
    adaptive_size = file_sizes[adaptive_name]
    
    # Calculate metrics
    fg_ssim_improvement = ((adaptive_fg['ssim_mean'] / baseline_fg['ssim_mean']) - 1) * 100
    bg_ssim_degradation = ((adaptive_bg['ssim_mean'] / baseline_bg['ssim_mean']) - 1) * 100
    fg_psnr_improvement = adaptive_fg['psnr_mean'] - baseline_fg['psnr_mean']
    bg_psnr_degradation = adaptive_bg['psnr_mean'] - baseline_bg['psnr_mean']
    size_increase = ((adaptive_size / baseline_size) - 1) * 100
    
    # Overall quality change
    overall_psnr_change = overall_results[adaptive_name]['psnr'] - overall_results[baseline_name]['psnr']
    overall_ssim_change = overall_results[adaptive_name]['ssim'] - overall_results[baseline_name]['ssim']
    
    print(f"\n📊 KEY METRICS COMPARISON:")
    print(f"   • File Size: {baseline_size:.2f} MB → {adaptive_size:.2f} MB ({size_increase:+.1f}%)")
    print(f"   • Overall PSNR: {overall_results[baseline_name]['psnr']:.2f} dB → {overall_results[adaptive_name]['psnr']:.2f} dB ({overall_psnr_change:+.2f} dB)")
    print(f"   • Overall SSIM: {overall_results[baseline_name]['ssim']:.4f} → {overall_results[adaptive_name]['ssim']:.4f} ({overall_ssim_change:+.4f})")
    
    print(f"\n🎯 FOREGROUND (IMPORTANT OBJECTS) ANALYSIS:")
    print(f"   • SSIM: {baseline_fg['ssim_mean']:.4f} → {adaptive_fg['ssim_mean']:.4f} ({fg_ssim_improvement:+.1f}%)")
    print(f"   • PSNR: {baseline_fg['psnr_mean']:.2f} dB → {adaptive_fg['psnr_mean']:.2f} dB ({fg_psnr_improvement:+.2f} dB)")
    print(f"   • Block Count: {baseline_fg['block_count']:,} blocks")
    
    print(f"\n🌅 BACKGROUND ANALYSIS:")
    print(f"   • SSIM: {baseline_bg['ssim_mean']:.4f} → {adaptive_bg['ssim_mean']:.4f} ({bg_ssim_degradation:+.1f}%)")
    print(f"   • PSNR: {baseline_bg['psnr_mean']:.2f} dB → {adaptive_bg['psnr_mean']:.2f} dB ({bg_psnr_degradation:+.2f} dB)")
    print(f"   • Block Count: {baseline_bg['block_count']:,} blocks")
    
    # Calculate perceptual weights (foreground is typically more important)
    fg_weight = 0.7  # 70% weight for foreground
    bg_weight = 0.3  # 30% weight for background
    
    weighted_ssim_change = (fg_ssim_improvement * fg_weight) + (bg_ssim_degradation * bg_weight)
    weighted_psnr_change = (fg_psnr_improvement * fg_weight) + (bg_psnr_degradation * bg_weight)
    
    print(f"\n⚖️  PERCEPTUAL TRADE-OFF ANALYSIS (FG:70%, BG:30%):")
    print(f"   • Weighted SSIM Change: {weighted_ssim_change:+.2f}%")
    print(f"   • Weighted PSNR Change: {weighted_psnr_change:+.2f} dB")
    
    # Efficiency analysis
    fg_blocks_ratio = baseline_fg['block_count'] / (baseline_fg['block_count'] + baseline_bg['block_count'])
    bg_blocks_ratio = baseline_bg['block_count'] / (baseline_fg['block_count'] + baseline_bg['block_count'])
    
    print(f"\n📈 EFFICIENCY ANALYSIS:")
    print(f"   • Foreground Coverage: {fg_blocks_ratio*100:.1f}% of total blocks")
    print(f"   • Background Coverage: {bg_blocks_ratio*100:.1f}% of total blocks")
    print(f"   • Bitrate Allocation: {size_increase:+.1f}% increase for {fg_ssim_improvement:+.1f}% FG improvement")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    
    if weighted_ssim_change > 0:
        print(f"   ✅ POSITIVE: The adaptive encoding provides a net perceptual benefit")
        if fg_ssim_improvement > 15:
            print(f"   ✅ EXCELLENT: Foreground improvement ({fg_ssim_improvement:+.1f}%) is substantial")
        elif fg_ssim_improvement > 5:
            print(f"   ✅ GOOD: Foreground improvement ({fg_ssim_improvement:+.1f}%) is meaningful")
        else:
            print(f"   ⚠️  MODEST: Foreground improvement ({fg_ssim_improvement:+.1f}%) is small")
    else:
        print(f"   ❌ NEGATIVE: The adaptive encoding reduces overall perceptual quality")
    
    if abs(bg_ssim_degradation) < 10:
        print(f"   ✅ ACCEPTABLE: Background degradation ({bg_ssim_degradation:+.1f}%) is minimal")
    elif abs(bg_ssim_degradation) < 20:
        print(f"   ⚠️  MODERATE: Background degradation ({bg_ssim_degradation:+.1f}%) is noticeable")
    else:
        print(f"   ❌ SIGNIFICANT: Background degradation ({bg_ssim_degradation:+.1f}%) is substantial")
    
    if size_increase < 5:
        print(f"   ✅ EFFICIENT: Bitrate increase ({size_increase:+.1f}%) is minimal")
    elif size_increase < 15:
        print(f"   ⚠️  MODERATE: Bitrate increase ({size_increase:+.1f}%) is acceptable")
    else:
        print(f"   ❌ EXPENSIVE: Bitrate increase ({size_increase:+.1f}%) is significant")
    
    # Overall recommendation
    print(f"\n🏆 OVERALL RECOMMENDATION:")
    
    if weighted_ssim_change > 5 and size_increase < 15:
        print(f"   🌟 HIGHLY RECOMMENDED: Adaptive encoding provides excellent quality/bitrate trade-off")
    elif weighted_ssim_change > 0 and size_increase < 25:
        print(f"   ✅ RECOMMENDED: Adaptive encoding provides good perceptual benefits")
    elif weighted_ssim_change > -5:
        print(f"   ⚠️  CONDITIONAL: Consider adaptive encoding only if foreground quality is critical")
    else:
        print(f"   ❌ NOT RECOMMENDED: Adaptive encoding degrades overall quality too much")
    
    # Technical suggestions
    print(f"\n🔧 TECHNICAL SUGGESTIONS:")
    if fg_ssim_improvement < 10:
        print(f"   • Consider increasing ROI quantization contrast (higher qoffset range)")
        print(f"   • Verify UFO mask accuracy - low foreground improvement may indicate poor object detection")
    
    if abs(bg_ssim_degradation) > 25:
        print(f"   • Reduce background quantization penalty to limit quality loss")
        print(f"   • Consider selective background filtering instead of aggressive quantization")
    
    if size_increase > 20:
        print(f"   • Reduce target bitrate or adjust rate control parameters")
        print(f"   • Implement more aggressive background compression")
    
    print(f"\n📁 ANALYSIS ARTIFACTS:")
    print(f"   • Quality heatmaps: experiment/quality_heatmaps/")
    print(f"   • UFO masks: experiment/UFO_masks/")
    print(f"   • Encoded videos: experiment/*.mp4")
    
    print(f"\n{'='*100}")

    # Show detailed overall comparisons between methods
    if len(encoded_videos) > 1:
        print(f"\n{'='*80}")
        print(f"DETAILED OVERALL COMPARISONS (vs {reference_name}):")
        print(f"{'='*80}")
        ref_psnr = quality_results[reference_name]["psnr"]
        ref_ssim = quality_results[reference_name]["ssim"]
        ref_size = file_sizes[reference_name]
        
        for video_name in list(encoded_videos.keys())[1:]:
            psnr = quality_results[video_name]["psnr"]
            ssim = quality_results[video_name]["ssim"]
            size_mb = file_sizes[video_name]
            
            psnr_diff = psnr - ref_psnr
            ssim_diff = ssim - ref_ssim
            size_ratio = size_mb / ref_size if ref_size > 0 else 0
            size_reduction = (1 - size_ratio) * 100
            
            print(f"\n{video_name}:")
            print(f"  PSNR difference: {psnr_diff:+.2f} dB")
            print(f"  SSIM difference: {ssim_diff:+.4f}")
            print(f"  Size ratio: {size_ratio:.2f}x")
            print(f"  Size change: {-size_reduction:+.1f}%")
    
    # Print comprehensive summary if foreground/background analysis was performed
    if 'fg_bg_results' in locals():
        print_comprehensive_summary(fg_bg_results, quality_results, file_sizes)
    
    print(f"\n{'='*80}")

def concat_segments(segment_files: List[str], output_video: str) -> None:
    """
    Concatenate video segments into a single video file.
    
    Args:
        segment_files: List of segment file paths
        output_video: Output video file path
    """
    # Create a temporary file list for ffmpeg concat
    temp_dir = os.path.dirname(output_video)
    filelist_path = os.path.join(temp_dir, "segments_list.txt")
    
    try:
        with open(filelist_path, 'w') as f:
            for segment_file in segment_files:
                f.write(f"file '{os.path.abspath(segment_file)}'\n")
        
        # Concatenate using ffmpeg
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "warning",
            "-f", "concat",
            "-safe", "0",
            "-i", filelist_path,
            "-c", "copy",
            "-y", output_video
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Concatenation failed: {result.stderr}")
            
    finally:
        # Clean up temporary file list
        if os.path.exists(filelist_path):
            os.remove(filelist_path)

def main():
    # Example usage parameters

    reference_video = "davis_test/bear.mp4"
    width, height = 640, 368
    square_size = 16
    segment_size = 30  # Number of frames per segment for processing

    # Calculate appropriate target bitrate based on video characteristics
    framerate = cv2.VideoCapture(reference_video).get(cv2.CAP_PROP_FPS)
    target_bitrate = calculate_target_bitrate(width, height, framerate, quality_factor=1.2)
    print(f"Calculated target bitrate: {target_bitrate} bps ({target_bitrate/1000000:.1f} Mbps) for {width}x{height}@{framerate:.1f}fps")

    # SERVER SIDE

    # Start recording time
    start = time.time()

    # Create experiment directory for all experiment-related files
    experiment_dir = "experiment"
    os.makedirs(experiment_dir, exist_ok=True)

    # resize video based on required resolution, save raw and frames for reference
    print(f"Processing video: {reference_video}")
    print(f"Target resolution: {width}x{height}")
    print(f"Calculated target bitrate: {target_bitrate} bps ({target_bitrate/1000000:.1f} Mbps) for {width}x{height}@{framerate:.1f}fps")

    reference_frames_dir = os.path.join(experiment_dir, "reference_frames")
    os.makedirs(reference_frames_dir, exist_ok=True)

    print("Converting video to raw YUV format...")
    raw_video_path = os.path.join(experiment_dir, "reference_raw.yuv")
    os.system(f"ffmpeg -hide_banner -loglevel error -i {reference_video} -vf scale={width}:{height} -c:v rawvideo -pix_fmt yuv420p {raw_video_path}")

    print("Extracting reference frames...")
    os.system(f"ffmpeg -hide_banner -loglevel error -video_size {width}x{height} -r {framerate} -pixel_format yuv420p -i {raw_video_path} -q:v 2 {reference_frames_dir}/%05d.jpg")

    end = time.time()
    print(f"Video preprocessing completed in {end - start:.2f} seconds.\n")
    start = time.time()

    # Calculate removability scores with integrated temporal smoothing
    print(f"Calculating removability scores with block size: {square_size}x{square_size}")
    removability_scores = calculate_removability_scores(
        raw_video_file=raw_video_path,
        reference_frames_folder=reference_frames_dir,
        width=width,
        height=height,
        square_size=square_size,
        alpha=0.5,
        working_dir=experiment_dir,
        smoothing_beta=0.5
    )
    end = time.time()
    print(f"Removability scores calculation completed in {end - start:.2f} seconds.\n")
    start = time.time()

    # Benchmark 1: traditional encoding with H.265
    print(f"Encoding reference frames with H.265 for baseline comparison...")
    baseline_video_h265 = os.path.join(experiment_dir, "baseline.mp4")

    # Use subprocess for better control and consistency with adaptive encoding
    baseline_cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "warning",
        "-framerate", str(framerate),
        "-i", f"{reference_frames_dir}/%05d.jpg",
        "-c:v", "libx265",
        "-b:v", str(target_bitrate),
        "-minrate", str(int(target_bitrate * 0.9)),
        "-maxrate", str(int(target_bitrate * 1.1)),
        "-bufsize", str(target_bitrate),  # Small buffer for strict bitrate control
        "-preset", "medium",
        "-g", "1",
        "-y", baseline_video_h265
    ]

    result = subprocess.run(baseline_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Baseline encoding failed: {result.stderr}")
        raise RuntimeError(f"Baseline encoding failed: {result.stderr}")
    else:
        print("Baseline encoding completed successfully")

    end = time.time()
    print(f"Baseline encoding completed in {end - start:.2f} seconds.\n")
    start = time.time()

    # Benchmark 2: adaptive encoding
    print(f"Encoding frames with ROI-based adaptive quantization...")
    adaptive_video_h265 = os.path.join(experiment_dir, "adaptive.mp4")

    # Use same encoder parameters as baseline encoding for fair comparison
    encode_with_h265_roi(
        input_frames_dir=reference_frames_dir,
        output_video=adaptive_video_h265,
        removability_scores=removability_scores,
        square_size=square_size,
        framerate=framerate,
        width=width,
        height=height,
        segment_size=segment_size,
        target_bitrate=target_bitrate
    )

    end = time.time()
    print(f"Adaptive encoding completed in {end - start:.2f} seconds.\n")
    start = time.time()

    # Benchmark 3: ELVIS 1 (zero information removal and inpainting)

    # Method 1: 


    # Compare file sizes and quality metrics
    baseline_size = os.path.getsize(baseline_video_h265)
    adaptive_size = os.path.getsize(adaptive_video_h265)
    compression_ratio = baseline_size / adaptive_size

    print(f"\nEncoding Results (Target Bitrate: {target_bitrate} bps / {target_bitrate/1000000:.1f} Mbps):")
    print(f"Baseline H.265 video size: {baseline_size / 1024 / 1024:.2f} MB")
    print(f"Adaptive ROI video size: {adaptive_size / 1024 / 1024:.2f} MB")
    print(f"Size ratio: {adaptive_size / baseline_size:.2f}x")
    print(f"Size difference: {(adaptive_size - baseline_size) / baseline_size * 100:+.1f}%")

    # Comprehensive quality comparison
    encoded_videos = {
        "Baseline H.265": baseline_video_h265,
        "Adaptive ROI": adaptive_video_h265
    }

    compare_encoding_quality(
        reference_frames_dir=reference_frames_dir,
        encoded_videos=encoded_videos,
        framerate=framerate,
        width=width,
        height=height,
        temp_dir=experiment_dir,
        square_size=square_size
    )

    end = time.time()
    print(f"Quality comparison completed in {end - start:.2f} seconds.\n")

if __name__ == "__main__":
    main()