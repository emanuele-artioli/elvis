import time
import shutil
import os
import subprocess
from pathlib import Path
import cv2
import numpy as np
from typing import List, Callable, Tuple

# Utility functions

def normalize_array(arr: np.ndarray) -> np.ndarray:
    """Normalizes a NumPy array to the range [0, 1]."""
    min_val = arr.min()
    max_val = arr.max()
    if max_val - min_val > 0:
        return (arr - min_val) / (max_val - min_val)
    return arr

def calculate_removability_scores(raw_video_file: str, reference_frames_folder: str, width: int, height: int, square_size: int, alpha: float = 0.5, working_dir: str = ".") -> np.ndarray:
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

        return removability_scores
    
    except Exception as e:
        print(f"Error in calculate_removability_scores: {e}")
        raise
    finally:
        # Always return to original directory
        os.chdir(original_dir)

def apply_temporal_smoothing(scores: np.ndarray, beta: float = 0.5) -> np.ndarray:
    """
    Applies temporal smoothing across frames to a score array.

    This function helps reduce flicker or jitter in decisions made based on these
    scores by blending the scores of a frame with the scores of the preceding frame.

    Args:
        scores: A 3D NumPy array of scores (num_frames, num_blocks_y, num_blocks_x).
        beta: The smoothing factor. A value of 1.0 means no smoothing (only use
              current frame), while a value of 0.0 would mean only using the
              previous frame's scores.

    Returns:
        A new 3D NumPy array of the same shape with smoothed scores.
    """
    if scores.ndim != 3 or scores.shape[0] < 2:
        # Not enough frames to smooth, return the original scores
        return scores

    smoothed_scores = np.zeros_like(scores)
    
    # The first frame has no prior frame, so its scores remain unchanged.
    smoothed_scores[0] = scores[0]

    # For all subsequent frames, apply the smoothing formula.
    # This is a vectorized operation, which is very fast.
    smoothed_scores[1:] = beta * scores[1:] + (1 - beta) * scores[:-1]
    
    return smoothed_scores

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

# Example usage parameters

reference_video = "davis_test/bear.mp4"
width, height = 640, 480
square_size = 20

# SERVER SIDE

# Start recording time
start = time.time()

# Create experiment directory for all experiment-related files
experiment_dir = "experiment"
os.makedirs(experiment_dir, exist_ok=True)

# resize video based on required resolution, save raw and frames for reference
print(f"Processing video: {reference_video}")
print(f"Target resolution: {width}x{height}")

framerate = cv2.VideoCapture(reference_video).get(cv2.CAP_PROP_FPS)
reference_frames_dir = os.path.join(experiment_dir, "reference_frames")
os.makedirs(reference_frames_dir, exist_ok=True)

print("Converting video to raw YUV format...")
raw_video_path = os.path.join(experiment_dir, "reference_raw.yuv")
os.system(f"ffmpeg -hide_banner -loglevel error -i {reference_video} -vf scale={width}:{height} -c:v rawvideo -pix_fmt yuv420p {raw_video_path}")

print("Extracting reference frames...")
os.system(f"ffmpeg -hide_banner -loglevel error -video_size {width}x{height} -r {framerate} -pixel_format yuv420p -i {raw_video_path} -q:v 2 {reference_frames_dir}/%05d.jpg")

print(f"Calculating removability scores with block size: {square_size}x{square_size}")
removability_scores = calculate_removability_scores(
    raw_video_file=raw_video_path,
    reference_frames_folder=reference_frames_dir,
    width=width,
    height=height,
    square_size=square_size,
    alpha=0.5,
    working_dir=experiment_dir
)

# Removability scores are now ready for use in further processing
print(f"Removability scores calculated successfully. Shape: {removability_scores.shape}")

# End recording time and report
end = time.time()
print(f"Total processing time: {end - start:.2f} seconds")

