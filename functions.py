from concurrent.futures import ProcessPoolExecutor
import json
import os
import shutil
import subprocess
from typing import Union
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

# TODO: have 2 extra dimensions to group by stride, so that the top left of each stride is always (0,0) and more complex masks can be made
def split_image_into_squares(image: np.array, l: int) -> np.array:
    """
    Split an image into squares of a specific size.

    Args:
    - image: numpy array representing the image with shape [n, m, c]
    - l: integer representing the side length of each square

    Returns:
    - numpy array with shape [n//l, m//l, l, l, c] containing the squares
    """
    n, m, c = image.shape
    num_rows = n // l
    num_cols = m // l
    squares = np.zeros((num_rows, num_cols, l, l, c), dtype=image.dtype)
    for i in range(num_rows):
        for j in range(num_cols):
            squares[i, j] = image[i*l:(i+1)*l, j*l:(j+1)*l, :]
    return squares

def filter_squares(squares: np.array, horizontal_stride: int, vertical_stride: int) -> Union[np.array, np.array]:
    """
    Filter squares based on their indices:
    Start by setting a stride, which determines the size of a block group. E.g., stride 3 means blocks are grouped in 3 by 3.
    Then, choose which blocks from each group will be kept by setting the row and column from where to start counting.
    E.g., in a 3 stride context, (0, 0) will select each top left block in a group, (0, 1) each top center block, 
    (0, 2) each top right, (1, 0) center left, etc. For each block that needs to be kept, add a 2d tuple like just shown
    to the tuple of blocks_to_keep.

    Args:
    - squares: numpy array with shape [n, m, l, l, c] containing the squares

    Returns:
    - filtered_squares: numpy array with shape [n_filtered, m_filtered, l, l, c] containing the filtered squares
    - filter_mask: numpy array with shape [n, m] indicating which blocks were kept (1) or filtered out (0)
    """
    filter_mask = np.ones(squares.shape, dtype=int) * 255
    filter_mask[::horizontal_stride, ::vertical_stride] = 0
    filtered_squares = squares[::horizontal_stride, ::vertical_stride]
    return filtered_squares, filter_mask

def flatten_squares_into_image(squares: np.array) -> np.array:
    """
    Reconstruct the original image from split squares.

    Args:
    - squares: numpy array with shape [n, m, l, l, c] containing the split squares

    Returns:
    - numpy array representing the reconstructed image
    """
    n, m, l, _, c = squares.shape
    num_rows = n * l
    num_cols = m * l
    image = np.zeros((num_rows, num_cols, c), dtype=squares.dtype)
    for i in range(n):
        for j in range(m):
            image[i*l:(i+1)*l, j*l:(j+1)*l, :] = squares[i, j]
    return image

def save_image(frame: np.array, output_folder: str, file_name: str) -> None:

    # Create the output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    # save filtered frame, overwrite if it exists
    cv2.imwrite(output_folder + '/' + file_name, frame, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])

def process_frame_server_side(frame_name, frames_folder, square_size, horizontal_stride, vertical_stride):
    frame = cv2.imread(frames_folder + '/' + frame_name)
    squared_frame = split_image_into_squares(frame, square_size)
    filtered_squares, mask_squares = filter_squares(squared_frame, horizontal_stride, vertical_stride)
    filtered_flattened = flatten_squares_into_image(filtered_squares)
    mask_flattened = flatten_squares_into_image(mask_squares)
    parent_folder, _ = frames_folder.rsplit('/', 1)
    save_image(filtered_flattened, parent_folder + '/' + 'shrunk', frame_name)
    save_image(mask_flattened, parent_folder + '/' + 'masks', frame_name)
    return frame_name

def process_frames_in_parallel(frames_folder, processing_function, num_processes, **kwargs):
    frame_names = [frame_name for frame_name in os.listdir(frames_folder) if frame_name.endswith('.png')]
    
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        results = []
        for frame_name in frame_names:
            results.append(executor.submit(processing_function, frame_name, frames_folder, **kwargs))
        
        # Retrieve results
        processed_frame_names = [future.result() for future in results]

    return processed_frame_names

def stretch_frame(frame: np.array, mask: np.array, horizontal_stride: int, vertical_stride: int) -> np.array:
    
    '''
    Arguments:
    - frame: numpy array with shape [n, m, l, l, c] containing the squares
    - mask: numpy array with shape [N, M, l, l, c] containing [n//l, m//l] white squares and [N - n//l, M - m//l] black squares
    - l: an integer specifying the side in pixel of a square

    Returns:
    - stretched_frame: numpy array with shape [N, M, l, l, c] where the squares of frame are substituted to the white squares of mask
    '''

    # Get dimensions
    n, m, _, _, _ = frame.shape
    N, _, _, _, _ = mask.shape

    stretched_frame = np.copy(mask)

    for i in range(n):
        for j in range(m):
            # iterate through each frame block
            frame_block = frame[i, j, :, :, :]
            # put it into the corresponding mask white block by multiplying its indices by the stride
            stretched_frame[int(i * horizontal_stride), int(j * vertical_stride), :, :, :] = frame_block
    return stretched_frame

def process_frame_client_side(frame_name, frames_folder, square_size, mask_squares, horizontal_stride, vertical_stride):
    frame = cv2.imread(frames_folder + '/' + frame_name)
    squared_frame = split_image_into_squares(frame, square_size)
    stretched_squares = stretch_frame(squared_frame, mask_squares, horizontal_stride, vertical_stride)
    stretched_flattened = flatten_squares_into_image(stretched_squares)
    parent_folder, _ = frames_folder.rsplit('/', 1)
    save_image(stretched_flattened, parent_folder + '/stretched', frame_name)
    return frame_name