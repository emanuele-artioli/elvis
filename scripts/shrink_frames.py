from concurrent.futures import ProcessPoolExecutor
import csv
import os
import cv2
import numpy as np

def normalize_array(arr):
    """
    Normalize a NumPy array to the range [0, 1].
    
    Parameters:
    arr (numpy.ndarray): Input array to be normalized.
    
    Returns:
    numpy.ndarray: Normalized array with values in the range [0, 1].
    """
    arr_min = np.min(arr)
    arr_max = np.max(arr)
    normalized_arr = (arr - arr_min) / (arr_max - arr_min)
    return normalized_arr

def get_coordinates_to_remove(temporal_file, spatial_file, width, height, square_size, alpha, to_remove, smoothing_factor=0.5):
    # Load the CSV files into 2D NumPy arrays
    temporal_array = np.loadtxt(temporal_file, delimiter=',', skiprows=1)
    spatial_array = np.loadtxt(spatial_file, delimiter=',', skiprows=1)

    num_blocks_x = width // square_size  # Number of horizontal blocks
    num_blocks_y = height // square_size  # Number of vertical blocks
    num_frames = temporal_array.shape[1]  # Number of frames (number of columns in CSV)

    # Reshape the arrays to (num_frames, num_blocks_y, num_blocks_x)
    temporal_3d_array = temporal_array.T.reshape(num_frames, num_blocks_y, num_blocks_x)
    spatial_3d_array = spatial_array.T.reshape(num_frames, num_blocks_y, num_blocks_x)

    # Normalize arrays
    temporal_3d_array = normalize_array(temporal_3d_array)
    spatial_3d_array = normalize_array(spatial_3d_array)

    # Get the shape details
    num_frames, num_blocks_y, num_blocks_x = spatial_3d_array.shape

    # Initialize the importance array
    importance = np.zeros((num_frames, num_blocks_y, num_blocks_x))

    # Calculate the importance values (for the last frame, there is no successive temporal complexity, so we rely only on spatial)
    for i in range(num_frames):
        if i == num_frames - 1:
            importance[i] = spatial_3d_array[i]
        else:
            importance[i] = alpha * spatial_3d_array[i] + (1 - alpha) * temporal_3d_array[i + 1]

    # Initialize the list to store coordinates of the lowest values
    removed_values_coords = []

    # to_remove can be a percentage, or a number of blocks
    if to_remove < 1:
        num_blocks_to_remove = int(to_remove * num_blocks_x)
    else:
        num_blocks_to_remove = int(to_remove)

    # Initialize a previous_importance array for smoothing
    previous_importance = None

    # Loop through each frame
    for i in range(num_frames):
        frame_coords = []
        # Loop through each row in the current frame
        for j in range(num_blocks_y):
            # Get the current row
            current_row = importance[i, j, :]

            # Apply smoothing with the previous frame's importance
            if previous_importance is not None:
                current_row = smoothing_factor * current_row + (1 - smoothing_factor) * previous_importance[j, :]

            # Find the indices of the highest values in the current row based on the percentage
            if len(current_row) > num_blocks_to_remove:
                indices_to_remove = np.argsort(current_row)[-num_blocks_to_remove:]
            else:
                indices_to_remove = np.argsort(current_row)

            # Store the coordinates for column (frame and row given by indices)
            row_coords = [k for k in indices_to_remove]
            frame_coords.append(row_coords)

            # Reduce the chance of these blocks to be removed from the next frame
            if i < num_frames - 1:
                importance[i + 1, j, indices_to_remove] /= 2

        # Update previous_importance for the next iteration
        previous_importance = importance[i]

        removed_values_coords.append(frame_coords)

    return removed_values_coords

def get_coordinates_to_remove(temporal_file, spatial_file, width, height, square_size, alpha, to_remove, focused_folder, smoothing_factor=0.5):
    # Load the CSV files into 2D NumPy arrays
    temporal_array = np.loadtxt(temporal_file, delimiter=',', skiprows=1)
    spatial_array = np.loadtxt(spatial_file, delimiter=',', skiprows=1)

    num_blocks_x = width // square_size  # Number of horizontal blocks
    num_blocks_y = height // square_size  # Number of vertical blocks
    num_frames = temporal_array.shape[1]  # Number of frames (number of columns in CSV)

    # Reshape the arrays to (num_frames, num_blocks_y, num_blocks_x)
    temporal_3d_array = temporal_array.T.reshape(num_frames, num_blocks_y, num_blocks_x)
    spatial_3d_array = spatial_array.T.reshape(num_frames, num_blocks_y, num_blocks_x)

    # Normalize arrays
    temporal_3d_array = normalize_array(temporal_3d_array)
    spatial_3d_array = normalize_array(spatial_3d_array)

    # Get the shape details
    num_frames, num_blocks_y, num_blocks_x = spatial_3d_array.shape

    # Initialize the importance array
    importance = np.zeros((num_frames, num_blocks_y, num_blocks_x))

    # Calculate the importance values (for the last frame, there is no successive temporal complexity, so we rely only on spatial)
    for i in range(num_frames):
        if i == num_frames - 1:
            importance[i] = spatial_3d_array[i]
        else:
            importance[i] = alpha * spatial_3d_array[i] + (1 - alpha) * temporal_3d_array[i + 1]

    # Initialize the list to store coordinates of the lowest values
    removed_values_coords = []

    # Load masks for each frame
    masks = [cv2.imread(f"{focused_folder}/{i:05d}.png", cv2.IMREAD_GRAYSCALE) for i in range(num_frames)]
    masks = [cv2.resize(mask, (num_blocks_x, num_blocks_y)) for mask in masks]

    # to_remove can be a percentage, or a number of blocks
    if to_remove < 1:
        num_blocks_to_remove = int(to_remove * num_blocks_x)
    else:
        num_blocks_to_remove = int(to_remove)

    # Initialize a previous_importance array for smoothing
    previous_importance = None

    # Loop through each frame
    for i in range(num_frames):
        frame_coords = []
        # Loop through each row in the current frame
        for j in range(num_blocks_y):
            # Get the current row
            current_row = importance[i, j, :]

            # Apply smoothing with the previous frame's importance
            if previous_importance is not None:
                current_row = smoothing_factor * current_row + (1 - smoothing_factor) * previous_importance[j, :]

            # Apply the mask: Only allow block removal if the corresponding mask region is black (background)
            mask_row = masks[i][j, :]  # Get the mask row
            background_indices = np.where(mask_row == 0)[0]  # Indices where mask is black (background)

            # Restrict block removal to background areas TODO: instead of this, set their importance to a very high number, so they can still be removed but only after all the rest
            if len(background_indices) > num_blocks_to_remove:
                indices_to_remove = background_indices[np.argsort(current_row[background_indices])[-num_blocks_to_remove:]]
            else:
                indices_to_remove = background_indices[np.argsort(current_row[background_indices])]

            # Store the coordinates for column (frame and row given by indices)
            row_coords = [k for k in indices_to_remove]
            frame_coords.append(row_coords)

            # Reduce the chance of these blocks to be removed from the next frame TODO: maybe remove this 
            if i < num_frames - 1:
                importance[i + 1, j, indices_to_remove] /= 2

        # Update previous_importance for the next iteration
        previous_importance = importance[i]

        removed_values_coords.append(frame_coords)

    return removed_values_coords

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

def filter_squares(squares: np.array, block_coords: list) -> tuple:
    """
    Remove specified blocks from a squares array based on block coordinates.

    Args:
    - squares: numpy array with shape [num_rows, num_cols, l, l, c] representing the image split into squares
    - block_coords: list of lists containing column indices of blocks to be removed for each row

    Returns:
    - new_squares: numpy array with shape [num_rows, new_num_cols, l, l, c] containing squares without the removed blocks
    - mask: list of length num_rows * num_cols indicating where blocks were removed (1) and unchanged areas (0)
    """
    num_rows, num_cols, _, _, _ = squares.shape
    new_squares = []
    mask = []

    for i in range(num_rows):
        row_squares = []
        for j in range(num_cols):
            if j not in block_coords[i]:
                row_squares.append(squares[i, j])
                mask.append(0)
            else:
                mask.append(1)
        new_squares.append(row_squares)

    # Convert new_squares list to numpy array
    new_squares = np.array(new_squares)

    return new_squares, mask

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

def save_mask(frame_number, binary_representation, csv_file):
    # Open the CSV file in append mode
    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        # Write the frame number and binary representation as a new row
        writer.writerow([frame_number] + binary_representation)

def process_frame_server_side(frame_name, experiment_folder, square_size, block_coords):

    frame = cv2.imread(f'{experiment_folder}/benchmark/{frame_name}')
    frame_squares = split_image_into_squares(frame, square_size)
    shrunk_squares, mask = filter_squares(frame_squares, block_coords)
    shrunk_flat = flatten_squares_into_image(shrunk_squares)
    save_image(shrunk_flat, f'{experiment_folder}/shrunk', frame_name)
    save_mask(frame_name, mask, f'{experiment_folder}/masks.csv')

    return frame_name

def sort_and_compress_masks(input_csv, output_compressed_npz):
    # Read the CSV file and store the rows
    rows = []
    with open(input_csv, mode='r', newline='') as file:
        reader = csv.reader(file)
        for row in reader:
            frame_number = int(row[0].split('.')[0])  # Convert '0000.png' to int 0
            binary_values = [int(val) for val in row[1:]]  # Convert binary values to int
            rows.append([frame_number] + binary_values)
    
    # Convert to NumPy array with appropriate smaller data types
    frame_numbers = np.array([row[0] for row in rows], dtype=np.uint16)
    binary_data = np.array([row[1:] for row in rows], dtype=bool)
    
    # Combine frame numbers and binary data into a single structured array
    structured_array = np.empty((len(rows),), dtype=[('frame', np.uint16), ('data', bool, binary_data.shape[1])])
    structured_array['frame'] = frame_numbers
    structured_array['data'] = binary_data
    
    # Sort the array based on the frame number
    sorted_array = np.sort(structured_array, order='frame')
    
    # Save the sorted array to a compressed .npz file
    np.savez_compressed(output_compressed_npz, sorted_data=sorted_array)

if __name__ == '__main__':
    # get parameters from orchestrator
    video_name = os.environ.get('video_name')
    scene_number = os.environ.get('scene_number')
    resolution = os.environ.get('resolution')
    square_size = int(os.environ.get('square_size'))
    to_remove = float(os.environ.get('to_remove'))
    alpha = float(os.environ.get('alpha'))
    smoothing_factor = float(os.environ.get('smoothing_factor'))
    experiment_name = os.environ.get('experiment_name')
    # calculate derivate parameters
    width, height = resolution.split('x')
    width = int(width)
    height = int(height)
    experiment_folder = f'experiments/{experiment_name}'
    frame_names = [frame_name for frame_name in os.listdir(f'{experiment_folder}/benchmark') if frame_name.endswith('.png')]
    temporal_file = f'{experiment_folder}/complexity/reference_TC_blocks.csv'
    spatial_file = f'{experiment_folder}/complexity/reference_SC_blocks.csv'
    focused_folder = f'{experiment_folder}/focus_masks'

    removed_values_coords = get_coordinates_to_remove(temporal_file, spatial_file, width, height, square_size, alpha, to_remove, focused_folder, smoothing_factor)

    with ProcessPoolExecutor() as executor:
        results = []
        for frame_name in frame_names:
            frame_number = int(frame_name.split('.')[0])
            results.append(
                executor.submit(
                    process_frame_server_side, 
                    frame_name, 
                    experiment_folder, 
                    square_size, 
                    removed_values_coords[frame_number]
                )
            )
        # Retrieve results
        processed_frame_names = [future.result() for future in results]

    # order csv by frame number and convert into npz
    sort_and_compress_masks(f'{experiment_folder}/masks.csv', f'{experiment_folder}/masks.npz')
