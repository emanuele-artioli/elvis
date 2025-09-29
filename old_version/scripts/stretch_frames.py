from concurrent.futures import ProcessPoolExecutor
import csv
import os
import cv2
import numpy as np

def decompress_and_save_as_csv(input_compressed_npz, output_csv):
    # Load the compressed .npz file
    data = np.load(input_compressed_npz)
    sorted_array = data['sorted_data']
    
    # Convert frame numbers back to '0000.png' format
    rows = []
    for row in sorted_array:
        frame_number = f'{row[0]:05}.png'
        binary_values = [int(val) for val in row[1]]
        rows.append([frame_number] + binary_values)
    
    # Write the rows to the CSV file
    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(rows)

def create_mask_from_binary(binary_representation, block_size, image_size):
    # invert the image width and height since that is how numpy interprets them 
    width, height = image_size
    num_blocks_y = height // block_size
    num_blocks_x = width // block_size

    # Initialize an empty mask image
    mask = np.zeros((height, width), dtype=np.uint8)

    # Fill in the mask based on the binary representation
    for idx, binary_value in enumerate(binary_representation):
        y = (idx // num_blocks_x) * block_size
        x = (idx % num_blocks_x) * block_size
        color = 255 if binary_value else 0
        mask[y:y+block_size, x:x+block_size] = color

    return mask

def convert_binary_to_masks(input_csv, output_folder, block_size, image_size):
    os.makedirs(output_folder, exist_ok=True)

    with open(input_csv, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            frame_number = row[0].split('.')[0]  # Extract the frame number from '0000.png'
            binary_representation = list(map(int, row[1:]))
            
            # Reconstruct the mask image
            mask = create_mask_from_binary(binary_representation, block_size, image_size)
            
            # Save the reconstructed mask image
            output_path = os.path.join(output_folder, f'{frame_number}.png')
            cv2.imwrite(output_path, mask)

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

def combine_squares_with_mask(new_squares: np.array, mask_squares: np.array, l: int) -> np.array:
    """
    Combine new_squares with mask_squares where each block of new_squares is placed into a white block of mask_squares row by row.

    Args:
    - new_squares: numpy array with shape [num_rows, num_cols, l, l, c] containing squares without removed blocks
    - mask_squares: numpy array with shape [n, m, l, l, c] indicating where blocks were removed (white) and unchanged areas (black)
    - l: integer representing the side length of each square

    Returns:
    - numpy array with shape [n, m, c] representing the combined image with modified blocks placed into mask_squares
    """
    num_rows, num_cols, l, _, c = new_squares.shape
    # flatten new_squares so that blocks can be taken as a list
    new_squares = new_squares.reshape(num_rows * num_cols, l, l, c)
    n, m, _, _, _ = mask_squares.shape

    # Iterate through mask blocks, and if you find a white one, replace it with the next block from new_squares
    square_index = 0
    for i in range(n):
        for j in range(m):
            if np.mean(mask_squares[i, j]) == 0:
                mask_squares[i, j] = new_squares[square_index]
                square_index += 1

    return mask_squares

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

def pad_image_to_match(image1, image2):
    """
    Pads image1 with black bars to match the dimensions of image2 if image1 is smaller.
    
    Parameters:
    image1 (numpy.ndarray): The first image (which may be smaller).
    image2 (numpy.ndarray): The second image (the target size).
    
    Returns:
    numpy.ndarray: The padded version of image1 to match the dimensions of image2.
    """
    height1, width1 = image1.shape[:2]
    height2, width2 = image2.shape[:2]
    
    # Determine padding needed
    pad_height = max(0, height2 - height1)
    pad_width = max(0, width2 - width1)
    
    # Pad the image with zeros (black bars)
    padded_image = np.pad(image1, ((0, pad_height), (0, pad_width), (0, 0)), mode='constant', constant_values=0)
    
    return padded_image

def save_image(frame: np.array, output_folder: str, file_name: str) -> None:

    # Create the output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    # save filtered frame, overwrite if it exists
    cv2.imwrite(output_folder + '/' + file_name, frame, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])

def process_frame_client_side(frame_name, experiment_folder, square_size):

    frame = cv2.imread(f'{experiment_folder}/shrunk_decoded/{frame_name}')
    frame_squares = split_image_into_squares(frame, square_size)
    mask = cv2.imread(f'{experiment_folder}/decoded_masks/{frame_name}')
    mask_squares = split_image_into_squares(mask, square_size)
    stretched_squares = combine_squares_with_mask(frame_squares, mask_squares, square_size)
    stretched_flat = flatten_squares_into_image(stretched_squares)
    stretched_flat = pad_image_to_match(stretched_flat, mask)
    save_image(stretched_flat, f'{experiment_folder}/stretched', frame_name)

    return frame_name

if __name__ == '__main__':
    # get parameters from orchestrator
    video_name = os.environ.get('video_name')
    scene_number = os.environ.get('scene_number')
    resolution = os.environ.get('resolution')
    square_size = int(os.environ.get('square_size'))
    to_remove = float(os.environ.get('to_remove'))
    alpha = float(os.environ.get('alpha'))
    # smoothing_factor = float(os.environ.get('smoothing_factor'))
    experiment_name = os.environ.get('experiment_name')
    # calculate derivate parameters
    width, height = resolution.split('x')
    width = int(width)
    height = int(height)
    experiment_folder = f'experiments/{experiment_name}'
    decoded_frames = f'{experiment_folder}/shrunk_decoded'
    frame_names = [frame_name for frame_name in os.listdir(decoded_frames) if frame_name.endswith('.png')]
    compressed_masks = f'{experiment_folder}/masks.npz'

    # Decompress and reconstruct CSV
    decoded_masks = f'{experiment_folder}/decoded_masks.csv'
    decompress_and_save_as_csv(compressed_masks, decoded_masks)

    # Convert binary to masks
    convert_binary_to_masks(decoded_masks, f'{experiment_folder}/decoded_masks', square_size, (width, height))

    with ProcessPoolExecutor() as executor:
        results = []
        for frame_name in frame_names:
            frame_number = int(frame_name.split('.')[0])
            results.append(
                executor.submit(
                    process_frame_client_side, 
                    frame_name, 
                    experiment_folder, 
                    square_size
                )
            )
        # Retrieve results
        processed_frame_names = [future.result() for future in results]