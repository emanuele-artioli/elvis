from concurrent.futures import ProcessPoolExecutor
import json
import os
import shutil
import subprocess
from typing import Union
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim


def get_video_info(video_file: str) -> dict:
    """
    Retrieves specific information about a video file using FFprobe.

    Parameters:
    - video_path (str): Path to the video file.

    Returns:
    - dict: A dictionary containing the extracted video information,
      including codec details, dimensions, frame rate, duration, bit rate,
      encoder information, file name, format name, and size.

    Example:
    >>> video_info = get_video_info("example.mp4")
    >>> print(video_info)
    {'codec_name': 'h264', 'codec_tag_string': 'avc1', 'width': 1920, 'height': 1080,
     'average_frame_rate': '30/1', 'duration': '128.550000', 'bit_rate': '1234567',
     'bits_per_raw_sample': None, 'encoder': 'Lavc', 'file_name': 'example.mp4',
     'format_name': 'mov,mp4,m4a,3gp,3g2,mj2', 'size': '1234567'}
    """
    # FFprobe command to get specific video information in JSON format
    ffprobe_cmd = [
        'ffprobe', '-v', 'error', '-print_format', 'json', '-show_format',
        '-show_streams', '-select_streams', 'v:0', video_file
    ]

    # Execute FFprobe command and capture output
    result = subprocess.run(ffprobe_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Check if FFprobe command was successful
    if result.returncode != 0:
        print("Error: Failed to execute FFprobe command")
        return None

    # Parse JSON output
    video_info = json.loads(result.stdout)

    # Extract specific information
    info_dict = {
        'codec_name': video_info['streams'][0]['codec_name'],
        'codec_tag_string': video_info['streams'][0]['codec_tag_string'],
        'width': video_info['streams'][0]['width'],
        'height': video_info['streams'][0]['height'],
        'average_frame_rate': video_info['streams'][0]['avg_frame_rate'],
        'duration': video_info['format']['duration'],
        'bit_rate': video_info['format']['bit_rate'],
        'bits_per_raw_sample': video_info['streams'][0].get('bits_per_raw_sample', None),
        'encoder': video_info['streams'][0].get('encoder', None),
        'file_name': video_info['format']['filename'],
        'format_name': video_info['format']['format_name'],
        'size': video_info['format']['size']
    }

    return info_dict

def split_video_into_scenes(video_path: str, resolution: tuple, threshold: int = 100, max_scenes: int = None) -> str:
    """
    Split a video into scenes based on the difference between consecutive frames.

    Args:
    - video_path (str): Path to the input video file.
    - threshold (int): Threshold value for detecting scene changes. Default is 100.
    - max_scenes (int): Maximum number of scenes to extract. Default is None (extract all scenes).

    Returns:
    - str: Path to the folder containing the extracted scenes.

    This function reads a video file frame by frame, calculates the absolute difference between consecutive frames,
    and detects scene changes based on the mean difference exceeding a threshold. It saves each scene as a separate
    video file in a folder named after the input video file.

    """

    # Extract folder path and create the folder if it doesn't exist
    folder_path, _ = video_path.rsplit('.', 1)
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
    os.mkdir(folder_path)

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return ""

    prev_frame = None
    scene_start = 0
    scene_number = 1
    scenes_extracted = 0

    # Parse the video frame by frame
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if prev_frame is not None:
            # Calculate absolute difference between frames
            diff = cv2.absdiff(prev_frame, frame)
            mean_diff = diff.mean()
            # Detect scene changes based on mean difference exceeding the threshold
            if mean_diff > threshold:
                # Get times of scene start and end, and save new video between those times
                scene_end = cap.get(cv2.CAP_PROP_POS_MSEC)
                # Save scene using ffmpeg
                output_file = os.path.join(folder_path, f"scene_{scene_number}.avi")
                # Silently execute ffmpeg command
                resolution_str = f'{resolution[0]}:{resolution[1]}'
                ffmpeg_command = ["ffmpeg", "-i", video_path, "-ss", f"{scene_start/1000}", "-to", f"{scene_end/1000}", "-c:v", 'ffvhuff', '-vf', f'scale={resolution_str}:force_original_aspect_ratio=increase,crop={resolution_str}', "-pix_fmt", "yuv420p", "-an", output_file]
                with open(os.devnull, 'w') as null_file:
                    subprocess.run(ffmpeg_command, stdout=null_file, stderr=null_file)
                scene_start = scene_end
                scene_number += 1
                scenes_extracted += 1
                if max_scenes is not None and scenes_extracted >= max_scenes:
                    break

        prev_frame = frame.copy()

    cap.release()

    return folder_path

def encode_video(input_file: str, output_file: str, vcodec: str = 'libx265', resolution: tuple = None, preset: str = None, bitrate: str = None, crf: str = None) -> str:
    """
    Encode a video file using FFmpeg with specified parameters.

    Args:
    - input_file (str): Path to the input video file.
    - output_file (str): Path to the output video file.
    - vcodec (str): Video codec to be used for encoding. Default is 'libx265'.
    - resolution (tuple): Desired resolution of the output video in the format (width, height). Default is None.
    - preset (str): Preset for encoding. Default is None.
    - bitrate (str): Bitrate for encoding. Default is None.
    - crf (str): Constant rate factor for encoding. Default is None.

    Returns:
    - str: Message indicating success or failure of video encoding.

    This function encodes a video file using FFmpeg with the specified parameters.
    It supports setting the video codec, resolution, preset, bitrate, and constant rate factor (CRF).
    The output file will be created with the specified path.

    """

    try:
        # Create output folder if it doesn't exist
        output_folder, _ = os.path.split(output_file)
        if not os.path.exists(output_folder):
            os.makedirs(output_folder, exist_ok=True)

        command = ['ffmpeg', '-y', '-i', input_file, '-c:v', vcodec]

        if resolution:
            resolution_str = f'{resolution[0]}:{resolution[1]}'
            command.extend(['-vf', f'scale={resolution_str}:force_original_aspect_ratio=increase,crop={resolution_str}'])

        if crf:
            command.extend(['-crf', crf])
        elif preset:
            command.extend(['-preset', preset])
        elif bitrate:
            command.extend(['-b:v', bitrate])
        
        command.append(output_file)

        # Execute FFmpeg command silently
        with open(os.devnull, 'w') as null_file:
            subprocess.run(command, stdout=null_file, stderr=null_file, check=True)

        return "Video encoding successful!"
    except subprocess.CalledProcessError as e:
        return f"Video encoding failed: {e}"

def encode_video(video_path, output_file):
    ffmpeg_command = ["ffmpeg", "-i", video_path, "-c:v", 'ffvhuff', "-pix_fmt", "yuv420p", "-an", output_file]
    with open(os.devnull, 'w') as null_file:
        subprocess.run(ffmpeg_command, stdout=null_file, stderr=null_file)
    return "Video encoded successfully."

# TODO: I was enhancing these functions with chatgpt's best practices, continue from here
def split_video_into_frames(video_path: str, frames_folder: str) -> str:
    
    # delete folder is if exists, then create it
    if os.path.exists(frames_folder):
        shutil.rmtree(frames_folder)
    os.mkdir(frames_folder)
    ffmpeg_command = f'ffmpeg -i {video_path} {frames_folder}/frame%04d.png'
    with open(os.devnull, 'w') as null_file:
        subprocess.run(ffmpeg_command, stdout=null_file, stderr=null_file, shell=True)
    return frames_folder

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

def filter_squares(squares: np.array, filter_factor: int, inpaint_white: bool = True) -> Union[np.array, np.array]:
    """
    Filter squares based on their indices.

    Args:
    - squares: numpy array with shape [n, m, l, l, c] containing the squares

    Returns:
    - filtered_squares: numpy array with shape [n_filtered, m_filtered, l, l, c] containing the filtered squares
    - filter_mask: numpy array with shape [n, m] indicating which blocks were kept (1) or filtered out (0)
    """
    n, m, _, _, _ = squares.shape
    if inpaint_white:
        filter_mask = np.ones(squares.shape, dtype=int) * 255
        filter_mask[::filter_factor, ::filter_factor] = 0
    else:
        filter_mask = np.zeros(squares.shape, dtype=int)
        filter_mask[::filter_factor, ::filter_factor] = 255

    filtered_squares = squares[::filter_factor, ::filter_factor]
    # filtered_squares = squares[filter_mask == 1]

    return filtered_squares, filter_mask

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

def process_frame_server_side(frame_name, frames_folder, square_size, filter_factor):
    frame = cv2.imread(frames_folder + '/' + frame_name)
    squared_frame = split_image_into_squares(frame, square_size)
    filtered_squares, mask_squares = filter_squares(squared_frame, filter_factor)
    filtered_flattened = flatten_squares_into_image(filtered_squares)
    mask_flattened = flatten_squares_into_image(mask_squares)
    parent_folder, _ = frames_folder.rsplit('/', 1)
    save_image(filtered_flattened, parent_folder + '/' + 'shrunk', frame_name)
    save_image(mask_flattened, parent_folder + '/' + 'masks', frame_name)
    return frame_name

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

def reconstruct_video_from_frames(frames_folder, output_video_path, frame_rate=30):

    # If the video file exists, delete it
    if os.path.exists(output_video_path):
        os.remove(output_video_path)

    # Construct ffmpeg command
    cmd = ['ffmpeg', '-framerate', str(frame_rate), '-pattern_type', 'glob', '-i', f'{frames_folder}/*.png',
                  '-c:v', 'ffvhuff', '-pix_fmt', 'yuv420p', output_video_path]

    # Execute FFmpeg command silently
    with open(os.devnull, 'w') as null_file:
        subprocess.run(cmd, stdout=null_file, stderr=null_file, check=True)

def calculate_mse(img1, img2):
    err = np.sum((img1.astype("float") - img2.astype("float")) ** 2)
    err /= float(img1.shape[0] * img1.shape[1])
    return err

def calculate_psnr(img1, img2):
    mse = calculate_mse(img1, img2)
    if mse == 0:
        return float('inf')
    PIXEL_MAX = 255.0
    return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))

def calculate_ssim(img1, img2):
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    return ssim(img1_gray, img2_gray)

def calculate_vmaf(distorted_video_path, original_video_path, model_path):
    command = [
        'ffmpeg',
        '-i', distorted_video_path,
        '-i', original_video_path,
        '-filter_complex libvmaf -f null -'
    ]
    result = subprocess.run(' '.join(command), shell=True, capture_output=True, text=True)
    print(result)
    output = result.stdout
    print(output)
    vmaf_score = json.loads(output)['aggregate']['VMAF_score']
    return vmaf_score

def compare_videos(video1_path, video2_path, model_path):
    cap1 = cv2.VideoCapture(video1_path)
    cap2 = cv2.VideoCapture(video2_path)
    frame_count = min(int(cap1.get(cv2.CAP_PROP_FRAME_COUNT)), int(cap2.get(cv2.CAP_PROP_FRAME_COUNT)))
    mse_total = 0
    psnr_total = 0
    ssim_total = 0
    # vmaf_score = calculate_vmaf(video1_path, video2_path, model_path)
    vmaf_score = 0
    for _ in range(frame_count):
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        if not (ret1 and ret2):
            break
        mse_total += calculate_mse(frame1, frame2)
        psnr_total += calculate_psnr(frame1, frame2)
        ssim_total += calculate_ssim(frame1, frame2)
    cap1.release()
    cap2.release()
    mse_avg = mse_total / frame_count
    psnr_avg = psnr_total / frame_count
    ssim_avg = ssim_total / frame_count
    return mse_avg, psnr_avg, ssim_avg, vmaf_score

def stretch_frame(frame: np.array, mask: np.array, filter_factor: int) -> np.array:
    
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
            # put it into the corresponding mask white block by multiplying its indices by the filter factor
            stretched_frame[int(i * filter_factor), int(j * filter_factor), :, :, :] = frame_block
    return stretched_frame

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

def process_frame_client_side(frame_name, frames_folder, square_size, mask_squares, filter_factor):
    frame = cv2.imread(frames_folder + '/' + frame_name)
    squares = split_image_into_squares(frame, square_size)
    stretched_squares = stretch_frame(squares, mask_squares, filter_factor)
    stretched_flattened = flatten_squares_into_image(stretched_squares)
    parent_folder, _ = frames_folder.rsplit('/', 1)
    save_image(stretched_flattened, parent_folder + '/stretched', frame_name)
    return frame_name

def process_frame_client_side(frame_name, frames_folder, square_size, mask_squares, horizontal_stride, vertical_stride):
    frame = cv2.imread(frames_folder + '/' + frame_name)
    squared_frame = split_image_into_squares(frame, square_size)
    stretched_squares = stretch_frame(squared_frame, mask_squares, horizontal_stride, vertical_stride)
    stretched_flattened = flatten_squares_into_image(stretched_squares)
    parent_folder, _ = frames_folder.rsplit('/', 1)
    save_image(stretched_flattened, parent_folder + '/stretched', frame_name)
    return frame_name

# TODO: take this out and into the shell script, also that should fix the error that this gives about using source (or conda)
def inpaint_video(input_video_path, mask_path, output_video_path):
    command = [
        'source activate propainter &&',
        'cd ProPainter &&',
        'cp', input_video_path, 'inputs/video_completion/stretched.mp4 &&',
        'cp', mask_path, 'inputs/video_completion/frame0001.png &&',
        'python inference_propainter.py --video inputs/video_completion/stretched.mp4 --mask inputs/video_completion/frame0001.png --fp16 &&',
        'mv results/stretched/inpaint_out.mp4', output_video_path
    ]

    result = subprocess.run(' '.join(command), shell=True, capture_output=True, text=True)
    print(result)
    return None

