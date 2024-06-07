# import packages
import os
from functions import *

video_name = os.environ.get('video_name')
scene_number = os.environ.get('scene_number')
resolution = os.environ.get('resolution')
square_size = int(os.environ.get('square_size'))
horizontal_stride = int(os.environ.get('horizontal_stride'))
vertical_stride = int(os.environ.get('vertical_stride'))

# TODO: now we're passing a predetermined mask, needs a function that takes as input one or more masks, 
# and yields one mask after another then loops back from the first, each time it's called

mask_frame_path = f'videos/{video_name}/scene_{scene_number}/{resolution}/squ_{square_size}_hor_{horizontal_stride}_ver_{vertical_stride}/masks/0000.png'
mask_frame = cv2.imread(mask_frame_path)
mask_squares = split_image_into_squares(mask_frame, square_size)
shrunk_frames_folder = f'videos/{video_name}/scene_{scene_number}/{resolution}/squ_{square_size}_hor_{horizontal_stride}_ver_{vertical_stride}/shrunk'

processed_frame_names = process_frames_in_parallel(
    shrunk_frames_folder, 
    process_frame_client_side, 
    num_processes=32,
    square_size=square_size,
    mask_squares=mask_squares,
    horizontal_stride=horizontal_stride,
    vertical_stride=vertical_stride
)