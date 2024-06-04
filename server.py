# import packages
import os
from functions import *

video_name = os.environ.get('video_name')
scene_number = os.environ.get('scene_number')
resolution = os.environ.get('resolution')
original_frames_folder = f'videos/{video_name}/scene_{scene_number}/{resolution}/original'

# break frames into blocks, remove blocks based on mask, reconstruct frames without removed blocks, 
processed_frame_names = process_frames_in_parallel(
    original_frames_folder, 
    process_frame_server_side, 
    num_processes=32,
    square_size=int(os.environ.get('square_size')), 
    horizontal_stride=int(os.environ.get('horizontal_stride')),
    vertical_stride=int(os.environ.get('vertical_stride'))
)