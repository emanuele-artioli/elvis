# import packages
import os
import sys
from functions import *

# Get the log file path from environment variable
log_file = os.environ.get('log_file')

# Redirect stdout and stderr to the log file
sys.stdout = open(log_file, 'a')
sys.stderr = open(log_file, 'a')

# Log information
print("Server script started.")

# get video info
unprocessed_video_file = os.environ.get('unprocessed_video_file')
unprocessed_video_info = get_video_info(unprocessed_video_file)
print(f'{unprocessed_video_file} info: {unprocessed_video_info}')

# split video into scenes with parameters from orchestrator
scene_similarity_threshold = int(os.environ.get('scene_similarity_threshold'))
max_scenes = int(os.environ.get('max_scenes'))
width = os.environ.get('width')
height = os.environ.get('height')
video_folder = split_video_into_scenes(
    unprocessed_video_file, 
    (width, height),
    scene_similarity_threshold,
    max_scenes
)
print(f'{unprocessed_video_file} was split into scenes and stored in {video_folder}')

# choose scene
scene_number = os.environ.get('scene_number')
scene_file = f'{video_folder}/scene_{scene_number}.avi'
# split scene into frames
frames_folder, _ = scene_file.rsplit('.', 1)
split_video_into_frames(scene_file, frames_folder)
print(f'Scene number {scene_number} was split into frames and stored in {frames_folder}')

# break frames into blocks, remove blocks based on mask, reconstruct frames without removed blocks, 
processed_frame_names = process_frames_in_parallel(
    frames_folder, 
    process_frame_server_side, 
    int(os.environ.get('num_processes')),
    square_size=int(os.environ.get('square_size')), 
    filter_factor=int(os.environ.get('filter_factor'))
)

# reconstruct video from shrunk frames
frame_rate = eval(unprocessed_video_info['average_frame_rate'])
reconstruct_video_from_frames(video_folder + '/shrunk', video_folder + '/shrunk.avi', frame_rate)
print('Frames have been processed and shrunk video was saved as' + video_folder + '/shrunk.avi')

print('Server side process completed.')