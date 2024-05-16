# import packages
import os
import sys
import time
from functions import *

start = time.time()

# Get the log file path from environment variable
log_file = os.environ.get('log_file')

# Redirect stdout and stderr to the log file
sys.stdout = open(log_file, 'a')
sys.stderr = open(log_file, 'a')

# Log information
print("Client script started.")

# TODO: check which parameters the client would not have access to, and send them from the server instead of just taking them like this
unprocessed_video_file = os.environ.get('unprocessed_video_file')
unprocessed_video_info = get_video_info(unprocessed_video_file)
width = os.environ.get('width')
height = os.environ.get('height')
scene_similarity_threshold = int(os.environ.get('scene_similarity_threshold'))
video_folder = os.environ.get('video_folder')
square_size = int(os.environ.get('square_size'))
num_processes = int(os.environ.get('num_processes'))

# TODO: now we're passing a predetermined mask, needs a function that takes as input one or more masks, 
# and yields one mask after another then loops back from the first, each time it's called
mask_frame_path = video_folder + '/masks/frame0001.png'
mask_frame = cv2.imread(mask_frame_path)
mask_squares = split_image_into_squares(mask_frame, square_size)
shrunk_frames_folder = video_folder + '/shrunk'
print(mask_frame_path, 'was chosen as static mask for each frame.')

processed_frame_names = process_frames_in_parallel(
    shrunk_frames_folder, 
    process_frame_client_side, 
    num_processes,
    square_size=square_size,
    mask_squares=mask_squares,
    horizontal_stride=int(os.environ.get('horizontal_stride')),
    vertical_stride=int(os.environ.get('vertical_stride'))
)
frame_rate = eval(unprocessed_video_info['average_frame_rate'])
reconstruct_video_from_frames(video_folder + '/stretched', video_folder + '/stretched.avi', frame_rate)
print('shrunk video was stretched based on the mask and saved at:', video_folder + '/stretched.avi')

end = time.time()
print('Client side pre-inpainting process completed in', end - start, 'seconds.')