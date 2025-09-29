import time
import shutil
import os
import subprocess
from pathlib import Path
import cv2

# SERVER SIDE

# Start recording time
start=time.time()

# resize video based on required resolution, save raw and frames for reference
reference_video="davis_test/bear.mp4"
width, height=640, 480
framerate=cv2.VideoCapture(reference_video).get(cv2.CAP_PROP_FPS)
os.makedirs("reference_frames", exist_ok=True)
os.system(f"ffmpeg -hide_banner -loglevel error -i {reference_video} -vf scale={width}:{height} -c:v rawvideo -pix_fmt yuv420p reference_raw.yuv")
os.system(f"ffmpeg -hide_banner -loglevel error -video_size {width}x{height} -r {framerate} -pixel_format yuv420p -i reference_raw.yuv -q:v 2 reference_frames/frame_%05d.jpg")

# Calculate scene complexity with EVCA
square_size=20
frame_count=len(os.listdir("reference_frames"))
os.chdir("../")
os.system(f"python EVCA/main.py -i elvis/reference_raw.yuv -r {width}x{height} -b {square_size} -f {frame_count} -c elvis/evca/evca.csv -bi 1")

# Calculate ROI with UFO
UFO_folder="UFO/datasets/elvis/image/reference_frames"
if os.path.exists(UFO_folder):
    shutil.rmtree(UFO_folder)
shutil.copytree("elvis/reference_frames", UFO_folder)
os.chdir("UFO")
os.system(f"python test.py --model='weights/video_best.pth' --data_path='datasets/elvis' --output_dir='VSOD_results/wo_optical_flow/elvis' --task='VSOD'")
os.chdir("../")
# Create UFO_masks directory
os.makedirs("elvis/UFO_masks", exist_ok=True)
# Move mask files
mask_source = "UFO/VSOD_results/wo_optical_flow/elvis/reference_frames"
if os.path.exists(mask_source):
    for mask_file in os.listdir(mask_source):
        shutil.move(f"{mask_source}/{mask_file}", "elvis/UFO_masks/")
os.chdir("elvis")

