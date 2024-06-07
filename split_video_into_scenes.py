import os
import subprocess
import sys
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

def split_video_into_scenes(video_file: str, threshold: int = 0.9, max_scenes: int = None) -> str:
    
    # Extract folder path and create the folder if it doesn't exist, or exit if it already exists
    folder_path, _ = video_file.rsplit('.', 1)
    if os.path.exists(folder_path):
        print("Folder already exists.")
        return ""
    os.mkdir(folder_path)

    # Open the video file
    cap = cv2.VideoCapture(video_file)
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
        # Resize frame to speed up computation
        height, width = frame.shape[:2]
        aspect_ratio = height / width
        new_height = int(640 * aspect_ratio)
        frame = cv2.resize(frame, (640, new_height))
        if not ret:
            break
        if prev_frame is not None:

            # # Calculate histogram correlation between frames
            # hist = cv2.calcHist([frame], [0], None, [256], [0, 256])
            # prev_hist = cv2.calcHist([prev_frame], [0], None, [256], [0, 256])
            # score = cv2.compareHist(hist, prev_hist, cv2.HISTCMP_CORREL)
            # if (score < threshold) and (cap.get(cv2.CAP_PROP_POS_MSEC) - scene_start > 2000):

            # Calculate SSIM between frames
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
            score, _ = ssim(gray_frame, gray_prev_frame, full=True)
            if (score < threshold) and (cap.get(cv2.CAP_PROP_POS_MSEC) - scene_start > 2000):

            # # Calculate optical flow between previous and current frame
            # gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # gray_prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
            # flow = cv2.calcOpticalFlowFarneback(gray_prev_frame, gray_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            # mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            # score = np.mean(mag)
            # # bring threshold to appropriate order of magnitude
            # threshold *= 10
            # if (score > threshold) and (cap.get(cv2.CAP_PROP_POS_MSEC) - scene_start > 2000):

                print(score)
                # Get times of scene start and end, and save new video between those times
                scene_end = cap.get(cv2.CAP_PROP_POS_MSEC)
                # Save scene using ffmpeg
                output_file = os.path.join(folder_path, f"scene_{scene_number}.mp4")
                ffmpeg_command = [
                    "ffmpeg", "-i", video_file, "-ss", f"{scene_start/1000}", "-to", f"{scene_end/1000}", 
                    "-r", "24", "-c:v", "libx265", "-crf", "20", "-pix_fmt", "yuv420p", "-an", output_file
                ]
                subprocess.run(ffmpeg_command)
                scene_start = scene_end
                scene_number += 1
                scenes_extracted += 1
                if max_scenes is not None and scenes_extracted >= max_scenes:
                    break

        prev_frame = frame.copy()

    cap.release()

    return folder_path

if __name__ == "__main__":
    split_video_into_scenes(
        video_file=sys.argv[1], 
        threshold=float(sys.argv[2]), 
        max_scenes=int(sys.argv[3])
    )

