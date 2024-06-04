import os
import shutil
import subprocess
import sys
import cv2

def split_video_into_scenes(video_file: str, threshold: int = 100, max_scenes: int = None) -> str:
    
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
                output_file = os.path.join(folder_path, f"scene_{scene_number}.mp4")
                # Silently execute ffmpeg command
                ffmpeg_command = ["ffmpeg", "-i", video_file, "-ss", f"{scene_start/1000}", "-to", f"{scene_end/1000}", "-c:v", 'libx265', "-crf", "0", "-pix_fmt", "yuv420p", "-an", output_file]
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

if __name__ == "__main__":
    split_video_into_scenes(
        video_file=sys.argv[1], 
        threshold=int(sys.argv[2]), 
        max_scenes=int(sys.argv[3])
    )