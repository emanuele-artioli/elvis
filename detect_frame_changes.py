# import cv2
# import sys
# import csv
# from skimage.metrics import structural_similarity as ssim
# import numpy as np
# from concurrent.futures import ProcessPoolExecutor

# def calculate_ssim(prev_frame, frame):
#     gray_prev = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
#     gray_current = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     score, diff = ssim(gray_prev, gray_current, full=True)
#     diff = (diff * 255).astype("uint8")
#     return score

# def detect_frame_changes(input_file, output_file, threshold=0.5, max_workers=4):
#     ''' function that parses a video frame by frame, and if two consecutive frames are too different,
#         appends their scene number, timestamp, and difference to the output file in CSV format '''
#     cap = cv2.VideoCapture(input_file)
#     if not cap.isOpened():
#         print("Error: Could not open video file.")
#         sys.exit(1)

#     frame_rate = cap.get(cv2.CAP_PROP_FPS)
#     prev_frames = [None] * max_workers
#     frame_number = 0
#     scene_number = 0
#     changes = []

#     def process_frames(worker_id):
#         nonlocal frame_number, scene_number, changes
#         while True:
#             cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
#             ret, frame = cap.read()
#             if not ret:
#                 break
#             current_time = (frame_number / frame_rate) * 1000  # Convert to milliseconds
#             if prev_frames[worker_id] is not None:
#                 score = cv2.absdiff(prev_frames[worker_id], frame).mean() # calculate_ssim(prev_frames[worker_id], frame)
#                 if score < threshold:
#                     changes.append((scene_number, current_time, score))
#                     scene_number += 1
#             prev_frames[worker_id] = frame
#             frame_number += max_workers

#     with ProcessPoolExecutor(max_workers=max_workers) as executor:
#         executor.map(process_frames, range(max_workers))

#     cap.release()

#     with open(output_file, 'w', newline='') as f:
#         writer = csv.writer(f)
#         writer.writerow(["Scene Number", "Timestamp (ms)", "Difference"])
#         writer.writerows(changes)

# if __name__ == "__main__":
#     input_file = sys.argv[1]
#     output_file = sys.argv[2]
#     threshold = float(sys.argv[3])
#     detect_frame_changes(input_file, output_file, threshold, 32)

import sys
import cv2
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

# Function to calculate histogram difference between two frames
def histogram_diff(frame1, frame2):
    hist1 = cv2.calcHist([frame1], [0], None, [256], [0, 256])
    hist2 = cv2.calcHist([frame2], [0], None, [256], [0, 256])
    return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)

# Function to process a chunk of frames and detect scene changes
def process_chunk(frames, threshold, start_frame_idx):
    scene_changes = []
    for i in range(len(frames) - 1):
        frame1 = frames[i]
        frame2 = frames[i + 1]
        
        diff = histogram_diff(frame1, frame2)
        
        if diff < threshold:
            scene_changes.append(start_frame_idx + i + 1)
    
    return scene_changes

def main(video_path, threshold):
    cap = cv2.VideoCapture(video_path)
    frames = []
    all_scene_changes = []

    with ThreadPoolExecutor() as executor:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)

            if len(frames) >= 2:
                chunk_frames = frames.copy()
                frames = [frames[-1]]
                
                future = executor.submit(process_chunk, chunk_frames, threshold, len(all_scene_changes))
                scene_changes = future.result()
                all_scene_changes.extend(scene_changes)

    cap.release()
    
    # Save scene changes timestamps to a csv file
    df = pd.DataFrame(data={'Timestamp': all_scene_changes})
    df.to_csv(sys.argv[2], index=False)

if __name__ == '__main__':
    video_path = sys.argv[1]
    threshold = float(sys.argv[3])
    main(video_path, threshold)
