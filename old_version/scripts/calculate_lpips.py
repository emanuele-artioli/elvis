import sys
import os
import cv2
import lpips
import pandas as pd

# Initialize the LPIPS model (default: VGG)
loss_fn = lpips.LPIPS(net='vgg')

def calculate_lpips(frame1, frame2):
    # Convert frames from BGR to RGB and normalize to [-1, 1] for LPIPS
    frame1 = (frame1 / 255.0) * 2 - 1
    frame2 = (frame2 / 255.0) * 2 - 1
    # Convert frames to tensors
    frame1_tensor = lpips.im2tensor(frame1)
    frame2_tensor = lpips.im2tensor(frame2)
    # Compute LPIPS score
    return loss_fn(frame1_tensor, frame2_tensor).item()

def process_videos(reference_path, distorted_path, csv_path):
    # Load CSV
    df = pd.read_csv(csv_path)
    
    # Add LPIPS column if not exists
    if 'LPIPS' not in df.columns:
        df['LPIPS'] = None

    # Open both videos
    ref_cap = cv2.VideoCapture(reference_path)
    dist_cap = cv2.VideoCapture(distorted_path)

    frame_index = 0

    while ref_cap.isOpened() and dist_cap.isOpened():
        ref_ret, ref_frame = ref_cap.read()
        dist_ret, dist_frame = dist_cap.read()

        if not ref_ret or not dist_ret:
            break

        # Calculate LPIPS for the current pair of frames
        lpips_score = calculate_lpips(ref_frame, dist_frame)

        # Update the CSV with the new LPIPS score
        df.at[frame_index, 'LPIPS'] = lpips_score
        frame_index += 1

    # Release video captures
    ref_cap.release()
    dist_cap.release()

    # Save updated CSV
    df.to_csv(csv_path, index=False)

if __name__ == "__main__":
    # Arguments passed from the bash script
    reference_video = sys.argv[1]
    distorted_video = sys.argv[2]
    csv_file = sys.argv[3]

    # Check if the CSV file exists
    if not os.path.exists(csv_file):
        print(f"Error: CSV file {csv_file} does not exist.")
        sys.exit(1)

    # Process the videos and update the CSV
    process_videos(reference_video, distorted_video, csv_file)
