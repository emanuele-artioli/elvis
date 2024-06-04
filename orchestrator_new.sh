#!/bin/bash

# Function to call the Python script for detecting frame changes
detect_frame_changes() {
    local input_file=$1
    local output_file=$2
    local threshold=$3

    python3 detect_frame_changes.py "$input_file" "$output_file" "$threshold"
}

# Function to extract a portion of a video
extract_video_segment() {
    local input_file=$1
    local start_time=$2
    local end_time=$3
    local output_file="${input_file}_${start_time}_${end_time}.mp4"

    # Convert milliseconds to seconds
    local start_seconds=$(echo "scale=3; $start_time/1000" | bc)
    local end_seconds=$(echo "scale=3; $end_time/1000" | bc)
    local duration=$(echo "scale=3; $end_seconds - $start_seconds" | bc)

    # Use ffmpeg to extract the segment
    ffmpeg -i "$input_file" -ss "$start_seconds" -t "$duration" -c:v libx265 -crf 0 "$output_file"
}

detect_frame_changes videos/Tears_of_Steel_1080p.mov videos/Tears_of_Steel_1080p.csv 0.9