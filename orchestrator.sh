#!/bin/bash

# SETUP

# Function to determine the appropriate bitrate for a given width
function determine_bitrate() {
    local original_width=${1}
    local division_factor=${2}
    
    if [ -z "$original_width" ] || [ -z "$division_factor" ]; then
        echo "Usage: determine_bitrate <video_width> <division_factor>"
        return 1
    fi

    local new_width=$((original_width / division_factor))

    local bitrate=0

    # Determine bitrate based on the new width
    if (( new_width <= 320 )); then
        bitrate=145000    # 180p24 - 145 kbps
    elif (( new_width <= 480 )); then
        bitrate=300000   # 270p24 - 300 kbps
    elif (( new_width <= 640 )); then
        bitrate=660000    # 360p24 - 660 kbps
    elif (( new_width <= 960 )); then
        bitrate=1700000   # 540p24 - 1.7 Mbps
    elif (( new_width <= 1280 )); then
        bitrate=2400000   # 720p24 - 2.4 Mbps
    elif (( new_width <= 1920 )); then
        bitrate=4500000   # 1080p24 - 4.5 Mbps
    elif (( new_width <= 2560 )); then
        bitrate=8100000   # 1440p24 - 8.1 Mbps
    else
        bitrate=11600000  # 2160p24 - 11.6 Mbps
    fi

    echo "$bitrate"
}
bitrate=$(determine_bitrate ${3} ${6})

# pass parameters to python scripts
export video_name=${1}
export scene_number=${2}
export resolution="${3}x${4}"
export square_size=${5}
export horizontal_stride=${6}
export vertical_stride=${7}
export neighbor_length=${8}
export ref_stride=${9}
export subvideo_length=${10}
export bitrate=$bitrate

# Iterate over all video files in the input directory and split them into scenes
for file in "videos"/*.{flv,mp4,mov,mkv,avi}; do
    if [[ -f "$file" ]]; then
        echo "Detected $file not split into scenes. Splitting..."
        python split_video_into_scenes.py $file 0.5 10
    fi
done

# EXPERIMENT

# Create the experiment folder
experiment_name="squ_${5}_hor_${6}_ver_${7}"
mkdir -p "videos/${1}/scene_${2}/"${3}x${4}"/$experiment_name"

resize_and_split_video() {
    local input_file=${1}
    local output_dir=${2}
    local resolution=${3}
    
    # Check if the output directory already exists, if not create it
    if [[ -d "$output_dir" ]]; then
        echo "$output_dir already exists"
        return 1        
    fi

    mkdir $output_dir
    ffmpeg -i "$input_file" -vf scale="$resolution" -start_number 0 "$output_dir/%04d.png"
}
# resize scene based on experiment resolution, save into 
resize_and_split_video "videos/${1}/scene_${2}.mp4" "videos/${1}/scene_${2}/${3}x${4}/original" "${3}:${4}"

frames_into_video() {
    local input_dir="${1}"
    local output_file="${2}"
    local bitrate="${3}"

    # Check if the output already exists
    if [[ -f "$output_file" ]]; then
        echo "$output_file already exists"
        return 1
    fi

    # Determine if encoding should be lossless or not
    if [[ "$bitrate" == "lossless" ]]; then
        ffmpeg -framerate 24 -i "$input_dir/%04d.png" -c:v libx265 -x265-params lossless=1 -pix_fmt yuv420p "$output_file"
    else
        ffmpeg -framerate 24 -i "$input_dir/%04d.png" -b:v "$bitrate" -maxrate "$bitrate" -minrate "$bitrate" -bufsize "$bitrate" -c:v libx265 -pix_fmt yuv420p "$output_file"
    fi
}
# get original video from frames
frames_into_video videos/${1}/scene_${2}/"${3}x${4}"/original videos/${1}/scene_${2}/"${3}x${4}"/$experiment_name/original.mp4 $bitrate

# run server script to get masks and shrunk frames
python server.py 

# move shrunk and masks into experiment folder
mv -f "videos/${1}/scene_${2}/"${3}x${4}"/shrunk/" "videos/${1}/scene_${2}/"${3}x${4}"/$experiment_name/shrunk"
mv -f "videos/${1}/scene_${2}/"${3}x${4}"/masks/" "videos/${1}/scene_${2}/"${3}x${4}"/$experiment_name/masks"

# get shrunk video from frames
frames_into_video "videos/${1}/scene_${2}/"${3}x${4}"/$experiment_name/shrunk" "videos/${1}/scene_${2}/"${3}x${4}"/$experiment_name/shrunk.mp4" $bitrate

# run client script to get stretched frames
python client.py

# get stretched video from frames
frames_into_video "videos/${1}/scene_${2}/"${3}x${4}"/$experiment_name/stretched" "videos/${1}/scene_${2}/"${3}x${4}"/$experiment_name/stretched.mp4" "lossless"

# INPAINTING

cd
stretched_video_path="embrace/videos/${1}/scene_${2}/"${3}x${4}"/$experiment_name/stretched.mp4"
mask_path="embrace/videos/${1}/scene_${2}/"${3}x${4}"/$experiment_name/masks/0000.png"
cp $stretched_video_path "ProPainter/inputs/video_completion/stretched.mp4"
cp $mask_path "ProPainter/inputs/video_completion/0000.png"
# TODO: we can change the mask at each frame, and set masks to alternate the block they keep so that each block has more references.
cd ProPainter
python inference_propainter.py \
    --video inputs/video_completion/stretched.mp4 \
    --mask inputs/video_completion/0000.png \
    --mask_dilation 0 \
    --neighbor_length ${8} \
    --ref_stride ${9} \
    --subvideo_length ${10} \
    --raft_iter 20 \
    --save_frames \
    --fp16

# move inpainted frames to experiment folder
cd
mv -f "ProPainter/results/stretched/frames" "embrace/videos/${1}/scene_${2}/"${3}x${4}"/$experiment_name/nei_${8}_ref_${9}_sub_${10}"
cd embrace
# get inpainted video from frames
frames_into_video "videos/${1}/scene_${2}/"${3}x${4}"/$experiment_name/nei_${8}_ref_${9}_sub_${10}" "videos/${1}/scene_${2}/"${3}x${4}"/$experiment_name/nei_${8}_ref_${9}_sub_${10}.mp4" "lossless"

# QUALITY MEASUREMENT

reference_output_path="videos/${1}/scene_${2}.mp4"

# compare reference with original video
original_input_path="videos/${1}/scene_${2}/"${3}x${4}"/$experiment_name/original.mp4"
original_csv_path="videos/${1}/scene_${2}/"${3}x${4}"/$experiment_name/original.csv"
# Check if the output already exists, if not create it
if [[ -f "$original_csv_path" ]]; then
    echo "$original_csv_path already exists"   
else
    ffmpeg-quality-metrics $original_input_path $reference_output_path -m  psnr ssim vmaf -of csv > "$original_csv_path"
fi

# compare reference with inpainted video
inpainted_input_path="videos/${1}/scene_${2}/"${3}x${4}"/$experiment_name/nei_${8}_ref_${9}_sub_${10}.mp4"
inpainted_csv_path="videos/${1}/scene_${2}/"${3}x${4}"/$experiment_name/nei_${8}_ref_${9}_sub_${10}.csv"
# Check if the output already exists, if not create it
if [[ -f "$inpainted_csv_path" ]]; then
    echo "$inpainted_csv_path already exists"   
else
    ffmpeg-quality-metrics $inpainted_input_path $reference_output_path -m  psnr ssim vmaf -of csv > "$inpainted_csv_path"
    # run metrics script
    python collect_metrics.py
fi

# CLEANING UP

# delete shrunk folder
rm -r "videos/${1}/scene_${2}/"${3}x${4}"/$experiment_name/shrunk"
# delete stretched folder
rm -r "videos/${1}/scene_${2}/"${3}x${4}"/$experiment_name/stretched"
# delete inpainted folder
rm -r "videos/${1}/scene_${2}/"${3}x${4}"/$experiment_name/nei_${8}_ref_${9}_sub_${10}"
