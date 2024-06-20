#!/bin/bash

# SETUP

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

# Iterate over all video files in the input directory and split them into scenes
for file in "videos"/*.{flv,mp4,mov,mkv,avi}; do
    if [[ -f "$file" ]]; then
        echo "Detected $file not split into scenes. Splitting..."
        python split_video_into_scenes.py $file 0.5 10
    fi
done

# SERVER SIDE

server_start_time=$(date +%s)

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
    ffmpeg -i "$input_file" -vf "fps=24,scale=${resolution}" -q:v 1 -fps_mode passthrough -copyts -start_number 0 "$output_dir/%04d.png"
}
# resize scene based on experiment resolution, save into 
resize_and_split_video "videos/${1}/scene_${2}.mp4" "videos/${1}/scene_${2}/${3}x${4}/original" "${3}:${4}"

# run server script to get masks and shrunk frames
python server.py 

# move shrunk and masks into experiment folder
mv -f "videos/${1}/scene_${2}/"${3}x${4}"/shrunk/" "videos/${1}/scene_${2}/"${3}x${4}"/$experiment_name/shrunk"
mv -f "videos/${1}/scene_${2}/"${3}x${4}"/masks/" "videos/${1}/scene_${2}/"${3}x${4}"/$experiment_name/masks"

# get bitrate from shrunk size
shrunk_frame="videos/${1}/scene_${2}/"${3}x${4}"/$experiment_name/shrunk/0000.png"
shrunk_width=$(ffprobe -v error -select_streams v:0 -show_entries stream=width -of csv=p=0 "$shrunk_frame")
shrunk_height=$(ffprobe -v error -select_streams v:0 -show_entries stream=height -of csv=p=0 "$shrunk_frame")
bitrate=$(python3 calculate_bitrate.py "$shrunk_width" "$shrunk_height")
export bitrate=$bitrate

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
        ffmpeg -framerate 24 -i "$input_dir/%04d.png" -c:v libx265 -x265-params lossless=1 -pix_fmt yuv420p -q:v 1 -fps_mode passthrough -copyts "$output_file"
    else
        ffmpeg -framerate 24 -i "$input_dir/%04d.png" -b:v "$bitrate" -maxrate "$bitrate" -minrate "$bitrate" -bufsize "$bitrate" -c:v libx265 -pix_fmt yuv420p -q:v 1 -fps_mode passthrough -copyts "$output_file"
    fi
}

# get shrunk video from frames
frames_into_video "videos/${1}/scene_${2}/"${3}x${4}"/$experiment_name/shrunk" "videos/${1}/scene_${2}/"${3}x${4}"/$experiment_name/shrunk.mp4" $bitrate

# get original video from frames
frames_into_video "videos/${1}/scene_${2}/"${3}x${4}"/original" "videos/${1}/scene_${2}/"${3}x${4}"/$experiment_name/original.mp4" $bitrate

# CLIENT SIDE

client_start_time=$(date +%s)

video_into_frames() {
    local input_file=${1}
    local output_dir=${2}

    # Check if the output directory already exists, if not create it
    if [[ -d "$output_dir" ]]; then
        echo "$output_dir already exists"
        return 1        
    fi

    mkdir $output_dir
    ffmpeg -i "$input_file" -vf "fps=24" -q:v 1 -fps_mode passthrough -copyts -start_number 0 "$output_dir/%04d.png"
}
# get encoded_shrunk frames from video
video_into_frames "videos/${1}/scene_${2}/"${3}x${4}"/$experiment_name/shrunk.mp4" "videos/${1}/scene_${2}/"${3}x${4}"/$experiment_name/encoded_shrunk"

# run client script to get stretched frames
python client.py

# get stretched video from frames
frames_into_video "videos/${1}/scene_${2}/"${3}x${4}"/$experiment_name/stretched" "videos/${1}/scene_${2}/"${3}x${4}"/$experiment_name/stretched.mp4" "lossless"

# QUALITY MEASUREMENT ORIGINAL

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

inpainted_input_path="videos/${1}/scene_${2}/"${3}x${4}"/$experiment_name/nei_${8}_ref_${9}_sub_${10}.mp4"
# Check if the output already exists, if not create it
if [[ -f "$inpainted_input_path" ]]; then
    echo "$inpainted_input_path already exists"   
else
    # TODO: we can change the mask at each frame, and set masks to alternate the block they keep so that each block has more references.
    cd
    stretched_video_path="embrace/videos/${1}/scene_${2}/"${3}x${4}"/$experiment_name/stretched.mp4"
    mask_path="embrace/videos/${1}/scene_${2}/"${3}x${4}"/$experiment_name/masks/0000.png"
    mkdir -p "ProPainter/inputs/video_completion"
    cp $stretched_video_path "ProPainter/inputs/video_completion/stretched.mp4"
    cp $mask_path "ProPainter/inputs/video_completion/0000.png"
    cd ProPainter
    # INPAINTING
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
    frames_into_video "videos/${1}/scene_${2}/"${3}x${4}"/$experiment_name/nei_${8}_ref_${9}_sub_${10}" "$inpainted_input_path" "lossless"
fi

# QUALITY MEASUREMENT INPAINTED

# compare reference with inpainted video
inpainted_csv_path="videos/${1}/scene_${2}/"${3}x${4}"/$experiment_name/nei_${8}_ref_${9}_sub_${10}.csv"
# Check if the output already exists, if not create it
if [[ -f "$inpainted_csv_path" ]]; then
    echo "$inpainted_csv_path already exists"   
else
    ffmpeg-quality-metrics $inpainted_input_path $reference_output_path -m  psnr ssim vmaf -of csv > "$inpainted_csv_path"
    # calculate time elapsed
    end_time=$(date +%s)
    export server_start_time=$server_start_time
    export client_start_time=$client_start_time
    export end_time=$end_time
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

cd
cd ProPainter
rm -r results/stretched/
rm -r inputs/video_completion/
