#!/bin/bash

# SETUP

make_lower_multiple() {
  num=$1
  multiple=$2
  remainder=$((num % multiple))
  if [ $remainder -ne 0 ]; then
    num=$((num - remainder))
  fi
  echo $num
}

run_evca() {
    local input_video_path=$1
    local output_video_path=$2
    local resolution=$3
    local square_size=$4
    local csv_path=$5

    # Get the number of frames in the input video using ffprobe
    local frame_count=$(ffprobe -v error -count_frames -select_streams v:0 -show_entries stream=nb_read_frames -of csv=p=0 "$input_video_path")

    if [[ -f "$output_video_path" ]]; then
        echo "$output_video_path already exists"
    else
        ffmpeg -loglevel warning -i "$input_video_path" -c:v rawvideo -pix_fmt yuv420p "$output_video_path"
        cd
        python3 EVCA/main.py -i "embrace/$output_video_path" -r $resolution -b $square_size -f $frame_count -c $csv_path -bi 1
        cd ~/embrace
    fi
}

video_into_resized_frames() {
    local input_file=$1
    local output_dir=$2
    local resolution=$3
    
    # Check if the output directory already exists, if not create it
    if [[ -d "$output_dir" ]]; then
        echo "$output_dir already exists"
        return 1        
    fi

    mkdir $output_dir
    ffmpeg -loglevel warning -i "$input_file" -vf "fps=24,scale=${resolution}" -q:v 1 -fps_mode passthrough -copyts -start_number 0 "$output_dir/%04d.png"
}

frames_into_video() {
    local input_dir=$1
    local output_file=$2
    local bitrate=$3

    # Check if the output already exists
    if [[ -f "$output_file" ]]; then
        echo "$output_file already exists"
        return 1
    fi

    # Determine if encoding should be lossless or not
    if [[ "$bitrate" == "lossless" ]]; then
        ffmpeg -loglevel warning -framerate 24 -i "$input_dir/%04d.png" -c:v libx265 -x265-params lossless=1 -pix_fmt yuv420p -fps_mode passthrough -copyts "$output_file"
    else
        ffmpeg -loglevel warning -framerate 24 -i "$input_dir/%04d.png" -b:v "$bitrate" -maxrate "$bitrate" -minrate "$bitrate" -bufsize "$bitrate" -c:v libx265 -pix_fmt yuv420p -fps_mode passthrough -copyts "$output_file"
    fi
}

video_into_frames() {
    local input_file=$1
    local output_dir=$2

    # Check if the output directory already exists, if not create it
    if [[ -d "$output_dir" ]]; then
        echo "$output_dir already exists"
        return 1        
    fi

    mkdir $output_dir
    ffmpeg -loglevel warning -i "$input_file" -vf "fps=24" -q:v 1 -fps_mode passthrough -copyts -start_number 0 "$output_dir/%04d.png"
}

# TODO: add more parameters from ProPainter
video_name=$1
scene_number=$2
width=$3
height=$4
square_size=$5
percentage_to_remove=$6
alpha=$7
neighbor_length=$8
ref_stride=$9
subvideo_length=${10}

# check whether width and height are multiples of square_size, otherwise make them
width=$(make_lower_multiple $width $square_size)
height=$(make_lower_multiple $height $square_size)

resolution="${width}x${height}"

# pass parameters to python scripts
export video_name=$video_name
export scene_number=$scene_number
export resolution=$resolution
export square_size=$square_size
export percentage_to_remove=$percentage_to_remove
export alpha=$alpha
export neighbor_length=$neighbor_length
export ref_stride=$ref_stride
export subvideo_length=$subvideo_length

# Iterate over all video files in the input directory, split them into scenes
for file in "videos"/*.{flv,mp4,mov,mkv,avi}; do
    if [[ -f "$file" ]]; then
        python split_video_into_scenes.py $file 0.5 10
    fi
done

# SERVER SIDE

server_start_time=$(date +%s)

# Create the experiment folder
experiment_name="squ_${square_size}_rem_${percentage_to_remove}_alp_${alpha}"
mkdir -p "videos/${video_name}/scene_${scene_number}/"${width}x${height}"/$experiment_name"

# resize scene based on experiment resolution, save into 
video_into_resized_frames "videos/${video_name}/scene_${scene_number}.mp4" "videos/${video_name}/scene_${scene_number}/${width}x${height}/original" "${width}:${height}"

# get reference video from frames
frames_into_video "videos/${video_name}/scene_${scene_number}/"${width}x${height}"/original" "videos/${video_name}/scene_${scene_number}/"${width}x${height}"/reference.mp4" "lossless"

# calculate scene complexities
run_evca "videos/${video_name}/scene_${scene_number}/"${width}x${height}"/reference.mp4" "videos/${video_name}/scene_${scene_number}/"${width}x${height}"/reference.yuv" "${resolution}" "${square_size}" "embrace/videos/${video_name}/scene_${scene_number}/"${width}x${height}"/complexity/reference.csv"

# run script to get smart masks and shrunk frames
python shrink_frames.py

# get bitrate from shrunk size
shrunk_frame="videos/${video_name}/scene_${scene_number}/"${width}x${height}"/$experiment_name/shrunk/0000.png"
shrunk_width=$(ffprobe -v error -select_streams v:0 -show_entries stream=width -of csv=p=0 "$shrunk_frame")
shrunk_height=$(ffprobe -v error -select_streams v:0 -show_entries stream=height -of csv=p=0 "$shrunk_frame")
bitrate=$(python3 calculate_bitrate.py "$shrunk_width" "$shrunk_height")
export bitrate=$bitrate

# get shrunk video from frames
frames_into_video "videos/${video_name}/scene_${scene_number}/"${width}x${height}"/$experiment_name/shrunk" "videos/${video_name}/scene_${scene_number}/"${width}x${height}"/$experiment_name/shrunk.mp4" $bitrate

# get original video from frames
frames_into_video "videos/${video_name}/scene_${scene_number}/"${width}x${height}"/original" "videos/${video_name}/scene_${scene_number}/"${width}x${height}"/$experiment_name/original.mp4" $bitrate

# CLIENT SIDE

client_start_time=$(date +%s)

# get encoded_shrunk frames from video
video_into_frames "videos/${video_name}/scene_${scene_number}/"${width}x${height}"/$experiment_name/shrunk.mp4" "videos/${video_name}/scene_${scene_number}/"${width}x${height}"/$experiment_name/encoded_shrunk"

stretched_file="videos/${video_name}/scene_${scene_number}/"${width}x${height}"/$experiment_name/stretched.mp4"
# Check if the stretched folder already exists, if not create it
    if [[ -f "$stretched_file" ]]; then
        echo "$stretched_file already exists"
    else
        # run client script to get stretched frames
        python stretch_frames.py
        # get stretched video from frames
        frames_into_video "videos/${video_name}/scene_${scene_number}/"${width}x${height}"/$experiment_name/stretched" $stretched_file "lossless"
    fi

# QUALITY MEASUREMENT ORIGINAL

reference_output_path="videos/${video_name}/scene_${scene_number}.mp4"

# compare reference with original video
original_input_path="videos/${video_name}/scene_${scene_number}/"${width}x${height}"/$experiment_name/original.mp4"
original_csv_path="videos/${video_name}/scene_${scene_number}/"${width}x${height}"/$experiment_name/original.csv"
# Check if the output already exists, if not create it
if [[ -f "$original_csv_path" ]]; then
    echo "$original_csv_path already exists"   
else
    ffmpeg-quality-metrics $original_input_path $reference_output_path -m  psnr ssim vmaf -of csv > "$original_csv_path"
fi

# INPAINTING

inpainted_input_path="videos/${video_name}/scene_${scene_number}/"${width}x${height}"/$experiment_name/nei_${neighbor_length}_ref_${ref_stride}_sub_${subvideo_length}.mp4"
# Check if the output already exists, if not create it
if [[ -f "$inpainted_input_path" ]]; then
    echo "$inpainted_input_path already exists"   
else
    # TODO: we can change the mask at each frame, and set masks to alternate the block they keep so that each block has more references.
    cd
    stretched_video_path="embrace/videos/${video_name}/scene_${scene_number}/"${width}x${height}"/$experiment_name/stretched.mp4"
    mask_path="embrace/videos/${video_name}/scene_${scene_number}/"${width}x${height}"/$experiment_name/reconstructed_masks"
    mkdir -p "ProPainter/inputs/video_completion"
    cp $stretched_video_path "ProPainter/inputs/video_completion/stretched.mp4"
    cp -r $mask_path "ProPainter/inputs/video_completion/masks"
    cd ProPainter
    # INPAINTING
    python inference_propainter.py \
        --video inputs/video_completion/stretched.mp4 \
        --mask inputs/video_completion/masks \
        --mask_dilation 0 \
        --neighbor_length ${neighbor_length} \
        --ref_stride ${ref_stride} \
        --subvideo_length ${subvideo_length} \
        --raft_iter 20 \
        --save_frames \
        --fp16

    # move inpainted frames to experiment folder
    cd
    mv -f "ProPainter/results/stretched/frames" "embrace/videos/${video_name}/scene_${scene_number}/"${width}x${height}"/$experiment_name/nei_${neighbor_length}_ref_${ref_stride}_sub_${subvideo_length}"
    cd embrace
    # get inpainted video from frames
    frames_into_video "videos/${video_name}/scene_${scene_number}/"${width}x${height}"/$experiment_name/nei_${neighbor_length}_ref_${ref_stride}_sub_${subvideo_length}" "$inpainted_input_path" "lossless"
fi

# QUALITY MEASUREMENT INPAINTED

# compare reference with inpainted video
inpainted_csv_path="videos/${video_name}/scene_${scene_number}/"${width}x${height}"/$experiment_name/nei_${neighbor_length}_ref_${ref_stride}_sub_${subvideo_length}.csv"
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
rm -r "videos/${video_name}/scene_${scene_number}/"${width}x${height}"/$experiment_name/shrunk"
# delete stretched folder
rm -r "videos/${video_name}/scene_${scene_number}/"${width}x${height}"/$experiment_name/stretched"
# delete inpainted folder
rm -r "videos/${video_name}/scene_${scene_number}/"${width}x${height}"/$experiment_name/nei_${neighbor_length}_ref_${ref_stride}_sub_${subvideo_length}"

cd
cd ProPainter
rm -r results/stretched/
rm -r inputs/video_completion/
