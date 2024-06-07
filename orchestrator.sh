#!/bin/bash

# SETUP

# pass parameters to python scripts
export video_name=$1
export scene_number=$2
export resolution="${3}x${4}"
export square_size=$6
export horizontal_stride=$7
export vertical_stride=$8
export neighbor_length=$9
export ref_stride=${10}
export subvideo_length=${11}
export bitrate=${12}

# Iterate over all video files in the input directory and split them into scenes
for file in "videos"/*.{flv,mp4,mov,mkv,avi}; do
    if [[ -f "$file" ]]; then
        echo "Detected $file not split into scenes. Splitting..."
        python split_video_into_scenes.py $file 0.5 10
    fi
done

# EXPERIMENT

# Create the experiment folder
experiment_name="squ_${6}_hor_${7}_ver_${8}"
mkdir -p "videos/$1/scene_$2/"${3}x${4}"/$experiment_name"

resize_video() {
    local input_file=$1
    local output_dir=$2
    local resolution=$3
    
    # Check if the output directory already exists, if not create it
    if [[ -d "$output_dir" ]]; then
        echo "$output_dir already exists"
        return 1        
    fi

    mkdir $output_dir
    ffmpeg -i "$input_file" -vf scale="$resolution":force_original_aspect_ratio=increase,crop="$resolution" -start_number 0 "$output_dir/%04d.png"
}
# resize scene based on experiment resolution, save into 
resize_video "videos/$1/scene_$2.mp4" "videos/$1/scene_$2/${3}x${4}/original" "${3}:${4}"

frames_into_video() {
    local input_dir="$1"
    local output_file="$2"
    local bitrate="$3"

    # Check if the output already exists
    if [[ -f "$output_file" ]]; then
        echo "$output_file already exists"
        return 1
    fi

    # Determine if encoding should be lossless or not
    if [[ "$bitrate" == "lossless" ]]; then
        ffmpeg -i "$input_dir/%04d.png" \
               -c:v libx265 -preset ultrafast -x265-params lossless=1 -pix_fmt yuv420p "$output_file"
    else
        ffmpeg -i "$input_dir/%04d.png" \
               -b:v "$bitrate" -maxrate "$bitrate" -minrate "$bitrate" -bufsize "$bitrate" \
               -c:v libx265 -pix_fmt yuv420p "$output_file"
    fi
}
# get original video from frames
frames_into_video videos/$1/scene_$2/"${3}x${4}"/original videos/$1/scene_$2/"${3}x${4}"/original.mp4 ${12}

# run server script to get masks and shrunk frames
python server.py 

# move shrunk and masks into experiment folder
mv -f "videos/$1/scene_$2/"${3}x${4}"/shrunk/" "videos/$1/scene_$2/"${3}x${4}"/$experiment_name/shrunk"
mv -f "videos/$1/scene_$2/"${3}x${4}"/masks/" "videos/$1/scene_$2/"${3}x${4}"/$experiment_name/masks"

# get shrunk video from frames
frames_into_video "videos/$1/scene_$2/"${3}x${4}"/$experiment_name/shrunk" "videos/$1/scene_$2/"${3}x${4}"/$experiment_name/shrunk.mp4" ${12}

# run client script to get stretched frames
python client.py

# get stretched video from frames
frames_into_video "videos/$1/scene_$2/"${3}x${4}"/$experiment_name/stretched" "videos/$1/scene_$2/"${3}x${4}"/$experiment_name/stretched.mp4" "lossless"

# INPAINTING

cd
stretched_video_path="embrace/videos/$1/scene_$2/"${3}x${4}"/$experiment_name/stretched.mp4"
mask_path="embrace/videos/$1/scene_$2/"${3}x${4}"/$experiment_name/masks/0000.png"
cp $stretched_video_path "ProPainter/inputs/video_completion/stretched.mp4"
cp $mask_path "ProPainter/inputs/video_completion/0000.png"
# TODO: we can change the mask at each frame, and set masks to alternate the block they keep so that each block has more references.
cd ProPainter
python inference_propainter.py \
    --video inputs/video_completion/stretched.mp4 \
    --mask inputs/video_completion/0000.png \
    --mask_dilation 0 \
    --neighbor_length $9 \
    --ref_stride ${10} \
    --subvideo_length ${11} \
    --raft_iter 20 \
    --save_frames \
    --fp16

# move inpainted frames to experiment folder
cd
mv -f "ProPainter/results/stretched/frames" "embrace/videos/$1/scene_$2/"${3}x${4}"/$experiment_name/nei_${9}_ref_${10}_sub_${11}"
cd embrace
# get inpainted video from frames
frames_into_video "videos/$1/scene_$2/"${3}x${4}"/$experiment_name/nei_${9}_ref_${10}_sub_${11}" "videos/$1/scene_$2/"${3}x${4}"/$experiment_name/nei_${9}_ref_${10}_sub_${11}.mp4" "lossless"

# QUALITY MEASUREMENT TODO: VMAF scores seem to be wrong... and libvmaf does not have psnr or ssim, move away from libvmaf

# Function to run ffmpeg command and extract metrics values
calculate_metrics() {
    local reference_file=$1
    local distorted_file=$2
    local log_file=$3

    # Default filter_complex for 1080p
    filter_complex="[0:v]scale=1920x1080:flags=bicubic[main]; [1:v]scale=1920x1080:flags=bicubic[ref]; [main][ref]libvmaf=log_path=${log_file}:log_fmt=csv"
    ffmpeg -i "$reference_file" -i "$distorted_file" -filter_complex "$filter_complex" -f null -
}

# # encode reference video lossless TODO: no need, take the scene_n.mp4
# reference_input_path="videos/$1/scene_$2/"${3}x${4}"/original/%04d.png"
# reference_output_path="videos/$1/scene_$2/"${3}x${4}"/reference.mp4"
# # Check if the output already exists, if not create it
# if [[ -f "$reference_output_path" ]]; then
#     echo "$reference_output_path already exists"   
# else
#     ffmpeg -i $reference_input_path -c:v libx265 -crf 0 -pix_fmt yuv420p $reference_output_path
# fi

reference_output_path="videos/$1/scene_$2.mp4"

# compare reference with original video
original_input_path="videos/$1/scene_$2/"${3}x${4}"/original.mp4"
original_csv_path="videos/$1/scene_$2/"${3}x${4}"/original.csv"
# Check if the output already exists, if not create it
if [[ -f "$original_csv_path" ]]; then
    echo "$original_csv_path already exists"   
else
    calculate_metrics "$reference_output_path" "$original_input_path" "$original_csv_path"
fi

# compare reference with inpainted video
inpainted_input_path="videos/$1/scene_$2/"${3}x${4}"/$experiment_name/nei_${9}_ref_${10}_sub_${11}.mp4"
inpainted_csv_path="videos/$1/scene_$2/"${3}x${4}"/$experiment_name/nei_${9}_ref_${10}_sub_${11}_inpainted.csv"
# Check if the output already exists, if not create it
if [[ -f "$inpainted_csv_path" ]]; then
    echo "$inpainted_csv_path already exists"   
else
    calculate_metrics "$reference_output_path" "$inpainted_input_path" "$inpainted_csv_path"
fi

# run metrics script
python collect_metrics.py

# CLEANING UP

# save storage when running many experiments by deleting files and folders that will not be needed anymore
# delete shrunk folder
rm -r "videos/$1/scene_$2/"${3}x${4}"/$experiment_name/shrunk"
# delete stretched folder
rm -r "videos/$1/scene_$2/"${3}x${4}"/$experiment_name/stretched"
# delete inpainted folder
rm -r "videos/$1/scene_$2/"${3}x${4}"/$experiment_name/nei_${9}_ref_${10}_sub_${11}"