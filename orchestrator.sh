#!/bin/bash

# source /etc/profile.d/opt-local.sh

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
        cd ..
        python3 EVCA/main.py -i "embrace/$output_video_path" -r $resolution -b $square_size -f $frame_count -c $csv_path -bi 1
        cd embrace
    fi
}

video_into_resized_frames() {
    local input_file=$1
    local output_dir=$2
    local width=$3
    local height=$4

    # Check if the output directory already exists, if not create it
    if [[ -d "$output_dir" ]]; then
        echo "$output_dir already exists"
        return 1        
    fi

    mkdir $output_dir

    # Apply scale and crop filters to center and crop the image
    ffmpeg -loglevel warning -i "$input_file" \
        -vf "scale=${width}:${height}:force_original_aspect_ratio=increase,crop=${width}:${height}" \
        -q:v 1 -fps_mode passthrough -copyts -start_number 0 "$output_dir/%05d.png"
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
        ffmpeg -loglevel warning -framerate 24 -i "$input_dir/%05d.png" -c:v libx265 -x265-params lossless=1 -pix_fmt yuv420p -fps_mode passthrough -copyts "$output_file"
    else
        ffmpeg -loglevel warning -framerate 24 -i "$input_dir/%05d.png" -b:v "$bitrate" -maxrate "$bitrate" -minrate "$bitrate" -bufsize "$bitrate" -c:v libx265 -pix_fmt yuv420p -fps_mode passthrough -copyts "$output_file"
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
    ffmpeg -loglevel warning -i "$input_file" -vf "fps=24" -q:v 1 -fps_mode passthrough -copyts -start_number 0 "$output_dir/%05d.png"
}

# Function to get the last created directory in a given parent directory
get_last_created_directory() {
    # Define the parent directory as the first argument
    local parent_dir="$1"

    # Navigate to the parent directory
    cd "$parent_dir" || { echo "Failed to navigate to $parent_dir"; exit 1; }

    # Get the name of the last created directory
    local last_created_dir
    last_created_dir=$(ls -dt */ | head -n 1)

    # Remove the trailing slash from the directory name
    last_created_dir=${last_created_dir%/}

    # Return the full path to the last created directory
    echo "$parent_dir/$last_created_dir"
}

# Function to cut the right half of video frames and rename them
rename_and_cut_frames() {
  # Define the target folder (pass this as an argument to the function)
  target_folder="$1"

  # Initialize a counter for the new file names
  counter=0

  # Loop through each file in the target folder that matches the pattern "pred_*.png"
  for file in "$target_folder"/pred_*.png; do
    # Format the new filename with leading zeros (e.g., 00000.png)
    new_name=$(printf "%05d.png" "$counter")

    # Get the width of the image
    double_width=$(identify -format "%w" "$file")

    # Calculate the width of half the image
    single_width=$((double_width / 2))

    # Cut the right half of the image and save it with the new name
    convert "$file" -crop "${single_width}x+${single_width}+0" "$target_folder/$new_name"

    # Increment the counter
    counter=$((counter + 1))

    # Remove the original file after processing
    rm "$file"
  done
}

# Function to perform encoding with HNeRV
function encode_with_hnerv {
    local video_name=$1
    local scene_number=$2
    local width=$3
    local height=$4
    local experiment_name=$5
    local original_file=$6
    local shrunk_width=$7
    local shrunk_height=$8
    local square_size=$9
    local hnerv_input_folder=${10}
    local encoding_size=${11}

    cd
    if [ ! -d "HNeRV/$hnerv_input_folder" ]; then
        mkdir -p "HNeRV/$hnerv_input_folder"
        cp "embrace/videos/${video_name}/scene_${scene_number}/${width}x${height}/$experiment_name/shrunk"/* "HNeRV/$hnerv_input_folder"/
    fi

    cd HNeRV
    export CUDA_VISIBLE_DEVICES=0
    export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

    # Use HNeRV to encode the video
    python train_nerv_all.py \
        --outf "embrace/${video_name}/scene_${scene_number}/${width}x${height}" \
        --data_path "$hnerv_input_folder" \
        --vid "$experiment_name" \
        --conv_type convnext pshuffel \
        --act gelu \
        --norm none \
        --crop_list "${shrunk_height}_${shrunk_width}" \
        --resize_list -1 \
        --loss L2 \
        --enc_strds 5 4 4 2 2 \
        --enc_dim "${square_size}_${square_size}" \
        --dec_strds 5 4 4 2 2 \
        --ks 0_3_3 \
        --reduce 1.2 \
        --modelsize "$encoding_size" \
        -e 300 \
        --eval_freq 30 \
        --lower_width 12 \
        -b 2 \
        --lr 0.001
}

# Function to perform decoding with HNeRV
function decode_with_hnerv {
    local video_name=$1
    local scene_number=$2
    local width=$3
    local height=$4
    local experiment_name=$5
    local shrunk_width=$6
    local shrunk_height=$7
    local square_size=$8
    local encoding_size=$9
    local hnerv_input_folder=${10}

    local encoded_shrunk_folder="videos/${video_name}/scene_${scene_number}/${width}x${height}/$experiment_name/encoded_shrunk"
    local weight_folder=$(get_last_created_directory "output/embrace/${video_name}/scene_${scene_number}/${width}x${height}/$experiment_name")

    cd HNeRV
    python train_nerv_all.py \
        --outf "embrace/${video_name}/scene_${scene_number}/${width}x${height}" \
        --data_path "$hnerv_input_folder" \
        --vid "$experiment_name" \
        --conv_type convnext pshuffel \
        --act gelu \
        --norm none \
        --crop_list "${shrunk_height}_${shrunk_width}" \
        --resize_list -1 \
        --loss L2 \
        --enc_strds 5 4 4 2 2 \
        --enc_dim "${square_size}_${square_size}" \
        --dec_strds 5 4 4 2 2 \
        --ks 0_3_3 \
        --reduce 1.2 \
        --modelsize "$encoding_size" \
        -e 300 \
        --eval_freq 30 \
        --lower_width 12 \
        -b 2 \
        --lr 0.001 \
        --eval_only \
        --weight "${weight_folder}/model_latest.pth" \
        --quant_model_bit 8 \
        --quant_embed_bit 6 \
        --dump_images

    # Move frames back into the embrace folder and rename them
    cd ..
    mkdir -p "embrace/${encoded_shrunk_folder}"
    cp "HNeRV/${weight_folder}/visualize_model_quant"/* "embrace/${encoded_shrunk_folder}/"
    cd embrace
    rename_and_cut_frames "$encoded_shrunk_folder"
}

# Function to perform inpainting with ProPainter
function inpaint_with_propainter {
    local video_name=$1
    local scene_number=$2
    local width=$3
    local height=$4
    local experiment_name=$5
    local neighbor_length=$6
    local ref_stride=$7
    local subvideo_length=$8
    
    local stretched_video_path="embrace/videos/${video_name}/scene_${scene_number}/${width}x${height}/$experiment_name/stretched.mp4"
    local mask_path="embrace/videos/${video_name}/scene_${scene_number}/${width}x${height}/$experiment_name/reconstructed_masks/"
    local inpainted_input_path="videos/${video_name}/scene_${scene_number}/${width}x${height}/$experiment_name/nei_${neighbor_length}_ref_${ref_stride}_sub_${subvideo_length}.mp4"

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
        --raft_iter 50 \
        --save_frames \
        --fp16

    # Move inpainted frames to experiment folder
    cd ..
    mv -f "ProPainter/results/stretched/frames" "embrace/videos/${video_name}/scene_${scene_number}/${width}x${height}/$experiment_name/nei_${neighbor_length}_ref_${ref_stride}_sub_${subvideo_length}"
    rm -r ProPainter/results/stretched/
    rm -r ProPainter/inputs/video_completion/
    
    cd embrace

    # Get inpainted video from frames
    frames_into_video "videos/${video_name}/scene_${scene_number}/${width}x${height}/$experiment_name/nei_${neighbor_length}_ref_${ref_stride}_sub_${subvideo_length}" "$inpainted_input_path" "lossless"
}

# Function to perform inpainting with E2FGVI
function inpaint_with_e2fgvi {
    local video_name=$1
    local scene_number=$2
    local width=$3
    local height=$4
    local experiment_name=$5
    local neighbor_length=$6
    local ref_stride=$7
    local subvideo_length=$8
    
    local stretched_video_path="embrace/videos/${video_name}/scene_${scene_number}/${width}x${height}/$experiment_name/stretched.mp4"
    local mask_path="embrace/videos/${video_name}/scene_${scene_number}/${width}x${height}/$experiment_name/reconstructed_masks/"
    local inpainted_input_path="videos/${video_name}/scene_${scene_number}/${width}x${height}/$experiment_name/nei_${neighbor_length}_ref_${ref_stride}_sub_${subvideo_length}.mp4"

    cp $stretched_video_path "E2FGVI/examples/stretched.mp4"
    cp -r $mask_path "E2FGVI/examples/stretched_mask/"
    cd E2FGVI
    
    # INPAINTING
    python test.py \
        --model e2fgvi_hq \
        --video examples/stretched.mp4 \
        --mask examples/stretched_mask \
        --ckpt release_model/E2FGVI-HQ-CVPR22.pth \
        --width ${width} \
        --height ${height}
    
    # Move inpainted video to experiment folder
    cd ..
    mv -f "E2FGVI/results/stretched_results.mp4" "embrace/videos/${video_name}/scene_${scene_number}/${width}x${height}/$experiment_name/nei_${neighbor_length}_ref_${ref_stride}_sub_${subvideo_length}.mp4"
    rm "E2FGVI/examples/stretched.mp4"
    rm -r "E2FGVI/examples/stretched_mask/"
    cd embrace
}

# TODO: add more parameters from ProPainter?
# TODO: add more parameters from HNeRV?
# TODO: add more parameters from E2FGVI?
# TODO: drop simple frames by passing full masks, so there is also some frame interpolation
# TODO: those different ways of calculating the mask, could be good to make into alternatives and compare them
# TODO: add LPIPS from tests into the pipeline
# TODO: implement row and column block selection
# TODO: implement *NeRV as codec
# TODO: implement HNeRV as codec + inpainter
# TODO: instead of server and client time, collect inpainting time
# TODO: instead of checking the videos folder for any unprocessed videos, take as input a video path and check whether you already have its scene folder
# TODO: we can also use image inpainting, since we have masked frames
# TODO: to improve scene change detection, don't set a threshold but compute it by taking the rolling median of the differences in the first 2 seconds, then checking whether each new difference is much higher, if it is, that's a scene change, if not, add it to the rolling median and go on

video_name=$1
scene_number=$2
width=$3
height=$4
square_size=$5
to_remove=$6
alpha=$7
smoothing_factor=$8
neighbor_length=$9
ref_stride=${10}
subvideo_length=${11}

# check whether width and height are multiples of square_size, otherwise make them
width=$(make_lower_multiple $width $square_size)
height=$(make_lower_multiple $height $square_size)

resolution="${width}x${height}"

# pass parameters to python scripts
export video_name=$video_name
export scene_number=$scene_number
export resolution=$resolution
export square_size=$square_size
export to_remove=$to_remove
export alpha=$alpha
export smoothing_factor=$smoothing_factor
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
experiment_name="squ_${square_size}_rem_${to_remove}_alp_${alpha}"
mkdir -p "videos/${video_name}/scene_${scene_number}/"${width}x${height}"/$experiment_name"

# resize scene based on experiment resolution, save into 
video_into_resized_frames "videos/${video_name}/scene_${scene_number}.mp4" "videos/${video_name}/scene_${scene_number}/${width}x${height}/original" $width $height

# get reference video from frames
frames_into_video "videos/${video_name}/scene_${scene_number}/"${width}x${height}"/original" "videos/${video_name}/scene_${scene_number}/"${width}x${height}"/reference_${square_size}.mp4" "lossless"

# calculate scene complexities
run_evca "videos/${video_name}/scene_${scene_number}/"${width}x${height}"/reference_${square_size}.mp4" "videos/${video_name}/scene_${scene_number}/"${width}x${height}"/reference_${square_size}.yuv" "${resolution}" "${square_size}" "embrace/videos/${video_name}/scene_${scene_number}/"${width}x${height}"/complexity_${square_size}/reference.csv"

# run script to get smart masks and shrunk frames
python shrink_frames.py

# get bitrate from shrunk size
shrunk_frame="videos/${video_name}/scene_${scene_number}/"${width}x${height}"/$experiment_name/shrunk/00000.png"
shrunk_width=$(ffprobe -v error -select_streams v:0 -show_entries stream=width -of csv=p=0 "$shrunk_frame")
shrunk_height=$(ffprobe -v error -select_streams v:0 -show_entries stream=height -of csv=p=0 "$shrunk_frame")
bitrate=$(python3 calculate_bitrate.py "$shrunk_width" "$shrunk_height")
export bitrate=$bitrate

# get original video from frames TODO: do this for every long path, also maybe the original folder should be reference?
original_folder="videos/${video_name}/scene_${scene_number}/"${width}x${height}"/original"
original_file="videos/${video_name}/scene_${scene_number}/"${width}x${height}"/$experiment_name/original.mp4"
frames_into_video $original_folder $original_file $bitrate

# ENCODING

shrunk_file="videos/${video_name}/scene_${scene_number}/"${width}x${height}"/$experiment_name/shrunk.mp4"
shrunk_folder="videos/${video_name}/scene_${scene_number}/"${width}x${height}"/$experiment_name/shrunk"

hnerv_input_folder="data/${video_name}/scene_${scene_number}/${width}x${height}/$experiment_name"
encoding_size=$(echo "scale=2; $(stat -c%s $original_file)/1024/1024/8" | bc)

# Uncomment one of the following lines based on the method you want to use:
# frames_into_video $shrunk_folder $shrunk_file $bitrate # encode with ffmpeg
encode_with_hnerv "$video_name" "$scene_number" "$width" "$height" "$experiment_name" "$original_file" "$shrunk_width" "$shrunk_height" "$square_size" "$hnerv_input_folder" "$encoding_size"

# CLIENT SIDE

client_start_time=$(date +%s)

# DECODING

encoded_shrunk_folder="videos/${video_name}/scene_${scene_number}/"${width}x${height}"/$experiment_name/encoded_shrunk"

# Uncomment one of the following lines based on the method you want to use:
decode_with_hnerv "$video_name" "$scene_number" "$width" "$height" "$experiment_name" "$shrunk_width" "$shrunk_height" "$square_size" "$encoding_size" "$hnerv_input_folder"
# video_into_frames $shrunk_file $encoded_shrunk_folder # decode with ffmpeg

# STRETCHING

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

reference_output_path="videos/${video_name}/scene_${scene_number}/${width}x${height}/reference_${square_size}.mp4"

# compare reference with original video
original_input_path="videos/${video_name}/scene_${scene_number}/"${width}x${height}"/$experiment_name/original.mp4"
original_csv_path="videos/${video_name}/scene_${scene_number}/"${width}x${height}"/$experiment_name/original.csv"
# Check if the output already exists, if not create it
if [[ -f "$original_csv_path" ]]; then
    echo "$original_csv_path already exists"   
else
    ffmpeg-quality-metrics $original_input_path $reference_output_path -m psnr ssim vmaf -of csv > "$original_csv_path"
fi

# INPAINTING

inpainted_input_path="videos/${video_name}/scene_${scene_number}/${width}x${height}/$experiment_name/nei_${neighbor_length}_ref_${ref_stride}_sub_${subvideo_length}.mp4"

if [[ -f "$inpainted_input_path" ]]; then
    echo "$inpainted_input_path already exists"
else
    cd ..

    # Uncomment one of the following lines based on the method you want to use:
    # inpaint_with_propainter "$video_name" "$scene_number" "$width" "$height" "$experiment_name" "$neighbor_length" "$ref_stride" "$subvideo_length"
    inpaint_with_e2fgvi "$video_name" "$scene_number" "$width" "$height" "$experiment_name" "$neighbor_length" "$ref_stride" "$subvideo_length"
fi

# QUALITY MEASUREMENT INPAINTED

# compare reference with inpainted video
inpainted_csv_path="videos/${video_name}/scene_${scene_number}/"${width}x${height}"/$experiment_name/nei_${neighbor_length}_ref_${ref_stride}_sub_${subvideo_length}.csv"
# Check if the output already exists, if not create it
if [[ -f "$inpainted_csv_path" ]]; then
    echo "$inpainted_csv_path already exists"   
else
    ffmpeg-quality-metrics $inpainted_input_path $reference_output_path -m psnr ssim vmaf -of csv > "$inpainted_csv_path"
    # calculate time elapsed
    end_time=$(date +%s)
    export server_start_time=$server_start_time
    export client_start_time=$client_start_time
    export end_time=$end_time
    # run metrics script
    python collect_metrics.py
fi

# # CLEANING UP TODO: move to run.sh, we launch orchestrator directly only in development, so keeping this stuff is good, while in run.sh we run into storage limitation, and there we need this

# # delete shrunk folder
# rm -r "videos/${video_name}/scene_${scene_number}/"${width}x${height}"/$experiment_name/shrunk"
# # delete stretched folder
# rm -r "videos/${video_name}/scene_${scene_number}/"${width}x${height}"/$experiment_name/stretched"
# # delete inpainted folder
# rm -r "videos/${video_name}/scene_${scene_number}/"${width}x${height}"/$experiment_name/nei_${neighbor_length}_ref_${ref_stride}_sub_${subvideo_length}"
