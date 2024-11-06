#!/bin/bash
# source /etc/profile.d/opt-local.sh
export CUDA_VISIBLE_DEVICES=0
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# SETUP

function make_lower_multiple {
  num=$1
  multiple=$2
  remainder=$((num % multiple))
  if [ $remainder -ne 0 ]; then
    num=$((num - remainder))
  fi
  echo $num
}

function run_evca {
    local input_video_path=$1
    local output_video_path=$2
    local resolution=$3
    local square_size=$4
    local csv_path=$5

    # Get the number of frames in the input video using ffprobe
    local frame_count=$(ffprobe -hide_banner -loglevel error -count_frames -select_streams v:0 -show_entries stream=nb_read_frames -of csv=p=0 "$input_video_path")

    ffmpeg -hide_banner -loglevel error -i $input_video_path -c:v rawvideo -pix_fmt yuv420p $output_video_path
    cd ..
    python3 EVCA/main.py -i "embrace/${output_video_path}" -r $resolution -b $square_size -f $frame_count -c "embrace/${csv_path}" -bi 1
    cd embrace
}

scale_frames() {
    input_folder=$1
    width=$2
    height=$3
    output_folder=$4
    
    # Create the output folder if it doesn't exist
    mkdir -p "$output_folder"

    # Initialize a counter for renaming output frames
    counter=0

    # Loop over all image files in the input folder
    for frame in "$input_folder"/*; do
        formatted_counter=$(printf "%05d" "$counter") # Format the counter as 5-digit number (e.g., 00000, 00001, ...)
        output_frame="${output_folder}/${formatted_counter}.png"  # Set output frame path with counter
        
        # Run ffmpeg to scale the frame and suppress console output
        ffmpeg -hide_banner -loglevel error -i "$frame" -vf "scale=${width}:${height}:force_original_aspect_ratio=increase,crop=${width}:${height}" "$output_frame"
        
        # Increment the counter
        counter=$((counter + 1))
    done
}

function video_into_frames {
    local input_file=$1
    local output_dir=$2
    local width=$3
    local height=$4

    mkdir -p $output_dir

    # Construct the ffmpeg command
    if [[ -n $width && -n $height ]]; then
        # Apply scale and crop filters if width and height are provided
        ffmpeg -hide_banner -loglevel error -i $input_file \
            -vf "scale=${width}:${height}:force_original_aspect_ratio=increase,crop=${width}:${height}" \
            -start_number 0 "$output_dir/%05d.png"
    else
        # No resizing, just extract frames
        ffmpeg -hide_banner -loglevel error -i "$input_file" \
            -start_number 0 "$output_dir/%05d.png"
    fi
}

function frames_into_video {
    local input_dir=$1
    local output_file=$2
    local bitrate=$3

    # Determine if encoding should be lossless or not
    if [[ "$bitrate" == "lossless" ]]; then
        # Use HEVC (libx265) for lossless encoding
        ffmpeg -hide_banner -loglevel error -framerate 24 -i "${input_dir}/%05d.png" -q:v 1 -pix_fmt yuv420p -c:v libx265 -x265-params lossless=1 $output_file
    else
        # Use AVC (libx264) for lossy encoding
        ffmpeg -hide_banner -loglevel error -framerate 24 -i "${input_dir}/%05d.png" -pix_fmt yuv420p -c:v libx264 -b:v $bitrate -bufsize $bitrate $output_file
    fi
}

function get_last_created_directory {
    # Function to get the last created directory in a given parent directory
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

function rename_and_cut_frames {
    # Function to cut the right half of video frames and rename them
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

function encode_with_hnerv {
    local experiment_name=$1
    local frame_type=$2
    local shrunk_width=$3
    local shrunk_height=$4
    local encoding_size=$5
    local ks=$6
    local enc_strds=(${7// / })
    local enc_dim=$8
    local fc_hw=$9
    local reduce=${10}
    local lower_width=${11}
    local dec_strds=(${12// / })
    local conv_type=(${13// / })
    local norm=${14}
    local act=${15}
    local workers=${16}
    local batchSize=${17}
    local epochs=${18}
    local lr=${19}
    local loss=${20}
    local out_bias=${21}
    local eval_freq=${22}
    local quant_model_bit=${23}
    local quant_embed_bit=${24}

    cd ..
    # Determine the input frames folder based on frame type
    local input_frames="data/${experiment_name}_${frame_type}"
    local output_frames="embrace/experiments/${experiment_name}/${frame_type}_decoded"
    mkdir -p "HNeRV/${input_frames}"
    cp "embrace/experiments/${experiment_name}/${frame_type}"/* "HNeRV/${input_frames}"/
    cd HNeRV

    # Use HNeRV to encode the video
    python train_nerv_all.py \
        --outf embrace \
        --data_path $input_frames \
        --vid "${experiment_name}_${frame_type}" \
        --crop_list "${shrunk_height}_${shrunk_width}" \
        --ks $ks \
        --enc_strds "${enc_strds[@]}" \
        --enc_dim $enc_dim \
        --modelsize $encoding_size \
        --fc_hw $fc_hw \
        --reduce $reduce \
        --lower_width $lower_width \
        --dec_strds "${dec_strds[@]}" \
        --conv_type "${conv_type[@]}" \
        --norm $norm \
        --act $act \
        --workers $workers \
        --batchSize $batchSize \
        --epochs $epochs \
        --lr $lr \
        --loss $loss \
        --out_bias $out_bias \
        --eval_freq $eval_freq \
        --overwrite

    cd ../embrace
}

function decode_with_hnerv {
    local experiment_name=$1
    local frame_type=$2  # New parameter: 'benchmark' or 'shrunk'
    local shrunk_width=$3
    local shrunk_height=$4
    local encoding_size=$5
    local ks=$6
    local enc_strds=(${7// / })
    local enc_dim=$8
    local fc_hw=$9
    local reduce=${10}
    local lower_width=${11}
    local dec_strds=(${12// / })
    local conv_type=(${13// / })
    local norm=${14}
    local act=${15}
    local workers=${16}
    local batchSize=${17}
    local epochs=${18}
    local lr=${19}
    local loss=${20}
    local out_bias=${21}
    local eval_freq=${22}
    local quant_model_bit=${23}
    local quant_embed_bit=${24}

    cd
    local output_frames="embrace/experiments/${experiment_name}/${frame_type}_decoded"
    cd HNeRV
    local input_frames="data/${experiment_name}_${frame_type}"
    local weight_folder=$(get_last_created_directory "output/embrace/${experiment_name}_${frame_type}")
    local model_weights="${weight_folder}/model_latest.pth"
    local quantized_weights="${weight_folder}/quant_vid.pth"
    # encoding sizes
    echo "Targeted encoding size: ${encoding_size}"
    file_size=$(stat -c%s "$model_weights")
    echo "Actual model size: ${file_size}"
    file_size=$(stat -c%s "$quantized_weights")
    echo "Actual quantized size: ${file_size}"

    # Use HNeRV to decode the video
    python train_nerv_all.py \
        --outf embrace \
        --data_path $input_frames \
        --vid "${experiment_name}_${frame_type}" \
        --crop_list "${shrunk_height}_${shrunk_width}" \
        --ks $ks \
        --enc_strds "${enc_strds[@]}" \
        --enc_dim $enc_dim \
        --modelsize $encoding_size \
        --fc_hw $fc_hw \
        --reduce $reduce \
        --lower_width $lower_width \
        --dec_strds "${dec_strds[@]}" \
        --conv_type "${conv_type[@]}" \
        --norm "$norm" \
        --act "$act" \
        --workers "$workers" \
        --batchSize "$batchSize" \
        --epochs "$epochs" \
        --lr "$lr" \
        --loss "$loss" \
        --out_bias "$out_bias" \
        --eval_freq "$eval_freq" \
        --quant_model_bit "$quant_model_bit" \
        --quant_embed_bit "$quant_embed_bit" \
        --eval_only \
        --weight $model_weights \
        --dump_images

    # Move frames back into the embrace folder and rename them
    cd ..
    mkdir -p $output_frames
    cp "HNeRV/${weight_folder}/visualize_model_quant"/* $output_frames/
    rename_and_cut_frames $output_frames
    cd embrace
}

function inpaint_with_propainter {
    local stretched_video=$1
    local mask_frames=$2
    local inpainted_frames=$3
    local neighbor_length=$4
    local ref_stride=$5
    local subvideo_length=$6
    local mask_dilation=$7
    local raft_iter=$8

    local stretched_video="embrace/${stretched_video}"
    local mask_frames="embrace/${mask_frames}/"
    local inpainted_frames="embrace/${inpainted_frames}"

    cd ..
    cp $stretched_video "ProPainter/inputs/video_completion/stretched.mp4"
    cp -r $mask_frames "ProPainter/inputs/video_completion/masks"
    cd ProPainter
    
    # INPAINTING
    python inference_propainter.py \
        --video inputs/video_completion/stretched.mp4 \
        --mask inputs/video_completion/masks \
        --mask_dilation "${mask_dilation}" \
        --neighbor_length "${neighbor_length}" \
        --ref_stride "${ref_stride}" \
        --subvideo_length "${subvideo_length}" \
        --raft_iter "${raft_iter}" \
        --save_frames \
        --fp16

    # Move inpainted frames to experiment folder
    cd ..
    mv -f "ProPainter/results/stretched/frames" $inpainted_frames
    rm -r ProPainter/results/stretched/
    rm -r ProPainter/inputs/video_completion/stretched.mp4
    rm -r ProPainter/inputs/video_completion/masks

    # Get inpainted video from frames
    frames_into_video $inpainted_frames "${inpainted_frames}.mp4" "lossless"
    cd embrace
}

function inpaint_with_e2fgvi {
    local stretched_video=$1
    local mask_frames=$2
    local inpainted_video=$3
    local width=$4
    local height=$5
    local step=$6
    local num_ref=$7
    local neighbor_stride=$8
    local savefps=$9

    local stretched_video="embrace/${stretched_video}"
    local mask_frames="embrace/${mask_frames}"
    local inpainted_video="embrace/${inpainted_video}"

    cd ..
    cp $stretched_video "E2FGVI/examples/stretched.mp4"
    cp -r $mask_frames "E2FGVI/examples/stretched_masks"
    cd E2FGVI
    
    # INPAINTING
    python test.py \
        --model e2fgvi_hq \
        --video examples/stretched.mp4 \
        --mask examples/stretched_masks \
        --ckpt release_model/E2FGVI-HQ-CVPR22.pth \
        --set_size \
        --width "${width}" \
        --height "${height}" \
        --step "${step}" \
        --num_ref "${num_ref}" \
        --neighbor_stride "${neighbor_stride}" \
        --savefps "${savefps}"
    
    # Move inpainted video to experiment folder
    cd ..
    mv -f "E2FGVI/results/stretched_results.mp4" $inpainted_video
    rm "E2FGVI/examples/stretched.mp4"
    rm -r "E2FGVI/examples/stretched_masks/"
    cd embrace
}

# TODO: drop simple frames by passing full masks, so there is also some frame interpolation
# TODO: those different ways of calculating the mask, could be good to make into alternatives and compare them
# TODO: implement column block selection
# TODO: instead of server and client time, collect inpainting time
# TODO: we can also use image inpainting, since we have masked frames
# TODO: silence ffmpeg

# PARAMETERS
codec_params=""
inpainter_params=""

# Embrace
video_name=$1
raw_frames=$2
width=$3
height=$4
square_size=$5
# Check whether width and height are multiples of square_size, otherwise make them
width=$(make_lower_multiple $width $square_size)
height=$(make_lower_multiple $height $square_size)
resolution="${width}x${height}"
to_remove=$6
alpha=$7
smoothing_factor=$8
codec=$9
inpainter=${10}
# Initialize the experiment_name with common parameters
experiment_name="${video_name}_${width}x${height}_ss_${square_size}_tr_${to_remove}_alp_${alpha}_sf_${smoothing_factor}"
# Shift off embrace's parameters
shift 10

# avc
if [[ "${codec}" == "avc" ]]; then
    experiment_name+="_avc"
fi

# hnerv
if [[ "${codec}" == "hnerv" ]]; then
    ks=$1
    enc_strds=$2
    enc_dim=$3
    fc_hw=$4
    reduce=$5
    lower_width=$6
    dec_strds=$7
    conv_type=$8
    norm=$9
    act=${10}
    workers=${11}
    batchSize=${12}
    epochs=${13}
    lr=${14}
    loss=${15}
    out_bias=${16}
    eval_freq=${17}
    quant_model_bit=${18}
    quant_embed_bit=${19}
    
    # Export HNeRV parameters with unique keywords
    export hnerv_ks=$ks
    export hnerv_enc_strds=$enc_strds
    export hnerv_enc_dim=$enc_dim
    export hnerv_fc_hw=$fc_hw
    export hnerv_reduce=$reduce
    export hnerv_lower_width=$lower_width
    export hnerv_dec_strds=$dec_strds
    export hnerv_conv_type=$conv_type
    export hnerv_norm=$norm
    export hnerv_act=$act
    export hnerv_workers=$workers
    export hnerv_batchSize=$batchSize
    export hnerv_epochs=$epochs
    export hnerv_lr=$lr
    export hnerv_loss=$loss
    export hnerv_out_bias=$out_bias
    export hnerv_eval_freq=$eval_freq
    export hnerv_quant_model_bit=$quant_model_bit
    export hnerv_quant_embed_bit=$quant_embed_bit

    # Prepare enc_strds and dec_strds for experiment_name by replacing spaces with underscores
    enc_strds_for_name="${enc_strds// /_}"
    dec_strds_for_name="${dec_strds// /_}"
    conv_type_for_name="${conv_type// /_}"

    # Add HNeRV parameters to the experiment name
    experiment_name+="_hnerv_${ks}_${enc_strds_for_name}_${enc_dim}_${fc_hw}_${reduce}_${lower_width}_${dec_strds_for_name}_${conv_type_for_name}_${norm}_${act}_${workers}_${batchSize}_${epochs}_${lr}_${loss}_${out_bias}_${eval_freq}_${quant_model_bit}_${quant_embed_bit}"

    # Shift off the HNeRV-specific parameters
    shift 19
fi

# propainter
if [[ "${inpainter}" == "propainter" ]]; then
    neighbor_length=$1
    ref_stride=$2
    subvideo_length=$3
    mask_dilation=$4
    raft_iter=$5

    # Export ProPainter parameters with unique keywords
    export propainter_neighbor_length=$neighbor_length
    export propainter_ref_stride=$ref_stride
    export propainter_subvideo_length=$subvideo_length
    export propainter_mask_dilation=$mask_dilation
    export propainter_raft_iter=$raft_iter

    # Construct the inpainter_params string for ProPainter
    inpainter_params="$neighbor_length $ref_stride $subvideo_length $mask_dilation $raft_iter"

    # Add ProPainter parameters to the experiment name
    experiment_name+="_propainter_${neighbor_length}_${ref_stride}_${subvideo_length}_${mask_dilation}_${raft_iter}"

    # Shift off the ProPainter-specific parameters
    shift 5
fi

# e2fgvi
if [[ "${inpainter}" == "e2fgvi" ]]; then
    step=$1
    num_ref=$2
    neighbor_stride=$3
    savefps=$4

    # Export E2FGVI parameters with unique keywords
    export e2fgvi_step=$step
    export e2fgvi_num_ref=$num_ref
    export e2fgvi_neighbor_stride=$neighbor_stride
    export e2fgvi_savefps=$savefps

    # Construct the inpainter_params string for E2FGVI
    inpainter_params="$step $num_ref $neighbor_stride $savefps"

    # Add E2FGVI parameters to the experiment name
    experiment_name+="_e2fgvi_${step}_${num_ref}_${neighbor_stride}_${savefps}"

    # Shift off the E2FGVI-specific parameters
    shift 4
fi

# Generate an MD5 hash and extract the first part as a shorter folder name
experiment_name=$(echo -n "$experiment_name" | md5sum | awk '{print $1}')

# check if experiment was already run TODO: set experiment name as an encoding to make it shorter, and add that encoding to the results.csv to be able to look it up
if [[ -d "experiments/${experiment_name}" ]]; then
    echo "experiment already run"
    exit 1
fi

# Print embrace-speecific parameters
echo "Running EMBRACE with parameters:"
echo "Video Name: ${video_name}"
echo "Raw Frames Path: ${raw_frame}"
echo "Resolution: ${resolution}"
echo "Square Size: ${square_size}"
echo "To Remove: ${to_remove}"
echo "Alpha: ${alpha}"
echo "Smoothing Factor: ${smoothing_factor}"
echo "Codec: ${codec}"
echo "Inpainter: ${inpainter}"
echo "Experiment name: ${experiment_name}"

# Print hnerv-specific parameters
if [[ "$codec" == "hnerv" ]]; then
    echo "ks: ${ks}"
    echo "enc_strds: ${enc_strds}"
    echo "enc_dim: ${enc_dim}"
    echo "fc_hw: ${fc_hw}"
    echo "reduce: ${reduce}"
    echo "lower_width: ${lower_width}"
    echo "dec_strds: ${dec_strds}"
    echo "conv_type: ${conv_type}"
    echo "norm: ${norm}"
    echo "act: ${act}"
    echo "workers: ${workers}"
    echo "batchSize: ${batchSize}"
    echo "epochs: ${epochs}"
    echo "lr: ${lr}"
    echo "loss: ${loss}"
    echo "out_bias: ${out_bias}"
    echo "eval_freq: ${eval_freq}"
    echo "quant_model_bit: ${quant_model_bit}"
    echo "quant_embed_bit: ${quant_embed_bit}"
fi

# Print propainter-specific parameters
if [[ "$inpainter" == "propainter" ]]; then
    echo "Neighbor Length: ${neighbor_length}"
    echo "Ref Stride: ${ref_stride}"
    echo "Subvideo Length: ${subvideo_length}"
    echo "Mask Dilation: ${mask_dilation}"
    echo "Raft Iter: ${raft_iter}"
fi

# Print e2fgvi-specific parameters
if [[ "$inpainter" == "e2fgvi" ]]; then
    echo "Step: ${step}"
    echo "Num Ref: ${num_ref}"
    echo "Neighbor Stride: ${neighbor_stride}"
    echo "Save FPS: ${savefps}"
fi

# Pass parameters to Python scripts
export video_name=$video_name
export resolution=$resolution
export square_size=$square_size
export to_remove=$to_remove
export alpha=$alpha
export smoothing_factor=$smoothing_factor
export experiment_name=$experiment_name
export codec=$codec
export inpainter=$inpainter

# SERVER SIDE

server_start_time=$(date +%s)

# resize video based on experiment resolution, save into experiment folder
echo "Scaling frames to required resolution..."
# task_start_time=$(date +%s)
experiment_folder="experiments/${experiment_name}"
benchmark_frames="${experiment_folder}/benchmark"
cd ..
scale_frames $raw_frames $width $height "embrace/${benchmark_frames}"
cd embrace
# task_end_time=$(date +%s)
# task_duration=$(( task_end_time - task_start_time ))
# echo "Task completed in ${task_duration} seconds."

# get reference video from frames
echo "Encoding reference video..."
# task_start_time=$(date +%s)
reference_video="${experiment_folder}/reference.mp4"
frames_into_video $benchmark_frames $reference_video "lossless"
# task_end_time=$(date +%s)
# # task_duration=$(( task_end_time - task_start_time ))
# echo "Task completed in ${task_duration} seconds."

# calculate scene complexities
echo "Calculating scene complexity with EVCA..."
# task_start_time=$(date +%s)
reference_raw="${experiment_folder}/reference.yuv"
reference_complexity="${experiment_folder}/complexity/reference.csv"
run_evca $reference_video $reference_raw $resolution $square_size $reference_complexity
# task_end_time=$(date +%s)
# task_duration=$(( task_end_time - task_start_time ))
# echo "Task completed in ${task_duration} seconds."

# generate focused masks to know what is the main object of a scene that needs to not be removed
echo "Generating focus masks with UFO..."
# task_start_time=$(date +%s)
cd
UFO_folder="datasets/embrace/image/${experiment_name}"
mkdir -p "UFO/${UFO_folder}"
cp -r "embrace/${experiment_folder}/benchmark"/* "UFO/${UFO_folder}"/
cd UFO 
python test.py --model="weights/video_best.pth" --data_path="datasets/embrace/" --output_dir="VSOD_results/wo_optical_flow/embrace" --task="VSOD"
# copy output folder back into embrace
cd
mv "UFO/VSOD_results/wo_optical_flow/embrace/${experiment_name}" "embrace/${experiment_folder}/focus_masks"
cd embrace
# task_end_time=$(date +%s)
# task_duration=$(( task_end_time - task_start_time ))
# echo "Task completed in ${task_duration} seconds."

# run script to get smart masks and shrunk frames
echo "Shrinking frames..."
# task_start_time=$(date +%s)
python scripts/shrink_frames.py
# task_end_time=$(date +%s)
# task_duration=$(( task_end_time - task_start_time ))
# echo "Task completed in ${task_duration} seconds."

# get bitrate from shrunk size
shrunk_frame="${experiment_folder}/shrunk/00000.png"
shrunk_width=$(ffprobe -hide_banner -loglevel error -select_streams v:0 -show_entries stream=width -of csv=p=0 $shrunk_frame)
shrunk_height=$(ffprobe -hide_banner -loglevel error -select_streams v:0 -show_entries stream=height -of csv=p=0 $shrunk_frame)
bitrate=$(python scripts/calculate_bitrate.py $shrunk_width $shrunk_height)
export bitrate=$bitrate

# ENCODING
echo "Encoding video with ${codec}..."
# task_start_time=$(date +%s)

shrunk_video="${experiment_folder}/shrunk.mp4"
shrunk_frames="${experiment_folder}/shrunk"
benchmark_video="${experiment_folder}/benchmark.mp4"

if [[ $codec == "hnerv" ]]; then
    # Get the duration of the reference video (in seconds)
    reference_video_duration=$(ffprobe -hide_banner -loglevel error -select_streams v:0 -show_entries stream=duration -of default=noprint_wrappers=1:nokey=1 $reference_video)
    # Calculate encoding size based on duration and bitrate
    encoding_size=$(echo "scale=2; $reference_video_duration * $bitrate / 1024 / 1024" | bc)
    # Calculate the padding required to make width and height multiples of 320 (HNeRV breaks for videos not multiple of 320x320)
    # Calculate padding required for the width and height
    pad_right=$(( ((width + 319) / 320) * 320 - width ))
    pad_bottom=$(( ((height + 319) / 320) * 320 - height ))
    shrunk_pad_right=$(( ((shrunk_width + 319) / 320) * 320 - shrunk_width ))
    shrunk_pad_bottom=$(( ((shrunk_height + 319) / 320) * 320 - shrunk_height ))
    # Ensure padded frames directories exist
    mkdir -p "${experiment_folder}/benchmark_padded"
    mkdir -p "${experiment_folder}/shrunk_padded"
    # Pad all benchmark frames
    ffmpeg -hide_banner -loglevel error -i "${benchmark_frames}/%05d.png" -vf "pad=iw+$pad_right:ih+$pad_bottom:0:0:black" -start_number 0 "${experiment_folder}/benchmark_padded/%05d.png"
    # Pad all shrunk frames
    ffmpeg -hide_banner -loglevel error -i "${shrunk_frames}/%05d.png" -vf "pad=iw+$shrunk_pad_right:ih+$shrunk_pad_bottom:0:0:black" -start_number 0 "${experiment_folder}/shrunk_padded/%05d.png"
    # Encode the padded benchmark frames with HNeRV
    encode_with_hnerv $experiment_name "benchmark_padded" $((width + pad_right)) $((height + pad_bottom)) $encoding_size $ks "$enc_strds" $enc_dim $fc_hw $reduce $lower_width "$dec_strds" "$conv_type" $norm $act $workers $batchSize $epochs $lr $loss $out_bias $eval_freq $quant_model_bit $quant_embed_bit
    # Encode the padded shrunk frames with HNeRV
    encode_with_hnerv $experiment_name "shrunk_padded" $((shrunk_width + shrunk_pad_right)) $((shrunk_height + shrunk_pad_bottom)) $encoding_size $ks "$enc_strds" $enc_dim $fc_hw $reduce $lower_width "$dec_strds" "$conv_type" $norm $act $workers $batchSize $epochs $lr $loss $out_bias $eval_freq $quant_model_bit $quant_embed_bit

elif [[ $codec == "avc" ]]; then
    # Get encoded video from frames
    frames_into_video $benchmark_frames $benchmark_video $bitrate
    # Encode the shrunk frames with AVC
    frames_into_video $shrunk_frames $shrunk_video $bitrate
fi

# task_end_time=$(date +%s)
# task_duration=$(( task_end_time - task_start_time ))
# echo "Task completed in ${task_duration} seconds."

# CLIENT SIDE

client_start_time=$(date +%s)

# DECODING

echo "Decoding video with ${codec}..."
# task_start_time=$(date +%s)

benchmark_decoded_frames="${experiment_folder}/benchmark_decoded"
shrunk_decoded_frames="${experiment_folder}/shrunk_decoded"

# Call the decoding function with HNeRV parameters if necessary
if [[ $codec == "hnerv" ]]; then
    # Decode the benchmark NN into padded frames with HNeRV
    decode_with_hnerv $experiment_name "benchmark_padded" $((width + pad_right)) $((height + pad_bottom)) $encoding_size $ks "$enc_strds" $enc_dim $fc_hw $reduce $lower_width "$dec_strds" "$conv_type" $norm $act $workers $batchSize $epochs $lr $loss $out_bias $eval_freq $quant_model_bit $quant_embed_bit
    # Decode the shrunk NN into padded frames with HNeRV
    decode_with_hnerv $experiment_name "shrunk_padded" $((shrunk_width + shrunk_pad_right)) $((shrunk_height + shrunk_pad_bottom)) $encoding_size $ks "$enc_strds" $enc_dim $fc_hw $reduce $lower_width "$dec_strds" "$conv_type" $norm $act $workers $batchSize $epochs $lr $loss $out_bias $eval_freq $quant_model_bit $quant_embed_bit

    # Ensure padded frames directories exist
    mkdir -p $benchmark_decoded_frames
    mkdir -p $shrunk_decoded_frames
    # Remove black bands (crop back to original size... it would be required to send these values to the client, but we assume a real implementation would fix HNeRV and not need to pad it at all)
    ffmpeg -hide_banner -loglevel error -i "${experiment_folder}/benchmark_padded_decoded/%05d.png" -vf "crop=${width}:${height}:0:0" -start_number 0 "${benchmark_decoded_frames}/%05d.png"
    ffmpeg -hide_banner -loglevel error -i "${experiment_folder}/shrunk_padded_decoded/%05d.png" -vf "crop=${shrunk_width}:${shrunk_height}:0:0" -start_number 0 "${shrunk_decoded_frames}/%05d.png"

    # Take frames and create benchmark video on the client side
    frames_into_video $benchmark_decoded_frames $benchmark_video "lossless"

elif [[ $codec == "avc" ]]; then
    # Decode the shrunk video using AVC
    video_into_frames $shrunk_video $shrunk_decoded_frames
fi

# task_end_time=$(date +%s)
# task_duration=$(( task_end_time - task_start_time ))
# echo "Task completed in ${task_duration} seconds."

# STRETCHING

echo "Stretching frames..."
# task_start_time=$(date +%s)

stretched_video="${experiment_folder}/stretched.mp4"
# run client script to get stretched frames
python scripts/stretch_frames.py
# get stretched video from frames
frames_into_video "${experiment_folder}/stretched" $stretched_video "lossless"

# task_end_time=$(date +%s)
# task_duration=$(( task_end_time - task_start_time ))
# echo "Task completed in ${task_duration} seconds."

# QUALITY MEASUREMENT benchmark

# compare reference with benchmark video
benchmark_metrics="${experiment_folder}/benchmark_metrics.csv"
ffmpeg-quality-metrics $benchmark_video $reference_video -m psnr ssim vmaf -of csv > "$benchmark_metrics"

# LPIPS
python scripts/calculate_lpips.py "$reference_video" "$benchmark_video" "$benchmark_metrics"

# INPAINTING

echo "Inpainting videos with ${inpainter}..."
# task_start_time=$(date +%s)

mask_frames="${experiment_folder}/decoded_masks"
inpainted_frames="${experiment_folder}/inpainted"
inpainted_video="${experiment_folder}/inpainted.mp4"

# Call the appropriate inpainting function
if [[ "$inpainter" == "propainter" ]]; then
    # Call the propainter function
    inpaint_with_propainter $stretched_video $mask_frames $inpainted_frames $inpainter_params
elif [[ "$inpainter" == "e2fgvi" ]]; then
    # Call the e2fgvi function
    inpaint_with_e2fgvi $stretched_video $mask_frames $inpainted_video $width $height $inpainter_params
fi

# task_end_time=$(date +%s)
# task_duration=$(( task_end_time - task_start_time ))
# echo "Task completed in ${task_duration} seconds."

# QUALITY MEASUREMENT INPAINTED

echo "Evaluating video quality..."
# task_start_time=$(date +%s)

# compare reference with inpainted video
inpainted_metrics="${experiment_folder}/inpainted_metrics.csv"
ffmpeg-quality-metrics $inpainted_video $reference_video -m psnr ssim vmaf -of csv > "$inpainted_metrics"

# LPIPS
python scripts/calculate_lpips.py "$reference_video" "$inpainted_video" "$inpainted_metrics"

# task_end_time=$(date +%s)
# task_duration=$(( task_end_time - task_start_time ))
# echo "Task completed in ${task_duration} seconds."

# calculate time elapsed
end_time=$(date +%s)
export server_start_time=$server_start_time
export client_start_time=$client_start_time
export end_time=$end_time
# run metrics script
python scripts/collect_metrics.py

# return experiment_name to run.sh for cleanup
echo "$experiment_name" > experiment_name.txt
echo "Experiment completed."