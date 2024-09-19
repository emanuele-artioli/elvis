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
    local frame_count=$(ffprobe -v error -count_frames -select_streams v:0 -show_entries stream=nb_read_frames -of csv=p=0 "$input_video_path")

    if [[ -f $output_video_path ]]; then
        echo "${output_video_path} already exists"
    else
        ffmpeg -loglevel warning -i $input_video_path -c:v rawvideo -pix_fmt yuv420p $output_video_path
        cd ..
        python3 EVCA/main.py -i "embrace/${output_video_path}" -r $resolution -b $square_size -f $frame_count -c "embrace/${csv_path}" -bi 1
        cd embrace
    fi
}

function video_into_frames {
    local input_file=$1
    local output_dir=$2
    local width=$3
    local height=$4

    # Check if the output directory already exists; if not, create it
    if [[ -d "$output_dir" ]]; then
        echo "$output_dir already exists"
        return 1        
    fi

    mkdir -p $output_dir

    # Construct the ffmpeg command
    if [[ -n $width && -n $height ]]; then
        # Apply scale and crop filters if width and height are provided
        ffmpeg -loglevel warning -i $input_file \
            -vf "scale=${width}:${height}:force_original_aspect_ratio=increase,crop=${width}:${height}" \
            -start_number 0 "$output_dir/%05d.png"
    else
        # No resizing, just extract frames
        ffmpeg -loglevel warning -i "$input_file" \
            -start_number 0 "$output_dir/%05d.png"
    fi
}

function frames_into_video {
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
        # Use HEVC (libx265) for lossless encoding
        ffmpeg -loglevel warning -framerate 24 -i "${input_dir}/%05d.png" -q:v 1 -pix_fmt yuv420p -c:v libx265 -x265-params lossless=1 $output_file
    else
        # Use AVC (libx264) for lossy encoding
        ffmpeg -loglevel warning -framerate 24 -i "${input_dir}/%05d.png" -pix_fmt yuv420p -c:v libx264 -b:v $bitrate -bufsize $bitrate $output_file
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
    local shrunk_width=$2
    local shrunk_height=$3
    local encoding_size=$4
    local ks=$5
    local enc_strds=(${6// / })
    local enc_dim=$7
    local fc_hw=$8
    local reduce=$9
    local lower_width=${10}
    local dec_strds=(${11// / })
    local conv_type=(${12// / })
    local norm=${13}
    local act=${14}
    local workers=${15}
    local batchSize=${16}
    local epochs=${17}
    local lr=${18}
    local loss=${19}
    local out_bias=${20}
    local eval_freq=${21}
    local quant_model_bit=${22}
    local quant_embed_bit=${23}

    local input_frames="data/${experiment_name}"

    cd ..
    mkdir -p "HNeRV/${input_frames}"
    cp "embrace/experiments/${experiment_name}/shrunk"/* "HNeRV/${input_frames}"/
    cd HNeRV

    # Use HNeRV to encode the video
    python train_nerv_all.py \
        --outf embrace \
        --data_path $input_frames \
        --vid $experiment_name \
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
}

function decode_with_hnerv {
    local experiment_name=$1
    local shrunk_width=$2
    local shrunk_height=$3
    local encoding_size=$4
    local ks=$5
    local enc_strds=(${6// / })
    local enc_dim=$7
    local fc_hw=$8
    local reduce=$9
    local lower_width=${10}
    local dec_strds=(${11// / })
    local conv_type=(${12// / })
    local norm=${13}
    local act=${14}
    local workers=${15}
    local batchSize=${16}
    local epochs=${17}
    local lr=${18}
    local loss=${19}
    local out_bias=${20}
    local eval_freq=${21}
    local quant_model_bit=${22}
    local quant_embed_bit=${23}

    cd ../HNeRV
    local input_frames="data/${experiment_name}"
    local weight_folder=$(get_last_created_directory "output/embrace/${experiment_name}")
    local output_frames="embrace/experiments/${experiment_name}/decoded"

    # Use HNeRV to decode the video
    python train_nerv_all.py \
        --outf embrace \
        --data_path $input_frames \
        --vid "$experiment_name" \
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
        --weight "${weight_folder}/model_latest.pth" \
        --dump_images

    # Move frames back into the embrace folder and rename them
    cd ..
    mkdir -p $output_frames
    cp "HNeRV/${weight_folder}/visualize_model_quant"/* $output_frames/
    rename_and_cut_frames $output_frames
    # clean HNeRV folders
    rm -r "HNeRV/${input_frames}"
    rm -r "HNeRV/output/embrace/${experiment_name}"
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
# TODO: add LPIPS from tests into the pipeline
# TODO: implement row and column block selection
# TODO: implement HNeRV as inpainter
# TODO: instead of server and client time, collect inpainting time
# TODO: instead of checking the videos folder for any unprocessed videos, take as input a video path and check whether you already have its scene folder
# TODO: we can also use image inpainting, since we have masked frames
# TODO: to improve scene change detection, don't set a threshold but compute it by taking the rolling median of the differences in the first 2 seconds, then checking whether each new difference is much higher, if it is, that's a scene change, if not, add it to the rolling median and go on

# PARAMETERS
codec_params=""
inpainter_params=""

# Embrace
video_name=$1
video_path=$2
width=$3
height=$4
square_size=$5
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
if [[ "$codec" == "avc" ]]; then
    experiment_name+="_avc"
fi

# hnerv
if [[ "$codec" == "hnerv" ]]; then
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
if [[ "$inpainter" == "propainter" ]]; then
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
if [[ "$inpainter" == "e2fgvi" ]]; then
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

# Check whether width and height are multiples of square_size, otherwise make them
width=$(make_lower_multiple $width $square_size)
height=$(make_lower_multiple $height $square_size)

resolution="${width}x${height}"

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
experiment_folder="experiments/${experiment_name}"
reference_frames="${experiment_folder}/reference"
video_into_frames $video_path $reference_frames $width $height

# get reference video from frames
reference_video="${experiment_folder}/reference.mp4"
frames_into_video $reference_frames $reference_video "lossless"

# calculate scene complexities
reference_raw="${experiment_folder}/reference.yuv"
reference_complexity="${experiment_folder}/complexity/reference.csv"
run_evca $reference_video $reference_raw $resolution $square_size $reference_complexity

# run script to get smart masks and shrunk frames
python scripts/shrink_frames.py

# get bitrate from shrunk size
shrunk_frame="${experiment_folder}/shrunk/00000.png"
shrunk_width=$(ffprobe -v error -select_streams v:0 -show_entries stream=width -of csv=p=0 $shrunk_frame)
shrunk_height=$(ffprobe -v error -select_streams v:0 -show_entries stream=height -of csv=p=0 $shrunk_frame)
bitrate=$(python scripts/calculate_bitrate.py $shrunk_width $shrunk_height)
export bitrate=$bitrate

# get original video from frames TODO: do this for every long path, also maybe the original folder should be reference?
original_video="${experiment_folder}/original.mp4"
frames_into_video $reference_frames $original_video $bitrate

# ENCODING

shrunk_video="${experiment_folder}/shrunk.mp4"
shrunk_frames="${experiment_folder}/shrunk"
encoding_size=$(echo "scale=2; $(stat -c%s $original_video)/1024/1024" | bc)

# Call the encoding function with HNeRV parameters if necessary
if [[ $codec == "hnerv" ]]; then
    encode_with_hnerv $experiment_name $shrunk_width $shrunk_height $encoding_size $ks "$enc_strds" $enc_dim $fc_hw $reduce $lower_width "$dec_strds" "$conv_type" $norm $act $workers $batchSize $epochs $lr $loss $out_bias $eval_freq $quant_model_bit $quant_embed_bit
elif [[ $codec == "avc" ]]; then
    frames_into_video $shrunk_frames $shrunk_video $bitrate
fi

# CLIENT SIDE

client_start_time=$(date +%s)

# DECODING

decoded_frames="${experiment_folder}/decoded"
# Call the decoding function with HNeRV parameters if necessary
if [[ $codec == "hnerv" ]]; then
    decode_with_hnerv $experiment_name $shrunk_width $shrunk_height $encoding_size $ks "$enc_strds" $enc_dim $fc_hw $reduce $lower_width "$dec_strds" "$conv_type" $norm $act $workers $batchSize $epochs $lr $loss $out_bias $eval_freq $quant_model_bit $quant_embed_bit
elif [[ $codec == "avc" ]]; then
    video_into_frames $shrunk_video $decoded_frames
fi

# STRETCHING

stretched_video="${experiment_folder}/stretched.mp4"
# Check if the stretched file already exists, if not create it
    if [[ -f "${stretched_video}" ]]; then
        echo "${stretched_video} already exists"
    else
        # run client script to get stretched frames
        python scripts/stretch_frames.py
        # get stretched video from frames
        frames_into_video "${experiment_folder}/stretched" $stretched_video "lossless"
    fi

# QUALITY MEASUREMENT ORIGINAL

# compare reference with original video
original_metrics="${experiment_folder}/original_metrics.csv"
# Check if the output already exists, if not create it
if [[ -f $original_metrics ]]; then
    echo "${original_metrics} already exists"
else
    ffmpeg-quality-metrics $original_video $reference_video -m psnr ssim vmaf -of csv > "$original_metrics"
fi

# INPAINTING

mask_frames="${experiment_folder}/decoded_masks"
inpainted_frames="${experiment_folder}/inpainted"
inpainted_video="${experiment_folder}/inpainted.mp4"

if [[ -f "$inpainted_video" ]]; then
    echo "$inpainted_video already exists"
else
    # Call the appropriate inpainting function
    if [[ "$inpainter" == "propainter" ]]; then
        # Call the propainter function
        inpaint_with_propainter $stretched_video $mask_frames $inpainted_frames $inpainter_params
    elif [[ "$inpainter" == "e2fgvi" ]]; then
        # Call the e2fgvi function
        inpaint_with_e2fgvi $stretched_video $mask_frames $inpainted_video $width $height $inpainter_params
    fi
fi

# QUALITY MEASUREMENT INPAINTED

# compare reference with inpainted video
inpainted_metrics="${experiment_folder}/inpainted_metrics.csv"
# Check if the output already exists, if not create it
if [[ -f $inpainted_metrics ]]; then
    echo "${inpainted_metrics} already exists"   
else
    ffmpeg-quality-metrics $inpainted_video $reference_video -m psnr ssim vmaf -of csv > "$inpainted_metrics"
fi

# calculate time elapsed
end_time=$(date +%s)
export server_start_time=$server_start_time
export client_start_time=$client_start_time
export end_time=$end_time
# run metrics script
python scripts/collect_metrics.py

# # CLEANING UP TODO: move to run.sh, we launch orchestrator directly only in development, so keeping this stuff is good, while in run.sh we run into storage limitation, and there we need this

# # delete shrunk folder
# rm -r "videos/${video_name}/scene_${scene_number}/"${width}x${height}"/$experiment_name/shrunk"
# # delete stretched folder
# rm -r "videos/${video_name}/scene_${scene_number}/"${width}x${height}"/$experiment_name/stretched"
# # delete inpainted folder
# rm -r "videos/${video_name}/scene_${scene_number}/"${width}x${height}"/$experiment_name/nei_${neighbor_length}_ref_${ref_stride}_sub_${subvideo_length}"
