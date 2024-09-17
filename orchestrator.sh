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

    # Construct the filter options
    local filter_options=""
    if [[ -n "$width" && -n "$height" ]]; then
        filter_options="scale=${width}:${height}:force_original_aspect_ratio=increase,crop=${width}:${height}"
    fi

    # Apply filters if specified
    ffmpeg -loglevel warning -i "$input_file" \
        -vf "$filter_options" \
        -q:v 1 -start_number 0 "$output_dir/%05d.png"
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

function encode_with_hnerv_old {
    local experiment_name=$1
    local shrunk_width=$2
    local shrunk_height=$3
    local encoding_size=$4
    local input_frames="data/${experiment_name}"

    # # Check if the input directory already exists; if not, create it
    # if [[ -d "$input_frames" ]]; then
    #     echo "$input_frames already exists"
    #     return 1        
    # fi
    # cd ..
    # cp "embrace/${experiment_name}/shrunk"/* "HNeRV/${input_frames}"/

    # cd HNeRV
    # export CUDA_VISIBLE_DEVICES=0
    # export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

    # Use HNeRV to encode the video
    python train_nerv_all.py \
        --outf $experiment_name \
        --data_path $input_frames \
        --vid "$experiment_name" \
        --crop_list "${shrunk_height}_${shrunk_width}" \
        --ks "0_3_3" \
        --enc_strds 5 4 4 2 2 \
        --enc_dim "64_16" \
        --modelsize "$encoding_size" \
        --fc_hw "9_16" \
        --reduce 1.2 \
        --lower_width 32 \
        --dec_strds 5 4 4 2 2 \
        --conv_type "convnext" \
        --norm "none" \
        --act "gelu" \
        --workers -1 \
        --batchSize 1 \
        --epochs 300 \
        --lr 0.001 \
        --loss "L2" \
        --out_bias "tanh" \
        --eval_freq 30 \
        --overwrite

    # cd ../embrace
}

function encode_with_hnerv {
    local experiment_name=$1
    local shrunk_width=$2
    local shrunk_height=$3
    local encoding_size=$4
    local ks=$5
    local enc_strds=$6
    local enc_dim=$7
    local fc_hw=$8
    local reduce=$9
    local lower_width=${10}
    local dec_strds=${11}
    local conv_type=${12}
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
    cp "embrace/experiments/${experiment_name}/shrunk"/* "HNeRV/${input_frames}"/
    cd HNeRV

    # Use HNeRV to encode the video
    python train_nerv_all.py \
        --outf $experiment_name \
        --data_path $input_frames \
        --vid "$experiment_name" \
        --crop_list "${shrunk_height}_${shrunk_width}" \
        --ks $ks \
        --enc_strds $enc_strds \
        --enc_dim $enc_dim \
        --modelsize $encoding_size \
        --fc_hw $fc_hw \
        --reduce $reduce \
        --lower_width $lower_width \
        --dec_strds $dec_strds \
        --conv_type $conv_type \
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

function decode_with_hnerv_old {
    local experiment_name=$1
    local shrunk_width=$2
    local shrunk_height=$3
    local encoding_size=$4
    local input_frames="data/${experiment_name}"
    local weight_folder=$(get_last_created_directory "output/${experiment_name}")
    local output_frames="embrace/${experiment_name}/encoded_shrunk"

    # cd ../HNeRV
    python train_nerv_all.py \
        --outf $experiment_name \
        --data_path $input_frames \
        --vid "$experiment_name" \
        --crop_list "${shrunk_height}_${shrunk_width}" \
        --ks "0_3_3" \
        --enc_strds 5 4 4 2 2 \
        --enc_dim "64_16" \
        --modelsize "$encoding_size" \
        --fc_hw "9_16" \
        --reduce 1.2 \
        --lower_width 32 \
        --dec_strds 5 4 4 2 2 \
        --conv_type "convnext" \
        --norm "none" \
        --act "gelu" \
        --workers -1 \
        --batchSize 1 \
        --epochs 300 \
        --lr 0.001 \
        --loss "L2" \
        --out_bias "tanh" \
        --eval_freq 30 \
        --overwrite \
        --eval_only \
        --weight "${weight_folder}/model_latest.pth" \
        --quant_model_bit 8 \
        --quant_embed_bit 6 \
        --dump_images

    # # Move frames back into the embrace folder and rename them
    # cd ..
    # mkdir -p $output_frames
    # cp "HNeRV/${weight_folder}/visualize_model_quant"/* $output_frames/
    # rename_and_cut_frames $output_frames
    # cd embrace
}

function decode_with_hnerv {
    local experiment_name=$1
    local shrunk_width=$2
    local shrunk_height=$3
    local encoding_size=$4
    local ks=$5
    local enc_strds=$6
    local enc_dim=$7
    local fc_hw=$8
    local reduce=$9
    local lower_width=${10}
    local dec_strds=${11}
    local conv_type=${12}
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
    local weight_folder=$(get_last_created_directory "output/${experiment_name}")
    local output_frames="embrace/${experiment_name}/encoded_shrunk"

    # Use HNeRV to decode the video
    python train_nerv_all.py \
        --outf $experiment_name \
        --data_path $input_frames \
        --vid "$experiment_name" \
        --crop_list "${shrunk_height}_${shrunk_width}" \
        --ks "$ks" \
        --enc_strds $enc_strds \
        --enc_dim "$enc_dim" \
        --modelsize "$encoding_size" \
        --fc_hw "$fc_hw" \
        --reduce "$reduce" \
        --lower_width "$lower_width" \
        --dec_strds $dec_strds \
        --conv_type "$conv_type" \
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
        --overwrite \
        --eval_only \
        --weight "${weight_folder}/model_latest.pth" \
        --dump_images
}

function inpaint_with_propainter_old {
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

function inpaint_with_propainter {
    local video_name=$1
    local scene_number=$2
    local width=$3
    local height=$4
    local experiment_name=$5
    local neighbor_length=$6
    local ref_stride=$7
    local subvideo_length=$8
    local mask_dilation=$9
    local raft_iter=${10}

    local stretched_video_path="embrace/${experiment_name}/stretched.mp4"
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
        --mask_dilation "${mask_dilation}" \
        --neighbor_length "${neighbor_length}" \
        --ref_stride "${ref_stride}" \
        --subvideo_length "${subvideo_length}" \
        --raft_iter "${raft_iter}" \
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

function inpaint_with_e2fgvi_old {
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
        --set_size \
        --width ${width} \
        --height ${height} \
        --step 10 \
        --num_ref -1 \
        --neighbor_stride 5 \
        --savefps 24 \
    
    # Move inpainted video to experiment folder
    cd ..
    mv -f "E2FGVI/results/stretched_results.mp4" "embrace/videos/${video_name}/scene_${scene_number}/${width}x${height}/$experiment_name/nei_${neighbor_length}_ref_${ref_stride}_sub_${subvideo_length}.mp4"
    rm "E2FGVI/examples/stretched.mp4"
    rm -r "E2FGVI/examples/stretched_mask/"
    cd embrace
}

function inpaint_with_e2fgvi {
    local video_name=$1
    local scene_number=$2
    local width=$3
    local height=$4
    local experiment_name=$5
    local neighbor_length=$6
    local ref_stride=$7
    local subvideo_length=$8
    local step=$9
    local num_ref=${10}
    local neighbor_stride=${11}
    local savefps=${12}

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
        --set_size \
        --width "${width}" \
        --height "${height}" \
        --step "${step}" \
        --num_ref "${num_ref}" \
        --neighbor_stride "${neighbor_stride}" \
        --savefps "${savefps}"
    
    # Move inpainted video to experiment folder
    cd ..
    mv -f "E2FGVI/results/stretched_results.mp4" "embrace/videos/${video_name}/scene_${scene_number}/${width}x${height}/$experiment_name/nei_${neighbor_length}_ref_${ref_stride}_sub_${subvideo_length}.mp4"
    rm "E2FGVI/examples/stretched.mp4"
    rm -r "E2FGVI/examples/stretched_mask/"
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
encoder_params=""
inpainter_params=""

# embrace
video_name=$1
video_path=$2
width=$3
height=$4
square_size=$5
to_remove=$6
alpha=$7
smoothing_factor=$8
encoder=$9
inpainter=${10}
# Initialize the experiment_name with common parameters
experiment_name="${video_name}_${width}x${height}_ss_${square_size}_tr_${to_remove}_alp_${alpha}_sf_${smoothing_factor}"
# Shift off embrace's parameters
shift 10

# avc
if [[ "$encoder" == "avc" ]]; then
    experiment_name+="_avc"
fi

# hnerv
if [[ "$encoder" == "hnerv" ]]; then
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
    # Fill encoder_params
    encoder_params="$ks $enc_strds $enc_dim $fc_hw $reduce $lower_width $dec_strds $conv_type $norm $act $workers $batchSize $epochs $lr $loss $out_bias $eval_freq $quant_model_bit $quant_embed_bit"
    # Add HNeRV parameters to the experiment name
    experiment_name+="_hnerv_ks_${ks}_es_${enc_strds}_ed_${enc_dim}_fh_${fc_hw}_red_${reduce}_lw_${lower_width}_ds_${dec_strds}_ct_${conv_type}_nor_${norm}_act_${act}_wor_${workers}_bs_${batchSize}_epo_${epochs}_lr_${lr}_los_${loss}_ob_${out_bias}_ef_${eval_freq}_qmb_${quant_model_bit}_qeb_${quant_embed_bit}"
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
    # Construct the inpainter_params string for ProPainter
    inpainter_params="$neighbor_length $ref_stride $subvideo_length $mask_dilation $raft_iter"
    # Add ProPainter parameters to the experiment name
    experiment_name+="_propainter_nl_${neighbor_length}_rs_${ref_stride}_sl_${subvideo_length}_md_${mask_dilation}_ri_${raft_iter}"
    # Shift off the ProPainter-specific parameters
    shift 5
fi

# e2fgvi
if [[ "$inpainter" == "e2fgvi" ]]; then
    step=$1
    num_ref=$2
    neighbor_stride=$3
    savefps=$4
    # Construct the inpainter_params string for E2FGVI
    inpainter_params="$step $num_ref $neighbor_stride $savefps"
    # Add E2FGVI parameters to the experiment name
    experiment_name+="e2fgvi_ste_${step}_nr_${num_ref}_ns_${neighbor_stride}_sf_${savefps}"
    # Shift off the E2FGVI-specific parameters
    shift 4
fi

# check whether width and height are multiples of square_size, otherwise make them
width=$(make_lower_multiple $width $square_size)
height=$(make_lower_multiple $height $square_size)

resolution="${width}x${height}"

# pass parameters to python scripts
export resolution=$resolution
export square_size=$square_size
export to_remove=$to_remove
export alpha=$alpha
export smoothing_factor=$smoothing_factor
export experiment_name=$experiment_name

# SERVER SIDE

server_start_time=$(date +%s)

# resize video based on experiment resolution, save into experiment folder
experiment_folder="experiments/${experiment_name}"
reference_frames="${experiment_folder}/reference_frames"
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
shrunk_folder="${experiment_folder}/shrunk"
encoding_size=$(echo "scale=2; $(stat -c%s $original_file)/1024/1024" | bc)

# Call the encoding function with HNeRV parameters if necessary
if [[ $encoder == "hnerv" ]]; then
    hnerv_input_folder="data/${experiment_name}"
    encode_with_hnerv $experiment_name $shrunk_width $shrunk_height $encoding_size $encoder_params
elif [[ $encoder == "avc" ]]; then
    frames_into_video $shrunk_folder $shrunk_video $bitrate
fi

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
