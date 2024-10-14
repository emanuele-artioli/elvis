#!/bin/bash

# Define lists for each parameter
videos=("bmx-trees") # ("bear" "blackswan" "bmx-bumps" "bmx-trees" "breakdance-flare")
widths=("1280") # ("640" "960" "1280" "1600")
heights=("960") # ("640" "960" "1280" "1600")
square_sizes=("16") # ("16" "32" "64")
to_remove=("20.0") # ("10.0" "20.0")
alpha=("0.5") # ("0.0" "0.25" "0.5" "0.75" "1.0")
smoothing_factor=("0.0") # ("0.0" "0.25" "0.5" "0.75" "1.0")
codecs=("hnerv") # ("avc" "hnerv")
inpainters=("e2fgvi") # ("propainter" "e2fgvi")

# Define parameters for each codec
hnerv_params_ks=("0_3_3") # ("2_2_2" "0_3_3" "0_4_4")
hnerv_params_enc_strds=("5 4 4 2 2") # ("4 3 3 2" "5 4 4 2 2" "6 6 5 5 4 2")
hnerv_params_enc_dim=("64_16") # ("32_8" "64_16" "128_32")
hnerv_params_fc_hw=("9_16") # ("4_9" "9_16" "16_25")
hnerv_params_reduce=("1.2") # ("1.0" "1.2" "1.5")
hnerv_params_lower_width=("32") # ("12" "32" "64")
hnerv_params_dec_strds=("5 4 4 2 2") # ("4 3 3 2" "5 4 4 2 2" "6 6 5 5 4 2")
hnerv_params_conv_type=("convnext pshuffel") # ("convnext pshuffel")
hnerv_params_norm=("none") # ("none" "bn" "in")
hnerv_params_act=("relu") # ("relu" "leaky" "gelu")
hnerv_params_workers=("8") # ("4" "8" "16" "32")
hnerv_params_batchSize=("1") # ("1" "2" "4")
hnerv_params_epochs=(100) # (10 30 100)
hnerv_params_lr=(0.001) # (0.01 0.001 0.0001)
hnerv_params_loss=("Fusion6") # ("Fusion6" "L2")
hnerv_params_out_bias=("tanh") # ("tanh")
hnerv_params_eval_freq=(10) # (5 10)
hnerv_params_quant_model_bit=(8) # (6 8)
hnerv_params_quant_embed_bit=(6) # (6 8)

propainter_params_neighbor_length=(8 16 24) # (8 16 24)
propainter_params_ref_stride=(2 4) # (2 4)
propainter_params_subvideo_length=(24 48) # (24 48)
propainter_params_mask_dilation=(0) # (0)
propainter_params_raft_iter=(10 30 100) # (10 30 100)

e2fgvi_params_step=(10) # (5 10)
e2fgvi_params_num_ref=(-1) # (-1 1 2 4)
e2fgvi_params_neighbor_stride=(4) # (2 4)
e2fgvi_params_savefps=(24) # (24)

# Randomly select codec and inpainter
codec=${codecs[$RANDOM % ${#codecs[@]}]}
inpainter=${inpainters[$RANDOM % ${#inpainters[@]}]}

# Randomly select values for the parameters based on codec and inpainter
video=${videos[$RANDOM % ${#videos[@]}]}
video_path="videos/${video}/scene_1.mp4"
width=${widths[$RANDOM % ${#widths[@]}]}
height=${heights[$RANDOM % ${#heights[@]}]}
square_size=${square_sizes[$RANDOM % ${#square_sizes[@]}]}
to_remove=${to_remove[$RANDOM % ${#to_remove[@]}]}
alpha=${alpha[$RANDOM % ${#alpha[@]}]}
smoothing_factor=${smoothing_factor[$RANDOM % ${#smoothing_factor[@]}]}

# Initialize an empty argument list
args=()

# Add common arguments
args+=("$video")
args+=("$video_path")
args+=("$width")
args+=("$height")
args+=("$square_size")
args+=("$to_remove")
args+=("$alpha")
args+=("$smoothing_factor")
args+=("$codec")
args+=("$inpainter")

# Add codec-specific arguments
if [[ "$codec" == "hnerv" ]]; then
    ks=${hnerv_params_ks[$RANDOM % ${#hnerv_params_ks[@]}]}
    enc_strds=${hnerv_params_enc_strds[$RANDOM % ${#hnerv_params_enc_strds[@]}]}
    enc_dim=${hnerv_params_enc_dim[$RANDOM % ${#hnerv_params_enc_dim[@]}]}
    fc_hw=${hnerv_params_fc_hw[$RANDOM % ${#hnerv_params_fc_hw[@]}]}
    reduce=${hnerv_params_reduce[$RANDOM % ${#hnerv_params_reduce[@]}]}
    lower_width=${hnerv_params_lower_width[$RANDOM % ${#hnerv_params_lower_width[@]}]}
    dec_strds=${hnerv_params_dec_strds[$RANDOM % ${#hnerv_params_dec_strds[@]}]}
    conv_type=${hnerv_params_conv_type[$RANDOM % ${#hnerv_params_conv_type[@]}]}
    norm=${hnerv_params_norm[$RANDOM % ${#hnerv_params_norm[@]}]}
    act=${hnerv_params_act[$RANDOM % ${#hnerv_params_act[@]}]}
    workers=${hnerv_params_workers[$RANDOM % ${#hnerv_params_workers[@]}]}
    batchSize=${hnerv_params_batchSize[$RANDOM % ${#hnerv_params_batchSize[@]}]}
    epochs=${hnerv_params_epochs[$RANDOM % ${#hnerv_params_epochs[@]}]}
    lr=${hnerv_params_lr[$RANDOM % ${#hnerv_params_lr[@]}]}
    loss=${hnerv_params_loss[$RANDOM % ${#hnerv_params_loss[@]}]}
    out_bias=${hnerv_params_out_bias[$RANDOM % ${#hnerv_params_out_bias[@]}]}
    eval_freq=${hnerv_params_eval_freq[$RANDOM % ${#hnerv_params_eval_freq[@]}]}
    quant_model_bit=${hnerv_params_quant_model_bit[$RANDOM % ${#hnerv_params_quant_model_bit[@]}]}
    quant_embed_bit=${hnerv_params_quant_embed_bit[$RANDOM % ${#hnerv_params_quant_embed_bit[@]}]}

    args+=("$ks" "$enc_strds" "$enc_dim" "$fc_hw" "$reduce" "$lower_width" "$dec_strds" "$conv_type" "$norm" "$act" "$workers" "$batchSize" "$epochs" "$lr" "$loss" "$out_bias" "$eval_freq" "$quant_model_bit" "$quant_embed_bit")
fi

# Add inpainter-specific arguments
if [[ "$inpainter" == "propainter" ]]; then
    neighbor_length=${propainter_params_neighbor_length[$RANDOM % ${#propainter_params_neighbor_length[@]}]}
    ref_stride=${propainter_params_ref_stride[$RANDOM % ${#propainter_params_ref_stride[@]}]}
    subvideo_length=${propainter_params_subvideo_length[$RANDOM % ${#propainter_params_subvideo_length[@]}]}
    mask_dilation=${propainter_params_mask_dilation[$RANDOM % ${#propainter_params_mask_dilation[@]}]}
    raft_iter=${propainter_params_raft_iter[$RANDOM % ${#propainter_params_raft_iter[@]}]}

    args+=("$neighbor_length" "$ref_stride" "$subvideo_length" "$mask_dilation" "$raft_iter")
fi

if [[ "$inpainter" == "e2fgvi" ]]; then
    step=${e2fgvi_params_step[$RANDOM % ${#e2fgvi_params_step[@]}]}
    num_ref=${e2fgvi_params_num_ref[$RANDOM % ${#e2fgvi_params_num_ref[@]}]}
    neighbor_stride=${e2fgvi_params_neighbor_stride[$RANDOM % ${#e2fgvi_params_neighbor_stride[@]}]}
    savefps=${e2fgvi_params_savefps[$RANDOM % ${#e2fgvi_params_savefps[@]}]}

    args+=("$step" "$num_ref" "$neighbor_stride" "$savefps")
fi

# Run the orchestrator with the dynamically constructed arguments
echo "Chosen parameters:"
echo "Video: $video"
echo "Video Path: $video_path"
echo "Width: $width"
echo "Height: $height"
echo "Square Size: $square_size"
echo "To Remove: $to_remove"
echo "Alpha: $alpha"
echo "Smoothing Factor: $smoothing_factor"
echo "Codec: $codec"
echo "Inpainter: $inpainter"

# Print codec-specific parameters
if [[ "$codec" == "hnerv" ]]; then
    echo "ks: $ks"
    echo "enc_strds: $enc_strds"
    echo "enc_dim: $enc_dim"
    echo "fc_hw: $fc_hw"
    echo "reduce: $reduce"
    echo "lower_width: $lower_width"
    echo "dec_strds: $dec_strds"
    echo "conv_type: $conv_type"
    echo "norm: $norm"
    echo "act: $act"
    echo "workers: $workers"
    echo "batchSize: $batchSize"
    echo "epochs: $epochs"
    echo "lr: $lr"
    echo "loss: $loss"
    echo "out_bias: $out_bias"
    echo "eval_freq: $eval_freq"
    echo "quant_model_bit: $quant_model_bit"
    echo "quant_embed_bit: $quant_embed_bit"
fi

# Print inpainter-specific parameters
if [[ "$inpainter" == "propainter" ]]; then
    echo "Neighbor Length: $neighbor_length"
    echo "Ref Stride: $ref_stride"
    echo "Subvideo Length: $subvideo_length"
    echo "Mask Dilation: $mask_dilation"
    echo "Raft Iter: $raft_iter"
fi

if [[ "$inpainter" == "e2fgvi" ]]; then
    echo "Step: $step"
    echo "Num Ref: $num_ref"
    echo "Neighbor Stride: $neighbor_stride"
    echo "Save FPS: $savefps"
fi

# EXCEPTION RULES

# Check if hnerv is chosen and the product of square_size and to_remove is a multiple of 320 (otherwise HNeRV does not work)
product=$(echo "$square_size * $to_remove" | bc)
if [[ "$codec" == "hnerv" ]]; then
    if ! (( $(echo "$product % 320 == 0" | bc) )); then
        echo "Error: The product of square_size ($square_size) and to_remove ($to_remove) is $product and is not a multiple of 320."
        exit 1
    fi
fi

# Check if propainter is chosen and ensure width and height are less than 1280
if [[ "$inpainter" == "propainter" ]]; then
    if (( width >= 1280 || height >= 1280 )); then
        echo "Error: When using propainter, both width ($width) and height ($height) must be less than 1280."
        exit 1
    fi
fi

./orchestrator.sh "${args[@]}"
