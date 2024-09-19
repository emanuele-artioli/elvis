#!/bin/bash

# Define lists for each parameter
videos=("bear")
widths=("1280")
heights=("960")
square_sizes=("16")
to_remove=("20.0")
alpha=("0.5")
smoothing_factor=("0.0")
codecs=("hnerv")
inpainters=("e2fgvi")

# Define parameters for each codec
hnerv_params_ks=("0_3_3")
hnerv_params_enc_strds=("5 4 4 2 2")
hnerv_params_enc_dim=("64_16")
hnerv_params_fc_hw=("9_16")
hnerv_params_reduce=("1.2")
hnerv_params_lower_width=("32")
hnerv_params_dec_strds=("5 4 4 2 2")
hnerv_params_conv_type=("convnext pshuffel")
hnerv_params_norm=("none")
hnerv_params_act=("gelu")
hnerv_params_workers=("8")
hnerv_params_batchSize=("1")
hnerv_params_epochs=(100)
hnerv_params_lr=(0.001)
hnerv_params_loss=("Fusion6")
hnerv_params_out_bias=("tanh")
hnerv_params_eval_freq=(10)
hnerv_params_quant_model_bit=(8)
hnerv_params_quant_embed_bit=(6)

propainter_params_neighbor_length=(8)
propainter_params_ref_stride=(2)
propainter_params_subvideo_length=(48)
propainter_params_mask_dilation=(0)
propainter_params_raft_iter=(10)

e2fgvi_params_step=(10)
e2fgvi_params_num_ref=(-1)
e2fgvi_params_neighbor_stride=(2)
e2fgvi_params_savefps=(24)

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
./orchestrator.sh "${args[@]}"
