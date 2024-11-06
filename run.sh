#!/bin/bash

# TODO: log which combinations go into error, so the parameters can be adjusted to avoid them

# Define lists for each parameter
videos=("bear" "bike-packing" "blackswan" "bmx-bumps" "bmx-trees" "breakdance-flare" "bus" "camel" "car-roundabout"
        "car-turn" "cat-girl" "classic-car" "cows" "dog" "dog-gooses" "dogs-scale" "drift-chicane" "drift-straight"
        "drift-turn" "drone" "elephant" "flamingo" "goat" "gold-fish" "hike" "hockey" "horsejump-high" "horsejump-low"
        "india" "judo" "kid-football" "kite-surf" "kite-walk" "koala" "lab-coat" "lady-running" "libby" "lindy-hop"
        "loading" "longboard" "lucia" "mallard-fly" "mallard-water" "mbike-trick" "miami-surf"
)
widths=("960" "1280" "1600")
heights=("540" "720" "900")
square_sizes=("8" "16" "32" "64") # ("16" "32" "64")
to_remove=("0.25" "0.5" "0.75") # ("10.0" "20.0")
alpha=("0.0" "0.5" "1.0") # ("0.0" "0.25" "0.5" "0.75" "1.0")
smoothing_factor=("0.0" "0.5" "1.0") # ("0.0" "0.25" "0.5" "0.75" "1.0")
codecs=("hnerv") # ("avc" "hnerv")
inpainters=("propainter" "e2fgvi") # ("propainter" "e2fgvi")

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
hnerv_params_workers=(16) # ("4" "8" "16" "32")
hnerv_params_batchSize=(2 4) # ("1" "2" "4")
hnerv_params_epochs=(50 75 100) # (10 30 100)
hnerv_params_lr=(0.001) # (0.01 0.001 0.0001)
hnerv_params_loss=("Fusion6") # ("Fusion6" "L2")
hnerv_params_out_bias=("tanh") # ("tanh")
hnerv_params_eval_freq=(1 3 10) # (5 10)
hnerv_params_quant_model_bit=(8) # (6 8 10)
hnerv_params_quant_embed_bit=(6) # (6)

propainter_params_neighbor_length=(8 16 24) # (8 16 24)
propainter_params_ref_stride=(2 4) # (2 4)
propainter_params_subvideo_length=(24 48) # (24 48)
propainter_params_mask_dilation=(0 2 4) # (0)
propainter_params_raft_iter=(3 10 30) # (10 30 100)

e2fgvi_params_step=(5 10) # (5 10)
e2fgvi_params_num_ref=(-1 1) # (-1 1 2 4)
e2fgvi_params_neighbor_stride=(2 4) # (2 4)
e2fgvi_params_savefps=(24) # (24)

# Randomly select codec and inpainter
codec=${codecs[$RANDOM % ${#codecs[@]}]}
inpainter=${inpainters[$RANDOM % ${#inpainters[@]}]}

# Randomly select values for the parameters based on codec and inpainter
video=${videos[$RANDOM % ${#videos[@]}]}
raw_frames="Datasets/DAVIS/${video}"
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
args+=("$raw_frames")
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

# EXCEPTION RULES

# # Check if hnerv is chosen and the product of square_size and to_remove is a multiple of 320 (otherwise HNeRV does not work)
# # Ensure the product is also smaller than the initial width, or there will be no video to inpaint 
# product=$(echo "$square_size * $to_remove" | bc)
# if [[ "$codec" == "hnerv" ]]; then
#     if ! (( $(echo "$product % 320 == 0" | bc) )); then
#         echo "Error: The product of square_size ($square_size) and to_remove ($to_remove) is $product and is not a multiple of 320."
#         exit 1
#     elif ! (( $(echo "$product < $width" | bc) )); then
#         echo "Error: The product of square_size ($square_size) and to_remove ($to_remove) is not smaller than video width $width."
#         exit 1
#     fi
# fi

# Check if propainter is chosen and ensure width and height are less than 1280
max_width=1920
max_height=1080
if [[ "$inpainter" == "propainter" ]]; then
    if (( width > 1920 || height > 1080 )); then
        echo "Error: When using propainter, both width ($width) and height ($height) must be less than specified max values."
        exit 1
    fi
fi

# Run orchestrator.sh and capture the experiment_name
./orchestrator.sh "${args[@]}"
# Capture experiment_name from the file
experiment_name=$(cat experiment_name.txt)
rm experiment_name.txt  # Clean up the temporary file

# # CLEANING UP: if the inpainted video exists, it means the orchestrator ran successfully.
# then delete everything about that experiment except inpainted video, 
# otherwise keep everything for debugging.

# Set paths
EXPERIMENTS_DIR="experiments/${experiment_name}"
HNeRV_DATA="../HNeRV/data"
HNeRV_OUTPUT="../HNeRV/output/embrace"
UFO_DATASETS="../UFO/datasets/embrace"
UFO_RESULTS="../UFO/VSOD_results/wo_optical_flow/embrace"

# Check if inpainted.mp4 exists in the specific experiment folder
if [ -f "$EXPERIMENTS_DIR/inpainted.mp4" ]; then
    echo "inpainted.mp4 exists in $experiment_name, cleaning up..."

    # Remove everything except inpainted.mp4 and benchmark.mp4 in the experiments folder
    find "$EXPERIMENTS_DIR" -mindepth 1 ! -name 'inpainted.mp4' ! -name 'benchmark.mp4' -exec rm -rf {} +

    # Remove everything from the other specified folders
    rm -rf "$HNeRV_DATA"/*
    rm -rf "$HNeRV_OUTPUT"/*
    rm -rf "$UFO_DATASETS"/*
    rm -rf "$UFO_RESULTS"/*

    echo "Cleanup complete."

    # Launch run.sh once more
    echo "Relaunching run.sh..."
    ./run.sh
else
    echo "inpainted.mp4 not found in ${experiment_name}, stopping recursive execution."
fi
