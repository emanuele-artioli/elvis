#!/bin/bash

# SETUP

# pass parameters to python scripts
export video_name=$1
export scene_number=$2
export resolution="$3:$4"
export frame_rate=$5 # TODO: add the damn frame interpolation, of course!!
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
mkdir -p "videos/$1/scene_$2/"$3:$4"/$experiment_name"

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
    ffmpeg -i "$input_file" -vf scale="$resolution":force_original_aspect_ratio=increase,crop="$resolution" "$output_dir/frame_%04d.png"
}
# resize scene based on experiment resolution, save into 
resize_video "videos/$1/scene_$2.mp4" "videos/$1/scene_$2/$3:$4/original" "$3:$4"

frames_into_video() {
    local input_dir="$1"
    local output_file="$2"
    local frame_rate="$3"
    local bitrate="$4"

    # Check if the output already exists
    if [[ -f "$output_file" ]]; then
        echo "$output_file already exists"
        return 1
    fi

    ffmpeg -framerate "$frame_rate" -i "$input_dir/frame_%04d.png" \
           -b:v ${bitrate} -maxrate ${bitrate} -minrate ${bitrate} -bufsize ${bitrate} \
           -c:v libx265 -pix_fmt yuv420p "$output_file"
}

frames_into_video videos/$1/scene_$2/"$3:$4"/original videos/$1/scene_$2/"$3:$4"/original.mp4 $5 ${12}

# run server script
python server.py 

# move shrunk and masks into experiment folder
mv -f "videos/$1/scene_$2/"$3:$4"/shrunk/" "videos/$1/scene_$2/"$3:$4"/$experiment_name/shrunk"
mv -f "videos/$1/scene_$2/"$3:$4"/masks/" "videos/$1/scene_$2/"$3:$4"/$experiment_name/masks"

# get shrunk video from frames
frames_into_video "videos/$1/scene_$2/"$3:$4"/$experiment_name/shrunk" "videos/$1/scene_$2/"$3:$4"/$experiment_name/shrunk.mp4" $5 ${12}

# run client script
python client.py

# get stretched video from frames
frames_into_video "videos/$1/scene_$2/"$3:$4"/$experiment_name/stretched" "videos/$1/scene_$2/"$3:$4"/$experiment_name/stretched.mp4" $5 ${12}

# INPAINTING

cd
stretched_video_path="embrace/videos/$1/scene_$2/"$3:$4"/$experiment_name/stretched.mp4"
mask_path="embrace/videos/$1/scene_$2/"$3:$4"/$experiment_name/masks/frame_0001.png"
cp $stretched_video_path "ProPainter/inputs/video_completion/stretched.mp4"
cp $mask_path "ProPainter/inputs/video_completion/frame_0001.png"
# TODO: we can change the mask at each frame, and set masks to alternate the block they keep so that each block has more references.
cd ProPainter
python inference_propainter.py \
    --video inputs/video_completion/stretched.mp4 \
    --mask inputs/video_completion/frame_0001.png \
    --neighbor_length $9 \
    --ref_stride ${10} \
    --subvideo_length ${11}

# move inpainted to experiment folder
cd
mv -f "ProPainter/results/stretched/inpaint_out.mp4" "embrace/videos/$1/scene_$2/"$3:$4"/$experiment_name/nei_${9}_ref_${10}_sub_${11}.mp4"
cd embrace

# QUALITY MEASUREMENT

# Function to run ffmpeg command and extract VMAF and PSNR values
calculate_vmaf() {
    local reference_file=$1
    local distorted_file=$2
    local log_file=$3
    local model_file=$4

    # Default filter_complex for 1080p
    filter_complex="[0:v]scale=1920x1080:flags=bicubic[main]; [1:v]scale=1920x1080:flags=bicubic[ref]; [main][ref]libvmaf=log_path=${log_file}:log_fmt=csv"

    if [ "$model_file" == "vmaf_4k_v0.6.1" ]; then
        filter_complex="[0:v]scale=3840x2160:flags=bicubic[main]; [1:v]scale=3840x2160:flags=bicubic[ref]; [main][ref]libvmaf=model=version=${model_file}:log_path=${log_file}:log_fmt=csv"
    elif [ "$model_file" == "vmaf_v0.6.1" ]; then
        filter_complex="[0:v]scale=1920x1080:flags=bicubic[main]; [1:v]scale=1920x1080:flags=bicubic[ref]; [main][ref]libvmaf=model=version=${model_file}:log_path=${log_file}:log_fmt=csv"
    fi

    ffmpeg -i "$reference_file" -i "$distorted_file" -filter_complex "$filter_complex" -f null -
}

# encode reference video
reference_input_path="videos/$1/scene_$2/"$3:$4"/original/frame_%04d.png"
reference_output_path="videos/$1/scene_$2/"$3:$4"/reference.yuv"
# Check if the output already exists, if not create it
if [[ -f "$reference_output_path" ]]; then
    echo "$reference_output_path already exists"   
else
    ffmpeg -i $reference_input_path -f rawvideo -pix_fmt yuv420p $reference_output_path
fi

# compare reference with original video
original_input_path="videos/$1/scene_$2/"$3:$4"/original.mp4"
original_output_path="videos/$1/scene_$2/"$3:$4"/original.yuv"
# Check if the output already exists, if not create it
if [[ -f "$original_output_path" ]]; then
    echo "$original_output_path already exists"   
else
    ffmpeg -i $original_input_path -f rawvideo -pix_fmt yuv420p $original_output_path
    # calculate quality degradation and save it as csv
    # vmafossexec yuv420p $3 $4 $reference_output_path $original_output_path /home/shared/athena/vmaf/model/vmaf_v0.6.1.pkl \
    #     --log videos/$1/scene_$2/"$3:$4"/original.csv\
    #     --log-fmt csv --psnr --ssim --ms-ssim
    calculate_vmaf "$original_input_path" "$original_input_path" "videos/$1/scene_$2/"$3:$4"/original.csv" "vmaf_v0.6.1"
fi

# compare reference with inpainted video
inpainted_input_path="videos/$1/scene_$2/"$3:$4"/$experiment_name/nei_${9}_ref_${10}_sub_${11}.mp4"
inpainted_output_path="videos/$1/scene_$2/"$3:$4"/$experiment_name/inpainted.yuv"
# Check if the output already exists, if not create it
if [[ -f "$inpainted_output_path" ]]; then
    echo "$inpainted_output_path already exists"   
else
    ffmpeg -i $inpainted_input_path -f rawvideo -pix_fmt yuv420p $inpainted_output_path
    # calculate quality degradation and save it as csv
    # vmafossexec yuv420p $3 $4 $reference_output_path $inpainted_output_path /home/shared/athena/vmaf/model/vmaf_v0.6.1.pkl \
    #     --log videos/$1/scene_$2/"$3:$4"/$experiment_name/nei_${9}_ref_${10}_sub_${11}_inpainted.csv\
    #     --log-fmt csv --psnr --ssim --ms-ssim
    calculate_vmaf "$reference_input_path" "$inpainted_input_path" "videos/$1/scene_$2/"$3\:$4"/$experiment_name/nei_${9}_ref_${10}_sub_${11}_inpainted.csv" "vmaf_v0.6.1"
fi

# run metrics script
python metrics.py

# save storage when running many experiments by deleting files and folders that will not be needed anymore
# delete shrunk folder
rm -r "videos/$1/scene_$2/"$3:$4"/$experiment_name/shrunk"
# delete stretched folder
rm -r "videos/$1/scene_$2/"$3:$4"/$experiment_name/stretched"