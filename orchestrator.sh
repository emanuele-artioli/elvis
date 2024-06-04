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
    if [[ -f "$output_dir" ]]; then
        echo "$output_dir already exists"
        return 1
    else
        mkdir $output_dir
    fi

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

# # run server script
# python server.py 

# # move shrunk and masks into experiment folder
# mv -f "videos/$1/scene_$2/"$3:$4"/shrunk/" "videos/$1/scene_$2/"$3:$4"/$experiment_name/shrunk"
# mv -f "videos/$1/scene_$2/"$3:$4"/masks/" "videos/$1/scene_$2/"$3:$4"/$experiment_name/masks"

# # get shrunk video from frames
# frames_into_video "videos/$1/scene_$2/"$3:$4"/$experiment_name/shrunk" "videos/$1/scene_$2/"$3:$4"/$experiment_name/shrunk.mp4" $5 ${12}

# # run client script
# python client.py

# # get stretched video from frames
# frames_into_video "videos/$1/scene_$2/"$3:$4"/$experiment_name/stretched" "videos/$1/scene_$2/"$3:$4"/$experiment_name/stretched.mp4" $5 ${12}

# # inpaint video
# cd
# stretched_video_path="embrace/videos/$1/scene_$2/"$3:$4"/$experiment_name/stretched.mp4"
# mask_path="embrace/videos/$1/scene_$2/"$3:$4"/$experiment_name/masks/frame_0001.png"
# cp $stretched_video_path "ProPainter/inputs/video_completion/stretched.mp4"
# cp $mask_path "ProPainter/inputs/video_completion/frame0001.png"
# # TODO: we can change the mask at each frame, and set masks to alternate the block they keep so that each block has more references.
# cd ProPainter
# python inference_propainter.py \
#     --video inputs/video_completion/stretched.mp4 \
#     --mask inputs/video_completion/frame0001.png \
#     --neighbor_length $9 \
#     --ref_stride ${10} \
#     --subvideo_length ${11}

# # move inpainted to experiment folder
# cd
# mv -f "ProPainter/results/stretched/inpaint_out.mp4" "embrace/videos/$1/scene_$2/"$3:$4"/$experiment_name/nei_${9}_ref_${10}_sub_${11}.mp4"

# cd embrace

# # quality degradation of inpainted video
# reference_video_path="videos/$1/scene_$2/"$3:$4"/original.mp4"
# distorted_video_path="videos/$1/scene_$2/"$3:$4"/$experiment_name/nei_${9}_ref_${10}_sub_${11}.mp4"
# reference_target_path="videos/$1/scene_$2/"$3:$4"/reference.yuv"
# distorted_target_path="videos/$1/scene_$2/"$3:$4"/distorted.yuv"
# # encode inpainted for vmaf, and save it in embrace folder renamed based on its parameters configuration
# ffmpeg -i $reference_video_path -f rawvideo -pix_fmt yuv420p $reference_target_path
# ffmpeg -i $distorted_video_path -f rawvideo -pix_fmt yuv420p $distorted_target_path
# # calculate quality degradation and save it as csv
# vmafossexec yuv420p $3 $4 $reference_target_path $distorted_target_path /home/shared/athena/vmaf/model/vmaf_v0.6.1.pkl \
#     --log videos/$1/scene_$2/"$3:$4"/$experiment_name/nei_${9}_ref_${10}_sub_${11}_inpainted.csv\
#     --log-fmt csv --psnr --ssim --ms-ssim
# # delete yuv files
# rm "videos/$1/scene_$2/"$3:$4"/reference.yuv"
# rm "videos/$1/scene_$2/"$3:$4"/distorted.yuv"

# # quality degradation of original video
# reference_video_path="videos/$1/scene_$2/"$3:$4"/original.mp4"
# distorted_video_path="videos/$1/scene_$2/"$3:$4"/$experiment_name/original.mp4"
# reference_target_path="videos/$1/scene_$2/"$3:$4"/reference.yuv"
# distorted_target_path="videos/$1/scene_$2/"$3:$4"/distorted.yuv"
# # encode inpainted for vmaf, and save it in embrace folder renamed based on its parameters configuration
# ffmpeg -i $reference_video_path -f rawvideo -pix_fmt yuv420p $reference_target_path
# ffmpeg -i $distorted_video_path -f rawvideo -pix_fmt yuv420p $distorted_target_path
# # calculate quality degradation and save it as csv
# vmafossexec yuv420p $3 $4 $reference_target_path $distorted_target_path /home/shared/athena/vmaf/model/vmaf_v0.6.1.pkl \
#     --log videos/$1/scene_$2/"$3:$4"/$experiment_name/nei_${9}_ref_${10}_sub_${11}_original.csv\
#     --log-fmt csv --psnr --ssim --ms-ssim
# # delete yuv files
# rm "videos/$1/scene_$2/"$3:$4"/reference.yuv"
# rm "videos/$1/scene_$2/"$3:$4"/distorted.yuv"

# # run metrics script
# python metrics.py

# # save storage when running many experiments by deleting files and folders that will not be needed anymore
# # delete shrunk folder
# rm -r "videos/$1/scene_$2/"$3:$4"/$experiment_name/shrunk"
# # delete stretched folder
# rm -r "videos/$1/scene_$2/"$3:$4"/$experiment_name/stretched"