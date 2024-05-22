cd
cd embrace
# should the list of packages be installed at this point?
source activate embrace

echo "Orchestrator script started."

# define running parameters TODO: put them as arguments of the script, so many config can be run in chain
unprocessed_video_file="inputs/Tears_of_Steel_1080p.mov"
scene_similarity_threshold="45"
max_scenes="2"
scene_number="2"
width="640"
height="384"
square_size="16"
horizontal_stride="2"
vertical_stride="2"
num_processes="64"
fp="fp16" #TODO: to allow for fp32, make this parameter in python like "if fp else None"
neighbor_length="3"
ref_stride="3"
subvideo_length="11"

# define video name
video_name="scene_${scene_number}_${height}p_square_${square_size}_stride_${horizontal_stride}x${vertical_stride}_inpaint_${fp}_${neighbor_length}_${ref_stride}_${subvideo_length}"

# Define the directory for log files
log_dir="logs"
# Ensure the log directory exists
mkdir -p "$log_dir"
# Generate name for log file
log_file="${log_dir}/${video_name}.txt"
# Redirect stdout and stderr to the new log file
exec > >(tee -a "$log_file") 2>&1

# export running parameters
export log_file="$log_file"
export unprocessed_video_file="$unprocessed_video_file"
export scene_similarity_threshold="$scene_similarity_threshold"
export max_scenes="$max_scenes"
export scene_number="$scene_number"
export width="$width"
export height="$height"
export square_size="$square_size"
export horizontal_stride="$horizontal_stride"
export vertical_stride="$vertical_stride"
export num_processes="$num_processes"

# running server script and extracting video folder and file size ratio from logs
output=$(python server.py)
video_folder=$(grep "video_folder:" "$log_file" | cut -d " " -f 2)
video_size_ratio=$(grep "Video size ratio:" "$log_file" | cut -d " " -f 2)

# passing variables to client script and running it
export video_folder="$video_folder"
python client.py

# inpaint video
SECONDS=0
stretched_video_path="${PWD}/${video_folder}/stretched.avi"
mask_path="${PWD}/${video_folder}/masks/frame0001.png"
cd
cd ProPainter
source activate propainter
cp $stretched_video_path "inputs/video_completion/stretched.avi"
cp $mask_path "inputs/video_completion/frame0001.png"
# TODO: we can change the mask at each frame, and set masks to alternate the block they keep so that each block has more references.
python inference_propainter.py --video inputs/video_completion/stretched.avi --mask inputs/video_completion/frame0001.png\
 --$fp --neighbor_length $neighbor_length --ref_stride $ref_stride --subvideo_length $subvideo_length > /dev/null
echo "Inpainting process run in $SECONDS seconds."

# compare quality degradation of inpainted video with regularly encoded video at same bitrate
cd
reference_video_path="${PWD}/embrace/${video_folder}/scene_${scene_number}.avi"
distorted_video_path="${PWD}/ProPainter/results/stretched/inpaint_out.avi"
reference_target_path="${PWD}/embrace/inpainted/${video_name}_reference.yuv"
distorted_target_path="${PWD}/embrace/inpainted/${video_name}_distorted.yuv"
# encode inpainted for vmaf, and save it in embrace folder renamed based on its parameters configuration
ffmpeg -i $reference_video_path -f rawvideo -pix_fmt yuv420p -s ${width}x${height} $reference_target_path > /dev/null
ffmpeg -i $distorted_video_path -f rawvideo -pix_fmt yuv420p -s ${width}x${height} $distorted_target_path > /dev/null
# calculate quality degradation and save it as csv
vmafossexec yuv420p $width $height $reference_target_path $distorted_target_path /home/shared/athena/vmaf/model/vmaf_v0.6.1.pkl\
 --log ${PWD}/embrace/logs/${video_name}.csv\
 --log-fmt csv --psnr --ssim --ms-ssim



# calculate flolpips
# cd flolpips
# source activate embrace
# export reference_target_path="$reference_target_path"
# export distorted_target_path="$distorted_target_path"
# python embrace.py