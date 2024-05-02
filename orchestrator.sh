cd
cd embrace
# should the list of packages be installed at this point?
source activate embrace

# Define the directory for log files
log_dir="logs"

# Ensure the log directory exists
mkdir -p "$log_dir"

# Generate timestamp for log file
timestamp=$(date +"%Y%m%d_%H%M%S")
log_file="${log_dir}/script_log_${timestamp}.txt"

# Redirect stdout and stderr to the new log file
exec > >(tee -a "$log_file") 2>&1

echo "Orchestrator script started."

# define running parameters
export log_file="$log_file"
export unprocessed_video_file="inputs/Tears_of_Steel_1080p.mov"

export scene_similarity_threshold="45"
export max_scenes="2"

export scene_number="2"
export width="1280"
export height="720"
export square_size="16"
export filter_factor="2"
export num_processes="16"

python server.py

# check bitrate saving from shrinking video TODO: mask size should be added


python client.py