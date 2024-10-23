import os
import pandas as pd

def get_codec_params(codec):
    codec_params = {
        "hnerv": ["ks", "enc_strds", "enc_dim", "fc_hw", "reduce", "lower_width", "dec_strds", "conv_type", "norm", "act", "workers", "batchSize", "epochs", "lr", "loss", "out_bias", "eval_freq", "quant_model_bit", "quant_embed_bit"],
        "avc": [],
        # Add other codecs if needed
    }
    params = codec_params.get(codec, [])
    return {f'{codec}_{key}': os.environ.get(f'{codec}_{key}') for key in params}

def get_inpainter_params(inpainter):
    inpainter_params = {
        "propainter": ["neighbor_length", "ref_stride", "subvideo_length", "mask_dilation", "raft_iter"],
        "e2fgvi": ["step", "num_ref", "neighbor_stride", "savefps"],
        # Add other inpainters if needed
    }
    params = inpainter_params.get(inpainter, [])
    return {f'{inpainter}_{key}': os.environ.get(f'{inpainter}_{key}') for key in params}

# Get parameters from environment
video_name = os.environ.get('video_name')
experiment_name = os.environ.get('experiment_name')
resolution = os.environ.get('resolution')
square_size = os.environ.get('square_size')
to_remove = float(os.environ.get('to_remove'))  # Changed to float
alpha = float(os.environ.get('alpha'))
smoothing_factor = float(os.environ.get('smoothing_factor'))
codec = os.environ.get('codec')
codec_params = get_codec_params(codec)
inpainter = os.environ.get('inpainter')
inpainter_params = get_inpainter_params(inpainter)
bitrate = os.environ.get('bitrate')
server_start_time = int(os.environ.get('server_start_time'))
client_start_time = int(os.environ.get('client_start_time'))
end_time = int(os.environ.get('end_time'))

# Calculate elapsed times
server_runtime = client_start_time - server_start_time
client_runtime = end_time - client_start_time

# Read experiment metrics
experiment_folder = f'experiments/{experiment_name}'
inpainted_metrics = f'{experiment_folder}/inpainted_metrics.csv'
benchmark_metrics = f'{experiment_folder}/benchmark_metrics.csv'

# Check if files exist to avoid errors
if os.path.isfile(inpainted_metrics) and os.path.isfile(benchmark_metrics):
    inpainted_df = pd.read_csv(inpainted_metrics)
    benchmark_df = pd.read_csv(benchmark_metrics)
    
    # Calculate aggregates
    inpainted_means = inpainted_df.mean(axis=0, numeric_only=True)
    inpainted_stds = inpainted_df.std(axis=0, numeric_only=True)
    benchmark_means = benchmark_df.mean(axis=0, numeric_only=True)
    benchmark_stds = benchmark_df.std(axis=0, numeric_only=True)
else:
    print("Metrics files do not exist.")
    inpainted_means = inpainted_stds = benchmark_means = benchmark_stds = pd.Series()

# Initialize results dictionary with basic parameters
results_dict = {
    'experiment_name': experiment_name,
    'video_name': video_name,
    'resolution': resolution,
    'square_size': square_size,
    'to_remove': to_remove,
    'alpha': alpha,
    'bitrate': bitrate,
    'server_runtime': server_runtime,
    'client_runtime': client_runtime,
    'smoothing_factor': smoothing_factor,
}

# Initialize codec and inpainter parameters to None with specific codec and inpainter names
all_codec_params = {
    "hnerv": ["ks", "enc_strds", "enc_dim", "fc_hw", "reduce", "lower_width", "dec_strds", "conv_type", "norm", "act", "workers", "batchSize", "epochs", "lr", "loss", "out_bias", "eval_freq", "quant_model_bit", "quant_embed_bit"],
    "avc": [],
    # Add other codecs if needed
}

all_inpainter_params = {
    "propainter": ["neighbor_length", "ref_stride", "subvideo_length", "mask_dilation", "raft_iter"],
    "e2fgvi": ["step", "num_ref", "neighbor_stride", "savefps"],
    # Add other inpainters if needed
}

# Set all codec params to None, prefixed with the specific codec name
for codec_name, params in all_codec_params.items():
    for param in params:
        results_dict[f'{codec_name}_{param}'] = None

# Set all inpainter params to None, prefixed with the specific inpainter name
for inpainter_name, params in all_inpainter_params.items():
    for param in params:
        results_dict[f'{inpainter_name}_{param}'] = None

# Add codec parameters to the dictionary with the specific codec name
results_dict.update(codec_params)

# Add inpainter parameters to the dictionary with the specific inpainter name
results_dict.update(inpainter_params)

# Add inpainted and benchmark metrics
results_dict['mse_ori_mean'] = benchmark_means.get('mse_avg', None)
results_dict['psnr_ori_mean'] = benchmark_means.get('psnr_avg', None)
results_dict['ssim_ori_mean'] = benchmark_means.get('ssim_avg', None)
results_dict['vmaf_ori_mean'] = benchmark_means.get('vmaf', None)
results_dict['lpips_ori_mean'] = benchmark_means.get('LPIPS', None)
results_dict['mse_inp_mean'] = inpainted_means.get('mse_avg', None)
results_dict['psnr_inp_mean'] = inpainted_means.get('psnr_avg', None)
results_dict['ssim_inp_mean'] = inpainted_means.get('ssim_avg', None)
results_dict['vmaf_inp_mean'] = inpainted_means.get('vmaf', None)
results_dict['lpips_inp_mean'] = inpainted_means.get('LPIPS', None)
results_dict['empty'] = ''

# Convert results to DataFrame
results = pd.DataFrame([results_dict])

# Append results to CSV
results_csv = 'experiment_results.csv'
if os.path.isfile(results_csv):
    results.to_csv(results_csv, mode='a', header=False, index=False)
else:
    results.to_csv(results_csv, mode='w', header=True, index=False)

print("Results appended to CSV.")
