# import packages
import os
import pandas as pd

# get parameters from orchestrator
video_name = os.environ.get('video_name')
scene_number = os.environ.get('scene_number')
resolution = os.environ.get('resolution')
square_size = os.environ.get('square_size')
percentage_to_remove = float(os.environ.get('percentage_to_remove'))
alpha = float(os.environ.get('alpha'))
neighbor_length = os.environ.get('neighbor_length')
ref_stride = int(os.environ.get('ref_stride'))
subvideo_length = int(os.environ.get('subvideo_length'))
bitrate = os.environ.get('bitrate')
server_start_time = int(os.environ.get('server_start_time'))
client_start_time = int(os.environ.get('client_start_time'))
end_time = int(os.environ.get('end_time'))

# calculate elapsed times
server_runtime = client_start_time - server_start_time
client_runtime = end_time - client_start_time

# read experiment metrics
experiment_folder = f'videos/{video_name}/scene_{scene_number}/{resolution}/squ_{square_size}_rem_{percentage_to_remove}_alp_{alpha}'
metrics_inpainted_file = f'{experiment_folder}/nei_{neighbor_length}_ref_{ref_stride}_sub_{subvideo_length}.csv'
metrics_inpainted_df = pd.read_csv(metrics_inpainted_file)
metrics_original_file = f'{experiment_folder}/original.csv'
metrics_original_df = pd.read_csv(metrics_original_file)

# calculate aggregates
inpainted_means = metrics_inpainted_df.mean(axis=0, numeric_only=True)
inpainted_stds = metrics_inpainted_df.std(axis=0, numeric_only=True)
original_means = metrics_original_df.mean(axis=0, numeric_only=True)
original_stds = metrics_original_df.std(axis=0, numeric_only=True)

# write a row of results.csv NOTE: some elements were passed as lists to avoid a pandas error
results = pd.DataFrame.from_dict({
    'video_name': [video_name],
    'scene_number': [scene_number],
    'resolution': [resolution],
    'square_size': [square_size],
    'percentage_to_remove': [percentage_to_remove],
    'alpha': [alpha],
    'neighbor_length': [neighbor_length],
    'ref_stride': [ref_stride],
    'subvideo_length': [subvideo_length],
    'bitrate': [bitrate],
    'server_runtime': [server_runtime],
    'client_runtime': [client_runtime],

    'mse_ori_mean': original_means['mse_avg'],
    # 'mse_ori_std': original_stds['mse_avg'],
    'psnr_ori_mean': original_means['psnr_avg'],
    # 'psnr_ori_std': original_stds['psnr_avg'],
    'ssim_ori_mean': original_means['ssim_avg'],
    # 'ssim_ori_std': original_stds['ssim_avg'],
    'vmaf_ori_mean': original_means['vmaf'],
    # 'vmaf_ori_std': original_stds['vmaf'],

    'mse_inp_mean': inpainted_means['mse_avg'],
    # 'mse_inp_std': inpainted_stds['mse_avg'],
    'psnr_inp_mean': inpainted_means['psnr_avg'],
    # 'psnr_inp_std': inpainted_stds['psnr_avg'],
    'ssim_inp_mean': inpainted_means['ssim_avg'],
    # 'ssim_inp_std': inpainted_stds['ssim_avg'],
    'vmaf_inp_mean': inpainted_means['vmaf'],
    # 'vmaf_inp_std': inpainted_stds['vmaf'],

})

# append row to results.csv, create file is it does not exist
results_path='results.csv'
results.to_csv(results_path, mode='a', index=False, header=not os.path.exists(results_path))