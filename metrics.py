# import packages
import os
import pandas as pd

# get parameters from orchestrator
video_name = os.environ.get('video_name')
scene_number = os.environ.get('scene_number')
resolution = os.environ.get('resolution')
frame_rate = int(os.environ.get('frame_rate'))
square_size = os.environ.get('square_size')
horizontal_stride = int(os.environ.get('horizontal_stride'))
vertical_stride = int(os.environ.get('vertical_stride'))
neighbor_length = os.environ.get('neighbor_length')
ref_stride = int(os.environ.get('ref_stride'))
subvideo_length = int(os.environ.get('subvideo_length'))

# read experiment metrics
experiment_folder = f'videos/{video_name}/scene_{scene_number}/{resolution}/squ_{square_size}_hor_{horizontal_stride}_ver_{vertical_stride}'
metrics_inpainted_file = f'{experiment_folder}/nei_{neighbor_length}_ref_{ref_stride}_sub_{subvideo_length}_inpainted.csv'
metrics_inpainted_df = pd.read_csv(metrics_inpainted_file)
metrics_original_file = f'{experiment_folder}/nei_{neighbor_length}_ref_{ref_stride}_sub_{subvideo_length}_original.csv'
metrics_original_df = pd.read_csv(metrics_original_file)

# calculate aggregates
inpainted_means = metrics_inpainted_df.mean(axis=0)
inpainted_stds = metrics_inpainted_df.std(axis=0)
original_means = metrics_original_df.mean(axis=0)
original_stds = metrics_original_df.std(axis=0)

# get file sizes
shrunk_file = f'{experiment_folder}/shrunk.mp4'
original_file = f'{experiment_folder}/original.mp4'
shrunk_size = os.path.getsize(shrunk_file)
original_size = os.path.getsize(original_file)

# write a row of results.csv NOTE: some elements were passed as lists to avoid a pandas error
results = pd.DataFrame.from_dict({
    'video_name': [video_name],
    'scene_number': [scene_number],
    'resolution': [resolution],
    'frame_rate': [frame_rate],
    'square_size': [square_size],
    'horizontal_stride': [horizontal_stride],
    'vertical_stride': [vertical_stride],
    'neighbor_length': [neighbor_length],
    'ref_stride': [ref_stride],
    'subvideo_length': [subvideo_length],

    'adm2_ori_mean': original_means['adm2'],
    'adm2_ori_std': original_stds['adm2'],
    'motion2_ori_mean': original_means['motion2'],
    'motion2_ori_std': original_stds['motion2'],
    'ms_ssim_ori_mean': original_means['ms_ssim'],
    'ms_ssim_ori_std': original_stds['ms_ssim'],
    'psnr_ori_mean': original_means['psnr'],
    'psnr_ori_std': original_stds['psnr'],
    'ssim_ori_mean': original_means['ssim'],
    'ssim_ori_std': original_stds['ssim'],
    'vif_scale0_ori_mean': original_means['vif_scale0'],
    'vif_scale0_ori_std': original_stds['vif_scale0'],
    'vif_scale1_ori_mean': original_means['vif_scale1'],
    'vif_scale1_ori_std': original_stds['vif_scale1'],
    'vif_scale2_ori_mean': original_means['vif_scale2'],
    'vif_scale2_ori_std': original_stds['vif_scale2'],
    'vif_scale3_ori_mean': original_means['vif_scale3'],
    'vif_scale3_ori_std': original_stds['vif_scale3'],
    'vmaf_ori_mean': original_means['vmaf'],
    'vmaf_ori_std': original_stds['vmaf'],

    'adm2_inp_mean': inpainted_means['adm2'],
    'adm2_inp_std': inpainted_stds['adm2'],
    'motion2_inp_mean': inpainted_means['motion2'],
    'motion2_inp_std': inpainted_stds['motion2'],
    'ms_ssim_inp_mean': inpainted_means['ms_ssim'],
    'ms_ssim_inp_std': inpainted_stds['ms_ssim'],
    'psnr_inp_mean': inpainted_means['psnr'],
    'psnr_inp_std': inpainted_stds['psnr'],
    'ssim_inp_mean': inpainted_means['ssim'],
    'ssim_inp_std': inpainted_stds['ssim'],
    'vif_scale0_inp_mean': inpainted_means['vif_scale0'],
    'vif_scale0_inp_std': inpainted_stds['vif_scale0'],
    'vif_scale1_inp_mean': inpainted_means['vif_scale1'],
    'vif_scale1_inp_std': inpainted_stds['vif_scale1'],
    'vif_scale2_inp_mean': inpainted_means['vif_scale2'],
    'vif_scale2_inp_std': inpainted_stds['vif_scale2'],
    'vif_scale3_inp_mean': inpainted_means['vif_scale3'],
    'vif_scale3_inp_std': inpainted_stds['vif_scale3'],
    'vmaf_inp_mean': inpainted_means['vmaf'],
    'vmaf_inp_std': inpainted_stds['vmaf'],

    'shrunk_size': [shrunk_size],
    'original_size': [original_size],
    'bitrate_saving': [1-(shrunk_size/original_size)], # TODO: figure out why the last column doesn't print
})

# append row to results.csv, create file is it does not exist
results_path='results.csv'
results.to_csv(results_path, mode='a', header=not os.path.exists(results_path))