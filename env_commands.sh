# works on GPU3 server, which has an NVIDIA Quadro RTX 8000 GPU and NVIDIA RTX A6000 GPU.
# does not work on GPU6 server, which as 2x NVIDIA RTX 6000 Ada GPU. So it needs somewhat older GPUs.

conda clean –all
pip cache purge
conda create -n elvis python=3.8
conda activate elvis
conda clean –all
pip cache purge
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge
pip install mmcv-full==1.6.2 -f https://download.openmmlab.com/mmcv/dist/cu116/torch1.12/index.html –no-cache-dir
conda install pandas tqdm regex matplotlib
conda install imageio tensorboard libstdcxx-ng timm
pip install decord pytorch_msssim dahuffman pytorchvideo lpips