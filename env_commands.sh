conda create -n embrace python=3.9 numpy=1.26.3 pandas=2.2.0 pytorch=2.3.0 torchvision cudatoolkit=11.8 -c pytorch -c conda-forge

conda activate embrace

conda install ipykernel scipy scikit-image tensorboard tqdm=4.59.0 pyarrow seaborn tensorflow-datasets tensorflow gcsfs lpips pyyaml requests -c conda-forge

pip install contourpy==1.2.0 cycler==0.12.1 filelock==3.13.1 fonttools==4.47.2 fsspec==2023.12.2 Jinja2==3.1.3 kiwisolver==1.4.5 MarkupSafe==2.1.4 matplotlib==3.8.2 mpmath==1.3.0 networkx==3.2.1 packaging==23.2 pillow==10.2.0 pyparsing==3.1.1 python-dateutil==2.8.2 pytorch-wavelets==1.3.0 pytz==2023.4 PyWavelets==1.5.0 six==1.16.0 sympy==1.12 torch-dct==0.1.6 typing_extensions==4.9.0 tzdata==2023.4 av addict einops future timm==0.6.7 yapf dahuffman==0.4.1 decord==0.6.0 pytorch_msssim==0.2.1 imageio==2.33 pytorchvideo==0.1.5

ln -s /opt/local/bin/ffmpeg /home/itec/emanuele/.conda/envs/embrace-test/bin/ffmpeg

conda install imageio-ffmpeg -c conda-forge

pip install ffmpeg-quality-metrics

pip install mmcv==2.2.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.3/index.html

conda install platformdirs

conda install termcolor