# ELVIS - Adaptive Video Compression Framework

This repository implements ELVIS (Efficient Low-complexity Video Intelligent Streaming), a framework for adaptive video compression using content-aware methods.

## Setup Instructions

### One-time Setup for Restoration Models

Before running ELVIS v2 methods, you need to set up the restoration models:

#### LaplacianVCAR (for DCT dampening restoration)
```bash
cd LaplacianVCAR
cd ops/dcn/
bash build.sh
python simple_check.py
```

#### SwinTormer (for Gaussian blur restoration)
```bash
cd swintormer
python basicsr/train.py -opt options/train/swintormer/train_swintormer.yml
```

Note: The training step for SwinTormer may take several hours. This only needs to be done once.

#### SinSR (for downsampling restoration)
SinSR is ready to use without additional setup. The model will automatically use the appropriate pre-trained weights.

## Usage

Run the main script:
```bash
python elvis.py
```

This will process a test video through all ELVIS methods and generate quality analysis results.

## Methods Implemented

- **Baseline**: Standard H.265 encoding
- **Adaptive ROI**: ROI-based quantization using removability scores
- **ELVIS v1**: Block removal with inpainting restoration
- **ELVIS v2 DCT**: DCT coefficient dampening with LaplacianVCAR restoration
- **ELVIS v2 Downsample**: Adaptive downsampling with SinSR restoration
- **ELVIS v2 Blur**: Gaussian blur with SwinTormer restoration

## TODOs
1. We need to ensure every video we compare is at the same bitrate. Maybe with two pass encoding? Maybe we encode them all the end? Maybe we have a function that based on the baseline bitrate, re-encodes the others multiple times until they reach the correct bitrate? Maybe we find the right tweak of parameters that gives us the right bitrate for each method every time? Or maybe we don't need to, and we just run the methods at a bunch of different settings, get those bitrates, and interpolate the quality curves. But then we need a general parameter, like bitrate, that even if it does not make them all the same, makes them vary in quality in the same way.
2. Right now, SinSR takes the whole image and does a 4x upscaling. Instead, we have applied multiple levels of downscaling on the server side, so on the server side we need to save the downsampling strenghts the same way we do with shrink masks, to understand how much each block was downsampled (but here, we don't have a single bit per block, we have two, representing the 4 possible levels: 1x, 2x, 4x, 8x). On the client side, we decompress this map, and can process the frame, in this way: First, we downsample the whole frame to the smallest level. We upsample 2x, then introduce the 2x original blocks. Then, we upsample another 2x, adn introduce the 4x blocks, and so on until we have them all.

