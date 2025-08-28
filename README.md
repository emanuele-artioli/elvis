# ELVIS: End-to-end Model for Video Streaming Bitrate Reduction

ELVIS (End-to-end Model for video streaming Bitrate Reduction: model-Agnostic Compression Enhancement) is a research project designed to explore and evaluate different techniques for reducing video bitrate while preserving perceptual quality. It provides a flexible framework for experimenting with various video codecs, inpainting algorithms, and other video manipulation techniques.

## Features

- **Model-Agnostic Framework:** Easily integrate and test different video compression models and algorithms.
- **Comprehensive Video Processing Pipeline:** Includes steps for scene complexity analysis, smart frame shrinking, encoding, decoding, stretching, and inpainting.
- **Multiple Codec Support:** Supports standard codecs like AVC (H.264) and advanced neural network-based codecs like HNeRV.
- **Multiple Inpainter Support:** Integrates state-of-the-art video inpainting models like ProPainter and E2FGVI.
- **Automated Experimentation:** The `run.sh` script allows for running a series of experiments with randomized parameters to explore the parameter space.
- **Detailed Metrics Collection:** Automatically calculates and aggregates various quality metrics (PSNR, SSIM, VMAF, LPIPS) to evaluate the performance of different configurations.

## Project Structure

The project is organized into several directories and scripts:

- `elvis/`: The main directory containing the orchestration and experiment scripts.
  - `orchestrator.sh`: The core script that runs a single experiment pipeline.
  - `run.sh`: A script to run multiple experiments with random parameters.
  - `scripts/`: A collection of Python scripts for various tasks like calculating bitrate, LPIPS, and collecting metrics.
  - `experiments/`: This directory is created to store the output of each experiment, including videos, frames, and metrics.
- `HNeRV/`, `ProPainter/`, `E2FGVI/`, `UFO/`, `EVCA/`: These are external projects that ELVIS depends on. They should be placed in the same parent directory as the `elvis/` repository.

## Dependencies

ELVIS relies on several external tools and projects.

### Core Tools

- **conda:** For managing Python environments.
- **ffmpeg:** For video manipulation.
- **ffmpeg-quality-metrics:** For calculating video quality metrics.

### External Projects

The following projects are expected to be cloned into the same parent directory as the `elvis` repository:

- **[HNeRV](https://github.com/hmkx/HiNeRV):** A hierarchical neural video representation model. Note that the version used in this project might be a specific fork or version, as the script names differ from the main repository.
- **[ProPainter](https://github.com/sczhou/ProPainter):** A video inpainting model.
- **[E2FGVI](https://github.com/MCG-NKU/E2FGVI):** An end-to-end flow-guided video inpainting framework.
- **UFO:** A video object segmentation tool. A public repository for this tool could not be located. You will need to obtain it from its original source.
- **EVCA:** A tool for video complexity analysis. A public repository for this tool could not be located. You will need to obtain it from its original source.

## Installation

1.  **Clone the ELVIS repository:**
    ```bash
    git clone <repository_url> elvis
    cd elvis
    ```

2.  **Install Core Tools:**
    Make sure you have `conda` and `ffmpeg` installed on your system. You can install `ffmpeg-quality-metrics` with pip:
    ```bash
    pip install ffmpeg-quality-metrics
    ```

3.  **Set up External Projects:**
    Clone the required external projects into the same parent directory as `elvis`.
    ```bash
    cd .. # Go to the parent directory of elvis/
    git clone https://github.com/hmkx/HiNeRV HNeRV
    git clone https://github.com/sczhou/ProPainter ProPainter
    git clone https://github.com/MCG-NKU/E2FGVI E2FGVI
    # Obtain and place UFO and EVCA directories here
    ```
    For each external project, follow its specific installation instructions (e.g., setting up a conda environment, installing Python dependencies from `requirements.txt`).

4.  **Prepare Video Datasets:**
    Place your video datasets in a directory. The `run.sh` script assumes a `Datasets/DAVIS` directory in the parent folder of `elvis`.

5.  **Make scripts executable:**
    ```bash
    cd elvis
    chmod +x orchestrator.sh
    chmod +x run.sh
    ```

## Usage

### Running a Single Experiment

You can run a single experiment by calling `orchestrator.sh` with a specific set of parameters. The script takes a large number of arguments to control the experiment.

Example:
```bash
./orchestrator.sh "bear" "Datasets/DAVIS/bear" "640" "360" "16" "0.5" "0.5" "0.5" "hnerv" "propainter" ... (and many more parameters)
```
Refer to the `orchestrator.sh` script for the full list of parameters and their order.

### Running Multiple Experiments

The `run.sh` script is designed to run multiple experiments with randomized parameters. You can configure the parameter lists within `run.sh` to define the search space for your experiments.

To start the experimentation process, simply run:
```bash
./run.sh
```
The script will randomly select a combination of parameters, run an experiment, and if it completes successfully, it will clean up the intermediate files and start a new experiment.

## Output

- **`experiments/` directory:** For each experiment, a directory is created with a unique name (an MD5 hash of the parameters). This directory contains:
  - `benchmark.mp4`: The original video, re-encoded.
  - `inpainted.mp4`: The final video after shrinking, encoding, decoding, stretching, and inpainting.
  - Various other intermediate files, frames, and masks.
- **`experiment_results.csv`:** A CSV file that aggregates the results of all experiments. Each row contains the parameters used for the experiment and the resulting quality metrics (PSNR, SSIM, VMAF, LPIPS), making it easy to analyze the results across different configurations.
