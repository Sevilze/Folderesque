Python script for running and inferencing image upscaling models on image folders using optimization techniques like batched tile processing and automatic mixed precision (AMP).

# Prerequisites

---

- Python 3.7+
- NVIDIA GPU that supports CUDA version 12.4 or higher (This Script uses PyTorch with CUDA 12.4).

# Installation

---

1. Create a virtual environment using Python's module or using Anaconda Prompt and activate the environment.

   ```bash
   python -m venv [env_name]
   .\[env_name]\Scripts\activate
   ```

   or

   ```bash
   conda create --name [env_name]
   conda activate [env_name]
   ```

2. Update PyPI if necessary:

   ```bash
   pip install --upgrade pip
   ```

3. Install the package:

   ```bash
   pip install Folderesque --extra-index-url https://download.pytorch.org/whl/cu124
   ```

---

# Usage

1. Create a new project or workspace.
2. Copy the folder of images that you want to upscale to the workspace.
3. Download the pre-trained model:
   - Place your pretrained model in the project directory. The example for this script uses the `RealESRGAN_x4plus_anime_6B.pth` model.
   - Download from [Real-ESRGAN repository](https://github.com/xinntao/Real-ESRGAN).
4. Create a config.py file with contents matching the one shown below:

   ```Python
   INPUT_PATH = "daskruns"
   OUTPUT_PATH = "testscaling"
   MODEL_PATH = "models\RealESRGAN_x4plus_anime_6B.pth"
   SCALE_FACTOR = 4
   DEVICE = "cuda"
   TILE_SIZE = 400
   THREAD_WORKERS = 4
   BATCH_SIZE = 16
   ```

5. Ensure that INPUT_PATH and MODEL_PATH points to the correct path of your input folder and the model path respectively.
6. To run the script, copy the path of the config file and run the command:

   ```bash
   Folderesque --conf [conf_path]
   ```

Output images are saved with "ESRGAN_" prefix in the filenames.

# ❤ Credits

---
Immense thanks to:

- Real-ESRGAN authors: [Xintao Wang](https://github.com/xinntao)
- BasicSR framework: [BasicSR](https://github.com/xinntao/BasicSR)

# Troubleshooting

---
**Common Issues:**

1. **CUDA Out of Memory**:
   - Reduce `TILE_SIZE`.
   - Decrease `THREAD_WORKERS`.
   - Decrease `BATCH_SIZE`.

2. **Command not recognized**:
   - You may want to add the folder path of your environment to the system's path.
   - Alternatively, you can use ```python -m``` prefix when running commands.

# Disclaimer

The `SCALE_FACTOR` parameter should match the model's intended scale (for now).
