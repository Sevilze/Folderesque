A Python script for running and inferencing image upscaling models on image folders using optimization techniques like batched tile processing and automatic mixed precision (AMP).

## Prerequisites
---
- Python 3.7+
- NVIDIA GPU (recommended) with CUDA support
- NVIDIA cuDNN Driver (You need to adjust your download version so that it is compatible with your CUDA version)
- PyTorch with CUDA

## Installation
---
1. Clone the repository:
   ```bash
   git clone https://github.com/Sevilze/Folderesque.git
   cd Folderesque
   ```

2. Install required dependencies:
   ```bash
   pip install numpy opencv-python torch torchvision tqdm Pillow basicsr
   ```

3. Download the pre-trained model:
   - Place your pretrained model in the project directory. The example for this script uses the `RealESRGAN_x4plus_anime_6B.pth` model.
   - Download from [Real-ESRGAN repository](https://github.com/xinntao/Real-ESRGAN)

## Notes
---
- Input images should be placed in the specified input folder directory.
- Output images are saved with "ESRGAN_" prefix in filenames.
- Recommended tile sizes:
  - 400-600 for 8GB GPUs
  - 800-1000 for 16GB+ GPUs
- You can reduce the batch size if you're encountering memory issues.

## ❤ Credits
---
Immense thanks to:
- Real-ESRGAN authors: [Xintao Wang](https://github.com/xinntao)
- BasicSR framework: [BasicSR](https://github.com/xinntao/BasicSR)

## Troubleshooting
---
**Common Issues:**
1. **CUDA Out of Memory**:
   - Reduce `tile_size`.
   - Decrease `batch_size`.

2. **Model File Not Found**:
   - Ensure `RealESRGAN_x4plus_anime_6B.pth` is in the correct path.
   - Download from official sources if it's missing.

## Disclaimer
---
This implementation is specifically optimized for anime-style images. Results may vary depending on input quality and image content. Always keep backups of original files before processing.