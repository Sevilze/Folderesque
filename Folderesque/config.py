import torch

INPUT_PATH = 'daskruns'
OUTPUT_PATH = 'testscaling'
MODEL_PATH = 'models\RealESRGAN_x4plus_anime_6B.pth'
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu')
