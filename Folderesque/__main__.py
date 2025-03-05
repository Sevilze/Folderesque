import argparse
import importlib.util
from .esrgan import ESRGAN
import torch


def main():
    parser = argparse.ArgumentParser(description="Upscale images using RealESRGAN.")
    parser.add_argument(
        "--conf",
        type=str,
        required=True,
        help="Provide the path to the .py configuration file.",
    )
    args = parser.parse_args()

    spec = importlib.util.spec_from_file_location("config", args.conf)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)

    device = (
        torch.device(config.DEVICE)
        if torch.cuda.is_available() and config.DEVICE == "cuda"
        else torch.device("cpu")
    )
    upscaler = ESRGAN(
        config.INPUT_PATH,
        config.OUTPUT_PATH,
        config.MODEL_PATH,
        config.SCALE_FACTOR,
        device,
        config.TILE_SIZE,
        config.THREAD_WORKERS,
        config.BATCH_SIZE,
        config.RETAIN_MODE
    )
    upscaler.process_folder(input_dir=config.INPUT_PATH, output_dir=config.OUTPUT_PATH)
    print("Upscaling complete!")


if __name__ == "__main__":
    main()
