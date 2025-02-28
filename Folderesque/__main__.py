import argparse
import importlib.util
from .esrgan import AnimeESRGAN
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

    input_dir = config.INPUT_PATH
    output_dir = config.OUTPUT_PATH
    model_path = config.MODEL_PATH
    scale_factor = config.SCALE_FACTOR
    device = (
        torch.device(config.DEVICE)
        if torch.cuda.is_available() and config.DEVICE == "cuda"
        else torch.device("cpu")
    )
    upscaler = AnimeESRGAN(output_dir, model_path, scale_factor, device)
    upscaler.process_folder(input_dir=input_dir, output_dir=output_dir)
    print("Upscaling complete!")


if __name__ == "__main__":
    main()
