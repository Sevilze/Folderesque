from esrgan import AnimeESRGAN
from config import INPUT_PATH, OUTPUT_PATH, MODEL_PATH, DEVICE


def main():
    input_dir = INPUT_PATH
    output_dir = OUTPUT_PATH
    upscaler = AnimeESRGAN(output_dir, MODEL_PATH, DEVICE)
    upscaler.process_folder(input_dir=input_dir, output_dir=output_dir)
    print("Upscaling complete!")


if __name__ == "__main__":
    main()
