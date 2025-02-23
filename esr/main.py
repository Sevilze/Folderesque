from esrgan import AnimeESRGAN

def main():
    input_dir='daskruns'
    output_dir='testscaling'
    upscaler = AnimeESRGAN(output_dir)
    upscaler.process_folder(
        input_dir=input_dir,
        output_dir=output_dir
    )
    print("Upscaling complete!")

if __name__ == '__main__':
    main()