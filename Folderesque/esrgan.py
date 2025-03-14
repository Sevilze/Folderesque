import os
import shutil
import re
import numpy as np
import signal
import time
import cv2
import torch
from PIL import Image
from tqdm import tqdm
import concurrent
from concurrent.futures import ThreadPoolExecutor
from torchvision.transforms import ToTensor, ToPILImage
import torchvision.transforms.functional as F
import sys

sys.modules["torchvision.transforms.functional_tensor"] = F
from basicsr.archs.rrdbnet_arch import RRDBNet  # noqa: E402


class ESRGAN:
    def __init__(
        self,
        input_dir,
        output_dir,
        model_path,
        scale_factor,
        device,
        tile_size,
        thread_workers,
        batch_size,
        retain_mode,
    ):
        os.makedirs(output_dir, exist_ok=True)
        self.scale_factor = scale_factor
        self.device = device
        self.tile_size = tile_size
        self.thread_workers = thread_workers
        self.batch_size = batch_size
        self.mode = retain_mode
        self.saved = set(os.listdir(output_dir)) if input_dir != output_dir else set()
        self.corrupt_dir = "corrupted_images"
        os.makedirs(self.corrupt_dir, exist_ok=True)
        self.discrete_model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=6,
            num_grow_ch=32,
            scale=scale_factor,
        )
        self.terminate = False
        signal.signal(signal.SIGINT, self.handle_termination)

        integrated_device = (
            torch.device("xpu") if torch.xpu.is_available() else torch.device("cpu")
        )
        self.integrated_model = self.discrete_model

        state_dict = torch.load(model_path, map_location=device, weights_only=True)
        if "params" in state_dict:
            self.discrete_model.load_state_dict(state_dict["params"], strict=True)
            self.integrated_model.load_state_dict(state_dict["params"], strict=True)
        elif "params_ema" in state_dict:
            self.discrete_model.load_state_dict(state_dict["params_ema"], strict=True)
            self.integrated_model.load_state_dict(state_dict["params_ema"], strict=True)
        else:
            self.discrete_model.load_state_dict(state_dict, strict=True)
            self.integrated_model.load_state_dict(state_dict, strict=True)

        self.discrete_model.to(device)
        self.discrete_model.eval()

        model_fp32 = self.discrete_model
        model_int8 = torch.quantization.quantize_dynamic(
            model_fp32, {torch.nn.Conv2d}, dtype=torch.qint8
        )
        self.discrete_model = model_int8

        self.integrated_model.eval()
        self.integrated_model.to(integrated_device)

    def handle_termination(self, signum, frame):
        print("\nTermination signal received. Cleaning up...")
        sys.stdout.flush()
        self.terminate = True
        self.shutdown_executor()
        sys.exit(130)

    def shutdown_executor(self):
        if hasattr(self, "executor") and self.executor:
            self.executor.shutdown(wait=False)
            self.executor = None

    def process_tile(self, img, x, y, tile_size):
        tile = img[y : y + tile_size, x : x + tile_size]
        to_tensor = ToTensor()
        to_pil = ToPILImage()

        tile_tensor = to_tensor(tile).unsqueeze(0).to(self.device)

        with torch.no_grad(), torch.autocast(device_type=self.device.type):
            upscaled_tile = (
                self.discrete_model(tile_tensor).squeeze(0).cpu().clamp(0, 1)
            )

        return (x * self.scale_factor, y * self.scale_factor, to_pil(upscaled_tile))

    def parallel_upscale(self, image):
        try:
            tile_size = self.tile_size
            num_workers = self.thread_workers
            batch_size = self.batch_size
            h, w = image.shape[:2]
            output_img = np.zeros(
                (h * self.scale_factor, w * self.scale_factor, 3), dtype=np.uint8
            )
            tile_coords = [
                (x, y) for y in range(0, h, tile_size) for x in range(0, w, tile_size)
            ]
            total_tiles = len(tile_coords)

            with tqdm(
                total=total_tiles, desc="Processing Tiles", unit="tile", leave=False
            ) as tile_pbar:
                with ThreadPoolExecutor(max_workers=num_workers) as executor:
                    futures = {}
                    remaining_tiles = iter(tile_coords)
                    initial_batch_size = min(batch_size * num_workers, total_tiles)

                    for _ in range(initial_batch_size):
                        x, y = next(remaining_tiles)
                        future = executor.submit(
                            self.process_tile, image, x, y, tile_size
                        )
                        futures[future] = (x, y)

                    while futures:
                        done, _ = concurrent.futures.wait(
                            futures.keys(),
                            return_when=concurrent.futures.FIRST_COMPLETED,
                        )

                        for future in done:
                            x, y = futures.pop(future)
                            x_out, y_out, upscaled_tile = future.result()

                            upscaled_tile_np = np.array(upscaled_tile)
                            output_img[
                                y_out : y_out + upscaled_tile_np.shape[0],
                                x_out : x_out + upscaled_tile_np.shape[1],
                            ] = upscaled_tile_np
                            tile_pbar.update(1)

                            try:
                                x_new, y_new = next(remaining_tiles)
                                new_future = executor.submit(
                                    self.process_tile, image, x_new, y_new, tile_size
                                )
                                futures[new_future] = (x_new, y_new)
                            except StopIteration:
                                pass

                        allocated = torch.cuda.memory_allocated() / 1024**2
                        max_allocated = torch.cuda.max_memory_allocated() / 1024**2
                        tile_pbar.set_postfix(
                            memory=f"GPU: {allocated:.2f}MB/{max_allocated:.2f}MB",
                            temp=f"Temp: {torch.cuda.temperature():.2f}°C"
                            if hasattr(torch.cuda, "temperature")
                            else "",
                        )

            torch.cuda.empty_cache()
            return Image.fromarray(output_img)

        except KeyboardInterrupt:
            self.shutdown_executor()
            raise

    def save_image(self, img, output_path):
        cv2.setNumThreads(self.thread_workers * 2)
        cv2.imwrite(output_path, cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR))

    def process_folder(self, input_dir, output_dir):
        try:
            image_files = [file for file in os.listdir(input_dir)]
            remaining_files = [
                file
                for file in image_files
                if f"ESRGAN_{os.path.splitext(file)[0]}" not in self.saved
            ]

            with tqdm(
                initial=len(self.saved),
                total=len(image_files),
                desc="Total Images Processed",
                unit="img",
            ) as main_pbar:
                for img_file in remaining_files:
                    name, extension = os.path.splitext(img_file)
                    name = re.sub(r"\.(jpg|jpeg|png)$", "", name, flags=re.IGNORECASE)
                    filename = f"ESRGAN_{name}{extension}"
                    input_path = os.path.join(input_dir, img_file)
                    output_path = os.path.join(output_dir, filename)
                    start_time = time.time()

                    with tqdm(
                        total=100,
                        desc=f"Processing {img_file[:15]}",
                        unit="%",
                        bar_format="{l_bar}{bar}| ({n_fmt}%){postfix}",
                    ) as img_pbar:
                        try:

                            def get_elapsed():
                                return time.time() - start_time

                            img_pbar.set_description(f"Loading {img_file[:15]}...")
                            img = cv2.imread(input_path)
                            if img is None:
                                print(
                                    f"Skipping corrupted/unreadable file: {input_path}"
                                )
                                corrupted_path = os.path.join(self.corrupt_dir, filename)
                                shutil.move(input_path, corrupted_path)
                                main_pbar.total -= 1
                                main_pbar.refresh()
                                continue
                            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                            img_pbar.update(33)
                            img_pbar.set_postfix_str(f"Elapsed: {get_elapsed():.3f}s")

                            img_pbar.set_description(f"Upscaling {img_file[:15]}...")
                            upscaled_img = self.parallel_upscale(img)
                            img_pbar.update(33)
                            img_pbar.set_postfix_str(f"Elapsed: {get_elapsed():.3f}s")

                            img_pbar.set_description(f"Saving {img_file[:15]}...")
                            self.save_image(upscaled_img, output_path)
                            img_pbar.update(34)
                            img_pbar.set_postfix_str(f"Elapsed: {get_elapsed():.3f}s")
                            if not self.mode:
                                os.remove(input_path)
                            img_pbar.close()

                        except Exception as e:
                            error_message = (
                                f"Exception on processing {img_file[:15]}: {str(e)}"
                            )
                            tqdm.write(error_message)
                            main_pbar.set_postfix(
                                status="Error", file=img_file[:15], error=True
                            )
                            main_pbar.close()
                            raise RuntimeError(error_message)

                    filesize = os.path.getsize(output_path) / 1024**2
                    main_pbar.update(1)
                    main_pbar.set_postfix(
                        status="Saved.",
                        resolution=f"{img.shape[:2]}→{upscaled_img.size}",
                        file=img_file[:15],
                        size=f"{filesize:.2f}MB",
                    )

        except KeyboardInterrupt:
            self.shutdown_executor()
            raise
