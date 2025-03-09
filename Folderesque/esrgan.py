import os
import re
import numpy as np
import time
import cv2
import torch
from tqdm import tqdm
import concurrent
from concurrent.futures import ThreadPoolExecutor
from torchvision.transforms import ToTensor
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
        self.discrete_model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=6,
            num_grow_ch=32,
            scale=scale_factor,
        )
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

        self.discrete_model.eval()
        self.discrete_model.to(device)
        self.discrete_model = torch.jit.enable_onednn_fusion(True)
        self.discrete_model = torch.jit.trace(
            self.discrete_model, torch.rand(batch_size, tile_size, tile_size)
        )
        self.discrete_model = torch.jit.freeze(self.discrete_model)
        self.integrated_model.eval()
        self.integrated_model.to(integrated_device)

    def process_batch(self, image, batch_coords):
        tiles = []
        for x, y in batch_coords:
            tile = image[y : y + self.tile_size, x : x + self.tile_size]
            tiles.append(ToTensor()(tile))

        batch_tensor = torch.cat(tiles, dim=0).to(self.device)
        with torch.no_grad(), torch.autocast(self.device.type):
            output_batch = self.discrete_model(batch_tensor)

        return [
            (
                x * self.scale_factor,
                y * self.scale_factor,
                output.squeeze().cpu().clamp(0, 1),
            )
            for (x, y), output in zip(batch_coords, output_batch)
        ]

    def parallel_upscale(self, image):
        tile_size = self.tile_size
        h, w = image.shape[:2]
        output_tensor = torch.zeros(
            (3, h * self.scale_factor, w * self.scale_factor), dtype=torch.uint8
        )
        tile_coords = [
            (x, y) for y in range(0, h, tile_size) for x in range(0, w, tile_size)
        ]

        with tqdm(
            total=len(tile_coords), desc="Processing Tiles", unit="tile", leave=False
        ) as tile_pbar:
            with ThreadPoolExecutor(max_workers=self.thread_workers) as executor:
                batch_futures = []
                pending_batches = 0
                batch_index = 0
                max_pending = self.thread_workers
                
                while batch_index < len(tile_coords) or pending_batches > 0:
                    while pending_batches < max_pending and batch_index < len(tile_coords):
                        end_idx = min(batch_index + self.batch_size, len(tile_coords))
                        batch_coords = tile_coords[batch_index:end_idx]
                        future = executor.submit(self.process_batch, image, batch_coords)
                        batch_futures.append(future)
                        batch_index = end_idx
                        pending_batches += 1
                    
                    done_futures, batch_futures = concurrent.futures.wait(
                        batch_futures, 
                        return_when=concurrent.futures.FIRST_COMPLETED
                    )
                    
                    for future in done_futures:
                        batch_results = future.result()
                        for x_out, y_out, tile_tensor in batch_results:
                            h_tile, w_tile = tile_tensor.shape[1], tile_tensor.shape[2]
                            output_tensor[
                                :, y_out : y_out + h_tile, x_out : x_out + w_tile
                            ] = tile_tensor
                        
                        pending_batches -= 1
                        tile_pbar.update(len(batch_results))
                        
                        torch.cuda.reset_max_memory_allocated()
                        allocated = torch.cuda.memory_allocated() / 1024**2
                        max_allocated = torch.cuda.max_memory_allocated() / 1024**2
                        tile_pbar.set_postfix(
                            memory=f"GPU: {allocated:.2f}MB/{max_allocated:.2f}MB",
                            temp=f"Temp: {torch.cuda.temperature:.2f}°C",
                        )

        torch.cuda.empty_cache()
        return output_tensor

    def save_image(self, tensor, output_path):
        try:
            img_np = (tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            cv2.setNumThreads(self.thread_workers * 4)
            cv2.imwrite(output_path, cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))
        except Exception as e:
            error_message = f"Exception on saving image tensors: {str(e)}"
            raise RuntimeError(error_message)

    def process_folder(self, input_dir, output_dir):
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
                            print(f"Skipping corrupted/unreadable file: {input_path}")
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
