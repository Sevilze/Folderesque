import unittest
import numpy as np
import os
import cv2
from Folderesque.esrgan import ESRGAN

class TestIO(unittest.TestCase):
    def test_corrupted_image_handling(tmp_path):
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        output_dir = tmp_path / "output"
        
        valid_img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        cv2.imwrite(str(input_dir / "valid.jpg"), valid_img)
        
        (input_dir / "fake_image.jpg").write_text("invalid image data")
        cv2.imwrite(str(input_dir / "4channel.png"), np.random.rand(64,64,4)*255)

        esrgan = ESRGAN(
            input_dir=str(input_dir),
            output_dir=str(output_dir),
            model_path="tests/test_data/models/test_model.pth",
            scale_factor=4,
            device="cpu",
            tile_size=32,
            thread_workers=2,
            batch_size=4,
            retain_mode=True
        )

        esrgan.process_folder(str(input_dir), str(output_dir))
        corrupted_dir = output_dir / "corrupted"
        assert (output_dir / "ESRGAN_valid.jpg").exists()
        
        assert len(os.listdir(corrupted_dir)) == 2
        assert "fake_image.jpg" in os.listdir(corrupted_dir)
        assert "4channel.png" in os.listdir(corrupted_dir)
        assert len(os.listdir(input_dir)) == 0