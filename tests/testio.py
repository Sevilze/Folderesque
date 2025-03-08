import unittest
import numpy as np
import os
import cv2
from Folderesque.esrgan import ESRGAN


class TestIO(unittest.TestCase):
    def test_large_image_processing(self):
        self.create_test_image("large_image.jpg", size=(2048, 2048))
        esrgan = ESRGAN(
            input_dir=str(self.input_dir),
            output_dir=str(self.output_dir),
            **self.esrgan_config,
        )
        esrgan.process_folder(str(self.input_dir), str(self.output_dir))

        output_path = self.output_dir / "ESRGAN_large_image.jpg"
        self.assertTrue(output_path.exists())

        img = cv2.imread(str(output_path))
        self.assertEqual(img.shape[:2], (2048 * 4, 2048 * 4))

    def test_non_image_files(self):
        (self.input_dir / "text_file.txt").write_text("Not an image")
        (self.input_dir / "fake_image.jpg").write_bytes(os.urandom(1024))

        esrgan = ESRGAN(
            input_dir=str(self.input_dir),
            output_dir=str(self.output_dir),
            **self.esrgan_config,
        )
        esrgan.process_folder(str(self.input_dir), str(self.output_dir))

        corrupted_dir = self.output_dir / "corrupted"
        self.assertTrue(corrupted_dir.exists())
        self.assertIn("text_file.txt", os.listdir(corrupted_dir))
        self.assertIn("fake_image.jpg", os.listdir(corrupted_dir))

    def test_unsupported_color_space(self):
        gray_img = np.random.randint(0, 255, (64, 64), dtype=np.uint8)
        cv2.imwrite(str(self.input_dir / "grayscale.jpg"), gray_img)

        esrgan = ESRGAN(
            input_dir=str(self.input_dir),
            output_dir=str(self.output_dir),
            **self.esrgan_config,
        )
        with self.assertLogs(level="WARNING") as cm:
            esrgan.process_folder(str(self.input_dir), str(self.output_dir))
            self.assertIn("Unsupported color space", cm.output[0])

    def test_output_image_quality(self):
        test_img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        cv2.imwrite(str(self.input_dir / "quality_test.jpg"), test_img)

        esrgan = ESRGAN(
            input_dir=str(self.input_dir),
            output_dir=str(self.output_dir),
            **self.esrgan_config,
        )
        esrgan.process_folder(str(self.input_dir), str(self.output_dir))

        output_img = cv2.imread(str(self.output_dir / "ESRGAN_quality_test.jpg"))

        self.assertNotEqual(np.sum(test_img), np.sum(output_img))
        self.assertEqual(output_img.shape, (256, 256, 3))
