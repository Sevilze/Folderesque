import unittest
import torch
import tempfile
from basicsr.archs.rrdbnet_arch import RRDBNet
from Folderesque.esrgan import ESRGAN

class TestESRGANStateDictHandling(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dummy_model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=2,
            num_grow_ch=32,
            scale=4
        )
        
        cls.base_params = {"params": cls.dummy_model.state_dict()}
        cls.base_params_ema = {"params_ema": cls.dummy_model.state_dict()}
        cls.base_raw = cls.dummy_model.state_dict()

    def _test_state_dict_error(self, modified_dict, error_type):
        with tempfile.NamedTemporaryFile() as tmpfile:
            torch.save(modified_dict, tmpfile.name)
            with self.assertRaises(RuntimeError) as cm:
                ESRGAN(
                    model_path=tmpfile.name,
                    scale_factor=4,
                    device=torch.device("cpu")
                )
            self.assertIn(error_type, str(cm.exception))

    def test_missing_keys_in_params(self):
        modified = dict(self.base_params)
        first_key = next(iter(modified["params"].keys()))
        del modified["params"][first_key]
        self._test_state_dict_error(modified, "Missing key(s)")

    def test_unexpected_keys_in_params_ema(self):
        modified = dict(self.base_params_ema)
        modified["params_ema"]["extra_layer.weight"] = torch.rand(1)
        self._test_state_dict_error(modified, "Unexpected key(s)")

    def test_both_errors_in_raw_state_dict(self):
        modified = dict(self.base_raw)
        del modified[next(iter(modified.keys()))]
        modified["extra_layer.bias"] = torch.rand(1)
        self._test_state_dict_error(modified, "both missing and unexpected")

    def test_corrupted_state_dict_structure(self):
        modified = {
            "conv1.weight": torch.rand(3, 3, 3, 3),
            "invalid_layer": torch.rand(5, 5)
        }
        self._test_state_dict_error(modified, "Missing key(s)")

    def test_unexpected_top_level_key(self):
        modified = {"model_weights": self.base_raw, "metadata": {"epoch": 100}}
        self._test_state_dict_error(modified, "Missing key(s)")

    def test_partial_missing_keys(self):
        modified = dict(self.base_params)
        params = modified["params"]
        
        del params[next(k for k in params if "conv" in k)]
        del params[next(k for k in params if "RRDB" in k)]
        
        self._test_state_dict_error(modified, "Missing key(s)")

    def test_mixed_valid_and_invalid_keys(self):
        modified = dict(self.base_params_ema)
        params_ema = modified["params_ema"]
        
        params_ema["new_block.0.conv.weight"] = torch.rand(64, 64, 3, 3)
        params_ema["classification_head.bias"] = torch.rand(10)
        
        self._test_state_dict_error(modified, "Unexpected key(s)")

    def test_valid_state_dicts(self):
        valid_cases = [
            self.base_params,
            self.base_params_ema,
            self.base_raw,
            {"params": self.base_raw, "other_data": 42}
        ]
        
        for case in valid_cases:
            with tempfile.NamedTemporaryFile() as tmpfile:
                torch.save(case, tmpfile.name)
                try:
                    ESRGAN(
                        model_path=tmpfile.name,
                        scale_factor=4,
                        device=torch.device("cpu")
                    )
                except RuntimeError as e:
                    self.fail(f"Valid state dict raised unexpected error: {e}")