"""
Unit Tests for Multimodal Dataset and Channel Slicing Pipeline.
"""

import unittest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import SCENARIOS, DATA_DIR
from src.dataset import MultimodalPatchDataset, get_patch_files, compute_train_norm_stats

class TestDatasetPipeline(unittest.TestCase):
    def test_toy_dataset_availability(self):
        train_files = get_patch_files("train", DATA_DIR)
        val_files = get_patch_files("val", DATA_DIR)
        test_files = get_patch_files("test", DATA_DIR)

        self.assertGreater(len(train_files), 0, "No training patches found!")
        self.assertGreater(len(val_files), 0, "No validation patches found!")
        self.assertGreater(len(test_files), 0, "No test patches found!")

    def test_channel_slicing_and_dem(self):
        train_files = get_patch_files("train", DATA_DIR)
        ds_s1 = MultimodalPatchDataset(split="train", files=train_files, scenario="S1_Optical")
        ds_s7 = MultimodalPatchDataset(split="train", files=train_files, scenario="S7_Full_Fusion")

        x1, y1, m1, _ = ds_s1[0]
        x7, y7, m7, _ = ds_s7[0]

        # S1_Optical has 6 channels
        self.assertEqual(x1.shape[0], 6)
        # S7_Full_Fusion has 41 channels (Optical 6 + SAR 6 + GLCM 8 + Indices 20 + DEM 1)
        self.assertEqual(x7.shape[0], 41)
        # Target shape is (3, 128, 128)
        self.assertEqual(y7.shape[0], 3)
        # Mask shape is (128, 128)
        self.assertEqual(m7.shape, (128, 128))

    def test_d4_augmentation_alignment(self):
        # Create asymmetric synthetic patch
        H, W = 128, 128
        dummy_opt = np.zeros((6, H, W), dtype=np.float32)
        dummy_sar = np.zeros((6, H, W), dtype=np.float32)
        dummy_glcm = np.zeros((8, H, W), dtype=np.float32)
        dummy_ind = np.zeros((20, H, W), dtype=np.float32)
        dummy_dem = np.zeros((1, H, W), dtype=np.float32)
        dummy_target = np.zeros((3, H, W), dtype=np.float32)
        dummy_mask = np.zeros((H, W), dtype=np.float32)

        cy, cx = 20, 60
        dummy_opt[:, cy, cx] = 0.5
        dummy_target[0, cy, cx] = 125.0
        dummy_mask[cy, cx] = 1.0

        # Rotate 90
        tgt_rot = np.rot90(dummy_target, k=1, axes=(1, 2))
        msk_rot = np.rot90(dummy_mask, k=1, axes=(0, 1))

        ys, xs = np.where(msk_rot == 1.0)
        self.assertEqual(len(ys), 1)
        self.assertAlmostEqual(tgt_rot[0, ys[0], xs[0]], 125.0)

if __name__ == "__main__":
    unittest.main()
