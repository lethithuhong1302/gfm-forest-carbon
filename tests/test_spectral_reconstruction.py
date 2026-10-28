"""
Unit Tests for Algebraic Spectral Inversion.
"""

import unittest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.spectral_reconstruction import reconstruct_hls_numpy

class TestSpectralReconstruction(unittest.TestCase):
    def test_algebraic_inversion_equations(self):
        # 6-channel optical: [LS_R, LS_G, LS_B, S2_R, S2_G, S2_B]
        optical = np.zeros((6, 4, 4), dtype=np.float32)
        optical[3] = 0.10 # S2 Red
        optical[4] = 0.15 # S2 Green
        optical[5] = 0.05 # S2 Blue

        # 20-channel indices: NDVI_S2 at 15, MNDWI_S2 at 9, NBR_S2 at 11
        indices = np.zeros((20, 4, 4), dtype=np.float32)
        indices[15] = 0.666667 # NDVI
        indices[9] = 0.200000  # MNDWI
        indices[11] = 0.818182 # NBR

        hls = reconstruct_hls_numpy(optical, indices, source="S2")
        self.assertEqual(hls.shape, (6, 4, 4))
        # Blue, Green, Red
        self.assertAlmostEqual(float(hls[0, 0, 0]), 0.05, places=4)
        self.assertAlmostEqual(float(hls[1, 0, 0]), 0.15, places=4)
        self.assertAlmostEqual(float(hls[2, 0, 0]), 0.10, places=4)
        # NIR recon = Red * (1 + NDVI) / (1 - NDVI) = 0.10 * 1.666667 / 0.333333 = 0.50
        self.assertAlmostEqual(float(hls[3, 0, 0]), 0.50, places=3)
        # SWIR1 recon = Green * (1 - MNDWI) / (1 + MNDWI) = 0.15 * 0.8 / 1.2 = 0.10
        self.assertAlmostEqual(float(hls[4, 0, 0]), 0.10, places=3)
        # SWIR2 recon = NIR * (1 - NBR) / (1 + NBR) = 0.50 * 0.181818 / 1.818182 = 0.05
        self.assertAlmostEqual(float(hls[5, 0, 0]), 0.05, places=3)

if __name__ == "__main__":
    unittest.main()
