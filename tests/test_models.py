"""Optional real-model smoke tests; skipped when weights are absent."""
import unittest
import numpy as np
from src.config import DETECTOR, EMBEDDER
from src.embed import ArcFace
from src.landmarks import FaceDetector


@unittest.skipUnless(DETECTOR.is_file() and DETECTOR.stat().st_size > 1024,
                     "Download YuNet for inference smoke test")
class DetectorSmokeTest(unittest.TestCase):
    def test_blank_frame_has_no_faces(self):
        detector = FaceDetector()
        self.assertEqual(detector.detect(np.zeros((480, 640, 3), np.uint8)), [])


@unittest.skipUnless(EMBEDDER.is_file() and EMBEDDER.stat().st_size > 1024,
                     "Download ArcFace for inference smoke test")
class EmbeddingSmokeTest(unittest.TestCase):
    def test_real_model_returns_repeatable_unit_vector(self):
        model = ArcFace()
        crop = np.random.default_rng(7).integers(0, 256, (112, 112, 3), dtype=np.uint8)
        first, second = model.embed(crop), model.embed(crop)
        self.assertEqual(first.shape, (512,))
        self.assertAlmostEqual(float(np.linalg.norm(first)), 1.0, places=5)
        np.testing.assert_allclose(first, second, atol=1e-6)
