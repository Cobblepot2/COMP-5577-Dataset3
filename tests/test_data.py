from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
from PIL import Image
import torch

from dataset3.common import ImageRecords


class ImageLoaderTests(unittest.TestCase):
    def test_rgb_layout_and_pixel_scaling(self):
        with TemporaryDirectory() as directory:
            pixels = np.zeros((64, 64, 3), dtype=np.uint8)
            pixels[:, :, 0] = 255
            pixels[:, :, 1] = 128
            Image.fromarray(pixels).save(Path(directory) / "7.png")
            data = ImageRecords([{"id": "7", "label": "3"}], directory, labeled=True)
            x, y = data[0]
            self.assertEqual(x.shape, (3, 64, 64))
            self.assertEqual(x.dtype, torch.float32)
            self.assertEqual(y, 3)
            self.assertEqual(data.ids, ["7"])
            torch.testing.assert_close(x[:, 0, 0], torch.tensor([1., 128/255, 0.]))


if __name__ == "__main__":
    unittest.main()
