# *****************************************************************************
# *
# *  This program is free software; you can redistribute it and/or modify
# *  it under the terms of the GNU General Public License as published by
# *  the Free Software Foundation; either version 2 of the License, or
# *  (at your option) any later version.
# *
# ****************************************************************************

import unittest

try:
    import numpy as np
except ImportError:
    np = None

try:
    from localrec.protocols.protocol_localized_extraction import (
        ProtLocalizedExtraction,
    )
except ImportError:
    ProtLocalizedExtraction = None


@unittest.skipIf(ProtLocalizedExtraction is None or np is None,
                 'Scipion/PWEM runtime is not available')
class TestLocalizedExtractionScaling(unittest.TestCase):
    def test_crop_center_preserves_old_scale_when_sampling_matches(self):
        xpos, ypos = ProtLocalizedExtraction._computeMicrographCropCenter(
            particleX=100, particleY=200, xOffset=5, yOffset=-6,
            coordMicSampling=1.5, particleSampling=1.5,
            outputSampling=1.5)

        self.assertEqual((xpos, ypos), (105, 194))

    def test_crop_center_scales_downsampled_parent_offsets(self):
        xpos, ypos = ProtLocalizedExtraction._computeMicrographCropCenter(
            particleX=100, particleY=200, xOffset=5, yOffset=-6,
            coordMicSampling=1.0, particleSampling=2.0,
            outputSampling=1.0)

        self.assertEqual((xpos, ypos), (110, 188))

    def test_crop_center_uses_downsampled_extraction_grid(self):
        xpos, ypos = ProtLocalizedExtraction._computeMicrographCropCenter(
            particleX=100, particleY=200, xOffset=5, yOffset=-6,
            coordMicSampling=1.0, particleSampling=2.0,
            outputSampling=2.0)

        self.assertEqual((xpos, ypos), (55, 94))

    def test_fourier_downsampling_preserves_constant_micrograph_level(self):
        data = np.ones((8, 8), dtype=np.float32) * 7

        downsampled = ProtLocalizedExtraction._downsampleMicrographData(
            data, 2.0)

        self.assertEqual(downsampled.shape, (4, 4))
        np.testing.assert_allclose(downsampled, 7, rtol=1e-6)


if __name__ == '__main__':
    unittest.main()
