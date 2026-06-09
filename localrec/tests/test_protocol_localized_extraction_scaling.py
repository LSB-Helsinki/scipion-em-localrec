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

    def test_output_set_dimensions_uses_set_dim(self):
        class DummySet:
            def __init__(self):
                self.dim = None

            def setDim(self, dim):
                self.dim = dim

        outputSet = DummySet()

        ProtLocalizedExtraction._setOutputSetDimensions(outputSet, 26)

        self.assertEqual(outputSet.dim, (26, 26, 1))

    def test_output_set_dimensions_falls_back_to_dimensions_attribute(self):
        class DummyDimensions:
            def __init__(self):
                self.value = None

            def set(self, value):
                self.value = value

        class DummySet:
            def __init__(self):
                self._dimensions = DummyDimensions()

        outputSet = DummySet()

        ProtLocalizedExtraction._setOutputSetDimensions(outputSet, 26)

        self.assertEqual(outputSet._dimensions.value, '26 26 1')

    def test_validate_written_output_image_accepts_matching_dimensions(self):
        class DummyImage:
            def getDimensions(self):
                return 26, 26, 1, 1

        class DummyImageHandler:
            def __init__(self):
                self.location = None

            def read(self, location):
                self.location = location
                return DummyImage()

        ih = DummyImageHandler()

        ProtLocalizedExtraction._validateWrittenOutputImage(
            ih, 'particles.mrcs', 1, 26)

        self.assertEqual(ih.location, (1, 'particles.mrcs'))

    def test_validate_written_output_image_rejects_bad_dimensions(self):
        class DummyImage:
            def getDimensions(self):
                return 24, 24, 1, 1

        class DummyImageHandler:
            def read(self, location):
                return DummyImage()

        with self.assertRaisesRegex(RuntimeError, 'expected 26x26'):
            ProtLocalizedExtraction._validateWrittenOutputImage(
                DummyImageHandler(), 'particles.mrcs', 1, 26)

    def test_clone_micrograph_coordinate_removes_nested_subparticle(self):
        class DummyValue:
            def __init__(self, value):
                self.value = value

            def clone(self):
                return DummyValue(self.value)

            def get(self):
                return self.value

        class DummyCoordinate:
            def __init__(self):
                self.x = None
                self.y = None
                self.micId = None
                self.micName = None
                self._micId = DummyValue(7)
                self._subparticle = object()

            def clone(self):
                clone = DummyCoordinate()
                clone._micId = self._micId
                clone._subparticle = self._subparticle
                return clone

            def setX(self, value):
                self.x = value

            def setY(self, value):
                self.y = value

            def setMicId(self, value):
                self.micId = value

            def setMicName(self, value):
                self.micName = value

        class DummyParticleCoordinate:
            def getMicId(self):
                return 3

            def getMicName(self):
                return 'mic-a'

        outputCoord = ProtLocalizedExtraction._cloneMicrographCoordinate(
            DummyCoordinate(), DummyParticleCoordinate(), 11, 13, 7)

        self.assertFalse(hasattr(outputCoord, '_subparticle'))
        self.assertEqual(outputCoord.x, 11)
        self.assertEqual(outputCoord.y, 13)
        self.assertEqual(outputCoord.micId, 3)
        self.assertEqual(outputCoord.micName, 'mic-a')
        self.assertEqual(outputCoord._parentParticleId.get(), 7)

    def test_sanitize_output_coordinate_returns_coordinate(self):
        class DummyCoordinate:
            pass

        coord = DummyCoordinate()
        coord._subparticle = object()

        sanitized = ProtLocalizedExtraction._sanitizeOutputCoordinate(coord)

        self.assertIs(sanitized, coord)
        self.assertFalse(hasattr(coord, '_subparticle'))


if __name__ == '__main__':
    unittest.main()
