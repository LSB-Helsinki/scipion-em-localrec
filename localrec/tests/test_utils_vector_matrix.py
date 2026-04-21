#!/usr/bin/env python
# **************************************************************************
# *
# * Authors: LocalRec maintainers
# *
# * Unidad de  Bioinformatica of Centro Nacional de Biotecnologia , CSIC
# *
# * This program is free software; you can redistribute it and/or modify
# * it under the terms of the GNU General Public License as published by
# * the Free Software Foundation; either version 2 of the License, or
# * (at your option) any later version.
# *
# * This program is distributed in the hope that it will be useful,
# * but WITHOUT ANY WARRANTY; without even the implied warranty of
# * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# * GNU General Public License for more details.
# *
# * You should have received a copy of the GNU General Public License
# * along with this program; if not, write to the Free Software
# * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA
# * 02111-1307  USA
# *
# **************************************************************************

import unittest
import numpy as np

from localrec.utils import Vector3, load_vectors


class TestVectorMatrix(unittest.TestCase):
    def _assertVectorAlignedToMatrixZ(self, vector):
        v = Vector3()
        v.set_vector(vector)
        v.compute_unit_vector()
        v.compute_matrix()
        mat = v.get_matrix()
        # Third column is rotated Z axis.
        rotated_z = mat[:, 2]
        np.testing.assert_allclose(rotated_z, v.vector, atol=1e-7)

    def testVectorAlignmentPositiveX(self):
        self._assertVectorAlignedToMatrixZ([1.0, 0.0, 0.0])

    def testVectorAlignmentNegativeX(self):
        self._assertVectorAlignedToMatrixZ([-1.0, 0.0, 0.0])

    def testLoadVectorsNegativeXWithAndWithoutAlternativeLength(self):
        no_alt = load_vectors("", "-1,0,0", "0", 1.0)[0]
        with_alt = load_vectors("", "-1,0,0", "5", 1.0)[0]

        np.testing.assert_allclose(no_alt.vector, np.array([-1.0, 0.0, 0.0]), atol=1e-7)
        np.testing.assert_allclose(with_alt.vector, np.array([-1.0, 0.0, 0.0]), atol=1e-7)
        self.assertAlmostEqual(no_alt.get_length(), 1.0, places=7)
        self.assertAlmostEqual(with_alt.get_length(), 5.0, places=7)


if __name__ == '__main__':
    unittest.main()
