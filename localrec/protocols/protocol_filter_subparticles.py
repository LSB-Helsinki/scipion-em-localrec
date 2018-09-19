# **************************************************************************
# *
# * Authors:   Vahid Abrishami (vahid.abrishami@helsinki.fi)
# *
# * Laboratory of Structural Biology,
# * Helsinki Institute of Life Science HiLIFE
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
# *  All comments concerning this program package may be sent to the
# *  e-mail address 'scipion@cnb.csic.es'
# *
# **************************************************************************

import numpy as np

from pyworkflow import VERSION_1_1
from pyworkflow.em import ImageHandler
from pyworkflow.protocol.params import (PointerParam, BooleanParam, StringParam,
                                        EnumParam, NumericRangeParam,
                                        PathParam, FloatParam, LEVEL_ADVANCED)
from pyworkflow.em.protocol import ProtParticles, IntParam

from localrec.utils import *

class ProtLocalizedExtraction(ProtParticles):
    """ Extract computed sub-particles from a SetOfParticles. """
    _label = 'filter_subunits'
    _lastUpdateVersion = VERSION_1_1

    # -------------------------- DEFINE param functions -----------------------
    def _defineParams(self, form):
        form.addSection(label='Input')
        form.addParam('inputCoordinates', PointerParam,
                      pointerClass='SetOfCoordinates',
                      important=True,
                      label='Input coordinates')
        form.addSection('Sub-particles')
        form.addParam('unique', FloatParam, default=-1,
                      label='Angle to keep unique sub-particles (deg)',
                      help='Keep only unique subparticles within angular '
                           'distance. It is useful to remove overlapping '
                           'sub-particles on symmetry axis.')
        form.addParam('mindist', FloatParam, default=-1,
                      label='Minimum distance between sub-particles (px)',
                      help='In pixels. Minimum distance between the '
                           'subparticles in the image. All overlapping ones '
                           'will be discarded.')
        form.addParam('side', FloatParam, default=-1,
                      label='Angle to keep sub-particles from side views (deg)',
                      help='Keep only particles within specified angular '
                           'distance from side views. All others will be '
                           'discarded. ')
        form.addParam('top', FloatParam, default=-1,
                      label='Angle to keep sub-particles from top views (deg)',
                      help='Keep only particles within specified angular '
                           'distance from top views. All others will be '
                           'discarded. ')

        form.addParallelSection(threads=0, mpi=0)

        form.addParallelSection(threads=0, mpi=0)

    # -------------------------- INSERT steps functions -----------------------
    def _insertAllSteps(self):
        partsId = self.inputParticles.get().getObjId()
        self._insertFunctionStep('createOutputStep',
                                 self._getInputParticles().getObjId(),
                                 self.inputCoordinates.get().getObjId(),
                                 self.boxSize.get())

    # -------------------------- STEPS functions ------------------------------
    def createOutputStep(self, particlesId, coordsId, boxSize):
        """ Create the input file in STAR format as expected by Relion.
        Params:
            particlesId: use this parameters just to force redo of convert if
                the input particles are changed.
        """

        inputCoords = self.inputCoordinates.get()
        outputSet = self._createSetOfCoordinates()
        outputSet.copyInfo(inputCoords)

        lastPartId = None

        for coord in inputCoords.iterItems(orderBy=['_subparticle._micId',
                                                    '_micId', 'id']):
            # The original particle id is stored in the sub-particle as micId
            partId = coord._micId.get()

            # Load the particle if it has changed from the last sub-particle
            if partId != lastPartId:
                particle = inputParticles[partId]

                if particle is None:
                    partIdExcluded.append(partId)
                    self.info("WARNING: Missing particle with id %s from "
                              "input particles set" % partId)
                else:
                    # Now load the particle image to extract later sub-particles
                    img = ih.read(particle)
                    x, y, _, _ = img.getDimensions()
                    data = img.getData()

                lastPartId = partId

            # If particle is not in inputParticles, subparticles will not be
            # generated. Now, subtract from a subset of original particles is
            # supported.
            if not partId in partIdExcluded:
                xpos = coord.getX()
                ypos = coord.getY()

                # Check that the sub-particle will not lay out of the particle
                if (ypos - b2 < 0 or ypos + b2 > y or
                        xpos - b2 < 0 or xpos + b2 > x):
                    outliers += 1
                    continue

                # Crop the sub-particle data from the whole particle image
                center[:, :] = data[ypos - b2:ypos + b2, xpos - b2:xpos + b2]
                outputImg.setData(center)
                i += 1
                outputImg.write((i, outputStack))
                subpart = coord._subparticle
                subpart.setLocation(
                    (i, outputStack))  # Change path to new stack
                subpart.setObjId(None)  # Force to insert as a new item
                outputSet.append(subpart)

        if outliers:
            self.info("WARNING: Discarded %s particles because laid out of the "
                      "particle (for a box size of %d" % (outliers, boxSize))

        self._defineOutputs(outputParticles=outputSet)
        self._defineSourceRelation(self.inputParticles, outputSet)

    # -------------------------- INFO functions -------------------------------
    def _validate(self):
        errors = []
        inputCoords = self.inputCoordinates.get()
        firstCoord = inputCoords.getFirstItem()

        if not firstCoord.hasAttribute('_subparticle'):
            errors.append('The selected input coordinates does not are the '
                          'output from a localized-subparticles protocol.')

        return errors

    def _citations(self):
        return ['Serban2015']

    def _summary(self):
        summary = []
        return summary

    def _methods(self):
        return []

    # -------------------------- UTILS functions ------------------------------
    def _getInputParticles(self):


        return self.inputParticles.get()
