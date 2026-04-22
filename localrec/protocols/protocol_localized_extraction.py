# **************************************************************************
# *
# * Authors:     J.M. De la Rosa Trevin (delarosatrevin@scilifelab.se) [1]
# *
# * [1] SciLifeLab, Stockholm University
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

from pyworkflow import VERSION_1_2
from pwem.emlib.image import ImageHandler
from pyworkflow.protocol.params import PointerParam, BooleanParam, LEVEL_ADVANCED
from pwem.protocols import ProtParticles
from pyworkflow.protocol.params import IntParam

# eventually progressbar will be move to scipion core
from pyworkflow.utils import ProgressBar
from pwem.objects import SetOfParticles
from localrec.utils import has_subparticle_provenance


class ProtLocalizedExtraction(ProtParticles):
    """ Extract computed sub-particles from a SetOfParticles. """
    _label = 'extract subparticles'
    _lastUpdateVersion = VERSION_1_2
    OUTPUTPARTICLESNAME = "outputParticles"
    _possibleOutputs = {OUTPUTPARTICLESNAME: SetOfParticles}

    # -------------------------- DEFINE param functions -----------------------
    def _defineParams(self, form):
        form.addSection(label='Input')
        form.addParam('inputParticles', PointerParam,
                      pointerClass='SetOfParticles',
                      pointerCondition='hasAlignmentProj',
                      important=True,
                      label="Input particles",
                      help='Select the input images from the project.')

        form.addParam('inputCoordinates', PointerParam,
                      pointerClass='SetOfCoordinates',
                      important=True,
                      label='Input coordinates')

        form.addParam('extractFromMicrographs', BooleanParam, default=False,
                      label='Extract from micrographs?',
                      help='If set to Yes, subparticles will be extracted from '
                           'the original micrographs using particle '
                           'coordinates and origin shifts.')

        form.addParam('inputMicrographs', PointerParam,
                      pointerClass='SetOfMicrographs',
                      condition='extractFromMicrographs',
                      label='Micrographs',
                      help='Set of micrographs used to extract the input '
                           'particles. This must match the original '
                           'micrographs.')

        form.addParam('boxSize', IntParam,
                      label='Subparticle box size (px)',
                      help='Select the amount of pixels to extract the '
                           'sub-particles.')
        form.addParam('extractAll', BooleanParam, default=False,
                      expertLevel=LEVEL_ADVANCED,
                      label='Extract all',
                      help='Extract all sub-particles, even those that extend '
                           'outside of the particle box')

        form.addParallelSection(threads=0, mpi=0)

    # -------------------------- INSERT steps functions -----------------------
    def _insertAllSteps(self):

        self._insertFunctionStep('createOutputStep')

    # -------------------------- STEPS functions ------------------------------
    def createOutputStep(self):

        ih = ImageHandler()
        outputStack = self._getPath('particles.mrcs')
        outputImg = ih.createImage()

        inputParticles = self.inputParticles.get()
        inputCoords = self.inputCoordinates.get()
        inputMicrographs = (self.inputMicrographs.get()
                            if self.extractFromMicrographs.get()
                            else None)
        outputSet = self._createSetOfParticles()
        outputSet.copyInfo(inputParticles)

        boxSize = self.boxSize.get()
        b2 = int(round(boxSize / 2))
        halfParticleDim = int(round(inputParticles.getXDim() / 2))
        center = np.zeros((boxSize, boxSize))

        ih = ImageHandler()

        i = 0
        discardedOutliers = 0
        paddedOutliers = 0
        partIdExcluded = []
        lastPartId = None
        lastMicId = None
        missingMicIds = set()
        missingProvenanceWarned = False
        data = None
        currentParticle = None
        currentParticleCoord = None

        progress = ProgressBar(len(inputCoords), fmt=ProgressBar.NOBAR)
        progress.start()
        step = max(100, len(inputCoords) // 100)
        for i, coord in enumerate(inputCoords.iterItems(orderBy=['_subparticle._micId',
                                                    '_micId', 'id'])):
            if i % step == 0:
                progress.update(i+1)

            # NOTE: In localrec subparticle coordinates, _micId stores the
            # parent particle ObjId (not a true micrograph id). Keep this
            # convention for backwards compatibility.
            partId = self._getParentParticleId(coord)
            if partId is None:
                discardedOutliers += 1
                self.info("WARNING: Missing parent particle id (_micId) in "
                          "subparticle coordinate id %s" % coord.getObjId())
                continue

            # Load the particle if it has changed from the last sub-particle
            if partId != lastPartId:
                particle = inputParticles[partId]
                currentParticle = particle
                currentParticleCoord = None

                if particle is None:
                    partIdExcluded.append(partId)
                    self.info("WARNING: Missing particle with id %s from "
                              "input particles set" % partId)
                else:
                    if self.extractFromMicrographs.get():
                        particleCoord = particle.getCoordinate()
                        currentParticleCoord = particleCoord
                        micId = particleCoord.getMicId()
                        if micId in missingMicIds:
                            data = None
                            lastMicId = None
                        elif micId != lastMicId:
                            mic = inputMicrographs[micId]
                            if mic is None:
                                self.info("WARNING: Missing micrograph with "
                                          "id %s from input micrographs set"
                                          % micId)
                                lastMicId = None
                                data = None
                                missingMicIds.add(micId)
                            else:
                                img = ih.read(mic)
                                x, y, _, _ = img.getDimensions()
                                data = img.getData()
                                lastMicId = micId
                    else:
                        # Load the particle image to extract later sub-particles
                        img = ih.read(particle)
                        x, y, _, _ = img.getDimensions()
                        data = img.getData()

                lastPartId = partId

            # If particle is not in inputParticles, subparticles will not be
            # generated. Now, subtract from a subset of original particles is
            # supported.
            if partId not in partIdExcluded:
                if self.extractFromMicrographs.get():
                    if data is None:
                        discardedOutliers += 1
                        continue
                    xpos, ypos = self._computeMicrographPosition(
                        currentParticleCoord, coord, halfParticleDim)
                else:
                    xpos = coord.getX()
                    ypos = coord.getY()

                centerData, usedPadding = self._extractWindowWithPadding(
                    data, xpos, ypos, boxSize, self.extractAll.get())
                if centerData is None:
                    discardedOutliers += 1
                    continue

                if usedPadding:
                    paddedOutliers += 1

                center[:, :] = centerData
                outputImg.setData(center)
                i += 1
                outputImg.write((i, outputStack))
                subpart = coord._subparticle
                if (not missingProvenanceWarned and
                        not has_subparticle_provenance(subpart)):
                    self.warning("Subparticle provenance fields "
                                 "(_symmetryGroup/_symmetryOperatorId) are "
                                 "missing for some items. Continuing without "
                                 "failing.")
                    missingProvenanceWarned = True
                subpart.setLocation(
                    (i, outputStack))  # Change path to new stack
                subpart.setObjId(i)  # Ids will be always the same no mater the number of outliers 
                outputSet.append(subpart)

        progress.finish()
        if discardedOutliers:
            self.info("WARNING: Discarded %s particles because laid out of the "
                      "particle (for a box size of %d)" %
                      (discardedOutliers, boxSize))
        if paddedOutliers:
            self.info("INFO: Extracted %s out-of-box sub-particles with "
                      "boundary padding (box size of %d)." %
                      (paddedOutliers, boxSize))
        outputSet.setIsSubparticles(True)
        self._defineOutputs(**{self.OUTPUTPARTICLESNAME: outputSet})
        self._defineSourceRelation(self.inputParticles, outputSet)

    @staticmethod
    def _extractWindowWithPadding(data, xpos, ypos, boxSize, extractAll):
        """Extract a centered window. If extractAll is True, windows that
        extend outside boundaries are padded by clamping to image edges.
        """
        b2 = int(round(boxSize / 2))
        xDim = data.shape[1]
        yDim = data.shape[0]

        x0 = xpos - b2
        x1 = xpos + b2
        y0 = ypos - b2
        y1 = ypos + b2

        inBounds = (y0 >= 0 and y1 <= yDim and x0 >= 0 and x1 <= xDim)
        if inBounds:
            return data[y0:y1, x0:x1], False

        if not extractAll:
            return None, False

        xIndices = np.clip(np.arange(x0, x1), 0, xDim - 1)
        yIndices = np.clip(np.arange(y0, y1), 0, yDim - 1)
        return data[np.ix_(yIndices, xIndices)], True

    @staticmethod
    def _getParentParticleId(coord):
        """Return the parent particle ObjId encoded in coordinate _micId."""
        if not hasattr(coord, '_micId'):
            return None

        parentId = coord._micId
        if hasattr(parentId, 'get'):
            parentId = parentId.get()
        return parentId

    @staticmethod
    def _computeMicrographPosition(particleCoord, subpartCoord, halfParticleDim):
        """Compute subparticle center in the original micrograph frame.

        Coordinate frames:
        - subpartCoord (x/y): particle-box image frame, top-left origin.
        - xOffset/yOffset: particle-centered frame (subtract half box size).
        - particleCoord + offset: original micrograph frame.
        """
        xOffset = subpartCoord.getX() - halfParticleDim
        yOffset = subpartCoord.getY() - halfParticleDim
        xpos = int(particleCoord.getX() + xOffset)
        ypos = int(particleCoord.getY() + yOffset)

        return xpos, ypos

    # -------------------------- INFO functions -------------------------------
    def _validate(self):
        errors = []
        inputParticles = self.inputParticles.get()
        inputCoords = self.inputCoordinates.get()
        firstCoord = inputCoords.getFirstItem()

        if firstCoord is None:
            errors.append('Input coordinates set is empty.')
            return errors

        if not firstCoord.hasAttribute('_subparticle'):
            errors.append('The selected input coordinates does not are the '
                          'output from a localized-subparticles protocol.')
        if self.extractFromMicrographs.get():
            if self.inputMicrographs.get() is None:
                errors.append('Micrographs input is required when "Extract '
                              'from micrographs?" is set to Yes.')
            else:
                inputMicIds = {m.getObjId() for m in self.inputMicrographs.get()}

                particleMicIds = set()
                particleMics = (inputParticles.getMicrographs()
                                if hasattr(inputParticles, 'getMicrographs')
                                else None)

                if particleMics is not None:
                    particleMicIds = {m.getObjId() for m in particleMics}
                else:
                    for particle in inputParticles:
                        particleCoord = (particle.getCoordinate()
                                         if hasattr(particle, 'getCoordinate')
                                         else None)
                        if particleCoord is None:
                            continue
                        micId = (particleCoord.getMicId()
                                 if hasattr(particleCoord, 'getMicId')
                                 else None)
                        if micId is not None:
                            particleMicIds.add(micId)

                if not particleMicIds:
                    errors.append('Input particles do not contain enough '
                                  'micrograph linkage information to validate '
                                  'against the provided micrographs.')
                elif not particleMicIds.issubset(inputMicIds):
                    errors.append('Input micrographs do not contain all '
                                  'micrographs referenced by input particles.')

        return errors

    def _citations(self):
        return ['Serban2015', 'Abrishami2020']

    def _summary(self):
        summary = []
        return summary

    def _methods(self):
        return []
