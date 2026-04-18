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
from pyworkflow.protocol.params import PointerParam, BooleanParam
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
        outliers = 0
        partIdExcluded = []
        lastPartId = None
        lastMicId = None
        missingProvenanceWarned = False

        progress = ProgressBar(len(inputCoords), fmt=ProgressBar.NOBAR)
        progress.start()
        step = max(100, len(inputCoords) // 100)
        for i, coord in enumerate(inputCoords.iterItems(orderBy=['_subparticle._micId',
                                                    '_micId', 'id'])):
            if i % step == 0:
                progress.update(i+1)

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
                    if self.extractFromMicrographs.get():
                        particleCoord = particle.getCoordinate()
                        micId = particleCoord.getMicId()
                        if micId != lastMicId:
                            mic = inputMicrographs[micId]
                            if mic is None:
                                self.info("WARNING: Missing micrograph with "
                                          "id %s from input micrographs set"
                                          % micId)
                                lastMicId = None
                                data = None
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
                        outliers += 1
                        continue
                    particle = inputParticles[partId]
                    particleCoord = particle.getCoordinate()
                    xOffset = coord.getX() - halfParticleDim
                    yOffset = coord.getY() - halfParticleDim
                    xpos = int(particleCoord.getX() + xOffset)
                    ypos = int(particleCoord.getY() + yOffset)
                else:
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
        if outliers:
            self.info("WARNING: Discarded %s particles because laid out of the "
                      "particle (for a box size of %d" % (outliers, boxSize))
        outputSet.setIsSubparticles(True)
        self._defineOutputs(**{self.OUTPUTPARTICLESNAME: outputSet})
        self._defineSourceRelation(self.inputParticles, outputSet)

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
