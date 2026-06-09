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
from pyworkflow.protocol.params import (PointerParam, BooleanParam,
                                        LEVEL_ADVANCED, FloatParam, IntParam)
from pwem.protocols import ProtParticles

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

        form.addParam('downFactor', FloatParam, default=1.0,
                      condition='extractFromMicrographs',
                      label='Micrograph downsampling factor',
                      help='Downsample the input micrographs before extracting '
                           'subparticles. Use 1.0 for no downsampling. If '
                           'parent particles were previously extracted with a '
                           'downsampling factor from full-resolution '
                           'micrographs, use the same factor here to preserve '
                           'the output sampling rate.')

        form.addParam('boxSize', IntParam,
                      label='Subparticle box size (px)',
                      help='Select the output sub-particle box size in pixels. '
                           'When extracting from micrographs with a '
                           'downsampling factor, this is the final box size '
                           'after micrograph downsampling.')
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

        extractFromMicrographs = self.extractFromMicrographs.get()
        downFactor = self._getExtractionDownFactor()
        particleSampling = self._getSamplingRate(inputParticles)
        micSampling = (self._getSamplingRate(inputMicrographs)
                       if extractFromMicrographs else None)
        coordMicSampling = (self._getParticleCoordinateMicSampling(
            inputParticles, inputMicrographs) if extractFromMicrographs else None)
        outputSampling = (micSampling * downFactor
                          if extractFromMicrographs else particleSampling)
        if extractFromMicrographs:
            outputSet.setSamplingRate(outputSampling)

        boxSize = self.boxSize.get()
        self._setOutputSetDimensions(outputSet, boxSize)
        halfParticleDim = inputParticles.getXDim() / 2.0
        center = np.zeros((boxSize, boxSize))

        ih = ImageHandler()

        outIndex = 0
        firstOutputValidated = False
        discardedOutliers = 0
        paddedOutliers = 0
        partIdExcluded = []
        lastPartId = None
        lastMicId = None
        missingProvenanceWarned = False

        progress = ProgressBar(len(inputCoords), fmt=ProgressBar.NOBAR)
        progress.start()
        step = max(100, len(inputCoords) // 100)
        for coordIndex, coord in enumerate(inputCoords.iterItems(
                orderBy=['_subparticle._micId', '_micId', 'id'])):
            if coordIndex % step == 0:
                progress.update(coordIndex + 1)

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
                    if extractFromMicrographs:
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
                                data = img.getData()
                                data = self._downsampleMicrographData(
                                    data, downFactor)
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
                outputCoord = None
                if extractFromMicrographs:
                    if data is None:
                        discardedOutliers += 1
                        continue
                    particle = inputParticles[partId]
                    particleCoord = particle.getCoordinate()
                    xOffset = coord.getX() - halfParticleDim
                    yOffset = coord.getY() - halfParticleDim
                    xpos, ypos = self._computeMicrographCropCenter(
                        particleCoord.getX(), particleCoord.getY(),
                        xOffset, yOffset, coordMicSampling, particleSampling,
                        outputSampling)
                    outputX, outputY = self._computeMicrographCropCenter(
                        particleCoord.getX(), particleCoord.getY(),
                        xOffset, yOffset, coordMicSampling, particleSampling,
                        micSampling)
                    outputCoord = self._cloneMicrographCoordinate(
                        coord, particleCoord, outputX, outputY, partId)
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
                outIndex += 1
                outputImg.write((outIndex, outputStack))
                if not firstOutputValidated:
                    self._validateWrittenOutputImage(ih, outputStack, outIndex,
                                                     boxSize)
                    firstOutputValidated = True
                subpart = coord._subparticle.clone()
                if (not missingProvenanceWarned and
                        not has_subparticle_provenance(subpart)):
                    self.warning("Subparticle provenance fields "
                                 "(_symmetryGroup/_symmetryOperatorId) are "
                                 "missing for some items. Continuing without "
                                 "failing.")
                    missingProvenanceWarned = True
                if extractFromMicrographs:
                    subpart.setCoordinate(outputCoord)
                    self._scaleSubparticleOriginShift(
                        subpart, particleSampling / outputSampling)
                subpart.setLocation(
                    (outIndex, outputStack))  # Change path to new stack
                # Ids will be always contiguous despite skipped outliers.
                subpart.setObjId(outIndex)
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
    def _setOutputSetDimensions(outputSet, boxSize):
        """Set output particle dimensions without reading the first item.

        Localized extraction can emit subparticles with a box size that differs
        from the parent particles.  Record those dimensions eagerly so Scipion
        does not need to infer them by opening the first output stack item while
        the set is being defined.
        """
        boxDim = (boxSize, boxSize, 1)

        if hasattr(outputSet, 'setDim'):
            outputSet.setDim(boxDim)
            return

        if hasattr(outputSet, 'setDimensions'):
            outputSet.setDimensions(boxDim)
            return

        dimensions = getattr(outputSet, '_dimensions', None)
        if dimensions is not None:
            dimText = '%d %d %d' % boxDim
            if hasattr(dimensions, 'set'):
                dimensions.set(dimText)
            else:
                outputSet._dimensions = dimText

    @staticmethod
    def _validateWrittenOutputImage(ih, outputStack, outIndex, boxSize):
        """Ensure the first written subparticle can be read as boxSize x boxSize."""
        location = (outIndex, outputStack)
        try:
            img = ih.read(location)
        except Exception as exc:
            raise RuntimeError(
                'Could not read the first extracted subparticle from %s at '
                'index %d; expected a %dx%d image.'
                % (outputStack, outIndex, boxSize, boxSize)) from exc

        xDim, yDim, _, _ = img.getDimensions()
        if xDim != boxSize or yDim != boxSize:
            raise RuntimeError(
                'First extracted subparticle in %s at index %d has dimensions '
                '%dx%d, but expected %dx%d.'
                % (outputStack, outIndex, xDim, yDim, boxSize, boxSize))

    @staticmethod
    def _getSamplingRate(itemSet):
        """Return a positive sampling rate from a Scipion image set."""
        if itemSet is None or not hasattr(itemSet, 'getSamplingRate'):
            return None
        sampling = itemSet.getSamplingRate()
        return sampling if sampling and sampling > 0 else None

    def _getExtractionDownFactor(self):
        """Return the micrograph downsampling factor, preserving old runs."""
        if not hasattr(self, 'downFactor'):
            return 1.0
        downFactor = self.downFactor.get()
        return float(downFactor) if downFactor else 1.0

    @classmethod
    def _computeMicrographCropCenter(cls, particleX, particleY, xOffset,
                                     yOffset, coordMicSampling,
                                     particleSampling, outputSampling):
        """Convert parent-center plus subparticle offset to an image grid.

        particleX/Y are in the coordinate micrograph grid, xOffset/yOffset are
        in the parent particle grid, and the returned crop center is in the
        output extraction grid.
        """
        xpos = (particleX * coordMicSampling / outputSampling +
                xOffset * particleSampling / outputSampling)
        ypos = (particleY * coordMicSampling / outputSampling +
                yOffset * particleSampling / outputSampling)
        return int(round(xpos)), int(round(ypos))

    def _getParticleCoordinateMicSampling(self, inputParticles,
                                          inputMicrographs):
        """Return sampling for the micrograph grid used by particle coords."""
        particleMics = (inputParticles.getMicrographs()
                        if hasattr(inputParticles, 'getMicrographs')
                        else None)
        sampling = self._getSamplingRate(particleMics)
        if sampling is None:
            sampling = self._getSamplingRate(inputMicrographs)
            self.info('WARNING: Could not determine the sampling rate of the '
                      'micrographs associated with input particles. Assuming '
                      'input particle coordinates use the provided micrograph '
                      'sampling rate.')
        return sampling

    @staticmethod
    def _downsampleMicrographData(data, downFactor):
        """Fourier-downsample a micrograph array before extraction."""
        if downFactor <= 1.0:
            return data

        yDim, xDim = data.shape[:2]
        newYDim = int(round(yDim / downFactor))
        newXDim = int(round(xDim / downFactor))
        if newYDim <= 0 or newXDim <= 0:
            raise ValueError('Invalid downsampling factor %.3f for micrograph '
                             'dimensions %dx%d.' % (downFactor, xDim, yDim))

        spectrum = np.fft.fftshift(np.fft.fft2(data))
        yStart = max((yDim - newYDim) // 2, 0)
        xStart = max((xDim - newXDim) // 2, 0)
        cropped = spectrum[yStart:yStart + newYDim,
                           xStart:xStart + newXDim]
        downsampled = np.fft.ifft2(np.fft.ifftshift(cropped)).real
        downsampled *= (float(newYDim * newXDim) / float(yDim * xDim))
        return downsampled.astype(data.dtype, copy=False)

    @staticmethod
    def _cloneMicrographCoordinate(coord, particleCoord, xpos, ypos,
                                   parentParticleId):
        """Clone a coordinate and set it in source-micrograph coordinates."""
        outputCoord = coord.clone()
        ProtLocalizedExtraction._sanitizeOutputCoordinate(outputCoord)
        outputCoord.setX(xpos)
        outputCoord.setY(ypos)
        outputCoord.setMicId(particleCoord.getMicId())
        if (hasattr(particleCoord, 'getMicName') and
                hasattr(outputCoord, 'setMicName')):
            outputCoord.setMicName(particleCoord.getMicName())
        if hasattr(coord, '_micId'):
            outputCoord._parentParticleId = coord._micId.clone()
        else:
            outputCoord._parentParticleId = parentParticleId
        return outputCoord

    @staticmethod
    def _sanitizeOutputCoordinate(coord):
        """Remove localized-coordinate payloads before storing in a particle.

        Coordinates produced by localized reconstruction carry the source
        subparticle in ``_subparticle``.  Keeping that nested particle inside the
        coordinate of an extracted output particle causes Scipion's SQLite mapper
        to see many more fields than the output SetOfParticles schema contains.
        """
        if hasattr(coord, '_subparticle'):
            delattr(coord, '_subparticle')
        return coord

    @staticmethod
    def _scaleSubparticleOriginShift(subpart, scale):
        """Scale in-plane fractional shifts when output sampling changes."""
        if scale is None or abs(scale - 1.0) < 1e-6:
            return

        for attrName in ('getTransform', '_transorg'):
            if attrName == 'getTransform':
                transform = (subpart.getTransform()
                             if hasattr(subpart, 'getTransform') else None)
            else:
                transform = getattr(subpart, attrName, None)
            if transform is None or not hasattr(transform, 'getMatrix'):
                continue
            matrix = np.array(transform.getMatrix(), copy=True)
            matrix[0, 3] *= scale
            matrix[1, 3] *= scale
            transform.setMatrix(matrix)

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
            downFactor = self._getExtractionDownFactor()
            if downFactor < 1.0:
                errors.append('Micrograph downsampling factor must be >= 1.0.')

            particleSampling = self._getSamplingRate(inputParticles)
            if particleSampling is None:
                errors.append('Input particles do not define a valid sampling '
                              'rate.')

            if self.inputMicrographs.get() is None:
                errors.append('Micrographs input is required when "Extract '
                              'from micrographs?" is set to Yes.')
            else:
                inputMicrographs = self.inputMicrographs.get()
                micSampling = self._getSamplingRate(inputMicrographs)
                if micSampling is None:
                    errors.append('Input micrographs do not define a valid '
                                  'sampling rate.')
                elif particleSampling is not None:
                    expectedDownFactor = particleSampling / micSampling
                    if abs(expectedDownFactor - downFactor) > 1e-3:
                        self.info('WARNING: Input particle sampling suggests a '
                                  'micrograph downsampling factor of %.4f, but '
                                  '%.4f was selected. This is valid if you '
                                  'intentionally want a different output '
                                  'sampling.' % (expectedDownFactor, downFactor))

                inputMicIds = {m.getObjId() for m in inputMicrographs}

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
        if self.extractFromMicrographs.get():
            inputMicrographs = self.inputMicrographs.get()
            micSampling = self._getSamplingRate(inputMicrographs)
            downFactor = self._getExtractionDownFactor()
            if micSampling is not None:
                summary.append('Extracted from micrographs with downsampling '
                               'factor %.4f and output sampling %.4f.'
                               % (downFactor, micSampling * downFactor))
        return summary

    def _methods(self):
        return self._summary()
