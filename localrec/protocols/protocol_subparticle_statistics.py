# **************************************************************************
# *
# * Authors:   Scipion localrec maintainers
# *
# * This program is free software; you can redistribute it and/or modify
# * it under the terms of the GNU General Public License as published by
# * the Free Software Foundation; either version 2 of the License, or
# * (at your option) any later version.
# *
# **************************************************************************

import csv
import json
import os

from pyworkflow import VERSION_3_0
from pyworkflow.protocol.params import PointerParam, IntParam
from pwem.protocols import ProtParticles


class ProtSubparticleStatistics(ProtParticles):
    """Compute class-occupancy statistics for subparticles.

    Histogram type 1:
        - X axis categories: number of subparticles assigned to class X per
          original particle (0..N).
        - Y axis: number of original particles in each category.
    """

    _label = 'subparticle statistics'
    _lastUpdateVersion = VERSION_3_0

    def _defineParams(self, form):
        form.addSection(label='Input')
        form.addParam('inputClassification', PointerParam,
                      pointerClass='SetOfClasses3D, SetOfClasses2D',
                      important=True,
                      label='Input classification',
                      help='Select a classification output where each class '
                           'contains subparticles.')

        form.addParam('targetClass', IntParam, default=1,
                      label='Target class (X)',
                      help='Class id X used to compute occupancy histogram.')

        form.addParam('maxCount', IntParam, default=-1,
                      label='Maximum count N (optional)',
                      help='If > 0, force histogram bins 0..N. If <= 0, N is '
                           'automatically inferred from classification data.')

        form.addParallelSection(threads=0, mpi=0)

    def _insertAllSteps(self):
        self._insertFunctionStep('computeStatisticsStep')

    def computeStatisticsStep(self):
        classification = self.inputClassification.get()

        per_particle_count = {}
        target_class = int(self.targetClass.get())

        for parent_id, class_id in self._iterAssignments(classification):
            if parent_id not in per_particle_count:
                per_particle_count[parent_id] = 0
            if class_id == target_class:
                per_particle_count[parent_id] += 1

        if not per_particle_count:
            raise Exception('No subparticle assignments found in the selected '
                            'classification input.')

        auto_n = max(per_particle_count.values())
        user_n = int(self.maxCount.get())
        n = user_n if user_n > 0 else auto_n

        if n < auto_n:
            self.warning('Provided maxCount (%d) is smaller than observed '
                         'maximum (%d). Using observed maximum.' % (n, auto_n))
            n = auto_n

        histogram = {k: 0 for k in range(n + 1)}
        overflow = 0

        for count in per_particle_count.values():
            if count <= n:
                histogram[count] += 1
            else:
                overflow += 1

        if overflow:
            self.warning('%d particles exceeded maximum bin N=%d and were '
                         'grouped in overflow.' % (overflow, n))

        bins = list(range(n + 1))
        counts = [histogram[k] for k in bins]

        data = {
            'histogramType': 1,
            'targetClass': target_class,
            'maxCount': n,
            'overflow': overflow,
            'totalParticles': len(per_particle_count),
            'bins': bins,
            'counts': counts
        }

        json_path = self._getExtraPath('histogram_type1.json')
        csv_path = self._getExtraPath('histogram_type1.csv')

        with open(json_path, 'w') as handle:
            json.dump(data, handle, indent=2)

        with open(csv_path, 'w') as handle:
            writer = csv.writer(handle)
            writer.writerow(['category_k', 'particle_count'])
            for k in bins:
                writer.writerow([k, histogram[k]])

        self._histSummary = data

    def _iterAssignments(self, classification):
        """Yield tuples: (parent_particle_id, class_id)."""
        for cls in classification:
            class_id = self._getClassId(cls)
            if hasattr(cls, 'iterItems'):
                cls_items = cls.iterItems()
            else:
                cls_items = cls
            for item in cls_items:
                parent_id = self._getParentParticleId(item)
                yield parent_id, class_id

    @staticmethod
    def _getClassId(cls):
        for attr in ('_objId', 'getObjId', 'id'):
            if hasattr(cls, attr):
                value = getattr(cls, attr)
                value = value() if callable(value) else value
                if value is not None:
                    return int(value)
        raise Exception('Could not read class id from classification class.')

    @staticmethod
    def _getParentParticleId(subparticle):
        if hasattr(subparticle, 'getCoordinate'):
            coord = subparticle.getCoordinate()
            if coord is not None and hasattr(coord, '_micId'):
                return int(coord._micId)

        if hasattr(subparticle, '_micId'):
            return int(subparticle._micId)

        raise Exception('Could not read parent particle id for subparticle.')

    def _validate(self):
        errors = []
        if self.targetClass.get() < 1:
            errors.append('Target class (X) must be >= 1.')
        return errors

    def _summary(self):
        summary = []
        json_path = self._getExtraPath('histogram_type1.json')
        if os.path.exists(json_path):
            summary.append('Histogram type 1 computed for class %d.'
                           % self.targetClass.get())
            summary.append('Results: %s' % json_path)
        return summary

    def _methods(self):
        return []
