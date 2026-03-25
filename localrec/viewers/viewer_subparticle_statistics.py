# ***************************************************************************
# *
# * Authors:   Scipion localrec maintainers
# *
# * This program is free software; you can redistribute it and/or modify
# * it under the terms of the GNU General Public License as published by
# * the Free Software Foundation; either version 2 of the License, or
# * (at your option) any later version.
# *
# ***************************************************************************

import json
import os

from pyworkflow.viewer import DESKTOP_TKINTER, Viewer
from pwem.viewers import EmPlotter

from localrec.protocols.protocol_subparticle_statistics import ProtSubparticleStatistics


class ProtSubparticleStatisticsViewer(Viewer):
    """Visualize histogram outputs from ProtSubparticleStatistics."""

    _label = 'viewer subparticle statistics'
    _targets = [ProtSubparticleStatistics]
    _environments = [DESKTOP_TKINTER]

    def _visualize(self, obj, **args):
        json_path = self.protocol._getExtraPath('histogram_type1.json')
        if not os.path.exists(json_path):
            raise Exception('Histogram output not found: %s' % json_path)

        with open(json_path) as handle:
            data = json.load(handle)

        bins = data.get('bins', [])
        counts = data.get('counts', [])
        class_ids = data.get('targetClassIds')
        if class_ids is None:
            class_ids = [data.get('targetClass', self.protocol.targetClass.get())]
        class_ids_str = ','.join([str(cid) for cid in class_ids])

        plotter = EmPlotter(windowTitle='Subparticle statistics')
        axis = plotter.createSubPlot('Subparticle occupancy (class ID(s): %s)'
                                     % class_ids_str,
                                     'Subparticles in selected class ID(s)',
                                     'Number of parent particles')
        axis.bar(bins, counts)
        axis.set_xticks(bins)

        return [plotter]
