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
        class_id = data.get('targetClass', self.protocol.targetClass.get())

        plotter = EmPlotter(windowTitle='Subparticle statistics')
        axis = plotter.createSubPlot('Histogram type 1 (class %s)' % class_id,
                                     'Subparticles in target class',
                                     'Number of parent particles')
        axis.bar(bins, counts)
        axis.set_xticks(bins)

        return [plotter]
