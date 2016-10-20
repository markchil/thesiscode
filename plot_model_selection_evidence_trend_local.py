# Copyright 2016 Mark Chilenski
# This program is distributed under the terms of the GNU General Purpose License (GPL).
# Refer to http://www.gnu.org/licenses/gpl.txt
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

# This script makes figure 3.52, which demonstrates how to select the
# appropriate level of complexity in D and V for simple profiles.

from __future__ import division

import os
import pymultinest
import scipy

num_eig = range(1, 5)
lnev = scipy.zeros_like(num_eig, dtype=float)
errlnev = scipy.zeros_like(num_eig, dtype=float)

for i, n in enumerate(num_eig):
    basename = os.path.abspath('/Users/markchilenski/src/bayesimp/chains_1101014006_24/c-D%d-V%d-' % (n, n))
    if os.path.isfile(basename + '.txt'):
        a = pymultinest.Analyzer(
            n_params=2 + n + n - 1 + n + n - 1,
            outputfiles_basename=basename
        )
        stats = a.get_stats()
        lnev[i] = stats['global evidence']
        errlnev[i] = stats['global evidence error']
    else:
        lnev[i] = None
        errlnev[i] = None

import setupplots
setupplots.thesis_format()
import matplotlib.pyplot as plt
plt.ion()

f = plt.figure(figsize=(0.75 * setupplots.TEXTWIDTH, 0.75 * setupplots.TEXTWIDTH / 1.618))
ax = f.add_subplot(1, 1, 1)
ax.plot(num_eig, lnev, 'o--')
ax.set_xlabel("number of coefficients")
ax.set_ylabel("log-evidence")
ax.set_title("Selecting a model by maximizing the evidence")
ax.set_xlim((num_eig[0] - 0.5, num_eig[-1] + 0.5))
ax.set_xticks(num_eig)
setupplots.apply_formatter(f)
f.savefig("modelSelectionBasic.pdf", bbox_inches='tight')
f.savefig("modelSelectionBasic.pgf", bbox_inches='tight')
