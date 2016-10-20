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

# This script makes figure 3.54, which shows the results of model selection for
# the case where D and V are taken to be given by NTH's old results.

from __future__ import division

import os
import pymultinest
import scipy

num_eig = range(1, 8)
lnev_ins = scipy.zeros_like(num_eig, dtype=float)
lnev_ns = scipy.zeros_like(num_eig, dtype=float)

for i, n in enumerate(num_eig):
    basename = os.path.abspath('/Users/markchilenski/src/bayesimp/chains_1101014006_22/c-D%d-V%d-' % (n, n))
    a = pymultinest.Analyzer(
        n_params=2 + n + n - 1 + n + n - 1,
        outputfiles_basename=basename
    )
    stats = a.get_stats()
    lnev_ins[i] = stats['nested importance sampling global log-evidence']
    lnev_ns[i] = stats['nested sampling global log-evidence']

# Good knots:
n = 5
basename = os.path.abspath('/Users/markchilenski/src/bayesimp/chains_1101014006_22/c-D5-V5-opt_knots-')
a = pymultinest.Analyzer(
    n_params=2 + n + n - 1 + n + n - 1,
    outputfiles_basename=basename
)
stats = a.get_stats()
lnev_ins_opt_knots = stats['nested importance sampling global log-evidence']
lnev_ns_opt_knots = stats['nested sampling global log-evidence']

# Bad knots:
basename = os.path.abspath('/Users/markchilenski/src/bayesimp/chains_1101014006_22/c-D5-V5-bad_knots-')
a = pymultinest.Analyzer(
    n_params=2 + n + n - 1 + n + n - 1,
    outputfiles_basename=basename
)
stats = a.get_stats()
lnev_ins_bad_knots = stats['nested importance sampling global log-evidence']
lnev_ns_bad_knots = stats['nested sampling global log-evidence']

import setupplots
setupplots.thesis_format()
import matplotlib.pyplot as plt
plt.ion()

f = plt.figure(figsize=(0.75 * setupplots.TEXTWIDTH, 2 * 0.75 * setupplots.TEXTWIDTH / 1.618))
a = f.add_subplot(2, 1, 1)
a.plot(num_eig, lnev_ins, 'bs--', label=r'linearly-spaced, \textsc{ins}')
a.plot(num_eig, lnev_ns, 'cD:', label=r'linearly-spaced, \textsc{ns}')
a.plot(5, lnev_ins_opt_knots, 'g^', label=r'near-optimal, \textsc{ins}')
a.plot(5, lnev_ns_opt_knots, 'yv', label=r'near-optimal, \textsc{ns}')
a.plot(5, lnev_ins_bad_knots, 'ro', label=r'bad, \textsc{ins}')
a.plot(5, lnev_ns_bad_knots, 'mh', label=r'bad, \textsc{ns}')
a.set_ylabel("log-evidence")
a.set_xlim((num_eig[0] - 0.5, num_eig[-1] + 0.5))
a.set_xticks(num_eig)
a.set_title("Full")
a.legend(loc='lower right', numpoints=1)

az = f.add_subplot(2, 1, 2)
az.plot(num_eig[3:], lnev_ins[3:], 'bs--')
az.plot(num_eig[3:], lnev_ns[3:], 'cD:')
az.plot(5, lnev_ins_opt_knots, 'g^')
az.plot(5, lnev_ns_opt_knots, 'yv')
az.plot(5, lnev_ins_bad_knots, 'ro')
az.plot(5, lnev_ns_bad_knots, 'mh')
az.set_xlabel("number of coefficients")
az.set_ylabel("log-evidence")
az.set_xlim((num_eig[3] - 0.5, num_eig[-1] + 0.5))
az.set_xticks(num_eig[3:])
az.set_title("Zoomed")

f.suptitle("Model selection with complicated synthetic data")
f.subplots_adjust(hspace=0.23)
setupplots.apply_formatter(f)
f.savefig("complexModelSelection.pdf", bbox_inches='tight')
f.savefig("complexModelSelection.pgf", bbox_inches='tight')
