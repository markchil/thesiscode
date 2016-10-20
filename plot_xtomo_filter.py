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

# This script makes figure 3.28, which shows the transmission of the XTOMO
# filter.

from __future__ import division

import scipy
import setupplots
setupplots.thesis_format()
import matplotlib.pyplot as plt
plt.ion()

import sys
sys.path.insert(0, '/Users/markchilenski/src/bayesimp')

import lines

lines.read_filter_file(
    '/Users/markchilenski/src/bayesimp/spectral_modeling/Be_filter_50_um.dat',
    plot=True,
    title=r'$\SI{50}{\micro m}$ Be filter',
    figsize=(0.5 * setupplots.TEXTWIDTH, 0.5 * setupplots.TEXTWIDTH / 1.618)
)
f = plt.gcf()
a = plt.gca()
a2 = a.twiny()
a2.set_xlim(a.get_xlim())
lam_locs = scipy.asarray([10, 1, 0.5, 0.25, 0.125], dtype=float)
lam_s = [r'$10\vphantom{0123456789}$', r'$1\vphantom{0123456789}$', r'$0.5\vphantom{0123456789}$', r'$0.25\vphantom{0123456789}$', r'$0.125\vphantom{0123456789}$']
E_locs = 1e-3 * scipy.constants.h * scipy.constants.c / (scipy.constants.e * lam_locs * 1e-9)
a2.set_xticks(E_locs)
a2.set_xticklabels(lam_s)
a2.set_xlabel(r"$\lambda$ [nm]")
a.set_title(r'$\SI{50}{\micro m}$ Be filter', y=1.275)
setupplots.apply_formatter(f)
f.savefig("XTOMO_filter.pdf", bbox_inches='tight')
f.savefig("XTOMO_filter.pgf", bbox_inches='tight')
