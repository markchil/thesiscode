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

# This script makes figure D.5, which shows a comparison between the IDL and
# Python tools for converting (R, Z) to psi.

from __future__ import division

import cPickle as pkl
import setupplots
setupplots.thesis_format()
import matplotlib.pyplot as plt
plt.ion()
from mpl_toolkits.axes_grid1 import make_axes_locatable
import scipy

with open('outputs/efit_IDL_compare_results.pkl', 'rb') as pf:
    out = pkl.load(pf)

f = plt.figure()
a = f.add_subplot(1, 1, 1)
a.set_aspect('equal')
CS = a.pcolormesh(
    setupplots.make_pcolor_grid(out['R_grid']),
    setupplots.make_pcolor_grid(out['Z_grid']),
    1e7 * scipy.absolute(out["rz2psi_out"] - out["IDL_rz2psi_out"]),
    cmap='plasma'
)
a.set_xlim(out['R_grid'].min(), out['R_grid'].max())
a.set_ylim(out['Z_grid'].min(), out['Z_grid'].max())
divider = make_axes_locatable(a)
a_cb = divider.append_axes("right", size="10%", pad=0.05)
cbar = f.colorbar(CS, cax=a_cb)
cbar.ax.set_ylabel(r'[$\SI{e-7}{Wb/rad}$]')
a.set_xlabel('$R$ [m]')
a.set_ylabel('$Z$ [m]')
a.set_title(r'$|$\texttt{eqtools} - \textsc{idl}$|$')
a.xaxis.set_major_locator(plt.MaxNLocator(nbins=6))
f.set_figwidth(0.6 * setupplots.TEXTWIDTH)
f.tight_layout()
setupplots.apply_formatter(f)
f.savefig("efit_test.pdf", bbox_inches='tight')
f.savefig("efit_test.pgf", bbox_inches='tight')
