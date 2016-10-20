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

# This script makes figure D.2, which shows the vacuum and total toroidal field
# components.

from __future__ import division

import eqtools
import scipy

e = eqtools.CModEFITTree(1120907032)
idx = 47
R = e.getRGrid()
Z0 = e.getMagZ()[idx]
t = e.getTimeBase()[idx]
psinorm_mid = e.rz2psinorm(R, Z0 * scipy.zeros_like(R), t)
core_mask = psinorm_mid <= 1.0
RLCFS_in = R[core_mask].min()
RLCFS_out = R[core_mask].max()
BT_tot = e.rz2BT(R, Z0 * scipy.ones_like(R), t)
BT_vac = e.getBtVac()[idx] * e.getMagR()[idx] / R

import setupplots
setupplots.thesis_format()
import matplotlib.pyplot as plt
plt.ion()

f = plt.figure(figsize=(0.75 * setupplots.TEXTWIDTH, 0.75 * setupplots.TEXTWIDTH / 1.618))
a = f.add_subplot(1, 1, 1)
a.plot(R, BT_vac, '--', linewidth=setupplots.lw, label='vacuum')
a.plot(R, BT_tot, '-', linewidth=setupplots.lw, label='total')
a.axvspan(RLCFS_in, RLCFS_out, color='k', alpha=0.25, label='plasma')
a.legend(loc='upper right')
a.set_xlabel('$R$ [m]')
a.set_ylabel(r'$B_\phi$ [T]')
a.set_title("Comparison of vacuum to total toroidal field")
a.set_xlim(0.4, 1.1)
setupplots.apply_formatter(f)
f.savefig("EFITBtor.pdf", bbox_inches='tight')
f.savefig("EFITBtor.pgf", bbox_inches='tight')
