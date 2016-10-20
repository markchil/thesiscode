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

# This script makes figure 2.26, which shows the toroidal rotation profiles for
# two L-mode plasmas with slightly different densities.

from __future__ import division

import scipy
import MDSplus
import eqtools

t_min = 0.9
t_max = 1.2
abscissa = 'r/a'

# Norman re-ran this analysis in THT=6 to fix a broken radial grid.
# This doesn't quite look like figure 3 from White PoP 2013 on account of that
# figure using time-averaged data.

class Case(object):
    def __init__(self, shot, color, label, style):
        self.shot = shot
        self.color = color
        self.label = label
        self.style = style
        
        self.e = eqtools.CModEFITTree(self.shot)
        
        tree = MDSplus.Tree('spectroscopy', self.shot)
        N_pro = tree.getNode('hirexsr.analysis6.helike.profiles.z.pro')
        self.vtor = scipy.asarray(N_pro.data(), dtype=float)[1, :, :]
        N_rho = tree.getNode('hirexsr.analysis6.helike.profiles.z.rho')
        self.rho = scipy.asarray(N_rho.data(), dtype=float)
        self.t = scipy.asarray(N_rho.dim_of().data(), dtype=float)
        N_proerr = tree.getNode('hirexsr.analysis6.helike.profiles.z.proerr')
        self.err_vtor = scipy.asarray(N_proerr.data(), dtype=float)[1, :, :]
        
        t_mask = (self.t >= t_min) & (self.t <= t_max)
        
        self.vtor = self.vtor[t_mask, :]
        self.rho = self.rho[t_mask, :]
        self.t = self.t[t_mask]
        self.err_vtor = self.err_vtor[t_mask, :]
        
        t_pattern = scipy.reshape(scipy.tile(self.t, self.rho.shape[1]), (self.rho.shape[1], self.rho.shape[0])).T
        self.roa = self.e.psinorm2roa(self.rho, t_pattern, each_t=False)

cases = [Case(1120221011, 'r', 'hollow', 'o'), Case(1120221012, 'b', 'peaked', '^')]

import setupplots
setupplots.thesis_format()
import matplotlib.pyplot as plt
plt.ion()

f = plt.figure(figsize=(0.75 * setupplots.TEXTWIDTH, 0.75 * setupplots.TEXTWIDTH / 1.618))
a = f.add_subplot(1, 1, 1)
for c in cases:
    a.errorbar(c.roa.ravel(), c.vtor.ravel(), yerr=c.err_vtor.ravel(), label=c.label, color=c.color, fmt=c.style)
a.axhline(0.0, color='k', alpha=0.5)
a.set_xlabel("$r/a$")
a.set_ylabel(r"$\omega_{\mathrm{T}}$ [kHz]")
a.set_xlim(0, 1)
a.set_ylim(-5, 12)
a.legend(loc='lower right', ncol=2)
a.set_title("Toroidal rotation profiles")
setupplots.apply_formatter(f)
f.savefig("rotProf.pdf", bbox_inches='tight')
f.savefig("rotProf.pgf", bbox_inches='tight')
