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

# This script makes figures 2.19, 2.23, 2.27, 2.31 and 2.35, which show the
# magnetic equilibrium and diagnostic locations for various discharges.

from __future__ import division
import setupplots
setupplots.thesis_format()
import scipy
import scipy.io
import eqtools
import profiletools
import matplotlib
matplotlib.rcParams.update({'image.cmap': 'Blues_r'})
import matplotlib.pyplot as plt
plt.ion()

class Case(object):
    def __init__(self, shot, t_min, t_max, idx, plot_TCI=False, plot_ECE=False, plot_FRCECE=False):
        self.shot = shot
        self.t_min = t_min
        self.t_max = t_max
        self.idx = idx
        self.plot_TCI = plot_TCI
        self.plot_ECE = plot_ECE
        self.plot_FRCECE = plot_FRCECE

cases = [
    # Shot used for impurity transport analysis:
    Case(1101014006, 0.965, 1.365, 47, plot_ECE=True, plot_FRCECE=True),
    # Shot used for TCI analysis:
    Case(1120907032, 0.8, 0.9, 40, plot_TCI=True),
    # High density shot from second derivative analysis:
    Case(1120221011, 0.9, 1.2, 47, plot_ECE=True),
    # Low density shot from second derivative analysis:
    Case(1120221012, 0.9, 1.2, 47, plot_ECE=True),
    # Sawtooth-free shot from 2D fit:
    Case(1110329013, 1.0, 1.4, 47),
    # H-mode shot for mtanh fit:
    Case(1110201035, 1.35, 1.5, 65)
    # For talk:
    # Case(1110201018, 1.35, 1.5, 65, plot_ECE=True)
]

for c in cases:
    shot = c.shot
    t_min = c.t_min
    t_max = c.t_max
    idx = c.idx
    plot_TCI = c.plot_TCI
    plot_ECE = c.plot_ECE
    plot_FRCECE = c.plot_FRCECE
    
    e = eqtools.CModEFITTree(shot)
    p_CTS = profiletools.TeCTS(shot, t_min=t_min, t_max=t_max, efit_tree=e)
    p_CTS.time_average(weighted=True)
    p_ETS = profiletools.TeETS(shot, t_min=t_min, t_max=t_max, efit_tree=e)
    p_ETS.time_average(weighted=True)
    if plot_ECE:
        p_GPC = profiletools.TeGPC(shot, t_min=t_min, t_max=t_max, efit_tree=e)
        p_GPC.time_average()
        p_GPC2 = profiletools.TeGPC2(shot, t_min=t_min, t_max=t_max, efit_tree=e)
        p_GPC2.time_average()
    if plot_TCI:
        p_TCI = profiletools.neTCI_old(shot, t_min=t_min, t_max=t_max, efit_tree=e)
        p_TCI.time_average()
    
    f, sl = e.plotFlux(fill=False, lw=1, add_title=False)
    sl.set_val(idx)
    if plot_TCI:
        f.axes[0].axvline(p_TCI.transformed[0].X[0, 0, 0], color='m', label=r'\textsc{tci}')
        for t in p_TCI.transformed[1:]:
            f.axes[0].axvline(t.X[0, 0, 0], color='m')
    f.axes[0].plot(p_CTS.X[:, 0], p_CTS.X[:, 1], 'bs', label=r'\textsc{cts}')
    f.axes[0].plot(p_ETS.X[:, 0], p_ETS.X[:, 1], 'gs', label=r'\textsc{ets}')
    if plot_ECE:
        f.axes[0].plot(p_GPC.X[:, 0], scipy.zeros_like(p_GPC.y), 'c^', label=r'\textsc{gpc}')
        f.axes[0].plot(p_GPC2.X[:, 0], scipy.zeros_like(p_GPC2.y), 'm^', label=r'\textsc{gpc}2')
    if plot_FRCECE:
        sf = scipy.io.readsav('outputs/frcece%d.sav' % (shot,))
        R = sf['rf']
        Z = sf['zf']
        t = sf['t']
        mask = (t >= t_min) & (t <= t_max)
        R = scipy.mean(R[mask, :], axis=0)
        Z = scipy.mean(Z[mask, :], axis=0)
        f.axes[0].plot(R, Z, 'r^', label=r'\textsc{frcece}')
        
    f.axes[0].legend(loc='upper left', numpoints=1, ncol=2)
    f.axes[0].set_title(r'%d, $t=\SI{%.2f}{s}$' % (shot, e.getTimeBase()[idx]))
    
    sl.ax.set_visible(False)
    f.set_figwidth(0.5 * setupplots.TEXTWIDTH)
    f.set_figheight(setupplots.TEXTWIDTH)
    
    f.canvas.draw()
    setupplots.apply_formatter(f)
    f.savefig("diags_%d.pdf" % (shot,), bbox_inches='tight')
    f.savefig("diags_%d.pgf" % (shot,), bbox_inches='tight')
