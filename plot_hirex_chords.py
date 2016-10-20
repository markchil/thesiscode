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

# This script makes figure 3.33, which shows the spectrometer lines of sight.

from __future__ import division

import os
cdir = os.getcwd()
os.chdir('/Users/markchilenski/src/bayesimp')

import bayesimp
import gptools
import scipy
import TRIPPy

r = bayesimp.Run(
    shot=1101014006,
    version=14,
    time_1=1.165,
    time_2=1.265,
    Te_args=['--system', 'TS', 'GPC', 'GPC2'],
    ne_args=['--system', 'TS'],
    debug_plots=1,
    num_eig_D=1,
    num_eig_V=1,
    method='linterp',
    free_knots=False,
    use_scaling=False,
    use_shift=False,
    include_loweus=True,
    source_file='/Users/markchilenski/src/bayesimp/Caflx_delta_1165.dat',
    sort_knots=True,
    params_true=scipy.concatenate((
        [1.0, -10.0],
        [1.0,] * 9,
        [0.0,] * 3,
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    )),
    time_spec=(  # For fast sampling:
        "    1.165     0.000050               1.01                      1\n"
        "    1.167     0.000075               1.1                       10\n"
        "    1.175     0.000100               1.50                      25\n"
        "    1.265     0.000100               1.50                      25\n"
    ),
    D_lb=0.0,
    D_ub=10.0,
    V_lb=-100.0,
    V_ub=10.0,
    V_lb_outer=-100.0,
    V_ub_outer=10.0,
    num_eig_ne=3,
    num_eig_Te=3,
    free_ne=False,
    free_Te=False,
    normalize=False,
    use_line_integral=True,
    use_local=False,
    hirex_time_res=6e-3,
    vuv_time_res=2e-3,
    synth_noises=[5e-2, 5e-2, 5e-2],
    signal_mask=[True, True, False]
)

os.chdir(cdir)

# Make plots:
import setupplots
setupplots.thesis_format()
import matplotlib
matplotlib.rcParams.update({'image.cmap': 'Blues_r'})
import matplotlib.pyplot as plt
plt.ion()
from TRIPPy.plot.pyplot import plotTokamak, plotLine
import matplotlib.lines as mlines

tokamak = TRIPPy.plasma.Tokamak(r.efit_tree)
hirex_sr_rays = [TRIPPy.beam.pos2Ray(p, tokamak) for p in r.signals[0].pos]
XEUS_beam = TRIPPy.beam.pos2Ray(bayesimp.XEUS_POS, tokamak)
LoWEUS_beam = TRIPPy.beam.pos2Ray(bayesimp.LOWEUS_POS, tokamak)

e = r.efit_tree
idx = 47

# Spectrometers:
f, sl = e.plotFlux(fill=False, lw=1, add_title=False)
sl.set_val(idx)
a = f.axes[0]
plt.sca(a)
red_line = mlines.Line2D([], [], color='r', label=r'\hirexsr')
for r in hirex_sr_rays:
    plotLine(r, pargs='r')
plotLine(XEUS_beam, pargs='b', lw=3, label=r'\xeus')
plotLine(LoWEUS_beam, pargs='g', lw=3, label=r'\loweus')

a.set_title("Spectrometer lines of sight")
a.legend(handles=[red_line,] + a.get_legend_handles_labels()[0], loc='upper right')

sl.ax.set_visible(False)
f.set_figwidth(0.5 * setupplots.TEXTWIDTH)
f.set_figheight(setupplots.TEXTWIDTH)
setupplots.apply_formatter(f)
f.canvas.draw()

f.savefig("specChords.pdf", bbox_inches='tight')
f.savefig("specChords.pgf", bbox_inches='tight')
