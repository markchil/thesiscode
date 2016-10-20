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

# This script makes figure 3.29, which shows the synthetic specturm obtained
# from the toy spectral model.

from __future__ import division

import os
cdir = os.getcwd()
os.chdir('/Users/markchilenski/src/bayesimp')

import bayesimp
import lines
import gptools
import scipy

r = bayesimp.Run(
    shot=1101014006,
    version=2,
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
    params_true=[1.0, -10.0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0.0, 0.0, 0.0, 0, 0, 0, 0, 0, 0, 0, 0],
    time_spec=(  # For fast sampling:
        "    1.165     0.000050               1.01                      1\n"
        "    1.167     0.000075               1.1                       10\n"
        "    1.175     0.000100               1.50                      25\n"
        "    1.265     0.000100               1.50                      25\n"
    ),
    D_lb=0.0,
    D_ub=30.0,
    V_lb=-200.0,
    V_ub=200.0,
    V_ub_outer=200.0,
    V_lb_outer=-200.0,
    num_eig_ne=5,
    num_eig_Te=3,
    free_ne=False,
    free_Te=False,
    normalize=False
)

cs_den, sqrtpsinorm, time, ne, Te = r.DV2cs_den(r.params_true)
em, lam, E, q, comment = lines.compute_lines(
    20, cs_den, ne, Te, atdata=r.atdata, sindat=r.sindat, PEC=r.PEC, full_return=True, E_thresh=1.0
)
# Convert to power at detector:
em *= r.filter_trans[None, :, None] * E[None, :, None] * scipy.constants.e * 1e3 * 1e6

# Make plots:
os.chdir(cdir)
import setupplots
setupplots.thesis_format()
import matplotlib.pyplot as plt
plt.ion()

i_time = 45
i_space = 0

f = plt.figure(figsize=(0.75 * setupplots.TEXTWIDTH, 0.75 * setupplots.TEXTWIDTH / 1.618))
a = f.add_subplot(1, 1, 1)

had_19 = False
had_18 = False
had_other = False

thresh = 1e-4
for em_v, E_v, q_v, c_v in zip(em[i_time, :, i_space], E, q, comment):
    # Only plot lines that matter:
    if em_v > thresh:
        l_v = scipy.constants.h * scipy.constants.c / (E_v * 1e3 * scipy.constants.e) * 1e9
        if q_v == 19:
            c = 'b'
            lw = 2 * setupplots.lw
        elif q_v == 18:
            c = 'g'
            lw = setupplots.lw
        else:
            c = 'r'
            lw = setupplots.lw
        if q_v == 19 and not had_19:
            label = 'H-like'
            had_19 = True
        elif q_v == 18 and not had_18:
            label = 'He-like'
            had_18 = True
        elif q_v < 18 and not had_other:
            label = 'lower states'
            had_other = False
        else:
            label = '_nolegend_'
        a.semilogy([l_v, l_v], [thresh, em_v], c, label=label, linewidth=lw)

a.set_xlabel(r'$\lambda$ [nm]')
a.set_ylabel(r'$T\epsilon$ [$\si{W/m^3}$]')
a.legend(loc='upper left', ncol=2)
a.set_title(r"Synthetic spectrum, $t-t_{\text{inj}}=\SI{%.0f}{ms}$, $r/a=0$" % (1e3 * (time[i_time] - r.time_1),))
setupplots.apply_formatter(f)
f.savefig("synth_spec.pdf", bbox_inches='tight')
f.savefig("synth_spec.pgf", bbox_inches='tight')
