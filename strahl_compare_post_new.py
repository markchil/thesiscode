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

# This script makes figure 2.41, which shows the D and V profiles with various
# profile fitting and uncertainty propagation schemes.

from __future__ import division
import setupplots
setupplots.thesis_format()
import scipy
from scipy.stats import norm, probplot
import matplotlib.pyplot as plt
import matplotlib.widgets as mplw
import matplotlib.gridspec as mplgs

import strahltools

plt.close('all')
plt.ion()

grid = None

res_MCMC = strahltools.STRAHLResult(filename='outputs/savepoint_backup_MCMC_141005', dims=[400, 100])
D_conv_MCMC, s_D_conv_MCMC, V_conv_MCMC, s_V_conv_MCMC = res_MCMC.form_convergence(grid=grid)

res_MAP = strahltools.STRAHLResult(filename='outputs/savepoint_backup_MAP_141005', dims=[400, 100])
D_conv_MAP, s_D_conv_MAP, V_conv_MAP, s_V_conv_MAP = res_MAP.form_convergence(grid=grid)

res_spline = strahltools.STRAHLResult(filename='outputs/savepoint_backup_spline', dims=[400, 99])
D_conv_spline, s_D_conv_spline, V_conv_spline, s_V_conv_spline = res_spline.form_convergence(grid=grid)

# Make plot of final form:
f = plt.figure(figsize=[0.75 * setupplots.TEXTWIDTH, 2 * 0.75 * setupplots.TEXTWIDTH / 1.618])
aD = f.add_subplot(2, 1, 1)
aD.set_title(r"Comparison of \textsc{strahl} results for different fitting schemes")
aD.plot(res_spline.grid, D_conv_spline[-1, :], 'k--', label='spline')
aD.fill_between(
    res_MCMC.grid,
    D_conv_spline[-1, :] - s_D_conv_spline[-1, :],
    D_conv_spline[-1, :] + s_D_conv_spline[-1, :],
    alpha=0.375,
    facecolor='k'
)
aD.plot(res_MAP.grid, D_conv_MAP[-1, :], 'r-.', label=r'\textsc{map}')
aD.fill_between(
    res_MAP.grid,
    D_conv_MAP[-1, :] - s_D_conv_MAP[-1, :],
    D_conv_MAP[-1, :] + s_D_conv_MAP[-1, :],
    alpha=0.375,
    facecolor='r'
)
aD.plot(res_MCMC.grid, D_conv_MCMC[-1, :], 'b', label=r'\textsc{mcmc}')
aD.fill_between(
    res_MCMC.grid,
    D_conv_MCMC[-1, :] - s_D_conv_MCMC[-1, :],
    D_conv_MCMC[-1, :] + s_D_conv_MCMC[-1, :],
    alpha=0.375,
    facecolor='b'
)
plt.setp(aD.get_xticklabels(), visible=False)
aD.set_ylabel('$D$ [$\si{m^2/s}$]')

aV = f.add_subplot(2, 1, 2, sharex=aD)
aV.plot(res_spline.grid, V_conv_spline[-1, :], 'k--', label='spline')
aV.fill_between(
    res_MCMC.grid,
    V_conv_spline[-1, :] - s_V_conv_spline[-1, :],
    V_conv_spline[-1, :] + s_V_conv_spline[-1, :],
    alpha=0.375,
    facecolor='k'
)
aV.plot(res_MAP.grid, V_conv_MAP[-1, :], 'r-.', label=r'\textsc{map}')
aV.fill_between(
    res_MAP.grid,
    V_conv_MAP[-1, :] - s_V_conv_MAP[-1, :],
    V_conv_MAP[-1, :] + s_V_conv_MAP[-1, :],
    alpha=0.375,
    facecolor='r'
)
aV.plot(res_MCMC.grid, V_conv_MCMC[-1, :], 'b', label=r'\textsc{mcmc}')
aV.fill_between(
    res_MCMC.grid,
    V_conv_MCMC[-1, :] - s_V_conv_MCMC[-1, :],
    V_conv_MCMC[-1, :] + s_V_conv_MCMC[-1, :],
    alpha=0.375,
    facecolor='b'
)
aV.set_xlabel('$r/a$')
aV.set_ylabel('$V$ [m/s]')

aD.axvline(x=0.339, color='m', ls=':', label='$q=1$')
aV.axvline(x=0.339, color='m', ls=':', label='$q=1$')

aD.legend(loc='upper left')

aD.set_xlim(0, 0.6001)

setupplots.apply_formatter(f)
f.canvas.draw()

f.savefig("DV.pgf", bbox_inches='tight')
f.savefig("DV.pdf", bbox_inches='tight')
