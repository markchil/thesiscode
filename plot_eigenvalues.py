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

# This script makes figure 3.30, which shows the eigenvalue spectrum of the ne
# and Te profiles used for impurity transport analysis.

from __future__ import division

import os
cdir = os.getcwd()
os.chdir('/Users/markchilenski/src/bayesimp')

import bayesimp
import lines
import gptools
import scipy
import sys

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

lam_ne, Q_ne = scipy.linalg.eigh(r.run_data.ne_res['cov'] + 1e3 * sys.float_info.epsilon * scipy.eye(len(r.run_data.ne_X)))
lam_Te, Q_Te = scipy.linalg.eigh(r.run_data.Te_res['cov'] + 1e3 * sys.float_info.epsilon * scipy.eye(len(r.run_data.ne_X)))

os.chdir(cdir)

import setupplots
setupplots.thesis_format()
import matplotlib.pyplot as plt
plt.ion()

f = plt.figure(figsize=(0.75 * setupplots.TEXTWIDTH, 2 * 0.75 * setupplots.TEXTWIDTH / 1.618))
a_ne = f.add_subplot(2, 1, 1)
a_ne.semilogy(scipy.sqrt(lam_ne)[::-1], 'o')
plt.setp(a_ne.get_xticklabels(), visible=False)
a_ne.set_ylabel(r'$\lambda^{1/2}$ [$\SI{e20}{m^{-3}}$]')
a_ne.set_title(r'$n_{\text{e}}$')

a_Te = f.add_subplot(2, 1, 2)
a_Te.semilogy(scipy.sqrt(lam_Te)[::-1], 'o')
a_Te.set_xlabel('eigenvalue index')
a_Te.set_ylabel(r'$\lambda^{1/2}$ [$\si{keV}$]')
a_Te.set_title(r'$T_{\text{e}}$')
setupplots.apply_formatter(f)
f.savefig("profEig.pdf", bbox_inches='tight')
f.savefig("profEig.pgf", bbox_inches='tight')
