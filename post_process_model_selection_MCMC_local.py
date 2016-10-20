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

# This script makes figure 3.53, which shows the inferred D and V profiles for
# various levels of model complexity.

from __future__ import division

import os
cdir = os.getcwd()

import bayesimp
import gptools
import scipy
import scipy.io
import profiletools
import os
import eqtools
import pymultinest
import cPickle as pkl
import itertools

num_eig = range(1, 5)

data = {}
D_mean = {}
D_std = {}
V_mean = {}
V_std = {}
free_param_names = {}
true_params = {}

for i in num_eig:
    os.chdir('/Users/markchilenski/src/bayesimp')
    
    num_eig_D = i
    num_eig_V = i
    
    # Linearly-spaced knots:
    knots_D = scipy.linspace(0, 1.05, num_eig_D + 1)[1:-1]
    knots_V = scipy.linspace(0, 1.05, num_eig_V + 1)[1:-1]
    knotflag = ''
    
    # Set up the actual STRAHL run:
    r = bayesimp.Run(
        shot=1101014006,
        version=24,
        time_1=1.165,
        time_2=1.265,
        Te_args=['--system', 'TS', 'GPC', 'GPC2'],
        ne_args=['--system', 'TS'],
        debug_plots=1,
        num_eig_D=num_eig_D,
        num_eig_V=num_eig_V,
        method='linterp',
        free_knots=False,
        use_scaling=False,
        use_shift=False,
        include_loweus=True,
        source_file='/Users/markchilenski/src/bayesimp/Caflx_delta_1165.dat',
        sort_knots=False,
        params_true=scipy.concatenate((
            [1.0,] * num_eig_D,
            scipy.linspace(0, -10, num_eig_V + 1)[1:],
            knots_D,
            knots_V,
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        )),
        time_spec=(  # For fast sampling:
            "    1.165     0.000050               1.01                      1\n"
            "    1.167     0.000075               1.1                       10\n"
            "    1.175     0.000100               1.50                      25\n"
            "    1.265     0.000100               1.50                      25\n"
        ),
        D_lb=0.0,
        D_ub=15.0,
        V_lb=-60.0,
        V_ub=20.0,
        V_lb_outer=-60.0,
        V_ub_outer=20.0,
        num_eig_ne=3,
        num_eig_Te=3,
        free_ne=False,
        free_Te=False,
        normalize=False,
        use_line_integral=False,
        use_local=True,
        local_time_res=6e-3,
        num_local_space=32,
        local_synth_noise=5e-2,
        local_cs=[18,]
    )
    
    basename = os.path.abspath('../chains_%d_%d/c-D%d-V%d-%s' % (r.shot, r.version, r.num_eig_D, r.num_eig_V, knotflag))
    a = pymultinest.Analyzer(
        n_params=~(r.fixed_params).sum(),
        outputfiles_basename=basename
    )
    data[i] = a.get_data()
    
    D_mean[i], D_std[i], V_mean[i], V_std[i] = r.compute_marginalized_DV(
        data[i][:, 2:],
        weights=data[i][:, 0]
    )
    
    free_param_names[i] = r.free_param_names
    true_params[i] = r.params_true[~r.fixed_params]

# This is the same for all cases:
D, V = r.eval_DV(r.params_true)

os.chdir(cdir)

# Make plots:
import setupplots
setupplots.thesis_format()
import matplotlib.pyplot as plt
plt.ion()

ls_cycle = itertools.cycle(['-', '--', '-.', ':'])

f_DV = plt.figure(figsize=(0.75 * setupplots.TEXTWIDTH, 2 * 0.75 * setupplots.TEXTWIDTH / 1.618))
a_D = f_DV.add_subplot(2, 1, 1)
a_V = f_DV.add_subplot(2, 1, 2, sharex=a_D)
a_D.plot(r.roa_grid_DV, D, label='true', linewidth=3 * setupplots.lw)
a_V.plot(r.roa_grid_DV, V, label='true', linewidth=3 * setupplots.lw)

for i in num_eig:
    # mean, ci_l, ci_u = gptools.summarize_sampler(data[i][:, 2:], weights=data[i][:, 0])
    
    # Marginals:
    # f = gptools.plot_sampler(
    #     data[i][:, 2:],
    #     weights=data[i][:, 0],
    #     labels=free_param_names[i],
    #     chain_alpha=1.0,
    #     cutoff_weight=0.001,
    #     cmap='plasma',
    #     fixed_width=0.9 * setupplots.TEXTWIDTH,
    #     suptitle='Posterior distribution of $D$ and $V$, %d coefficients' % (i,),
    #     points=scipy.asarray([mean, true_params[i]]),
    #     bottom_sep=0.12,
    #     suptitle_space=0.07,
    #     ax_space=0.175,
    #     label_fontsize=11,
    #     chain_ticklabel_fontsize=9
    # )
    # f.savefig("samplerDVD%dV%d.pdf" % (num_eig_D, num_eig_V), bbox_inches='tight')
    # f.savefig("samplerDVD3V3.pgf", bbox_inches='tight')
    
    # Marginalized D, V:
    ls = ls_cycle.next()
    gptools.univariate_envelope_plot(r.roa_grid_DV, D_mean[i], D_std[i], ax=a_D, label=str(i), linestyle=ls, envelopes=[1,])
    gptools.univariate_envelope_plot(r.roa_grid_DV, V_mean[i], V_std[i], ax=a_V, label=str(i), linestyle=ls, envelopes=[1,])

a_V.legend(loc='lower left')
a_D.set_ylabel(r"$D$ [$\si{m^2/s}$]")
a_V.set_ylabel(r"$V$ [m/s]")
a_V.set_xlabel('$r/a$')
a_D.set_title(r"Transport coefficient profiles for various levels of complexity")
plt.setp(a_D.get_xticklabels(), visible=False)
a_D.set_xlim(r.roa_grid_DV[0], r.roa_grid_DV[-1])
a_D.set_ylim(0, 2)
f_DV.subplots_adjust(hspace=0.1)
setupplots.apply_formatter(f_DV)
f_DV.savefig("DVModelSelectionBasic.pdf", bbox_inches='tight')
f_DV.savefig("DVModelSelectionBasic.pgf", bbox_inches='tight')
