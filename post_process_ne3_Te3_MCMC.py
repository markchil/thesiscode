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

# This script makes figure 3.43, which shows the result of attempting to infer
# D and V with ne and Te free to vary and local density measurements.

from __future__ import division

import os
cdir = os.getcwd()
os.chdir('/Users/markchilenski/src/bayesimp')

import bayesimp
import gptools
import scipy
import pymultinest

r = bayesimp.Run(
    shot=1101014006,
    version=25,
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
    params_true=[1.0, -10.0, 0.0, 0.0, 0.0, 0, 0, 0],
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
    free_ne=True,
    free_Te=True,
    normalize=False,
    local_time_res=6e-3,
    num_local_space=32,
    local_synth_noise=5e-2,
    use_line_integral=False,
    use_local=True,
    local_cs=[18,]
)

basename = os.path.abspath('../chains_%d_%d/c-' % (r.shot, r.version))
a = pymultinest.Analyzer(
    n_params=~(r.fixed_params).sum(),
    outputfiles_basename=basename
)
data = a.get_data()

os.chdir(cdir)

# Make plots:
import setupplots
setupplots.thesis_format()
import matplotlib.pyplot as plt
plt.ion()

mean, ci_l, ci_u = gptools.summarize_sampler(data[:, 2:], weights=data[:, 0])

# f = gptools.plot_sampler(
#     data[:, 2:],
#     weights=data[:, 0],
#     labels=r.free_param_names, #['$D$', '$V$'],
#     chain_alpha=1.0,
#     cutoff_weight=0.01,
#     cmap='plasma',
#     fixed_width=0.9 * setupplots.TEXTWIDTH,
#     suptitle='Posterior distribution of $D$ and $V$',
#     points=scipy.asarray([mean, r.params_true[~r.fixed_params]]),
#     bottom_sep=0.12,
#     suptitle_space=0.07,
#     ax_space=0.175,
#     label_fontsize=11,
#     chain_ticklabel_fontsize=5
# )
# f.savefig("samplerDVne3Te3.pdf", bbox_inches='tight')
# f.savefig("samplerDVne3Te3.pgf", bbox_inches='tight')

f, a = gptools.plot_sampler_cov(
    data[:, 2:],
    weights=data[:, 0],
    labels=[
        r'$D\vphantom{D_{T_{\text{e}}}}$',
        r'$V\vphantom{D_{T_{\text{e}}}}$',
        r'$u_{n_{\text{e}},1}\vphantom{D_{T_{\text{e}}}}$',
        r'$u_{n_{\text{e}},2}\vphantom{D_{T_{\text{e}}}}$',
        r'$u_{n_{\text{e}},3}\vphantom{D_{T_{\text{e}}}}$',
        r'$u_{T_{\text{e}},1}\vphantom{D_{T_{\text{e}}}}$',
        r'$u_{T_{\text{e}},2}\vphantom{D_{T_{\text{e}}}}$',
        r'$u_{T_{\text{e}},3}\vphantom{D_{T_{\text{e}}}}$'
    ],
    figsize=(0.8 * setupplots.TEXTWIDTH, 0.8 * setupplots.TEXTWIDTH),
    title='Posterior correlation matrix, local measurements',
    cbar_label=r'$\rho$'
)
a.title.set_position([0.5, 1.125])
f.savefig("corrDVne3Te3Local.pdf", bbox_inches='tight')
f.savefig("corrDVne3Te3Local.pgf", bbox_inches='tight')

# post_sum = r"""$D$ & {muD:.8f} & [ & {cilD:.8f} & {ciuD:.8f} & ]\\
# $V$ & {muV:.8f} & [ & {cilV:.8f} & {ciuV:.8f} & ]\\
# """.format(
#     muD=mean[0],
#     cilD=ci_l[0],
#     ciuD=ci_u[0],
#     muV=mean[1],
#     cilV=ci_l[1],
#     ciuV=ci_u[1]
# )
# with open('samplerDVne3Te3PostSum.tex', 'w') as f:
#     f.write(post_sum)
