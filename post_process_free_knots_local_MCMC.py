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

# This script makes figures 3.36 and 3.37 which show the results of trying to
# infer D and V with free knots.

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
    version=18,
    time_1=1.165,
    time_2=1.265,
    Te_args=['--system', 'TS', 'GPC', 'GPC2'],
    ne_args=['--system', 'TS'],
    debug_plots=1,
    num_eig_D=2,
    num_eig_V=2,
    method='linterp',
    free_knots=True,
    use_scaling=False,
    use_shift=False,
    include_loweus=True,
    source_file='/Users/markchilenski/src/bayesimp/Caflx_delta_1165.dat',
    sort_knots=True,
    params_true=[1.0, 1.0, -5.0, -10.0, 1.05 / 2.0, 1.05 / 2.0, 0.0, 0.0, 0.0, 0, 0, 0],
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
    local_time_res=6e-3,
    num_local_space=5,
    local_synth_noise=5e-2,
    use_line_integral=False,
    use_local=True
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

f = gptools.plot_sampler(
    data[:, 2:],
    weights=data[:, 0],
    labels=[
        r'$D_1$', r'$D_2$',
        '$V_1$', '$V_2$',
        '$t_D$', '$t_V$'
    ],
    chain_alpha=1.0,
    cutoff_weight=0.01,
    cmap='plasma',
    fixed_width=6.5,
    suptitle=r'Posterior distribution of $D$ and $V$' + '\n1 free knot',
    points=scipy.asarray([mean, r.params_true[~r.fixed_params]]),
    bottom_sep=0.09,
    suptitle_space=0.0,
    ax_space=0.175,
    label_fontsize=11,
    chain_ticklabel_fontsize=11,
    ticklabel_fontsize=11,
    max_hist_ticks=5,
    hide_chain_ylabels=True
)
setupplots.apply_formatter(f)
f.savefig("samplerDVFreeKnots.pdf", bbox_inches='tight')
f.savefig("samplerDVFreeKnots.pgf", bbox_inches='tight')

# Marginalized D, V:
D_mean, D_std, V_mean, V_std = r.compute_marginalized_DV(
    data[:, 2:],
    weights=data[:, 0],
    cutoff_weight=0.01
)
D, V = r.eval_DV(r.params_true)
f = plt.figure(figsize=(0.75 * setupplots.TEXTWIDTH, 2 * 0.75 * setupplots.TEXTWIDTH / 1.618))
a_D = f.add_subplot(2, 1, 1)
a_D.plot(r.roa_grid_DV, D, label='true')
gptools.univariate_envelope_plot(r.roa_grid_DV, D_mean, D_std, ax=a_D, label='inferred', linestyle='--')
a_V = f.add_subplot(2, 1, 2, sharex=a_D)
a_V.plot(r.roa_grid_DV, V, label='true')
gptools.univariate_envelope_plot(r.roa_grid_DV, V_mean, V_std, ax=a_V, label='inferred', linestyle='--')
a_V.legend(loc='lower left')
a_D.set_ylabel(r"$D$ [$\si{m^2/s}$]")
a_V.set_ylabel(r"$V$ [m/s]")
a_V.set_xlabel('$r/a$')
a_D.set_title(r"\textcolor{MPLb}{True} and \textcolor{MPLg}{inferred} transport coefficient profiles" + "\n1 free knot")
plt.setp(a_D.get_xticklabels(), visible=False)
a_D.set_xlim(r.roa_grid_DV[0], r.roa_grid_DV[-1])
a_D.set_ylim(0, 2)
f.subplots_adjust(hspace=0.1)
setupplots.apply_formatter(f)
f.savefig("DVFreeKnots.pdf", bbox_inches='tight')
f.savefig("DVFreeKnots.pgf", bbox_inches='tight')

post_sum = r"""One free knot & $D$ & {muD:.8f} & [ & {cilD:.8f} & {ciuD:.8f} & ]\\
& $V$ & {muV:.8f} & [ & {cilV:.8f} & {ciuV:.8f} & ]\\
""".format(
    muD=mean[0],
    cilD=ci_l[0],
    ciuD=ci_u[0],
    muV=mean[3],
    cilV=ci_l[3],
    ciuV=ci_u[3]
)
with open('samplerDVFreeKnotPostSum.tex', 'w') as f:
    f.write(post_sum)
