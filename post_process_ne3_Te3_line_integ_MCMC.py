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

# This script makes figures 3.38, 3.39, 3.40, 3.41 and 3.42 which show the
# result of attempting to infer D and V with ne and Te free to vary.

from __future__ import division

import os
cdir = os.getcwd()
os.chdir('/Users/markchilenski/src/bayesimp')

import bayesimp
import gptools
import scipy
import pymultinest
import profiletools
import itertools

r = bayesimp.Run(
    shot=1101014006,
    version=16,
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
    free_ne=True,
    free_Te=True,
    normalize=False,
    use_line_integral=True,
    use_local=False,
    hirex_time_res=6e-3,
    vuv_time_res=2e-3,
    synth_noises=[5e-2, 5e-2, 5e-2],
    signal_mask=[True, True, False]
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
        r'$D$', '$V$', r'$u_{n_{\text{e}},1}$', r'$u_{n_{\text{e}},2}$', r'$u_{n_{\text{e}},3}$',
        r'$u_{T_{\text{e}},1}$', r'$u_{T_{\text{e}},2}$', r'$u_{T_{\text{e}},3}$'
    ],
    chain_alpha=1.0,
    cutoff_weight=0.01,
    cmap='plasma',
    fixed_width=6.5,
    suptitle=r'Posterior distribution of $D$, $V$, $n_{\text{e}}$, $T_{\text{e}}$',
    points=scipy.asarray([mean, r.params_true[~r.fixed_params]]),
    bottom_sep=0.1,
    suptitle_space=0.0,
    ax_space=0.175,
    label_fontsize=11,
    chain_ticklabel_fontsize=11,
    ticklabel_fontsize=11,
    max_hist_ticks=5,
    max_chain_ticks=5,
    hide_chain_ylabels=True
)
setupplots.apply_formatter(f)
f.savefig("samplerDVne3Te3.pdf", bbox_inches='tight')
f.savefig("samplerDVne3Te3.pgf", bbox_inches='tight')

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
a_D.set_title(r"\textcolor{MPLb}{True} and \textcolor{MPLg}{inferred} transport coefficient profiles" + "\nmarginalized over " + r"$n_{\text{e}}$, $T_{\text{e}}$")
plt.setp(a_D.get_xticklabels(), visible=False)
a_D.set_xlim(r.roa_grid_DV[0], r.roa_grid_DV[-1])
a_D.set_ylim(0, 2)
f.subplots_adjust(hspace=0.1)
setupplots.apply_formatter(f)
f.savefig("DVne3Te3.pdf", bbox_inches='tight')
f.savefig("DVne3Te3.pgf", bbox_inches='tight')

# Marginalized ne, Te:
ne_samps = r.run_data.ne_p.gp.draw_sample(
    r.run_data.ne_X,
    rand_vars=data[:, 4:7].T,
    method='eig',
    num_eig=3,
    mean=r.run_data.ne_res['mean_val'],
    cov=r.run_data.ne_res['cov']
)
mean_ne = profiletools.meanw(ne_samps.T, weights=data[:, 0], axis=0)
std_ne = profiletools.stdw(ne_samps.T, weights=data[:, 0], axis=0, ddof=1)

Te_samps = r.run_data.Te_p.gp.draw_sample(
    r.run_data.Te_X,
    rand_vars=data[:, 7:].T,
    method='eig',
    num_eig=3,
    mean=r.run_data.Te_res['mean_val'],
    cov=r.run_data.Te_res['cov']
)
mean_Te = profiletools.meanw(Te_samps.T, weights=data[:, 0], axis=0)
std_Te = profiletools.stdw(Te_samps.T, weights=data[:, 0], axis=0, ddof=1)

f = plt.figure(figsize=(0.75 * setupplots.TEXTWIDTH, 2 * 0.75 * setupplots.TEXTWIDTH / 1.618))
a_ne = f.add_subplot(2, 1, 1)
gptools.univariate_envelope_plot(
    r.run_data.ne_X,
    r.run_data.ne_res['mean_val'],
    r.run_data.ne_res['std_val'],
    ax=a_ne,
    color='b',
    linestyle='-',
    label='prior'
)
gptools.univariate_envelope_plot(
    r.run_data.ne_X,
    mean_ne,
    std_ne,
    ax=a_ne,
    color='g',
    linestyle='--',
    label='posterior'
)

a_Te = f.add_subplot(2, 1, 2, sharex=a_ne)
gptools.univariate_envelope_plot(
    r.run_data.Te_X,
    r.run_data.Te_res['mean_val'],
    r.run_data.Te_res['std_val'],
    ax=a_Te,
    color='b',
    linestyle='-',
    label='prior'
)
gptools.univariate_envelope_plot(
    r.run_data.Te_X,
    mean_Te,
    std_Te,
    ax=a_Te,
    color='g',
    linestyle='--',
    label='posterior'
)

plt.setp(a_ne.get_xticklabels(), visible=False)
a_ne.set_ylabel(r"$n_{\text{e}}$ [$\SI{e20}{m^{-3}}$]")
a_ne.set_title(r"$n_{\text{e}}$")
a_ne.set_ylim(bottom=0.0)
a_ne.set_xlim(right=1.05)

a_Te.set_xlabel(r"$r/a$")
a_Te.set_ylabel(r"$T_{\text{e}}$ [keV]")
a_Te.set_title(r"$T_{\text{e}}$")
a_Te.legend(loc='lower left')
a_Te.set_ylim(bottom=0.0)

setupplots.apply_formatter(f)
f.savefig("ne3Te3PostProf.pdf", bbox_inches='tight')
f.savefig("ne3Te3PostProf.pgf", bbox_inches='tight')

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
    title='Posterior correlation matrix',
    cbar_label=r'$\rho$'
)
a.title.set_position([0.5, 1.125])
# setupplots.apply_formatter(f)
f.savefig("corrDVne3Te3.pdf", bbox_inches='tight')
f.savefig("corrDVne3Te3.pgf", bbox_inches='tight')

post_sum = r"""Free $n_{{\text{{e}}}}$, $T_{{\text{{e}}}}$ & $D$ [$\si{{m^2/s}}$] & {muD:.8f} & [ & {cilD:.8f} & {ciuD:.8f} & ]\\
& $V$ [m/s] & {muV:.8f} & [ & {cilV:.8f} & {ciuV:.8f} & ]\\
""".format(
    muD=mean[0],
    cilD=ci_l[0],
    ciuD=ci_u[0],
    muV=mean[1],
    cilV=ci_l[1],
    ciuV=ci_u[1]
)
with open('samplerDVne3Te3PostSum.tex', 'w') as f:
    f.write(post_sum)

# Plot first three eigenvectors:
f = plt.figure(figsize=(0.75 * setupplots.TEXTWIDTH, 2 * 0.75 * setupplots.TEXTWIDTH / 1.618))
a_ne = f.add_subplot(2, 1, 1)
a_Te = f.add_subplot(2, 1, 2, sharex=a_ne)
ls_cycle = itertools.cycle(['b-', 'g--', 'r:'])
for k in range(0, 3):
    ls = ls_cycle.next()
    rand_vars = scipy.zeros((3, 1), dtype=float)
    rand_vars[k] = 1.0
    ne = r.run_data.ne_p.gp.draw_sample(
        r.run_data.ne_X,
        rand_vars=rand_vars,
        method='eig',
        num_eig=3,
        mean=r.run_data.ne_res['mean_val'],
        cov=r.run_data.ne_res['cov']
    )
    Te = r.run_data.Te_p.gp.draw_sample(
        r.run_data.Te_X,
        rand_vars=rand_vars,
        method='eig',
        num_eig=3,
        mean=r.run_data.Te_res['mean_val'],
        cov=r.run_data.Te_res['cov']
    )
    a_ne.plot(r.run_data.ne_X, ne[:, 0] - r.run_data.ne_res['mean_val'], ls, label=k + 1)
    a_Te.plot(r.run_data.Te_X, Te[:, 0] - r.run_data.Te_res['mean_val'], ls, label=k + 1)

plt.setp(a_ne.get_xticklabels(), visible=False)
a_ne.set_ylabel(r'$\lambda^{1/2}q$ [$\SI{e20}{m^{-3}}$]')
a_Te.set_ylabel(r"$\lambda^{1/2}q$ [keV]")
a_ne.set_title(r"$n_{\text{e}}$")
a_Te.set_title(r"$T_{\text{e}}$")
a_Te.set_xlabel("$r/a$")
a_Te.legend(loc='lower right')
f.suptitle(r"$n_{\text{e}}$, $T_{\text{e}}$ eigenvectors")
setupplots.apply_formatter(f)
f.savefig("neTeEig.pdf", bbox_inches='tight')
f.savefig("neTeEig.pgf", bbox_inches='tight')
