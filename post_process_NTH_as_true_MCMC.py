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

# This script makes figures 3.55, 3.57, 3.58, 3.59, C.6, C.7, C.8, C.9, C.10,
# C.11, C.12 and C.13, which show the results of inferring D and V profiles for
# various levels of complexity.

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

os.chdir('/Users/markchilenski/src/bayesimp')

# Import the NTH profiles to use as the truth data:
roa_grid = scipy.linspace(0, 1.05, 100)
e = eqtools.CModEFITTree(1101014006)
sqrtpsinorm_grid = e.roa2psinorm(roa_grid, (1.165 + 1.265) / 2.0, sqrt=True)
filename = 'results_from_NTH_code/savepoint_backup_MAP_141005'
f = scipy.io.readsav(filename)
chisqd = scipy.asarray(f.chisqd, dtype=float)
roasave = scipy.asarray(f.roasave, dtype=float)
D_results = scipy.asarray(f.dvresults[:, :, 0], dtype=float)
V_results = scipy.asarray(f.dvresults[:, :, 1], dtype=float)
# Filter out the bad fits:
valid = (chisqd != 0.0) & (chisqd != 1.0) & (chisqd != -999)
chisqd = chisqd[valid]
roasave = roasave[valid, :]
D_results = D_results[valid, :]
V_results = V_results[valid, :]
D_out = scipy.zeros((len(chisqd), len(roa_grid)))
V_out = scipy.zeros((len(chisqd), len(roa_grid)))
for i in range(0, len(chisqd)):
    D_out[i] = scipy.interpolate.InterpolatedUnivariateSpline(roasave[i, :], D_results[i, :], k=1)(roa_grid)
    V_out[i] = scipy.interpolate.InterpolatedUnivariateSpline(roasave[i, :], V_results[i, :], k=1)(roa_grid)

weights = chisqd.min() / chisqd
explicit_D = profiletools.meanw(D_out, axis=0, weights=weights)
explicit_V = profiletools.meanw(V_out, axis=0, weights=weights)

num_eig = range(1, 8)

data = {}
D_mean = {}
D_std = {}
V_mean = {}
V_std = {}
free_param_names = {}
sig = {}

for i in num_eig:
    os.chdir('/Users/markchilenski/src/bayesimp')
    
    num_eig_D = i
    num_eig_V = i
    
    # Linearly-spaced knots:
    knots_D = scipy.linspace(0, 1.05, num_eig_D + 1)[1:-1]
    knots_V = scipy.linspace(0, 1.05, num_eig_V + 1)[1:-1]
    knotflag = ''
    
    free_knots = False
    
    # Set up the actual STRAHL run:
    r = bayesimp.Run(
        shot=1101014006,
        version=22,
        time_1=1.165,
        time_2=1.265,
        Te_args=['--system', 'TS', 'GPC', 'GPC2'],
        ne_args=['--system', 'TS'],
        debug_plots=1,
        num_eig_D=num_eig_D,
        num_eig_V=num_eig_V,
        method='linterp',
        free_knots=free_knots,
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
        use_line_integral=True,
        use_local=False,
        hirex_time_res=6e-3,
        vuv_time_res=2e-3,
        synth_noises=[5e-2, 5e-2, 5e-2],
        signal_mask=[True, True, False],
        explicit_D=explicit_D,
        explicit_D_grid=sqrtpsinorm_grid,
        explicit_V=explicit_V,
        explicit_V_grid=sqrtpsinorm_grid
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
    
    # Compute the signals from the marginalized profiles:
    sig[i] = r.DV2sig(
        params=r.params_true,
        explicit_D=D_mean[i],
        explicit_D_grid=scipy.sqrt(r.psinorm_grid_DV),
        explicit_V=V_mean[i],
        explicit_V_grid=scipy.sqrt(r.psinorm_grid_DV)
    )

# Optimal knots:
os.chdir('/Users/markchilenski/src/bayesimp')

num_eig_D = 5
num_eig_V = 5

# Near-optimal knots:
knots_D = scipy.asarray([0.3, 0.45, 0.6, 0.8], dtype=float)
knots_V = scipy.asarray([0.3, 0.45, 0.6, 0.8], dtype=float)

knotflag = 'opt_knots-'

free_knots = False

# Set up the actual STRAHL run:
r = bayesimp.Run(
    shot=1101014006,
    version=22,
    time_1=1.165,
    time_2=1.265,
    Te_args=['--system', 'TS', 'GPC', 'GPC2'],
    ne_args=['--system', 'TS'],
    debug_plots=1,
    num_eig_D=num_eig_D,
    num_eig_V=num_eig_V,
    method='linterp',
    free_knots=free_knots,
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
    use_line_integral=True,
    use_local=False,
    hirex_time_res=6e-3,
    vuv_time_res=2e-3,
    synth_noises=[5e-2, 5e-2, 5e-2],
    signal_mask=[True, True, False],
    explicit_D=explicit_D,
    explicit_D_grid=sqrtpsinorm_grid,
    explicit_V=explicit_V,
    explicit_V_grid=sqrtpsinorm_grid
)

basename = os.path.abspath('../chains_%d_%d/c-D%d-V%d-%s' % (r.shot, r.version, r.num_eig_D, r.num_eig_V, knotflag))
a = pymultinest.Analyzer(
    n_params=~(r.fixed_params).sum(),
    outputfiles_basename=basename
)
data['opt'] = a.get_data()

D_mean['opt'], D_std['opt'], V_mean['opt'], V_std['opt'] = r.compute_marginalized_DV(
    data['opt'][:, 2:],
    weights=data['opt'][:, 0]
)

# Bad knots:
os.chdir('/Users/markchilenski/src/bayesimp')

num_eig_D = 5
num_eig_V = 5

# Purposefully bad knots for 5+5:
knots_D = scipy.asarray([0.12, 0.4, 0.8, 1.0], dtype=float)
knots_V = scipy.asarray([0.12, 0.4, 0.8, 1.0], dtype=float)

knotflag = 'bad_knots-'

free_knots = False

# Set up the actual STRAHL run:
r = bayesimp.Run(
    shot=1101014006,
    version=22,
    time_1=1.165,
    time_2=1.265,
    Te_args=['--system', 'TS', 'GPC', 'GPC2'],
    ne_args=['--system', 'TS'],
    debug_plots=1,
    num_eig_D=num_eig_D,
    num_eig_V=num_eig_V,
    method='linterp',
    free_knots=free_knots,
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
    use_line_integral=True,
    use_local=False,
    hirex_time_res=6e-3,
    vuv_time_res=2e-3,
    synth_noises=[5e-2, 5e-2, 5e-2],
    signal_mask=[True, True, False],
    explicit_D=explicit_D,
    explicit_D_grid=sqrtpsinorm_grid,
    explicit_V=explicit_V,
    explicit_V_grid=sqrtpsinorm_grid
)

basename = os.path.abspath('../chains_%d_%d/c-D%d-V%d-%s' % (r.shot, r.version, r.num_eig_D, r.num_eig_V, knotflag))
a = pymultinest.Analyzer(
    n_params=~(r.fixed_params).sum(),
    outputfiles_basename=basename
)
data['bad'] = a.get_data()

D_mean['bad'], D_std['bad'], V_mean['bad'], V_std['bad'] = r.compute_marginalized_DV(
    data['bad'][:, 2:],
    weights=data['bad'][:, 0]
)

os.chdir(cdir)

# Make plots:
import setupplots
setupplots.thesis_format()
import matplotlib.pyplot as plt
plt.ion()

ls_cycle = itertools.cycle(['-', '--', '-.', ':'])

f_DV = plt.figure(figsize=(0.75 * setupplots.TEXTWIDTH, 2 * 0.75 * setupplots.TEXTWIDTH / 1.618))
a_D = f_DV.add_subplot(2, 1, 1)
a_D.plot(roa_grid, explicit_D, label='true', linewidth=3 * setupplots.lw)
a_V = f_DV.add_subplot(2, 1, 2, sharex=a_D)
a_V.plot(roa_grid, explicit_V, label='true', linewidth=3 * setupplots.lw)

for i in num_eig:
    mean, ci_l, ci_u = gptools.summarize_sampler(data[i][:, 2:], weights=data[i][:, 0])
    
    # Marginals:
    f = gptools.plot_sampler(
        data[i][:, 2:],
        weights=data[i][:, 0],
        labels=['$D_{%d}$' % (j + 1,) for j in range(0, i)] + ['$V_{%d}$' % (j + 1,) for j in range(0, i)],
        chain_alpha=1.0,
        cutoff_weight=0.001,
        cmap='plasma',
        fixed_width=setupplots.TEXTWIDTH if i < 4 else 6.5,
        suptitle='Posterior distribution of $D$ and $V$, %d+%d' % (i, i),
        # points=mean,
        bottom_sep=0.12 if i < 4 else 0.095,
        suptitle_space=0.05 if i < 4 else 0.0,
        ax_space=0.175,
        label_fontsize=11,
        chain_ticklabel_fontsize=11,
        ticklabel_fontsize=11,
        max_chain_ticks=6 if i < 4 else 3,
        max_hist_ticks=6 if i < 4 else 4,
        hide_chain_ylabels=True,
    )
    setupplots.apply_formatter(f)
    f.savefig("samplerDVD%dV%d.pdf" % (i, i), bbox_inches='tight')
    f.savefig("samplerDVD%dV%d.pgf" % (i, i), bbox_inches='tight')
    
    # Marginalized D, V:
    ls = ls_cycle.next()
    gptools.univariate_envelope_plot(r.roa_grid_DV, D_mean[i], D_std[i], ax=a_D, label=str(i), linestyle=ls, envelopes=[1,])
    gptools.univariate_envelope_plot(r.roa_grid_DV, V_mean[i], V_std[i], ax=a_V, label=str(i), linestyle=ls, envelopes=[1,])

# Opt marginals:
mean, ci_l, ci_u = gptools.summarize_sampler(data['opt'][:, 2:], weights=data['opt'][:, 0])
f = gptools.plot_sampler(
    data['opt'][:, 2:],
    weights=data['opt'][:, 0],
    labels=['$D_{%d}$' % (j + 1,) for j in range(0, 5)] + ['$V_{%d}$' % (j + 1,) for j in range(0, 5)],
    chain_alpha=1.0,
    cutoff_weight=0.001,
    cmap='plasma',
    fixed_width=6.5,
    suptitle='Posterior distribution of $D$ and $V$, %d+%d, near-optimal' % (5, 5),
    # points=mean,
    bottom_sep=0.1,
    suptitle_space=0.0,
    ax_space=0.175,
    label_fontsize=11,
    chain_ticklabel_fontsize=11,
    ticklabel_fontsize=11,
    max_chain_ticks=4,
    max_hist_ticks=4,
    hide_chain_ylabels=True,
)
setupplots.apply_formatter(f)
f.savefig("samplerDVD%dV%dopt.pdf" % (5, 5), bbox_inches='tight')
f.savefig("samplerDVD%dV%dopt.pgf" % (5, 5), bbox_inches='tight')

# Bad marginals:
mean, ci_l, ci_u = gptools.summarize_sampler(data['bad'][:, 2:], weights=data['bad'][:, 0])
f = gptools.plot_sampler(
    data['bad'][:, 2:],
    weights=data['bad'][:, 0],
    labels=['$D_{%d}$' % (j + 1,) for j in range(0, 5)] + ['$V_{%d}$' % (j + 1,) for j in range(0, 5)],
    chain_alpha=1.0,
    cutoff_weight=0.001,
    cmap='plasma',
    fixed_width=6.5,
    suptitle='Posterior distribution of $D$ and $V$, %d+%d, bad' % (5, 5),
    # points=mean,
    bottom_sep=0.1,
    suptitle_space=0.0,
    ax_space=0.175,
    label_fontsize=11,
    chain_ticklabel_fontsize=11,
    ticklabel_fontsize=11,
    max_chain_ticks=4,
    max_hist_ticks=4,
    hide_chain_ylabels=True,
)
setupplots.apply_formatter(f)
f.savefig("samplerDVD%dV%dbad.pdf" % (5, 5), bbox_inches='tight')
f.savefig("samplerDVD%dV%dbad.pgf" % (5, 5), bbox_inches='tight')

a_V.legend(loc='lower left', bbox_to_anchor=(1.025, 0.0))
a_D.set_ylabel(r"$D$ [$\si{m^2/s}$]")
a_V.set_ylabel(r"$V$ [m/s]")
a_V.set_xlabel('$r/a$')
a_D.set_title(r"Transport coefficient profiles for various levels of complexity")
plt.setp(a_D.get_xticklabels(), visible=False)
a_D.set_xlim(r.roa_grid_DV[0], r.roa_grid_DV[-1])
a_D.set_ylim(bottom=0.0, top=12)
a_V.set_ylim(bottom=-50, top=10)
f_DV.subplots_adjust(hspace=0.1)
setupplots.apply_formatter(f_DV)
f_DV.savefig("DVcomplex.pdf", bbox_inches='tight')
f_DV.savefig("DVcomplex.pgf", bbox_inches='tight')

# Good versus bad knots:
f_DV = plt.figure(figsize=(0.75 * setupplots.TEXTWIDTH, 2 * 0.75 * setupplots.TEXTWIDTH / 1.618))
a_D = f_DV.add_subplot(2, 1, 1)
a_D.plot(roa_grid, explicit_D, label='true', linewidth=3 * setupplots.lw)
a_V = f_DV.add_subplot(2, 1, 2, sharex=a_D)
a_V.plot(roa_grid, explicit_V, label='true', linewidth=3 * setupplots.lw)

gptools.univariate_envelope_plot(r.roa_grid_DV, D_mean[5], D_std[5], ax=a_D, label='linearly-spaced', linestyle='--', color='b', envelopes=[1,])
gptools.univariate_envelope_plot(r.roa_grid_DV, V_mean[5], V_std[5], ax=a_V, label='linearly-spaced', linestyle='--', color='b', envelopes=[1,])

gptools.univariate_envelope_plot(r.roa_grid_DV, D_mean['opt'], D_std['opt'], ax=a_D, label='near-optimal', linestyle='-.', color='g', envelopes=[1,])
gptools.univariate_envelope_plot(r.roa_grid_DV, V_mean['opt'], V_std['opt'], ax=a_V, label='near-optimal', linestyle='-.', color='g', envelopes=[1,])

gptools.univariate_envelope_plot(r.roa_grid_DV, D_mean['bad'], D_std['bad'], ax=a_D, label='bad', linestyle=':', color='r', envelopes=[1,])
gptools.univariate_envelope_plot(r.roa_grid_DV, V_mean['bad'], V_std['bad'], ax=a_V, label='bad', linestyle=':', color='r', envelopes=[1,])

a_V.legend(loc='lower left', ncol=2)
a_D.set_ylabel(r"$D$ [$\si{m^2/s}$]")
a_V.set_ylabel(r"$V$ [m/s]")
a_V.set_xlabel('$r/a$')
a_D.set_title(r"Effect of knot locations on five coefficient case")
plt.setp(a_D.get_xticklabels(), visible=False)
a_D.set_xlim(r.roa_grid_DV[0], r.roa_grid_DV[-1])
a_D.set_ylim(bottom=0.0, top=10)
a_V.set_ylim(bottom=-60, top=10)
f_DV.subplots_adjust(hspace=0.1)
setupplots.apply_formatter(f_DV)
f_DV.savefig("DVcomplexKnots.pdf", bbox_inches='tight')
f_DV.savefig("DVcomplexKnots.pgf", bbox_inches='tight')

# Make plot of signals:
ls_cycle = itertools.cycle(['-', '--', '-.', ':'])
color_cycle = itertools.cycle(['g', 'r', 'c', 'm', 'y', 'k', 'b'])
f = plt.figure(figsize=(0.75 * setupplots.TEXTWIDTH, 2 * 0.75 * setupplots.TEXTWIDTH / 1.618))
a_core = f.add_subplot(2, 1, 1)
a_core.errorbar(r.signals[0].t, r.signals[0].y[:, 0] * 1e9, yerr=r.signals[0].std_y[:, 0] * 1e9, fmt='o', ms=1.5 * setupplots.ms)
a_edge = f.add_subplot(2, 1, 2, sharex=a_core)
a_edge.errorbar(r.signals[0].t, r.signals[0].y[:, -1] * 1e9, yerr=r.signals[0].std_y[:, -1] * 1e9, fmt='o', ms=1.5 * setupplots.ms)
for i in num_eig:
    ls = color_cycle.next() + ls_cycle.next()
    a_core.plot(r.signals[0].t, sig[i][0][:, 0] * 1e9, ls, label=str(i))
    a_edge.plot(r.signals[0].t, sig[i][0][:, -1] * 1e9, ls, label=str(i))
a_edge.legend(loc='lower left', bbox_to_anchor=(1.025, 0.0))
plt.setp(a_core.get_xticklabels(), visible=False)
a_core.set_title(r"Core")
a_edge.set_title("Edge")
a_edge.set_xlabel(r"$t - t_{\text{inj}}$ [s]")
a_core.set_ylabel(r"$s$ [\textsc{au}]")
a_edge.set_ylabel(r"$s$ [\textsc{au}]")
a_core.set_ylim(bottom=0.0)
a_edge.set_ylim(bottom=0.0)
setupplots.apply_formatter(f)
f.savefig('signalCompare.pdf', bbox_inches='tight')
f.savefig('signalCompare.pgf', bbox_inches='tight')
