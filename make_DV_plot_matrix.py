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

# This script makes figures 3.2, 3.3, 3.4 and 3.5, which show the general
# properties of the impurity density following an impurity injection.

from __future__ import division

import os
cdir = os.getcwd()
os.chdir('/Users/markchilenski/src/bayesimp')

import bayesimp
import gptools
import scipy
import cPickle as pkl
import multiprocessing
from emcee.interruptible_pool import InterruptiblePool as iPool
import itertools

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
    use_scaling=True,
    include_loweus=True,
    source_file='/Users/markchilenski/src/bayesimp/Caflx_delta_1165.dat',
    sort_knots=True,
    params_true=[1.0, -10.0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0.0, 0.0, 0.0, 0, 0, 0, 0, 0, 0, 0, 0],
    time_spec=(  # For dense sampling of cs_den properties:
        "    {time_1:.5f}     0.000010               1.00                      1\n"
        "    {time_2:.5f}     0.000010               1.00                      1\n"
    )
)

rr_true = bayesimp._ComputeCSDenEval(r)(r.params_true[0:2])

# Get data for cs_den plots:
p_vals = [
    [0.1, 0.0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0.0, 0.0, 0.0],
    [1.0, 0.0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0.0, 0.0, 0.0],
    [2.0, 0.0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0.0, 0.0, 0.0],
    [0.1, -10.0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0.0, 0.0, 0.0],
    [1.0, -10.0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0.0, 0.0, 0.0],
    [2.0, -10.0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0.0, 0.0, 0.0],
    [0.1, -20.0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0.0, 0.0, 0.0],
    [1.0, -20.0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0.0, 0.0, 0.0],
    [2.0, -20.0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0.0, 0.0, 0.0]
]
n_vals = []
for p in p_vals:
    cs_den, sqrtpsinorm, time, ne, Te = r.DV2cs_den(p)
    n_vals.append(cs_den.sum(axis=1))
roa = r.efit_tree.psinorm2roa(sqrtpsinorm**2.0, (r.time_1 + r.time_2) / 2.0)
idx, = scipy.where(sqrtpsinorm <= 1.0)
idx = idx[-1]

# Evaluate N:
n = n_vals[4]
volnorm_grid = r.efit_tree.psinorm2volnorm(
    sqrtpsinorm**2.0,
    (r.time_1 + r.time_2) / 2.0
)
V = r.efit_tree.psinorm2v(1.0, (r.time_1 + r.time_2) / 2.0)
mask = ~scipy.isnan(volnorm_grid)
volnorm_grid = volnorm_grid[mask]
nn = n[:, mask] * 1e6 # convert to m^-3.
# Use the trapezoid rule:
N = V * 0.5 * ((volnorm_grid[1:] - volnorm_grid[:-1]) * (nn[:, 1:] + nn[:, :-1])).sum(axis=1)

t_mask = (time > rr_true.t_N_peak + 0.01) & (N > 0.0) & (~scipy.isinf(N)) & (~scipy.isnan(N))
X = scipy.hstack((scipy.ones((t_mask.sum(), 1)), scipy.atleast_2d(time[t_mask]).T))
theta, dum1, dum2, dum3 = scipy.linalg.lstsq(X.T.dot(X), X.T.dot(scipy.log(N[t_mask])))

os.chdir(cdir)

# Write output file:
class Result(object):
    def __init__(self):
        pass

re = Result()
re.p_vals = p_vals
re.n_vals = n_vals
re.roa = roa
re.time = time
re.N = N
re.theta = theta
re.t_mask = t_mask
re.n = n
re.rr_true = rr_true

with open("DV_plot_matrix.pkl", 'wb') as pf:
    pkl.dump(re, pf)

# Make plots:
import setupplots
setupplots.thesis_format()
import matplotlib.pyplot as plt
plt.ion()
import matplotlib.patches as mplp
import matplotlib.gridspec as mplgs
from matplotlib.colors import LogNorm
import matplotlib
matplotlib.rcParams['contour.negative_linestyle'] = 'solid'
import matplotlib.lines as mlines

# Plots of n:
f = plt.figure(figsize=(setupplots.TEXTWIDTH, setupplots.TEXTWIDTH))
gs = mplgs.GridSpec(3, 4, width_ratios=[5, 5, 5, 1])
a = []
k = 0
j = 0
for i, (p, n) in enumerate(zip(re.p_vals, re.n_vals)):
    a.append(f.add_subplot(gs[j, k], sharex=a[0] if i > 0 else None, sharey=a[0] if i > 0 else None))
    k += 1
    if k == 3:
        k = 0
        j += 1
    pcm = a[-1].pcolormesh(
        setupplots.make_pcolor_grid(re.roa),
        setupplots.make_pcolor_grid(re.time[::100]) - r.time_1,
        n[::100, :] * 1e6 / 1e12,
        cmap='plasma',
        vmax=5e6 * 1e6 / 1e12,
        vmin=0.0
    )
    # This is converted to 1e12 m^-3
    pcm.cmap.set_over('white')
    if i < 6:
        plt.setp(a[-1].get_xticklabels(), visible=False)
    else:
        a[-1].set_xlabel(r'$r/a$' + '\n' + r'$D = \SI{%.1f}{m^2/s}$' % (p[0],))
    if i % 3 != 0:
        plt.setp(a[-1].get_yticklabels(), visible=False)
    else:
        a[-1].set_ylabel(r'$V = \SI{%.0f}{m/s}$' % (p[1],) + '\n' + r'$t - t_{\text{inj}}$ [s]')
a[0].set_xlim(0, re.roa[-1])
a[0].xaxis.set_major_locator(plt.MaxNLocator(nbins=3))
a[0].set_ylim(re.time[0] - r.time_1, re.time[-1] - r.time_1)
cax = f.add_subplot(gs[:, 3])
cb = plt.colorbar(pcm, extend='max', cax=cax)
cb.set_label(r"$n_Z$ [$\SI{e12}{m^{-3}}$]")
f.suptitle("Effect of $D$, $V$ on $n_Z(r, t)$")
f.subplots_adjust(left=0.15, bottom=0.12, right=0.9, top=0.93, wspace=0.2, hspace=0.12)
setupplots.apply_formatter(f)
f.savefig("DVeffect_n.pdf")
f.savefig("DVeffect_n.pgf")

# Plots of n(0, t):
f = plt.figure(figsize=(0.75 * setupplots.TEXTWIDTH, 0.75 * setupplots.TEXTWIDTH / 1.618))
a = f.add_subplot(1, 1, 1)
roa_idxs = [0, scipy.where(re.roa <= 0.5)[0][-1], scipy.where(re.roa <= 1.0)[0][-1]]
labels = ['$r/a=0$', '$r/a=0.5$', '$r/a=1$']
linestyles = ['b', 'g--', 'r:']
for idx, l, ls in zip(roa_idxs, labels, linestyles):
    a.plot(re.time - r.time_1, re.n[:, idx] * 1e6 / 1e12, ls, label=l)
a.set_xlabel(r'$t-t_{\text{inj}}$ [s]')
a.set_ylabel(r'$n_Z$ [$\SI{e12}{m^{-3}}$]')
a.set_title("Temporal evolution of $n_Z$ at various radii")
a.legend(loc='upper right')
a.set_ylim(0, 5)
a.set_xlim(left=0)
setupplots.apply_formatter(f)
f.savefig("nZtBasic.pdf", bbox_inches='tight')
f.savefig("nZtBasic.pgf", bbox_inches='tight')

# Plot of N(t):
f = plt.figure(figsize=(0.75 * setupplots.TEXTWIDTH, 0.75 * setupplots.TEXTWIDTH / 1.618))
a = f.add_subplot(1, 1, 1)
a.plot(re.time - r.time_1, re.N / 1e12, label='simulated')
a.plot(
    re.time - r.time_1,
    scipy.exp((scipy.hstack((scipy.ones((len(re.time), 1)), scipy.atleast_2d(re.time).T))).dot(re.theta)) / 1e12,
    'g--',
    linewidth=2 * setupplots.lw,
    label='fit'
)
a.set_xlabel(r'$t-t_{\text{inj}}$ [s]')
a.set_ylabel(r'$N_Z$ [$\SI{e12}{particles}$]')
a.set_title("Total impurity content")
a.set_xlim(left=0)
a.set_ylim(0, 6)
a.legend(loc='upper right')
setupplots.apply_formatter(f)
f.savefig("NNZtBasic.pdf", bbox_inches='tight')
f.savefig("NNZtBasic.pgf", bbox_inches='tight')

# Plots of n(r, t):
f = plt.figure(figsize=(0.75 * setupplots.TEXTWIDTH, 0.75 * setupplots.TEXTWIDTH / 1.618))
gs = mplgs.GridSpec(1, 2, width_ratios=[15, 1])
a = f.add_subplot(gs[0, 0])
time_targets = scipy.concatenate((scipy.linspace(0, 10, 11), [20, 30, 40]))
ls = itertools.cycle(['-', '--', ':', '-.'])
for i, t in enumerate(time_targets):
    idx = scipy.where(re.time - r.time_1 <= t * 1e-3)[0][-1]
    a.plot(re.roa, re.n[idx, :] * 1e6 / 1e12, color=matplotlib.cm.plasma((t - time_targets.min()) / (time_targets.max() - time_targets.min())))
a_cb = f.add_subplot(gs[0, 1])
norm = matplotlib.colors.Normalize(vmin=min(time_targets), vmax=max(time_targets))
cb1 = matplotlib.colorbar.ColorbarBase(a_cb, cmap='plasma', norm=norm)
cb1.set_label(r'$t-t_{\text{inj}}$ [ms]')
a.set_xlabel("$r/a$")
a.set_ylabel(r"$n_Z$ [$\SI{e12}{m^{-3}}$]")
a.set_xlim(0, re.roa[-1])
a.set_title("Temporal evolution of impurity density profile")
setupplots.apply_formatter(f)
f.savefig("nZroaBasic.pdf", bbox_inches='tight')
f.savefig("nZroaBasic.pgf", bbox_inches='tight')
