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

# This script makes figures 3.45, 3.46, 3.47 and 3.48, which show the
# correlations between D, V and ne, Te and the knot locations for 1101014006.

from __future__ import division

import scipy
import scipy.io
import scipy.stats
import os
import eqtools

shotnum = 1101014006
e = eqtools.CModEFITTree(shotnum)

results_files = [
    '/Users/markchilenski/cmodws/home/markchil/strahl/strahl_1101014006_as_of_140930/savepoint_backup_MCMC_141005',
    '/Users/markchilenski/cmodws/home/markchil/strahl/strahl_1101014006_as_of_140930/savepoint_backup_MAP_141005',
    # '/Users/markchilenski/cmodws/home/markchil/strahl/strahl_1101014006_as_of_140930/savepoint_backup_spline',
]
dims = [[400, 100], [400, 100], [400, 99]]

def eval_prof(knots, coeffs, grid):
    return scipy.interpolate.InterpolatedUnivariateSpline(
        scipy.asarray(knots, dtype=float),
        scipy.asarray(coeffs, dtype=float),
        k=1
    )(grid)

class Result(object):
    def __init__(self, fn, dim):
        self.filename = fn
        f = scipy.io.readsav(fn)
        mask = (f.chisqd != 0.0) & (f.chisqd != 1.0) & (f.chisqd != -999) & (f.chisqd <= 1e5)
        self.chisqd = f.chisqd[mask]
        self.roasave = f.roasave[mask, :]
        self.D_results = f.dvresults[mask, :, 0]
        self.V_results = f.dvresults[mask, :, 1]
        
        self.ne = self.load_file(fn, 'ne', dim)[f.input.tstart:f.input.tstop + 1, :][mask, :]
        self.Te = self.load_file(fn, 'Te', dim)[f.input.tstart:f.input.tstop + 1, :][mask, :]
        self.Rmaj = self.load_file(fn, 'Rmaj', dim)[0, :]
        self.roa = e.rmid2roa(self.Rmaj, 1.0)
        
        self.D = scipy.asarray([eval_prof(roa, DD, self.roa) for roa, DD in zip(self.roasave, self.D_results)], dtype=float)
        self.V = scipy.asarray([eval_prof(roa, V, self.roa) for roa, V in zip(self.roasave, self.V_results)], dtype=float)
    
    def load_file(self, fn, name, dim):
        file_dir, file_name = os.path.split(os.path.abspath(fn))
        suffix = file_name[len('savepoint_backup'):]
        return scipy.reshape(
            scipy.fromfile(
                os.path.join(file_dir, name + suffix + '.dat'),
                dtype=scipy.float32
            ),
            dim
        ).T

results = [Result(fn, d) for fn, d in zip(results_files, dims)]
D = scipy.vstack([r.D for r in results])
V = scipy.vstack([r.V for r in results])
ne = scipy.vstack([r.ne for r in results])
Te = scipy.vstack([r.Te for r in results])
knots = scipy.vstack([r.roasave[:, 1:-1] for r in results])

# Efficiently compute the correlation matrices:
corr_D_ne = scipy.corrcoef(scipy.hstack((D, ne)).T)
corr_D_Te = scipy.corrcoef(scipy.hstack((D, Te)).T)
corr_V_ne = scipy.corrcoef(scipy.hstack((V, ne)).T)
corr_V_Te = scipy.corrcoef(scipy.hstack((V, Te)).T)
corr_D_knots = scipy.corrcoef(scipy.hstack((D, knots)).T)
corr_V_knots = scipy.corrcoef(scipy.hstack((V, knots)).T)

# Chop down to relevant portion:
corr_D_ne = corr_D_ne[:len(results[0].roa), len(results[0].roa):]
corr_D_Te = corr_D_Te[:len(results[0].roa), len(results[0].roa):]
corr_V_ne = corr_V_ne[:len(results[0].roa), len(results[0].roa):]
corr_V_Te = corr_V_Te[:len(results[0].roa), len(results[0].roa):]
corr_D_knots = corr_D_knots[:len(results[0].roa), len(results[0].roa):]
corr_V_knots = corr_V_knots[:len(results[0].roa), len(results[0].roa):]

# Compute the t-values:
N = D.shape[0]
nu = N - 2.0
t_D_ne = corr_D_ne * scipy.sqrt(nu / (1.0 - corr_D_ne**2.0))
t_D_Te = corr_D_Te * scipy.sqrt(nu / (1.0 - corr_D_Te**2.0))
t_V_ne = corr_V_ne * scipy.sqrt(nu / (1.0 - corr_V_ne**2.0))
t_V_Te = corr_V_Te * scipy.sqrt(nu / (1.0 - corr_V_Te**2.0))

t_D_knots = corr_D_knots * scipy.sqrt(nu / (1.0 - corr_D_knots**2.0))
t_V_knots = corr_V_knots * scipy.sqrt(nu / (1.0 - corr_V_knots**2.0))

sig_D_ne = scipy.special.betainc(0.5 * nu, 0.5, nu / (nu + t_D_ne**2.0))
sig_D_Te = scipy.special.betainc(0.5 * nu, 0.5, nu / (nu + t_D_Te**2.0))
sig_V_ne = scipy.special.betainc(0.5 * nu, 0.5, nu / (nu + t_V_ne**2.0))
sig_V_Te = scipy.special.betainc(0.5 * nu, 0.5, nu / (nu + t_V_Te**2.0))

sig_D_knots = scipy.special.betainc(0.5 * nu, 0.5, nu / (nu + t_D_knots**2.0))
sig_V_knots = scipy.special.betainc(0.5 * nu, 0.5, nu / (nu + t_V_knots**2.0))

# Mask to the threshold p-value:
pval = 0.05

mask_D_ne = scipy.zeros_like(sig_D_ne)
mask_D_ne[sig_D_ne <= pval] = None
mask_D_Te = scipy.zeros_like(sig_D_Te)
mask_D_Te[sig_D_Te <= pval] = None
mask_V_ne = scipy.zeros_like(sig_V_ne)
mask_V_ne[sig_V_ne <= pval] = None
mask_V_Te = scipy.zeros_like(sig_V_Te)
mask_V_Te[sig_V_Te <= pval] = None

mask_D_knots = scipy.zeros_like(sig_D_knots)
mask_D_knots[sig_D_knots <= pval] = None
mask_V_knots = scipy.zeros_like(sig_V_knots)
mask_V_knots[sig_V_knots <= pval] = None

mean_knots = scipy.mean(knots, axis=0)
std_knots = scipy.std(knots, axis=0, ddof=1)

# Get the peak correlation of D with ne:
abs_corr_D_ne = scipy.absolute(corr_D_ne)
i, j = scipy.where(abs_corr_D_ne == abs_corr_D_ne.max())
i_corr_D_ne = i[0]
j_corr_D_ne = j[0]

# Get the peak correlation of D with Te:
abs_corr_D_Te = scipy.absolute(corr_D_Te)
i, j = scipy.where(abs_corr_D_Te == abs_corr_D_Te.max())
i_corr_D_Te = i[0]
j_corr_D_Te = j[0]

# Get the peak correlation of V with ne:
abs_corr_V_ne = scipy.absolute(corr_V_ne)
i, j = scipy.where(abs_corr_V_ne == abs_corr_V_ne.max())
i_corr_V_ne = i[0]
j_corr_V_ne = j[0]

# Get the peak correlation of V with Te:
abs_corr_V_Te = scipy.absolute(corr_V_Te)
i, j = scipy.where(abs_corr_V_Te == abs_corr_V_Te.max())
i_corr_V_Te = i[0]
j_corr_V_Te = j[0]

# Get the peak correlation of D with knots:
abs_corr_D_knots = scipy.absolute(corr_D_knots)
i, j = scipy.where(abs_corr_D_knots == abs_corr_D_knots.max())
i_corr_D_knots = i[0]
j_corr_D_knots = j[0]

# Get the peak correlation of V with knots:
abs_corr_V_knots = scipy.absolute(corr_V_knots)
i, j = scipy.where(abs_corr_V_knots == abs_corr_V_knots.max())
i_corr_V_knots = i[0]
j_corr_V_knots = j[0]

# Make plots:
import setupplots
setupplots.thesis_format()
import matplotlib.pyplot as plt
plt.ion()
import matplotlib.gridspec as mplgs
import matplotlib.cm

df = 2

cmap = matplotlib.cm.gray
cmap.set_bad(alpha=0.0)

roa_grid_plot = setupplots.make_pcolor_grid(results[0].roa)
roa_grid_plot_dec = setupplots.make_pcolor_grid(results[0].roa[::df])
knot_grid_plot = setupplots.make_pcolor_grid(range(0, knots.shape[1]))

# Knots scatter plot:
f = plt.figure(figsize=(0.75 * setupplots.TEXTWIDTH, 2 * 0.75 * setupplots.TEXTWIDTH / 1.618))

# Peak D vs. knots scatter:
a_D = f.add_subplot(2, 1, 1)
a_D.plot(knots[:, j_corr_D_knots], D[:, i_corr_D_knots], 'o')
# a_D.set_xlabel(r'$r/a$ of knot %d' % (j_corr_D_knots,))
plt.setp(a_D.get_xticklabels(), visible=False)
a_D.set_ylabel(r'$D(%.2f)$ [$\si{m^2/s}$]' % (results[0].roa[i_corr_D_knots],))
a_D.set_title(
    r"Peak correlation of $D$ with knot locations, $\rho=%.2f$" % (
        corr_D_knots[i_corr_D_knots, j_corr_D_knots],
    )
)

# Peak V vs. knots scatter:
a_V = f.add_subplot(2, 1, 2, sharex=a_D)
a_V.plot(knots[:, j_corr_V_knots], V[:, i_corr_V_knots], 'o')
a_V.set_xlabel(r'$r/a$ of knot %d' % (j_corr_V_knots,))
a_V.set_ylabel(r'$V(%.2f)$ [$\si{m/s}$]' % (results[0].roa[i_corr_V_knots],))
a_V.set_title(
    r"Peak correlation of $V$ with knot locations, $\rho=%.2f$" % (
        corr_V_knots[i_corr_V_knots, j_corr_V_knots],
    )
)
setupplots.apply_formatter(f)
f.savefig("ScatterDVKnots.pdf", bbox_inches='tight')
f.savefig("ScatterDVKnots.pgf", bbox_inches='tight')

# D, V; ne, Te scatter:
f = plt.figure(figsize=(6, 1.1 * 6 / 1.618))

# Peak D vs. ne scatter:
a_Dne = f.add_subplot(2, 2, 1)
a_Dne.plot(ne[:, j_corr_D_ne] / 1e19, D[:, i_corr_D_ne], 'o')
a_Dne.set_xlabel(r'$n_{\text{e}}(%.2f)$ [$\SI{e19}{m^{-3}}$]' % (results[0].roa[j_corr_D_ne],))
a_Dne.set_ylabel(r'$D(%.2f)$ [$\si{m^2/s}$]' % (results[0].roa[i_corr_D_ne],))
a_Dne.set_title(
    r"Peak correlation of $D$ with $n_{\text{e}}$, $\rho=%.2f$" % (
        corr_D_ne[i_corr_D_ne, j_corr_D_ne],
    )
)

# Peak D vs. Te scatter:
a_DTe = f.add_subplot(2, 2, 2)
a_DTe.plot(Te[:, j_corr_D_Te], D[:, i_corr_D_Te], 'o')
a_DTe.set_xlabel(r'$T_{\text{e}}(%.2f)$ [keV]' % (results[0].roa[j_corr_D_Te],))
a_DTe.set_ylabel(r'$D(%.2f)$ [$\si{m^2/s}$]' % (results[0].roa[i_corr_D_Te],))
a_DTe.set_title(
    r"Peak correlation of $D$ with $T_{\text{e}}$, $\rho=%.2f$" % (
        corr_D_Te[i_corr_D_Te, j_corr_D_Te],
    )
)
a_DTe.set_ylim(bottom=0.0)

# Peak V vs. ne scatter:
a_Vne = f.add_subplot(2, 2, 3)
a_Vne.plot(ne[:, j_corr_V_ne] / 1e19, V[:, i_corr_V_ne], 'o')
a_Vne.xaxis.set_major_locator(plt.MaxNLocator(nbins=7))
a_Vne.set_xlabel(r'$n_{\text{e}}(%.2f)$ [$\SI{e19}{m^{-3}}$]' % (results[0].roa[j_corr_V_ne],))
a_Vne.set_ylabel(r'$V(%.2f)$ [m/s]' % (results[0].roa[i_corr_V_ne],))
a_Vne.set_title(
    r"Peak correlation of $V$ with $n_{\text{e}}$, $\rho=%.2f$" % (
        corr_V_ne[i_corr_V_ne, j_corr_V_ne],
    )
)

# Peak V vs. Te scatter:
a_VTe = f.add_subplot(2, 2, 4)
a_VTe.plot(Te[:, j_corr_V_Te], V[:, i_corr_V_Te], 'o')
a_VTe.set_xlabel(r'$T_{\text{e}}(%.2f)$ [keV]' % (results[0].roa[j_corr_V_Te],))
a_VTe.set_ylabel(r'$V(%.2f)$ [m/s]' % (results[0].roa[i_corr_V_Te],))
a_VTe.set_title(r"Peak correlation of $V$ with $T_{\text{e}}$")
a_VTe.set_title(
    r"Peak correlation of $V$ with $T_{\text{e}}$, $\rho=%.2f$" % (
        corr_V_Te[i_corr_V_Te, j_corr_V_Te],
    )
)
f.subplots_adjust(left=0.1, right=0.97, top=0.93, wspace=0.24, hspace=0.46)
setupplots.apply_formatter(f)
f.savefig("ScatterDVneTe.pdf", bbox_inches='tight')
f.savefig("ScatterDVneTe.pgf", bbox_inches='tight')

# D, V; knots corr:
f = plt.figure(figsize=(setupplots.TEXTWIDTH, setupplots.TEXTWIDTH / 1.618))
gs = mplgs.GridSpec(2, 3, width_ratios=[10, 10, 1])

# D vs. knots:
a_D = f.add_subplot(gs[0, 0])
pcm = a_D.pcolormesh(
    knot_grid_plot,
    roa_grid_plot,
    corr_D_knots,
    cmap='seismic',
    vmin=-1,
    vmax=1
)
a_D.set_ylabel('$r/a$ for $D$')
# a_D.set_xlabel(r'knot index')
plt.setp(a_D.get_xticklabels(), visible=False)
a_D.set_title(
    r"$\corr[D, \text{knots}]$, $\max|\rho|=%.2f$" % (
        scipy.absolute(corr_D_knots).max(),
    )
)
a_D.set_xlim(left=-0.5, right=knots.shape[1] - 0.5)
a_D.set_ylim(bottom=0.0, top=results[0].roa[-1])
a_D.set_xticks(range(0, knots.shape[1]))
for i, (k, sk) in enumerate(zip(mean_knots, std_knots)):
    a_D.plot([i - 0.5, i + 0.5], [k, k], 'g:', linewidth=3.0 * setupplots.lw)
    # a_D.fill_between([i - 0.5, i + 0.5], [k - sk, k - sk], [k + sk, k + sk], facecolor='g', alpha=0.25)

# V vs. knots:
a_V = f.add_subplot(gs[1, 0])
pcm = a_V.pcolormesh(
    knot_grid_plot,
    roa_grid_plot,
    corr_V_knots,
    cmap='seismic',
    vmin=-1,
    vmax=1
)
a_cb = f.add_subplot(gs[:, 2])
cb = f.colorbar(pcm, cax=a_cb)
cb.set_label(r'$\rho$')
a_V.set_ylabel('$r/a$ for $V$')
a_V.set_xlabel(r'knot index')
a_V.set_title(
    r"$\corr[V, \text{knots}]$, $\max|\rho|=%.2f$" % (
        scipy.absolute(corr_V_knots).max(),
    )
)
a_V.set_xlim(left=-0.5, right=knots.shape[1] - 0.5)
a_V.set_ylim(bottom=0.0, top=results[0].roa[-1])
a_V.set_xticks(range(0, knots.shape[1]))
for i, (k, sk) in enumerate(zip(mean_knots, std_knots)):
    a_V.plot([i - 0.5, i + 0.5], [k, k], 'g:', linewidth=3.0 * setupplots.lw)
    # a_V.fill_between([i - 0.5, i + 0.5], [k - sk, k - sk], [k + sk, k + sk], facecolor='g', alpha=0.25)

# D vs. knots masked:
a_D = f.add_subplot(gs[0, 1])
pcm = a_D.pcolormesh(
    knot_grid_plot,
    roa_grid_plot,
    corr_D_knots,
    cmap='seismic',
    vmin=-1,
    vmax=1
)
# a_D.set_ylabel('$r/a$ for $D$')
plt.setp(a_D.get_yticklabels(), visible=False)
# a_D.set_xlabel(r'knot index')
plt.setp(a_D.get_xticklabels(), visible=False)
a_D.set_title("Masked, $p\le 0.05$")
# a_D.set_title(
#     r"$\corr[D, \text{knots}]$, $\max|\rho|=%.2f$" % (
#         scipy.absolute(corr_D_knots).max(),
#     )
# )
a_D.set_xlim(left=-0.5, right=knots.shape[1] - 0.5)
a_D.set_ylim(bottom=0.0, top=results[0].roa[-1])
a_D.set_xticks(range(0, knots.shape[1]))
for i, (k, sk) in enumerate(zip(mean_knots, std_knots)):
    a_D.plot([i - 0.5, i + 0.5], [k, k], 'g:', linewidth=3.0 * setupplots.lw)
    # a_D.fill_between([i - 0.5, i + 0.5], [k - sk, k - sk], [k + sk, k + sk], facecolor='g', alpha=0.25)
a_D.pcolormesh(
    knot_grid_plot,
    roa_grid_plot,
    scipy.ma.masked_where(scipy.isnan(mask_D_knots), mask_D_knots),
    cmap=cmap,
    alpha=0.75
)

# V vs. knots:
a_V = f.add_subplot(gs[1, 1])
pcm = a_V.pcolormesh(
    knot_grid_plot,
    roa_grid_plot,
    corr_V_knots,
    cmap='seismic',
    vmin=-1,
    vmax=1
)
# a_V.set_ylabel('$r/a$ for $V$')
plt.setp(a_V.get_yticklabels(), visible=False)
a_V.set_xlabel(r'knot index')
a_V.set_title("Masked, $p\le 0.05$")
# a_V.set_title(
#     r"$\corr[V, \text{knots}]$, $\max|\rho|=%.2f$" % (
#         scipy.absolute(corr_V_knots).max(),
#     )
# )
a_V.set_xlim(left=-0.5, right=knots.shape[1] - 0.5)
a_V.set_ylim(bottom=0.0, top=results[0].roa[-1])
a_V.set_xticks(range(0, knots.shape[1]))
for i, (k, sk) in enumerate(zip(mean_knots, std_knots)):
    a_V.plot([i - 0.5, i + 0.5], [k, k], 'g:', linewidth=3.0 * setupplots.lw)
    # a_V.fill_between([i - 0.5, i + 0.5], [k - sk, k - sk], [k + sk, k + sk], facecolor='g', alpha=0.25)
a_V.pcolormesh(
    knot_grid_plot,
    roa_grid_plot,
    scipy.ma.masked_where(scipy.isnan(mask_V_knots), mask_V_knots),
    cmap=cmap,
    alpha=0.75
)

f.subplots_adjust(hspace=0.27)
setupplots.apply_formatter(f)
f.savefig("CorrDVKnots.pdf", bbox_inches='tight')
f.savefig("CorrDVKnots.pgf", bbox_inches='tight')

# D, V; ne, Te corr:
f = plt.figure(figsize=(setupplots.TEXTWIDTH, setupplots.TEXTWIDTH / 1.618))
gs = mplgs.GridSpec(2, 3, width_ratios=[10, 10, 1])

# D vs. ne:
a_Dne = f.add_subplot(gs[0, 0])
pcm = a_Dne.pcolormesh(
    roa_grid_plot_dec,
    roa_grid_plot_dec,
    corr_D_ne[::df, ::df],
    cmap='seismic',
    vmin=-1,
    vmax=1
)
a_cb = f.add_subplot(gs[:, 2])
cb = f.colorbar(pcm, cax=a_cb)
cb.set_label(r'$\rho$')
a_Dne.set_ylabel('$r/a$ for $D$')
# a_Dne.set_xlabel(r'$r/a$ for $n_{\text{e}}$')
plt.setp(a_Dne.get_xticklabels(), visible=False)
a_Dne.set_title(
    r"$\corr[D, n_{\text{e}}]$, $\max|\rho|=%.2f$" % (
        scipy.absolute(corr_D_ne).max(),
    )
)
a_Dne.set_xlim(left=0.0, right=results[0].roa[-1])
a_Dne.set_ylim(bottom=0.0, top=results[0].roa[-1])
a_Dne.axhline(1.0, color='g', ls=':')
a_Dne.axvline(1.0, color='g', ls=':')
a_Dne.axhline(0.0, color='g', ls=':')
a_Dne.axvline(0.0, color='g', ls=':')
for i, k in enumerate(mean_knots):
    a_Dne.axhline(k, color='g', ls=':')
    a_Dne.axvline(k, color='g', ls=':')

# D vs. Te:
a_DTe = f.add_subplot(gs[0, 1], sharex=a_Dne, sharey=a_Dne)
pcm = a_DTe.pcolormesh(
    roa_grid_plot_dec,
    roa_grid_plot_dec,
    corr_D_Te[::df, ::df],
    cmap='seismic',
    vmin=-1,
    vmax=1
)
# a_DTe.set_ylabel('$r/a$ for $D$')
# a_DTe.set_xlabel(r'$r/a$ for $T_{\text{e}}$')
plt.setp(a_DTe.get_xticklabels(), visible=False)
plt.setp(a_DTe.get_yticklabels(), visible=False)
a_DTe.set_title(
    r"$\corr[D, T_{\text{e}}]$, $\max|\rho|=%.2f$" % (
        scipy.absolute(corr_D_Te).max(),
    )
)
a_DTe.set_xlim(left=0.0, right=results[0].roa[-1])
a_DTe.set_ylim(bottom=0.0, top=results[0].roa[-1])
a_DTe.axhline(1.0, color='g', ls=':')
a_DTe.axvline(1.0, color='g', ls=':')
a_DTe.axhline(0.0, color='g', ls=':')
a_DTe.axvline(0.0, color='g', ls=':')
for i, k in enumerate(mean_knots):
    a_DTe.axhline(k, color='g', ls=':')
    a_DTe.axvline(k, color='g', ls=':')

# V vs. ne:
a_Vne = f.add_subplot(gs[1, 0], sharex=a_Dne, sharey=a_Dne)
pcm = a_Vne.pcolormesh(
    roa_grid_plot_dec,
    roa_grid_plot_dec,
    corr_V_ne[::df, ::df],
    cmap='seismic',
    vmin=-1,
    vmax=1
)
a_Vne.set_ylabel('$r/a$ for $V$')
a_Vne.set_xlabel(r'$r/a$ for $n_{\text{e}}$')
a_Vne.set_title(
    r"$\corr[V, n_{\text{e}}]$, $\max|\rho|=%.2f$" % (
        scipy.absolute(corr_V_ne).max(),
    )
)
a_Vne.set_xlim(left=0.0, right=results[0].roa[-1])
a_Vne.set_ylim(bottom=0.0, top=results[0].roa[-1])
a_Vne.axhline(1.0, color='g', ls=':')
a_Vne.axvline(1.0, color='g', ls=':')
a_Vne.axhline(0.0, color='g', ls=':')
a_Vne.axvline(0.0, color='g', ls=':')
for i, k in enumerate(mean_knots):
    a_Vne.axhline(k, color='g', ls=':')
    a_Vne.axvline(k, color='g', ls=':')

# V vs. Te:
a_VTe = f.add_subplot(gs[1, 1], sharex=a_Dne, sharey=a_Dne)
pcm = a_VTe.pcolormesh(
    roa_grid_plot_dec,
    roa_grid_plot_dec,
    corr_V_Te[::df, ::df],
    cmap='seismic',
    vmin=-1,
    vmax=1
)
# a_VTe.set_ylabel('$r/a$ for $V$')
plt.setp(a_VTe.get_yticklabels(), visible=False)
a_VTe.set_xlabel(r'$r/a$ for $T_{\text{e}}$')
a_VTe.set_title(
    r"$\corr[V, T_{\text{e}}]$, $\max|\rho|=%.2f$" % (
        scipy.absolute(corr_V_Te).max(),
    )
)
a_VTe.set_xlim(left=0.0, right=results[0].roa[-1])
a_VTe.set_ylim(bottom=0.0, top=results[0].roa[-1])
a_VTe.axhline(1.0, color='g', ls=':')
a_VTe.axvline(1.0, color='g', ls=':')
a_VTe.axhline(0.0, color='g', ls=':')
a_VTe.axvline(0.0, color='g', ls=':')
for i, k in enumerate(mean_knots):
    a_VTe.axhline(k, color='g', ls=':')
    a_VTe.axvline(k, color='g', ls=':')

f.subplots_adjust(hspace=0.26)
setupplots.apply_formatter(f)
f.savefig("CorrDVneTe.pdf", bbox_inches='tight')
f.savefig("CorrDVneTe.pgf", bbox_inches='tight')

a_Dne.pcolormesh(
    roa_grid_plot_dec,
    roa_grid_plot_dec,
    scipy.ma.masked_where(scipy.isnan(mask_D_ne), mask_D_ne)[::df, ::df],
    cmap=cmap,
    alpha=0.75
)
a_DTe.pcolormesh(
    roa_grid_plot_dec,
    roa_grid_plot_dec,
    scipy.ma.masked_where(scipy.isnan(mask_D_Te), mask_D_Te)[::df, ::df],
    cmap=cmap,
    alpha=0.75
)
a_Vne.pcolormesh(
    roa_grid_plot_dec,
    roa_grid_plot_dec,
    scipy.ma.masked_where(scipy.isnan(mask_V_ne), mask_V_ne)[::df, ::df],
    cmap=cmap,
    alpha=0.75
)
a_VTe.pcolormesh(
    roa_grid_plot_dec,
    roa_grid_plot_dec,
    scipy.ma.masked_where(scipy.isnan(mask_V_Te), mask_V_Te)[::df, ::df],
    cmap=cmap,
    alpha=0.75
)

f.savefig("CorrDVneTeMasked.pdf", bbox_inches='tight')
f.savefig("CorrDVneTeMasked.pgf", bbox_inches='tight')
