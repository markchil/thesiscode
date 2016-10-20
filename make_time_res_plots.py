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

# This script makes figure 3.1, 3.6, 3.7, 3.8, 3.9 and 3.11, which show the
# properties of the impurity density following an impurity injection. It
# evaluates the impurity density over a dense grid in D, V space.

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

# Load the dense sampling data:
D_grid = scipy.linspace(0, 10, 102)[1:]
V_grid = scipy.linspace(-50, 10, 100)
# The following code is what is used to produce cs_den.pkl. It takes a while...
# pool = bayesimp.make_pool(num_proc=24)
# res = r.parallel_compute_cs_den(D_grid, V_grid, pool)
# bayesimp.finalize_pool(pool)
# with open("../cs_den_new.pkl", 'wb') as pf:
#     pkl.dump(res, pf, protocol=pkl.HIGHEST_PROTOCOL)

with open("../cs_den_new.pkl", 'rb') as pf:
    res = pkl.load(pf)
DD, VV = scipy.meshgrid(D_grid, V_grid)
VD = VV / DD
S = -r.efit_tree.getAOutSpline()((r.time_1 + r.time_2) / 2.0) * VD / 2.0
orig_shape = DD.shape

tau_N = scipy.reshape(scipy.asarray([rr.tau_N for rr in res]), orig_shape)
t_peak_core = scipy.reshape(scipy.asarray([rr.t_n_peak_local[0] for rr in res]), orig_shape) - r.time_1
t_peak_edge = scipy.reshape(scipy.asarray([rr.t_n_peak_local[-1] for rr in res]), orig_shape) - r.time_1
t_peak = t_peak_core - t_peak_edge
n_peak_core = scipy.reshape(scipy.asarray([rr.n_peak_local[0] for rr in res]), orig_shape)
n075n0 = scipy.reshape(scipy.asarray([rr.n075n0 for rr in res]), orig_shape)

s_flat = S.ravel()
b_flat = n075n0.ravel()
d_flat = DD.ravel()

os.chdir(cdir)

# Save the variables for later use:
# https://stackoverflow.com/questions/2960864/how-can-i-save-all-the-variables-in-the-current-python-session
import shelve

shelf_name = 'make_time_res_plots.shelf'
shelf = shelve.open(shelf_name, 'n')
for key in dir():
    try:
        shelf[key] = globals()[key]
    except:
        print("Failed to shelve %s" % (key,))
shelf.close()

# To restore:
# shelf = shelve.open(shelf_name)
# for key in shelf:
#     globals()[key] = shelf[key]
# shelf.close()

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

D_grid_plot = setupplots.make_pcolor_grid(D_grid)
V_grid_plot = setupplots.make_pcolor_grid(V_grid)

# Plot of n(0.75) / n(0.0):
f = plt.figure(figsize=(0.75 * setupplots.TEXTWIDTH, 0.75 * setupplots.TEXTWIDTH / 1.618))
a = f.add_subplot(1, 1, 1)
pcm = a.pcolormesh(
    D_grid_plot,
    V_grid_plot,
    n075n0,
    vmin=0,
    vmax=1.0
)
pcm.cmap.set_over('white')
cb = f.colorbar(pcm, extend='max')
cb.set_label("$b_{0.75}$")
a.contour(
    D_grid,
    V_grid,
    n075n0,
    scipy.linspace(0, 1.0, 50),
    alpha=0.5,
    linewidths=setupplots.lw / 2.0,
    colors='w'
)
a.set_title("Impurity density profile broadness")
a.set_xlabel(r"$D$ [$\si{m^2/s}$]")
a.set_ylabel(r"$V$ [$\si{m/s}$]")
a.set_xlim(0, D_grid[-1])
a.set_ylim(V_grid[0], V_grid[-1])
setupplots.apply_formatter(f)
f.savefig("n075n0.pdf", bbox_inches='tight')
f.savefig("n075n0.pgf", bbox_inches='tight')

# Plot of tau_N:
f = plt.figure(figsize=(0.75 * setupplots.TEXTWIDTH, 0.75 * setupplots.TEXTWIDTH / 1.618))
a = f.add_subplot(1, 1, 1)
pcm = a.pcolormesh(
    D_grid_plot,
    V_grid_plot,
    tau_N,
    norm=LogNorm()
)
cb = f.colorbar(pcm)
cb.set_label(r"$\tau_{\text{imp}}$ [s]")
a.contour(D_grid, V_grid, scipy.log10(tau_N), 50, alpha=0.5, linewidths=setupplots.lw / 2.0, colors='w')
a.set_title("Impurity confinement time")
a.set_xlabel(r"$D$ [$\si{m^2/s}$]")
a.set_ylabel(r"$V$ [$\si{m/s}$]")
a.set_xlim(0, D_grid[-1])
a.set_ylim(V_grid[0], V_grid[-1])
setupplots.apply_formatter(f)
f.savefig("tauimp.pdf", bbox_inches='tight')
f.savefig("tauimp.pgf", bbox_inches='tight')

# Plot of n_peak_core:
f = plt.figure(figsize=(0.75 * setupplots.TEXTWIDTH, 0.75 * setupplots.TEXTWIDTH / 1.618))
a = f.add_subplot(1, 1, 1)
pcm = a.pcolormesh(
    D_grid_plot,
    V_grid_plot,
    n_peak_core * 1e6 / 1e12,
    vmin=0,
    vmax=25
)
pcm.cmap.set_over('white')
cb = f.colorbar(pcm, extend='max')
cb.set_label(r"$n_{Z,\text{p}}(0)$ [$\SI{e12}{m^{-3}}$]")
a.contour(D_grid, V_grid, n_peak_core * 1e6 / 1e12, scipy.linspace(0.0, 25.0, 50), alpha=0.5, linewidths=setupplots.lw / 2.0, colors='w')
a.set_title("Peak core impurity density")
a.set_xlabel(r"$D$ [$\si{m^2/s}$]")
a.set_ylabel(r"$V$ [$\si{m/s}$]")
a.set_xlim(0, D_grid[-1])
a.set_ylim(V_grid[0], V_grid[-1])
setupplots.apply_formatter(f)
f.savefig("npeakcore.pdf", bbox_inches='tight')
f.savefig("npeakcore.pgf", bbox_inches='tight')

# Plot of t_peak_core:
f = plt.figure(figsize=(0.75 * setupplots.TEXTWIDTH, 0.75 * setupplots.TEXTWIDTH / 1.618))
a = f.add_subplot(1, 1, 1)
pcm = a.pcolormesh(
    D_grid_plot,
    V_grid_plot,
    t_peak_core,
    norm=LogNorm(),
    vmin=1e-3,
    vmax=100e-3
)
cb = f.colorbar(pcm)
cb.set_label(r"$t_{\text{r}}$ [s]")
a.contour(D_grid, V_grid, scipy.log10(t_peak), 50, alpha=0.5, linewidths=setupplots.lw / 2.0, colors='w')
a.set_title("Core rise time")
a.set_xlabel(r"$D$ [$\si{m^2/s}$]")
a.set_ylabel(r"$V$ [$\si{m/s}$]")
a.set_xlim(0, D_grid[-1])
a.set_ylim(V_grid[0], V_grid[-1])
setupplots.apply_formatter(f)
f.savefig("dt.pdf", bbox_inches='tight')
f.savefig("dt.pgf", bbox_inches='tight')

# Plot of the model D, V.
f = plt.figure(figsize=(0.75 * setupplots.TEXTWIDTH, 2 * 0.75 * setupplots.TEXTWIDTH / 1.618))
a_D = f.add_subplot(2, 1, 1)
a_V = f.add_subplot(2, 1, 2, sharex=a_D)
D, V = r.eval_DV(r.params_true)
a_D.plot(r.roa_grid_DV, D)
a_V.plot(r.roa_grid_DV, V)
a_V.set_xlabel("$r/a$")
a_D.set_ylabel(r"$D$ [$\si{m^2/s}$]")
a_V.set_ylabel(r"$V$ [$\si{m/s}$]")
a_D.set_title("True transport coefficient profiles")
plt.setp(a_D.get_xticklabels(), visible=False)
a_D.set_xlim(r.roa_grid_DV[0], r.roa_grid_DV[-1])
a_D.set_ylim(0, 2)
f.subplots_adjust(hspace=0.1)
setupplots.apply_formatter(f)
f.savefig("DconstVlin.pdf", bbox_inches='tight')
f.savefig("DconstVlin.pgf", bbox_inches='tight')

# Make plot of b(S):
f = plt.figure(figsize=(0.75 * setupplots.TEXTWIDTH, 0.75 * setupplots.TEXTWIDTH / 1.618))
a = f.add_subplot(1, 1, 1)
a.set_yscale('log')
sp = a.plot(s_flat, b_flat, '.')
a.set_xlabel("$S$")
a.set_ylabel("$b_{0.75}$")
a.set_title("$b_{0.75}$ is a proxy for $S$")
a.set_xlim(left=-12, right=25)
a.set_ylim(bottom=1e-7)
setupplots.apply_formatter(f)
f.savefig("bOfS.pdf", bbox_inches='tight')
f.savefig("bOfS.pgf", bbox_inches='tight')
