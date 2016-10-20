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

# This script makes figures 2.24, 2.25, A.11, A.12 and A.13, which show
# profiles fit with and without the TCI data.

from __future__ import division
import scipy
import profiletools
import eqtools
import gptools
import cPickle as pkl
import copy
import time

class Case(object):
    def __init__(self, shot, t_min, t_max, abscissa, constrain_at_limiter=True, bad_core=False, bad_outer=False):
        self.shot = shot
        self.t_min = t_min
        self.t_max = t_max
        self.abscissa = abscissa
        self.constrain_at_limiter = constrain_at_limiter
        self.bad_core = bad_core
        self.bad_outer = bad_outer

c = Case(1120907032, 0.8, 0.9, 'r/a', bad_outer=True)
c.roa_star = scipy.linspace(0, 1.1, 200)
c.core_mask = c.roa_star <= 1.0
c.e = eqtools.CModEFITTree(c.shot)

print("Fetching CTS...")
c.p_CTS = profiletools.neCTS(c.shot, t_min=c.t_min, t_max=c.t_max, abscissa=c.abscissa, efit_tree=c.e)
c.p_CTS.time_average(weighted=True)
if c.bad_core:
    # Drop the coremost point since it seems to be systematically low:
    mask = scipy.zeros_like(c.p_CTS.y, dtype=bool)
    mask[0] = True
    c.p_CTS.remove_points(mask)
if c.bad_outer:
    # Drop the outermost CTS point since it seems to be systematically high:
    mask = scipy.zeros_like(c.p_CTS.y, dtype=bool)
    mask[12] = True
    c.p_CTS.remove_points(mask)
c.p = copy.deepcopy(c.p_CTS)

print("Fetching ETS...")
c.p_ETS = profiletools.neETS(c.shot, t_min=c.t_min, t_max=c.t_max, abscissa=c.abscissa, efit_tree=c.e)
c.p_ETS.time_average(weighted=True)
c.p.add_profile(c.p_ETS)

print("Fetching TCI...")
c.p_TCI = profiletools.neTCI_old(c.shot, t_min=c.t_min, t_max=c.t_max, abscissa=c.abscissa, efit_tree=c.e)
c.p_TCI.time_average(weighted=False)
c.p_TCI.remove_quadrature_points_outside_of_limiter()
c.TCI_labels = [t.y_label for t in c.p_TCI.transformed]
c.TCI_y = [t.y[0] for t in c.p_TCI.transformed]
c.TCI_err_y = [t.err_y[0] for t in c.p_TCI.transformed]
c.TCI_axis = range(0, len(c.TCI_y))

print("Fitting with only TS...")
c.p.create_gp(constrain_at_limiter=c.constrain_at_limiter)
c.p.find_gp_MAP_estimate(verbose=True)
c.params_MAP_TS_only = c.p.gp.free_params[:]
c.l_MAP_TS_only = c.p.gp.k.l_func(c.roa_star, 0, *c.p.gp.k.params[1:])
c.out_MAP_TS_only = c.p.compute_a_over_L(c.roa_star, return_prediction=True)
c.out_TCI_vals_MAP_TS_only = [
    c.p.smooth(scipy.vstack(t.X), output_transform=scipy.linalg.block_diag(*t.T))
    for t in c.p_TCI.transformed
]
c.y_TCI_vals_MAP_TS_only = [v[0] for v in c.out_TCI_vals_MAP_TS_only]
c.err_y_TCI_vals_MAP_TS_only = [v[1] for v in c.out_TCI_vals_MAP_TS_only]
c.sampler_TS_only = c.p.gp.sample_hyperparameter_posterior(nsamp=500, sampler_a=8.0)
c.sampler_TS_only.pool.close()
c.sampler_TS_only.pool = None
c.out_MCMC_TS_only = c.p.compute_a_over_L(
    c.roa_star, return_prediction=True, use_MCMC=True, sampler=c.sampler_TS_only,
    burn=400, thin=100
)
c.out_TCI_vals_MCMC_TS_only = [
    c.p.smooth(
        scipy.vstack(t.X), output_transform=scipy.linalg.block_diag(*t.T),
        use_MCMC=True, sampler=c.sampler_TS_only, burn=400, thin=100
    )
    for t in c.p_TCI.transformed
]
c.y_TCI_vals_MCMC_TS_only = [v[0] for v in c.out_TCI_vals_MCMC_TS_only]
c.err_y_TCI_vals_MCMC_TS_only = [v[1] for v in c.out_TCI_vals_MCMC_TS_only]
res = c.p.gp.compute_l_from_MCMC(c.roa_star, sampler=c.sampler_TS_only, burn=400)
res = scipy.asarray(res)
c.mean_l_MCMC_TS_only = scipy.mean(res, axis=0)
c.std_l_MCMC_TS_only = scipy.std(res, axis=0, ddof=1)

print("Fitting with TS+TCI...")
c.p.add_profile(c.p_TCI)
c.p.create_gp(constrain_at_limiter=c.constrain_at_limiter)
t_start = time.time()
c.p.find_gp_MAP_estimate(verbose=True)
c.params_MAP_TS_TCI = c.p.gp.free_params[:]
c.l_MAP_TS_TCI = c.p.gp.k.l_func(c.roa_star, 0, *c.p.gp.k.params[1:])
c.out_MAP_TS_TCI = c.p.compute_a_over_L(c.roa_star, return_prediction=True)
c.out_TCI_vals_MAP_TS_TCI = [
    c.p.smooth(scipy.vstack(t.X), output_transform=scipy.linalg.block_diag(*t.T))
    for t in c.p_TCI.transformed
]
c.y_TCI_vals_MAP_TS_TCI = [v[0] for v in c.out_TCI_vals_MAP_TS_TCI]
c.err_y_TCI_vals_MAP_TS_TCI = [v[1] for v in c.out_TCI_vals_MAP_TS_TCI]
t_elapsed = time.time() - t_start
t_start = time.time()
c.sampler_TS_TCI = c.p.gp.sample_hyperparameter_posterior(nsamp=500, sampler_a=8.0)
c.sampler_TS_TCI.pool.close()
c.sampler_TS_TCI.pool = None
c.out_MCMC_TS_TCI = c.p.compute_a_over_L(
    c.roa_star, return_prediction=True, use_MCMC=True, sampler=c.sampler_TS_TCI,
    burn=400, thin=100
)
c.out_TCI_vals_MCMC_TS_TCI = [
    c.p.smooth(
        scipy.vstack(t.X), output_transform=scipy.linalg.block_diag(*t.T),
        use_MCMC=True, sampler=c.sampler_TS_TCI, burn=400, thin=100
    )
    for t in c.p_TCI.transformed
]
c.y_TCI_vals_MCMC_TS_TCI = [v[0] for v in c.out_TCI_vals_MCMC_TS_TCI]
c.err_y_TCI_vals_MCMC_TS_TCI = [v[1] for v in c.out_TCI_vals_MCMC_TS_TCI]
res = c.p.gp.compute_l_from_MCMC(c.roa_star, sampler=c.sampler_TS_TCI, burn=400)
res = scipy.asarray(res)
c.mean_l_MCMC_TS_TCI = scipy.mean(res, axis=0)
c.std_l_MCMC_TS_TCI = scipy.std(res, axis=0, ddof=1)
t_elapsed_mcmc = time.time() - t_start

print("Fitting with ETS+TCI...")
c.p2 = copy.deepcopy(c.p_ETS)
c.p2.add_profile(c.p_TCI)
c.p2.create_gp(constrain_at_limiter=c.constrain_at_limiter)
c.p2.gp.k.hyperprior = c.p.gp.k.hyperprior
c.p2.find_gp_MAP_estimate(verbose=True)
c.params_MAP_ETS_TCI = c.p2.gp.free_params[:]
c.l_MAP_ETS_TCI = c.p2.gp.k.l_func(c.roa_star, 0, *c.p2.gp.k.params[1:])
c.out_MAP_ETS_TCI = c.p2.compute_a_over_L(c.roa_star, return_prediction=True)
c.out_TCI_vals_MAP_ETS_TCI = [
    c.p2.smooth(scipy.vstack(t.X), output_transform=scipy.linalg.block_diag(*t.T))
    for t in c.p_TCI.transformed
]
c.y_TCI_vals_MAP_ETS_TCI = [v[0] for v in c.out_TCI_vals_MAP_ETS_TCI]
c.err_y_TCI_vals_MAP_ETS_TCI = [v[1] for v in c.out_TCI_vals_MAP_ETS_TCI]
c.sampler_ETS_TCI = c.p2.gp.sample_hyperparameter_posterior(nsamp=500, sampler_a=8.0)
c.sampler_ETS_TCI.pool.close()
c.sampler_ETS_TCI.pool = None
c.out_MCMC_ETS_TCI = c.p2.compute_a_over_L(
    c.roa_star, return_prediction=True, use_MCMC=True, sampler=c.sampler_ETS_TCI,
    burn=400, thin=100
)
c.out_TCI_vals_MCMC_ETS_TCI = [
    c.p2.smooth(
        scipy.vstack(t.X), output_transform=scipy.linalg.block_diag(*t.T),
        use_MCMC=True, sampler=c.sampler_ETS_TCI, burn=400, thin=100
    )
    for t in c.p_TCI.transformed
]
c.y_TCI_vals_MCMC_ETS_TCI = [v[0] for v in c.out_TCI_vals_MCMC_ETS_TCI]
c.err_y_TCI_vals_MCMC_ETS_TCI = [v[1] for v in c.out_TCI_vals_MCMC_ETS_TCI]
res = c.p2.gp.compute_l_from_MCMC(c.roa_star, sampler=c.sampler_ETS_TCI, burn=400)
res = scipy.asarray(res)
c.mean_l_MCMC_ETS_TCI = scipy.mean(res, axis=0)
c.std_l_MCMC_ETS_TCI = scipy.std(res, axis=0, ddof=1)

with open('ne_TCI_test.pkl', 'wb') as pf:
    pkl.dump(c, pf)

print("Making figures...")
import setupplots
setupplots.thesis_format()
import matplotlib.pyplot as plt
import matplotlib.gridspec as mplgs
plt.ion()

# TCI values:
f_TCI = plt.figure(figsize=(0.75 * setupplots.TEXTWIDTH, 2 * 0.75 * setupplots.TEXTWIDTH / 1.618))
a_TCI = f_TCI.add_subplot(2, 1, 1)
a_TCI.errorbar(c.TCI_axis, c.TCI_y, yerr=c.TCI_err_y, label='measured', fmt='o', ms=setupplots.ms)
a_TCI.errorbar(
    c.TCI_axis, c.y_TCI_vals_MCMC_TS_only, yerr=c.err_y_TCI_vals_MCMC_TS_only,
    label=r'\textsc{cts}+\textsc{ets}', fmt='r^', ms=setupplots.ms
)
a_TCI.errorbar(
    c.TCI_axis, c.y_TCI_vals_MCMC_TS_TCI, yerr=c.err_y_TCI_vals_MCMC_TS_TCI,
    label=r'\textsc{cts}+\textsc{ets}+\textsc{tci}', fmt='gs', ms=setupplots.ms
)
a_TCI.errorbar(
    c.TCI_axis, c.y_TCI_vals_MCMC_ETS_TCI, yerr=c.err_y_TCI_vals_MCMC_ETS_TCI,
    label=r'\textsc{ets}+\textsc{tci}', fmt='yD', ms=setupplots.ms
)
# a_TCI.legend(loc='lower left', ncol=2)
a_TCI.set_title(r"Predicted and observed \textsc{tci} values")
# a_TCI.set_xlabel("channel")
a_TCI.set_ylabel(r"$nL$ [$\SI{e20}{m^{-2}}$]")
a_TCI.set_xticks(c.TCI_axis)
# a_TCI.set_xticklabels(c.TCI_labels)
plt.setp(a_TCI.get_xticklabels(), visible=False)

a_TCI_res = f_TCI.add_subplot(2, 1, 2, sharex=a_TCI)
a_TCI_res.errorbar(c.TCI_axis, c.TCI_y - scipy.asarray(c.TCI_y), yerr=c.TCI_err_y, label='measured', fmt='o', ms=setupplots.ms)
a_TCI_res.errorbar(
    c.TCI_axis, scipy.asarray(c.y_TCI_vals_MCMC_TS_only).ravel() - scipy.asarray(c.TCI_y), yerr=c.err_y_TCI_vals_MCMC_TS_only,
    label=r'\textsc{cts}+\textsc{ets}', fmt='r^', ms=setupplots.ms
)
a_TCI_res.errorbar(
    c.TCI_axis, scipy.asarray(c.y_TCI_vals_MCMC_TS_TCI).ravel() - scipy.asarray(c.TCI_y), yerr=c.err_y_TCI_vals_MCMC_TS_TCI,
    label=r'\textsc{cts}+\textsc{ets}+\textsc{tci}', fmt='gs', ms=setupplots.ms
)
a_TCI_res.errorbar(
    c.TCI_axis, scipy.asarray(c.y_TCI_vals_MCMC_ETS_TCI).ravel() - scipy.asarray(c.TCI_y), yerr=c.err_y_TCI_vals_MCMC_ETS_TCI,
    label=r'\textsc{ets}+\textsc{tci}', fmt='yD', ms=setupplots.ms
)
a_TCI_res.set_title(r"\textsc{tci} residuals")
a_TCI_res.set_xlabel("channel")
a_TCI_res.set_ylabel(r"$nL - nL_{\mathrm{obs}}$ [$\SI{e20}{m^{-2}}$]")
a_TCI_res.set_xticks(c.TCI_axis)
a_TCI_res.set_xticklabels([lbl[:-1] + r'\vphantom{nL_{0123456789}}$' for lbl in c.TCI_labels])
a_TCI_res.axhline(0.0, color='b')

a_TCI.set_xlim((c.TCI_axis[0] - 0.5, c.TCI_axis[-1] + 0.5))
f_TCI.savefig("neTCITestTCI_%d.pgf" % (c.shot,), bbox_inches='tight')
f_TCI.savefig("neTCITestTCI_%d.pdf" % (c.shot,), bbox_inches='tight')

# Put legend in seperate figure:
f_leg = plt.figure()
l = f_leg.legend(*a_TCI.get_legend_handles_labels(), ncol=2, loc='center')
f_leg.canvas.draw()
f_leg.set_figwidth(l.get_window_extent().width / 72)
f_leg.set_figheight(l.get_window_extent().height / 72)
f_leg.savefig("legTCI%d.pdf" % (c.shot,), bbox_inches='tight')
f_leg.savefig("legTCI%d.pgf" % (c.shot,), bbox_inches='tight')

# Profiles:
f = plt.figure(figsize=[setupplots.TEXTWIDTH, 0.5 * 3.5 * setupplots.TEXTWIDTH / 1.618])
gs = mplgs.GridSpec(4, 2, height_ratios=[1, 0.5, 1, 1])

a_val_MAP = f.add_subplot(gs[0, 0])
a_l_MAP = f.add_subplot(gs[1, 0], sharex=a_val_MAP)
a_grad_MAP = f.add_subplot(gs[2, 0], sharex=a_val_MAP)
a_a_L_MAP = f.add_subplot(gs[3, 0], sharex=a_val_MAP)

a_val_MCMC = f.add_subplot(gs[0, 1], sharex=a_val_MAP, sharey=a_val_MAP)
a_l_MCMC = f.add_subplot(gs[1, 1], sharex=a_val_MAP, sharey=a_l_MAP)
a_grad_MCMC = f.add_subplot(gs[2, 1], sharex=a_val_MAP, sharey=a_grad_MAP)
a_a_L_MCMC = f.add_subplot(gs[3, 1], sharex=a_val_MAP, sharey=a_a_L_MAP)

# MAP:
c.p_CTS.plot_data(ax=a_val_MAP, label_axes=False, label=r'\textsc{cts}', fmt='s', markersize=setupplots.ms)
c.p_ETS.plot_data(ax=a_val_MAP, label_axes=False, label=r'\textsc{ets}', fmt='s', markersize=setupplots.ms)
gptools.univariate_envelope_plot(
    c.roa_star, c.out_MAP_TS_only['mean_val'], c.out_MAP_TS_only['std_val'],
    ax=a_val_MAP, label=r'\textsc{cts}+\textsc{ets}', color='r', linewidth=setupplots.lw, ls='-',
    envelopes=[1,]
)
gptools.univariate_envelope_plot(
    c.roa_star, c.out_MAP_TS_TCI['mean_val'], c.out_MAP_TS_TCI['std_val'],
    ax=a_val_MAP, label=r'\textsc{cts}+\textsc{ets}+\textsc{tci}', color='g', linewidth=setupplots.lw, ls='--',
    envelopes=[1,]
)
gptools.univariate_envelope_plot(
    c.roa_star, c.out_MAP_ETS_TCI['mean_val'], c.out_MAP_ETS_TCI['std_val'],
    ax=a_val_MAP, label=r'\textsc{ets}+\textsc{tci}', color='y', linewidth=setupplots.lw, ls=':',
    envelopes=[1,]
)

a_l_MAP.plot(c.roa_star, c.l_MAP_TS_only, label=r'\textsc{cts}+\textsc{ets}', color='r', linewidth=setupplots.lw, ls='-')
a_l_MAP.plot(c.roa_star, c.l_MAP_TS_TCI, label=r'\textsc{cts}+\textsc{ets}+\textsc{tci}', color='g', linewidth=setupplots.lw, ls='--')
a_l_MAP.plot(c.roa_star, c.l_MAP_ETS_TCI, label=r'\textsc{ets}+\textsc{tci}', color='y', linewidth=setupplots.lw, ls=':')

gptools.univariate_envelope_plot(
    c.roa_star, c.out_MAP_TS_only['mean_grad'], c.out_MAP_TS_only['std_grad'],
    ax=a_grad_MAP, label=r'\textsc{cts}+\textsc{ets}', color='r', linewidth=setupplots.lw, ls='-',
    envelopes=[1,]
)
gptools.univariate_envelope_plot(
    c.roa_star, c.out_MAP_TS_TCI['mean_grad'], c.out_MAP_TS_TCI['std_grad'],
    ax=a_grad_MAP, label=r'\textsc{cts}+\textsc{ets}+\textsc{tci}', color='g', linewidth=setupplots.lw, ls='--',
    envelopes=[1,]
)
gptools.univariate_envelope_plot(
    c.roa_star, c.out_MAP_ETS_TCI['mean_grad'], c.out_MAP_ETS_TCI['std_grad'],
    ax=a_grad_MAP, label=r'\textsc{ets}+\textsc{tci}', color='y', linewidth=setupplots.lw, ls=':',
    envelopes=[1,]
)

gptools.univariate_envelope_plot(
    c.roa_star[c.core_mask], c.out_MAP_TS_only['mean_a_L'][c.core_mask], c.out_MAP_TS_only['std_a_L'][c.core_mask],
    ax=a_a_L_MAP, label=r'\textsc{cts}+\textsc{ets}', color='r', linewidth=setupplots.lw, ls='-',
    envelopes=[1,]
)
gptools.univariate_envelope_plot(
    c.roa_star[c.core_mask], c.out_MAP_TS_TCI['mean_a_L'][c.core_mask], c.out_MAP_TS_TCI['std_a_L'][c.core_mask],
    ax=a_a_L_MAP, label=r'\textsc{cts}+\textsc{ets}+\textsc{tci}', color='g', linewidth=setupplots.lw, ls='--',
    envelopes=[1,]
)
gptools.univariate_envelope_plot(
    c.roa_star[c.core_mask], c.out_MAP_ETS_TCI['mean_a_L'][c.core_mask], c.out_MAP_ETS_TCI['std_a_L'][c.core_mask],
    ax=a_a_L_MAP, label=r'\textsc{ets}+\textsc{tci}', color='y', linewidth=setupplots.lw, ls=':',
    envelopes=[1,]
)
a_a_L_MAP.axvspan(1.0, c.roa_star.max(), color='grey')

# MCMC:
c.p_CTS.plot_data(ax=a_val_MCMC, label_axes=False, label=r'\textsc{cts}', fmt='s', markersize=setupplots.ms)
c.p_ETS.plot_data(ax=a_val_MCMC, label_axes=False, label=r'\textsc{ets}', fmt='s', markersize=setupplots.ms)
gptools.univariate_envelope_plot(
    c.roa_star, c.out_MCMC_TS_only['mean_val'], c.out_MCMC_TS_only['std_val'],
    ax=a_val_MCMC, label=r'\textsc{cts}+\textsc{ets}', color='r', linewidth=setupplots.lw, ls='-',
    envelopes=[1,]
)
gptools.univariate_envelope_plot(
    c.roa_star, c.out_MCMC_TS_TCI['mean_val'], c.out_MCMC_TS_TCI['std_val'],
    ax=a_val_MCMC, label=r'\textsc{cts}+\textsc{ets}+\textsc{tci}', color='g', linewidth=setupplots.lw, ls='--',
    envelopes=[1,]
)
gptools.univariate_envelope_plot(
    c.roa_star, c.out_MCMC_ETS_TCI['mean_val'], c.out_MCMC_ETS_TCI['std_val'],
    ax=a_val_MCMC, label=r'\textsc{ets}+\textsc{tci}', color='y', linewidth=setupplots.lw, ls=':',
    envelopes=[1,]
)

gptools.univariate_envelope_plot(
    c.roa_star, c.mean_l_MCMC_TS_only, c.std_l_MCMC_TS_only,
    ax=a_l_MCMC, label=r'\textsc{cts}+\textsc{ets}', color='r', linewidth=setupplots.lw, ls='-',
    envelopes=[1,]
)
gptools.univariate_envelope_plot(
    c.roa_star, c.mean_l_MCMC_TS_TCI, c.std_l_MCMC_TS_TCI,
    ax=a_l_MCMC, label=r'\textsc{cts}+\textsc{ets}+\textsc{tci}', color='g', linewidth=setupplots.lw, ls='--',
    envelopes=[1,]
)
gptools.univariate_envelope_plot(
    c.roa_star, c.mean_l_MCMC_ETS_TCI, c.std_l_MCMC_ETS_TCI,
    ax=a_l_MCMC, label=r'\textsc{ets}+\textsc{tci}', color='y', linewidth=setupplots.lw, ls=':',
    envelopes=[1,]
)

gptools.univariate_envelope_plot(
    c.roa_star, c.out_MCMC_TS_only['mean_grad'], c.out_MCMC_TS_only['std_grad'],
    ax=a_grad_MCMC, label=r'\textsc{cts}+\textsc{ets}', color='r', linewidth=setupplots.lw, ls='-',
    envelopes=[1,]
)
gptools.univariate_envelope_plot(
    c.roa_star, c.out_MCMC_TS_TCI['mean_grad'], c.out_MCMC_TS_TCI['std_grad'],
    ax=a_grad_MCMC, label=r'\textsc{cts}+\textsc{ets}+\textsc{tci}', color='g', linewidth=setupplots.lw, ls='--',
    envelopes=[1,]
)
gptools.univariate_envelope_plot(
    c.roa_star, c.out_MCMC_ETS_TCI['mean_grad'], c.out_MCMC_ETS_TCI['std_grad'],
    ax=a_grad_MCMC, label=r'\textsc{ets}+\textsc{tci}', color='y', linewidth=setupplots.lw, ls=':',
    envelopes=[1,]
)

gptools.univariate_envelope_plot(
    c.roa_star[c.core_mask], c.out_MCMC_TS_only['mean_a_L'][c.core_mask], c.out_MCMC_TS_only['std_a_L'][c.core_mask],
    ax=a_a_L_MCMC, label=r'\textsc{cts}+\textsc{ets}', color='r', linewidth=setupplots.lw, ls='-',
    envelopes=[1,]
)
gptools.univariate_envelope_plot(
    c.roa_star[c.core_mask], c.out_MCMC_TS_TCI['mean_a_L'][c.core_mask], c.out_MCMC_TS_TCI['std_a_L'][c.core_mask],
    ax=a_a_L_MCMC, label=r'\textsc{cts}+\textsc{ets}+\textsc{tci}', color='g', linewidth=setupplots.lw, ls='--',
    envelopes=[1,]
)
gptools.univariate_envelope_plot(
    c.roa_star[c.core_mask], c.out_MCMC_ETS_TCI['mean_a_L'][c.core_mask], c.out_MCMC_ETS_TCI['std_a_L'][c.core_mask],
    ax=a_a_L_MCMC, label=r'\textsc{ets}+\textsc{tci}', color='y', linewidth=setupplots.lw, ls=':',
    envelopes=[1,]
)
a_a_L_MCMC.axvspan(1.0, c.roa_star.max(), color='grey')

a_val_MAP.set_ylabel(r'$n_{\mathrm{e}}$ [$\SI{e20}{m^{-3}}$]')
a_l_MAP.set_ylabel(r'$\ell$')
a_grad_MAP.set_ylabel(r'$\mathrm{d}n_{\mathrm{e}}/\mathrm{d}(r/a)$ [$\SI{e20}{m^{-3}}$]')
a_a_L_MAP.set_ylabel(r'$a/L_{n_{\mathrm{e}}}$')

a_a_L_MAP.set_xlabel('$r/a$')
a_a_L_MCMC.set_xlabel('$r/a$')

a_val_MAP.set_xlim([0, 1.1])
a_val_MAP.set_ylim([0, 2.5])
a_l_MAP.set_ylim([0.0, 1.5])
a_grad_MAP.set_ylim([-20, 0])
a_a_L_MAP.set_ylim([0, 1.0])

a_l_MAP.get_yaxis().set_ticks([0, 0.5, 1.0, 1.5])
a_grad_MAP.get_yaxis().set_ticks([0, -5, -10, -15, -20])

plt.setp(a_val_MAP.get_xticklabels(), visible=False)
plt.setp(a_l_MAP.get_xticklabels(), visible=False)
plt.setp(a_grad_MAP.get_xticklabels(), visible=False)
plt.setp(a_val_MCMC.get_xticklabels(), visible=False)
plt.setp(a_l_MCMC.get_xticklabels(), visible=False)
plt.setp(a_grad_MCMC.get_xticklabels(), visible=False)
plt.setp(a_val_MCMC.get_yticklabels(), visible=False)
plt.setp(a_l_MCMC.get_yticklabels(), visible=False)
plt.setp(a_grad_MCMC.get_yticklabels(), visible=False)
plt.setp(a_a_L_MCMC.get_yticklabels(), visible=False)

f.suptitle(r"Complete $n_{\mathrm{e}}$ profile, with and without \textsc{tci}")
a_val_MAP.set_title(r'\textsc{map}')
a_val_MCMC.set_title(r'\textsc{mcmc}')

f.subplots_adjust(left=0.12, bottom=0.08, right=0.99, top=0.92, wspace=0.07)
setupplots.apply_formatter(f)
f.canvas.draw()

f.savefig("neTCITestProf_%d.pgf" % (c.shot,))
f.savefig("neTCITestProf_%d.pdf" % (c.shot,))

# Put legend in seperate figure:
f_leg = plt.figure()
l = f_leg.legend(*a_val_MCMC.get_legend_handles_labels(), ncol=2, loc='center')
f_leg.canvas.draw()
f_leg.set_figwidth(l.get_window_extent().width / 72)
f_leg.set_figheight(l.get_window_extent().height / 72)
f_leg.savefig("legTCIne.pdf", bbox_inches='tight')
f_leg.savefig("legTCIne.pgf", bbox_inches='tight')

# Samplers:
gptools.plot_sampler(
    c.sampler_TS_only,
    labels=[r'$\sigma_f$ [$\SI{e20}{m^{-3}}$]', r'$\ell_1$', r'$\ell_2$', r'$\ell_{\text{w}}$', r'$x_0$'],
    burn=400, suptitle=r'\textsc{cts}+\textsc{ets}', fixed_width=setupplots.TEXTWIDTH,
    label_fontsize=11, chain_ytick_pad=0.8, chain_ticklabel_fontsize=11,
    ax_space=0.225, bottom_sep=0.125, suptitle_space=0.0, max_hist_ticks=6, cmap='plasma',
    hide_chain_ylabels=True, ticklabel_fontsize=11
    # points=c.params_MAP_TS_only
)
setupplots.apply_formatter(plt.gcf())
plt.savefig("TCIMargTSOnly.pdf", bbox_inches='tight')
plt.savefig("TCIMargTSOnly.pgf", bbox_inches='tight')

gptools.plot_sampler(
    c.sampler_TS_TCI,
    labels=[r'$\sigma_f$ [$\SI{e20}{m^{-3}}$]', r'$\ell_1$', r'$\ell_2$', r'$\ell_{\text{w}}$', r'$x_0$'],
    burn=400, suptitle=r'\textsc{cts}+\textsc{ets}+\textsc{tci}', fixed_width=setupplots.TEXTWIDTH,
    label_fontsize=11, chain_ytick_pad=0.8, chain_ticklabel_fontsize=11,
    ax_space=0.225, bottom_sep=0.125, suptitle_space=0.0, max_hist_ticks=6, cmap='plasma',
    hide_chain_ylabels=True, ticklabel_fontsize=11
    # points=c.params_MAP_TS_TCI
)
setupplots.apply_formatter(plt.gcf())
plt.savefig("TCIMargTSTCI.pdf", bbox_inches='tight')
plt.savefig("TCIMargTSTCI.pgf", bbox_inches='tight')

gptools.plot_sampler(
    c.sampler_ETS_TCI,
    labels=[r'$\sigma_f$ [$\SI{e20}{m^{-3}}$]', r'$\ell_1$', r'$\ell_2$', r'$\ell_{\text{w}}$', r'$x_0$'],
    burn=400, suptitle=r'\textsc{ets}+\textsc{tci}', fixed_width=setupplots.TEXTWIDTH,
    label_fontsize=11, chain_ytick_pad=0.8, chain_ticklabel_fontsize=11,
    ax_space=0.225, bottom_sep=0.125, suptitle_space=0.0, max_hist_ticks=6, cmap='plasma',
    hide_chain_ylabels=True, ticklabel_fontsize=11
    # points=c.params_MAP_ETS_TCI
)
setupplots.apply_formatter(plt.gcf())
plt.savefig("TCIMargETSTCI.pdf", bbox_inches='tight')
plt.savefig("TCIMargETSTCI.pgf", bbox_inches='tight')

# Generate MAP estimate summary table:
map_sum = setupplots.generate_latex_tabular_rows(
    ['%s', '%.5g', '%.5g', '%.5g', '%.5g', '%.5g'],
    [0],
    [r'\textsc{cts}+\textsc{ets}',] + list(c.params_MAP_TS_only),
    [r'\textsc{cts}+\textsc{ets}+\textsc{tci}',] + list(c.params_MAP_TS_TCI),
    [r'\textsc{ets}+\textsc{tci}',] + list(c.params_MAP_ETS_TCI),
)
with open('TCIMAPSum.tex', 'w') as tf:
    tf.write(map_sum)

# Generate posterior summary:
post_sum = setupplots.generate_post_sum(
    [c.params_MAP_TS_only, c.params_MAP_TS_TCI, c.params_MAP_ETS_TCI],
    [c.sampler_TS_only, c.sampler_TS_TCI, c.sampler_ETS_TCI],
    [400,] * 3,
    [
        [
            [r'\textsc{cts}+\textsc{ets}', r'$\sigma_f$ [$\SI{e20}{m^{-3}}$]'],
            [r'', r'$\ell_1$'],
            [r'', r'$\ell_2$'],
            [r'', r'$\ell_{\text{w}}$'],
            [r'', r'$x_0$']
        ],
        [
            [r'\textsc{cts}+\textsc{ets}+\textsc{tci}', r'$\sigma_f$ [$\SI{e20}{m^{-3}}$]'],
            [r'', r'$\ell_1$'],
            [r'', r'$\ell_2$'],
            [r'', r'$\ell_{\text{w}}$'],
            [r'', r'$x_0$']
        ],
        [
            [r'\textsc{ets}+\textsc{tci}', r'$\sigma_f$ [$\SI{e20}{m^{-3}}$]'],
            [r'', r'$\ell_1$'],
            [r'', r'$\ell_2$'],
            [r'', r'$\ell_{\text{w}}$'],
            [r'', r'$x_0$']
        ]
    ]
)
with open("TCIPostSum.tex", 'w') as tf:
    tf.write(post_sum)

c.median_unc_val_MAP_TS_only = 100*scipy.median(scipy.absolute(c.out_MAP_TS_only['std_val'][c.core_mask] / c.out_MAP_TS_only['mean_val'][c.core_mask]))
c.median_unc_grad_MAP_TS_only = 100*scipy.median(scipy.absolute(c.out_MAP_TS_only['std_grad'][c.core_mask] / c.out_MAP_TS_only['mean_grad'][c.core_mask]))
c.median_unc_a_L_MAP_TS_only = 100*scipy.median(scipy.absolute(c.out_MAP_TS_only['std_a_L'][c.core_mask] / c.out_MAP_TS_only['mean_a_L'][c.core_mask]))

c.median_unc_val_MAP_TS_TCI = 100*scipy.median(scipy.absolute(c.out_MAP_TS_TCI['std_val'][c.core_mask] / c.out_MAP_TS_TCI['mean_val'][c.core_mask]))
c.median_unc_grad_MAP_TS_TCI = 100*scipy.median(scipy.absolute(c.out_MAP_TS_TCI['std_grad'][c.core_mask] / c.out_MAP_TS_TCI['mean_grad'][c.core_mask]))
c.median_unc_a_L_MAP_TS_TCI = 100*scipy.median(scipy.absolute(c.out_MAP_TS_TCI['std_a_L'][c.core_mask] / c.out_MAP_TS_TCI['mean_a_L'][c.core_mask]))

c.median_unc_val_MAP_ETS_TCI = 100*scipy.median(scipy.absolute(c.out_MAP_ETS_TCI['std_val'][c.core_mask] / c.out_MAP_ETS_TCI['mean_val'][c.core_mask]))
c.median_unc_grad_MAP_ETS_TCI = 100*scipy.median(scipy.absolute(c.out_MAP_ETS_TCI['std_grad'][c.core_mask] / c.out_MAP_ETS_TCI['mean_grad'][c.core_mask]))
c.median_unc_a_L_MAP_ETS_TCI = 100*scipy.median(scipy.absolute(c.out_MAP_ETS_TCI['std_a_L'][c.core_mask] / c.out_MAP_ETS_TCI['mean_a_L'][c.core_mask]))

c.median_unc_val_MCMC_TS_only = 100*scipy.median(scipy.absolute(c.out_MCMC_TS_only['std_val'][c.core_mask] / c.out_MCMC_TS_only['mean_val'][c.core_mask]))
c.median_unc_grad_MCMC_TS_only = 100*scipy.median(scipy.absolute(c.out_MCMC_TS_only['std_grad'][c.core_mask] / c.out_MCMC_TS_only['mean_grad'][c.core_mask]))
c.median_unc_a_L_MCMC_TS_only = 100*scipy.median(scipy.absolute(c.out_MCMC_TS_only['std_a_L'][c.core_mask] / c.out_MCMC_TS_only['mean_a_L'][c.core_mask]))

c.median_unc_val_MCMC_TS_TCI = 100*scipy.median(scipy.absolute(c.out_MCMC_TS_TCI['std_val'][c.core_mask] / c.out_MCMC_TS_TCI['mean_val'][c.core_mask]))
c.median_unc_grad_MCMC_TS_TCI = 100*scipy.median(scipy.absolute(c.out_MCMC_TS_TCI['std_grad'][c.core_mask] / c.out_MCMC_TS_TCI['mean_grad'][c.core_mask]))
c.median_unc_a_L_MCMC_TS_TCI = 100*scipy.median(scipy.absolute(c.out_MCMC_TS_TCI['std_a_L'][c.core_mask] / c.out_MCMC_TS_TCI['mean_a_L'][c.core_mask]))

c.median_unc_val_MCMC_ETS_TCI = 100*scipy.median(scipy.absolute(c.out_MCMC_ETS_TCI['std_val'][c.core_mask] / c.out_MCMC_ETS_TCI['mean_val'][c.core_mask]))
c.median_unc_grad_MCMC_ETS_TCI = 100*scipy.median(scipy.absolute(c.out_MCMC_ETS_TCI['std_grad'][c.core_mask] / c.out_MCMC_ETS_TCI['mean_grad'][c.core_mask]))
c.median_unc_a_L_MCMC_ETS_TCI = 100*scipy.median(scipy.absolute(c.out_MCMC_ETS_TCI['std_a_L'][c.core_mask] / c.out_MCMC_ETS_TCI['mean_a_L'][c.core_mask]))

unc_sum = setupplots.generate_latex_tabular(
    ['%s', '%s', '%.4g', '%.4g', '%.4g'],
    [0, 2, 4],
    [r'\textsc{cts}+\textsc{ets}', '', r'\textsc{cts}+\textsc{ets}+\textsc{tci}', '', r'\textsc{ets}+\textsc{tci}', ''],
    [r'\textsc{map}', r'\textsc{mcmc}', r'\textsc{map}', r'\textsc{mcmc}', r'\textsc{map}', r'\textsc{mcmc}'],
    [c.median_unc_val_MAP_TS_only, c.median_unc_val_MCMC_TS_only, c.median_unc_val_MAP_TS_TCI, c.median_unc_val_MCMC_TS_TCI, c.median_unc_val_MAP_ETS_TCI, c.median_unc_val_MCMC_ETS_TCI],
    [c.median_unc_grad_MAP_TS_only, c.median_unc_grad_MCMC_TS_only, c.median_unc_grad_MAP_TS_TCI, c.median_unc_grad_MCMC_TS_TCI, c.median_unc_grad_MAP_ETS_TCI, c.median_unc_grad_MCMC_ETS_TCI],
    [c.median_unc_a_L_MAP_TS_only, c.median_unc_a_L_MCMC_TS_only, c.median_unc_a_L_MAP_TS_TCI, c.median_unc_a_L_MCMC_TS_TCI, c.median_unc_a_L_MAP_ETS_TCI, c.median_unc_a_L_MCMC_ETS_TCI],
)
with open('TCIUncSumTCI.tex', 'w') as tf:
    tf.write(unc_sum)
