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

# This script makes figures 2.36, 2.37, 2.38 and 2.39, which show the fits to
# H-mode ne profiles with and without the mtanh mean function included.

from __future__ import division
import scipy
import profiletools
import eqtools
import gptools
import copy
import cPickle as pkl

class Case(object):
    def __init__(self, shot, t_min, t_max, abscissa, constrain_at_limiter):
        self.shot = shot
        self.t_min = t_min
        self.t_max = t_max
        self.abscissa = abscissa
        self.constrain_at_limiter = constrain_at_limiter

c = Case(1110201035, 1.35, 1.5, 'psinorm', False)
c.roa_star = scipy.linspace(0, 1.1, 400)
c.core_mask = c.roa_star <= 1.0
c.constrain_at_limiter = False

print("Fetching TS...")
# Apply a 13mm vertical shift to correct for ETS fiber location:
c.p_CTS = profiletools.neCTS(c.shot, t_min=c.t_min, t_max=c.t_max, abscissa=c.abscissa, Z_shift=0.0)
c.p_ETS = profiletools.neETS(c.shot, t_min=c.t_min, t_max=c.t_max, abscissa=c.abscissa, Z_shift=-13e-3)
c.p_CTS.time_average(weighted=True)
c.p_ETS.time_average(weighted=True)
c.p = copy.deepcopy(c.p_CTS)
c.p.add_profile(c.p_ETS)

print("Fitting without mean function...")
c.p.create_gp(constrain_at_limiter=True, constrain_slope_on_axis=False)
c.p.find_gp_MAP_estimate(verbose=True)
c.params = c.p.gp.free_params[:]
c.sampler = c.p.gp.sample_hyperparameter_posterior(nsamp=1000, sampler_a=16.0)
c.sampler.pool.close()
c.sampler.pool = None
c.out = c.p.compute_a_over_L(
    c.roa_star, return_prediction=True, use_MCMC=True, sampler=c.sampler, burn=800, thin=100
)

print("Fitting with mean function...")
c.p_mtanh = copy.deepcopy(c.p)
k = gptools.SquaredExponentialKernel(
    hyperprior=gptools.UniformJointPrior([(0.0, 30.0),]) * gptools.GammaJointPriorAlt([1.0], [0.3])
)
c.p_mtanh.create_gp(
    constrain_at_limiter=c.constrain_at_limiter, constrain_slope_on_axis=False,
    mu=gptools.MtanhMeanFunction1d(), k=k, use_hyper_deriv=True
)
c.p_mtanh.find_gp_MAP_estimate(verbose=True)
c.params_mtanh = c.p_mtanh.gp.free_params[:]
c.p_mtanh.gp.use_hyper_deriv = False
c.sampler_mtanh = c.p_mtanh.gp.sample_hyperparameter_posterior(nsamp=1000, sampler_a=4.0)
c.sampler_mtanh.pool.close()
c.sampler_mtanh.pool = None
c.out_mtanh = c.p_mtanh.compute_a_over_L(
    c.roa_star, return_prediction=True, return_mean_func=True, use_MCMC=True,
    sampler=c.sampler_mtanh, burn=800, thin=100
)

print("Fitting with ONLY mean function...")
k = gptools.ZeroKernel()
c.p_mtanh_only = copy.deepcopy(c.p)
c.p_mtanh_only.create_gp(
    constrain_at_limiter=c.constrain_at_limiter, constrain_slope_on_axis=False,
    mu=gptools.MtanhMeanFunction1d(), k=k, use_hyper_deriv=True
)
c.p_mtanh_only.find_gp_MAP_estimate(verbose=True)
c.params_mtanh_only = c.p_mtanh_only.gp.free_params[:]
c.p_mtanh_only.gp.use_hyper_deriv = False
c.sampler_mtanh_only = c.p_mtanh_only.gp.sample_hyperparameter_posterior(nsamp=1000, sampler_a=16.0)
c.sampler_mtanh_only.pool.close()
c.sampler_mtanh_only.pool = None
c.out_mtanh_only = c.p_mtanh_only.compute_a_over_L(
    c.roa_star, return_prediction=True, return_mean_func=True, use_MCMC=True,
    sampler=c.sampler_mtanh_only, burn=800, thin=100
)

with open('ne_mtanh.pkl', 'wb') as pf:
    pkl.dump(c, pf)

print("Making figures...")
import setupplots
setupplots.thesis_format()
import matplotlib.pyplot as plt
import matplotlib.gridspec as mplgs
plt.ion()

gptools.plot_sampler(
    c.sampler,
    labels=[r'$\sigma_f$ [$\SI{e20}{m^{-3}}$]', r'$\ell_1$', r'$\ell_2$', r'$\ell_{\text{w}}$', '$x_0$'],
    burn=800, suptitle=r'Gibbs+$\tanh$', fixed_width=6.5,
    label_fontsize=11, chain_ytick_pad=0.8, chain_ticklabel_fontsize=11,
    suptitle_space=0.0, ax_space=0.2, bottom_sep=0.09,
    cmap='plasma', hide_chain_ylabels=True, ticklabel_fontsize=11,
    max_hist_ticks=6
)
setupplots.apply_formatter(plt.gcf())
plt.savefig('mufit_gibbs_tanh_post.pdf', bbox_inches='tight')
plt.savefig('mufit_gibbs_tanh_post.pgf', bbox_inches='tight')

gptools.plot_sampler(
    c.sampler_mtanh,
    labels=[r'$\sigma_f$' + '\n' + r'[$\SI{e20}{m^{-3}}$]', r'$\ell$', '$x_0$', r'$\delta$', r'$\alpha$', r'$h$' + '\n' + r'[$\SI{e20}{m^{-3}}$]', r'$b$' + '\n' + r'[$\SI{e20}{m^{-3}}$]'],
    burn=800, suptitle=r'\textsc{se}+mtanh', fixed_width=6.5,
    label_fontsize=11, chain_ytick_pad=0.8, chain_ticklabel_fontsize=11,
    suptitle_space=0.0, ax_space=0.25, bottom_sep=0.12, max_hist_ticks=5,
    cmap='plasma', hide_chain_ylabels=True, ticklabel_fontsize=11
)
setupplots.apply_formatter(plt.gcf())
plt.savefig('mufit_se_mtanh_post.pdf', bbox_inches='tight')
plt.savefig('mufit_se_mtanh_post.pgf', bbox_inches='tight')

gptools.plot_sampler(
    c.sampler_mtanh_only,
    labels=['$x_0$', r'$\delta$', r'$\alpha$', r'$h$' + '\n' + r'[$\SI{e20}{m^{-3}}$]', r'$b$' + '\n' + r'[$\SI{e20}{m^{-3}}$]'],
    burn=800, suptitle=r'mtanh only', fixed_width=6.5,
    label_fontsize=11, chain_ytick_pad=0.8, chain_ticklabel_fontsize=11,
    suptitle_space=0.0, ax_space=0.2, bottom_sep=0.12, max_hist_ticks=6,
    cmap='plasma', hide_chain_ylabels=True, ticklabel_fontsize=11
)
setupplots.apply_formatter(plt.gcf())
plt.savefig('mufit_mtanh_only_post.pdf', bbox_inches='tight')
plt.savefig('mufit_mtanh_only_post.pgf', bbox_inches='tight')

lw = setupplots.lw
# Global:
f = plt.figure(figsize=[setupplots.TEXTWIDTH, 0.5 * 3.5 * setupplots.TEXTWIDTH / 1.618])
gs = mplgs.GridSpec(4, 2, height_ratios=[1, 0.5, 1, 1])
a_val = f.add_subplot(gs[0, 0])
a_res = f.add_subplot(gs[1, 0], sharex=a_val)
a_grad = f.add_subplot(gs[2, 0], sharex=a_val)
a_a_L = f.add_subplot(gs[3, 0], sharex=a_val)

c.p_CTS.plot_data(ax=a_val, label_axes=False, label=r'\textsc{cts}', fmt='bs', markersize=setupplots.ms)
c.p_ETS.plot_data(ax=a_val, label_axes=False, label=r'\textsc{ets}', fmt='gs', markersize=setupplots.ms)
gptools.univariate_envelope_plot(
    c.roa_star, c.out['mean_val'], c.out['std_val'], ax=a_val, label='Gibbs+tanh',
    color='r', lw=lw, ls='-', envelopes=[1.0,]
)
gptools.univariate_envelope_plot(
    c.roa_star, c.out_mtanh['mean_val'], c.out_mtanh['std_val'], ax=a_val,
    label=r'\textsc{se}+mtanh', color='b', lw=lw, ls='--', envelopes=[1.0,]
)
gptools.univariate_envelope_plot(
    c.roa_star, c.out_mtanh['mean_func_val'], c.out_mtanh['std_func_val'], ax=a_val,
    label='mean function', color='k', base_alpha=0.1, lw=lw, ls=':', envelopes=[1.0,]
)
gptools.univariate_envelope_plot(
    c.roa_star, c.out_mtanh_only['mean_val'], c.out_mtanh_only['std_val'], ax=a_val,
    label='mtanh only', color='g', lw=lw, ls='-.', envelopes=[1.0,]
)

gptools.univariate_envelope_plot(
    c.roa_star, c.out_mtanh['mean_without_func_val'], c.out_mtanh['std_without_func_val'],
    ax=a_res, label='residual', color='k', base_alpha=0.1, lw=lw, ls=':', envelopes=[1.0,]
)

gptools.univariate_envelope_plot(
    c.roa_star, c.out['mean_grad'], c.out['std_grad'], ax=a_grad, label='Gibbs+tanh',
    color='r', lw=lw, ls='-', envelopes=[1.0,]
)
gptools.univariate_envelope_plot(
    c.roa_star, c.out_mtanh['mean_grad'], c.out_mtanh['std_grad'], ax=a_grad,
    label=r'\textsc{se}+mtanh', color='b', lw=lw, ls='--', envelopes=[1.0,]
)
gptools.univariate_envelope_plot(
    c.roa_star, c.out_mtanh_only['mean_grad'], c.out_mtanh_only['std_grad'], ax=a_grad,
    label='mtanh only', color='g', lw=lw, ls='-.', envelopes=[1.0,]
)

gptools.univariate_envelope_plot(
    c.roa_star[c.core_mask], c.out['mean_a_L'][c.core_mask], c.out['std_a_L'][c.core_mask],
    ax=a_a_L, label='Gibbs+tanh', color='r', lw=lw, ls='-', envelopes=[1.0,]
)
gptools.univariate_envelope_plot(
    c.roa_star[c.core_mask], c.out_mtanh['mean_a_L'][c.core_mask], c.out_mtanh['std_a_L'][c.core_mask],
    ax=a_a_L, label=r'\textsc{se}+mtanh', color='b', lw=lw, ls='--', envelopes=[1.0,]
)
gptools.univariate_envelope_plot(
    c.roa_star[c.core_mask], c.out_mtanh_only['mean_a_L'][c.core_mask], c.out_mtanh_only['std_a_L'][c.core_mask],
    ax=a_a_L, label='mtanh only', color='g', lw=lw, ls='-.', envelopes=[1.0,]
)
a_a_L.axvspan(1.0, c.roa_star.max(), color='grey')

a_val.set_ylabel(r'$n_{\mathrm{e}}$' + '\n' + r'[$\SI{e20}{m^{-3}}$]')
a_res.set_ylabel(r'$r_{\textsc{gp}}$' + '\n' + r'[$\SI{e20}{m^{-3}}$]')
a_grad.set_ylabel(r'$\mathrm{d}n_{\mathrm{e}}/\mathrm{d}\psi_{\mathrm{n}}$' + '\n' + r'[$\SI{e20}{m^{-3}}$]')
a_a_L.set_ylabel(r'$a/L_{n_{\mathrm{e}}}$')

a_a_L.set_xlabel(r'$\psi_{\mathrm{n}}$')

a_val.set_xlim([0, 1.0])
a_val.set_ylim([1.2, 2.75])
a_res.set_ylim([-4, 1])
a_grad.set_ylim([-2.5, -0.5])
a_a_L.set_ylim([0, 1.5])

plt.setp(a_val.get_xticklabels(), visible=False)
plt.setp(a_res.get_xticklabels(), visible=False)
plt.setp(a_grad.get_xticklabels(), visible=False)

f.suptitle(r"H-mode $n_{\mathrm{e}}$ profile")
a_val.set_title("core")

# Pedestal:
a_val_ped = f.add_subplot(gs[0, 1])
a_res_ped = f.add_subplot(gs[1, 1], sharex=a_val_ped)
a_grad_ped = f.add_subplot(gs[2, 1], sharex=a_val_ped)
a_a_L_ped = f.add_subplot(gs[3, 1], sharex=a_val_ped)

c.p_CTS.plot_data(ax=a_val_ped, label_axes=False, label=r'\textsc{cts}', fmt='bs', markersize=setupplots.ms)
c.p_ETS.plot_data(ax=a_val_ped, label_axes=False, label=r'\textsc{ets}', fmt='gs', markersize=setupplots.ms)
gptools.univariate_envelope_plot(
    c.roa_star, c.out['mean_val'], c.out['std_val'], ax=a_val_ped, label='Gibbs+tanh',
    color='r', lw=lw, ls='-', envelopes=[1.0,]
)
gptools.univariate_envelope_plot(
    c.roa_star, c.out_mtanh['mean_val'], c.out_mtanh['std_val'], ax=a_val_ped,
    label=r'\textsc{se}+mtanh', color='b', lw=lw, ls='--', envelopes=[1.0,]
)
gptools.univariate_envelope_plot(
    c.roa_star, c.out_mtanh['mean_func_val'], c.out_mtanh['std_func_val'], ax=a_val_ped,
    label='mean function', color='k', base_alpha=0.1, lw=lw, ls=':', envelopes=[1.0,]
)
gptools.univariate_envelope_plot(
    c.roa_star, c.out_mtanh_only['mean_val'], c.out_mtanh_only['std_val'], ax=a_val_ped,
    label='mtanh only', color='g', lw=lw, ls='-.', envelopes=[1.0,]
)

gptools.univariate_envelope_plot(
    c.roa_star, c.out_mtanh['mean_without_func_val'], c.out_mtanh['std_without_func_val'],
    ax=a_res_ped, label='residual', color='k', base_alpha=0.1, lw=lw, ls=':', envelopes=[1.0,]
)

gptools.univariate_envelope_plot(
    c.roa_star, c.out['mean_grad'], c.out['std_grad'], ax=a_grad_ped, label='Gibbs+tanh',
    color='r', lw=lw, ls='-', envelopes=[1.0,]
)
gptools.univariate_envelope_plot(
    c.roa_star, c.out_mtanh['mean_grad'], c.out_mtanh['std_grad'], ax=a_grad_ped,
    label=r'\textsc{se}+mtanh', color='b', lw=lw, ls='--', envelopes=[1.0,]
)
gptools.univariate_envelope_plot(
    c.roa_star, c.out_mtanh_only['mean_grad'], c.out_mtanh_only['std_grad'], ax=a_grad_ped,
    label='mtanh only', color='g', lw=lw, ls='-.', envelopes=[1.0,]
)

gptools.univariate_envelope_plot(
    c.roa_star[c.core_mask], c.out['mean_a_L'][c.core_mask], c.out['std_a_L'][c.core_mask],
    ax=a_a_L_ped, label='Gibbs+tanh', color='r', lw=lw, ls='-', envelopes=[1.0,]
)
gptools.univariate_envelope_plot(
    c.roa_star[c.core_mask], c.out_mtanh['mean_a_L'][c.core_mask], c.out_mtanh['std_a_L'][c.core_mask],
    ax=a_a_L_ped, label=r'\textsc{se}+mtanh', color='b', lw=lw, ls='--', envelopes=[1.0,]
)
gptools.univariate_envelope_plot(
    c.roa_star[c.core_mask], c.out_mtanh_only['mean_a_L'][c.core_mask], c.out_mtanh_only['std_a_L'][c.core_mask],
    ax=a_a_L_ped, label='mtanh only', color='g', lw=lw, ls='-.', envelopes=[1.0,]
)
a_a_L_ped.axvspan(1.0, c.roa_star.max(), color='grey')

a_a_L_ped.set_xlabel(r'$\psi_{\mathrm{n}}$')

a_val_ped.set_xlim([0.85, 1.1])
a_val_ped.set_ylim([0, 1.75])
a_res_ped.set_ylim([-1, 0.5])
a_grad_ped.set_ylim([-30, 0])
a_a_L_ped.set_ylim([0, 5.0])

plt.setp(a_val_ped.get_xticklabels(), visible=False)
plt.setp(a_res_ped.get_xticklabels(), visible=False)
plt.setp(a_grad_ped.get_xticklabels(), visible=False)

a_val_ped.set_title('pedestal')

a_res.yaxis.set_major_locator(plt.MaxNLocator(nbins=5))
a_res_ped.yaxis.set_major_locator(plt.MaxNLocator(nbins=5))

f.subplots_adjust(left=0.15, bottom=0.075, right=0.97, top=0.92)
setupplots.apply_formatter(f)
f.canvas.draw()

f.savefig('mufit_prof.pdf', bbox_inches='tight')
f.savefig('mufit_prof.pgf', bbox_inches='tight')

# Put legend in separate figure:
f_leg = plt.figure()
l = f_leg.legend(*a_val.get_legend_handles_labels(), ncol=3, loc='center')
f_leg.canvas.draw()
f_leg.set_figwidth(l.get_window_extent().width / 72)
f_leg.set_figheight(l.get_window_extent().height / 72)
f_leg.savefig("mufit_leg.pdf", bbox_inches='tight')
f_leg.savefig("mufit_leg.pgf", bbox_inches='tight')

# Print summary:

c.median_unc_val = 100*scipy.median(scipy.absolute(c.out['std_val'][c.core_mask] / c.out['mean_val'][c.core_mask]))
c.median_unc_grad = 100*scipy.median(scipy.absolute(c.out['std_grad'][c.core_mask] / c.out['mean_grad'][c.core_mask]))
c.median_unc_a_L = 100*scipy.median(scipy.absolute(c.out['std_a_L'][c.core_mask] / c.out['mean_a_L'][c.core_mask]))

c.median_unc_val_mtanh = 100*scipy.median(scipy.absolute(c.out_mtanh['std_val'][c.core_mask] / c.out_mtanh['mean_val'][c.core_mask]))
c.median_unc_grad_mtanh = 100*scipy.median(scipy.absolute(c.out_mtanh['std_grad'][c.core_mask] / c.out_mtanh['mean_grad'][c.core_mask]))
c.median_unc_a_L_mtanh = 100*scipy.median(scipy.absolute(c.out_mtanh['std_a_L'][c.core_mask] / c.out_mtanh['mean_a_L'][c.core_mask]))

c.median_unc_val_mtanh_only = 100*scipy.median(scipy.absolute(c.out_mtanh_only['std_val'][c.core_mask] / c.out_mtanh_only['mean_val'][c.core_mask]))
c.median_unc_grad_mtanh_only = 100*scipy.median(scipy.absolute(c.out_mtanh_only['std_grad'][c.core_mask] / c.out_mtanh_only['mean_grad'][c.core_mask]))
c.median_unc_a_L_mtanh_only = 100*scipy.median(scipy.absolute(c.out_mtanh_only['std_a_L'][c.core_mask] / c.out_mtanh_only['mean_a_L'][c.core_mask]))

unc_sum = setupplots.generate_latex_tabular(
    ['%s', '%.4g', '%.4g', '%.4g'],
    [0],
    [r'Gibbs+$\tanh$', r'\textsc{se}+$\mtanh$', r'$\mtanh$ only'],
    [c.median_unc_val, c.median_unc_val_mtanh, c.median_unc_val_mtanh_only],
    [c.median_unc_grad, c.median_unc_grad_mtanh, c.median_unc_grad_mtanh_only],
    [c.median_unc_a_L, c.median_unc_a_L_mtanh, c.median_unc_a_L_mtanh_only],
)
with open('ne_mtanh_rel_unc.tex', 'w') as tf:
    tf.write(unc_sum)

post_sum = setupplots.generate_post_sum(
    [c.params, c.params_mtanh, c.params_mtanh_only],
    [c.sampler, c.sampler_mtanh, c.sampler_mtanh_only],
    [800, 800, 800],
    [
        [[r'Gibbs+$\tanh$', r'$\sigma_f$ [$\SI{e20}{m^{-3}}$]'], ['', r'$\ell_1$'], ['', r'$\ell_2$'], ['', r'$\ell_{\text{w}}$'], ['', r'$x_0$']],
        [[r'\textsc{se}+$\mtanh$', r'$\sigma_f$ [$\SI{e20}{m^{-3}}$]'], ['', r'$\ell$'], ['', r'$x_0$'], ['', r'$\delta$'], ['', r'$\alpha$'], ['', r'$h$ [$\SI{e20}{m^{-3}}$]'], ['', r'$b$ [$\SI{e20}{m^{-3}}$]']],
        [[r'$\mtanh$ only', r'$x_0$'], ['', r'$\delta$'], ['', r'$\alpha$'], ['', r'$h$ [$\SI{e20}{m^{-3}}$]'], ['', r'$b$ [$\SI{e20}{m^{-3}}$]']],
    ]
)
with open('ne_mtanh_post_sum.tex', 'w') as tf:
    tf.write(post_sum)
