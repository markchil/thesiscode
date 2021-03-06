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

# This script makes figures 2.20(b) and 2.22, which show the temperature
# profile for shot 1101014006, a basic L-mode plasma.

from __future__ import division
import profiletools
import scipy
import scipy.io
import scipy.interpolate
import eqtools
import gptools
import time
import copy
import cPickle as pkl

class Case(object):
    def __init__(self, shot, t_min, t_max, abscissa, use_NTH_grid=True):
        self.shot = shot
        self.t_min = t_min
        self.t_max = t_max
        self.abscissa = abscissa
        self.use_NTH_grid = use_NTH_grid

# Set shot, time range of interest:
start_time = time.time()
c = Case(1101014006, 0.965, 1.365, 'r/a', use_NTH_grid=True)
c.efit_tree = eqtools.CModEFITTree(c.shot)

if c.use_NTH_grid:
    fits_file = scipy.io.readsav('/Users/markchilenski/Dropbox/Research_and_Academic/GPR_paper/nth_samples_1101014006.save')
    efit_timebase = c.efit_tree.getTimeBase()
    ok_idxs = (efit_timebase >= c.t_min) & (efit_timebase <= c.t_max)
    t_NTH = efit_timebase[ok_idxs]
    # Use the grid from ne, since we don't want it moving around:
    R_NTH = fits_file.ne_fit.rmajor[0][:, 0]
    Z_mid_NTH = scipy.tile(c.efit_tree.getMagZ()[ok_idxs], (len(R_NTH), 1))
    R_mid_NTH = scipy.tile(R_NTH, (len(t_NTH), 1)).T
    roa_star = c.efit_tree.rz2rho(
        c.abscissa, R_mid_NTH, Z_mid_NTH, scipy.tile(t_NTH, (len(R_NTH), 1)), each_t=False
    )
    c.roa_star = scipy.mean(roa_star, axis=1)
    
    # Find the gradients and put everything on the same abscissa:
    R_raw = (fits_file.te_fit.rmajor[0][:, 32:72]).T
    Te_spline = (fits_file.te_fit.te_comb_fit[0][:, 32:72]).T
    
    Te_spline_corrected_abscissa = scipy.zeros_like(Te_spline)
    dTe_spline = scipy.zeros_like(Te_spline)
    dTe_dR_spline = scipy.zeros_like(Te_spline)
    for k in xrange(0, len(Te_spline)):
        spl = scipy.interpolate.InterpolatedUnivariateSpline(
            R_raw[k, :], Te_spline[k, :], k=3
        )
        Te_spline_corrected_abscissa[k, :] = spl(R_NTH)
        dTe_dR_spline[k, :] = spl(R_NTH, nu=1)
        
        dTe_spline[k, :] = scipy.interpolate.InterpolatedUnivariateSpline(
            c.roa_star, Te_spline_corrected_abscissa[k, :], k=3
        )(c.roa_star, nu=1)
    a = scipy.mean(c.efit_tree.getAOut()[ok_idxs])
    a_L_Te_spline = -a * dTe_dR_spline / Te_spline
    c.mean_val_spline = scipy.mean(Te_spline, axis=0)
    c.std_val_spline = scipy.std(Te_spline, axis=0, ddof=1)
    c.mean_grad_spline = scipy.mean(dTe_spline, axis=0)
    c.std_grad_spline = scipy.std(dTe_spline, axis=0, ddof=1)
    c.mean_a_L_spline = scipy.mean(a_L_Te_spline, axis=0)
    c.std_a_L_spline = scipy.std(a_L_Te_spline, axis=0, ddof=1)
else:
    c.roa_star = scipy.linspace(0.0, 1.2, 400)

c.core_mask = (c.roa_star <= 1.0)

# Get data:
print("Getting data...")
c.p_CTS = profiletools.TeCTS(
    c.shot, t_min=c.t_min, t_max=c.t_max, efit_tree=c.efit_tree, abscissa=c.abscissa
)
c.p_CTS.time_average(weighted=True)
c.p_ETS = profiletools.TeETS(
    c.shot, t_min=c.t_min, t_max=c.t_max, efit_tree=c.efit_tree, abscissa=c.abscissa
)
c.p_ETS.time_average(weighted=True)
c.p_GPC = profiletools.TeGPC(
    c.shot, t_min=c.t_min, t_max=c.t_max, efit_tree=c.efit_tree, abscissa='psinorm'
)
c.p_GPC.convert_abscissa(c.abscissa)
c.p_GPC.time_average(weighted=False)
c.p_GPC2 = profiletools.TeGPC2(
    c.shot, t_min=c.t_min, t_max=c.t_max, efit_tree=c.efit_tree, abscissa='psinorm'
)
c.p_GPC2.convert_abscissa(c.abscissa)
c.p_GPC2.time_average(weighted=False)
c.p_FRCECE = profiletools.TeFRCECE(
    c.shot, t_min=c.t_min, t_max=c.t_max, efit_tree=c.efit_tree, abscissa='psinorm'
)
c.p_FRCECE.convert_abscissa(c.abscissa)
c.p_FRCECE.time_average(weighted=False)

c.p = copy.deepcopy(c.p_CTS)
c.p.efit_tree = c.efit_tree
c.p.add_profile(c.p_ETS)
c.p.add_profile(c.p_GPC)
c.p.add_profile(c.p_GPC2)
c.p.add_profile(c.p_FRCECE)

# Create GP:
c.p.create_gp(constrain_at_limiter=False, constrain_slope_on_axis=(c.abscissa == 'r/a'))
b = 10.0
m = 1.0
be = 10.0
me = 0.5
c.p.gp.k.hyperprior = (
    gptools.UniformJointPrior([(0.0, 30.0)]) *
    gptools.GammaJointPrior(
        [1.0 + m * b, 1.0 + me * be, 1.0, 1.0 + 1.01 * 200.0],
        [b, be, 1.0 / 0.1, 200.0]
    )
)
c.roa_out = scipy.linspace(1.1, 1.4, 4)
zeros_out = scipy.zeros_like(c.roa_out)
c.p.gp.add_data(c.roa_out, zeros_out, err_y=0.01)
c.p.gp.add_data(c.roa_out, zeros_out, err_y=0.1, n=1)
# Find and evaluate the MAP estimate:
print("Finding MAP estimate...")
c.p.find_gp_MAP_estimate(random_starts=24, verbose=True)
c.MAP_params = c.p.gp.free_params[:]
print("Evaluating at MAP estimate...")
c.out_MAP = c.p.compute_a_over_L(c.roa_star, use_MCMC=False, return_prediction=True)
# Store l:
c.l_MAP = c.p.gp.k.l_func(c.roa_star, 0, *c.p.gp.k.params[1:])
# Marginalize with MCMC:
# Create the sampler separately so we can look at the burned/thinned cornerplot:
print("Running MCMC sampler...")
c.sampler = c.p.gp.sample_hyperparameter_posterior(nsamp=1000, sampler_a=32.0)
c.sampler.pool.close()
c.sampler.pool = None
print("Evaluating profiles with MCMC samples...")
c.out_MCMC = c.p.compute_a_over_L(
    c.roa_star, use_MCMC=True, return_prediction=True, sampler=c.sampler, burn=900, thin=100)
res = c.p.gp.compute_l_from_MCMC(c.roa_star, sampler=c.sampler, burn=900)
res = scipy.asarray(res)
c.mean_l = scipy.mean(res, axis=0)
c.std_l = scipy.std(res, axis=0, ddof=1)

with open('Te_profiletools.pkl', 'wb') as pf:
    pkl.dump(c, pf)

print("Plotting results...")
import setupplots
setupplots.thesis_format()
import matplotlib.pyplot as plt
plt.ion()
import matplotlib.gridspec as mplgs

gptools.plot_sampler(
    c.sampler,
    labels=[r'$\sigma_f$ [keV]', r'$\ell_1$', r'$\ell_2$', r'$\ell_{\text{w}}$', r'$x_0$'],
    burn=900, suptitle=r'$T_{\text{e}}$', fixed_width=setupplots.TEXTWIDTH,
    label_fontsize=11, chain_ytick_pad=0.8, chain_ticklabel_fontsize=11,
    ax_space=0.225, bottom_sep=0.12, suptitle_space=0.0, max_hist_ticks=6,
    cmap='plasma', hide_chain_ylabels=True, ticklabel_fontsize=11
)
setupplots.apply_formatter(plt.gcf())
plt.savefig("TeMarginalsNew.pdf", bbox_inches='tight')
plt.savefig("TeMarginalsNew.pgf", bbox_inches='tight')

# Set up plot:
lwd = setupplots.lw
ms = setupplots.ms

f = plt.figure(figsize=[0.5 * setupplots.TEXTWIDTH, 0.5 * 3.5 * setupplots.TEXTWIDTH / 1.618])
gs = mplgs.GridSpec(4, 1, height_ratios=[1, 0.5, 1, 1])
a_val = f.add_subplot(gs[0, :])
a_l = f.add_subplot(gs[1, :], sharex=a_val)
a_grad = f.add_subplot(gs[2, :], sharex=a_val)
a_a_L = f.add_subplot(gs[3, :], sharex=a_val)

# Plot value:
if c.use_NTH_grid:
    gptools.univariate_envelope_plot(
        c.roa_star, c.mean_val_spline, c.std_val_spline, ax=a_val, linewidth=lwd,
        linestyle='--', color='k', label='spline'
    )
gptools.univariate_envelope_plot(
    c.roa_star, c.out_MAP['mean_val'], c.out_MAP['std_val'], ax=a_val, linewidth=lwd,
    linestyle='-.', color='r', label=r'\textsc{map}'
)
gptools.univariate_envelope_plot(
    c.roa_star, c.out_MCMC['mean_val'], c.out_MCMC['std_val'], ax=a_val, linewidth=lwd,
    linestyle='-', color='b', label=r'\textsc{mcmc}'
)
c.p_CTS.plot_data(markersize=ms, fmt='bs', label=r'\textsc{cts}', ax=a_val, label_axes=False)
c.p_ETS.plot_data(markersize=ms, fmt='gs', label=r'\textsc{ets}', ax=a_val, label_axes=False)
c.p_GPC.plot_data(markersize=ms, fmt='c^', label=r'\textsc{gpc}', ax=a_val, label_axes=False)
c.p_GPC2.plot_data(markersize=ms, fmt='m^', label=r'\textsc{gpc}2', ax=a_val, label_axes=False)
c.p_FRCECE.plot_data(markersize=ms, fmt='r^', label=r'\textsc{frcece}', ax=a_val, label_axes=False)

# Plot l:
a_l.plot(c.roa_star, c.l_MAP, 'r', linewidth=lwd, linestyle='-.')
gptools.univariate_envelope_plot(
    c.roa_star, c.mean_l, c.std_l, ax=a_l, linewidth=lwd, linestyle='-', color='b'
)

# Plot gradient:
# gptools.univariate_envelope_plot(
#     c.roa_star, c.mean_grad_spline, c.std_grad_spline, ax=a_grad, linewidth=lwd,
#     linestyle='--', color='k'
# )
gptools.univariate_envelope_plot(
    c.roa_star, c.out_MAP['mean_grad'], c.out_MAP['std_grad'], ax=a_grad,
    linewidth=lwd, linestyle='-.', color='r'
)
gptools.univariate_envelope_plot(
    c.roa_star, c.out_MCMC['mean_grad'], c.out_MCMC['std_grad'], ax=a_grad,
    linewidth=lwd, linestyle='-', color='b'
)

# Plot a/L:
# gptools.univariate_envelope_plot(
#     c.roa_star[c.core_mask], c.mean_a_L_spline[c.core_mask], c.std_a_L_spline[c.core_mask],
#     ax=a_a_L, linewidth=lwd, linestyle='--', color='k'
# )
gptools.univariate_envelope_plot(
    c.roa_star[c.core_mask], c.out_MAP['mean_a_L'][c.core_mask], c.out_MAP['std_a_L'][c.core_mask],
    ax=a_a_L, linewidth=lwd, linestyle='-.', color='r'
)
gptools.univariate_envelope_plot(
    c.roa_star[c.core_mask], c.out_MCMC['mean_a_L'][c.core_mask], c.out_MCMC['std_a_L'][c.core_mask],
    ax=a_a_L, linewidth=lwd, linestyle='-', color='b'
)
a_a_L.axvspan(1.0, c.roa_star.max(), color='grey')

# Format plots:
a_val.set_ylabel(r'$T_{\mathrm{e}}$ [keV]')
a_l.set_ylabel(r'$\ell$')
a_grad.set_ylabel(r'$\mathrm{d}T_{\mathrm{e}}/\mathrm{d}(r/a)$ [keV]')
a_a_L.set_ylabel(r'$a/L_{T_{\mathrm{e}}}$')

a_a_L.set_xlabel('$r/a$')

a_val.set_xlim([0, 1.1])
a_val.set_ylim(bottom=0)
a_l.set_ylim([0, 1])
a_grad.set_ylim([-7, 0])
a_a_L.set_ylim([0, 6.5])

plt.setp(a_val.get_xticklabels(), visible=False)
plt.setp(a_l.get_xticklabels(), visible=False)
plt.setp(a_grad.get_xticklabels(), visible=False)

a_val.set_title(r"$T_{\mathrm{e}}$ profile")

f.subplots_adjust(hspace=0.15, left=0.24, bottom=0.075, right=0.98, top=0.96)
setupplots.apply_formatter(f)
f.canvas.draw()

f.savefig("TeFit.pdf")
f.savefig("TeFit.pgf")

# Put legend in seperate figure:
f_leg = plt.figure()
l = f_leg.legend(*a_val.get_legend_handles_labels(), ncol=4, loc='center')
f_leg.canvas.draw()
f_leg.set_figwidth(l.get_window_extent().width / 72)
f_leg.set_figheight(l.get_window_extent().height / 72)
f_leg.savefig("leg1101014006.pdf", bbox_inches='tight')
f_leg.savefig("leg1101014006.pgf", bbox_inches='tight')

# Make tables:
post_sum = setupplots.generate_post_sum(
    [c.MAP_params,],
    [c.sampler,],
    [900,],
    [[[r'$T_{\text{e}}$', r'$\sigma_f$ [keV]'], ['', r'$\ell_1$'], ['', r'$\ell_2$'], ['', r'$\ell_{\text{w}}$'], ['', r'$x_0$']]]
)
with open('TePostSum1101014006.tex', 'w') as tf:
    tf.write(post_sum)

c.median_unc_val_spline = 100*scipy.median(scipy.absolute(c.std_val_spline[c.core_mask] / c.mean_val_spline[c.core_mask]))
c.median_unc_grad_spline = 100*scipy.median(scipy.absolute(c.std_grad_spline[c.core_mask] / c.mean_grad_spline[c.core_mask]))
c.median_unc_a_L_spline = 100*scipy.median(scipy.absolute(c.std_a_L_spline[c.core_mask] / c.mean_a_L_spline[c.core_mask]))

c.median_unc_val_MAP = 100*scipy.median(scipy.absolute(c.out_MAP['std_val'][c.core_mask] / c.out_MAP['mean_val'][c.core_mask]))
c.median_unc_grad_MAP = 100*scipy.median(scipy.absolute(c.out_MAP['std_grad'][c.core_mask] / c.out_MAP['mean_grad'][c.core_mask]))
c.median_unc_a_L_MAP = 100*scipy.median(scipy.absolute(c.out_MAP['std_a_L'][c.core_mask] / c.out_MAP['mean_a_L'][c.core_mask]))

c.median_unc_val_MCMC = 100*scipy.median(scipy.absolute(c.out_MCMC['std_val'][c.core_mask] / c.out_MCMC['mean_val'][c.core_mask]))
c.median_unc_grad_MCMC = 100*scipy.median(scipy.absolute(c.out_MCMC['std_grad'][c.core_mask] / c.out_MCMC['mean_grad'][c.core_mask]))
c.median_unc_a_L_MCMC = 100*scipy.median(scipy.absolute(c.out_MCMC['std_a_L'][c.core_mask] / c.out_MCMC['mean_a_L'][c.core_mask]))

unc_sum = setupplots.generate_latex_tabular(
    ['%s', '%s', '%.4g', '%.4g', '%.4g'],
    [0],
    [r'$T_{\text{e}}$', '', ''],
    ['spline', r'\textsc{map}', r'\textsc{mcmc}'],
    [c.median_unc_val_spline, c.median_unc_val_MAP, c.median_unc_val_MCMC],
    [c.median_unc_grad_spline, c.median_unc_grad_MAP, c.median_unc_grad_MCMC],
    [c.median_unc_a_L_spline, c.median_unc_a_L_MAP, c.median_unc_a_L_MCMC],
)
with open('TeUncSum1101014006.tex', 'w') as tf:
    tf.write(unc_sum)

acor_sum = r'$T_{\text{e}}$'
for v in c.sampler.acor:
    acor_sum += ' & %.4g' % (v,)
acor_sum += r'\\'
with open('TeSamplerAcor1101014006.tex', 'w') as tf:
    tf.write(acor_sum)

# Print diagnostics:
print("Analysis complete.\nShot:\t%d\nt_min:\t%.3f\nt_max:\t%.3f" % (c.shot, c.t_min, c.t_max))
print("MCMC sampler mean acceptance fraction: %.2f%%" % (100*scipy.mean(c.sampler.acceptance_fraction),))
print("Median relative discrepancy between MCMC and spline estimate: %.2f%%\n"
      "Median relative discrepancy between MCMC and MAP estimate: %.2f%%" %
      (2*100*scipy.median(scipy.absolute(c.out_MCMC['mean_val'][c.core_mask] - c.mean_val_spline[c.core_mask]) /
            (c.out_MCMC['mean_val'][c.core_mask] + c.mean_val_spline[c.core_mask])),
       2*100*scipy.median(scipy.absolute(c.out_MCMC['mean_val'][c.core_mask] - c.out_MAP['mean_val'][c.core_mask]) /
            (c.out_MCMC['mean_val'][c.core_mask] + c.out_MAP['mean_val'][c.core_mask]))))

print("Elapsed time (including plotting): %.1fs" % (time.time() - start_time,))
