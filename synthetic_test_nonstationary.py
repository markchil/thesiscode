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

# This script makes figures 2.16, 2.17, 2.18, A.1, A.2, A.3, A.4, A.5, A.6,
# A.7, A.8, A.9 and A.10, which show synthetic profile data fit with a variety
# of nonstationary covariance kernels.

from __future__ import division
import gptools
import scipy
import scipy.interpolate
import numpy.random
import cPickle as pkl

# ne-like shape:
x_control = scipy.asarray([0, 0.3,  0.6, 0.85, 0.965, 1.02, 1.05, 1.08,   1.1, 1.125, 1.15])
y_control = scipy.asarray([1, 0.85, 0.7, 0.65, 0.6,   0.4,  0.08, 0.0005, 0.0, 0.0,   0.0 ])

spl = scipy.interpolate.InterpolatedUnivariateSpline(
    scipy.concatenate((-scipy.flipud(x_control[1:]), x_control)),
    scipy.concatenate((scipy.flipud(y_control[1:]), y_control)),
    k=3
)

x_grid = scipy.linspace(0, 1.1, 100)
y_grid = spl(x_grid)
dy_grid = spl(x_grid, nu=1)

edge_start = 0.97
core_rel_err = 0.03
edge_abs_err = 0.1

# Number of points in each system:
n = 15

# Make eval grids and synthetic, noisy data:
x_star = scipy.concatenate((x_grid, x_grid))
n_star = scipy.concatenate((scipy.zeros_like(x_grid), scipy.ones_like(x_grid)))

x_core = scipy.linspace(0, edge_start, n)
x_edge = scipy.linspace(edge_start, 1.07, n + 1)[1:]
x_meas = scipy.concatenate((x_core, x_edge))
y_meas = spl(x_meas)
err_y_meas = scipy.zeros_like(y_meas)
err_y_meas[:n] = core_rel_err
err_y_meas *= y_meas
err_y_meas[n:] = edge_abs_err

# First seed the generator to make the test repeatable:
RS = numpy.random.RandomState(8675309)
y_meas += RS.randn(y_meas.shape[0]) * err_y_meas
y_meas[y_meas < 0] = 0.0

class TestCase(object):
    """Object to store each test case and its parameters.
    """
    def __init__(self, k, lbl, tlbl, hp_labels, a=2.0, nsamp=500, burn=400, thin=100, MAP_starts=300):
        self.k = k
        self.lbl = lbl
        self.tlbl = tlbl
        self.hp_labels = hp_labels
        self.a = a
        self.nsamp = nsamp
        self.burn = burn
        self.thin = thin
        self.MAP_starts = MAP_starts

# Gibbs+tanh:
hp = (
    gptools.UniformJointPrior([(0, 10),]) *
    gptools.GammaJointPriorAlt([1.0, 0.5, 0.0, 1.0], [0.3, 0.25, 0.1, 0.05])
)
k_gibbs_tanh = gptools.GibbsKernel1dTanh(hyperprior=hp)

# Gibbs+1 exp-gauss:
n_gaussians = 1
hp = (
    gptools.UniformJointPrior([(0, 4),]) *
    gptools.GammaJointPriorAlt([1.0,], [0.3,]) *
    gptools.SortedUniformJointPrior(n_gaussians, x_grid.min(), x_grid.max()) *
    gptools.UniformJointPrior([(0, 2 * (x_grid.max() - x_grid.min())),] * n_gaussians) *
    gptools.NormalJointPrior([0.0,] * n_gaussians, [1.0,] * n_gaussians)
)
k_gibbs_exp_1 = gptools.GibbsKernel1dExpGauss(n_gaussians, hyperprior=hp)

# Gibbs+4 exp-gauss:
n_gaussians = 4
hp = (
    gptools.UniformJointPrior([(0, 4),]) *
    gptools.GammaJointPriorAlt([1.0,], [0.3,]) *
    gptools.SortedUniformJointPrior(n_gaussians, x_grid.min(), x_grid.max()) *
    gptools.UniformJointPrior([(0, 2 * (x_grid.max() - x_grid.min())),] * n_gaussians) *
    gptools.NormalJointPrior([0.0,] * n_gaussians, [1.0,] * n_gaussians)
)
k_gibbs_exp_4 = gptools.GibbsKernel1dExpGauss(n_gaussians, hyperprior=hp)
k_gibbs_exp_4.params[2:2 + n_gaussians] = scipy.linspace(x_grid.min(), x_grid.max(), n_gaussians)
k_gibbs_exp_4.params[2 + n_gaussians:2 + 2 * n_gaussians] = k_gibbs_exp_4.params[3] - k_gibbs_exp_4.params[2]
k_gibbs_exp_4.fixed_params[2:2 + 2 * n_gaussians] = True

# SE+beta warp:
hp_SE = gptools.UniformJointPrior([(0, 40),]) * gptools.GammaJointPriorAlt([1.0,], [0.3,])
k_SE = gptools.SquaredExponentialKernel(hyperprior=hp_SE)
hp_beta = gptools.GammaJointPriorAlt([1.0, 0.0], [1.0, 0.5])
k_SE_beta = gptools.BetaWarpedKernel(k_SE, hyperprior=hp_beta)
k_SE_beta_lin = gptools.LinearWarpedKernel(k_SE_beta, x_grid.min() - 1e-6, x_grid.max() + 1e-6)

# SE+I-spline warp:
nt = 2
hp_SE = gptools.UniformJointPrior([(0, 10), (0, 2)])
# Fix l_1, since the global scale is set by the spline coefficients:
k_SE = gptools.SquaredExponentialKernel(hyperprior=hp_SE, initial_params=[1, 1], fixed_params=[False, True])
# C_{0,1} gets trapped in a bad mode:
hp_I_spline = (
    gptools.SortedUniformJointPrior(nt, x_grid.min(), x_grid.max()) *
    gptools.UniformJointPrior([(1e-3, 10),] + [(0, 10),] * (nt + 3 - 2 - 1))
)
k_SE_is_2 = gptools.ISplineWarpedKernel(k_SE, nt, hyperprior=hp_I_spline)
k_SE_is_2.w.params[0] = x_grid.min()
k_SE_is_2.w.params[1] = x_grid.max()
k_SE_is_2.w.fixed_params[0:2] = True

# SE+I-spline warp, free internal knot:
nt = 3
hp_SE = gptools.UniformJointPrior([(0, 10), (0, 2)])
# Fix l_1, since the global scale is set by the spline coefficients:
k_SE = gptools.SquaredExponentialKernel(hyperprior=hp_SE, initial_params=[1, 1], fixed_params=[False, True])
# C_{0,1} gets trapped in a bad mode:
hp_I_spline = (
    gptools.SortedUniformJointPrior(nt, x_grid.min(), x_grid.max()) *
    gptools.UniformJointPrior([(1e-3, 10),] + [(0, 10),] * (nt + 3 - 2 - 1))
)
k_SE_is_3 = gptools.ISplineWarpedKernel(k_SE, nt, hyperprior=hp_I_spline)
k_SE_is_3.w.params[0] = x_grid.min()
k_SE_is_3.w.params[2] = x_grid.max()
k_SE_is_3.w.fixed_params[0] = True
k_SE_is_3.w.fixed_params[2] = True

test_cases = [
    TestCase(
        k_gibbs_tanh, 'Gibbs + tanh', 'GibbsTanh',
        [r'$\sigma_f$', r'$\ell_1$', r'$\ell_2$', r'$\ell_{\text{w}}$', r'$x_0$'],
        a=6.0
    ),
    TestCase(
        k_gibbs_exp_1, 'Gibbs + exp(1 Gaussian)', 'GibbsExp1',
        [r'$\sigma_f$', r'$\ell_0$', r'$\mu_1$', r'$\sigma_1$', r'$\beta_1$'],
        a=16.0, nsamp=2000, burn=1900
    ),
    TestCase(
        k_gibbs_exp_4, 'Gibbs + exp(4 Gaussians)', 'GibbsExp4',
        [r'$\sigma_f$', r'$\ell_0$', r'$\beta_1$', r'$\beta_2$', r'$\beta_3$', r'$\beta_4$'],
        a=8.0, nsamp=1500, burn=1400
    ),
    TestCase(
        k_SE_beta_lin, r'\textsc{se} + beta-\textsc{cdf} warp', 'SEBeta',
        [r'$\sigma_f$', r'$\ell$', r'$\alpha$', r'$\beta$'],
        a=8.0
    ),
    TestCase(
        k_SE_is_2, r'\textsc{se} + 2-knot I-spline warp', 'SEIS2',
        [r'$\sigma_f$', r'$C_1$', r'$C_2$', r'$C_3$'],
        a=4.0
    ),
    TestCase(
        k_SE_is_3, r'\textsc{se} + 3-knot I-spline warp', 'SEIS3',
        [r'$\sigma_f$', r'$t_2$', r'$C_1$', r'$C_2$', r'$C_3$', r'$C_4$'],
        a=8.0, nsamp=1500, burn=1400
    )
]

# Start loop here:
for tc in test_cases:
    print(tc.lbl)
    gp = gptools.GaussianProcess(tc.k)
    gp.add_data(x_meas, y_meas, err_y=err_y_meas)
    gp.add_data(0, 0, n=1)
    psin_out = [1.1,]
    zeros_out = scipy.zeros_like(psin_out)
    gp.add_data(psin_out, zeros_out, err_y=0.01)
    gp.add_data(psin_out, zeros_out, err_y=0.1, n=1)
    
    gp.optimize_hyperparameters(
        verbose=True,
        random_starts=tc.MAP_starts,
        opt_kwargs={
            'method': 'Nelder-Mead',
            'options': {'maxiter': 5000}
        }
    )
    tc.MAP_params = gp.params[:]
    
    mu, s = gp.predict(x_star, n=n_star)
    
    tc.y_fit_MAP = mu[:len(x_grid)]
    tc.s_y_fit_MAP = s[:len(x_grid)]
    tc.dy_fit_MAP = mu[len(x_grid):]
    tc.s_dy_fit_MAP = s[len(x_grid):]
    
    tc.sampler = gp.sample_hyperparameter_posterior(nsamp=tc.nsamp, sampler_a=tc.a)
    tc.sampler.pool.close()
    tc.sampler.pool = None
    mu, s = gp.predict(
        x_star, n=n_star, use_MCMC=True, sampler=tc.sampler, burn=tc.burn, thin=tc.thin
    )
    gp.update_hyperparameters(tc.MAP_params)
    if 'Gibbs' in tc.lbl:
        tc.l = tc.k.l_func(x_grid, 0, *tc.k.params[1:])
        tc.w = scipy.integrate.cumtrapz(1.0 / tc.l, x=x_grid, initial=0.0)
        res = gp.compute_l_from_MCMC(x_grid, sampler=tc.sampler, burn=tc.burn)
        res = scipy.asarray(res)
        tc.mean_l = scipy.mean(res, axis=0)
        tc.std_l = scipy.std(res, axis=0, ddof=1)
        res_w = scipy.integrate.cumtrapz(1.0 / res, x=x_grid, axis=1, initial=0.0)
        tc.mean_w = scipy.mean(res_w, axis=0)
        tc.std_w = scipy.std(res_w, axis=0, ddof=1)
    else:
        tc.w = tc.k.w_func(x_grid, 0, 0)
        tc.l = tc.k.params[1] / tc.k.w_func(x_grid, 0, 1)
        res = gp.compute_w_from_MCMC(x_grid, sampler=tc.sampler, burn=tc.burn)
        res = scipy.asarray(res)
        tc.mean_w = scipy.mean(res, axis=0)
        tc.std_w = scipy.std(res, axis=0, ddof=1)
        
        res = gp.compute_w_from_MCMC(x_grid, n=1, sampler=tc.sampler, burn=tc.burn)
        res = scipy.asarray(res)
        if not tc.k.fixed_params[1]:
            flat_trace = tc.sampler.chain[:, tc.burn:, :]
            flat_trace = flat_trace.reshape((-1, flat_trace.shape[2]))
            l_MCMC = scipy.atleast_2d(flat_trace[:, 1]).T
        else:
            l_MCMC = 1.0
        
        res = l_MCMC / res
        tc.mean_l = scipy.mean(res, axis=0)
        tc.std_l = scipy.std(res, axis=0, ddof=1)
        
        # Remove the first and last points, since they mess up:
        tc.mean_l[0] = scipy.nan
        tc.mean_l[-1] = scipy.nan
        tc.std_l[0] = scipy.nan
        tc.std_l[-1] = scipy.nan
        tc.l[0] = scipy.nan
        tc.l[-1] = scipy.nan
    
    tc.y_fit = mu[:len(x_grid)]
    tc.s_y_fit = s[:len(x_grid)]
    tc.dy_fit = mu[len(x_grid):]
    tc.s_dy_fit = s[len(x_grid):]

for tc in test_cases:
    tc.rms_err_y_MAP = scipy.sqrt(scipy.mean((tc.y_fit_MAP - y_grid)**2.0))
    tc.rms_std_y_MAP = scipy.sqrt(scipy.mean(tc.s_y_fit_MAP**2.0))
    tc.rms_err_dy_MAP = scipy.sqrt(scipy.mean((tc.dy_fit_MAP - dy_grid)**2.0))
    tc.rms_std_dy_MAP = scipy.sqrt(scipy.mean(tc.s_dy_fit_MAP**2.0))
    
    tc.rms_err_y = scipy.sqrt(scipy.mean((tc.y_fit - y_grid)**2.0))
    tc.rms_std_y = scipy.sqrt(scipy.mean(tc.s_y_fit**2.0))
    tc.rms_err_dy = scipy.sqrt(scipy.mean((tc.dy_fit - dy_grid)**2.0))
    tc.rms_std_dy = scipy.sqrt(scipy.mean(tc.s_dy_fit**2.0))

with open('synthetic_test_nonstationary.pkl', 'wb') as pf:
    pkl.dump(test_cases, pf)

import setupplots
setupplots.thesis_format()
import matplotlib.pyplot as plt
plt.ion()
import matplotlib.gridspec as mplgs

# Sampler plots:
for tc in test_cases:
    gptools.plot_sampler(
        tc.sampler,
        labels=tc.hp_labels,
        burn=tc.burn,
        suptitle=tc.lbl,
        fixed_width=setupplots.TEXTWIDTH,
        label_fontsize=11,
        chain_ytick_pad=0.8,
        bottom_sep=0.12,
        chain_ticklabel_fontsize=11,
        ticklabel_fontsize=11,
        # max_chain_ticks=3 if tc.tlbl == 'SE' else 6,
        suptitle_space=0.0,
        ax_space=0.2,
        max_hist_ticks=6,
        cmap='plasma',
        hide_chain_ylabels=True
    )
    setupplots.apply_formatter(plt.gcf())
    plt.savefig("NS%sMarg.pgf" % (tc.tlbl,), bbox_inches='tight')
    plt.savefig("NS%sMarg.pdf" % (tc.tlbl,), bbox_inches='tight')

post_sum = setupplots.generate_post_sum(
    [tc.MAP_params for tc in test_cases],
    [tc.sampler for tc in test_cases],
    [tc.burn for tc in test_cases],
    [
        [[r'$\sigma_f$'], [r'$\ell_1$'], [r'$\ell_2$'], [r'$\ell_{\text{w}}$'], [r'$x_0$']],
        [[r'$\sigma_f$'], [r'$\ell_0$'], [r'$\mu_1$'], [r'$\sigma_1$'], [r'$\beta_1$']],
        [[r'$\sigma_f$'], [r'$\ell_0$'], [r'$\beta_1$'], [r'$\beta_2$'], [r'$\beta_3$'], [r'$\beta_4$']],
        [[r'$\sigma_f$'], [r'$\ell$'], [r'$\alpha$'], [r'$\beta$']],
        [[r'$\sigma_f$'], [r'$C_1$'], [r'$C_2$'], [r'$C_3$']],
        [[r'$\sigma_f$'], [r'$t_2$'], [r'$C_1$'], [r'$C_2$'], [r'$C_3$'], [r'$C_4$']]
    ],
    header_lines=[
        r'\multicolumn{7}{l}{\itshape Gibbs covariance kernel + tanh covariance length scale function}',
        r'\multicolumn{7}{l}{\itshape Gibbs covariance kernel + exponential of one Gaussian}',
        r'\multicolumn{7}{l}{\itshape Gibbs covariance kernel + exponential of four Gaussians}',
        r'\multicolumn{7}{l}{\itshape \textsc{se} covariance kernel + beta-\textsc{cdf} input warping function}',
        r'\multicolumn{7}{l}{\itshape \textsc{se} covariance kernel + 2-knot I-spline input warping function}',
        r'\multicolumn{7}{l}{\itshape \textsc{se} covariance kernel + 3-knot I-spline input warping function}'
    ]
)
with open('NSPostSum.tex', 'w') as tf:
    tf.write(post_sum)

# Compute RMS and relative uncertainties:
print("case\terr_y_MAP\tstd_y_MAP\terr_dy_MAP\tstd_dy_MAP\terr_y\tstd_y\terr_dy\tstd_dy")
f_rms = plt.figure(figsize=[setupplots.TEXTWIDTH, 0.5 * setupplots.TEXTWIDTH * 1.618])
a_y_rms = f_rms.add_subplot(1, 2, 1)
a_dy_rms = f_rms.add_subplot(1, 2, 2, sharex=a_y_rms)
for i, tc in enumerate(test_cases):
    print(
        "%s\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f" % (
            tc.lbl, tc.rms_err_y_MAP, tc.rms_std_y_MAP, tc.rms_err_dy_MAP,
            tc.rms_std_dy_MAP, tc.rms_err_y, tc.rms_std_y, tc.rms_err_dy,
            tc.rms_std_dy
        )
    )
    
    a_y_rms.bar(i, tc.rms_err_y_MAP, width=1.0 / 5.0, color='r', hatch='/', alpha=0.5, label=r'error, \textsc{map}' if i == 0 else None)
    a_y_rms.bar(i + 1.0 / 5.0, tc.rms_std_y_MAP, width=1.0 / 5.0, color='r', label=r'$\sigma$, \textsc{map}' if i == 0 else None)
    a_y_rms.bar(i + 2.0 / 5.0, tc.rms_err_y, width=1.0 / 5.0, color='b', hatch='/', alpha=0.5, label=r'error, \textsc{mcmc}' if i == 0 else None)
    a_y_rms.bar(i + 3.0 / 5.0, tc.rms_std_y, width=1.0 / 5.0, color='b', label=r'$\sigma$, \textsc{mcmc}' if i == 0 else None)
    
    a_dy_rms.bar(i, tc.rms_err_dy_MAP, width=1.0 / 5.0, color='r', hatch='/', alpha=0.5, label=r'error, \textsc{map}' if i == 0 else None)
    a_dy_rms.bar(i + 1.0 / 5.0, tc.rms_std_dy_MAP, width=1.0 / 5.0, color='r', label=r'$\sigma$, \textsc{map}' if i == 0 else None)
    a_dy_rms.bar(i + 2.0 / 5.0, tc.rms_err_dy, width=1.0 / 5.0, color='b', hatch='/', alpha=0.5, label=r'error, \textsc{mcmc}' if i == 0 else None)
    a_dy_rms.bar(i + 3.0 / 5.0, tc.rms_std_dy, width=1.0 / 5.0, color='b', label=r'$\sigma$, \textsc{mcmc}' if i == 0 else None)

a_y_rms.set_xlim(left=-1.0 / 5.0, right=len(test_cases))
a_y_rms.legend(loc='lower left')
a_y_rms.set_xticks([i + 2.0 / 5.0 for i in xrange(0, len(test_cases))])
a_dy_rms.set_xticks([i + 2.0 / 5.0 for i in xrange(0, len(test_cases))])
a_y_rms.set_xticklabels([tc.lbl for tc in test_cases])
a_dy_rms.set_xticklabels([tc.lbl for tc in test_cases])
a_y_rms.set_title("$y$")
a_y_rms.set_ylabel(r'\textsc{rms} error')
a_dy_rms.set_title(r"$\mathrm{d}y/\mathrm{d}x$")
f_rms.suptitle(r"\textsc{rms} error and uncertainty estimates")
plt.setp(a_y_rms.xaxis.get_majorticklabels(), rotation=90)
plt.setp(a_dy_rms.xaxis.get_majorticklabels(), rotation=90)
f_rms.subplots_adjust(wspace=0.2, bottom=0.0)
f_rms.savefig("NSRMS.pgf", bbox_inches='tight')
f_rms.savefig("NSRMS.pdf", bbox_inches='tight')

val_bounds = [0.5, 1.1]
grad_bounds = [-1, 0.25]
l_bounds = [0, 2]
w_bounds = [0, 4]

val_bounds_ped = [0, 0.8]
grad_bounds_ped = [-15, 2.5]
l_bounds_ped = [0, 2]
w_bounds_ped = [0, 5]

for tc in test_cases:
    # Set up plots:
    f = plt.figure(figsize=(setupplots.TEXTWIDTH, 0.9 * 0.5 * setupplots.TEXTWIDTH * 4 / 1.618))
    f.suptitle(tc.lbl)
    gs1 = mplgs.GridSpec(4, 2)
    gs1.update(bottom=0.275)
    gs2 = mplgs.GridSpec(1, 2)
    gs2.update(top=0.2)
    a_y = f.add_subplot(gs1[0, 0])
    a_y.plot(x_grid, y_grid, 'k--', linewidth=2 * setupplots.lw, label='true curve')
    a_y.errorbar(x_meas, y_meas, yerr=err_y_meas, fmt='o', color='g', markersize=setupplots.ms, label='synthetic data')
    a_dy = f.add_subplot(gs1[1, 0], sharex=a_y)
    a_dy.plot(x_grid, dy_grid, 'k--', linewidth=2 * setupplots.lw)
    a_l = f.add_subplot(gs1[2, 0], sharex=a_y)
    a_w = f.add_subplot(gs1[3, 0], sharex=a_y)
    a_warped = f.add_subplot(gs2[0, 0])
    a_warped.plot(tc.mean_w, y_grid, 'k--', linewidth=2 * setupplots.lw)
    
    a_y_ped = f.add_subplot(gs1[0, 1])
    a_y_ped.plot(x_grid, y_grid, 'k--', linewidth=2 * setupplots.lw, label='true curve')
    a_y_ped.errorbar(x_meas, y_meas, yerr=err_y_meas, fmt='o', color='g', markersize=setupplots.ms, label='synthetic data')
    a_dy_ped = f.add_subplot(gs1[1, 1], sharex=a_y_ped)
    a_dy_ped.plot(x_grid, dy_grid, 'k--', linewidth=2 * setupplots.lw)
    a_l_ped = f.add_subplot(gs1[2, 1], sharex=a_y_ped)
    a_w_ped = f.add_subplot(gs1[3, 1], sharex=a_y_ped)
    a_warped_ped = f.add_subplot(gs2[0, 1])
    a_warped_ped.plot(tc.mean_w, y_grid, 'k--', linewidth=2 * setupplots.lw)
    
    # Core plot:
    gptools.univariate_envelope_plot(
        x_grid, tc.y_fit_MAP, tc.s_y_fit_MAP, ax=a_y, color='r', ls='-.', label=r'\textsc{map}', lb=val_bounds[0], ub=val_bounds[1], lw=setupplots.lw
    )
    gptools.univariate_envelope_plot(
        x_grid, tc.y_fit, tc.s_y_fit, ax=a_y, color='b', label=r'\textsc{mcmc}', lb=val_bounds[0], ub=val_bounds[1], lw=setupplots.lw
    )
    
    gptools.univariate_envelope_plot(
        x_grid, tc.dy_fit_MAP, tc.s_dy_fit_MAP, ax=a_dy, color='r', ls='-.', lb=grad_bounds[0], ub=grad_bounds[1], lw=setupplots.lw
    )
    gptools.univariate_envelope_plot(
        x_grid, tc.dy_fit, tc.s_dy_fit, ax=a_dy, color='b', lb=grad_bounds[0], ub=grad_bounds[1], lw=setupplots.lw
    )
    
    w = tc.w.copy()
    w[w > w_bounds[1]] = w_bounds[1]
    a_w.plot(x_grid, w, '-.', color='r', lw=setupplots.lw)
    gptools.univariate_envelope_plot(
        x_grid, tc.mean_w, tc.std_w, ax=a_w, color='b', lb=w_bounds[0], ub=w_bounds[1], lw=setupplots.lw
    )
    
    l = tc.l.copy()
    l[l > l_bounds[1]] = l_bounds[1]
    a_l.plot(x_grid, l, '-.', color='r', lw=setupplots.lw)
    gptools.univariate_envelope_plot(
        x_grid, tc.mean_l, tc.std_l, ax=a_l, color='b', lb=l_bounds[0], ub=l_bounds[1], lw=setupplots.lw
    )
    
    a_warped.plot(tc.mean_w, tc.y_fit, 'b', lw=setupplots.lw)
    
    # Pedestal plot:
    gptools.univariate_envelope_plot(
        x_grid, tc.y_fit_MAP, tc.s_y_fit_MAP, ax=a_y_ped, color='r', ls='-.', label=r'\textsc{map}', lb=val_bounds_ped[0], ub=val_bounds_ped[1], lw=setupplots.lw
    )
    gptools.univariate_envelope_plot(
        x_grid, tc.y_fit, tc.s_y_fit, ax=a_y_ped, color='b', label=r'\textsc{mcmc}', lb=val_bounds_ped[0], ub=val_bounds_ped[1], lw=setupplots.lw
    )
    
    gptools.univariate_envelope_plot(
        x_grid, tc.dy_fit_MAP, tc.s_dy_fit_MAP, ax=a_dy_ped, color='r', ls='-.', lb=grad_bounds_ped[0], ub=grad_bounds_ped[1], lw=setupplots.lw
    )
    gptools.univariate_envelope_plot(
        x_grid, tc.dy_fit, tc.s_dy_fit, ax=a_dy_ped, color='b', lb=grad_bounds_ped[0], ub=grad_bounds_ped[1], lw=setupplots.lw
    )
    
    w = tc.w.copy()
    w[w > w_bounds_ped[1]] = w_bounds_ped[1]
    a_w_ped.plot(x_grid, w, '-.', color='r', lw=setupplots.lw)
    gptools.univariate_envelope_plot(
        x_grid, tc.mean_w, tc.std_w, ax=a_w_ped, color='b', lb=w_bounds_ped[0], ub=w_bounds_ped[1], lw=setupplots.lw
    )
    
    l = tc.l.copy()
    l[l > l_bounds_ped[1]] = l_bounds_ped[1]
    a_l_ped.plot(x_grid, l, '-.', color='r', lw=setupplots.lw)
    gptools.univariate_envelope_plot(
        x_grid, tc.mean_l, tc.std_l, ax=a_l_ped, color='b', lb=l_bounds_ped[0], ub=l_bounds_ped[1], lw=setupplots.lw
    )
    
    a_warped_ped.plot(tc.mean_w, tc.y_fit, 'b', lw=setupplots.lw)
    
    a_y.set_xlim(0, 1.1)
    a_y.set_ylim(val_bounds)
    a_y.yaxis.set_major_locator(plt.MaxNLocator(nbins=5))
    a_dy.set_ylim(grad_bounds)
    a_dy.yaxis.set_major_locator(plt.MaxNLocator(nbins=5))
    a_l.set_ylim(l_bounds)
    a_l.yaxis.set_major_locator(plt.MaxNLocator(nbins=5))
    a_w.set_ylim(w_bounds)
    a_w.yaxis.set_major_locator(plt.MaxNLocator(nbins=5))
    a_warped.set_ylim(bottom=0, top=1.1)
    a_warped.yaxis.set_major_locator(plt.MaxNLocator(nbins=5))
    a_y.set_title('Core')
    plt.setp(a_y.get_xticklabels(), visible=False)
    plt.setp(a_dy.get_xticklabels(), visible=False)
    plt.setp(a_l.get_xticklabels(), visible=False)
    a_w.set_xlabel(r'$x$')
    a_y.set_ylabel(r'$y$')
    a_dy.set_ylabel(r'$\mathrm{d}y/\mathrm{d}x$')
    a_l.set_ylabel(r'$\ell$')
    a_w.set_ylabel(r'$w$')
    a_warped.set_ylabel(r'$y$')
    a_warped.set_xlabel(r'$w$')
    
    a_y_ped.set_xlim(0.9, 1.1)
    a_y_ped.set_ylim(val_bounds_ped)
    a_y_ped.yaxis.set_major_locator(plt.MaxNLocator(nbins=5))
    a_dy_ped.set_ylim(grad_bounds_ped)
    a_dy_ped.yaxis.set_major_locator(plt.MaxNLocator(nbins=5))
    a_l_ped.set_ylim(l_bounds_ped)
    a_l_ped.yaxis.set_major_locator(plt.MaxNLocator(nbins=5))
    a_w_ped.set_ylim(w_bounds_ped)
    a_w_ped.yaxis.set_major_locator(plt.MaxNLocator(nbins=5))
    a_y_ped.set_title('Edge')
    plt.setp(a_y_ped.get_xticklabels(), visible=False)
    plt.setp(a_dy_ped.get_xticklabels(), visible=False)
    plt.setp(a_l_ped.get_xticklabels(), visible=False)
    a_w_ped.set_xlabel(r'$x$')
    a_warped_ped.set_xlabel(r'$w$')
    a_warped_ped.set_ylim(val_bounds_ped)
    a_warped_ped.set_xlim(left=min(tc.mean_w[x_grid >= 0.9]))
    a_warped_ped.yaxis.set_major_locator(plt.MaxNLocator(nbins=5))
    a_warped_ped.xaxis.set_major_locator(plt.MaxNLocator(nbins=5))
    
    f.subplots_adjust(left=0.12, bottom=0.075, right=0.97, top=0.92)
    f.canvas.draw()
    setupplots.apply_formatter(f)
    f.savefig("NS%s.pdf" % (tc.tlbl,), bbox_inches='tight')
    f.savefig("NS%s.pgf" % (tc.tlbl,), bbox_inches='tight')

# Put legend in seperate figure:
f_leg = plt.figure()
l = f_leg.legend(*a_y.get_legend_handles_labels(), ncol=2, loc='center')
f_leg.canvas.draw()
f_leg.set_figwidth(l.get_window_extent().width / 72)
f_leg.set_figheight(l.get_window_extent().height / 72)
f_leg.savefig("NSleg.pdf", bbox_inches='tight')
f_leg.savefig("NSleg.pgf", bbox_inches='tight')
