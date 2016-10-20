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

# This script produces figures 2.13, 2.14 and 2.15, which show the synthetic
# core data fit with a variety of stationary covariance kernels.

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

x_grid = scipy.linspace(0, 0.97, 100)
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
x_edge = x_edge[x_edge <= 0.97]
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
    def __init__(self, k, lbl, tlbl, hp_labels, a=2.0, nsamp=500, burn=400, thin=100, MAP_starts=48, plt_width=0.66):
        self.k = k
        self.lbl = lbl
        self.tlbl = tlbl
        self.hp_labels = hp_labels
        self.a = a
        self.nsamp = nsamp
        self.burn = burn
        self.thin = thin
        self.MAP_starts = MAP_starts
        self.plt_width = plt_width

# SE:
k_SE = gptools.SquaredExponentialKernel(param_bounds=[(0, 10), (0, 5)])

# RQ, variable nu:
k_RQ = gptools.RationalQuadraticKernel(param_bounds=[(0, 10), (0.01, 10), (0, 5)])

# Matern, variable nu:
hp_M = gptools.UniformJointPrior([(0, 10), (1.01, 50)]) * gptools.GammaJointPriorAlt(1, 0.3)
k_M = gptools.MaternKernel(hyperprior=hp_M)

test_cases = [
    TestCase(k_SE, r'\textsc{se}', 'SE', [r'$\sigma_f$', r'$\ell$'], nsamp=300, burn=200, plt_width=0.45),
    TestCase(k_RQ, r'\textsc{rq}', 'RQ', [r'$\sigma_f$', r'$\alpha$', r'$\ell$'], nsamp=400, burn=300, a=6.0),
    TestCase(k_M, r'Mat\'ern', 'M', [r'$\sigma_f$', r'$\nu$', r'$\ell$'], nsamp=400, burn=300)
]

# Start loop here:
for tc in test_cases:
    print(tc.lbl)
    tc.gp = gptools.GaussianProcess(tc.k, verbose=True)
    tc.gp.add_data(x_meas, y_meas, err_y=err_y_meas)
    tc.gp.add_data(0, 0, n=1)
    
    tc.gp.optimize_hyperparameters(verbose=True, random_starts=tc.MAP_starts, opt_kwargs={'options': {'maxiter': 500}})
    tc.MAP_params = tc.gp.params[:]
    mu, s = tc.gp.predict(x_star, n=n_star)
    tc.y_fit_MAP = mu[:len(x_grid)]
    tc.s_y_fit_MAP = s[:len(x_grid)]
    tc.dy_fit_MAP = mu[len(x_grid):]
    tc.s_dy_fit_MAP = s[len(x_grid):]
    
    tc.sampler = tc.gp.sample_hyperparameter_posterior(nsamp=tc.nsamp, sampler_a=tc.a)
    tc.sampler.pool.close()
    tc.sampler.pool = None
    mu, s = tc.gp.predict(
        x_star, n=n_star, use_MCMC=True, sampler=tc.sampler, burn=tc.burn, thin=tc.thin
    )
    tc.gp.update_hyperparameters(tc.MAP_params)
    tc.y_fit = mu[:len(x_grid)]
    tc.s_y_fit = s[:len(x_grid)]
    tc.dy_fit = mu[len(x_grid):]
    tc.s_dy_fit = s[len(x_grid):]

# Compute RMS and relative uncertainties:

for tc in test_cases:
    tc.rms_err_y_MAP = scipy.sqrt(scipy.mean((tc.y_fit_MAP - y_grid)**2.0))
    tc.rms_std_y_MAP = scipy.sqrt(scipy.mean(tc.s_y_fit_MAP**2.0))
    tc.rms_err_dy_MAP = scipy.sqrt(scipy.mean((tc.dy_fit_MAP - dy_grid)**2.0))
    tc.rms_std_dy_MAP = scipy.sqrt(scipy.mean(tc.s_dy_fit_MAP**2.0))
    
    tc.rms_err_y = scipy.sqrt(scipy.mean((tc.y_fit - y_grid)**2.0))
    tc.rms_std_y = scipy.sqrt(scipy.mean(tc.s_y_fit**2.0))
    tc.rms_err_dy = scipy.sqrt(scipy.mean((tc.dy_fit - dy_grid)**2.0))
    tc.rms_std_dy = scipy.sqrt(scipy.mean(tc.s_dy_fit**2.0))

with open('synthetic_test_stationary.pkl', 'wb') as pf:
    pkl.dump(test_cases, pf)

# Make plots:
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
        fixed_width=tc.plt_width * setupplots.TEXTWIDTH,
        label_fontsize=11,
        chain_ytick_pad=0.8,
        bottom_sep=0.175 if tc.tlbl == 'SE' else 0.115,
        chain_ticklabel_fontsize=11,
        ticklabel_fontsize=11,
        hide_chain_ylabels=True,
        max_chain_ticks=3 if tc.tlbl == 'SE' else 6,
        suptitle_space=0.1 if tc.tlbl == 'SE' else 0.05,
        ax_space=0.15,
        cmap='plasma',
        max_hist_ticks=6
    )
    setupplots.apply_formatter(plt.gcf())
    plt.savefig("S%sMarg.pgf" % (tc.tlbl,), bbox_inches='tight')
    plt.savefig("S%sMarg.pdf" % (tc.tlbl,), bbox_inches='tight')

post_sum = setupplots.generate_post_sum(
    [tc.MAP_params for tc in test_cases],
    [tc.sampler for tc in test_cases],
    [tc.burn for tc in test_cases],
    [
        [[r'\textsc{se}', r'$\sigma_f$'], ['', r'$\ell$']],
        [[r'\textsc{rq}', r'$\sigma_f$'], ['', r'$\alpha$'], ['', r'$\ell$']],
        [[r'Mat\'ern', r'$\sigma_f$'], ['', r'$\nu$'], ['', r'$\ell$']],
    ]
)
with open('SPostSum.tex', 'w') as tf:
    tf.write(post_sum)

# RMS plot:
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
f_rms.subplots_adjust(wspace=0.24)
f_rms.savefig('SRMS.pgf', bbox_inches='tight')
f_rms.savefig('SRMS.pdf', bbox_inches='tight')

f = plt.figure(figsize=(setupplots.TEXTWIDTH, setupplots.TEXTWIDTH * 2.0 / len(test_cases) / 1.618))

for i, tc in enumerate(test_cases):
    a_y = f.add_subplot(2, len(test_cases), i + 1)
    a_y.plot(x_grid, y_grid, 'k--', linewidth=2 * setupplots.lw, label='true curve')
    a_y.errorbar(x_meas, y_meas, yerr=err_y_meas, fmt='o', color='g', markersize=setupplots.ms, label='synthetic data')
    a_dy = f.add_subplot(2, len(test_cases), i + 1 + len(test_cases), sharex=a_y)
    a_dy.plot(x_grid, dy_grid, 'k--', linewidth=2 * setupplots.lw)
    
    # Core plot:
    gptools.univariate_envelope_plot(
        x_grid, tc.y_fit_MAP, tc.s_y_fit_MAP, ax=a_y, color='r', ls='-.', label=r'\textsc{map}', lw=setupplots.lw
    )
    gptools.univariate_envelope_plot(
        x_grid, tc.y_fit, tc.s_y_fit, ax=a_y, color='b', label=r'\textsc{mcmc}', lw=setupplots.lw
    )
    
    gptools.univariate_envelope_plot(
        x_grid, tc.dy_fit_MAP, tc.s_dy_fit_MAP, ax=a_dy, color='r', ls='-.', lw=setupplots.lw
    )
    # a_dy.plot(x_grid[:-1], scipy.diff(y_fit_MAP) / (x_grid[1] - x_grid[0]))
    gptools.univariate_envelope_plot(
        x_grid, tc.dy_fit, tc.s_dy_fit, ax=a_dy, color='b', lw=setupplots.lw
    )
    
    a_y.set_xlim(0, 1.0)
    a_y.set_ylim(0.5, 1.1)
    a_dy.set_ylim(-1, 0.25)
    a_y.set_title(tc.lbl)
    plt.setp(a_y.get_xticklabels(), visible=False)
    if i == 0:
        a_y.set_ylabel(r'$y$')
        a_dy.set_ylabel(r'$\mathrm{d}y/\mathrm{d}x$')
    else:
        plt.setp(a_y.get_yticklabels(), visible=False)
        plt.setp(a_dy.get_yticklabels(), visible=False)
    a_dy.set_xlabel(r'$x$')

f.subplots_adjust(left=0.12, bottom=0.19, right=0.97)
f.canvas.draw()
setupplots.apply_formatter(f)
f.savefig('S.pdf')
f.savefig('S.pgf')

# Put legend in seperate figure:
f_leg = plt.figure()
l = f_leg.legend(*a_y.get_legend_handles_labels(), ncol=2, loc='center')
f_leg.canvas.draw()
f_leg.set_figwidth(l.get_window_extent().width / 72)
f_leg.set_figheight(l.get_window_extent().height / 72)
f_leg.savefig("Sleg.pdf", bbox_inches='tight')
f_leg.savefig("Sleg.pgf", bbox_inches='tight')
