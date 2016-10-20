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

# This script makes figures 2.11 and 2.12, which show the (lack of) effect of
# random errors in the abscissa on the GPR fit.

from __future__ import division
import gptools
import scipy
import scipy.interpolate
import numpy.random
import cPickle as pkl

# ne-like shape:
x_control = scipy.asarray([0, 0.3,  0.6, 0.85, 0.965, 1.02, 1.05, 1.08,   1.1, 1.125, 1.15])
y_control = scipy.asarray([1, 0.85, 0.7, 0.65, 0.6,   0.4,  0.08, 0.0005, 0.0, 0.0,   0.0])

spl = scipy.interpolate.InterpolatedUnivariateSpline(
    scipy.concatenate((-scipy.flipud(x_control[1:]), x_control)),
    scipy.concatenate((scipy.flipud(y_control[1:]), y_control)),
    k=3
)

x_grid = scipy.linspace(0, 1.1, 400)
y_grid = spl(x_grid)
dy_grid = spl(x_grid, nu=1)

edge_start = 0.97
core_rel_err = 0.04
edge_abs_err = 0.2

x_star = scipy.concatenate((x_grid, x_grid))
n_star = scipy.concatenate((scipy.zeros_like(x_grid), scipy.ones_like(x_grid)))

# First seed the generator to make the test repeatable:
RS = numpy.random.RandomState(8675309)

class Case(object):
    def __init__(self):
        pass

c = Case()

with open('outputs/cov_psi_TS.pkl', 'rb') as pf:
    res = pkl.load(pf)
c.x_meas_clean = res['X']
c.cov_x_meas = res['cov']
c.x_meas_noisy = c.x_meas_clean + 5 * RS.multivariate_normal(
    scipy.zeros_like(c.x_meas_clean), c.cov_x_meas, 1
).ravel()

c.y_meas = spl(c.x_meas_clean)
c.err_y_meas = scipy.array([
    0.03041753,  0.0394804 ,  0.02225024,  0.02984038,  0.02434629,
    0.03528642,  0.03234842,  0.03063433,  0.02532734,  0.03548691,
    0.01737603,  0.02349633,  0.02761289,  0.02459682,  0.01715875,
    0.05098487,  0.11185434,  0.10632183,  0.07904776,  0.07919467,
    0.06259312,  0.07857593,  0.14576103,  0.1595387 ,  0.23001953,
    0.12682407,  0.31037695,  0.27262479,  0.30911589,  0.33525105,
    0.54201707,  0.71100043,  0.37074277
])

c.y_meas += RS.randn(c.y_meas.shape[0]) * c.err_y_meas
c.y_meas[c.y_meas < 0] = 0

b = 10
m = 1
be = 10
me = 0.5
hp = (
    gptools.UniformJointPrior([(0, 30)]) *
    gptools.GammaJointPrior(
        [1 + m * b, 1 + me * be, 1, 1 + 1.01 * 200],
        [b, be, 1 / 0.1, 200]
    )
)
k = gptools.GibbsKernel1dTanh(hyperprior=hp)
psin_out = scipy.array([1.1])
zeros_out = scipy.zeros_like(psin_out)
c.gp_clean = gptools.GaussianProcess(k)
c.gp_clean.add_data(c.x_meas_clean, c.y_meas, err_y=c.err_y_meas)
c.gp_clean.add_data(0, 0, n=1)
c.gp_clean.add_data(psin_out, zeros_out, err_y=0.01)
c.gp_clean.add_data(psin_out, zeros_out, err_y=0.1, n=1)
c.sampler_clean = c.gp_clean.sample_hyperparameter_posterior(nsamp=500, sampler_a=4.0)
mu_clean, s_clean = c.gp_clean.predict(
    x_star, n=n_star, use_MCMC=True, burn=400, thin=100, sampler=c.sampler_clean, full_output=False
)
c.y_fit_clean = mu_clean[:len(x_grid)]
c.s_y_fit_clean = s_clean[:len(x_grid)]
c.dy_fit_clean = mu_clean[len(x_grid):]
c.s_dy_fit_clean = s_clean[len(x_grid):]

c.gp_noisy = gptools.GaussianProcess(k)
c.gp_noisy.add_data(c.x_meas_noisy, c.y_meas, err_y=c.err_y_meas)
c.gp_noisy.add_data(0, 0, n=1)
c.gp_noisy.add_data(psin_out, zeros_out, err_y=0.01)
c.gp_noisy.add_data(psin_out, zeros_out, err_y=0.1, n=1)
c.sampler_noisy = c.gp_noisy.sample_hyperparameter_posterior(nsamp=500, sampler_a=4.0)
mu_noisy, s_noisy = c.gp_noisy.predict(
    x_star, n=n_star, use_MCMC=True, burn=400, thin=100, sampler=c.sampler_noisy, full_output=False
)
c.y_fit_noisy = mu_noisy[:len(x_grid)]
c.s_y_fit_noisy = s_noisy[:len(x_grid)]
c.dy_fit_noisy = mu_noisy[len(x_grid):]
c.s_dy_fit_noisy = s_noisy[len(x_grid):]

c.sampler_clean.pool.close()
c.sampler_clean.pool = None
c.sampler_noisy.pool.close()
c.sampler_noisy.pool = None
with open('synthetic_test_EIV.pkl', 'wb') as pf:
    pkl.dump(c, pf)

# Make plots:
import setupplots
setupplots.thesis_format()
import matplotlib.pyplot as plt
import matplotlib.ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
plt.ion()

# gptools.plot_sampler(
#     c.sampler_clean, burn=400, suptitle="clean",
#     labels=['$%s$' % (l,) for l in c.gp_clean.free_param_names[:]]
# )
# gptools.plot_sampler(
#     c.sampler_noisy, burn=400, suptitle="noisy",
#     labels=['$%s$' % (l,) for l in c.gp_clean.free_param_names[:]]
# )

f = plt.figure(figsize=[setupplots.TEXTWIDTH, 2 * 0.5 * setupplots.TEXTWIDTH / 1.618])

a_y = f.add_subplot(2, 2, 1)
a_y.set_title("Core")
a_y.plot(x_grid, y_grid, 'k--', linewidth=2 * setupplots.lw, label='true curve')
gptools.univariate_envelope_plot(
    x_grid, c.y_fit_clean, c.s_y_fit_clean, ax=a_y, color='b', label='no $x$ error', ls='-', lw=setupplots.lw
)
gptools.univariate_envelope_plot(
    x_grid, c.y_fit_noisy, c.s_y_fit_noisy, ax=a_y, color='r', label='with $x$ error', ls='-.', lw=setupplots.lw
)
a_y.errorbar(
    c.x_meas_clean, c.y_meas, yerr=c.err_y_meas, fmt='o', color='b', markersize=setupplots.ms
)
a_y.errorbar(
    c.x_meas_noisy, c.y_meas, yerr=c.err_y_meas, xerr=5 * scipy.sqrt(scipy.diag(c.cov_x_meas)), fmt='^', color='r', markersize=setupplots.ms
)
a_dy = f.add_subplot(2, 2, 3)
a_dy.plot(x_grid, dy_grid, 'k--', linewidth=2 * setupplots.lw)
gptools.univariate_envelope_plot(
    x_grid, c.dy_fit_clean, c.s_dy_fit_clean, ax=a_dy, color='b', ls='-', lw=setupplots.lw
)
gptools.univariate_envelope_plot(
    x_grid, c.dy_fit_noisy, c.s_dy_fit_noisy, ax=a_dy, color='r', ls='-.', lw=setupplots.lw
)
a_y.set_xlim(0, 1.1)
a_y.set_ylim(0.5, 1.1)
a_dy.set_xlim(0, 1.1)
a_dy.set_ylim(-1, 0.25)
plt.setp(a_y.get_xticklabels(), visible=False)
a_dy.set_xlabel(r'$x$')
a_l = a_y
a_y.set_ylabel(r'$y$')
a_dy.set_ylabel(r'$\mathrm{d}y/\mathrm{d}x$')

a_y = f.add_subplot(2, 2, 2)
a_y.set_title('Edge')
a_y.plot(x_grid, y_grid, 'k--', linewidth=2 * setupplots.lw, label='true curve')
gptools.univariate_envelope_plot(
    x_grid, c.y_fit_clean, c.s_y_fit_clean, ax=a_y, color='b', ls='-',
    lw=setupplots.lw, label=r'no \textsc{eiv}'
)
gptools.univariate_envelope_plot(
    x_grid, c.y_fit_noisy, c.s_y_fit_noisy, ax=a_y, color='r', ls='-.',
    lw=setupplots.lw, label=r'with \textsc{eiv}'
)
a_y.errorbar(
    c.x_meas_clean, c.y_meas, yerr=c.err_y_meas, fmt='o', color='b',
    markersize=setupplots.ms, label=r'synthetic data, no \textsc{eiv}'
)
a_y.errorbar(
    c.x_meas_noisy, c.y_meas, yerr=c.err_y_meas, xerr=5 * scipy.sqrt(scipy.diag(c.cov_x_meas)),
    fmt='^', color='r', markersize=setupplots.ms, label=r'synthetic data, with \textsc{eiv}'
)
a_dy = f.add_subplot(2, 2, 4)
a_dy.plot(x_grid, dy_grid, 'k--', linewidth=2 * setupplots.lw, label='true curve')
gptools.univariate_envelope_plot(
    x_grid, c.dy_fit_clean, c.s_dy_fit_clean, ax=a_dy, color='b',
    label=r'no \textsc{eiv}', ls='-', lw=setupplots.lw
)
gptools.univariate_envelope_plot(
    x_grid, c.dy_fit_noisy, c.s_dy_fit_noisy, ax=a_dy, color='r',
    label=r'with \textsc{eiv}', ls='-.', lw=setupplots.lw
)
a_y.set_xlim(0.9, 1.1)
a_y.set_ylim(0, 1.1)
a_dy.set_xlim(0.9, 1.1)
a_dy.set_ylim(-15, 2.5)
plt.setp(a_y.get_xticklabels(), visible=False)
a_dy.set_xlabel(r'$x$')
a_l_ped = a_y
# a_dy.legend(loc='lower left')

f.subplots_adjust(left=0.13, bottom=0.13, right=0.97, top=0.93)
setupplots.apply_formatter(f)
f.savefig('EIV.pdf')
f.savefig('EIV.pgf')

# Put legend in seperate figure:
f_leg = plt.figure()
l = f_leg.legend(*a_y.get_legend_handles_labels(), ncol=1, loc='center')
f_leg.canvas.draw()
f_leg.set_figwidth(l.get_window_extent().width / 72)
f_leg.set_figheight(l.get_window_extent().height / 72)
f_leg.savefig("EIVleg.pdf", bbox_inches='tight')
f_leg.savefig("EIVleg.pgf", bbox_inches='tight')

# Plot the covariance matrix:
f_cov = plt.figure(figsize=[0.5 * setupplots.TEXTWIDTH, 0.5 * setupplots.TEXTWIDTH])
a_cov = f_cov.add_subplot(1, 1, 1)
a_cov.set_title('Covariance of scatter in $r/a$ coordinate', y=1.2)
vmax = scipy.absolute(c.cov_x_meas * 1e6).max()
cax = a_cov.pcolor(c.cov_x_meas * 1e6, cmap='seismic', vmin=-1 * vmax, vmax=vmax)
divider = make_axes_locatable(a_cov)
a_cb = divider.append_axes("right", size="10%", pad=0.05)
cbar = f_cov.colorbar(cax, label=r'$\mathrm{cov}[x_i, x_j] \times 10^6$', cax=a_cb)
cbar.update_ticks()
a_cov.set_xlabel('channel index')
a_cov.xaxis.set_label_position('top')
a_cov.set_ylabel('channel index')
a_cov.axis('square')
a_cov.invert_yaxis()
a_cov.xaxis.tick_top()
setupplots.apply_formatter(f_cov)
f_cov.savefig('xCov.pdf', bbox_inches='tight')
f_cov.savefig('xCov.pgf', bbox_inches='tight')
