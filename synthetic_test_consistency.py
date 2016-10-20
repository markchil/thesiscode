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

# This script makes figures 2.9 and 2.10, which show the results of testing the
# asymptotic consistency of nonstationary GPR with synthetic data. Figure 2.9
# uses the "profile plots" case, figure 2.10 uses the "asymptotic consistency"
# case.

from __future__ import division
import gptools
import scipy
import scipy.interpolate
import numpy
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
core_rel_err = 0.03
edge_abs_err = 0.1

x_star = scipy.concatenate((x_grid, x_grid))
n_star = scipy.concatenate((scipy.zeros_like(x_grid), scipy.ones_like(x_grid)))

class Case(object):
    def __init__(self, num_pts):
        self.num_pts = num_pts

# Use this for profile plots:
cases = [Case(15), Case(30), Case(60)]
num_pts = [c.num_pts for c in cases]
# Use this for asymptotic consistency:
# num_pts = 5 * [3, 7, 15, 30, 60, 120]
# cases = [Case(n) for n in num_pts]

# First seed the generator to make the test repeatable:
RS = numpy.random.RandomState(8675309)

for i, c in enumerate(cases):
    n = c.num_pts
    print("n=%d, i=%d..." % (n, i))
    x_core = scipy.linspace(0, edge_start, n)
    x_edge = scipy.linspace(edge_start, 1.07, n + 1)[1:]
    c.x_meas = scipy.concatenate((x_core, x_edge))
    c.y_meas = spl(c.x_meas)
    c.err_y_meas = scipy.zeros_like(c.y_meas)
    c.err_y_meas[:n] = core_rel_err
    c.err_y_meas *= c.y_meas
    c.err_y_meas[n:] = edge_abs_err
    
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
    gp = gptools.GaussianProcess(k)
    gp.add_data(c.x_meas, c.y_meas, err_y=c.err_y_meas)
    gp.add_data(0, 0, n=1)
    psin_out = scipy.array([1.1])
    zeros_out = scipy.zeros_like(psin_out)
    gp.add_data(psin_out, zeros_out, err_y=0.01)
    gp.add_data(psin_out, zeros_out, err_y=0.1, n=1)
    
    gp.optimize_hyperparameters(verbose=True)
    c.MAP_params = gp.free_params[:]
    
    mu, s = gp.predict(x_star, n=n_star)
    
    c.y_fit_MAP = mu[:len(x_grid)]
    c.s_y_fit_MAP = s[:len(x_grid)]
    c.dy_fit_MAP = mu[len(x_grid):]
    c.s_dy_fit_MAP = s[len(x_grid):]
    
    c.rms_err_y_MAP = scipy.sqrt(scipy.mean(((c.y_fit_MAP - y_grid))**2))
    c.rms_std_y_MAP = scipy.sqrt(scipy.mean((c.s_y_fit_MAP)**2))
    c.rms_err_dy_MAP = scipy.sqrt(scipy.mean(((c.dy_fit_MAP - dy_grid))**2))
    c.rms_std_dy_MAP = scipy.sqrt(scipy.mean((c.s_dy_fit_MAP)**2))
    
    c.sampler = gp.sample_hyperparameter_posterior(nsamp=500, sampler_a=4.0)
    c.sampler.pool.close()
    c.sampler.pool = None
    mu, s = gp.predict(
        x_star, n=n_star, use_MCMC=True, burn=400, thin=100, sampler=c.sampler,
        full_output=False
    )
    
    c.y_fit = mu[:len(x_grid)]
    c.s_y_fit = s[:len(x_grid)]
    c.dy_fit = mu[len(x_grid):]
    c.s_dy_fit = s[len(x_grid):]
    
    c.rms_err_y = scipy.sqrt(scipy.mean(((c.y_fit - y_grid))**2))
    c.rms_std_y = scipy.sqrt(scipy.mean((c.s_y_fit)**2))
    c.rms_err_dy = scipy.sqrt(scipy.mean(((c.dy_fit - dy_grid))**2))
    c.rms_std_dy = scipy.sqrt(scipy.mean((c.s_dy_fit)**2))

# Store the data for re-making plots:
with open('synthetic_test_consistency.pkl', 'wb') as pf:
    pkl.dump(cases, pf)

rms_err_y = [c.rms_err_y for c in cases]
rms_std_y = [c.rms_std_y for c in cases]
rms_err_dy = [c.rms_err_dy for c in cases]
rms_std_dy = [c.rms_std_dy for c in cases]

rms_err_y_MAP = [c.rms_err_y_MAP for c in cases]
rms_std_y_MAP = [c.rms_std_y_MAP for c in cases]
rms_err_dy_MAP = [c.rms_err_dy_MAP for c in cases]
rms_std_dy_MAP = [c.rms_std_dy_MAP for c in cases]


# Make plots:
import setupplots
setupplots.thesis_format()
import matplotlib.pyplot as plt
plt.ion()

# Set up figures:
# for c in cases:
#     gptools.plot_sampler(
#         c.sampler, burn=400, suptitle=str(c.num_pts),
#         labels=['$%s$' % (l,) for l in gp.free_param_names[:]]
#     )

f = plt.figure(figsize=[setupplots.TEXTWIDTH, setupplots.TEXTWIDTH / len(cases) * 2.0 / 1.618])
f.subplots_adjust(left=0.125, bottom=0.19, right=0.97, top=0.82)
f.suptitle("Core consistency")#, y=1.075)
f_ped = plt.figure(figsize=[setupplots.TEXTWIDTH, setupplots.TEXTWIDTH / len(cases) * 2.0 / 1.618])
f_ped.subplots_adjust(left=0.125, bottom=0.19, right=0.97, top=0.82)
f_ped.suptitle("Edge consistency")#, y=1.075)

for i, c in enumerate(cases):
    n = c.num_pts
    
    # Plot results:
    a_y = f.add_subplot(2, len(cases), i + 1)
    a_y.plot(x_grid, y_grid, 'k--', linewidth=2 * setupplots.lw, label='true curve')
    gptools.univariate_envelope_plot(
        x_grid, c.y_fit_MAP, c.s_y_fit_MAP, ax=a_y, color='r', ls='-.', label=r'\textsc{map}', lw=setupplots.lw
    )
    gptools.univariate_envelope_plot(
        x_grid, c.y_fit, c.s_y_fit, ax=a_y, color='b', label=r'\textsc{mcmc}', lw=setupplots.lw
    )
    a_y.errorbar(
        c.x_meas, c.y_meas, yerr=c.err_y_meas, fmt='o', color='g', markersize=setupplots.ms, label='synthetic data'
    )
    a_dy = f.add_subplot(2, len(cases), i + 1 + len(cases))
    a_dy.plot(x_grid, dy_grid, 'k--', linewidth=2 * setupplots.lw)
    gptools.univariate_envelope_plot(
        x_grid, c.dy_fit_MAP, c.s_dy_fit_MAP, ax=a_dy, color='r', ls='-.', lw=setupplots.lw
    )
    gptools.univariate_envelope_plot(
        x_grid, c.dy_fit, c.s_dy_fit, ax=a_dy, color='b', lw=setupplots.lw
    )
    a_y.set_xlim(0, 1.1)
    a_y.set_ylim(0.5, 1.1)
    a_dy.set_xlim(0, 1.1)
    a_dy.set_ylim(-1, 0.25)
    a_y.set_title('$n=%d$' % (n,))
    plt.setp(a_y.get_xticklabels(), visible=False)
    a_dy.set_xlabel(r'$x$')
    if i == 0:
        a_l = a_y
        a_y.set_ylabel(r'$y$')
        a_dy.set_ylabel(r'$\mathrm{d}y/\mathrm{d}x$')
        a_y.yaxis.set_major_locator(plt.MaxNLocator(nbins=4))
        a_dy.yaxis.set_major_locator(plt.MaxNLocator(nbins=4))
    else:
        plt.setp(a_y.get_yticklabels(), visible=False)
        plt.setp(a_dy.get_yticklabels(), visible=False)
    
    a_y = f_ped.add_subplot(2, len(cases), i + 1)
    a_y.plot(x_grid, y_grid, 'k--', linewidth=2 * setupplots.lw, label='true curve')
    gptools.univariate_envelope_plot(
        x_grid, c.y_fit_MAP, c.s_y_fit_MAP, ax=a_y, color='r', ls='-.', label=r'\textsc{map}', lw=setupplots.lw
    )
    gptools.univariate_envelope_plot(
        x_grid, c.y_fit, c.s_y_fit, ax=a_y, color='b', label=r'\textsc{mcmc}', lw=setupplots.lw
    )
    a_y.errorbar(
        c.x_meas, c.y_meas, yerr=c.err_y_meas, fmt='o', color='g', markersize=setupplots.ms, label='synthetic data'
    )
    a_dy = f_ped.add_subplot(2, len(cases), i + 1 + len(cases))
    a_dy.plot(x_grid, dy_grid, 'k--', linewidth=2 * setupplots.lw)
    gptools.univariate_envelope_plot(
        x_grid, c.dy_fit_MAP, c.s_dy_fit_MAP, ax=a_dy, color='r', ls='-.', lw=setupplots.lw
    )
    gptools.univariate_envelope_plot(
        x_grid, c.dy_fit, c.s_dy_fit, ax=a_dy, color='b', lw=setupplots.lw
    )
    a_y.set_xlim(0.9, 1.1)
    a_y.set_ylim(0, 1.1)
    a_dy.set_xlim(0.9, 1.1)
    a_dy.set_ylim(-15, 2.5)
    a_y.set_title('$n=%d$' % (n,))
    plt.setp(a_y.get_xticklabels(), visible=False)
    a_dy.set_xlabel(r'$x$')
    if i == 0:
        a_l_ped = a_y
        a_y.set_ylabel(r'$y$')
        a_dy.set_ylabel(r'$\mathrm{d}y/\mathrm{d}x$')
        a_y.yaxis.set_major_locator(plt.MaxNLocator(nbins=4))
        a_dy.yaxis.set_major_locator(plt.MaxNLocator(nbins=4))
    else:
        plt.setp(a_y.get_yticklabels(), visible=False)
        plt.setp(a_dy.get_yticklabels(), visible=False)
    
    f.canvas.draw()
    f_ped.canvas.draw()

setupplots.apply_formatter(f)
f.savefig('coreConsistency.pdf')
f.savefig('coreConsistency.pgf')

setupplots.apply_formatter(f_ped)
f_ped.savefig('pedestalConsistency.pdf')
f_ped.savefig('pedestalConsistency.pgf')

# Put legend in seperate figure:
f_leg = plt.figure()
l = f_leg.legend(*a_l_ped.get_legend_handles_labels(), ncol=2, loc='center')
f_leg.canvas.draw()
f_leg.set_figwidth(l.get_window_extent().width / 72)
f_leg.set_figheight(l.get_window_extent().height / 72)
f_leg.savefig("consistencyLeg.pdf", bbox_inches='tight')
f_leg.savefig("consistencyLeg.pgf", bbox_inches='tight')

# Make consistency plot:
ms = 6
alpha = 0.25

f = plt.figure(figsize=(0.75 * setupplots.TEXTWIDTH, 0.75 * 2 / 1.618 * setupplots.TEXTWIDTH))
a_y = f.add_subplot(2, 1, 1)
a_y.loglog(
    num_pts, rms_err_y_MAP, 'r^', markersize=ms, label=r'\textsc{map} $e_{\text{\textsc{rms}}}$', alpha=alpha
)
a_y.loglog(
    num_pts, rms_std_y_MAP, 'ro', markersize=ms,
    label=r'\textsc{map} $\sigma_{\text{\textsc{rms}}}$', alpha=alpha
)
a_y.loglog(
    num_pts, rms_err_y, 'bv', markersize=ms, label=r'\textsc{mcmc} $e_{\text{\textsc{rms}}}$', alpha=alpha
)
a_y.loglog(
    num_pts, rms_std_y, 'bs', markersize=ms,
    label=r'\textsc{mcmc} $\sigma_{\text{\textsc{rms}}}$', alpha=alpha
)
a_y.legend(loc='lower center', ncol=2)
a_y.set_title('(a) value')
a_y.set_ylabel(r'\textsc{rms} error')
plt.setp(a_y.get_xticklabels(), visible=False)
a_dy = f.add_subplot(2, 1, 2)
a_dy.loglog(
    num_pts, rms_err_dy_MAP, 'r^', markersize=ms, label=r'\textsc{map} $e_{\text{\textsc{rms}}}$', alpha=alpha
)
a_dy.loglog(
    num_pts, rms_std_dy_MAP, 'ro', markersize=ms,
    label='\textsc{map} $\sigma_{\text{\textsc{rms}}}$', alpha=alpha
)
a_dy.loglog(
    num_pts, rms_err_dy, 'bv', markersize=ms, label=r'\textsc{mcmc} $e_{\text{\textsc{rms}}}$', alpha=alpha
)
a_dy.loglog(
    num_pts, rms_std_dy, 'bs', markersize=ms,
    label=r'\textsc{mcmc} $\sigma_{\text{\textsc{rms}}}$', alpha=alpha
)

a_dy.set_title('(b) gradient')
a_dy.set_xlabel('number of points')
a_dy.set_ylabel(r'\textsc{rms} error')

f.suptitle(r"Asymptotic consistency of \textsc{gpr} fits")

setupplots.apply_formatter(f)
f.savefig("asymptoticConsistency.pdf", bbox_inches='tight')
f.savefig("asymptoticConsistency.pgf", bbox_inches='tight')
