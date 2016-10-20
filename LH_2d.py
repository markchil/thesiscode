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

# This script makes figures 2.32, 2.33, 2.34, A.15, A.16 and A.17, which show
# the 2D fits to sawtooth-free data.

from __future__ import division
import profiletools
import gptools
import scipy
import scipy.interpolate
import copy
import cPickle as pkl

class Case(object):
    def __init__(self, shot, t_min, t_max, abscissa):
        self.shot = shot
        self.t_min = t_min
        self.t_max = t_max
        self.abscissa = abscissa

class TimeCase(object):
    def __init__(self, t):
        self.t = t

c = Case(1110329013, 1.0, 1.4, 'r/a')
c.t_grid = scipy.linspace(c.t_min, c.t_max, 25)
c.roa_grid = scipy.linspace(0.0, 1.0, 50)

c.T, c.R = scipy.meshgrid(c.t_grid, c.roa_grid)
c.shape = c.T.shape
c.X = scipy.hstack((c.T.reshape(-1, 1), c.R.reshape(-1, 1)))

c.p_CTS = profiletools.TeCTS(c.shot, t_min=c.t_min, t_max=c.t_max, abscissa=c.abscissa)
c.p_ETS = profiletools.TeETS(c.shot, t_min=c.t_min, t_max=c.t_max, abscissa=c.abscissa)
c.p = copy.deepcopy(c.p_CTS)
c.p.add_profile(c.p_ETS)

k = gptools.SquaredExponentialKernel(
    num_dim=2,
    hyperprior=gptools.UniformJointPrior([(0.0, 10.0), (0.0, 100.0)]) * gptools.GammaJointPriorAlt(1.0, 0.2),
)
c.p.create_gp(k=k, constrain_at_limiter=True, constrain_slope_on_axis=True, use_hyper_deriv=False)
c.p.gp.noise_k = gptools.DiagonalNoiseKernel(num_dim=2, initial_noise=0.1, noise_bound=[0.0, 1.0])
c.p.find_gp_MAP_estimate(verbose=True, random_starts=8, method='SLSQP')
c.MAP_params_2d = c.p.gp.free_params[:]
c.sampler_2d = c.p.gp.sample_hyperparameter_posterior(nsamp=250, sampler_a=4.0)
c.sampler_2d.pool.close()
c.sampler_2d.pool = None
c.mean_val, c.std_val = c.p.smooth(
    c.X, n=scipy.tile([0, 0], (c.X.shape[0], 1)), use_MCMC=True, sampler=c.sampler_2d, burn=150, thin=100
)
c.mean_grad, c.std_grad = c.p.smooth(
    c.X, n=scipy.tile([0, 1], (c.X.shape[0], 1)), use_MCMC=True, sampler=c.sampler_2d, burn=150, thin=100
)

c.mean_val = c.mean_val.reshape(c.shape)
c.std_val = c.std_val.reshape(c.shape)
c.mean_grad = c.mean_grad.reshape(c.shape)
c.std_grad = c.std_grad.reshape(c.shape)

# Use this code if you want to account for cov(mean, grad). It is rather
# memory-intensive.
# mean, cov = c.p.smooth(
#     scipy.vstack((c.X, c.X)),
#     n=scipy.vstack((scipy.tile([0, 0], (c.X.shape[0], 1)), scipy.tile([0, 1], (c.X.shape[0], 1)))),
#     use_MCMC=True, sampler=c.sampler_2d, burn=100, thin=100
# )
# std = scipy.sqrt(scipy.diag(cov))
# c.mean_val = mean[:c.X.shape[0]].reshape(shape)
# c.std_val = std[:c.X.shape[0]].reshape(shape)
# c.mean_grad = mean[c.X.shape[0]:].reshape(shape)
# c.std_grad = std[c.X.shape[0]:].reshape(shape)

# Compute a/L:
c.mean_a_L = -c.mean_grad / c.mean_val
# Use this code if you are accounting for cov(mean, grad) (must also change
# above).
# i = range(0, len(c.X))
# j = range(len(c.X), 2 * len(c.X))
# c.cov_val_grad = scipy.asarray(cov[i, j]).reshape(c.shape)
# c.std_a_L = scipy.sqrt(
#     c.std_val**2.0 * c.mean_grad**2.0 / c.mean_val**4.0 +
#     c.std_grad**2.0 / c.mean_val**(2.0) -
#     2.0 * c.cov_val_grad * c.mean_grad / c.mean_val**3.0
# )
c.std_a_L = scipy.sqrt(c.std_val**2.0 * c.mean_grad**2.0 / c.mean_val**4.0 + c.std_grad**2.0 / c.mean_val**(2.0))

# Compare averaging schemes:
# All points:
print('all points')
c.p_all = copy.deepcopy(c.p)
c.p_all.drop_axis(0)
c.p_all_CTS = copy.deepcopy(c.p_CTS)
c.p_all_CTS.drop_axis(0)
c.p_all_ETS = copy.deepcopy(c.p_ETS)
c.p_all_ETS.drop_axis(0)
c.p_all.create_gp(k='SE', constrain_at_limiter=True, use_hyper_deriv=True)
c.p_all.gp.k.hyperprior = gptools.UniformJointPrior([(0.0, 10.0),]) * gptools.GammaJointPriorAlt(1.0, 0.2)
c.p_all.gp.noise_k = gptools.DiagonalNoiseKernel(initial_noise=0.1, noise_bound=[0.0, 1.0])
c.p_all.find_gp_MAP_estimate(verbose=True)
c.MAP_params_all = c.p_all.gp.free_params[:]
c.p_all.gp.use_hyper_deriv = False
c.sampler_all = c.p_all.gp.sample_hyperparameter_posterior(nsamp=200)
c.sampler_all.pool.close()
c.sampler_all.pool = None
c.out_all = c.p_all.compute_a_over_L(
    c.roa_grid, return_prediction=True, use_MCMC=True, sampler=c.sampler_all, burn=100, thin=100
)

# Average:
print('average')
c.p_avg = copy.deepcopy(c.p)
c.p_avg.time_average(weighted=True)
c.p_avg_CTS = copy.deepcopy(c.p_CTS)
c.p_avg_CTS.time_average(weighted=True)
c.p_avg_ETS = copy.deepcopy(c.p_ETS)
c.p_avg_ETS.time_average(weighted=True)
c.p_avg.create_gp(k='SE', constrain_at_limiter=True, use_hyper_deriv=True)
c.p_avg.gp.k.hyperprior = gptools.UniformJointPrior([(0.0, 10.0),]) * gptools.GammaJointPriorAlt(1.0, 0.2)
c.p_avg.gp.noise_k = gptools.DiagonalNoiseKernel(initial_noise=0.1, noise_bound=[0.0, 1.0])
c.p_avg.find_gp_MAP_estimate(verbose=True)
c.MAP_params_avg = c.p_avg.gp.free_params[:]
c.p_avg.gp.use_hyper_deriv = False
c.sampler_avg = c.p_avg.gp.sample_hyperparameter_posterior(nsamp=200)
c.sampler_avg.pool.close()
c.sampler_avg.pool = None
c.out_avg = c.p_avg.compute_a_over_L(
    c.roa_grid, return_prediction=True, use_MCMC=True, sampler=c.sampler_avg, burn=100, thin=100
)

# Of mean:
print('of mean')
c.p_of_mean = copy.deepcopy(c.p)
c.p_of_mean.time_average(weighted=True, y_method='of mean')
c.p_of_mean_CTS = copy.deepcopy(c.p_CTS)
c.p_of_mean_CTS.time_average(weighted=True, y_method='of mean')
c.p_of_mean_ETS = copy.deepcopy(c.p_ETS)
c.p_of_mean_ETS.time_average(weighted=True, y_method='of mean')
c.p_of_mean.create_gp(k='SE', constrain_at_limiter=True, use_hyper_deriv=True)
c.p_of_mean.gp.k.hyperprior = gptools.UniformJointPrior([(0.0, 10.0),]) * gptools.GammaJointPriorAlt(1.0, 0.2)
c.p_of_mean.gp.noise_k = gptools.DiagonalNoiseKernel(initial_noise=0.1, noise_bound=[0.0, 1.0])
c.p_of_mean.find_gp_MAP_estimate(verbose=True)
c.MAP_params_of_mean = c.p_of_mean.gp.free_params[:]
c.p_of_mean.gp.use_hyper_deriv = False
c.sampler_of_mean = c.p_of_mean.gp.sample_hyperparameter_posterior(nsamp=200)
c.sampler_of_mean.pool.close()
c.sampler_of_mean.pool = None
c.out_of_mean = c.p_of_mean.compute_a_over_L(
    c.roa_grid, return_prediction=True, use_MCMC=True, sampler=c.sampler_of_mean, burn=100, thin=100
)

# Averaging and single-slice comparisons:
c.t_cases = [TimeCase(1.1), TimeCase(1.2), TimeCase(1.3), TimeCase(1.4)]

for tc in c.t_cases:
    tv = tc.t
    print(tv)
    tc.p_CTS = copy.deepcopy(c.p_CTS)
    tc.p_ETS = copy.deepcopy(c.p_ETS)
    tc.p_CTS.keep_times(tv, tol=0.015)
    tc.p_ETS.keep_times(tv, tol=0.015)
    tc.p = copy.deepcopy(tc.p_CTS)
    tc.p.add_profile(tc.p_ETS)
    t = tc.p.X[0, 0]
    tc.p.drop_axis(0)
    tc.p_CTS.drop_axis(0)
    tc.p_ETS.drop_axis(0)
    tc.p.create_gp(k='SE', constrain_at_limiter=True, use_hyper_deriv=True)
    tc.p.gp.k.hyperprior = gptools.UniformJointPrior([(0.0, 10.0),]) * gptools.GammaJointPriorAlt(1.0, 0.2)
    tc.p.gp.noise_k = gptools.DiagonalNoiseKernel(initial_noise=0.1, noise_bound=[0.0, 1.0])
    tc.p.find_gp_MAP_estimate(verbose=True)
    tc.MAP_params = tc.p.gp.free_params[:]
    tc.p.gp.use_hyper_deriv = False
    tc.sampler = tc.p.gp.sample_hyperparameter_posterior(nsamp=200)
    tc.sampler.pool.close()
    tc.sampler.pool = None
    tc.out = tc.p.compute_a_over_L(
        c.roa_grid, return_prediction=True, use_MCMC=True, sampler=tc.sampler, burn=100, thin=100
    )
    
    X_t = scipy.zeros((2 * len(c.roa_grid), 2))
    X_t[:, 0] = t
    X_t[:, 1] = scipy.concatenate((c.roa_grid, c.roa_grid))
    n_t = scipy.vstack((scipy.tile([0, 0], (len(c.roa_grid), 1)), scipy.tile([0, 1], (len(c.roa_grid), 1))))
    mean_t, cov_t = c.p.gp.predict(
        X_t, n=n_t, return_cov=True, use_MCMC=True, sampler=c.sampler_2d, burn=150, thin=100
    )
    std_t = scipy.sqrt(scipy.diag(cov_t))
    tc.mean_val_t = mean_t[:len(c.roa_grid)]
    tc.std_val_t = std_t[:len(c.roa_grid)]
    tc.mean_grad_t = mean_t[len(c.roa_grid):]
    tc.std_grad_t = std_t[len(c.roa_grid):]
    tc.mean_a_L_t = -tc.mean_grad_t / tc.mean_val_t
    i = range(0, len(c.roa_grid))
    j = range(len(c.roa_grid), 2 * len(c.roa_grid))
    tc.cov_val_grad_t = scipy.asarray(cov_t[i, j]).flatten()
    tc.std_a_L_t = scipy.sqrt(
        tc.std_val_t**2.0 * tc.mean_grad_t**2.0 / tc.mean_val_t**4.0 +
        tc.std_grad_t**2.0 / tc.mean_val_t**(2.0) -
        2 * tc.cov_val_grad_t * tc.mean_grad_t / tc.mean_val_t**3.0
    )

with open('LH_2d.pkl', 'wb') as pf:
    pkl.dump(c, pf)

# Make plots:
import setupplots
setupplots.thesis_format()
import matplotlib.pyplot as plt
plt.ion()
import matplotlib.gridspec as mplgs

f_comp = plt.figure(figsize=(setupplots.TEXTHEIGHT, 3 * setupplots.TEXTHEIGHT / 1.618 / len(c.t_cases)))
gs = mplgs.GridSpec(3, len(c.t_cases))

for i, tc in enumerate(c.t_cases):
    t = tc.t
    print(t)
    
    gptools.plot_sampler(
        tc.sampler, labels=[r'$\sigma_f$ [keV]', r'$\ell$', r'$\sigma_{\text{n}}$ [keV]'], burn=100,
        suptitle=r"$t=\SI{%.1f}{s}$" % (t,), label_fontsize=11,
        chain_ytick_pad=0.8, fixed_width=0.5 * setupplots.TEXTWIDTH,
        chain_ticklabel_fontsize=11, bottom_sep=0.185, max_hist_ticks=6,
        max_chain_ticks=5, ax_space=0.225, suptitle_space=0.05,
        cmap='plasma', hide_chain_ylabels=True, ticklabel_fontsize=11
    )
    setupplots.apply_formatter(plt.gcf())
    plt.savefig("Two%dMarg.pgf" % (int(t * 10),), bbox_inches='tight')
    plt.savefig("Two%dMarg.pdf" % (int(t * 10),), bbox_inches='tight')
    
    a_val = f_comp.add_subplot(gs[0, i])
    a_grad = f_comp.add_subplot(gs[1, i], sharex=a_val)
    a_a_L = f_comp.add_subplot(gs[2, i], sharex=a_val)
    
    tc.p_CTS.plot_data(ax=a_val, label_axes=False, label=r'\textsc{cts}', fmt='bs', markersize=setupplots.ms)
    tc.p_ETS.plot_data(ax=a_val, label_axes=False, label=r'\textsc{ets}', fmt='gs', markersize=setupplots.ms)
    gptools.univariate_envelope_plot(c.roa_grid, tc.out['mean_val'], tc.out['std_val'], ax=a_val, label=r'1\textsc{d}', color='g', linewidth=setupplots.lw)
    gptools.univariate_envelope_plot(c.roa_grid, tc.out['mean_grad'], tc.out['std_grad'], ax=a_grad, label=r'1\textsc{d}', color='g', linewidth=setupplots.lw)
    gptools.univariate_envelope_plot(c.roa_grid, tc.out['mean_a_L'], tc.out['std_a_L'], ax=a_a_L, label=r'1\textsc{d}', color='g', linewidth=setupplots.lw)
    
    gptools.univariate_envelope_plot(c.roa_grid, tc.mean_val_t, tc.std_val_t, ax=a_val, label=r'2\textsc{d}', color='r', ls='--', linewidth=setupplots.lw)
    gptools.univariate_envelope_plot(c.roa_grid, tc.mean_grad_t, tc.std_grad_t, ax=a_grad, label=r'2\textsc{d}', color='r', ls='--', linewidth=setupplots.lw)
    gptools.univariate_envelope_plot(c.roa_grid, tc.mean_a_L_t, tc.std_a_L_t, ax=a_a_L, label=r'2\textsc{d}', color='r', ls='--', linewidth=setupplots.lw)
    
    a_val.set_ylim([0.0, 3.5])
    a_val.set_xlim([0.0, 1.0])
    a_grad.set_ylim([-5.0, 0.0])
    a_a_L.set_ylim([0.0, 15.0])
    plt.setp(a_val.get_xticklabels(), visible=False)
    plt.setp(a_grad.get_xticklabels(), visible=False)
    
    a_val.set_title(r"$t=\SI{%.1f}{s}$" % (t,))
    
    a_a_L.set_xlabel(r'$r/a$')
    if i == 0:
        a_val.set_ylabel(r'$T_{\mathrm{e}}$ [keV]')
        a_grad.set_ylabel(r'$\mathrm{d}T_{\mathrm{e}}/\mathrm{d}(r/a)$ [keV]')
        a_a_L.set_ylabel(r'$a/L_{T_{\mathrm{e}}}$')
    else:
        plt.setp(a_val.get_yticklabels(), visible=False)
        plt.setp(a_grad.get_yticklabels(), visible=False)
        plt.setp(a_a_L.get_yticklabels(), visible=False)
    
    tc.median_unc_val_1d = 100 * scipy.median(scipy.absolute(tc.out['std_val'] / tc.out['mean_val']))
    tc.median_unc_val_2d = 100 * scipy.median(scipy.absolute(tc.std_val_t / tc.mean_val_t))
    tc.median_unc_grad_1d = 100 * scipy.median(scipy.absolute(tc.out['std_grad'] / tc.out['mean_grad']))
    tc.median_unc_grad_2d = 100 * scipy.median(scipy.absolute(tc.std_grad_t / tc.mean_grad_t))
    tc.median_unc_a_L_1d = 100 * scipy.median(scipy.absolute(tc.out['std_a_L'] / tc.out['mean_a_L']))
    tc.median_unc_a_L_2d = 100 * scipy.median(scipy.absolute(tc.std_a_L_t / tc.mean_a_L_t))

f_comp.subplots_adjust(left=0.07, bottom=0.12, right=0.98, top=0.93)
setupplots.apply_formatter(f_comp)
f_comp.canvas.draw()
f_comp.savefig("TwotProf.pdf")
f_comp.savefig("TwotProf.pgf")

# Put legend in separate figure:
f_leg = plt.figure()
l = f_leg.legend(*a_val.get_legend_handles_labels(), ncol=4, loc='center')
f_leg.canvas.draw()
f_leg.set_figwidth(l.get_window_extent().width / 72)
f_leg.set_figheight(l.get_window_extent().height / 72)
f_leg.savefig("Twotleg.pdf", bbox_inches='tight')
f_leg.savefig("Twotleg.pgf", bbox_inches='tight')

# Make plots from 2D fit:
gptools.plot_sampler(
    c.sampler_2d, labels=[r'$\sigma_f$ [keV]', r'$\ell_t$ [s]', r'$\ell$', r'$\sigma_{\text{n}}$ [keV]'],
    burn=100, suptitle=r"Marginal posterior distributions for 2\textsc{d} \textsc{se} covariance kernel", label_fontsize=11,
    chain_ytick_pad=0.8, fixed_width=setupplots.TEXTWIDTH,
    chain_ticklabel_fontsize=11, bottom_sep=0.1125, ax_space=0.2, suptitle_space=0.05,
    cmap='plasma', hide_chain_ylabels=True, ticklabel_fontsize=11
)
setupplots.apply_formatter(plt.gcf())
plt.savefig("TwoMarg.pgf", bbox_inches='tight')
plt.savefig("TwoMarg.pdf", bbox_inches='tight')

f = plt.figure(figsize=(setupplots.TEXTWIDTH, setupplots.TEXTWIDTH / 1.618))
a = f.add_subplot(111, projection='3d')
c.p_CTS.plot_data(label_axes=False, label=r'\textsc{cts}', fmt='bs', ax=a, ms=3)
c.p_ETS.plot_data(label_axes=False, label=r'\textsc{ets}', fmt='gs', ax=a, ms=3)
surf = a.plot_surface(
    c.T, c.R, c.mean_val, cmap='seismic', linewidth=0, cstride=1, rstride=1,
    vmin=-scipy.absolute(c.mean_val).max(), vmax=scipy.absolute(c.mean_val).max()
)
a.set_zlim([0, 4.0])
a.set_xlim([c.t_min, c.t_max])
a.set_ylim([0, 1.0])
a.view_init(elev=34, azim=28)
a.set_title(r"2\textsc{d} fit to $T_{\mathrm{e}}$ profile")
a.set_xlabel('$t$ [s]', labelpad=0)
a.set_ylabel('$r/a$', labelpad=0)
a.set_zlabel(r'$T_{\text{e}}$ [keV]', labelpad=0)
for tick in a.get_yaxis().get_major_ticks():
    tick.set_pad(0)
    tick.label1 = tick._get_text1()
for tick in a.get_xaxis().get_major_ticks():
    tick.set_pad(0)
    tick.label1 = tick._get_text1()
for tick in a.zaxis.get_major_ticks():
    tick.set_pad(0)
    tick.label1 = tick._get_text1()
f.subplots_adjust(left=0.05, right=1, bottom=0.05, top=1)
setupplots.apply_formatter(f)
f.savefig("TwoTe.pgf")
f.savefig("TwoTe.pdf")

f_grad = plt.figure(figsize=(0.5 * setupplots.TEXTWIDTH, 0.5 * setupplots.TEXTWIDTH))
a_grad = f_grad.add_subplot(111, projection='3d')
surf = a_grad.plot_surface(
    c.T, c.R, c.mean_grad, cmap='seismic', linewidth=0, cstride=1, rstride=1,
    vmin=-scipy.absolute(c.mean_grad).max(), vmax=scipy.absolute(c.mean_grad).max()
)
a_grad.set_xlim([c.t_min, c.t_max])
a_grad.set_ylim([0, 1.0])
a_grad.view_init(elev=34, azim=28)
a_grad.set_xlabel('$t$ [s]', labelpad=0)
a_grad.set_ylabel('$r/a$', labelpad=0)
a_grad.set_zlabel('$\mathrm{d}T_{\mathrm{e}}/\mathrm{d}(r/a)$ [keV]', labelpad=0)
a_grad.set_title(r"2\textsc{d} temperature gradient")
f_grad.subplots_adjust(left=0.1, right=1, bottom=0.075, top=0.99)
for tick in a_grad.get_yaxis().get_major_ticks():
    tick.set_pad(0)
    tick.label1 = tick._get_text1()
for tick in a_grad.get_xaxis().get_major_ticks():
    tick.set_pad(0)
    tick.label1 = tick._get_text1()
for tick in a_grad.zaxis.get_major_ticks():
    tick.set_pad(0)
    tick.label1 = tick._get_text1()
a_grad.xaxis.set_major_locator(plt.MaxNLocator(nbins=4))
a_grad.yaxis.set_major_locator(plt.MaxNLocator(nbins=4))
a_grad.zaxis.set_major_locator(plt.MaxNLocator(nbins=4))
setupplots.apply_formatter(f_grad)
f_grad.savefig("TwoGrad.pgf")
f_grad.savefig("TwoGrad.pdf")

f_a_L = plt.figure(figsize=(0.5 * setupplots.TEXTWIDTH, 0.5 * setupplots.TEXTWIDTH))
a_a_L = f_a_L.add_subplot(111, projection='3d')
surf = a_a_L.plot_surface(
    c.T, c.R, c.mean_a_L, cmap='seismic', linewidth=0, cstride=1, rstride=1,
    vmin=-scipy.absolute(c.mean_a_L).max(), vmax=scipy.absolute(c.mean_a_L).max()
)
a_a_L.set_xlim([c.t_min, c.t_max])
a_a_L.set_ylim([0, 1.0])
a_a_L.view_init(elev=34, azim=28)
a_a_L.set_xlabel('$t$ [s]', labelpad=0)
a_a_L.set_ylabel('$r/a$', labelpad=0)
a_a_L.set_zlabel('$a/L_{T_{\mathrm{e}}}$', labelpad=0)
a_a_L.set_title(r"2\textsc{d} gradient scale length")
f_a_L.subplots_adjust(left=0.1, right=1, bottom=0.075, top=0.99)
for tick in a_a_L.get_yaxis().get_major_ticks():
    tick.set_pad(0)
    tick.label1 = tick._get_text1()
for tick in a_a_L.get_xaxis().get_major_ticks():
    tick.set_pad(0)
    tick.label1 = tick._get_text1()
for tick in a_a_L.zaxis.get_major_ticks():
    tick.set_pad(0)
    tick.label1 = tick._get_text1()
a_a_L.xaxis.set_major_locator(plt.MaxNLocator(nbins=4))
a_a_L.yaxis.set_major_locator(plt.MaxNLocator(nbins=4))
a_a_L.zaxis.set_major_locator(plt.MaxNLocator(nbins=4))
setupplots.apply_formatter(f_a_L)
f_a_L.savefig("TwoaOverL.pgf")
f_a_L.savefig("TwoaOverL.pdf")

# Make sampler plots for the averaging schemes:
gptools.plot_sampler(
    c.sampler_all, labels=[r'$\sigma_f$ [keV]', r'$\ell$', r'$\sigma_{\text{n}}$ [keV]'],
    burn=100, suptitle=r"All points", label_fontsize=11,
        chain_ytick_pad=0.8, fixed_width=0.5 * setupplots.TEXTWIDTH,
        chain_ticklabel_fontsize=11, bottom_sep=0.175, max_hist_ticks=6,
        max_chain_ticks=5, ax_space=0.225, suptitle_space=0.075,
        cmap='plasma', hide_chain_ylabels=True, ticklabel_fontsize=11
)
setupplots.apply_formatter(plt.gcf())
plt.savefig("TwoAllPointsMarg.pgf", bbox_inches='tight')
plt.savefig("TwoAllPointsMarg.pdf", bbox_inches='tight')

gptools.plot_sampler(
    c.sampler_avg, labels=[r'$\sigma_f$ [keV]', r'$\ell$', r'$\sigma_{\text{n}}$ [keV]'],
    burn=100, suptitle=r"Averaged", label_fontsize=11,
        chain_ytick_pad=0.8, fixed_width=0.5 * setupplots.TEXTWIDTH,
        chain_ticklabel_fontsize=11, bottom_sep=0.175, max_hist_ticks=6,
        max_chain_ticks=5, ax_space=0.225, suptitle_space=0.075,
        cmap='plasma', hide_chain_ylabels=True, ticklabel_fontsize=11
)
setupplots.apply_formatter(plt.gcf())
plt.savefig("TwoAveragedMarg.pgf", bbox_inches='tight')
plt.savefig("TwoAveragedMarg.pdf", bbox_inches='tight')

gptools.plot_sampler(
    c.sampler_of_mean, labels=[r'$\sigma_f$ [keV]', r'$\ell$', r'$\sigma_{\text{n}}$ [keV]'],
    burn=100, suptitle=r"Averaged, $\sigma/\sqrt{n}$", label_fontsize=11,
        chain_ytick_pad=0.8, fixed_width=0.5 * setupplots.TEXTWIDTH,
        chain_ticklabel_fontsize=11, bottom_sep=0.175, max_hist_ticks=6,
        max_chain_ticks=5, ax_space=0.225, suptitle_space=0.075,
        cmap='plasma', hide_chain_ylabels=True, ticklabel_fontsize=11
)
setupplots.apply_formatter(plt.gcf())
plt.savefig("TwoOfMeanMarg.pgf", bbox_inches='tight')
plt.savefig("TwoOfMeanMarg.pdf", bbox_inches='tight')

# Make plot comparing averaging schemes:
f_comp = plt.figure(figsize=(0.65 * setupplots.TEXTWIDTH, 0.65 * 3 * setupplots.TEXTWIDTH / 1.618))
a_val = f_comp.add_subplot(3, 1, 1)
a_grad = f_comp.add_subplot(3, 1, 2, sharex=a_val)
a_a_L = f_comp.add_subplot(3, 1, 3, sharex=a_val)

# All points:
c.p_all.plot_data(ax=a_val, label_axes=False, color='b', ms=3)
gptools.univariate_envelope_plot(
    c.roa_grid, c.out_all['mean_val'], c.out_all['std_val'], ax=a_val, label='all points', color='b', linestyle='-',
    envelopes=[1,]
)
gptools.univariate_envelope_plot(
    c.roa_grid, c.out_all['mean_grad'], c.out_all['std_grad'], ax=a_grad, label='all points', color='b', linestyle='-',
    envelopes=[1,]
)
gptools.univariate_envelope_plot(
    c.roa_grid, c.out_all['mean_a_L'], c.out_all['std_a_L'], ax=a_a_L, label='all points', color='b', linestyle='-',
    envelopes=[1,]
)

c.median_unc_val_all = 100 * scipy.median(scipy.absolute(c.out_all['std_val'] / c.out_all['mean_val']))
c.median_unc_grad_all = 100 * scipy.median(scipy.absolute(c.out_all['std_grad'] / c.out_all['mean_grad']))
c.median_unc_a_L_all = 100 * scipy.median(scipy.absolute(c.out_all['std_a_L'] / c.out_all['mean_a_L']))

print(
    "Median relative uncertainty in Te from all points: %.2f%%\n"
    "Median relative uncertainty in dTe/droa from all points: %.2f%%\n"
    "Median relative uncertainty in a/LTe from error propagation from all points: %.2f%%" %
    (c.median_unc_val_all, c.median_unc_grad_all, c.median_unc_a_L_all)
)

# Average:
c.p_avg.plot_data(ax=a_val, label_axes=False, color='g', ms=6)
gptools.univariate_envelope_plot(
    c.roa_grid, c.out_avg['mean_val'], c.out_avg['std_val'], ax=a_val, label='averaged', color='g', linestyle='-.',
    envelopes=[1,]
)
gptools.univariate_envelope_plot(
    c.roa_grid, c.out_avg['mean_grad'], c.out_avg['std_grad'], ax=a_grad, label='averaged', color='g', linestyle='-.',
    envelopes=[1,]
)
gptools.univariate_envelope_plot(
    c.roa_grid, c.out_avg['mean_a_L'], c.out_avg['std_a_L'], ax=a_a_L, label='averaged', color='g', linestyle='-.',
    envelopes=[1,]
)

c.median_unc_val_avg = 100 * scipy.median(scipy.absolute(c.out_avg['std_val'] / c.out_avg['mean_val']))
c.median_unc_grad_avg = 100 * scipy.median(scipy.absolute(c.out_avg['std_grad'] / c.out_avg['mean_grad']))
c.median_unc_a_L_avg = 100 * scipy.median(scipy.absolute(c.out_avg['std_a_L'] / c.out_avg['mean_a_L']))

print(
    "Median relative uncertainty in Te from average: %.2f%%\n"
    "Median relative uncertainty in dTe/droa from average: %.2f%%\n"
    "Median relative uncertainty in a/LTe from error propagation from average: %.2f%%" %
    (c.median_unc_val_avg, c.median_unc_grad_avg, c.median_unc_a_L_avg)
)

# Of mean:
c.p_of_mean.plot_data(ax=a_val, label_axes=False, color='y', ms=3)
gptools.univariate_envelope_plot(
    c.roa_grid, c.out_of_mean['mean_val'], c.out_of_mean['std_val'], ax=a_val, label=r'averaged, $\sigma/\sqrt{n}$', color='y', linestyle=':',
    envelopes=[1,]
)
gptools.univariate_envelope_plot(
    c.roa_grid, c.out_of_mean['mean_grad'], c.out_of_mean['std_grad'], ax=a_grad, label=r'averaged, $\sigma/\sqrt{n}$', color='y', linestyle=':',
    envelopes=[1,]
)
gptools.univariate_envelope_plot(
    c.roa_grid, c.out_of_mean['mean_a_L'], c.out_of_mean['std_a_L'], ax=a_a_L, label=r'averaged, $\sigma/\sqrt{n}$', color='y', linestyle=':',
    envelopes=[1,]
)

c.median_unc_val_of_mean = 100 * scipy.median(scipy.absolute(c.out_of_mean['std_val'] / c.out_of_mean['mean_val']))
c.median_unc_grad_of_mean = 100 * scipy.median(scipy.absolute(c.out_of_mean['std_grad'] / c.out_of_mean['mean_grad']))
c.median_unc_a_L_of_mean = 100 * scipy.median(scipy.absolute(c.out_of_mean['std_a_L'] / c.out_of_mean['mean_a_L']))

print(
    "Median relative uncertainty in Te from of mean: %.2f%%\n"
    "Median relative uncertainty in dTe/droa from of mean: %.2f%%\n"
    "Median relative uncertainty in a/LTe from error propagation from of mean: %.2f%%" %
    (c.median_unc_val_of_mean, c.median_unc_grad_of_mean, c.median_unc_a_L_of_mean)
)

# Add a slice of the 2D fit:
gptools.univariate_envelope_plot(
    c.roa_grid, c.mean_val[:, 0], c.std_val[:, 0], ax=a_val, label=r'2\textsc{d}, $t=1.0$s', color='r', linestyle='--',
    envelopes=[1,]
)
gptools.univariate_envelope_plot(
    c.roa_grid, c.mean_grad[:, 0], c.std_grad[:, 0], ax=a_grad, label=r'2\textsc{d}, $t=1.0$s', color='r', linestyle='--',
    envelopes=[1,]
)
gptools.univariate_envelope_plot(
    c.roa_grid, c.mean_a_L[:, 0], c.std_a_L[:, 0], ax=a_a_L, label=r'2\textsc{d}, $t=1.0$s', color='r', linestyle='--',
    envelopes=[1,]
)

a_val.set_ylim([0.0, 3.5])
a_val.set_xlim([0.0, 1.0])
a_grad.set_ylim([-4.0, 0.0])
a_a_L.set_ylim([0.0, 15.0])
a_a_L.legend(loc='upper left', ncol=1)
plt.setp(a_val.get_xticklabels(), visible=False)
plt.setp(a_grad.get_xticklabels(), visible=False)

a_val.set_title(r"Comparison of averaging schemes")

a_a_L.set_xlabel(r'$r/a$')
a_val.set_ylabel(r'$T_{\mathrm{e}}$ [keV]')
a_grad.set_ylabel(r'$\mathrm{d}T_{\mathrm{e}}/\mathrm{d}(r/a)$ [keV]')
a_a_L.set_ylabel(r'$a/L_{T_{\mathrm{e}}}$')

f_comp.subplots_adjust(left=0.14, bottom=0.06, right=0.96, top=0.96, hspace=0.08)
setupplots.apply_formatter(f_comp)
f_comp.canvas.draw()
f_comp.savefig("TwoAvgSliceNew.pdf", bbox_inches='tight')
f_comp.savefig("TwoAvgSliceNew.pgf", bbox_inches='tight')

# Generate posterior summary table:
post_sum = setupplots.generate_post_sum(
    [c.MAP_params_2d, c.MAP_params_all, c.MAP_params_avg, c.MAP_params_of_mean] + [tc.MAP_params for tc in c.t_cases],
    [c.sampler_2d, c.sampler_all, c.sampler_avg, c.sampler_of_mean] + [tc.sampler for tc in c.t_cases],
    [150,] + [100,] * (3 + len(c.t_cases)),
    [
        [[r'2\textsc{d} \textsc{se}', r'$\sigma_f$ [keV]'], ['', r'temporal $\ell_t$ [s]'], ['', r'spatial $\ell$'], ['', r'$\sigma_\text{n}$ [keV]']],
        [[r'1\textsc{d} \textsc{se},', r'$\sigma_f$ [keV]'], ['all points', r'spatial $\ell$'], ['', r'$\sigma_\text{n}$ [keV]']],
        [[r'1\textsc{d} \textsc{se},', r'$\sigma_f$ [keV]'], ['averaged', r'spatial $\ell$'], ['', r'$\sigma_\text{n}$ [keV]']],
        [[r'1\textsc{d} \textsc{se},', r'$\sigma_f$ [keV]'], ['averaged,', r'spatial $\ell$'], [r'$\sigma/\sqrt{n}$', r'$\sigma_\text{n}$ [keV]']],
    ] + [[[r'1\textsc{d} \textsc{se},', r'$\sigma_f$ [keV]'], ['$t = \\SI{%.1f}{s}$' % tc.t, r'spatial $\ell$'], ['', r'$\sigma_\text{n}$ [keV]']] for tc in c.t_cases]
)

with open("LH_2d_post_sum.tex", 'w') as tf:
    tf.write(post_sum)

# Generate relative unceratainty summary:
unc_sum = setupplots.generate_latex_tabular(
    ['%s', '%s', '%.4g', '%.4g', '%.4g'],
    [0, 3, 5, 7, 9],
    ['{$[1.0, 1.4]$}', '', '', '1.1', '', '1.2', '', '1.3', '', '1.4', ''],
    ['all points', 'averaged', r'averaged, $\sigma/\sqrt{n}$', r'2\textsc{d} \textsc{se}', r'1\textsc{d} \textsc{se}', r'2\textsc{d} \textsc{se}', r'1\textsc{d} \textsc{se}', r'2\textsc{d} \textsc{se}', r'1\textsc{d} \textsc{se}', r'2\textsc{d} \textsc{se}', r'1\textsc{d} \textsc{se}'],
    [c.median_unc_val_all, c.median_unc_val_avg, c.median_unc_val_of_mean] + sum([[tc.median_unc_val_2d, tc.median_unc_val_1d]  for tc in c.t_cases], []),
    [c.median_unc_grad_all, c.median_unc_grad_avg, c.median_unc_grad_of_mean] + sum([[tc.median_unc_grad_2d, tc.median_unc_grad_1d] for tc in c.t_cases], []),
    [c.median_unc_a_L_all, c.median_unc_a_L_avg, c.median_unc_a_L_of_mean] + sum([[tc.median_unc_a_L_2d, tc.median_unc_a_L_1d] for tc in c.t_cases], [])
)

with open("LH_2d_rel_unc.tex", 'w') as tf:
    tf.write(unc_sum)
