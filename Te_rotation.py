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

# This script makes figures 2.29, 2.30 and A.14, which show the Te profiles for
# the two shots on either side of the rotation reversal threshold.

from __future__ import division
import profiletools
import scipy
import scipy.io
import scipy.interpolate
import eqtools
import gptools
import cPickle as pkl

roa_star = scipy.linspace(0.0, 1.0, 400)
core_mask = (roa_star <= 1.0)

# Set shot, time range of interest:
class Case(object):
    def __init__(self, shot, color, label):
        self.shot = shot
        self.color = color
        self.label = label

cases = [Case(1120221011, 'r', 'hollow'), Case(1120221012, 'b', 'peaked')]
t_min = 0.9
t_max = 1.2
abscissa = 'r/a'

for case in cases:
    shot = case.shot
    c = case.color
    l = case.label
    
    print(l)
    # Get data:
    print("Getting data...")
    
    efit_tree = eqtools.CModEFITTree(shot)
    p = profiletools.TeCTS(
        shot, t_min=t_min, t_max=t_max, abscissa=abscissa, efit_tree=efit_tree,
        remove_edge=True
    )
    case.p = p
    p.time_average(weighted=True)
    p2 = profiletools.TeETS(
        shot, t_min=t_min, t_max=t_max, abscissa=abscissa, efit_tree=efit_tree,
        remove_edge=True
    )
    p2.time_average(weighted=True)
    p.add_profile(p2)
    p3 = profiletools.TeGPC(
        shot, t_min=t_min, t_max=t_max, abscissa=abscissa, efit_tree=efit_tree,
        remove_edge=True
    )
    p3.time_average(weighted=False)
    p.add_profile(p3)
    p4 = profiletools.TeGPC2(
        shot, t_min=t_min, t_max=t_max, abscissa=abscissa, efit_tree=efit_tree,
        remove_edge=True
    )
    p4.time_average(weighted=False)
    p.add_profile(p4)
    
    # Create GP:
    p.create_gp(constrain_at_limiter=False, k='SE')
    p.gp.k.hyperprior.bounds = [(0, 50.0), (0, 25.0)]
    # Find MAP estimate for reference:
    p.find_gp_MAP_estimate(verbose=True)
    case.MAP_params = p.gp.free_params[:]
    # Marginalize with MCMC:
    print("Running MCMC sampler...")
    case.sampler = p.gp.sample_hyperparameter_posterior(nsamp=600)
    case.sampler.pool.close()
    case.sampler.pool = None
    
    print("Evaluating profiles with MCMC samples...")
    case.out_MCMC = p.compute_a_over_L(
        roa_star, use_MCMC=True, return_prediction=True, sampler=case.sampler,
        burn=200, thin=200, full_MC=False, compute_2=True
    )
    # Remove the bad point at the origin:
    case.out_MCMC['mean_a_L_grad'][0] = case.out_MCMC['mean_a_L_grad'][1]
    case.out_MCMC['std_a_L_grad'][0] = case.out_MCMC['std_a_L_grad'][1]

# Make a Bayesian argument about possible differences:
c_grid = scipy.atleast_2d(scipy.linspace(0.0, 1.0, 100)).T

# For mean_2 directly:
mean_diff_2 = cases[0].out_MCMC['mean_2'] - cases[1].out_MCMC['mean_2']
std_diff_2 = scipy.sqrt(cases[0].out_MCMC['std_2']**2.0 + cases[1].out_MCMC['std_2']**2.0)
p_gtr_c_2 = 1 - scipy.stats.norm.cdf(c_grid, loc=mean_diff_2, scale=std_diff_2)
p_lt_nc_2 = scipy.stats.norm.cdf(-c_grid, loc=mean_diff_2, scale=std_diff_2)

# For a_L_grad:
mean_diff_a_L_grad = cases[0].out_MCMC['mean_a_L_grad'] - cases[1].out_MCMC['mean_a_L_grad']
std_diff_a_L_grad = scipy.sqrt(cases[0].out_MCMC['std_a_L_grad']**2.0 + cases[1].out_MCMC['std_a_L_grad']**2.0)
p_gtr_c_a_L_grad = 1 - scipy.stats.norm.cdf(c_grid, loc=mean_diff_a_L_grad, scale=std_diff_a_L_grad)
p_lt_nc_a_L_grad = scipy.stats.norm.cdf(-c_grid, loc=mean_diff_a_L_grad, scale=std_diff_a_L_grad)

# For a2_2:
mean_diff_a2_2 = cases[0].out_MCMC['mean_a2_2'] - cases[1].out_MCMC['mean_a2_2']
std_diff_a2_2 = scipy.sqrt(cases[0].out_MCMC['std_a2_2']**2.0 + cases[1].out_MCMC['std_a2_2']**2.0)
p_gtr_c_a2_2 = 1 - scipy.stats.norm.cdf(c_grid, loc=mean_diff_a2_2, scale=std_diff_a2_2)
p_lt_nc_a2_2 = scipy.stats.norm.cdf(-c_grid, loc=mean_diff_a2_2, scale=std_diff_a2_2)

# Save the outputs for later re-generation of figures:
with open("Te_rotation.pkl", "wb") as pf:
    pkl.dump(cases, pf, protocol=pkl.HIGHEST_PROTOCOL)

print("Plotting results...")
import setupplots
setupplots.thesis_format()
import matplotlib.pyplot as plt
plt.ion()
import matplotlib.gridspec as mplgs

# Plot hypothesis test:
f = plt.figure(figsize=(0.5 * setupplots.TEXTWIDTH, 3 * 0.5 * setupplots.TEXTWIDTH / 1.618))

a = f.add_subplot(3, 1, 1)
a.plot(c_grid, p_gtr_c_2[:, 200] + p_lt_nc_2[:, 200], 'b-', label=r'$|\Delta| > c$')
a.plot(c_grid, p_gtr_c_2[:, 200], 'g--', label=r'$\Delta > c$')
a.plot(c_grid, p_lt_nc_2[:, 200], 'r:', label=r'$\Delta < -c$')
plt.setp(a.get_xticklabels(), visible=False)
a.set_ylabel(r"$\mathbb{P}$")
a.set_title(r"(d) $\Delta\mathrm{d}^2T_{\mathrm{e}}/\mathrm{d}(r/a)^2$")

a_a_L_grad = f.add_subplot(3, 1, 2, sharex=a, sharey=a)
a_a_L_grad.plot(c_grid, p_gtr_c_a_L_grad[:, 200] + p_lt_nc_a_L_grad[:, 200], 'b-', label=r'$|\Delta| > c$')
a_a_L_grad.plot(c_grid, p_gtr_c_a_L_grad[:, 200], 'g--', label=r'$\Delta > c$')
a_a_L_grad.plot(c_grid, p_lt_nc_a_L_grad[:, 200], 'r:', label=r'$\Delta < -c$')
plt.setp(a_a_L_grad.get_xticklabels(), visible=False)
a_a_L_grad.set_title(r"(e) $\Delta a/L_{\nabla T_{\mathrm{e}}}$")
a_a_L_grad.set_ylabel(r"$\mathbb{P}$")

a_a2_2 = f.add_subplot(3, 1, 3, sharex=a, sharey=a)
a_a2_2.plot(c_grid, p_gtr_c_a2_2[:, 200] + p_lt_nc_a2_2[:, 200], 'b-', label=r'$|\Delta| > c$')
a_a2_2.plot(c_grid, p_gtr_c_a2_2[:, 200], 'g--', label=r'$\Delta > c$')
a_a2_2.plot(c_grid, p_lt_nc_a2_2[:, 200], 'r:', label=r'$\Delta < -c$')
a_a2_2.set_xlabel("$c$")
a_a2_2.set_title(r"(f) $\Delta \mathrm{d}^2T_{\mathrm{e}}/\mathrm{d}(r/a)^2/T_{\mathrm{e}}$")
a_a2_2.set_ylabel(r"$\mathbb{P}$")

a.set_ylim(0, 1)
f.subplots_adjust(hspace=0.25)
setupplots.apply_formatter(f)
f.savefig("prof_Te2_difference.pdf", bbox_inches='tight')
f.savefig("prof_Te2_difference.pgf", bbox_inches='tight')

# Set up plot:
lwd = setupplots.lw
ms = setupplots.ms

f = plt.figure(figsize=[setupplots.TEXTWIDTH, 3.0 / 2.0 * setupplots.TEXTWIDTH / 1.618])
gs = mplgs.GridSpec(3, 2)
a_val = f.add_subplot(gs[0, 0])
a_grad = f.add_subplot(gs[1, 0], sharex=a_val)
a_a_L = f.add_subplot(gs[2, 0], sharex=a_val)
a_2 = f.add_subplot(gs[0, 1], sharex=a_val)
a_a_L_grad = f.add_subplot(gs[1, 1], sharex=a_val)
a_a2_2 = f.add_subplot(gs[2, 1], sharex=a_val)

# Set plot bounds:
val_bounds = [0, 3.5]
grad_bounds = [-5, 0]
a_L_bounds = [0, 10]
grad_2_bounds = [-20, 15]
a_L_grad_bounds = [-4, 10]
a2_2_bounds = [-10, 15]

for i, case in enumerate(cases):
    shot = case.shot
    c = case.color
    l = case.label
    
    # Make the sampler plot:
    gptools.plot_sampler(
        case.sampler,
        labels=[r'$\sigma_f$ [keV]', r'$\ell$'],
        burn=200,
        suptitle=r'$T_{\text{e}}$, %s' % (l,),
        bottom_sep=0.13,
        chain_ytick_pad=0.8,
        suptitle_space=0.1,
        label_fontsize=11,
        fixed_width=0.5 * setupplots.TEXTWIDTH,
        cmap='plasma', chain_ticklabel_fontsize=11, hide_chain_ylabels=True,
        ticklabel_fontsize=11
    )
    setupplots.apply_formatter(plt.gcf())
    plt.savefig("%s-Te-marginals.pdf" % (l,), bbox_inches='tight')
    plt.savefig("%s-Te-marginals.pgf" % (l,), bbox_inches='tight')
    
    # Plot value:
    gptools.univariate_envelope_plot(
        roa_star, case.out_MCMC['mean_val'], case.out_MCMC['std_val'], ax=a_val,
        linewidth=lwd, linestyle='-' if i == 0 else '--', color=c, label=l, lb=val_bounds[0], ub=val_bounds[1]
    )
    case.p.plot_data(markersize=ms, fmt=c+'o' if i == 0 else '^', label='_nolegend_', ax=a_val, label_axes=False)
    
    # Plot gradient:
    gptools.univariate_envelope_plot(
        roa_star, case.out_MCMC['mean_grad'], case.out_MCMC['std_grad'], ax=a_grad,
        linewidth=lwd, linestyle='-' if i == 0 else '--', color=c, lb=grad_bounds[0], ub=grad_bounds[1]
    )
    
    # Plot a/L:
    gptools.univariate_envelope_plot(
        roa_star[core_mask], case.out_MCMC['mean_a_L'][core_mask],
        case.out_MCMC['std_a_L'][core_mask], ax=a_a_L, linewidth=lwd, linestyle='-' if i == 0 else '--',
        color=c, lb=a_L_bounds[0], ub=a_L_bounds[1]
    )
    
    # Plot 2nd deriv:
    gptools.univariate_envelope_plot(
        roa_star[core_mask], case.out_MCMC['mean_2'][core_mask],
        case.out_MCMC['std_2'][core_mask], ax=a_2, linewidth=lwd, linestyle='-' if i == 0 else '--',
        color=c, lb=grad_2_bounds[0], ub=grad_2_bounds[1]
    )
    
    # Plot 2nd deriv scale length:
    gptools.univariate_envelope_plot(
        roa_star[core_mask], case.out_MCMC['mean_a_L_grad'][core_mask],
        case.out_MCMC['std_a_L_grad'][core_mask], ax=a_a_L_grad, linewidth=lwd,
        linestyle='-' if i == 0 else '--', color=c, lb=a_L_grad_bounds[0], ub=a_L_grad_bounds[1]
    )
    
    # Plot normalized 2nd deriv:
    gptools.univariate_envelope_plot(
        roa_star[core_mask], case.out_MCMC['mean_a2_2'][core_mask],
        case.out_MCMC['std_a2_2'][core_mask], ax=a_a2_2, linewidth=lwd, linestyle='-' if i == 0 else '--',
        color=c, lb=a2_2_bounds[0], ub=a2_2_bounds[1]
    )

# Format plots:
a_val.set_xlim([0, 1.0])
a_val.set_ylim(val_bounds)
a_grad.set_ylim(grad_bounds)
a_a_L.set_ylim(a_L_bounds)
a_2.set_ylim(grad_2_bounds)
a_a_L_grad.set_ylim(a_L_grad_bounds)
a_a2_2.set_ylim(a2_2_bounds)

a_val.legend(loc='upper right')

a_val.set_title(r'(a) $T_{\text{e}}$ [keV]')
a_grad.set_title(r'(b) $\mathrm{d}T_{\text{e}}/\mathrm{d}(r/a)$ [keV]')
a_a_L.set_title(r'(c) $a/L_{T_{\text{e}}}$')
a_2.set_title(r'(d) $\mathrm{d}^2T_{\text{e}}/\mathrm{d}(r/a)^2$ [keV]')
a_a_L_grad.set_title(r'(e) $a/L_{\nabla T_{\text{e}}}$')
a_a2_2.set_title(r'(f) $\mathrm{d}^2T_{\text{e}}/\mathrm{d}(r/a)^2/T_{\text{e}}$')

a_a2_2.set_xlabel('$r/a$')
a_a_L.set_xlabel('$r/a$')

plt.setp(a_val.get_xticklabels(), visible=False)
plt.setp(a_grad.get_xticklabels(), visible=False)
plt.setp(a_2.get_xticklabels(), visible=False)
plt.setp(a_a_L_grad.get_xticklabels(), visible=False)

f.suptitle(r"$T_{\text{e}}$ profile, \textsc{se} covariance kernel")
f.subplots_adjust(left=0.07, bottom=0.09, right=0.97, top=0.9, wspace=0.17, hspace=0.26)
setupplots.apply_formatter(f)
f.canvas.draw()
f.savefig("prof2Te.pdf")
f.savefig("prof2Te.pgf")

# Make posterior summary table:
post_sum = setupplots.generate_post_sum(
    [c.MAP_params for c in cases],
    [c.sampler for c in cases],
    [200,] * len(cases),
    [[[r'$T_{\text{e}}$', c.label, r'$\sigma_f$ [keV]'], ['', '', r'$\ell$']] for c in cases]
)
with open('TeRotPostSum.tex', 'w') as tf:
    tf.write(post_sum)
