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

# This script makes figures 3.31 and 3.32, which show the result of inferring
# basic D and V profiles from local measurements.

from __future__ import division

import os
cdir = os.getcwd()
os.chdir('/Users/markchilenski/src/bayesimp')

import bayesimp
import gptools
import scipy
import pymultinest

r = bayesimp.Run(
    shot=1101014006,
    version=3,
    time_1=1.165,
    time_2=1.265,
    Te_args=['--system', 'TS', 'GPC', 'GPC2'],
    ne_args=['--system', 'TS'],
    debug_plots=1,
    num_eig_D=1,
    num_eig_V=1,
    method='linterp',
    free_knots=False,
    use_scaling=False,
    use_shift=False,
    include_loweus=True,
    source_file='/Users/markchilenski/src/bayesimp/Caflx_delta_1165.dat',
    sort_knots=True,
    params_true=[1.0, -10.0, 0.0, 0.0, 0.0, 0, 0, 0],
    time_spec=(  # For fast sampling:
        "    1.165     0.000050               1.01                      1\n"
        "    1.167     0.000075               1.1                       10\n"
        "    1.175     0.000100               1.50                      25\n"
        "    1.265     0.000100               1.50                      25\n"
    ),
    D_lb=0.0,
    D_ub=10.0,
    V_lb=-100.0,
    V_ub=10.0,
    V_lb_outer=-100.0,
    V_ub_outer=10.0,
    num_eig_ne=3,
    num_eig_Te=3,
    free_ne=False,
    free_Te=False,
    normalize=False,
    local_time_res=6e-3,
    num_local_space=5,
    local_synth_noise=5e-2,
    use_line_integral=False,
    use_local=True
)

basename = os.path.abspath('../chains_%d_%d/c-' % (r.shot, r.version))
a = pymultinest.Analyzer(
    n_params=~(r.fixed_params).sum(),
    outputfiles_basename=basename
)
data = a.get_data()

os.chdir(cdir)

# Restore the std_D, std_V data:
import shelve
shelf_name = 'outputs/bayes_time_res.shelf'
shelf = shelve.open(shelf_name)
globals()['std_D'] = shelf['std_D']
globals()['std_V'] = shelf['std_V']
globals()['cov_DV'] = shelf['cov_DV']
globals()['dt_grid'] = shelf['dt_grid']
globals()['noise_grid'] = shelf['noise_grid']
globals()['npts_grid'] = shelf['npts_grid']
shelf.close()

# Estimate ellipse:
std_D = scipy.interpolate.RectBivariateSpline(dt_grid, noise_grid, std_D[:, :, 2])(6e-3, 5e-2)[0, 0]
std_V = scipy.interpolate.RectBivariateSpline(dt_grid, noise_grid, std_V[:, :, 2])(6e-3, 5e-2)[0, 0]
cov_DV = scipy.interpolate.RectBivariateSpline(dt_grid, noise_grid, cov_DV[:, :, 2])(6e-3, 5e-2)[0, 0]
Sigma_DV = scipy.asarray([[std_D**2.0, cov_DV], [cov_DV, std_V**2.0]], dtype=float)

# Summary statistics from sampler:
sampler = data[:, 2:]
weights = data[:, 0]
k = sampler.shape[-1]
flat_trace = sampler.reshape((-1, k))
weights = weights.ravel()
cov = scipy.cov(flat_trace, rowvar=0, aweights=weights)

# Make plots:
import setupplots
setupplots.thesis_format()
import matplotlib.pyplot as plt
plt.ion()

mean, ci_l, ci_u = gptools.summarize_sampler(data[:, 2:], weights=data[:, 0])

f = gptools.plot_sampler(
    data[:, 2:],
    weights=data[:, 0],
    labels=[r'$D$ [$\si{m^2/s}$]', '$V$ [m/s]'],
    chain_alpha=1.0,
    cutoff_weight=0.01,
    cmap='plasma',
    fixed_width=0.9 * setupplots.TEXTWIDTH,
    suptitle='Posterior distribution of $D$ and $V$\nlocal measurements',
    points=scipy.asarray([mean, r.params_true[:2]]),
    covs=[None, Sigma_DV],
    bottom_sep=0.12,
    suptitle_space=0.07,
    ax_space=0.175,
    label_fontsize=11,
    chain_ticklabel_fontsize=11,
    ticklabel_fontsize=11
)
setupplots.apply_formatter(f)
f.savefig("samplerDVBasic.pdf", bbox_inches='tight')
f.savefig("samplerDVBasic.pgf", bbox_inches='tight')

# Marginalized D, V:
D_mean, D_std, V_mean, V_std = r.compute_marginalized_DV(
    data[:, 2:],
    weights=data[:, 0],
    cutoff_weight=0.01
)
D, V = r.eval_DV(r.params_true)
f = plt.figure(figsize=(0.75 * setupplots.TEXTWIDTH, 2 * 0.75 * setupplots.TEXTWIDTH / 1.618))
a_D = f.add_subplot(2, 1, 1)
a_D.plot(r.roa_grid_DV, D, label='true')
gptools.univariate_envelope_plot(r.roa_grid_DV, D_mean, D_std, ax=a_D, label='inferred', linestyle='--')
a_V = f.add_subplot(2, 1, 2, sharex=a_D)
a_V.plot(r.roa_grid_DV, V, label='true')
gptools.univariate_envelope_plot(r.roa_grid_DV, V_mean, V_std, ax=a_V, label='inferred', linestyle='--')
a_V.legend(loc='lower left')
a_D.set_ylabel(r"$D$ [$\si{m^2/s}$]")
a_V.set_ylabel(r"$V$ [m/s]")
a_V.set_xlabel('$r/a$')
a_D.set_title(r"\textcolor{MPLb}{True} and \textcolor{MPLg}{inferred} transport coefficient profiles" + "\nlocal measurements")
plt.setp(a_D.get_xticklabels(), visible=False)
a_D.set_xlim(r.roa_grid_DV[0], r.roa_grid_DV[-1])
a_D.set_ylim(0, 2)
f.subplots_adjust(hspace=0.1)
setupplots.apply_formatter(f)
f.savefig("DVBasic.pdf", bbox_inches='tight')
f.savefig("DVBasic.pgf", bbox_inches='tight')

post_sum = r"""Basic & $D$ [$\si{{m^2/s}}$] & {muD:.8f} & [ & {cilD:.8f} & {ciuD:.8f} & ]\\
& $V$ [m/s] & {muV:.8f} & [ & {cilV:.8f} & {ciuV:.8f} & ]\\
""".format(
    muD=mean[0],
    cilD=ci_l[0],
    ciuD=ci_u[0],
    muV=mean[1],
    cilV=ci_l[1],
    ciuV=ci_u[1]
)
with open('samplerDVBasicPostSum.tex', 'w') as f:
    f.write(post_sum)

lincomp = r"""Linearized & {stdDLin:.8f} & {stdVLin:.8f} & {covDVLin:.8f} & {corrDVLin:.8f}\\
\textsc{{MultiNest}} & {stdDFull:.8f} & {stdVFull:.8f} & {covDVFull:.8f} & {corrDVFull:.8f}\\
""".format(
    stdDLin=std_D,
    stdVLin=std_V,
    covDVLin=cov_DV,
    corrDVLin=cov_DV / (std_D * std_V),
    stdDFull=scipy.sqrt(cov[0, 0]),
    stdVFull=scipy.sqrt(cov[1, 1]),
    covDVFull=cov[0, 1],
    corrDVFull=cov[0, 1] / (scipy.sqrt(cov[0, 0]) * scipy.sqrt(cov[1, 1]))
)
with open('linCompBasic.tex', 'w') as f:
    f.write(lincomp)
