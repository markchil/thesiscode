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

# This script makes figure 3.44, which shows the uncertainties in the rate
# coefficients.

from __future__ import division

import os
cdir = os.getcwd()
os.chdir('/Users/markchilenski/src/bayesimp')

import bayesimp
import gptools
import scipy
import sys
sys.path.insert(0, '/Users/markchilenski/src/bayesimp/xxdata_11')
import adf11
sys.path.insert(0, '/Users/markchilenski/src/bayesimp/xxdata_15')
import adf15

# Load the run so we have ne, Te:
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
    time_spec=(  # For dense sampling of cs_den properties:
        "    {time_1:.5f}     0.000010               1.00                      1\n"
        "    {time_2:.5f}     0.000010               1.00                      1\n"
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

# Load the ADF files:
recombination = adf11.adf11('/Users/markchilenski/src/bayesimp/strahl/atomdat/newdat/acd85_ca.dat', 1)
ionization = adf11.adf11('/Users/markchilenski/src/bayesimp/strahl/atomdat/newdat/scd85_ca.dat', 2)
pec = adf15.adf15('/Users/markchilenski/src/bayesimp/strahl/atomdat/adf15/ca/pue#ca18.dat_mod')

X = scipy.asarray(r.run_data.ne_X, dtype=float)
mask = (X <= 1.0)
X = X[mask]

ne_samps = r.run_data.ne_p.gp.draw_sample(
    r.run_data.ne_X,
    mean=r.run_data.ne_res['mean_val'],
    cov=r.run_data.ne_res['cov'],
    num_samp=1000
)[mask, :] * 1e20 / 1e6
Te_samps = r.run_data.Te_p.gp.draw_sample(
    r.run_data.Te_X,
    mean=r.run_data.Te_res['mean_val'],
    cov=r.run_data.Te_res['cov'],
    num_samp=1000
)[mask, :] * 1e3

mu_alpha = scipy.zeros((recombination.num_blocks, len(X)))
std_alpha = scipy.zeros((recombination.num_blocks, len(X)))
mu_S = scipy.zeros((ionization.num_blocks, len(X)))
std_S = scipy.zeros((ionization.num_blocks, len(X)))
for i in range(0, recombination.num_blocks):
    alpha = 10.0**(
        scipy.interpolate.RectBivariateSpline(
            scipy.log10(recombination.ne),
            scipy.log10(recombination.Te),
            scipy.log10(recombination.rate_coeffs[i]).T,
            s=0
        )(scipy.log10(ne_samps), scipy.log10(Te_samps), grid=False)
    )
    mu_alpha[i, :] = alpha.mean(axis=1)
    std_alpha[i, :] = alpha.std(axis=1, ddof=1)
    S = 10.0**(
        scipy.interpolate.RectBivariateSpline(
            scipy.log10(ionization.ne),
            scipy.log10(ionization.Te),
            scipy.log10(ionization.rate_coeffs[i]).T,
            s=0
        )(scipy.log10(ne_samps), scipy.log10(Te_samps), grid=False)
    )
    mu_S[i, :] = S.mean(axis=1)
    std_S[i, :] = S.std(axis=1, ddof=1)
pec_i = scipy.interpolate.RectBivariateSpline(
    scipy.log10(pec.ne[:pec.num_dens[0], 0]),
    scipy.log10(pec.Te[:pec.num_temp[0], 0]),
    pec.pec[:pec.num_temp[0], :pec.num_dens[0], 0].T,
    s=0
)(scipy.log10(ne_samps), scipy.log10(Te_samps), grid=False)
mu_pec = pec_i.mean(axis=1)
std_pec = pec_i.std(axis=1, ddof=1)

# Make plots:
os.chdir(cdir)

import setupplots
setupplots.thesis_format()
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.cm import plasma
plt.ion()

f = plt.figure(figsize=(0.75 * setupplots.TEXTWIDTH, 2.75 * 0.75 * setupplots.TEXTWIDTH / 1.618))
gs = GridSpec(3, 2, width_ratios=[8, 1])
a_alpha = f.add_subplot(gs[0, 0])
a_S = f.add_subplot(gs[1, 0])
a_pec = f.add_subplot(gs[2, 0])

for i in range(0, recombination.num_blocks):
    gptools.univariate_envelope_plot(
        X,
        mu_alpha[i, :] / 1e6,
        std_alpha[i, :] / 1e6,
        ax=a_alpha,
        color=plasma(i / (recombination.num_blocks - 1)),
        lb=mu_alpha.min() / 10.0e6
    )
    gptools.univariate_envelope_plot(
        X,
        mu_S[i, :] / 1e6,
        std_S[i, :] / 1e6,
        ax=a_S,
        color=plasma(i / (recombination.num_blocks - 1)),
        lb=mu_S.min() / 10.0e6
    )
gptools.univariate_envelope_plot(
    X,
    mu_pec / 1e6 / 1e-18,
    std_pec / 1e6 / 1e-18,
    ax=a_pec,
    color='b',
    lb=mu_pec.min() / 10.0e6
)
a_S.set_ylim(bottom=1e-21)
a_pec.set_ylim(bottom=0)
a_cb = f.add_subplot(gs[0:2, 1])
norm = matplotlib.colors.Normalize(vmin=1, vmax=recombination.num_blocks)
cb1 = matplotlib.colorbar.ColorbarBase(a_cb, cmap='plasma', norm=norm)
cb1.set_label(r'$i_1$')
a_alpha.set_ylabel(r'$\alpha_{i_1 \to i_0}$ [$\si{m^3/s}$]')
a_S.set_ylabel(r'$S_{i_0 \to i_1}$ [$\si{m^3/s}$]')
a_pec.set_ylabel(r"$P_{ij}$ [$\SI{e-18}{m^3/s}$]")
plt.setp(a_alpha.get_xticklabels(), visible=False)
plt.setp(a_S.get_xticklabels(), visible=False)
a_pec.set_xlabel('$r/a$')
a_alpha.set_yscale('log')
a_S.set_yscale('log')
# a_pec.set_yscale('log')
f.suptitle("Rate coefficients")
a_alpha.set_title("Recombination")
a_S.set_title("Ionization")
a_pec.set_title("Photon emission")
f.subplots_adjust(left=0.12, bottom=0.06, right=0.9, top=0.93, hspace=0.15)
setupplots.apply_formatter(f)
f.savefig("rateCoeffs.pdf", bbox_inches='tight')
f.savefig("rateCoeffs.pgf", bbox_inches='tight')
