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

# This script makes figures C.1 and C.2, which show the various components of
# the flux and density change terms of the impurity transport equation.

from __future__ import division

import os
cdir = os.getcwd()
os.chdir('/Users/markchilenski/src/bayesimp')

import bayesimp
import gptools
import scipy
import scipy.io
import sys
sys.path.insert(0, '/Users/markchilenski/src/bayesimp/xxdata_11')
import adf11

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

r.DV2cs_den(r.params_true)

f = scipy.io.netcdf_file('result/strahl_result.dat', 'r')

# Load the STRAHL outputs:
r_vol = f.variables['radius_grid'][:]
D = f.variables['anomal_diffusion'][:]
V = f.variables['anomal_drift'][:]
nZi = f.variables['impurity_density'][:]
n_18 = nZi[:, 18, :]
n_17 = nZi[:, 17, :]
n_19 = nZi[:, 19, :]
n_tot = nZi[:, 1:, :].sum(axis=1)
ne = f.variables['electron_density'][:]
Te = f.variables['electron_temperature'][:]
t = f.variables['time'][:]
sqrtpsinorm = f.variables['rho_poloidal_grid'][:]
roa = r.efit_tree.psinorm2roa(sqrtpsinorm**2.0, (r.time_1 + r.time_2) / 2.0)
f.close()

# Get the S_{Z,i}, alpha_{Z,i}, S_{Z,i-1}, alpha_{Z,i+1} factors from the
# netCDF files:
recombination = adf11.adf11('/Users/markchilenski/src/bayesimp/strahl/atomdat/newdat/acd85_ca.dat', 1)
ionization = adf11.adf11('/Users/markchilenski/src/bayesimp/strahl/atomdat/newdat/scd85_ca.dat', 2)

alpha_18_to_17 = 10.0**(
    scipy.interpolate.RectBivariateSpline(
        scipy.log10(recombination.ne),
        scipy.log10(recombination.Te),
        scipy.log10(recombination.rate_coeffs[17]).T,
        s=0
    )(scipy.log10(ne[0, :]), scipy.log10(Te[0, :]), grid=False)
)
alpha_19_to_18 = 10.0**(
    scipy.interpolate.RectBivariateSpline(
        scipy.log10(recombination.ne),
        scipy.log10(recombination.Te),
        scipy.log10(recombination.rate_coeffs[18]).T,
        s=0
    )(scipy.log10(ne[0, :]), scipy.log10(Te[0, :]), grid=False)
)

S_18_to_19 = 10.0**(
    scipy.interpolate.RectBivariateSpline(
        scipy.log10(ionization.ne),
        scipy.log10(ionization.Te),
        scipy.log10(ionization.rate_coeffs[18]).T,
        s=0
    )(scipy.log10(ne[0, :]), scipy.log10(Te[0, :]), grid=False)
)

S_17_to_18 = 10.0**(
    scipy.interpolate.RectBivariateSpline(
        scipy.log10(ionization.ne),
        scipy.log10(ionization.Te),
        scipy.log10(ionization.rate_coeffs[17]).T,
        s=0
    )(scipy.log10(ne[0, :]), scipy.log10(Te[0, :]), grid=False)
)

# Find the transport fluxes:
dn_18_drvol = scipy.interpolate.RectBivariateSpline(t, r_vol, n_18, s=0)(t, r_vol, dy=1)
diffusive_flux = -D * dn_18_drvol
convective_flux = V * n_18
transport_contribution = -1.0 / r_vol * scipy.interpolate.RectBivariateSpline(
    t, r_vol, r_vol * (diffusive_flux + convective_flux)
)(t, r_vol, dy=1)
transport_contribution[scipy.isinf(transport_contribution)] = 0.0

# For total ion density:
dn_tot_drvol = scipy.interpolate.RectBivariateSpline(t, r_vol, n_tot, s=0)(t, r_vol, dy=1)
diffusive_flux_tot = -D * dn_tot_drvol
convective_flux_tot = V * n_tot


# Find the atomic physics contribution:
atomic_physics_contribution = (
    -ne * n_18 * S_18_to_19 - ne * n_18 * alpha_18_to_17 + ne * n_17 * S_17_to_18 + ne * n_19 * alpha_19_to_18
)

os.chdir(cdir)
import setupplots
setupplots.thesis_format()
import matplotlib.pyplot as plt
plt.ion()

t_plot = setupplots.make_pcolor_grid(t - r.time_1)
roa_plot = setupplots.make_pcolor_grid(roa)

# Plot the flux components:
f = plt.figure(figsize=(0.75 * setupplots.TEXTWIDTH, 3 * 0.75 * setupplots.TEXTWIDTH / 1.618))

# Plot the diffusive flux:
a_D = f.add_subplot(3, 1, 1)
pcm = a_D.pcolormesh(
    roa_plot,
    t_plot[::100],
    diffusive_flux[::100, :] * 1e4 / 1e12,
    cmap='seismic',
    vmin=-scipy.absolute(diffusive_flux * 1e4 / 1e12).max(),
    vmax=scipy.absolute(diffusive_flux * 1e4 / 1e12).max()
)
cb = f.colorbar(pcm)
cb.set_label(r"$-D\partial n/\partial r_{\mathrm{vol}}$ [$\SI{e12}{m^{-2}.s^{-1}}$]")
a_D.set_ylabel(r"$t-t_{\mathrm{inj}}$ [s]")
a_D.set_title("Diffusive flux")
plt.setp(a_D.get_xticklabels(), visible=False)

# Plot the convective flux:
a_V = f.add_subplot(3, 1, 2, sharex=a_D, sharey=a_D)
pcm = a_V.pcolormesh(
    roa_plot,
    t_plot[::100],
    convective_flux[::100, :] * 1e4 / 1e12,
    cmap='seismic',
    vmin=-scipy.absolute(diffusive_flux * 1e4 / 1e12).max(),
    vmax=scipy.absolute(diffusive_flux * 1e4 / 1e12).max()
)
cb = f.colorbar(pcm)
cb.set_label(r"$Vn$ [$\SI{e12}{m^{-2}.s^{-1}}$]")
a_V.set_ylabel(r"$t - t_{\mathrm{inj}}$ [s]")
a_V.set_title("Convective flux")
plt.setp(a_V.get_xticklabels(), visible=False)

a_D.set_xlim([0.0, roa.max()])
a_D.set_ylim(t[0] - r.time_1, t[-1] - r.time_1)

# Plot the total flux:
a_G = f.add_subplot(3, 1, 3, sharex=a_D, sharey=a_D)
pcm = a_G.pcolormesh(
    roa_plot,
    t_plot[::100],
    (diffusive_flux[::100, :] + convective_flux[::100, :]) * 1e4 / 1e12,
    cmap='seismic',
    vmin=-scipy.absolute(diffusive_flux * 1e4 / 1e12).max(),
    vmax=scipy.absolute(diffusive_flux * 1e4 / 1e12).max()
)
cb = f.colorbar(pcm)
cb.set_label(r"$\Gamma$ [$\SI{e12}{m^{-2}.s^{-1}}$]")
a_G.set_xlabel("$r/a$")
a_G.set_ylabel(r"$t - t_{\mathrm{inj}}$ [s]")
a_G.set_title("Total flux")

a_D.set_xlim([0.0, roa.max()])
a_D.set_ylim(t[0] - r.time_1, t[-1] - r.time_1)

f.suptitle("Components of Ca$^{18+}$ flux")
f.subplots_adjust(top=0.93)
setupplots.apply_formatter(f)
f.canvas.draw()

f.savefig("fluxComponents.pdf", bbox_inches='tight')
f.savefig("fluxComponents.pgf", bbox_inches='tight')

# Plot the contributions to dn/dt:
f = plt.figure(figsize=(0.75 * setupplots.TEXTWIDTH, 3 * 0.75 * setupplots.TEXTWIDTH / 1.618))

# Plot the transport term:
a_t = f.add_subplot(3, 1, 1)
pcm = a_t.pcolormesh(
    roa_plot,
    t_plot[::100],
    transport_contribution[::100, :] * 1e6 / 1e14,
    cmap='seismic',
    vmin=-scipy.absolute(atomic_physics_contribution * 1e6 / 1e14).max(),
    vmax=scipy.absolute(atomic_physics_contribution * 1e6 / 1e14).max()
)
cb = f.colorbar(pcm)
cb.set_label(r"$-\nabla\cdot\Gamma$ [$\SI{e14}{m^{-3}.s^{-1}}$]")
a_t.set_ylabel(r"$t - t_{\mathrm{inj}}$ [s]")
a_t.set_title("Transport contribution")
plt.setp(a_t.get_xticklabels(), visible=False)

# Plot the atomic physics term:
a_a = f.add_subplot(3, 1, 2, sharex=a_t, sharey=a_t)
pcm = a_a.pcolormesh(
    roa_plot,
    t_plot[::100],
    atomic_physics_contribution[::100, :] * 1e6 / 1e14,
    cmap='seismic',
    vmin=-scipy.absolute(atomic_physics_contribution * 1e6 / 1e14).max(),
    vmax=scipy.absolute(atomic_physics_contribution * 1e6 / 1e14).max()
)
cb = f.colorbar(pcm)
cb.set_label(r"$Q$ [$\SI{e14}{m^{-3}.s^{-1}}$]")
a_a.set_ylabel(r"$t - t_{\mathrm{inj}}$ [s]")
a_a.set_title("Atomic physics contribution")
plt.setp(a_a.get_xticklabels(), visible=False)

# Plot the total dn/dt:
a_tr = f.add_subplot(3, 1, 3, sharex=a_t, sharey=a_t)
pcm = a_tr.pcolormesh(
    roa_plot,
    t_plot[::100],
    (transport_contribution[::100, :] + atomic_physics_contribution[::100, :]) * 1e6 / 1e14,
    cmap='seismic',
    vmin=-scipy.absolute(atomic_physics_contribution * 1e6 / 1e14).max(),
    vmax=scipy.absolute(atomic_physics_contribution * 1e6 / 1e14).max()
)
cb = f.colorbar(pcm)
cb.set_label(r"$\partial n/\partial t$ [$\SI{e14}{m^{-3}.s^{-1}}$]")
a_tr.set_ylabel(r"$t - t_{\mathrm{inj}}$ [s]")
a_tr.set_title(r"Total $\partial n/\partial t$")
a_tr.set_xlabel(r"$r/a$")

a_t.set_xlim([0.0, roa.max()])
a_t.set_ylim(t[0] - r.time_1, t[-1] - r.time_1)

f.suptitle(r"Contributions to Ca$^{18+}$ density change")
f.subplots_adjust(top=0.93)
setupplots.apply_formatter(f)
f.savefig("dndtComponents.pdf", bbox_inches='tight')
f.savefig("dndtComponents.pgf", bbox_inches='tight')
