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

# This script makes figures 3.12, 3.13, 3.14, 3.15, 3.16, 3.17, 3.18, 3.19,
# 3.20, 3.21, 3.22, 3.23, 3.24, 3.25, 3.26, C.3, C.4 and C.5, which illustrate
# the linearized Bayesian analysis of impurity transport measurements.

from __future__ import division

import os
cdir = os.getcwd()
os.chdir('/Users/markchilenski/src/bayesimp')

import bayesimp
import gptools
import scipy
import cPickle as pkl
import multiprocessing
from emcee.interruptible_pool import InterruptiblePool as iPool
import itertools
from skimage import measure
import time as time_
import profiletools
import MDSplus

r = bayesimp.Run(
    shot=1101014006,
    version=2,
    time_1=1.165,
    time_2=1.265,
    Te_args=['--system', 'TS', 'GPC', 'GPC2'],
    ne_args=['--system', 'TS'],
    debug_plots=1,
    num_eig_D=1,
    num_eig_V=1,
    method='linterp',
    free_knots=False,
    use_scaling=True,
    include_loweus=True,
    source_file='/Users/markchilenski/src/bayesimp/Caflx_delta_1165.dat',
    sort_knots=True,
    params_true=[1.0, -10.0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0.0, 0.0, 0.0, 0, 0, 0, 0, 0, 0, 0, 0],
    time_spec=(  # For dense sampling of cs_den properties:
        "    {time_1:.5f}     0.000010               1.00                      1\n"
        "    {time_2:.5f}     0.000010               1.00                      1\n"
    )
)

D0 = r.params_true[0]
V0 = r.params_true[1]
target_precision = 0.1

# Load the dense sampling data:
D_grid = scipy.linspace(D0 - 2 * target_precision * D0, D0 + 2 * target_precision * D0, 50)
V_grid = scipy.linspace(V0 + 2 * target_precision * V0, V0 - 2 * target_precision * V0, 50)
# The following code is what is used to produce cs_den_local.pkl. It takes a while...
# pool = bayesimp.make_pool(num_proc=24)
# res = r.parallel_compute_cs_den(D_grid, V_grid, pool)
# bayesimp.finalize_pool(pool)
# with open("../cs_den_local.pkl", 'wb') as pf:
#     pkl.dump(res, pf, protocol=pkl.HIGHEST_PROTOCOL)

rr_true = bayesimp._ComputeCSDenEval(r)(r.params_true[0:2])
with open("../cs_den_local.pkl", 'rb') as pf:
    res = pkl.load(pf)
DD, VV = scipy.meshgrid(D_grid, V_grid)
VD = VV / DD
orig_shape = DD.shape

tauimp = scipy.reshape(scipy.asarray([rr.tau_N for rr in res]), orig_shape)
tp = scipy.reshape(scipy.asarray([rr.t_n_peak_local[0] for rr in res]), orig_shape) - r.time_1
b = scipy.reshape(scipy.asarray([rr.n075n0 for rr in res]), orig_shape)
profs = scipy.reshape(scipy.asarray([rr.prof for rr in res]), orig_shape + rr_true.prof.shape)

# Make a spline to extract the n(r)/n(0) value for any value of r:
roa_grid_true = r.efit_tree.psinorm2roa(r.truth_data.sqrtpsinorm**2.0, (r.time_1 + r.time_2) / 2.0)
roa_grid_true[0] = 0.0
def nrn0(roa):
    """Interpolate to find :math:`n(r)/n(0)` as a function of :math:`D`, :math:`V`.
    """
    out = scipy.zeros(orig_shape)
    for i in range(0, out.shape[0]):
        for j in range(0, out.shape[1]):
            out[i, j] = scipy.interpolate.InterpolatedUnivariateSpline(
                roa_grid_true,
                profs[i, j, :]
            )(roa)
    return out

def y2abc(y, debug_plots=False, s=0):
    """Linearize a given quantity to compute the a, b, c coefficients for the uncertainty analysis.
    """
    spl = scipy.interpolate.RectBivariateSpline(D_grid, V_grid, y.T, s=s)
    dydD = spl(D0, V0, dx=1, dy=0)[0, 0]
    dydV = spl(D0, V0, dx=0, dy=1)[0, 0]
    y0 = spl(D0, V0)[0, 0]
    aa = y0 - dydD * D0 - dydV * V0
    b = dydD
    c = dydV
    
    if debug_plots:
        y_lin = y0 + dydD * (DD - D0) + dydV * (VV - V0)
        
        f = plt.figure()
        a = f.add_subplot(111, projection='3d')
        a.plot_surface(DD, VV, spl(DD, VV, grid=False), cmap='Blues', alpha=0.5, rstride=1, cstride=1)
        a.plot_surface(DD, VV, y, cmap='Greens', alpha=0.5, rstride=1, cstride=1)
        a.plot_surface(DD, VV, y_lin, cmap='Reds', alpha=0.5, rstride=1, cstride=1)
        a.set_title("Surfaces of spline (b), actual (g), linear (r)")
        a.set_xlabel("D")
        a.set_ylabel("V")
        
        f = plt.figure()
        a = f.add_subplot(1, 1, 1)
        a.contour(D_grid, V_grid, spl(DD, VV, grid=False), 50, cmap='Blues')
        a.contour(D_grid, V_grid, y, 50, cmap='Greens')
        a.contour(D_grid, V_grid, y_lin, 50, cmap='Reds')
        a.set_title("Contours of spline (b), actual (g), linear (r)")
        a.set_xlabel("D")
        a.set_ylabel("V")
        
        f = plt.figure()
        a = f.add_subplot(1, 1, 1)
        d = (spl(DD, VV, grid=False) - y) / y
        pcm = a.pcolormesh(
            D_grid,
            V_grid,
            d,
            cmap='seismic',
            vmin=-scipy.absolute(d).max(),
            vmax=scipy.absolute(d).max()
        )
        cb = f.colorbar(pcm)
        a.set_title("Relative difference between spline and actual")
        a.set_xlabel("D")
        a.set_ylabel("V")
        
        f = plt.figure()
        a = f.add_subplot(1, 1, 1)
        d = (y_lin - y) / y
        pcm = a.pcolormesh(
            D_grid,
            V_grid,
            d,
            cmap='seismic',
            vmin=-scipy.absolute(d).max(),
            vmax=scipy.absolute(d).max()
        )
        cb = f.colorbar(pcm)
        a.set_title("Relative difference between linearization and actual")
        a.set_xlabel("D")
        a.set_ylabel("V")
        
        f = plt.figure()
        a = f.add_subplot(1, 1, 1)
        pcm = a.pcolormesh(D_grid, V_grid, spl(DD, VV, grid=False, dx=1, dy=0))
        f.colorbar(pcm)
        a.contour(D_grid, V_grid, spl(DD, VV, grid=False, dx=1, dy=0), 50, colors='w', alpha=0.5)
        a.set_title("dy/dD")
        a.set_xlabel("D")
        a.set_ylabel("V")
        
        f = plt.figure()
        a = f.add_subplot(1, 1, 1)
        pcm = a.pcolormesh(D_grid, V_grid, spl(DD, VV, grid=False, dx=0, dy=1))
        f.colorbar(pcm)
        a.contour(D_grid, V_grid, spl(DD, VV, grid=False, dx=0, dy=1), 50, colors='w', alpha=0.5)
        a.set_title("dy/dV")
        a.set_xlabel("D")
        a.set_ylabel("V")
        
    
    return aa, b, c, y0

def data2DV(a, C, y, std_y):
    """Convert the given data and parameters to estimates of D, V including uncertainties.
    
    Parameters
    ----------
    a : array of float, (`n`,)
        The :math:`a` coefficients for each measurement.
    C : array of float, (`n`, 2)
        The :math:`b` and :math:`c` coefficients for each measurement.
    y : array of float, (`n`,)
        The measurements.
    std_y : array of float, (`n`,)
        The uncertainties in the measurements.
    """
    a = scipy.asarray(a, dtype=float)
    C = scipy.asarray(C, dtype=float)
    y = scipy.asarray(y, dtype=float)
    std_y = scipy.asarray(std_y, dtype=float)
    
    sum_bb = (C[:, 0]**2.0 / std_y**2.0).sum()
    sum_cc = (C[:, 1]**2.0 / std_y**2.0).sum()
    sum_bc = (C[:, 0] * C[:, 1] / std_y**2.0).sum()
    denom = sum_bb * sum_cc - sum_bc**2.0
    
    sum_bya = (C[:, 0] * (y - a) / std_y**2.0).sum()
    sum_cya = (C[:, 1] * (y - a) / std_y**2.0).sum()
    
    std_D = scipy.sqrt(sum_cc / denom)
    std_V = scipy.sqrt(sum_bb / denom)
    cov_DV = -sum_bc / denom
    mu_D = std_D**2.0 * sum_bya + cov_DV * sum_cya
    mu_V = std_V**2.0 * sum_cya + cov_DV * sum_bya
    
    return mu_D, mu_V, std_D, std_V, cov_DV

# Compute the coefficients for the linear model:
a_tauimp, dtauimpdD, dtauimpdV, tauimp0 = y2abc(tauimp)
# tp needs to be smoothed because of the discretization error:
a_tp, dtpdD, dtpdV, tp0 = y2abc(tp, s=tp.size / 4.0)
a_b, dbdD, dbdV, b0 = y2abc(b)

aa = scipy.asarray([a_tauimp, a_tp, a_b], dtype=float)
C = scipy.asarray([[dtauimpdD, dtauimpdV], [dtpdD, dtpdV], [dbdD, dbdV]], dtype=float)

# Evaluate for typical values:
tauimp_tol = 3e-4
tp_tol = 5e-4
b_tol = 6e-3

mu_D_typ, mu_V_typ, std_D_typ, std_V_typ, cov_DV_typ = data2DV(aa, C, [tauimp0, tp0, b0], [tauimp_tol, tp_tol, b_tol])
Sigma_DV_typ = scipy.asarray([[std_D_typ**2.0, cov_DV_typ], [cov_DV_typ, std_V_typ**2.0]], dtype=float)
lam_typ, v_typ = scipy.linalg.eigh(Sigma_DV_typ)
chi2_95 = -scipy.log(1.0 - 0.95) * 2.0
a_typ = 2.0 * scipy.sqrt(chi2_95 * lam_typ[-1])
b_typ = 2.0 * scipy.sqrt(chi2_95 * lam_typ[-2])
alpha_typ = scipy.arctan2(v_typ[1, -1], v_typ[0, -1])

# Evaluate uncertainties over a dense grid:
std_tauimp_grid = scipy.logspace(-4, -2, 100)
std_tp_grid = scipy.logspace(-4, -2, 101)
std_b_grid = scipy.logspace(-3, -1, 102)

# First for just t_p, V/D:
std_tauimp = 1e-6
STD_TP, STD_B = scipy.meshgrid(std_tp_grid, std_b_grid)

sum_bb = C[0, 0]**2.0 / std_tauimp**2.0 + C[1, 0]**2.0 / STD_TP**2.0 + C[2, 0]**2.0 / STD_B**2.0
sum_cc = C[0, 1]**2.0 / std_tauimp**2.0 + C[1, 1]**2.0 / STD_TP**2.0 + C[2, 1]**2.0 / STD_B**2.0
sum_bc = C[0, 0] * C[0, 1] / std_tauimp**2.0 + C[1, 0] * C[1, 1] / STD_TP**2.0 + C[2, 0] * C[2, 1] / STD_B**2.0

denom = sum_bb * sum_cc - sum_bc**2.0

std_D_fixed_tauimp = scipy.sqrt(sum_cc / denom)
std_V_fixed_tauimp = scipy.sqrt(sum_bb / denom)
cov_DV_fixed_tauimp = -sum_bc / denom

# Now for all three:
STD_TAUIMP, STD_TP, STD_B = scipy.meshgrid(std_tauimp_grid, std_tp_grid, std_b_grid, indexing='ij')

sum_bb = C[0, 0]**2.0 / STD_TAUIMP**2.0 + C[1, 0]**2.0 / STD_TP**2.0 + C[2, 0]**2.0 / STD_B**2.0
sum_cc = C[0, 1]**2.0 / STD_TAUIMP**2.0 + C[1, 1]**2.0 / STD_TP**2.0 + C[2, 1]**2.0 / STD_B**2.0
sum_bc = C[0, 0] * C[0, 1] / STD_TAUIMP**2.0 + C[1, 0] * C[1, 1] / STD_TP**2.0 + C[2, 0] * C[2, 1] / STD_B**2.0

denom = sum_bb * sum_cc - sum_bc**2.0

std_D1 = scipy.sqrt(sum_cc / denom)
std_V1 = scipy.sqrt(sum_bb / denom)
cov_DV1 = -sum_bc / denom

# Extract the target precision surface using marching cubes:
verts_D, faces_D = measure.marching_cubes(
    std_D1,
    D0 * target_precision,
    spacing=(
        scipy.log10(std_tauimp_grid[1] / std_tauimp_grid[0]),
        scipy.log10(std_tp_grid[1] / std_tp_grid[0]),
        scipy.log10(std_b_grid[1] / std_b_grid[0])
    )
)
verts_D += scipy.log10([std_tauimp_grid[0], std_tp_grid[0], std_b_grid[0]])

verts_V, faces_V = measure.marching_cubes(
    std_V1,
    scipy.absolute(V0) * target_precision,
    spacing=(
        scipy.log10(std_tauimp_grid[1] / std_tauimp_grid[0]),
        scipy.log10(std_tp_grid[1] / std_tp_grid[0]),
        scipy.log10(std_b_grid[1] / std_b_grid[0])
    )
)
verts_V += scipy.log10([std_tauimp_grid[0], std_tp_grid[0], std_b_grid[0]])

# Now densely sample the space to figure out the necessary resolution:
ntrials = 1000
dt_grid = scipy.logspace(-4, -2, 50)
noise_grid = scipy.logspace(-3, 0, 51)
npts_grid = scipy.asarray([1, 3, 5, 32], dtype=int)
DT, N, NP = scipy.meshgrid(dt_grid, noise_grid, npts_grid, indexing='ij')

# Compare to real HiReX-SR data:
# SOMETHING IS OFF ABOUT THESE UNITS!
# T_sr = MDSplus.Tree('spectroscopy', 1101014006)
# n_sr = T_sr.getNode('hirex_sr.raw_data.h_like.inten')
# flux_sr = n_sr.data() # shape is [313, 16, 200]: [t, z, lam]
# lam_sr = n_sr.dim_of(0).data()
# z_sr = n_sr.dim_of(1).data()
# t_sr = n_sr.dim_of(2).data()
# summed_flux_sr = flux_sr.sum(axis=2)
# # dt_grid_sr = scipy.logspace(scipy.log10(6e-3), -2, 50)
# u_on_grid_sr = 1.0 / scipy.sqrt(summed_flux_sr.max() * dt_grid)
# dt_sr = 6e-3
# u_act_sr = 1.0 / scipy.sqrt(summed_flux_sr.max() * dt_sr)

dt_sr = 6e-3
u_act_sr = 0.05
u_on_grid_sr = u_act_sr * scipy.sqrt(dt_sr / dt_grid)

# And compare to VUV:
dt_vuv = 2e-3
u_act_vuv = 0.02
u_on_grid_vuv = u_act_vuv * scipy.sqrt(dt_vuv / dt_grid)

# And compare to real XTOMO data:
# THIS COMES UP WITH REALLY LOW NOISE!
T_xtomo = MDSplus.Tree('xtomo', 1101014006)
arrays = [1, 3]
nchords = 38
sigs = []
for ar in arrays:
    for c in range(0, nchords):
        sigs.append(T_xtomo.getNode('signals.array_%d.chord_%02d' % (ar, c + 1)).data())
t_xtomo = T_xtomo.getNode('signals.array_%d.chord_%02d' % (ar, c + 1)).dim_of().data()
sigs = scipy.asarray([s[:len(t_xtomo)] for s in sigs], dtype=float)
flux_xtomo = (sigs.max() - sigs.mean()) * 0.32e-9 / (scipy.constants.h * scipy.constants.c)
u_on_grid_xtomo = 1.0 / scipy.sqrt(flux_xtomo * dt_grid)
dt_xtomo = 4e-6
u_act_xtomo = 1.0 / scipy.sqrt(flux_xtomo * dt_xtomo)

# Compute sigma_b analytically:
std_b_est = scipy.sqrt(2.0 * DT[:, :, 0] / tauimp0) * N[:, :, 0] * b0

# Make a spline to generate synthetic data at any location:
cs_den_true, sqrtpsinorm, time, ne, Te = r.DV2cs_den(r.params_true)
spl = scipy.interpolate.RectBivariateSpline(
    time,
    sqrtpsinorm,
    cs_den_true.sum(axis=1),
    s=0
)

def make_inspection_plot(y, mu, std, name=''):
    """Make a plot with the histogram, fitted distribution and QQ plot.
    """
    # Estimate number of bins:
    lq, uq = scipy.stats.scoreatpercentile(y, [25, 75])
    h = 2.0 * (uq - lq) / len(y)**(1.0 / 3.0)
    nn = int(scipy.ceil((y.max() - y.min()) / h))
    
    f = plt.figure()
    a = f.add_subplot(2, 1, 1)
    a.set_title(name)
    a.hist(y, bins=nn, normed=True)
    y_grid = scipy.linspace(y.min(), y.max(), 1000)
    a.plot(y_grid, scipy.stats.norm.pdf(y_grid, loc=mu, scale=std))
    a = f.add_subplot(2, 1, 2)
    scipy.stats.probplot(y, plot=a)

def eval_dt_n_np(dtnnp):
    """Evaluate the uncertainty in tp, tauimp and b for the given parameters.
    
    Parameters
    ----------
    dtnnp : 3-tuple of floats
        The sampling interval (in seconds), the relative noise level and the
        number of spatial points to use.
    """
    dt, n, npts = dtnnp
    
    t = scipy.arange(r.time_1, r.time_2, dt)
    if npts == 1:
        roa_grid = [0.0, 1.0]
        sqrtpsinorm_grid = [0.0, 1.0]
    else:
        roa_grid = scipy.linspace(0.0, 1.0, int(npts))
        sqrtpsinorm_grid = r.efit_tree.roa2psinorm(
            roa_grid,
            (r.time_1 + r.time_2) / 2.0,
            sqrt=True
        )
    
    # Compute the linearization coefficients for each of the radii:
    if npts > 1:
        bcoeffs = scipy.zeros((npts - 1, 3))
        for i, roa in enumerate(roa_grid[1:]):
            yr = nrn0(roa)
            dum = y2abc(yr)
            bcoeffs[i, :] = dum[:-1]
    
    tp = scipy.zeros(ntrials)
    tauimp = scipy.zeros(ntrials)
    if npts > 1:
        b = scipy.zeros((ntrials, npts - 1))
    else:
        b = None
    
    for i in range(0, ntrials):
        tv = t - dt * scipy.rand()
        
        y = spl(tv, sqrtpsinorm_grid)
        std_y = scipy.absolute(n * y)
        std_y[std_y < 1e-3 * y.max()] = 1e-3 * y.max()
        y += std_y * scipy.randn(*y.shape)
        
        # Find the core peaking time without using an optimizer:
        syc = scipy.interpolate.UnivariateSpline(
            tv,
            y[:, 0],
            w=1.0 / std_y[:, 0],
            ext=1,
            s=len(y[:, 0])
        )
        t_grid = scipy.linspace(tv.min(), tv.min() + 2 * (tv[y[:, 0].argmax()] - tv.min()), 100)
        tp[i] = t_grid[syc(t_grid).argmax()] - r.time_1
        
        # f = plt.figure()
        # a = f.add_subplot(1, 1, 1)
        # a.errorbar(tv, y[:, 0], yerr=std_y[:, 0], fmt='.')
        # a.plot(t_grid, syc(t_grid))
        
        # Find the impurity confinement time:
        if npts > 1:
            N_proxy = y.sum(axis=1)
        else:
            N_proxy = y[:, 0]
        mask = (tv > r.time_1 + tp0 + 0.01) & (N_proxy > 0.0)
        if mask.sum() >= 2:
            X = scipy.hstack((
                scipy.ones((mask.sum(), 1)),
                scipy.atleast_2d(tv[mask]).T
            ))
            theta, dum1, dum2, dum3 = scipy.linalg.lstsq(
                X.T.dot(X), X.T.dot(scipy.log(N_proxy[mask]))
            )
            tauimp[i] = -1.0 / theta[1]
        else:
            tauimp[i] = scipy.nan
        
        # Find the broadness factors:
        if npts > 1:
            if mask.sum() >= 1:
                prof = y / y[:, 0][:, None]
                b[i, :] = scipy.median(prof[mask, 1:], axis=0)
            else:
                b[i, :] = scipy.nan
    
    # Estimate the values and uncertainties of D and V:
    mu_tp = scipy.nanmedian(tp)
    std_tp = profiletools.robust_std(tp[~scipy.isnan(tp)])
    
    # Using robust estimators with tauimp fails because at low noise, low
    # sampling rate there are lots of tail events that are ignored, giving too
    # rosy of a picture.
    # mu_tauimp = scipy.nanmedian(tauimp)
    # std_tauimp = profiletools.robust_std(tauimp[~scipy.isnan(tauimp)])
    
    # mu_tp = scipy.nanmean(tp)
    # std_tp = scipy.nanstd(tp, ddof=1)
    mu_tauimp = scipy.nanmean(tauimp)
    std_tauimp = scipy.nanstd(tauimp, ddof=1)
    
    # make_inspection_plot(tauimp, mu_tauimp, std_tauimp, name=r'$\tau_{\text{imp}}$')
    # make_inspection_plot(tp, mu_tp, std_tp, name=r"$t_{\text{p}}$")
    
    if npts > 1:
        # mu_b = scipy.nanmean(b, axis=0)
        # std_b = scipy.nanstd(b, axis=0, ddof=1)
        
        # Non-robust estimators are also a bad idea for b because there are strong
        # tails -- the theoretical distribution in fact has infinite variance!
        mu_b = scipy.nanmedian(b, axis=0)
        std_b = profiletools.robust_std(b, axis=0)
        
        # for idx in xrange(0, b.shape[1]):
        #     make_inspection_plot(b[:, idx], mu_b[idx], std_b[idx], name="$b$ %d" % (idx,))
    else:
        mu_b = None
        std_b = None
    
    aa = scipy.asarray([a_tauimp, a_tp], dtype=float)
    C = scipy.asarray([[dtauimpdD, dtauimpdV], [dtpdD, dtpdV]], dtype=float)
    yd = scipy.asarray([mu_tauimp, mu_tp], dtype=float)
    std_yd = scipy.asarray([std_tauimp, std_tp], dtype=float)
    if npts > 1:
        aa = scipy.concatenate((aa, bcoeffs[:, 0]))
        C = scipy.vstack((C, bcoeffs[:, 1:]))
        yd = scipy.concatenate((yd, mu_b))
        std_yd = scipy.concatenate((std_yd, std_b))
    
    mu_D, mu_V, std_D, std_V, cov_DV = data2DV(aa, C, yd, std_yd)
    
    return tp, tauimp, b, std_D, std_V, cov_DV, mu_D, mu_V, mu_tp, std_tp, mu_tauimp, std_tauimp, mu_b, std_b

t_start = time_.time()
pool = iPool(multiprocessing.cpu_count())
res_samp = pool.map(eval_dt_n_np, zip(DT.ravel(), N.ravel(), NP.ravel()))
pool.close()
t_elapsed = time_.time() - t_start
print("%.3f minutes elapsed" % (t_elapsed / 60.0,))

tp_exp = scipy.reshape([rs[0] for rs in res_samp], DT.shape + (ntrials,))
tauimp_exp = scipy.reshape([rs[1] for rs in res_samp], DT.shape + (ntrials,))
std_D = scipy.reshape([rs[3] for rs in res_samp], DT.shape)
std_V = scipy.reshape([rs[4] for rs in res_samp], DT.shape)
cov_DV = scipy.reshape([rs[5] for rs in res_samp], DT.shape)
mu_D = scipy.reshape([rs[6] for rs in res_samp], DT.shape)
mu_V = scipy.reshape([rs[7] for rs in res_samp], DT.shape)

mu_tp = scipy.reshape([rs[8] for rs in res_samp], DT.shape)
std_tp = scipy.reshape([rs[9] for rs in res_samp], DT.shape)
mu_tauimp = scipy.reshape([rs[10] for rs in res_samp], DT.shape)
std_tauimp = scipy.reshape([rs[11] for rs in res_samp], DT.shape)

# b(0.75) requires some gymnastics:
res_samp_arr = scipy.asarray(res_samp, dtype=object)
std_b_arr = scipy.reshape(res_samp_arr[:, 13], DT.shape)
std_b_3 = std_b_arr[:, :, 2]
std_b_75 = scipy.reshape([q[2] for q in std_b_3.ravel()], DT.shape[:2])
b_arr = scipy.reshape(res_samp_arr[:, 2], DT.shape)
b_3 = b_arr[:, :, 2]
b_75 = scipy.reshape([q[:, 2] for q in b_3.ravel()], DT.shape[:2] + (ntrials,))
mu_b_arr = scipy.reshape(res_samp_arr[:, 12], DT.shape)
mu_b_3 = mu_b_arr[:, :, 2]
mu_b_75 = scipy.reshape([q[2] for q in mu_b_3.ravel()], DT.shape[:2])

# Sampling of b:
n = 1e-1
n0 = 1.0
n075 = b0 * n0
std_b_lin = scipy.sqrt(2.0) * n * b0
std_n075 = n * n075
std_n0 = n * n0
b_grid = scipy.linspace(b0 - 5 * std_b_lin, b0 + 5 * std_b_lin, 1000)
a = scipy.sqrt(b_grid**2.0 / std_n075**2.0 + 1.0 / std_n0**2.0)
bb = n075 * b_grid / std_n075**2.0 + n0 / std_n0**2.0
c = n075**2.0 / std_n075**2.0 + n0**2.0 / std_n0**2.0
d = scipy.exp((bb**2.0 - c * a**2.0) / (2.0 * a**2.0))
p_b = (
    bb * d / (a**3.0 * scipy.sqrt(2.0 * scipy.pi) * std_n075 * std_n0) * (
        scipy.stats.norm.cdf(bb / a) - scipy.stats.norm.cdf(-bb / a)
    ) +
    scipy.exp(-c / 2.0) / (a**2.0 * scipy.pi * std_n075 * std_n0)
)

# Do it sampling-based:
# Look at the convergence of the mean:
n_pts = [10, 100, 1000, 10000]  # Number of points to use in each realization.
n_samps = 1e4  # Number of samples to use for each value in n_pts.
mean_samps = scipy.zeros((len(n_pts), n_samps))
median_samps = scipy.zeros((len(n_pts), n_samps))
for i, N in enumerate(n_pts):
    n0_samp = scipy.stats.norm.rvs(loc=n0, scale=n * n0, size=(n_samps, N))
    n075_samp = scipy.stats.norm.rvs(loc=n075, scale=n * n075, size=(n_samps, N))
    b_samp = n075_samp / n0_samp
    mean_samps[i, :] = b_samp.mean(axis=1)
    median_samps[i, :] = scipy.median(b_samp, axis=1)

# Basic case to illustrate it:
n0_samp = scipy.stats.norm.rvs(loc=n0, scale=n * n0, size=n_samps)
n075_samp = scipy.stats.norm.rvs(loc=n075, scale=n * n075, size=n_samps)
b_samp = n075_samp / n0_samp

os.chdir(cdir)

# Save the variables for later use:
# https://stackoverflow.com/questions/2960864/how-can-i-save-all-the-variables-in-the-current-python-session
import shelve

shelf_name = 'bayes_time_res.shelf'
shelf = shelve.open(shelf_name, 'n')
for key in dir():
    try:
        shelf[key] = globals()[key]
    except:
        print("Failed to shelve %s" % (key,))
shelf.close()

# To restore:
# shelf = shelve.open(shelf_name)
# for key in shelf:
#     try:
#         globals()[key] = shelf[key]
#     except:
#         print("Failed to restore %s" % (key,))
# shelf.close()

# Make some plots:
import setupplots
setupplots.thesis_format()
import matplotlib.pyplot as plt
plt.ion()
import matplotlib.lines as mlines
import matplotlib.patches as mplp
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.cm
import matplotlib.gridspec as mplgs
from matplotlib.colors import LogNorm

D_grid_plot = setupplots.make_pcolor_grid(D_grid)
V_grid_plot = setupplots.make_pcolor_grid(V_grid)

std_tp_grid_plot = 10.0**setupplots.make_pcolor_grid(scipy.log10(std_tp_grid))
std_b_grid_plot = 10.0**setupplots.make_pcolor_grid(scipy.log10(std_b_grid))

# Plot the 2d 10% precision contour for D:
f = plt.figure(figsize=(0.75 * setupplots.TEXTWIDTH, 0.75 * setupplots.TEXTWIDTH / 1.618))
a = f.add_subplot(1, 1, 1)
pcm = a.pcolormesh(
    std_tp_grid_plot,
    std_b_grid_plot,
    std_D_fixed_tauimp,
    vmin=0.0
)
cb = f.colorbar(pcm)
cb.set_label(r"$\sigma_D$ [$\si{m^2/s}$]")
a.contour(std_tp_grid, std_b_grid, std_D_fixed_tauimp, 50, alpha=0.5, linewidths=setupplots.lw / 2.0, colors='w')
cs = a.contour(std_tp_grid, std_b_grid, std_D_fixed_tauimp, [target_precision * D0,], colors='g', linewidths=3 * setupplots.lw)
cb.add_lines(cs)
a.set_xscale('log')
a.set_yscale('log')
a.set_xlabel(r'$\sigma_{t_{\text{r}}}$ [s]')
a.set_ylabel(r'$\sigma_{b_{0.75}}$')
a.set_title("Uncertainty in $D$")
setupplots.apply_formatter(f)
f.savefig("Dtol.pdf", bbox_inches='tight')
f.savefig("Dtol.pgf", bbox_inches='tight')

# Plot the 2d 10% precision contour for V:
f = plt.figure(figsize=(0.75 * setupplots.TEXTWIDTH, 0.75 * setupplots.TEXTWIDTH / 1.618))
a = f.add_subplot(1, 1, 1)
pcm = a.pcolormesh(
    std_tp_grid_plot,
    std_b_grid_plot,
    std_V_fixed_tauimp,
    vmin=0.0
)
cb = f.colorbar(pcm)
cb.set_label(r"$\sigma_V$ [$\si{m/s}$]")
a.contour(std_tp_grid, std_b_grid, std_V_fixed_tauimp, 50, alpha=0.5, linewidths=setupplots.lw / 2.0, colors='w')
cs = a.contour(std_tp_grid, std_b_grid, std_V_fixed_tauimp, [target_precision * scipy.absolute(V0),], colors='g', linewidths=3 * setupplots.lw)
cb.add_lines(cs)
a.set_xscale('log')
a.set_yscale('log')
a.set_xlabel(r'$\sigma_{t_{\text{r}}}$ [s]')
a.set_ylabel(r'$\sigma_{b_{0.75}}$')
a.set_title("Uncertainty in $V$")
setupplots.apply_formatter(f)
f.savefig("Vtol.pdf", bbox_inches='tight')
f.savefig("Vtol.pgf", bbox_inches='tight')

# Plot of 10% precision surface for D:
f = plt.figure(figsize=(setupplots.TEXTWIDTH, setupplots.TEXTWIDTH / 1.618))
a = f.add_subplot(111, projection='3d')
a.view_init(elev=32, azim=29)
v = verts_D[faces_D]
mesh_D = Poly3DCollection(
    v,
    linewidths=0,
    facecolors=matplotlib.cm.plasma(
        (v[:, 0, 2] - v[:, 0, 2].min()) / (v[:, 0, 2].max() - v[:, 0, 2].min())
    )
)
a.add_collection3d(mesh_D)
# Add contours at constant std_tauimp:
for k in range(0, len(std_tauimp_grid), 10):
    contours = measure.find_contours(std_D1[k, :, :], D0 * target_precision)
    if len(contours) > 0:
        l = contours[0]
        l *= [
            scipy.log10(std_tp_grid[1] / std_tp_grid[0]),
            scipy.log10(std_b_grid[1] / std_b_grid[0])
        ]
        l += [scipy.log10(std_tp_grid[0]), scipy.log10(std_b_grid[0])]
        a.plot(l[:, 0], l[:, 1], scipy.log10(std_tauimp_grid[k]), color='w', zdir='x', alpha=0.5)
# Add contours at constant std_tp:
for k in range(0, len(std_tp_grid), 10):
    contours = measure.find_contours(std_D1[:, k, :], D0 * target_precision)
    if len(contours) > 0:
        l = contours[0]
        l *= [
            scipy.log10(std_tauimp_grid[1] / std_tauimp_grid[0]),
            scipy.log10(std_b_grid[1] / std_b_grid[0])
        ]
        l += [scipy.log10(std_tauimp_grid[0]), scipy.log10(std_b_grid[0])]
        a.plot(l[:, 0], l[:, 1], scipy.log10(std_tp_grid[k]), color='w', zdir='y', alpha=0.5)
# Add contours at constant std_b:
for k in range(0, len(std_b_grid), 10):
    contours = measure.find_contours(std_D1[:, :, k], D0 * target_precision)
    if len(contours) > 0:
        l = contours[0]
        l *= [
            scipy.log10(std_tauimp_grid[1] / std_tauimp_grid[0]),
            scipy.log10(std_tp_grid[1] / std_tp_grid[0])
        ]
        l += [scipy.log10(std_tauimp_grid[0]), scipy.log10(std_tp_grid[0])]
        a.plot(l[:, 0], l[:, 1], scipy.log10(std_b_grid[k]), color='w', zdir='z', alpha=0.5)
a.set_xlim(scipy.around(verts_V[:, 0].min()), scipy.around(verts_V[:, 0].max()))
a.set_ylim(scipy.around(verts_V[:, 1].min()), scipy.around(verts_V[:, 1].max()))
a.set_zlim(scipy.around(verts_V[:, 2].min()), scipy.around(verts_V[:, 2].max()))
a.set_xticks(scipy.arange(scipy.around(verts_V[:, 0].min()), scipy.around(verts_V[:, 0].max()) + 1))
a.set_yticks(scipy.arange(scipy.around(verts_V[:, 1].min()), scipy.around(verts_V[:, 1].max()) + 1))
a.set_zticks(scipy.arange(scipy.around(verts_V[:, 2].min()), scipy.around(verts_V[:, 2].max()) + 1))
a.set_xticklabels([r'$10^{%d}\vphantom{0123456789}$' % (t,) for t in a.get_xticks()])
a.set_yticklabels([r'$10^{%d}\vphantom{0123456789}$' % (t,) for t in a.get_yticks()])
a.set_zticklabels([r'$10^{%d}\vphantom{0123456789}$' % (t,) for t in a.get_zticks()])
a.set_xlabel(r'$\sigma_{\tau_{\text{imp}}}$ [s]')
a.set_ylabel(r'$\sigma_{t_{\text{r}}}$ [s]')
a.set_zlabel(r'$\sigma_{b_{0.75}}$')
a.set_title(r"Requirements for 10\% uncertainty in $D$")
f.subplots_adjust(left=0.05, bottom=0.05, right=1.0, top=1.0)
f.savefig("Dsurf.pdf")
f.savefig("Dsurf.pgf")

# Plot of 10% precision surface for V:
f = plt.figure(figsize=(setupplots.TEXTWIDTH, setupplots.TEXTWIDTH / 1.618))
a = f.add_subplot(111, projection='3d')
a.view_init(elev=32, azim=29)
v = verts_V[faces_V]
mesh_V = Poly3DCollection(
    v,
    linewidths=0,
    facecolors=matplotlib.cm.plasma(
        (v[:, 0, 2] - v[:, 0, 2].min()) / (v[:, 0, 2].max() - v[:, 0, 2].min())
    )
)
a.add_collection3d(mesh_V)
# Add contours at constant std_tauimp:
for k in range(0, len(std_tauimp_grid), 10):
    contours = measure.find_contours(std_V1[k, :, :], scipy.absolute(V0) * target_precision)
    if len(contours) > 0:
        l = contours[0]
        l *= [
            scipy.log10(std_tp_grid[1] / std_tp_grid[0]),
            scipy.log10(std_b_grid[1] / std_b_grid[0])
        ]
        l += [scipy.log10(std_tp_grid[0]), scipy.log10(std_b_grid[0])]
        a.plot(l[:, 0], l[:, 1], scipy.log10(std_tauimp_grid[k]), color='w', zdir='x', alpha=0.5)
# Add contours at constant std_tp:
for k in range(0, len(std_tp_grid), 10):
    contours = measure.find_contours(std_V1[:, k, :], scipy.absolute(V0) * target_precision)
    if len(contours) > 0:
        l = contours[0]
        l *= [
            scipy.log10(std_tauimp_grid[1] / std_tauimp_grid[0]),
            scipy.log10(std_b_grid[1] / std_b_grid[0])
        ]
        l += [scipy.log10(std_tauimp_grid[0]), scipy.log10(std_b_grid[0])]
        a.plot(l[:, 0], l[:, 1], scipy.log10(std_tp_grid[k]), color='w', zdir='y', alpha=0.5)
# Add contours at constant std_b:
for k in range(0, len(std_b_grid), 10):
    contours = measure.find_contours(std_V1[:, :, k], scipy.absolute(V0) * target_precision)
    if len(contours) > 0:
        l = contours[0]
        l *= [
            scipy.log10(std_tauimp_grid[1] / std_tauimp_grid[0]),
            scipy.log10(std_tp_grid[1] / std_tp_grid[0])
        ]
        l += [scipy.log10(std_tauimp_grid[0]), scipy.log10(std_tp_grid[0])]
        a.plot(l[:, 0], l[:, 1], scipy.log10(std_b_grid[k]), color='w', zdir='z', alpha=0.5)
a.set_xlim(scipy.around(verts_V[:, 0].min()), scipy.around(verts_V[:, 0].max()))
a.set_ylim(scipy.around(verts_V[:, 1].min()), scipy.around(verts_V[:, 1].max()))
a.set_zlim(scipy.around(verts_V[:, 2].min()), scipy.around(verts_V[:, 2].max()))
a.set_xticks(scipy.arange(scipy.around(verts_V[:, 0].min()), scipy.around(verts_V[:, 0].max()) + 1))
a.set_yticks(scipy.arange(scipy.around(verts_V[:, 1].min()), scipy.around(verts_V[:, 1].max()) + 1))
a.set_zticks(scipy.arange(scipy.around(verts_V[:, 2].min()), scipy.around(verts_V[:, 2].max()) + 1))
a.set_xticklabels([r'$10^{%d}\vphantom{0123456789}$' % (t,) for t in a.get_xticks()])
a.set_yticklabels([r'$10^{%d}\vphantom{0123456789}$' % (t,) for t in a.get_yticks()])
a.set_zticklabels([r'$10^{%d}\vphantom{0123456789}$' % (t,) for t in a.get_zticks()])
a.set_xlabel(r'$\sigma_{\tau_{\text{imp}}}$ [s]')
a.set_ylabel(r'$\sigma_{t_{\text{r}}}$ [s]')
a.set_zlabel(r'$\sigma_{b_{0.75}}$')
a.set_title(r"Requirements for 10\% uncertainty in $V$")
f.subplots_adjust(left=0.05, bottom=0.05, right=1.0, top=1.0)
f.savefig("Vsurf.pdf")
f.savefig("Vsurf.pgf")

# Combined plots of 10% precision surfaces for D and V:
f = plt.figure(figsize=(setupplots.TEXTWIDTH, setupplots.TEXTWIDTH / 1.618))
a = f.add_subplot(111, projection='3d')
a.view_init(elev=32, azim=29)
v = verts_D[faces_D]
mesh_D = Poly3DCollection(
    v,
    linewidths=0,
    facecolors=matplotlib.cm.Blues_r(
        (v[:, 0, 2] - v[:, 0, 2].min()) / (v[:, 0, 2].max() - v[:, 0, 2].min())
    )
)
a.add_collection3d(mesh_D)
v = verts_V[faces_V]
mesh_V = Poly3DCollection(
    v,
    linewidths=0,
    facecolors=matplotlib.cm.Greens_r(
        (v[:, 0, 2] - v[:, 0, 2].min()) / (v[:, 0, 2].max() - v[:, 0, 2].min())
    )
)
a.add_collection3d(mesh_V)
# Contours for std_D:
# Add contours at constant std_tauimp:
for k in range(0, len(std_tauimp_grid), 10):
    contours = measure.find_contours(std_D1[k, :, :], D0 * target_precision)
    if len(contours) > 0:
        l = contours[0]
        l *= [
            scipy.log10(std_tp_grid[1] / std_tp_grid[0]),
            scipy.log10(std_b_grid[1] / std_b_grid[0])
        ]
        l += [scipy.log10(std_tp_grid[0]), scipy.log10(std_b_grid[0])]
        a.plot(l[:, 0], l[:, 1], scipy.log10(std_tauimp_grid[k]), color='b', zdir='x', alpha=0.5)
# Add contours at constant std_tp:
for k in range(0, len(std_tp_grid), 10):
    contours = measure.find_contours(std_D1[:, k, :], D0 * target_precision)
    if len(contours) > 0:
        l = contours[0]
        l *= [
            scipy.log10(std_tauimp_grid[1] / std_tauimp_grid[0]),
            scipy.log10(std_b_grid[1] / std_b_grid[0])
        ]
        l += [scipy.log10(std_tauimp_grid[0]), scipy.log10(std_b_grid[0])]
        a.plot(l[:, 0], l[:, 1], scipy.log10(std_tp_grid[k]), color='b', zdir='y', alpha=0.5)
# Add contours at constant std_b:
for k in range(0, len(std_b_grid), 10):
    contours = measure.find_contours(std_D1[:, :, k], D0 * target_precision)
    if len(contours) > 0:
        l = contours[0]
        l *= [
            scipy.log10(std_tauimp_grid[1] / std_tauimp_grid[0]),
            scipy.log10(std_tp_grid[1] / std_tp_grid[0])
        ]
        l += [scipy.log10(std_tauimp_grid[0]), scipy.log10(std_tp_grid[0])]
        a.plot(l[:, 0], l[:, 1], scipy.log10(std_b_grid[k]), color='b', zdir='z', alpha=0.5)
# Contours for std_V:
# Add contours at constant std_tauimp:
for k in range(0, len(std_tauimp_grid), 10):
    contours = measure.find_contours(std_V1[k, :, :], scipy.absolute(V0) * target_precision)
    if len(contours) > 0:
        l = contours[0]
        l *= [
            scipy.log10(std_tp_grid[1] / std_tp_grid[0]),
            scipy.log10(std_b_grid[1] / std_b_grid[0])
        ]
        l += [scipy.log10(std_tp_grid[0]), scipy.log10(std_b_grid[0])]
        a.plot(l[:, 0], l[:, 1], scipy.log10(std_tauimp_grid[k]), color='g', zdir='x', alpha=0.5)
# Add contours at constant std_tp:
for k in range(0, len(std_tp_grid), 10):
    contours = measure.find_contours(std_V1[:, k, :], scipy.absolute(V0) * target_precision)
    if len(contours) > 0:
        l = contours[0]
        l *= [
            scipy.log10(std_tauimp_grid[1] / std_tauimp_grid[0]),
            scipy.log10(std_b_grid[1] / std_b_grid[0])
        ]
        l += [scipy.log10(std_tauimp_grid[0]), scipy.log10(std_b_grid[0])]
        a.plot(l[:, 0], l[:, 1], scipy.log10(std_tp_grid[k]), color='g', zdir='y', alpha=0.5)
# Add contours at constant std_b:
for k in range(0, len(std_b_grid), 10):
    contours = measure.find_contours(std_V1[:, :, k], scipy.absolute(V0) * target_precision)
    if len(contours) > 0:
        l = contours[0]
        l *= [
            scipy.log10(std_tauimp_grid[1] / std_tauimp_grid[0]),
            scipy.log10(std_tp_grid[1] / std_tp_grid[0])
        ]
        l += [scipy.log10(std_tauimp_grid[0]), scipy.log10(std_tp_grid[0])]
        a.plot(l[:, 0], l[:, 1], scipy.log10(std_b_grid[k]), color='g', zdir='z', alpha=0.5)
a.set_xlim(scipy.around(verts_V[:, 0].min()), scipy.around(verts_V[:, 0].max()))
a.set_ylim(scipy.around(verts_V[:, 1].min()), scipy.around(verts_V[:, 1].max()))
a.set_zlim(scipy.around(verts_V[:, 2].min()), scipy.around(verts_V[:, 2].max()))
a.set_xticks(scipy.arange(scipy.around(verts_V[:, 0].min()), scipy.around(verts_V[:, 0].max()) + 1))
a.set_yticks(scipy.arange(scipy.around(verts_V[:, 1].min()), scipy.around(verts_V[:, 1].max()) + 1))
a.set_zticks(scipy.arange(scipy.around(verts_V[:, 2].min()), scipy.around(verts_V[:, 2].max()) + 1))
a.set_xticklabels([r'$10^{%d}\vphantom{0123456789}$' % (t,) for t in a.get_xticks()])
a.set_yticklabels([r'$10^{%d}\vphantom{0123456789}$' % (t,) for t in a.get_yticks()])
a.set_zticklabels([r'$10^{%d}\vphantom{0123456789}$' % (t,) for t in a.get_zticks()])
a.set_xlabel(r'$\sigma_{\tau_{\text{imp}}}$ [s]')
a.set_ylabel(r'$\sigma_{t_{\text{r}}}$ [s]')
a.set_zlabel(r'$\sigma_{b_{0.75}}$')
a.set_title(r"Requirements for 10\% uncertainty in $\textcolor{MPLb}{D}$ and $\textcolor{MPLg}{V}$")
f.subplots_adjust(left=0.05, bottom=0.05, right=1.0, top=1.0)
f.savefig("CombSurf.pdf")
f.savefig("CombSurf.pgf")

# Plots of sigma_D and sigma_V in experimental space:
dt_grid_plot = 10.0**(setupplots.make_pcolor_grid(scipy.log10(dt_grid)))
noise_grid_plot = 10.0**(setupplots.make_pcolor_grid(scipy.log10(noise_grid)))

f_D = plt.figure(figsize=(setupplots.TEXTWIDTH, setupplots.TEXTWIDTH / 1.618))
f_V = plt.figure(figsize=(setupplots.TEXTWIDTH, setupplots.TEXTWIDTH / 1.618))
gs = mplgs.GridSpec(2, 3, width_ratios=[5, 5, 10.0 / 15.0])
a_D = []
a_V = []
k = 0
j = 0
for i, npts in enumerate(npts_grid):
    a_D.append(f_D.add_subplot(gs[j, k], sharex=a_D[0] if i > 0 else None, sharey=a_D[0] if i > 0 else None))
    pcm_D = a_D[-1].pcolormesh(
        dt_grid_plot,
        noise_grid_plot,
        std_D[:, :, i].T,
        vmin=0,
        vmax=0.5
    )
    pcm_D.cmap.set_over('white')
    cs_D = a_D[-1].contour(
        dt_grid,
        noise_grid,
        std_D[:, :, i].T,
        [target_precision * D0,],
        colors='g',
        linewidths=3.0 * setupplots.lw
    )
    ln_D, = a_D[-1].plot(dt_grid, u_on_grid_sr, 'r--', lw=3.0 * setupplots.lw, label=r'\textsc{xics}')
    a_D[-1].plot(dt_sr, u_act_sr, 'ro', ms=2.0 * setupplots.ms)
    ln_D2, = a_D[-1].plot(dt_grid, u_on_grid_vuv, 'c:', lw=3.0 * setupplots.lw, label=r'\textsc{vuv}')
    a_D[-1].plot(dt_vuv, u_act_vuv, 'c^', ms=2.0 * setupplots.ms)
    a_D[-1].set_xscale('log')
    a_D[-1].set_yscale('log')
    a_D[-1].set_title("1 point" if npts == 1 else "%d points" % (npts,))
    if i < 2:
        plt.setp(a_D[-1].get_xticklabels(), visible=False)
    else:
        a_D[-1].set_xlabel(r'$\Delta t$ [s]')
    if i % 2 != 0:
        plt.setp(a_D[-1].get_yticklabels(), visible=False)
    else:
        a_D[-1].set_ylabel(r'$u$')
    
    a_V.append(f_V.add_subplot(gs[j, k], sharex=a_V[0] if i > 0 else None, sharey=a_V[0] if i > 0 else None))
    pcm_V = a_V[-1].pcolormesh(
        dt_grid_plot,
        noise_grid_plot,
        std_V[:, :, i].T,
        vmin=0,
        vmax=15
    )
    pcm_V.cmap.set_over('white')
    cs_V = a_V[-1].contour(
        dt_grid,
        noise_grid,
        std_V[:, :, i].T,
        [target_precision * scipy.absolute(V0),],
        colors='g',
        linewidths=3.0 * setupplots.lw
    )
    ln_V, = a_V[-1].plot(dt_grid, u_on_grid_sr, 'r--', lw=3.0 * setupplots.lw, label=r'\textsc{xics}')
    a_V[-1].plot(dt_sr, u_act_sr, 'ro', ms=2.0 * setupplots.ms)
    ln_V2, = a_V[-1].plot(dt_grid, u_on_grid_vuv, 'c:', lw=3.0 * setupplots.lw, label=r'\textsc{vuv}')
    a_V[-1].plot(dt_vuv, u_act_vuv, 'c^', ms=2.0 * setupplots.ms)
    a_V[-1].set_xscale('log')
    a_V[-1].set_yscale('log')
    a_V[-1].set_title("1 point" if npts == 1 else "%d points" % (npts,))
    if i < 2:
        plt.setp(a_V[-1].get_xticklabels(), visible=False)
    else:
        a_V[-1].set_xlabel(r'$\Delta t$ [s]')
    if i % 2 != 0:
        plt.setp(a_V[-1].get_yticklabels(), visible=False)
    else:
        a_V[-1].set_ylabel(r'$u$')
    
    k += 1
    if k == 2:
        k = 0
        j += 1

a_D[0].set_xlim(dt_grid[0], dt_grid[-1])
a_D[0].set_ylim(noise_grid[0], noise_grid[-1])
a_V[0].set_xlim(dt_grid[0], dt_grid[-1])
a_V[0].set_ylim(noise_grid[0], noise_grid[-1])
f_D.suptitle("Uncertainty in $D$")
cax_D = f_D.add_subplot(gs[:, 2])
cb_D = plt.colorbar(pcm_D, cax=cax_D, extend='max')
cb_D.add_lines(cs_D)
cb_D.set_label(r"$\sigma_D$ [$\si{m^2/s}$]")
f_V.suptitle("Uncertainty in $V$")
cax_V = f_V.add_subplot(gs[:, 2])
cb_V = plt.colorbar(pcm_V, cax=cax_V, extend='max')
cb_V.add_lines(cs_V)
cb_V.set_label(r"$\sigma_V$ [$\si{m/s}$]")
f_D.subplots_adjust(left=0.12, bottom=0.14, right=0.85, top=0.87, wspace=0.22, hspace=0.22)
f_V.subplots_adjust(left=0.12, bottom=0.14, right=0.85, top=0.87, wspace=0.22, hspace=0.22)
setupplots.apply_formatter(f_D)
setupplots.apply_formatter(f_V)
f_D.savefig("DtolExp.pdf")
f_D.savefig("DtolExp.pgf")
f_V.savefig("VtolExp.pdf")
f_V.savefig("VtolExp.pgf")

green_line_D = mlines.Line2D([], [], color='g', linestyle='-', label=r'$\sigma_D=0.1 D_0$')
green_line_V = mlines.Line2D([], [], color='g', linestyle='-', label=r'$\sigma_V=0.1 |V_0|$')

f_leg = plt.figure()
l = f_leg.legend(
    [green_line_D, ln_D, ln_D2],
    [green_line_D.get_label(), ln_D.get_label(), ln_D2.get_label()],
    ncol=3,
    loc='center'
)
f_leg.canvas.draw()
f_leg.set_figwidth(l.get_window_extent().width / 72)
f_leg.set_figheight(l.get_window_extent().height / 72)
f_leg.savefig("legDTolExp.pdf", bbox_inches='tight')
f_leg.savefig("legDTolExp.pgf", bbox_inches='tight')

f_leg = plt.figure()
l = f_leg.legend(
    [green_line_V, ln_V, ln_V2],
    [green_line_V.get_label(), ln_V.get_label(), ln_V2.get_label()],
    ncol=3,
    loc='center'
)
f_leg.canvas.draw()
f_leg.set_figwidth(l.get_window_extent().width / 72)
f_leg.set_figheight(l.get_window_extent().height / 72)
f_leg.savefig("legVTolExp.pdf", bbox_inches='tight')
f_leg.savefig("legVTolExp.pgf", bbox_inches='tight')

# Plot estimated sigma_b:
f = plt.figure(figsize=(0.75 * setupplots.TEXTWIDTH, 0.75 * setupplots.TEXTWIDTH / 1.618))
a = f.add_subplot(1, 1, 1)
pcm = a.pcolormesh(
    dt_grid_plot,
    noise_grid_plot,
    std_b_est.T,
    vmin=0.0
)
cb = f.colorbar(pcm)
cb.set_label(r"$\sigma_{b_{0.75}}$")
a.contour(dt_grid, noise_grid, std_b_est.T, 50, alpha=0.5, linewidths=setupplots.lw / 2.0, colors='w')
cs = a.contour(
    dt_grid,
    noise_grid,
    std_b_est.T,
    [b_tol, 1e-2,],
    linewidths=3.0 * setupplots.lw,
    colors=['g', 'r'],
    linestyles=['-', '--']
)
cb.add_lines(cs)
a.set_xscale('log')
a.set_yscale('log')
a.set_title("Estimated uncertainty in $b_{0.75}$")
a.set_xlabel(r'$\Delta t$ [s]')
a.set_ylabel(r'$u$')
a.set_xlim(dt_grid[0], dt_grid[-1])
a.set_ylim(noise_grid[0], noise_grid[-1])
setupplots.apply_formatter(f)
f.savefig('bTol.pdf', bbox_inches='tight')
f.savefig('bTol.pgf', bbox_inches='tight')

# Plot observed sigma_b:
f = plt.figure(figsize=(0.75 * setupplots.TEXTWIDTH, 0.75 * setupplots.TEXTWIDTH / 1.618))
a = f.add_subplot(1, 1, 1)
pcm = a.pcolormesh(
    dt_grid_plot,
    noise_grid_plot,
    std_b_75.T,
    vmin=0.0,
    vmax=std_b_est.max()
)
pcm.cmap.set_over('white')
cb = f.colorbar(pcm)#, extend='max')
cb.set_label(r"$\sigma_{b_{0.75}}$")
cs = a.contour(
    dt_grid,
    noise_grid,
    std_b_75.T,
    [b_tol, 1e-2],
    linewidths=3.0 * setupplots.lw,
    colors=['g', 'r'],
    linestyles=['-', '--']
)
cb.add_lines(cs)
a.set_xscale('log')
a.set_yscale('log')
a.set_title("Simulated uncertainty in $b_{0.75}$")
a.set_xlabel(r'$\Delta t$ [s]')
a.set_ylabel(r'$u$')
a.set_xlim(dt_grid[0], dt_grid[-1])
a.set_ylim(noise_grid[0], noise_grid[-1])

# Set up the histogram plot:
f_hist = plt.figure()
a_hist = f_hist.add_subplot(1, 1, 1)

def on_click_b(event):
    if f.canvas.manager.toolbar._active is not None:
        return
    if event.xdata is None or event.ydata is None:
        return
    
    print("Updating...")
    # Find out what dt, n are:
    dt_idx = profiletools.get_nearest_idx(event.xdata, dt_grid)
    noise_idx = profiletools.get_nearest_idx(event.ydata, noise_grid)
    
    print(dt_grid[dt_idx])
    print(noise_grid[noise_idx])
    
    # Just nuke the data, since we need to re-do it all:
    a_hist.clear()
    # Estimate number of bins:
    y = b_75[dt_idx, noise_idx]
    lq, uq = scipy.stats.scoreatpercentile(y, [25, 75])
    h = 2.0 * (uq - lq) / len(y)**(1.0 / 3.0)
    nn = int(scipy.ceil((y.max() - y.min()) / h))
    a_hist.hist(y, bins=nn, normed=True)
    grid = scipy.linspace(y.min(), y.max(), 100)
    a_hist.plot(
        grid,
        scipy.stats.norm.pdf(
            grid,
            loc=mu_b_75[dt_idx, noise_idx],
            scale=std_b_75[dt_idx, noise_idx]
        )
    )
    a_hist.plot(
        grid,
        scipy.stats.norm.pdf(
            grid,
            loc=scipy.nanmean(y),
            scale=scipy.nanstd(y, ddof=1)
        )
    )
    # a_hist.set_xlim(0.008, 0.014)
    f_hist.canvas.draw()
    
    print("Done!")

f.canvas.mpl_connect("button_press_event", on_click_b)
setupplots.apply_formatter(f)
f.savefig("bTolExp.pdf", bbox_inches='tight')
f.savefig("bTolExp.pgf", bbox_inches='tight')

# Plot observed sigma_t_p:
f = plt.figure(figsize=(0.75 * setupplots.TEXTWIDTH, 0.75 * setupplots.TEXTWIDTH / 1.618))
a = f.add_subplot(1, 1, 1)
pcm = a.pcolormesh(
    dt_grid_plot,
    noise_grid_plot,
    std_tp[:, :, 0].T,
    norm=LogNorm(),
    vmin=1e-4
    # vmin=std_b_est.min(),
    # vmax=std_b_est.max()
)
cb = f.colorbar(pcm)
cb.set_label(r"$\sigma_{t_{\text{r}}}$ [s]")
cs = a.contour(
    dt_grid,
    noise_grid,
    std_tp[:, :, 0].T,
    [tp_tol,],
    linewidths=3.0 * setupplots.lw,
    colors='g'
)
cb.add_lines(cs)
a.set_xscale('log')
a.set_yscale('log')
a.set_title(r"Uncertainty in $t_{\text{r}}$")
a.set_xlabel(r'$\Delta t$ [s]')
a.set_ylabel(r'$u$')
a.set_xlim(dt_grid[0], dt_grid[-1])
a.set_ylim(noise_grid[0], noise_grid[-1])

# Set up the histogram plot:
f_hist = plt.figure()
a_hist = f_hist.add_subplot(1, 1, 1)

def on_click_tp(event):
    if f.canvas.manager.toolbar._active is not None:
        return
    if event.xdata is None or event.ydata is None:
        return
    
    print("Updating...")
    npts_idx = 0
    # Find out what dt, n are:
    dt_idx = profiletools.get_nearest_idx(event.xdata, dt_grid)
    noise_idx = profiletools.get_nearest_idx(event.ydata, noise_grid)
    
    print(npts_grid[npts_idx])
    print(dt_grid[dt_idx])
    print(noise_grid[noise_idx])
    
    # Just nuke the data, since we need to re-do it all:
    a_hist.clear()
    # Estimate number of bins:
    y = tp_exp[dt_idx, noise_idx, npts_idx]
    lq, uq = scipy.stats.scoreatpercentile(y, [25, 75])
    h = 2.0 * (uq - lq) / len(y)**(1.0 / 3.0)
    nn = int(scipy.ceil((y.max() - y.min()) / h))
    a_hist.hist(y, bins=nn, normed=True)
    grid = scipy.linspace(y.min(), y.max(), 100)
    a_hist.plot(
        grid,
        scipy.stats.norm.pdf(
            grid,
            loc=mu_tp[dt_idx, noise_idx, npts_idx],
            scale=std_tp[dt_idx, noise_idx, npts_idx]
        )
    )
    a_hist.plot(
        grid,
        scipy.stats.norm.pdf(
            grid,
            loc=scipy.nanmean(y),
            scale=scipy.nanstd(y, ddof=1)
        )
    )
    a_hist.set_xlim(0.008, 0.014)
    f_hist.canvas.draw()
    
    print("Done!")

f.canvas.mpl_connect("button_press_event", on_click_tp)
setupplots.apply_formatter(f)
f.savefig("tpTolExp.pdf", bbox_inches='tight')
f.savefig("tpTolExp.pgf", bbox_inches='tight')

# Plot of observed sigma_tauimp:
f = plt.figure(figsize=(setupplots.TEXTWIDTH, setupplots.TEXTWIDTH / 1.618))
gs = mplgs.GridSpec(2, 3, width_ratios=[5, 5, 10.0 / 15.0])
a = []
k = 0
j = 0
for i, npts in enumerate(npts_grid):
    a.append(f.add_subplot(gs[j, k], sharex=a[0] if i > 0 else None, sharey=a[0] if i > 0 else None))
    pcm = a[-1].pcolormesh(
        dt_grid_plot,
        noise_grid_plot,
        std_tauimp[:, :, i].T,
        vmin=scipy.nanmin(std_tauimp),
        vmax=scipy.nanmax(std_tauimp[:, :, -1]),
        norm=LogNorm()
    )
    pcm.cmap.set_over('white')
    cs = a[-1].contour(
        dt_grid,
        noise_grid,
        std_tauimp[:, :, i].T,
        [tauimp_tol,],
        linewidths=3.0 * setupplots.lw,
        colors='g'
    )
    a[-1].set_xscale('log')
    a[-1].set_yscale('log')
    a[-1].set_title("1 point" if npts == 1 else "%d points" % (npts,))
    if i < 2:
        plt.setp(a[-1].get_xticklabels(), visible=False)
    else:
        a[-1].set_xlabel(r'$\Delta t$ [s]')
    if i % 2 != 0:
        plt.setp(a[-1].get_yticklabels(), visible=False)
    else:
        a[-1].set_ylabel(r'$u$')
    
    k += 1
    if k == 2:
        k = 0
        j += 1

a[0].set_xlim(dt_grid[0], dt_grid[-1])
a[0].set_ylim(noise_grid[0], noise_grid[-1])
f.suptitle(r"Uncertainty in $\tau_{\text{imp}}$")
cax = f.add_subplot(gs[:, 2])
cb = plt.colorbar(pcm, cax=cax, extend='max')
cb.add_lines(cs)
cb.set_label(r"$\sigma_{\tau_{\text{imp}}}$ [s]")
f.subplots_adjust(left=0.12, bottom=0.14, right=0.88, top=0.87, wspace=0.22, hspace=0.22)

# Set up the histogram plot:
f_hist = plt.figure()
a_hist = f_hist.add_subplot(1, 1, 1)

def on_click_tauimp(event):
    if f.canvas.manager.toolbar._active is not None:
        return
    if event.xdata is None or event.ydata is None:
        return
    
    print("Updating...")
    # Find out what axis we are in:
    for npts_idx in xrange(0, len(a)):
        if event.inaxes == a[npts_idx]:
            break
    else:
        # Not in any axes...do nothing!
        return
    # Find out what dt, n are:
    dt_idx = profiletools.get_nearest_idx(event.xdata, dt_grid)
    noise_idx = profiletools.get_nearest_idx(event.ydata, noise_grid)
    
    print(npts_grid[npts_idx])
    print(dt_grid[dt_idx])
    print(noise_grid[noise_idx])
    
    # Just nuke the data, since we need to re-do it all:
    a_hist.clear()
    # Estimate number of bins:
    y = tauimp_exp[dt_idx, noise_idx, npts_idx]
    lq, uq = scipy.stats.scoreatpercentile(y, [25, 75])
    h = 2.0 * (uq - lq) / len(y)**(1.0 / 3.0)
    nn = int(scipy.ceil((y.max() - y.min()) / h))
    a_hist.hist(y, bins=nn, normed=True)
    grid = scipy.linspace(y.min(), y.max(), 100)
    a_hist.plot(
        grid,
        scipy.stats.norm.pdf(
            grid,
            loc=mu_tauimp[dt_idx, noise_idx, npts_idx],
            scale=std_tauimp[dt_idx, noise_idx, npts_idx]
        )
    )
    a_hist.plot(
        grid,
        scipy.stats.norm.pdf(
            grid,
            loc=scipy.nanmean(y),
            scale=scipy.nanstd(y, ddof=1)
        )
    )
    a_hist.set_xlim(0.023, 0.026)
    f_hist.canvas.draw()
    
    print("Done!")

f.canvas.mpl_connect("button_press_event", on_click_tauimp)
setupplots.apply_formatter(f)
f.savefig("tauimpTolExp.pdf")
f.savefig("tauimpTolExp.pgf")

# Plot of tauimp:
f = plt.figure(figsize=(0.75 * setupplots.TEXTWIDTH, 0.75 * setupplots.TEXTWIDTH / 1.618))
a = f.add_subplot(1, 1, 1)
pcm = a.pcolormesh(
    D_grid_plot,
    V_grid_plot,
    tauimp * 1e3
)
cb = f.colorbar(pcm)
cb.set_label(r"$\tau_{\text{imp}}$ [ms]")
a.contour(D_grid, V_grid, tauimp, 50, alpha=0.5, linewidths=setupplots.lw / 2.0, colors='w')
a.set_title("Impurity confinement time")
a.set_xlabel(r"$D$ [$\si{m^2/s}$]")
a.set_ylabel(r"$V$ [$\si{m/s}$]")
a.set_xlim(D_grid[0], D_grid[-1])
a.set_ylim(V_grid[0], V_grid[-1])
a.xaxis.set_major_locator(plt.MaxNLocator(nbins=7))
setupplots.apply_formatter(f)
f.savefig("tauimpZoom.pdf", bbox_inches='tight')
f.savefig("tauimpZoom.pgf", bbox_inches='tight')

# Plot of t_p:
f = plt.figure(figsize=(0.75 * setupplots.TEXTWIDTH, 0.75 * setupplots.TEXTWIDTH / 1.618))
a = f.add_subplot(1, 1, 1)
pcm = a.pcolormesh(
    D_grid_plot,
    V_grid_plot,
    tp * 1e3
)
cb = f.colorbar(pcm)
cb.set_label(r"$t_{\text{r}}$ [ms]")
a.contour(D_grid, V_grid, tp, 50, alpha=0.5, linewidths=setupplots.lw / 2.0, colors='w')
a.set_title("Core rise time")
a.set_xlabel(r"$D$ [$\si{m^2/s}$]")
a.set_ylabel(r"$V$ [$\si{m/s}$]")
a.set_xlim(D_grid[0], D_grid[-1])
a.set_ylim(V_grid[0], V_grid[-1])
a.xaxis.set_major_locator(plt.MaxNLocator(nbins=7))
setupplots.apply_formatter(f)
f.savefig("tpZoom.pdf", bbox_inches='tight')
f.savefig("tpZoom.pgf", bbox_inches='tight')

# Plot of broadness:
f = plt.figure(figsize=(0.75 * setupplots.TEXTWIDTH, 0.75 * setupplots.TEXTWIDTH / 1.618))
a = f.add_subplot(1, 1, 1)
pcm = a.pcolormesh(
    D_grid_plot,
    V_grid_plot,
    b
)
cb = f.colorbar(pcm)
cb.set_label(r"$b_{0.75}$")
a.contour(D_grid, V_grid, b, 50, alpha=0.5, linewidths=setupplots.lw / 2.0, colors='w', linestyles='-')
a.set_title("Impurity density profile broadness")
a.set_xlabel(r"$D$ [$\si{m^2/s}$]")
a.set_ylabel(r"$V$ [$\si{m/s}$]")
a.set_xlim(D_grid[0], D_grid[-1])
a.set_ylim(V_grid[0], V_grid[-1])
a.xaxis.set_major_locator(plt.MaxNLocator(nbins=7))
setupplots.apply_formatter(f)
f.savefig("bZoom.pdf", bbox_inches='tight')
f.savefig("bZoom.pgf", bbox_inches='tight')

# Combined plot:
f = plt.figure(figsize=(0.75 * setupplots.TEXTWIDTH, 0.75 * setupplots.TEXTWIDTH / 1.618))
a = f.add_subplot(1, 1, 1)
# Truth contours:
a.contour(D_grid, V_grid, tauimp, [tauimp0,], colors='b')
blue_line = mlines.Line2D([], [], color='b', label=r'$\tau_{\text{imp}}$')
a.contour(D_grid, V_grid, tp, [tp0,], colors='g', linestyles='--')
green_line = mlines.Line2D([], [], color='g', linestyle='--', label=r'$t_{\text{r}}$')
a.contour(D_grid, V_grid, b, [b0,], colors='r', linestyles=':')
red_line = mlines.Line2D([], [], color='r', linestyle=':', label=r'$b_{0.75}$')
# Tolerance contours:
a.contourf(
    D_grid,
    V_grid,
    #tauimp0 + dtauimpdD * (DD - D0) + dtauimpdV * (VV - V0),
    tauimp,
    [tauimp0 - 2 * tauimp_tol, tauimp0 + 2 * tauimp_tol],
    colors='b',
    alpha=0.25
)
a.contourf(
    D_grid,
    V_grid,
    # tp0 + dtpdD * (DD - D0) + dtpdV * (VV - V0),
    tp,
    [tp0 - 2 * tp_tol, tp0 + 2 * tp_tol],
    colors='g',
    alpha=0.25
)
a.contourf(
    D_grid,
    V_grid,
    # b0 + dbdD * (DD - D0) + dbdV * (VV - V0),
    b,
    [b0 - 2 * b_tol, b0 + 2 * b_tol],
    colors='r',
    alpha=0.25
)
# Linear tolerance contours:
# a.contour(D_grid, V_grid, tauimp0 + dtauimpdD * (DD - D0) + dtauimpdV * (VV - V0), [tauimp0,], colors='y', linestyles='-')
# a.contour(D_grid, V_grid, tp0 + dtpdD * (DD - D0) + dtpdV * (VV - V0), [tp0,], colors='y', linestyles='--')
# a.contour(D_grid, V_grid, b0 + dbdD * (DD - D0) + dbdV * (VV - V0), [b0,], colors='y', linestyles=':')
# a.contourf(
#     D_grid,
#     V_grid,
#     tauimp0 + dtauimpdD * (DD - D0) + dtauimpdV * (VV - V0),
#     [tauimp0 - 2 * tauimp_tol, tauimp0 + 2 * tauimp_tol],
#     colors='y',
#     alpha=0.25
# )
# a.contourf(
#     D_grid,
#     V_grid,
#     tp0 + dtpdD * (DD - D0) + dtpdV * (VV - V0),
#     [tp0 - 2 * tp_tol, tp0 + 2 * tp_tol],
#     colors='y',
#     alpha=0.25
# )
# a.contourf(
#     D_grid,
#     V_grid,
#     b0 + dbdD * (DD - D0) + dbdV * (VV - V0),
#     [b0 - 2 * b_tol, b0 + 2 * b_tol],
#     colors='y',
#     alpha=0.25
# )
# Confidence ellipse:
ell = mplp.Ellipse(
    [mu_D_typ, mu_V_typ],
    a_typ,
    b_typ,
    angle=scipy.degrees(alpha_typ),
    facecolor='m',
    edgecolor='k',
    alpha=0.75,
    label=r'95\% \textsc{pr}',
    zorder=100
)
# a.axhspan(V0 - 2 * std_V_typ, V0 + 2 * std_V_typ, facecolor='m', alpha=0.125)
# a.axvspan(D0 - 2 * std_D_typ, D0 + 2 * std_D_typ, facecolor='m', alpha=0.125)
a.add_artist(ell)
a.plot(D0, V0, 'wo', markersize=setupplots.ms, label='true', zorder=101)
a.set_title(r"Intersection of $\textcolor{MPLb}{\tau_{\text{imp}}}$, $\textcolor{MPLg}{t_{\text{r}}}$ and $\textcolor{MPLr}{b_{0.75}}$")
a.set_xlabel(r"$D$ [$\si{m^2/s}$]")
a.set_ylabel(r"$V$ [$\si{m/s}$]")
a.set_xlim(D_grid[0], D_grid[-1])
a.set_ylim(V_grid[0], V_grid[-1])
leg = a.legend(handles=a.get_legend_handles_labels()[0] + [blue_line, green_line, red_line, ell], loc='lower left', numpoints=1)
leg.set_zorder(105)
setupplots.apply_formatter(f)
f.savefig("OverlayZoom.pdf", bbox_inches='tight')
f.savefig("OverlayZoom.pgf", bbox_inches='tight')

# Linearization goodness:
# tauimp:
f = plt.figure(figsize=(0.75 * setupplots.TEXTWIDTH, 0.75 * setupplots.TEXTWIDTH / 1.618))
a = f.add_subplot(1, 1, 1)
d = (tauimp0 + dtauimpdD * (DD - D0) + dtauimpdV * (VV - V0) - tauimp) / tauimp * 100
pcm = a.pcolormesh(
    D_grid_plot,
    V_grid_plot,
    d,
    vmin=-scipy.absolute(d).max(),
    vmax=scipy.absolute(d).max(),
    cmap='seismic'
)
cb = f.colorbar(pcm)
cb.set_label(r"$100\%\cdot(\hat{\tau}_{\text{imp}}-\tau_{\text{imp}})/\tau_{\text{imp}}$")
a.contour(
    D_grid,
    V_grid,
    tauimp,
    [tauimp0,],
    colors='g'
)
a.contour(
    D_grid,
    V_grid,
    tauimp,
    [0.9 * tauimp0, 1.1 * tauimp0],
    colors='k',
    linestyles=':'
)
a.set_xlabel(r"$D$ [$\si{m^2/s}$]")
a.set_ylabel(r"$V$ [m/s]")
a.set_title(r"Percent error in linearization of $\tau_{\text{imp}}$")
a.set_xlim(D_grid[0], D_grid[-1])
a.set_ylim(V_grid[0], V_grid[-1])
a.xaxis.set_major_locator(plt.MaxNLocator(nbins=7))
setupplots.apply_formatter(f)
f.savefig("tauimpLin.pdf", bbox_inches='tight')
f.savefig("tauimpLin.pgf", bbox_inches='tight')

# tp:
f = plt.figure(figsize=(0.75 * setupplots.TEXTWIDTH, 0.75 * setupplots.TEXTWIDTH / 1.618))
a = f.add_subplot(1, 1, 1)
d = (tp0 + dtpdD * (DD - D0) + dtpdV * (VV - V0) - tp) / tp * 100
pcm = a.pcolormesh(
    D_grid_plot,
    V_grid_plot,
    d,
    vmin=-scipy.absolute(d).max(),
    vmax=scipy.absolute(d).max(),
    cmap='seismic'
)
cb = f.colorbar(pcm)
cb.set_label(r"$100\%\cdot(\hat{t}_{\text{r}}-t_{\text{r}})/t_{\text{r}}$")
a.contour(
    D_grid,
    V_grid,
    tp,
    [tp0,],
    colors='g'
)
a.contour(
    D_grid,
    V_grid,
    tp,
    [0.9 * tp0, 1.1 * tp0],
    colors='k',
    linestyles=':'
)
a.set_xlabel(r"$D$ [$\si{m^2/s}$]")
a.set_ylabel(r"$V$ [m/s]")
a.set_title(r"Percent error in linearization of $t_{\text{r}}$")
a.set_xlim(D_grid[0], D_grid[-1])
a.set_ylim(V_grid[0], V_grid[-1])
a.xaxis.set_major_locator(plt.MaxNLocator(nbins=7))
setupplots.apply_formatter(f)
f.savefig("tpLin.pdf", bbox_inches='tight')
f.savefig("tpLin.pgf", bbox_inches='tight')

# b:
f = plt.figure(figsize=(0.75 * setupplots.TEXTWIDTH, 0.75 * setupplots.TEXTWIDTH / 1.618))
a = f.add_subplot(1, 1, 1)
d = (b0 + dbdD * (DD - D0) + dbdV * (VV - V0) - b) / b * 100
pcm = a.pcolormesh(
    D_grid_plot,
    V_grid_plot,
    d,
    vmin=-scipy.absolute(d).max(),
    vmax=scipy.absolute(d).max(),
    cmap='seismic'
)
cb = f.colorbar(pcm)
cb.set_label(r"$100\%\cdot(\hat{b}_{0.75}-b_{0.75})/b_{0.75}$")
a.contour(
    D_grid,
    V_grid,
    b,
    [b0,],
    colors='g'
)
a.contour(
    D_grid,
    V_grid,
    b,
    [0.9 * b0, 1.1 * b0],
    colors='k',
    linestyles=':'
)
a.set_xlabel(r"$D$ [$\si{m^2/s}$]")
a.set_ylabel(r"$V$ [m/s]")
a.set_title(r"Percent error in linearization of $b_{0.75}$")
a.set_xlim(D_grid[0], D_grid[-1])
a.set_ylim(V_grid[0], V_grid[-1])
a.xaxis.set_major_locator(plt.MaxNLocator(nbins=7))
setupplots.apply_formatter(f)
f.savefig("bLin.pdf", bbox_inches='tight')
f.savefig("bLin.pgf", bbox_inches='tight')

# Plot of the distribution of b:
f = plt.figure(figsize=(0.75 * setupplots.TEXTWIDTH, 0.75 * setupplots.TEXTWIDTH / 1.618))
a = f.add_subplot(1, 1, 1)
a.hist(
    b_samp,
    bins=setupplots.hist_bins(b_samp),
    normed=True,
    histtype='stepfilled',
    label='Monte Carlo',
)
a.plot(
    b_grid,
    scipy.stats.norm.pdf(b_grid, loc=b0, scale=std_b_lin),
    'g',
    label='linearized',
    lw=3.0 * setupplots.lw
)
a.plot(b_grid, p_b, 'r--', label='exact', lw=3.0 * setupplots.lw)
a.set_yscale('log')
a.set_ylim(1e-4, 1e1)
a.legend(loc='lower center')
a.set_xlabel('$b_{0.75}$')
a.set_ylabel('$f_{b_{0.75}}(b_{0.75})$')
a.set_title(r"\textsc{pdf} of $b_{0.75}$")
setupplots.apply_formatter(f)
f.savefig("bDist.pdf", bbox_inches='tight')
f.savefig("bDist.pgf", bbox_inches='tight')

# Plot of mean and median distributions:
f = plt.figure(figsize=(setupplots.TEXTWIDTH, setupplots.TEXTWIDTH / 1.618))
a = []
for i, (ms, mds, N) in enumerate(zip(mean_samps, median_samps, n_pts)):
    a.append(f.add_subplot(2, 2, i + 1))#, sharex=a[0] if i > 0 else None))
    a[-1].hist(
        ms,
        bins=setupplots.hist_bins(ms),
        normed=True,
        label='mean',
        histtype='stepfilled',
        color='b',
        alpha=0.5
    )
    a[-1].hist(
        mds,
        bins=setupplots.hist_bins(mds),
        normed=True,
        label='median',
        histtype='stepfilled',
        color='g',
        alpha=0.5
    )
    grid = scipy.linspace(min(ms.min(), mds.min()), max(ms.max(), mds.max()), 1000)
    ln, = a[-1].plot(
        grid,
        scipy.stats.norm.pdf(grid, loc=b0, scale=std_b_lin / scipy.sqrt(N)),
        'r:',
        label=r'$\sigma_{b_{0.75}}/\sqrt{N}$',
        lw=2.0 * setupplots.lw
    )
    a[-1].set_title("$10^{%d}$ points" % (scipy.log10(N),))
    if i % 2 == 0:
        a[-1].set_ylabel("$f_{b_{0.75}}(b_{0.75})$")
    if i >= 2:
        a[-1].set_xlabel("$b_{0.75}$")
    a[-1].xaxis.set_major_locator(plt.MaxNLocator(nbins=5))
f.suptitle(r"Evolution of the \textcolor{MPLb}{mean} and \textcolor{MPLg}{median} distributions for $b_{0.75}$")
f.subplots_adjust(left=0.125, bottom=0.13, right=0.96, top=0.87, hspace=0.39)
# a[0].legend(loc='upper right')
setupplots.apply_formatter(f)
f.savefig("meanbDist.pdf")
f.savefig("meanbDist.pgf")

f_leg = plt.figure()
blue_patch = mplp.Patch(color='b', label='mean', alpha=0.5)
green_patch = mplp.Patch(color='g', label='median', alpha=0.5)
l = f_leg.legend(
    [ln, blue_patch, green_patch],
    [ln.get_label(), blue_patch.get_label(), green_patch.get_label()],
    ncol=3,
    loc='center'
)
f_leg.canvas.draw()
f_leg.set_figwidth(l.get_window_extent().width / 72)
f_leg.set_figheight(l.get_window_extent().height / 72)
f_leg.savefig("legMeanbDist.pdf", bbox_inches='tight')
f_leg.savefig("legMeanbDist.pgf", bbox_inches='tight')
