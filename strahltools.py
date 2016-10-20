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

# This package is used to post-process results from the old IDL-based STRAHL
# workflow. It is used by strahl_compare_post_new.py.

from __future__ import division

import os.path
import scipy
import scipy.io
import scipy.interpolate
import gptools
import matplotlib.pyplot as plt
plt.ion()

class STRAHLResult(object):
    def __init__(self, filename=None, **kwargs):
        self.chisqd = None
        self.roasave = None
        self.D_results = None
        self.V_results = None
        self.D_hi = None
        self.V_lo = None
        self.Rmaj = None
        self.Te = None
        self.ne = None
        
        if filename is not None:
            self.read_strahl_savepoint(filename, **kwargs)
    
    def read_strahl_savepoint(self, filename, suffix=None, dims=(400, 100)):
        # Read the save file:
        f = scipy.io.readsav(filename)
        try:
            self.chisqd = scipy.concatenate((self.chisqd, f.chisqd))
            self.roasave = scipy.vstack((self.roasave, f.roasave))
            self.D_results = scipy.vstack((self.D_results, f.dvresults[:, :, 0]))
            self.V_results = scipy.vstack((self.V_results, f.dvresults[:, :, 1]))
        except ValueError:
            self.chisqd = f.chisqd
            self.roasave = f.roasave
            self.D_results = f.dvresults[:, :, 0]
            self.V_results = f.dvresults[:, :, 1]

        # Just use the value from the most recently-loaded file:
        self.D_hi = float(f.input.dhi[0])
        self.V_lo = float(f.input.vlo[0])
        
        # Read the binary profile files:
        file_dir, file_name = os.path.split(os.path.abspath(filename))
        if suffix is None:
            if file_name.startswith('savepoint_backup'):
                suffix = file_name[len('savepoint_backup'):]
            elif file_name.startswith('save_combined'):
                suffix = file_name[len('save_combined'):]
            else:
                raise ValueError("Cannot parse base filename '%s' to retrieve suffix." % file_name)
        
        nefile = os.path.join(file_dir, 'ne'+suffix+'.dat')
        ne_vals = scipy.reshape(scipy.fromfile(nefile, dtype=scipy.float32), dims).T
        Tefile = os.path.join(file_dir, 'Te'+suffix+'.dat')
        Te_vals = scipy.reshape(scipy.fromfile(Tefile, dtype=scipy.float32), dims).T
        Rmajfile = os.path.join(file_dir, 'Rmaj'+suffix+'.dat')
        Rmaj_vals = scipy.reshape(scipy.fromfile(Rmajfile, dtype=scipy.float32), dims).T
        
        try:
            self.ne = scipy.vstack((self.ne, ne_vals[f.input.tstart:f.input.tstop+1, :]))
            self.Te = scipy.vstack((self.Te, Te_vals[f.input.tstart:f.input.tstop+1, :]))
            self.Rmaj = scipy.vstack((self.Rmaj, Rmaj_vals[f.input.tstart:f.input.tstop+1, :]))
        except ValueError:
            self.ne = ne_vals[f.input.tstart:f.input.tstop+1, :]
            self.Te = Te_vals[f.input.tstart:f.input.tstop+1, :]
            self.Rmaj = Rmaj_vals[f.input.tstart:f.input.tstop+1, :]
    
    def form_profiles(self, grid, k=1, weighted=True, debug_plots=False):
        D_means = []
        V_means = []
        chisqds = []
        valid = ((self.chisqd != 0.0) & (self.chisqd != 1.0) & (self.chisqd != -999))
        for chisqd, roa, D_res, V_res, idx in zip(self.chisqd,
                                                  self.roasave,
                                                  self.D_results,
                                                  self.V_results,
                                                  range(0, len(self.chisqd))):
            if valid[idx]:
                # Use NTH's regular technique:
                # NOTE: NTH technically does this against sqrt(psi_norm).
                D_mean = scipy.interpolate.InterpolatedUnivariateSpline(roa, D_res, k=k)(grid)
                V_mean = scipy.interpolate.InterpolatedUnivariateSpline(roa, V_res, k=k)(grid)
            
                if debug_plots > 1:
                    f = plt.figure()
                    a_D = f.add_subplot(2, 1, 1)
                    a_D.plot(grid, D_mean)
                    a_D.plot(roa, D_res, 'k.', ms=10)
                
                    a_V = f.add_subplot(2, 1, 2)
                    a_V.plot(grid, V_mean)
                    if use_gp:
                        a_V.fill_between(grid, V_mean - scipy.sqrt(V_var), V_mean + scipy.sqrt(V_var), alpha=0.5)
                    a_V.plot(roa, V_res, 'k.', ms=10)
            
                if (D_mean.max() >= 1.2 * self.D_hi or
                    V_mean.min() <= 1.2 * self.V_lo or
                    scipy.isnan(D_mean).any() or
                    scipy.isnan(V_mean).any()):
                    valid[idx] = False
                    print("Rejecting a sample!")
                    if debug_plots > 1:
                        f.suptitle("REJECTED!")
                        f.canvas.draw()
                else:
                    D_means.append(D_mean)
                    V_means.append(V_mean)
                    chisqds.append(chisqd)
        self.valid_idxs = scipy.where(valid)[0]
        print("Kept %d out of %d samples." % (len(chisqds), len(self.chisqd),))
        self.D_means = scipy.asarray(D_means)
        self.V_means = scipy.asarray(V_means)
        self.grid = grid
        
    def compute_D_and_V(self, grid=None, stop_idx=None, **kwargs):
        if grid is not None:
            self.form_profiles(grid, **kwargs)
        try:
            chisqds = self.chisqd[self.valid_idxs]
        except AttributeError:
            if grid is None:
                grid = scipy.linspace(0, 1, 101)
            self.form_profiles(grid, **kwargs)
            chisqds = self.chisqd[self.valid_idxs]
        if kwargs.get('weighted', True):
            weights = chisqd2weight(chisqds)
        else:
            weights = scipy.ones_like(chisqds)
        
        D = weighted_mean(self.D_means[:stop_idx, :], weights[:stop_idx], axis=0)
        V = weighted_mean(self.V_means[:stop_idx, :], weights[:stop_idx], axis=0)
        s_D = weighted_std(self.D_means[:stop_idx, :], weights[:stop_idx], axis=0, ddof=1)
        s_V = weighted_std(self.V_means[:stop_idx, :], weights[:stop_idx], axis=0, ddof=1)
        
        if kwargs.get('debug_plots', False):
            f = plt.figure()
            a_D = f.add_subplot(2, 1, 1)
            a_D.plot(self.grid, D)
            a_D.fill_between(self.grid, D - s_D, D + s_D, alpha=0.375)
            
            a_V = f.add_subplot(2, 1, 2)
            a_V.plot(self.grid, V)
            a_V.fill_between(self.grid, V - s_V, V + s_V, alpha=0.375)
        
        return (D, s_D, V, s_V)
    
    def form_convergence(self, **kwargs):
        try:
            max_idx = len(self.valid_idxs)
        except AttributeError:
            self.compute_D_and_V(**kwargs)
            max_idx = len(self.valid_idxs)
        D_conv = scipy.zeros((max_idx, len(self.grid)))
        s_D_conv = scipy.zeros((max_idx, len(self.grid)))
        V_conv = scipy.zeros((max_idx, len(self.grid)))
        s_V_conv = scipy.zeros((max_idx, len(self.grid)))
        for k in xrange(0, max_idx):
            D, s_D, V, s_V = self.compute_D_and_V(stop_idx=1 + k, **kwargs)
            D_conv[k, :] = D
            s_D_conv[k, :] = s_D
            V_conv[k, :] = V
            s_V_conv[k, :] = s_V
        return (D_conv, s_D_conv, V_conv, s_V_conv)

def chisqd2weight(chisqd):
    chisqd = scipy.asarray(chisqd)
    return chisqd.min() / chisqd

def weighted_mean(q, weights, axis=0):
    return scipy.sum(scipy.atleast_2d(weights).T * q, axis=axis) / scipy.sum(weights)

def weighted_var(q, weights, axis=0, ddof=1):
    try:
        return (q.shape[axis] / (q.shape[axis] - ddof) *
                scipy.sum(scipy.atleast_2d(weights).T * (q - weighted_mean(q, weights, axis=axis))**2, axis=axis) /
                scipy.sum(weights))
    except ZeroDivisionError:
        return scipy.nan

def weighted_std(*args, **kwargs):
    return scipy.sqrt(weighted_var(*args, **kwargs))
