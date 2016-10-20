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

# This script makes figures B.2, B.4, B.6, B.7 and B.8, which show various 2d
# covariance kernels and their derivatives.

from __future__ import division
import setupplots
setupplots.thesis_format()
import scipy
import matplotlib
pgf_with_latex_tiny = {
    "axes.labelsize": 11,
    "font.size": 9,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.titlesize": 11,
    "figure.titlesize": 11,
}
matplotlib.rcParams.update(pgf_with_latex_tiny)
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
plt.ion()
import gptools

x = scipy.linspace(-3, 3, 51)
zeros = scipy.zeros_like(x)
ones = scipy.ones_like(x)
twos = 2.0 * ones

X, Y = scipy.meshgrid(x, x)
ZEROS, dum = scipy.meshgrid(zeros, zeros)
ONES, dum = scipy.meshgrid(ones, ones)
TWOS, dum = scipy.meshgrid(twos, twos)

shape = X.shape

X = X.reshape((-1, 1))
Y = Y.reshape((-1, 1))
ZEROS = ZEROS.reshape((-1, 1))
ONES = ONES.reshape((-1, 1))
TWOS = TWOS.reshape((-1, 1))

XY = scipy.hstack((X, Y))
ZZ = scipy.hstack((ZEROS, ZEROS))
OZ = scipy.hstack((ONES, ZEROS))
OO = scipy.hstack((ONES, ONES))
ZO = scipy.hstack((ZEROS, ONES))

def n2N(nix, njx, niy, njy, l):
    """Constructs the arguments `Ni`, `Nj` given the values to put in each column and a length.
    """
    Ni = scipy.zeros((l, 2))
    Nj = scipy.zeros((l, 2))
    Ni[:, 0] = nix
    Ni[:, 1] = niy
    Nj[:, 0] = njx
    Nj[:, 1] = njy
    
    return Ni, Nj

kernels = [
    gptools.SquaredExponentialKernel(num_dim=2, initial_params=[1.0, 1.0, 1.0]),
    gptools.RationalQuadraticKernel(num_dim=2, initial_params=[1.0, 0.5, 1.0, 1.0]),
    gptools.MaternKernel(num_dim=2, initial_params=[1.0, 0.5, 1.0, 1.0]),
]
names = [r'\textsc{se}', r'\textsc{rq}', 'M']
tnames = ['SE', 'RQ', 'M']
aux_params = [[None,], [0.5,], [0.5, 1.5, 2.5]]
aux_param_names = [None, r'\alpha', r'\nu']

for k, n, tn, ap, apn in zip(kernels, names, tnames, aux_params, aux_param_names):
    for p in ap:
        if n == 'M':
            if p <= 1:
                ncol = 1
                w = 6.4 / 3
            elif p <= 2:
                ncol = 3
                w = 6.4
            else:
                ncol = 4
                w = 6.4
        else:
            ncol = 4
            w = 6.4
        pi = 1
        f = plt.figure(figsize=(w, w))
        if p is not None:
            k.params[1] = p
        for niy in (0, 1):
            for nix in (0, 1):
                for njy in (0, 1):
                    for njx in (0, 1):
                        if n != 'M' or (n == 'M' and p > (nix + niy) and p > (njx + njy)):
                            k_val = k(ZZ, XY, *n2N(nix, njx, niy, njy, len(X)))
                            a = f.add_subplot(ncol, ncol, pi, projection='3d')
                            surf = a.plot_surface(
                                X.reshape(shape),
                                Y.reshape(shape),
                                k_val.reshape(shape),
                                rstride=1,
                                cstride=1,
                                cmap='seismic',
                                linewidth=0,
                                vmin=-1.0,
                                vmax=1.0
                            )
                            # plt.setp(a.get_xticklabels(), visible=False)
                            # plt.setp(a.get_yticklabels(), visible=False)
                            # plt.setp(a.get_zticklabels(), visible=False)
                            a.set_xlabel(r'$x$', labelpad=-10)
                            a.set_ylabel(r'$y$', labelpad=-10)
                            istr = ''
                            for xv in range(0, nix):
                                istr += 'x'
                            for yv in range(0, niy):
                                istr += 'y'
                            jstr = ''
                            for xv in range(0, njx):
                                jstr += 'x'
                            for xv in range(0, njy):
                                jstr += 'y'
                            a.set_title(r"$\cov[z_{%s}(0, 0), z_{%s}(x, y)]$" % (istr, jstr), y=0.95)
                            for tick in a.get_yaxis().get_major_ticks():
                                tick.set_pad(-5)
                                tick.label1 = tick._get_text1()
                            for tick in a.get_xaxis().get_major_ticks():
                                tick.set_pad(-5)
                                tick.label1 = tick._get_text1()
                            for tick in a.zaxis.get_major_ticks():
                                tick.set_pad(-2)
                                tick.label1 = tick._get_text1()
                            pi += 1
        f.subplots_adjust(left=0, right=0.98, top=0.99, bottom=0.01, wspace=0.05, hspace=0.05)
        f.canvas.draw()
        if apn is None:
            # f.savefig("%s2D.svg" % (tn,))
            f.savefig("%s2D.pdf" % (tn,))#, bbox_inches='tight')
            f.savefig("%s2D.pgf" % (tn,))#, bbox_inches='tight')
        else:
            f.savefig("%s%02d2D.pdf" % (tn, int(p * 10)))#, bbox_inches='tight')
            f.savefig("%s%02d2D.pgf" % (tn, int(p * 10)))#, bbox_inches='tight')
