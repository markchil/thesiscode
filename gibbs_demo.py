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

# This script makes figure B.9, which shows the Gibbs covariance kernel with
# tanh covariance length scale function and its derivatives.

from __future__ import division
import setupplots
setupplots.thesis_format()
import scipy
import matplotlib.pyplot as plt
plt.ion()
import gptools

X = scipy.atleast_2d(scipy.linspace(0.0, 1.5, 400)).T
zeros = scipy.zeros_like(X)
ones = scipy.ones_like(X)

xi_vals = [0.0, 0.5, 1.0, 1.2]

# Gibbs+tanh:
f = plt.figure(figsize=(setupplots.TEXTWIDTH, setupplots.TEXTWIDTH / 1.618))
k = gptools.GibbsKernel1dTanh(initial_params=[1.0, 1.0, 0.5, 0.02, 1.0])
for i, xi in enumerate(xi_vals):
    k_00 = k(zeros + xi, X, zeros, zeros)
    k_01 = k(zeros + xi, X, zeros, ones)
    k_10 = k(zeros + xi, X, ones, zeros)
    k_11 = k(zeros + xi, X, ones, ones)
    l = k.l_func(X.ravel(), 0, *k.params[1:])
    
    a = f.add_subplot(2, 2, i + 1)
    c, = a.plot(X.ravel(), k_00, label='$n_i=0$, $n_j=0$', lw=3)
    a.axvline(xi, color=c.get_color())
    a.plot(X.ravel(), k_01, '--', label='$n_i=0$, $n_j=1$')
    a.plot(X.ravel(), k_10, '--', label='$n_i=1$, $n_j=0$')
    a.plot(X.ravel(), k_11, '--', label='$n_i=1$, $n_j=1$')
    a.plot(X.ravel(), l, ':', label=r'$\ell(x)$', lw=3)
    
    a.set_title(r'$x_i/\ell_1=%.1f$' % (xi,))
    a.set_ylim([-2, 3])
    a.set_xlim([0, 1.5])
    
    if i in (1, 3):
        plt.setp(a.get_yticklabels(), visible=False)
    else:
        a.set_ylabel(r'$k_{\mathrm{G}}/\sigma_f^2$')
    
    if i in (0, 1):
        plt.setp(a.get_xticklabels(), visible=False)
    else:
        a.set_xlabel(r'$x_j/\ell_1$')

f.subplots_adjust(wspace=0.1)
setupplots.apply_formatter(f)
f.savefig("Gtanh.pgf", bbox_inches='tight')
f.savefig("Gtanh.pdf", bbox_inches='tight')

f_leg = plt.figure()
l = f_leg.legend(*a.get_legend_handles_labels(), ncol=2, loc='center')
f_leg.canvas.draw()
f_leg.set_figwidth(l.get_window_extent().width / 72)
f_leg.set_figheight(l.get_window_extent().height / 72)
f_leg.savefig("gibbs_kernel_1d_legend.pgf", bbox_inches='tight')
f_leg.savefig("gibbs_kernel_1d_legend.pdf", bbox_inches='tight')
