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

# This script makes figures B.1, B.3 and B.5, which show various 1d covariance
# kernels and their derivatives.

from __future__ import division
import scipy
import setupplots
setupplots.thesis_format()
import matplotlib.pyplot as plt
plt.ion()
import gptools

X = scipy.atleast_2d(scipy.linspace(0, 3, 100)).T
zeros = scipy.zeros_like(X)
ones = scipy.ones_like(X)
twos = 2.0 * ones

kernels = [
    gptools.SquaredExponentialKernel(initial_params=[1.0, 1.0]),
    gptools.RationalQuadraticKernel(initial_params=[1.0, 0.5, 1.0]),
    gptools.MaternKernel(initial_params=[1.0, 0.5, 1.0])
]
names = [r'\textsc{se}', r'\textsc{rq}', 'M']
tnames = ['SE', 'RQ', 'M']
long_names = ['squared exponential', 'rational quadratic', r'Mat\'ern']
aux_params = [[None,], [0.5, 2.0, 10.0], [0.5, 1.5, 2.5]]
aux_param_names = [None, r'\alpha', r'\nu']

for k, n, tn, ln, ap, apn in zip(kernels, names, tnames, long_names, aux_params, aux_param_names):
    f = plt.figure(figsize=(setupplots.TEXTWIDTH, setupplots.TEXTWIDTH / 1.618))
    if len(ap) > 1:
        a_comp = f.add_subplot(2, 2, 4)
    for i, p in enumerate(ap):
        if p is not None:
            k.params[1] = p
            a_comp.plot(X.ravel(), k(zeros, X, zeros, zeros), label=r'$%s=%.1f$' % (apn, p))
        if tn == 'SE':
            a = f.add_subplot(1, 1, 1)
        else:
            a = f.add_subplot(2, 2, i + 1)
        a.plot(X.ravel(), k(zeros, X, zeros, zeros), label='$n_i=0$, $n_j=0$', lw=3)
        if n != 'M' or (n == 'M' and p > 1):
            a.plot(X.ravel(), k(zeros, X, zeros, ones), '--', label='$n_i=0$, $n_j=1$')
            a.plot(X.ravel(), k(zeros, X, ones, zeros), '--', label='$n_i=1$, $n_j=0$')
            a.plot(X.ravel(), k(zeros, X, ones, ones), '--', label='$n_i=1$, $n_j=1$')
        if n != 'M' or (n == 'M' and p > 2):
            a.plot(X.ravel(), k(zeros, X, zeros, twos), ':', label='$n_i=0$, $n_j=2$; $n_i=2$, $n_j=0$')
            a.plot(X.ravel(), k(zeros, X, ones, twos), ':', label='$n_i=1$, $n_j=2$')
            a.plot(X.ravel(), k(zeros, X, twos, ones), ':', label='$n_i=2$, $n_j=1$')
            a.plot(X.ravel(), k(zeros, X, twos, twos), ':', label='$n_i=2$, $n_j=2$')
        if tn == 'SE' or i in (2, 3):
            a.set_xlabel(r'$x/\ell$')
        elif tn != 'SE':
            plt.setp(a.get_xticklabels(), visible=False)
        if tn == 'SE' or i in (0, 2):
            a.set_ylabel(r'$k_{\mathrm{%s}}/\sigma_f^2$' % (n,))
        if apn is not None:
            title = r"$k_{\mathrm{%s}}(0, x)$, $%s=%.1f$" % (n, apn, p)
        else:
            title = r"$k_{\mathrm{%s}}(0, x)$" % (n,)
        a.set_title(title)
        a.set_ylim([-2, 3])
        fn = '%s' % (tn,) if len(ap) == 1 else '%s%s%03d' % (tn, apn[1:], p * 10)
    if len(ap) > 1:
        a_comp.plot(X.ravel(), kernels[0](zeros, X, zeros, zeros), '--', label=r'\textsc{se} ($%s=\infty$)' % (apn,))
        a_comp.legend(loc='best', fontsize=7)
        a_comp.set_xlabel(r'$x/\ell$')
        a_comp.set_title(r"Effect of $%s$ on $k_{\mathrm{%s}}(0, x)$" % (apn, n))
        f.subplots_adjust(left=0.11, bottom=0.13, right=0.98, top=0.92, wspace=0.14, hspace=0.25)
    else:
        f.subplots_adjust(left=0.11, bottom=0.13, right=0.97)
    setupplots.apply_formatter(f)
    f.savefig('%s.pgf' % (tn,))
    f.savefig('%s.pdf' % (tn,))

f_leg = plt.figure()
l = f_leg.legend(*a.get_legend_handles_labels(), ncol=2, loc='center')
f_leg.canvas.draw()
f_leg.set_figwidth(l.get_window_extent().width / 72)
f_leg.set_figheight(l.get_window_extent().height / 72)
f_leg.savefig("kernel_1d_legend_3.pgf", bbox_inches='tight')
f_leg.savefig("kernel_1d_legend_3.pdf", bbox_inches='tight')
