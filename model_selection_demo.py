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

# This script makes figures 3.50 and 3.51, which demonstrate model selection
# for polynomial fitting.

from __future__ import division

import scipy
import itertools
RS = scipy.random.RandomState(seed=8675309)

err_y = 5.0

x = scipy.asarray([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
# True model:
p_true = [1.0, 2.0, -5.0, 1.0]
y_true = scipy.polyval(p_true, x)
y_noised = y_true + err_y * RS.randn(len(y_true))

# Dense grid:
x_d = scipy.linspace(x.min() - 0.5, x.max() + 0.5, 500)
y_true_d = scipy.polyval(p_true, x_d)

y_d = []
bic = []
deg = range(1, 6)
res = []

for d in deg:
    p = scipy.polyfit(x, y_noised, d)
    y_d.append(scipy.polyval(p, x_d))
    
    y = scipy.polyval(p, x)
    res.append(y_noised - y)
    rr = scipy.atleast_2d(res[-1])
    ll = (
        -len(x) * scipy.log(2 * scipy.pi) / 2.0 -
        len(x) * scipy.log(err_y) -
        1.0 / (2.0 * err_y**2.0) * rr.dot(rr.T)
    )[0, 0]
    bic.append(ll - len(p) * scipy.log(len(x)) / 2.0)

import setupplots
setupplots.thesis_format()
import matplotlib.pyplot as plt
plt.ion()

f = plt.figure(figsize=(0.75 * setupplots.TEXTWIDTH, 2 * 0.75 * setupplots.TEXTWIDTH / 1.618))
a = f.add_subplot(2, 1, 1)
ls_cycle = itertools.cycle(['--', '-.', ':', '-'])
a.plot(x_d, y_true_d, label='true', lw=3)
for i, d in enumerate(deg):
    a.plot(x_d, y_d[i], ls_cycle.next(), label='$d=%d$' % (d,))
a.errorbar(x, y_noised, yerr=err_y, fmt='go', label='data')
a.legend(loc='best', ncol=2)
a.set_ylabel('$y$')
a.set_title("Cubic data with various polynomial fits")
plt.setp(a.get_xticklabels(), visible=False)

a_res = f.add_subplot(2, 1, 2, sharex=a)
ls_cycle = itertools.cycle(['>--', '^-.', 's:', 'h-', 'D--'])
a_res.plot(x_d, y_true_d - y_true_d, lw=3)
for i, d in enumerate(deg):
    a_res.plot(x, res[i], ls_cycle.next(), ms=2 * setupplots.ms)
a_res.set_xlabel('$x$')
a_res.set_ylabel(r"$y_{\text{true}} - \hat{y}$")
a_res.set_title("Residuals")
a.set_xlim(left=x.min() - 0.5, right=x.max() + 0.5)
a_res.set_ylim(bottom=-10, top=10)
setupplots.apply_formatter(f)
f.savefig("polyfit.pdf", bbox_inches='tight')
f.savefig("polyfit.pgf", bbox_inches='tight')

f = plt.figure(figsize=(0.75 * setupplots.TEXTWIDTH, 2 * 0.75 * setupplots.TEXTWIDTH / 1.618))
a = f.add_subplot(2, 1, 1)
a.plot(deg, bic, 'o--', ms=2 * setupplots.ms)
a_zoom =  f.add_subplot(2, 1, 2)
a_zoom.plot(deg[1:], bic[1:], 'o--', ms=2 * setupplots.ms)
# plt.setp(a.get_xticklabels(), visible=False)
a.set_ylabel(r"$\textsc{bic}\approx\text{log-evidence}$")
a_zoom.set_ylabel(r"$\textsc{bic}\approx\text{log-evidence}$")
a.set_title("Model selection for polynomial fits")
a_zoom.set_title("Zoomed")
a_zoom.set_xlabel("$d$")
a.set_xlim(left=deg[0] - 0.5, right=deg[-1] + 0.5)
a_zoom.set_xlim(left=deg[1] - 0.5, right=deg[-1] + 0.5)
a_zoom.set_ylim(top=-19.0, bottom=-21)
a_zoom.set_xticks(deg[1:])
setupplots.apply_formatter(f)
f.savefig("polyfitBic.pdf", bbox_inches='tight')
f.savefig("polyfitBic.pgf", bbox_inches='tight')
