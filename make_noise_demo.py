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

# This script makes figure 2.4, which illustrates the effect of the error bars
# on the inferred covariance length scale. This script also makes figure 2.7,
# which shows how the GP mean can be expressed as a sum over copies of the
# covariance kernel.

from __future__ import division

import scipy
import gptools

RS = scipy.random.RandomState(seed=8675309)

n1 = 0.1
n2 = 0.5

x = scipy.linspace(0, 1, 8)[1:-1]
y = scipy.sin(x * 2.0 * scipy.pi)
xdens = scipy.linspace(0, 1, 1000)
ydens = scipy.sin(xdens * 2.0 * scipy.pi)
ygdens = 2.0 * scipy.pi * scipy.cos(xdens * 2.0 * scipy.pi)
y2 = y + n2 * RS.randn(len(y))

kSE1 = gptools.SquaredExponentialKernel(param_bounds=[(0, 2.0), (0, 1.0)])
gp1 = gptools.GaussianProcess(kSE1, X=x, y=y2, err_y=n1)
gp1.optimize_hyperparameters()
y_star_1, std_y_star_1 = gp1.predict(xdens)
yg_star_1, std_yg_star_1 = gp1.predict(xdens, n=1)

kSE2 = gptools.SquaredExponentialKernel(param_bounds=[(0, 2.0), (0, 1.0)])
gp2 = gptools.GaussianProcess(kSE2, X=x, y=y2, err_y=n2)
gp2.optimize_hyperparameters()
y_star_2, std_y_star_2 = gp2.predict(xdens)
yg_star_2, std_yg_star_2 = gp2.predict(xdens, n=1)

# Make stuff for the basis function illustration:
Xstar = scipy.atleast_2d(xdens).T
Kstar = gp2.compute_Kij(gp2.X, Xstar, gp2.n, scipy.zeros_like(Xstar))

import setupplots
setupplots.thesis_format()
import matplotlib.pyplot as plt
plt.ion()

f = plt.figure(figsize=(setupplots.TEXTWIDTH, setupplots.TEXTWIDTH / 1.618))
a1 = f.add_subplot(2, 2, 1)
a2 = f.add_subplot(2, 2, 2, sharex=a1, sharey=a1)
ag1 = f.add_subplot(2, 2, 3, sharex=a1)
ag2 = f.add_subplot(2, 2, 4, sharex=a1, sharey=ag1)

# a1.plot(xdens, ydens, 'b:', label='true', lw=2.0 * setupplots.lw)
a1.errorbar(x, y2, yerr=n1, label='noisy observations', color='b', fmt='o')
gptools.univariate_envelope_plot(xdens, y_star_1, std_y_star_1, color='g', ax=a1, label=r'\textsc{gp} fit')

# a2.plot(xdens, ydens, 'b:', label='true', lw=2.0 * setupplots.lw)
a2.errorbar(x, y2, yerr=n2, label='noisy observations', color='b', fmt='o')
gptools.univariate_envelope_plot(xdens, y_star_2, std_y_star_2, color='g', ax=a2, label=r'\textsc{gp} fit')

# ag1.plot(xdens, ygdens, 'b:', lw=2.0 * setupplots.lw)
gptools.univariate_envelope_plot(xdens, yg_star_1, std_yg_star_1, color='g', ax=ag1)

# ag2.plot(xdens, ygdens, 'b:', lw=2.0 * setupplots.lw)
gptools.univariate_envelope_plot(xdens, yg_star_2, std_yg_star_2, color='g', ax=ag2)

a1.set_title("Precise")
a1.set_ylabel("$y$")
plt.setp(a1.get_xticklabels(), visible=False)
a2.set_title("Noisy")
plt.setp(a2.get_yticklabels(), visible=False)
plt.setp(a2.get_xticklabels(), visible=False)
ag1.set_ylabel(r"$\mathrm{d}y/\mathrm{d}x$")
ag1.set_xlabel("$x$")
plt.setp(ag2.get_yticklabels(), visible=False)
ag2.set_xlabel("$x$")
f.suptitle(r"Effect of noise on \textsc{gpr} \textsc{map} estimate")
f.subplots_adjust(wspace=0.1, hspace=0.12, top=0.86)

setupplots.apply_formatter(f)

f.savefig("noise_demo.pdf", bbox_inches='tight')
f.savefig("noise_demo.pgf", bbox_inches='tight')

f_leg = plt.figure()
l = f_leg.legend(*a1.get_legend_handles_labels(), ncol=3, loc='center')
f_leg.canvas.draw()
f_leg.set_figwidth(l.get_window_extent().width / 72)
f_leg.set_figheight(l.get_window_extent().height / 72)
f_leg.savefig("legNoiseDemo.pdf", bbox_inches='tight')
f_leg.savefig("legNoiseDemo.pgf", bbox_inches='tight')

# Basis function illustration:
f = plt.figure(figsize=(0.75 * setupplots.TEXTWIDTH, 0.75 * setupplots.TEXTWIDTH / 1.618))
a = f.add_subplot(1, 1, 1)
a.plot(xdens, y_star_2, 'g', label=r'$\overline{y}_*$', lw=2 * setupplots.lw)
clrs = ['b', 'r', 'c', 'm', 'y', 'k']
for i, (ks, av, xv, yv, c) in enumerate(zip(Kstar, gp2.alpha, x, y2, clrs)):
    a.errorbar(xv, yv, yerr=n2, color=c, fmt='o', label='$y_i$' if i == 0 else None)
    a.plot(xdens, ks * av, c + ':', label=r'$\alpha_i k(x_i, x_*)$' if i == 0 else None)
a.set_xlabel("$x$")
a.set_ylabel("$y$")
a.legend(loc='best', numpoints=1)
a.set_title(r"\textsc{gp} mean and basis functions")
setupplots.apply_formatter(f)
f.savefig("gpWithBasis.pdf", bbox_inches='tight')
f.savefig("gpWithBasis.pgf", bbox_inches='tight')

out = "Underestimated & %.4f & %.4f\\\\\nCorrect & %.4f & %.4f\\\\" % (gp1.params[0], gp1.params[1], gp2.params[0], gp2.params[1])
with open('noise_demo_map.tex', 'w') as f:
    f.write(out)
