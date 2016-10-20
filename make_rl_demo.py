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

# This script makes figure 2.6(a), which shows the effect of r/l on the shape of
# the joint prior distribution.

from __future__ import division
import setupplots
setupplots.thesis_format()
import scipy
import scipy.linalg
import scipy.stats
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as mplgs

plt.ion()
plt.close('all')

sigma_f = 1.0
rl_vals = [0.5, 1.0, 3.0]
colors = ['m', 'g', 'b']
cmaps = ['Purples', 'Greens', 'Blues']

m = scipy.array([[0.0], [0.0]])

y = scipy.linspace(-3, 3, 200)
y_star = scipy.linspace(-3, 3, 201)
Y, Y_STAR = scipy.meshgrid(y - m[0], y_star - m[1])
Y_FLAT = Y.flatten()
Y_STAR_FLAT = Y_STAR.flatten()
YYSTAR = scipy.matrix(scipy.vstack((Y_FLAT, Y_STAR_FLAT)))

f = plt.figure(figsize=[setupplots.TEXTWIDTH, setupplots.TEXTWIDTH / len(rl_vals)])
gs = mplgs.GridSpec(1, len(rl_vals), width_ratios=[1,] * len(rl_vals), top=0.75, wspace=0, hspace=0)
a = []

levels = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]

for idx in xrange(0, len(rl_vals)):
    rl = rl_vals[idx]
    clr = colors[idx]
    cmap = cmaps[idx]
    
    K = scipy.matrix([[sigma_f**2, sigma_f**2 * scipy.exp(-rl**2 / 2.0)],
                      [sigma_f**2 * scipy.exp(-rl**2 / 2.0), sigma_f**2]])
    
    Kinv = scipy.linalg.inv(K)
    arg = scipy.array([p * Kinv * p.T for p in YYSTAR.T])
    fyystar = (
        (2 * scipy.pi)**(-1) * (scipy.linalg.det(K))**(-1.0 / 2.0) *
        scipy.exp(-1.0 / 2.0 * arg)
    )
    fyystar = scipy.reshape(fyystar, Y.shape)
    
    a.append(f.add_subplot(gs[:, idx], aspect='equal'))
    ax = a[-1]
    print(fyystar.max())
    print(fyystar.min())
    cs = ax.contour(Y_STAR.T, Y.T, fyystar.T, 5, linewidths=2, cmap=cmap)
    ax.axhline(y=1.0, ls=':', color='r', linewidth=2)
    ax.set_xlabel('$y_*$')
    ax.set_xticks([-2, -1, 0, 1, 2])
    ax.set_yticks([-2, -1, 0, 1, 2])
    ax.set_title(r'$r/\ell = \textcolor{MPL%s}{%.1f}$' % (clr, rl))

a[0].set_ylabel('$y$')
plt.setp(a[1].get_yticklabels(), visible=False)
plt.setp(a[2].get_yticklabels(), visible=False)

f.suptitle("Effect of $r/\ell$ on joint prior")

setupplots.apply_formatter(f)

f.savefig("lengthEffect.pdf", bbox_inches='tight')
f.savefig("lengthEffect.pgf", bbox_inches='tight')
