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

# This script makes figure 2.5, which shows how GPR works with one observation
# and one prediction.

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
rl = 1.0
y_obs = 1.0

m = scipy.array([[0.0], [0.0]])
K = scipy.matrix([[sigma_f**2, sigma_f**2 * scipy.exp(-rl**2 / 2.0)],
                  [sigma_f**2 * scipy.exp(-rl**2 / 2.0), sigma_f**2]])

y = scipy.linspace(-3, 3, 200)
y_star = scipy.linspace(-3, 3, 201)

Y, Y_STAR = scipy.meshgrid(y - m[0], y_star - m[1])
Y_FLAT = Y.flatten()
Y_STAR_FLAT = Y_STAR.flatten()
YYSTAR = scipy.matrix(scipy.vstack((Y_FLAT, Y_STAR_FLAT)))
Kinv = scipy.linalg.inv(K)
arg = scipy.array([p * Kinv * p.T for p in YYSTAR.T])
fyystar = (
    (2 * scipy.pi)**(-1) * (scipy.linalg.det(K))**(-1.0 / 2.0) *
    scipy.exp(-1.0 / 2.0 * arg)
)
fyystar = scipy.reshape(fyystar, Y.shape)

f = plt.figure(figsize=[0.75 * setupplots.TEXTWIDTH, 0.75 * setupplots.TEXTWIDTH])
gs = mplgs.GridSpec(2, 2, height_ratios=[1, 3], width_ratios=[3, 1])

a_j = f.add_subplot(gs[1, 0], aspect='equal')
cs = a_j.contour(Y_STAR.T, Y.T, fyystar.T, [0.0, 0.05, 0.1, 0.15, 0.2], cmap='binary', linewidths=3)
# a_j.clabel(cs, fontsize=7, inline=1, inline_spacing=2)
a_j.set_xlabel('$y_*$')
a_j.set_ylabel('$y$')
a_j.set_xticks([-2, -1, 0, 1, 2])
a_j.set_yticks([-2, -1, 0, 1, 2])
a_j.axhline(y=y_obs, linewidth=3, color='r', ls=':')

a_ystar = f.add_subplot(gs[0, 0], sharex=a_j)
a_ystar.plot(y_star, scipy.stats.norm.pdf(y_star, loc=0.0, scale=sigma_f), linewidth=3, label='$f_{Y_*}(y_*)$')
a_ystar.plot(
    y_star,
    scipy.stats.norm.pdf(
        y_star,
        loc=K[0, 1] * y_obs / K[0, 0],
        scale=scipy.sqrt(K[1, 1] - (K[0, 1])**2 / K[0, 0])
    ),
    'r:', linewidth=3, label='$f_{Y_*|Y}(y_*|y)$'
)
# a_ystar.axvline(x=K[0, 1] * y_obs / K[0, 0], linewidth=3, color='g', ls='--')
a_ystar.get_xaxis().set_visible(False)
a_ystar.set_ylabel('$f_{Y_*}(y_*)$')
a_ystar.set_ylim(bottom=0.0, top=0.6)
a_ystar.set_yticks([0.0, 0.2, 0.4, 0.6])

a_l = f.add_subplot(gs[0, 1])
a_l.set_visible(False)
a_ystar.legend(loc='lower left', bbox_to_anchor=(0.05, 0.05), bbox_transform=a_l.transAxes)

a_y = f.add_subplot(gs[1, 1], sharey=a_j)
a_y.plot(scipy.stats.norm.pdf(y, loc=0.0, scale=sigma_f), y, linewidth=3)
a_y.get_yaxis().set_visible(False)
a_y.set_xlabel('$f_{Y}(y)$')
a_y.set_xlim(left=0.0, right=0.6)
a_y.set_xticks([0.0, 0.2, 0.4, 0.6])
a_y.axhline(y=y_obs, linewidth=3, color='r', ls=':')

f.subplots_adjust(wspace=0, hspace=0)

f.suptitle(r"Joint, marginal and conditional \textsc{pdf}s")

setupplots.apply_formatter(f)

f.savefig("JointMarginalConditional.pdf", bbox_inches='tight')
f.savefig("JointMarginalConditional.pgf", bbox_inches='tight')
