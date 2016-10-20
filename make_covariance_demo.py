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

# This script makes figure 2.3, which illustrates the effect of the covariance
# kernel by showing how nearby points are highly correlated with an observation
# but distant points are more weakly correlated.

from __future__ import division

import setupplots
setupplots.thesis_format()
import scipy
import gptools
import matplotlib
import matplotlib.pyplot as plt
plt.ion()
import numpy.random

numpy.random.seed(seed=8675309)

k = gptools.SquaredExponentialKernel()
g = gptools.GaussianProcess(k)
g.add_data(0, 1)

x_grid = scipy.concatenate(([0.0, 0.05, 0.9], scipy.linspace(-0.2, 1, 100)))
y_samp = g.draw_sample(x_grid, num_samp=4)

f = plt.figure(figsize=(0.75 * setupplots.TEXTWIDTH, 0.75 * setupplots.TEXTWIDTH / 1.618))
a = f.add_subplot(1, 1, 1)

a.plot(x_grid[3:], y_samp[3:, :], linewidth=3)
a.plot([x_grid[0]] * y_samp.shape[1], y_samp[0, :], 'ko', markersize=8)
a.plot([x_grid[1]] * y_samp.shape[1], y_samp[1, :], 'g^', markersize=8)
a.plot([x_grid[2]] * y_samp.shape[1], y_samp[2, :], 'rs', markersize=8)

a.set_xlabel(r'independent variable, $x$')
a.set_ylabel(r'dependent variable, $y(x)$')
a.set_title(r'Random realizations of Gaussian process')

setupplots.apply_formatter(f)

f.savefig('GPSample.pdf', bbox_inches='tight')
f.savefig('GPSample.pgf', bbox_inches='tight')
