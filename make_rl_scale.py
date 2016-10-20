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

# This script makes figure 2.6(b), which shows the squared exponential
# covariance kernel.

from __future__ import division

import scipy
import gptools

k = gptools.SquaredExponentialKernel(initial_params=[1.0, 1.0])

X = scipy.atleast_2d(scipy.linspace(0, 4.0, 500)).T
X0 = scipy.zeros_like(X)

kv = k(X0, X, X0, X0)

import setupplots
setupplots.thesis_format()
import matplotlib.pyplot as plt
plt.ion()

f = plt.figure(figsize=(0.5 * setupplots.TEXTWIDTH, 0.5 * setupplots.TEXTWIDTH / 1.618))
a = f.add_subplot(1, 1, 1)
a.plot(X[:, 0], kv, lw=2.0 * setupplots.lw)
a.plot(
    0.5,
    k(scipy.asarray([[0.0]]), scipy.asarray([[0.5]]), scipy.asarray([[0.0]]), scipy.asarray([[0.0]])),
    'mo',
    markersize=2 * setupplots.ms
)
a.plot(
    1.0,
    k(scipy.asarray([[0.0]]), scipy.asarray([[1.0]]), scipy.asarray([[0.0]]), scipy.asarray([[0.0]])),
    'go',
    markersize=2 * setupplots.ms
)

a.plot(
    3.0,
    k(scipy.asarray([[0.0]]), scipy.asarray([[3.0]]), scipy.asarray([[0.0]]), scipy.asarray([[0.0]])),
    'bo',
    markersize=2 * setupplots.ms
)
a.set_xlabel(r"$r/\ell$")
a.set_ylabel(r"$\cov[y, y_*]/\sigma_f^2$")
a.set_title("Squared exponential covariance kernel")

setupplots.apply_formatter(f)

f.savefig("rlKse.pdf", bbox_inches='tight')
f.savefig("rlKse.pgf", bbox_inches='tight')