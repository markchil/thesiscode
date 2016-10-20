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

# This script makes figure 3.56, which shows how the number of forward model
# evaluations increases with model complexity.

from __future__ import division

import scipy

num_eig = range(1, 8)
shot = 1101014006
version = 22

num_calls = []
num_samp = []

for i in num_eig:
    with open(
        '/Users/markchilenski/src/bayesimp/chains_%d_%d/c-D%d-V%d-resume.dat' % (shot, version, i, i), 'r'
    ) as f:
        s = f.read().split()
        num_calls.append(int(s[2]))
        num_samp.append(int(s[1]))

poly = scipy.polyfit(num_eig[1:-1], scipy.log10(num_calls[1:-1]), 1)
expected_calls = lambda i: 10.0**(scipy.polyval(poly, i))

poly2 = scipy.polyfit(num_eig[:-1], num_samp[:-1], 2)
expected_samp = lambda i: scipy.polyval(poly2, i)

import setupplots
setupplots.thesis_format()
import matplotlib.pyplot as plt
plt.ion()

f = plt.figure(figsize=(0.75 * setupplots.TEXTWIDTH, 0.75 * setupplots.TEXTWIDTH / 1.618))
a = f.add_subplot(1, 1, 1)
a.semilogy(num_eig, num_calls, 'o--')
# a.semilogy(num_eig, expected_calls(num_eig), 'r:')
a.set_xticks(num_eig)
a.set_xlim((num_eig[0] - 0.5, num_eig[-1] + 0.5))
a.set_xlabel("number of coefficients")
a.set_ylabel("number of evaluations")
a.set_title("Complicated models are expensive")
setupplots.apply_formatter(f)
f.savefig("costGrowth.pdf", bbox_inches='tight')
f.savefig("costGrowth.pgf", bbox_inches='tight')

# f = plt.figure(figsize=(0.75 * setupplots.TEXTWIDTH, 0.75 * setupplots.TEXTWIDTH / 1.618))
# a = f.add_subplot(1, 1, 1)
# a.plot(num_eig, num_samp, 'o--')
# a.plot(num_eig, expected_samp(num_eig), 'r:')
# a.set_xticks(num_eig)
# a.set_xlim((num_eig[0] - 0.5, num_eig[-1] + 0.5))
# a.set_xlabel("number of coefficients")
# a.set_ylabel("number of samples")
