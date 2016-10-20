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

# This script makes figure 3.10, which shows the normalized first and second
# eigenvalues of the impulse response, as predicted by [Seguin PRL 1983] and
# [Fussmann NF 1986].

from __future__ import division

import scipy

S = scipy.linspace(-15, 25, 1000)
tD2_a2 = (77.0 + S**2.0) * (scipy.exp(S) - S - 1.0) / ((56.0 + S**2.0) * 4.0 * S**2.0)
tD2_a2[S == 0.0] = 77.0 / (56.0 * 8.0)

S_hi = scipy.linspace(30, S.max(), 1000)
t_hi = scipy.exp(S_hi) / (4.0 * S_hi**2.0)

S_lo = scipy.linspace(S.min(), -15, 1000)
t_lo = -1.0 / (4.0 * S_lo)

import setupplots
setupplots.thesis_format()
import matplotlib.pyplot as plt
plt.ion()

f = plt.figure(figsize=(0.75 * setupplots.TEXTWIDTH, 0.75 * setupplots.TEXTWIDTH / 1.618))
a = f.add_subplot(1, 1, 1)
a.semilogy(S, tD2_a2, label=r'$\tau_{\text{imp}}$')
a.axhline(0.03, label=r'$\tau_2$', ls='--', color='g')
a.legend(loc='upper left')
# a.semilogy(S_hi, t_hi, 'g--')
# a.semilogy(S_lo, t_lo, 'r:')
a.set_xlabel(r"$S=-aV/(2D)$")
a.set_ylabel(r"$\tau_{\text{imp}}D / a^2$")
a.set_title("Normalized impurity confinement time")
setupplots.apply_formatter(f)
f.savefig("tauimp_scaling.pdf", bbox_inches='tight')
f.savefig("tauimp_scaling.pgf", bbox_inches='tight')