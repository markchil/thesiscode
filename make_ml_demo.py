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

# This script makes figure 3.49, which explains how the marginal likelihood
# helps to select the appropriate level of model complexity.

from __future__ import division

import scipy
import setupplots
setupplots.thesis_format()
import matplotlib.pyplot as plt
plt.ion()

xgrid = scipy.linspace(0, 10, 1000)

f = plt.figure(figsize=(0.75 * setupplots.TEXTWIDTH, 0.75 * setupplots.TEXTWIDTH / 1.618))
a = f.add_subplot(1, 1, 1)
a.plot(xgrid, scipy.stats.norm.pdf(xgrid, loc=0, scale=0.75), 'b', label='simple')
a.plot(xgrid, scipy.stats.norm.pdf(xgrid, loc=0, scale=2.0), 'g--', label='moderate')
a.plot(xgrid, scipy.stats.norm.pdf(xgrid, loc=0, scale=5), 'r-.', label='complex')
a.axvline(2.0, color='k', linestyle=':', linewidth=2 * setupplots.lw, label='observation')
a.legend(loc='best')
a.set_title("Model evidence selects the appropriate level of complexity")
a.set_xlabel(r"data $\mathcal{D}$")
a.set_ylabel(r"evidence $f_{\mathcal{D}|\mathcal{M}}(\mathcal{D}|\mathcal{M})$")
setupplots.apply_formatter(f)
f.savefig("mlDemo.pdf", bbox_inches='tight')
f.savefig("mlDemo.pgf", bbox_inches='tight')
