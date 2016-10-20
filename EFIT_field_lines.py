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

# This script makes figures D.3 and D.4, which show the magnetic and current
# field lines in 3d.

from __future__ import division
import eqtools
eqtools.J_LABEL = r'$j$ [$\si{MA/m^2}$]'
import setupplots
setupplots.thesis_format()
import matplotlib.pyplot as plt
plt.ion()

e = eqtools.CModEFITTree(1120907032)
t = 1.0

f_B, a_B = e.plotField(
    t, num_rev=10, rev_method='poloidal', color='magnitude', arrows=False,
    alpha=0.25
)
a_B.set_title("Magnetic field lines")
f_B.set_size_inches(setupplots.TEXTWIDTH, 0.85 * setupplots.TEXTWIDTH)
f_B.subplots_adjust(left=0.0, bottom=0.0, right=0.91, top=1.0, wspace=0.11)
f_B.savefig("Blines.pdf", bbox_inches='tight')
f_B.savefig("Blines.pgf", bbox_inches='tight')

f_j, a_j = e.plotField(
    t, num_rev=10, rev_method='poloidal', color='magnitude', arrows=False,
    alpha=0.25, field='j', origin='Fnorm', rhomin=0.01, rhomax=0.99
)
a_j.set_title("Current lines")
f_j.set_size_inches(setupplots.TEXTWIDTH, 0.85 * setupplots.TEXTWIDTH)
f_j.subplots_adjust(left=0.0, bottom=0.0, right=0.89, top=1.0, wspace=0.11)
f_j.savefig("jlines.pdf", bbox_inches='tight')
f_j.savefig("jlines.pgf", bbox_inches='tight')
