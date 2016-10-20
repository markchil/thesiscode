#!/usr/bin/env python2.7

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

# This script generates the local synthetic data and runs MultiNest to infer D
# and V with ne and Te allowed to vary.

from __future__ import division

from signal import signal, SIGPIPE, SIG_DFL 
#Ignore SIG_PIPE and don't throw exceptions on it... (http://docs.python.org/library/signal.html)
signal(SIGPIPE, SIG_DFL) 

"""This is a settings file for bayesimp. Any valid Python syntax may be used, as
the file is simply imported like any other module.

This file can be run as a script, in which case it will perform the appropriate
operations depending on the state of the relevant directories.

If run as a script, it MUST be run from the directory containing bayesimp.
"""

# Put the modified version of emcee at the front of sys.path:
import sys
sys.path.insert(0, "/Users/markchilenski/src/emcee_adaptive")

import bayesimp
import gptools
import scipy

r = bayesimp.Run(
    shot=1101014006,
    version=25,
    time_1=1.165,
    time_2=1.265,
    Te_args=['--system', 'TS', 'GPC', 'GPC2'],
    ne_args=['--system', 'TS'],
    debug_plots=1,
    num_eig_D=1,
    num_eig_V=1,
    method='linterp',
    free_knots=False,
    use_scaling=False,
    use_shift=False,
    include_loweus=True,
    source_file='/Users/markchilenski/src/bayesimp/Caflx_delta_1165.dat',
    sort_knots=True,
    params_true=[1.0, -10.0, 0.0, 0.0, 0.0, 0, 0, 0],
    time_spec=(  # For fast sampling:
        "    1.165     0.000050               1.01                      1\n"
        "    1.167     0.000075               1.1                       10\n"
        "    1.175     0.000100               1.50                      25\n"
        "    1.265     0.000100               1.50                      25\n"
    ),
    D_lb=0.0,
    D_ub=10.0,
    V_lb=-100.0,
    V_ub=10.0,
    V_lb_outer=-100.0,
    V_ub_outer=10.0,
    num_eig_ne=3,
    num_eig_Te=3,
    free_ne=True,
    free_Te=True,
    normalize=False,
    local_time_res=6e-3,
    num_local_space=32,
    local_synth_noise=5e-2,
    use_line_integral=False,
    use_local=True,
    local_cs=[18,]
)

# Uncomment this line to perform the inference:
# r.run_multinest()

# Run the inference with:
# $ mpiexec -np 24 python2.7 settings_1101014006_25_ne3_Te3_local.py
