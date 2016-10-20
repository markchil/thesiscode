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

# This script generates the synthetic data and runs MultiNest to select the
# appropriate level of complexity for simple D and V profiles.

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
import scipy.io
import profiletools
import os
import eqtools
import connect
import cPickle as pkl

# Uncomment the appropriate block to perform model selection for that case:

# num_eig_D = 1
# num_eig_V = 1
# n_live_points = 400

# num_eig_D = 2
# num_eig_V = 2
# n_live_points = 400

# num_eig_D = 3
# num_eig_V = 3
# n_live_points = 400

num_eig_D = 4
num_eig_V = 4
n_live_points = 400

# num_eig_D = 5
# num_eig_V = 5
# n_live_points = 400

# num_eig_D = 6
# num_eig_V = 6
# n_live_points = 400

# num_eig_D = 7
# num_eig_V = 7
# n_live_points = 400

# num_eig_D = 8
# num_eig_V = 8
# n_live_points = 400

# num_eig_D = 9
# num_eig_V = 9
# n_live_points = 400

# Use linearly-spaced knots:
knots_D = scipy.linspace(0, 1.05, num_eig_D + 1)[1:-1]
knots_V = scipy.linspace(0, 1.05, num_eig_V + 1)[1:-1]
knotflag = ''

# Set up the actual STRAHL run:
r = bayesimp.Run(
    shot=1101014006,
    version=24,
    time_1=1.165,
    time_2=1.265,
    Te_args=['--system', 'TS', 'GPC', 'GPC2'],
    ne_args=['--system', 'TS'],
    debug_plots=1,
    num_eig_D=num_eig_D,
    num_eig_V=num_eig_V,
    method='linterp',
    free_knots=False,
    use_scaling=False,
    use_shift=False,
    include_loweus=True,
    source_file='/Users/markchilenski/src/bayesimp/Caflx_delta_1165.dat',
    sort_knots=False,
    params_true=scipy.concatenate((
        [1.0,] * num_eig_D,
        scipy.linspace(0, -10, num_eig_V + 1)[1:],
        knots_D,
        knots_V,
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    )),
    time_spec=(  # For fast sampling:
        "    1.165     0.000050               1.01                      1\n"
        "    1.167     0.000075               1.1                       10\n"
        "    1.175     0.000100               1.50                      25\n"
        "    1.265     0.000100               1.50                      25\n"
    ),
    D_lb=0.0,
    D_ub=15.0,
    V_lb=-60.0,
    V_ub=20.0,
    V_lb_outer=-60.0,
    V_ub_outer=20.0,
    num_eig_ne=3,
    num_eig_Te=3,
    free_ne=False,
    free_Te=False,
    normalize=False,
    use_line_integral=False,
    use_local=True,
    local_time_res=6e-3,
    num_local_space=32,
    local_synth_noise=5e-2,
    local_cs=[18,]
)

# Uncomment these lines to perform the inference:
# basename = os.path.abspath('../chains_%d_%d/c-D%d-V%d-%s' % (r.shot, r.version, r.num_eig_D, r.num_eig_V, knotflag))
# r.run_multinest(
#     local=True,
#     n_live_points=n_live_points,
#     basename=basename,
#     sampling_efficiency=0.3,
#     importance_nested_sampling=False
# )

# Attempt to find the MAP estimate:
# bayesimp.acquire_working_dir(lockmode='file')
# map_res = r.find_MAP_estimate(random_starts=1, num_proc=1, use_local=False)
# bayesimp.release_working_dir(lockmode='file')
# # bayesimp.finalize_pool(pool)
# with open('../map_res_%d_%d_D%d_V%d.pkl' % (r.shot, r.version, r.num_eig_D, r.num_eig_V), 'wb') as f:
#     pkl.dump(map_res, f)

# Run the inference with:
# $ mpiexec -np 24 python2.7 settings_1101014006_24_model_selection_local.py
