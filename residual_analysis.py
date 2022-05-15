#!/usr/bin/env python3
 
"""Residual analysis of the front end."""

import argparse
import functools
import json
import argparse
import functools
import json
import signal
import time

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
import optax
from scipy import optimize, stats

import gvi_slam
import utils


if __name__ == '__main__':
    # Define and parse command-line arguments
    parser = argparse.ArgumentParser(
        description=__doc__, fromfile_prefix_chars='@'
    )
    parser.add_argument('posegraph_file', type=open)
    parser.add_argument('output')
    parser.add_argument('--tails', default=10, type=int)
    parser.add_argument('--skip', default=5, type=int)
    args = parser.parse_args()

    # Parse JSON problem data
    link_specs, node_data, odo_pose, tbx_pose = gvi_slam.load_json(
        args.posegraph_file
    )
    i, j, y, cov = gvi_slam.problem_arrays(link_specs)

    # Create problem structure
    p = gvi_slam.Problem(i, j, y, cov, odo_pose[0])

    # Get residuals
    r = p.path_residuals(tbx_pose[1:])
    sr = (p.scale @ r[..., None])[..., 0]

    # Compute quantiles
    r_qq = [stats.probplot(r[:,i])[0] for i in range(3)]
    sr_qq = [stats.probplot(sr[:,i])[0] for i in range(3)]
    data = np.column_stack(sum(r_qq + sr_qq, ()))

    # Remove some rows for better visualization
    tails = args.tails
    skip = args.skip
    data_filt = np.r_[data[:tails], data[tails:-tails:skip], data[-tails:]]
    np.savetxt(args.output, data_filt)
    