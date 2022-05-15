#!/usr/bin/env python3
 
"""Evaluation of ELBO variance and error."""

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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('posegraph_file', type=open)
    parser.add_argument('solution', type=argparse.FileType('rb'))
    parser.add_argument(
        '--output', default='elbo_std.txt', type=argparse.FileType('w')
    )
    args = parser.parse_args()

    # Parse JSON problem data
    link_specs, node_data, odo_pose, tbx_pose = gvi_slam.load_json(
        args.posegraph_file
    )

    # Load solution
    solution = np.load(args.solution)
    dec = solution['mu'], solution['Sld']

    # Create GVI problem instances
    pg = gvi_slam.GlobalDenseProblem.from_link_specs(link_specs, odo_pose[0])
    pl = gvi_slam.LinkwiseDenseProblem.from_link_specs(link_specs, odo_pose[0])

    # Optimize function implementation
    pg.elbo = jax.jit(pg.elbo)
    pl.elbo = jax.jit(pl.elbo)
    
    # Initialize experiment and parameters
    np.random.seed(0)
    key = jax.random.PRNGKey(0)
    Nrep = 1000
    log2_n_max = 16

    # Print output file header
    print('n', 'global', 'linkwise_mc', 'linkwise_rqmc', file=args.output)


    for log2_n in range(log2_n_max + 1):
        n = 2 ** log2_n
        elbo_g = np.zeros(Nrep)
        elbo_l_mc = np.zeros(Nrep)
        elbo_l_rqmc = np.zeros(Nrep)

        print(f'{n = }')
        for i in range(Nrep):
            key, subkey = jax.random.split(key)
            rqmc_sampler = stats.qmc.MultivariateNormalQMC(np.zeros(6))
            e_g_mc = jax.random.normal(subkey, (n, pg.N * 3))
            e_l_mc = jax.random.normal(subkey, (n, 6))
            e_l_rqmc = jnp.asarray(rqmc_sampler.random(n))

            elbo_g[i] = pg.elbo(*dec, e_g_mc)
            elbo_l_mc[i] = pl.elbo(*dec, e_l_mc)
            elbo_l_rqmc[i] = pl.elbo(*dec, e_l_rqmc)

        print(n, np.std(elbo_g), np.std(elbo_l_mc), np.std(elbo_l_rqmc),
              file=args.output)
        args.output.flush()
