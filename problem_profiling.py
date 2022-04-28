#!/usr/bin/env python3
 
"""Evaluation of ELBO variance and error."""

import argparse
import functools
import json
import signal
import timeit

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
        '--output', default='profiling.txt', type=argparse.FileType('w')
    )
    args = parser.parse_args()

    # Parse JSON problem data
    link_specs, node_data, odo_pose, tbx_pose = gvi_slam.load_json(
        args.posegraph_file
    )

    # Load solution
    solution = np.load(args.solution)
    dec_cpu = solution['mu'], solution['Sld']
    dec_gpu = [jnp.array(d) for d in dec_cpu]

    # Create GVI problem instances
    pg = gvi_slam.GlobalDenseProblem.from_link_specs(link_specs, odo_pose[0])
    pl = gvi_slam.LinkwiseDenseProblem.from_link_specs(link_specs, odo_pose[0])

    # Optimize function implementation
    g_cpu = jax.jit(pg.elbo_grad, backend='cpu')
    g_gpu = jax.jit(pg.elbo_grad, backend='gpu')
    l_cpu = jax.jit(pl.elbo_grad, backend='cpu')
    l_gpu = jax.jit(pl.elbo_grad, backend='gpu')
    
    # Initialize experiment and parameters
    np.random.seed(0)
    key = jax.random.PRNGKey(0)
    log2_n_max = 14

    # Print output file header
    print('n', 'gcpu', 'ggpu', 'lcpu', 'lgpu', file=args.output)

    for log2_n in range(log2_n_max + 1):
        n = 2 ** log2_n
        print(f'{n = }')

        key, subkey = jax.random.split(key)
        eg_gpu = jax.random.normal(subkey, (n, pg.N * 3))
        el_gpu = jax.random.normal(subkey, (n, 6))
        eg_cpu = np.array(eg_gpu)
        el_cpu = np.array(el_gpu)

        # Call functions once for JIT
        g_gpu(*dec_gpu, eg_gpu)
        g_cpu(*dec_cpu, eg_cpu)
        l_gpu(*dec_gpu, el_gpu)
        l_cpu(*dec_cpu, el_cpu)

        def profile(stmt):
            num = 10
            t = timeit.repeat(stmt, globals=globals(), number=num, repeat=10)
            return np.median(t) / num

        t_gcpu = profile('g_cpu(*dec_cpu, eg_cpu)[0].block_until_ready()')
        t_lcpu = profile('l_cpu(*dec_cpu, el_cpu)[0].block_until_ready()')
        t_ggpu = profile('g_cpu(*dec_gpu, eg_gpu)[0].block_until_ready()')
        t_lgpu = profile('l_gpu(*dec_gpu, el_gpu)[0].block_until_ready()')

        print(n, t_gcpu, t_ggpu, t_lcpu, t_lgpu, file=args.output)
        args.output.flush()
