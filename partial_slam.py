#!/usr/bin/env python3
 
"""GVI over a partial SLAM session."""

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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('posegraph_file', type=open)
    parser.add_argument('--cutoff', default=484, type=int)
    parser.add_argument(
        '--stoch', default=True, action=argparse.BooleanOptionalAction,
        help='Perform GVI estimation with stochastic optimization.',
    )
    args = parser.parse_args()

    # Parse JSON problem data
    link_specs, node_data, odo_pose, tbx_pose = gvi_slam.load_json(
        args.posegraph_file
    )

    # Trim graph
    inrange = lambda s: s['source'] < args.cutoff > s['target']
    link_specs = list(filter(inrange, link_specs))

    # Create problem structure
    p = gvi_slam.LinkwiseDenseProblem.from_link_specs(link_specs, odo_pose[0])
    p.elbo = jax.jit(p.elbo)
    elbo_grad = jax.jit(p.elbo_grad)
    Nsamp = 2**13

    # Create initial guess
    logdiag0 = jnp.tile(np.log([0.2, 0.2, 0.025]), p.N)
    Sld0 = jnp.diag(logdiag0)
    dec = [p.link_odo[1:], Sld0]

    # Initialize stochastic optimization
    mincost = np.inf
    seed = 1
    last_save_time = -np.inf
    sched = gvi_slam.search_then_converge(2e-2, tau=500, c=5)
    optimizer = optax.adabelief(sched)
    opt_state = optimizer.init(dec)
    np.random.seed(seed)
    
    if not args.stoch:
        raise SystemExit

    # Perform optimization
    for i in range(100_000_000):
        # Sample random population
        sampler = stats.qmc.MultivariateNormalQMC(np.zeros(6))
        e = jnp.asarray(sampler.random(Nsamp))

        # Calculate cost and gradient
        cost_i = -p.elbo(*dec, e)
        grad_i = [-v for v in elbo_grad(*dec, e)]
        mincost = min(mincost, cost_i)

        fooc = [jnp.sum(v**2) ** 0.5 for v in grad_i]
        print(
            f'{i=}', f'cost={cost_i:1.5e}', f'mincost={mincost:1.5e}',
            f'{fooc[0]=:1.2e}', f'{fooc[1]=:1.2e}', f'{sched(i)=:1.1e}',
            sep='\t'
        )

        if any(jnp.any(~jnp.isfinite(v)) for v in grad_i):
            break

        # Compute and apply scaling (preconditioner)
        #S = dec[1]
        #sigma = jnp.sqrt(jnp.sum(S ** 2, axis=1)).reshape(p.N, 3)
        #scaled_mu_b = (S.T @ grad_i[0].flatten()).reshape(p.N, 3)
        #scaled_Sld_b = S.T @ jnp.tril(grad_i[1], -1) + jnp.diag(grad_i[1].diagonal())
        #scaled_Sld_b = Sld_scale * grad_i[1]
        #scaled_grad_i = [scaled_mu_b, scaled_Sld_b]

        #scaled_upd, opt_state = optimizer.update(scaled_grad_i, opt_state)
        #mu_upd = (S @ scaled_upd[0].flatten()).reshape(p.N, 3)
        #Sld_upd = S @ jnp.tril(scaled_upd[1], -1) + jnp.diag(scaled_upd[1].diagonal())
        #Sld_upd = Sld_scale * scaled_upd[1]
        #updates = [mu_upd, Sld_upd]

        updates, opt_state = optimizer.update(grad_i, opt_state)
        updates[0] = 10 * updates[0]
        dec = optax.apply_updates(dec, updates)

        curr_time = time.time()
        if curr_time - last_save_time > 10:
            np.savez('gvi_progress.npz', mu=dec[0], Sld=dec[1])
            last_save_time = curr_time
            print("progress saved")
