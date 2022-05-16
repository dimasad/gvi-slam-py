#!/usr/bin/env python3
 
"""Toy example for visual abstract."""

import argparse
import functools
import importlib
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
    parser.add_argument(
        '--posegraph_file', default='data/toy.posegraph.json', type=open
    )
    parser.add_argument(
        '--save-map', type=argparse.FileType('w'),  dest='save_map',
    )
    parser.add_argument(
        '--stoch', default=True, action=argparse.BooleanOptionalAction,
        help='Perform GVI estimation with stochastic optimization.',
    )
    parser.add_argument('--reload', default=[], nargs='*')
    args = parser.parse_args()

    # Reload modules needed
    for mod in args.reload:
        importlib.reload(importlib.import_module(mod))

    # Parse JSON problem data
    link_specs, node_data, odo_pose, tbx_pose = gvi_slam.load_json(
        args.posegraph_file
    )

    # Trim graph
    i, j, y, cov = gvi_slam.problem_arrays(link_specs)

    # Create problem structure
    p = gvi_slam.LinkwiseDenseProblem(i, j, y, cov, odo_pose[0])
    elbo_grad = jax.jit(p.elbo_grad)
    Nsamp = 2**17

    # Run MAP estimation
    x0map = p.link_odo[1:].flatten()
    mapur = lambda v: v.reshape(-1, 3)
    mapcost = lambda v: -p.logpdf(mapur(v)).to_py()
    mapgrad = lambda v: -p.logpdf_grad(mapur(v)).flatten()
    maphvp = lambda v, v_d: -p.logpdf_hvp(mapur(v), mapur(v_d)).flatten()
    mapsol = optimize.minimize(
        mapcost, x0map, jac=mapgrad,
        method='trust-constr',
        hessp=maphvp,
        tol=1e-10,
        options={'maxiter': 1500, 'gtol': 1e-10, 'verbose': 2}
    )

    # Run MAP estimation with Gaussian observation model
    pg = gvi_slam.GaussianProblem(i, j, y, cov, odo_pose[0])
    mapgcost = lambda v: -pg.logpdf(mapur(v)).to_py()
    mapggrad = lambda v: -pg.logpdf_grad(mapur(v)).flatten()
    mapghvp = lambda v, v_d: -pg.logpdf_hvp(mapur(v), mapur(v_d)).flatten()
    mapgsol = optimize.minimize(
        mapgcost, x0map, jac=mapggrad,
        method='trust-constr',
        hessp=mapghvp,
        tol=1e-10,
        options={'maxiter': 1500, 'gtol': 1e-10, 'verbose': 2}
    )
    gx = mapur(mapgsol.x)
    Hg = pg.logpdf_hess(gx).reshape(pg.N*3, pg.N*3)
    S_glap = np.linalg.cholesky(np.linalg.inv(-Hg))
    Sld_glap = p.disassemble_S(S_glap)
    S_gposition = p.S_position(Sld_glap)
    p.save('data/toy_gauss.laplace.npz', gx, Sld_glap)

    # Save updated JSON with MAP estimate
    if args.save_map is not None:
        args.posegraph_file.seek(0)
        obj = json.load(args.posegraph_file)
        obj['slam_toolbox_pose'] = p.prepend_anchor(mapur(mapsol.x)).tolist()
        json.dump(obj, args.save_map, indent=2)
        args.save_map.close()

    # Create initial guess
    mu0 = mapur(mapsol.x)
    H = p.logpdf_hess(mu0).reshape(p.N*3, p.N*3)
    S_lap = np.linalg.cholesky(np.linalg.inv(-H))
    Sld_lap = p.disassemble_S(S_lap)
    dec = [mu0, Sld_lap]
    
    # Exit if stochastic optimization not needed
    if not args.stoch:
        raise SystemExit
    
    # Initialize stochastic optimization
    mincost = np.inf
    seed = 1
    last_save_time = -np.inf
    sched = gvi_slam.search_then_converge(5e-3, tau=50, c=8)
    optimizer = optax.adam(sched)
    opt_state = optimizer.init(dec)
    np.random.seed(seed)
    
    # Perform optimization
    for i in range(10_000):
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
        
        updates, opt_state = optimizer.update(grad_i, opt_state)
        dec = optax.apply_updates(dec, updates)

        curr_time = time.time()
        if curr_time - last_save_time > 10:
            p.save('data/toy_progress.gvi.npz', *dec)
            last_save_time = curr_time
            print("progress saved")

    # Save final solution
    p.save('data/toy.gvi.npz', *dec)
