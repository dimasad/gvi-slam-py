#!/usr/bin/env python3
 
"""Gaussian Variational Inference SLAM."""

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

import utils

@jax.jit
@utils.jax_vectorize(signature='()->(2,2)')
def rotation_matrix_2d(angle):
    cos = jnp.cos(angle)
    sin = jnp.sin(angle)
    return jnp.array([[cos, -sin], [sin, cos]])


def normalize_angle(ang):
    return (ang + jnp.pi) % (2 * jnp.pi) - jnp.pi


@utils.jax_vectorize(signature='(3),(3),(3),(3,3)->()')
def link_logpdf(xi, xj, y, scale):
    inv_rot_i = rotation_matrix_2d(xi[2]).T
    poserr = inv_rot_i @ (xj[:2] - xi[:2]) - y[:2]
    angerr = normalize_angle(xj[2:] - xi[2:] - y[2:])

    errscaled = scale @ jnp.concatenate((poserr, angerr), -1)
    pos_logpdf = jsp.stats.t.logpdf(errscaled[:2], 4, scale=1.0).sum()
    ang_logpdf = jsp.stats.t.logpdf(errscaled[2], 4, scale=1.0)
    return pos_logpdf + ang_logpdf


class Problem:
    def __init__(self, i, j, y, cov, x0=None):
        self.i = jnp.asarray(i, int)
        """Graph link sources."""

        self.j = jnp.asarray(j, int)
        """Graph link targets."""

        self.y = jnp.asarray(y)
        """Measured 2D pose transformation from node i to node j."""

        self.cov = jnp.asarray(cov)
        """Measured 2D pose transformation from node i to node j."""

        self.M = len(self.i)
        """The number of links."""

        self.N = int(max(self.i.max(), self.j.max()))
        """The number of free nodes."""

        self.x0 = jnp.asarray(x0)
        """Known pose of first node (anchor)."""

        assert self.i.shape == (self.M,)
        assert self.j.shape == (self.M,)
        assert self.y.shape == (self.M, 3)
        assert self.cov.shape == (self.M, 3, 3)

        info = jnp.linalg.inv(self.cov)
        self.scale = jnp.linalg.cholesky(info).swapaxes(1, 2)
        """Link residual scaling matrix."""

    @classmethod
    def from_link_specs(cls, specs, x0):
        M = len(specs)
        i = np.zeros(M, int)
        j = np.zeros(M, int)
        y = np.zeros((M, 3), float)
        cov = np.zeros((M, 3, 3), float)

        for k, spec in enumerate(specs):
            i[k] = spec['source']
            j[k] = spec['target']
            y[k] = spec['pose_difference']
            cov[k] = spec['covariance']
        
        return cls(i, j, y, cov, x0)

    @property
    def link_odo(self):
        link_map = {(int(self.i[k]), int(self.j[k])): k for k in range(self.M)}
        poses = np.tile(self.x0, (self.N+1, 1))
        for i in range(self.N):
            j = i+1
            k = link_map[i, j]
            y = self.y[k]
            rot_i = rotation_matrix_2d(poses[i, 2])
            poses[j, :2] = poses[i, :2] + rot_i @ y[:2]
            poses[j, 2] = poses[i, 2] + y[2]
        return poses

    @utils.jax_vectorize_method(signature='(N,3)->(K,3)')
    def prepend_anchor(self, x):
        return jnp.r_[self.x0[None], x]

    @utils.jax_jit_method
    def logpdf(self, x):
        x_anchored = self.prepend_anchor(x)
        xi = x_anchored[..., self.i, :]
        xj = x_anchored[..., self.j, :]
        return link_logpdf(xi, xj, self.y, self.scale).sum(-1)
    
    @property
    @functools.cache
    def logpdf_grad(self):
        return jax.jit(jax.grad(self.logpdf))

    @property
    @functools.cache
    def logpdf_hvp(self):
        hvp = lambda x, x_d: jax.jvp(self.logpdf_grad, (x,), (x_d,))[1]
        return jax.jit(hvp)


class DenseProblem(Problem):
    @staticmethod
    def assemble_S(Sld):
        logdiag = Sld.diagonal()
        S = jnp.tril(Sld, -1) + jnp.diag(jnp.exp(logdiag))
        return logdiag, S

    @property
    @functools.cache
    def elbo_grad(self):
        return jax.jit(jax.grad(self.elbo, (0, 1)))

    @utils.jax_jit_method
    def elbo_hvp(self, mu, Sld, mu_d, Sld_d, *args):
        primals = mu, Sld, *args
        tangents = mu_d, Sld_d, *[jnp.zeros_like(a) for a in args]
        return jax.jvp(self.elbo_grad, primals, tangents)[1]


class GlobalDenseProblem(DenseProblem):
    @utils.jax_jit_method
    def elbo(self, mu, Sld, e):
        logdiag, S = self.assemble_S(Sld)
        Se = jnp.inner(e, S).reshape(-1, self.N, 3)
        x = mu + Se
        logpdf = self.logpdf(x).mean(0)
        entropy = logdiag.sum()
        return logpdf + entropy


class LinkwiseDenseProblem(DenseProblem):
    @utils.jax_jit_method
    def elbo(self, mu, Sld, e):
        # Assemble the scale matrix
        logdiag, S = self.assemble_S(Sld)

        # Anchor the scale matrix and separate its blocks
        Np1 = self.N + 1
        S_anchor = jnp.diag(jnp.repeat(1e-8, 3))
        S_anchored = jsp.linalg.block_diag(S_anchor, S)
        S_blk = S_anchored.reshape(Np1, 3, Np1, 3).swapaxes(1, 2)
        ST_blk = S_blk.swapaxes(-1, -2)

        # Build the joint covariance matrix for each link's node pair
        cov_node = jnp.sum(S_blk @ ST_blk, axis=1)
        cov_cross_link = jnp.sum(S_blk[self.i] @ ST_blk[self.j], axis=1)
        cov_cross_link_T = cov_cross_link.swapaxes(-1, -2)
        cov_link = jnp.block([[cov_node[self.i], cov_cross_link],
                              [cov_cross_link_T, cov_node[self.j]]])

        # Build the joint scale matrix for each link's node pair
        S_link = jnp.linalg.cholesky(cov_link)
        Se = jnp.inner(e, S_link)

        # Sample each link pair
        mu_anchored = self.prepend_anchor(mu)
        xi = mu_anchored[p.i] + Se[..., :3]
        xj = mu_anchored[p.j] + Se[..., 3:]
        logpdf = link_logpdf(xi, xj, self.y, self.scale).sum(-1).mean(0)
        entropy = logdiag.sum()
        return logpdf + entropy


def load_json(file):
    obj = json.load(file)
    link_specs = obj['links']
    node_data = obj['node_data']
    odo_pose = np.array(obj['odometric_pose'])
    tbx_pose = np.array(obj['slam_toolbox_pose'])
    return link_specs, node_data, odo_pose, tbx_pose


if __name__ == '__main__':
    # Define and parse command-line arguments
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('posegraph_file', type=open)
    parser.add_argument(
        '--map', default=False, action=argparse.BooleanOptionalAction,
        help='Perform MAP estimation.',
    )
    parser.add_argument(
        '--stoch', default=True, action=argparse.BooleanOptionalAction,
        help='Perform GVI estimation with stochastic optimization.',
    )
    args = parser.parse_args()

    # Parse JSON problem data
    link_specs, node_data, odo_pose, tbx_pose = load_json(args.posegraph_file)

    # Create problem structure
    p = LinkwiseDenseProblem.from_link_specs(link_specs, odo_pose[0])
    Nsamp = 2**12

    # Create initial guess
    logdiag0 = jnp.tile(np.log([0.2, 0.2, 0.025]), p.N)
    Sld0 = jnp.diag(logdiag0)
    dec = [p.link_odo[1:], Sld0]

    # Perform MAP estimation, if requested
    if args.map:
        x0map = dec[0]
        mapur = lambda v: v.reshape(-1, 3)
        mapcost = lambda v: -p.logpdf(mapur(v)).to_py()
        mapgrad = lambda v: -p.logpdf_grad(mapur(v)).flatten()
        maphvp = lambda v, v_d: -p.logpdf_hvp(mapur(v), mapur(v_d)).flatten()
        def map_callback(v, *args):
            pass
        mapsol = optimize.minimize(
            mapcost, x0map, jac=mapgrad, callback=map_callback,
            method='trust-constr',
            hessp=maphvp,
            tol=1e-10,
            options={'maxiter': 1500, 'gtol': 1e-10, 'verbose': 2}
        )

    # Initialize stochastic optimization
    mincost = np.inf
    seed = 0
    last_save_time = 0
    sched = lambda i: 2e-4 / (1 + i * 1e-4)
    optimizer = optax.sgd(sched)
    opt_state = optimizer.init(dec)
    sampler = stats.qmc.MultivariateNormalQMC(np.zeros(6), seed=seed)
    
    if not args.stoch:
        raise SystemExit

    # Perform optimization
    for i in range(100_000_000):
        e = jnp.asarray(sampler.random(Nsamp))

        cost_i = -p.elbo(*dec, e)
        grad_i = [-v for v in p.elbo_grad(*dec, e)]
        mincost = min(mincost, cost_i)

        fooc = max(jnp.abs(v).max() for v in grad_i)
        print(
            f'{i=}', f'cost={cost_i:1.3e}', f'mincost={mincost:1.3e}',
            f'{fooc=:1.2e}',
            sep='\t'
        )

        if any(jnp.any(~jnp.isfinite(v)) for v in grad_i):
            break
        
        updates, opt_state = optimizer.update(grad_i, opt_state)
        dec = optax.apply_updates(dec, updates)

        curr_time = time.time()
        if curr_time - last_save_time > 10:
            np.savez('gvi_progress.npz', mu=dec[0], Sld=dec[1])
            last_save_time = curr_time
            print("progress saved")
