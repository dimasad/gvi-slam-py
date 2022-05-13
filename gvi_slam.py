#!/usr/bin/env python3
 
"""Gaussian Variational Inference SLAM."""

import argparse
import functools
import json
import pathlib
import signal
import time

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
import optax
from scipy import optimize, stats

import utils


@utils.jax_vectorize(signature='()->(2,2)')
def rotation_matrix_2d(angle):
    cos = jnp.cos(angle)
    sin = jnp.sin(angle)
    return jnp.array([[cos, -sin], [sin, cos]])


def normalize_angle(ang):
    return (ang + jnp.pi) % (2 * jnp.pi) - jnp.pi


def multivariate_t(x, df):
    sqerrsum = jnp.sum(x ** 2, -1)
    n = jnp.shape(x)[-1]
    return -0.5 * (df + n) * jnp.log(1 + sqerrsum / df)


class Problem:

    scale_multiplier = 10.0
    """Tuning multiplier for link scale."""

    degf = 2.0
    """t-distribution degrees of freedom parameter."""

    def __init__(self, i, j, y, cov, x0, 
                 scale_multiplier=None, degf=None, jit='gpu'):
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

        self.jit = jit
        """Type of JIT optimization to use."""

        if scale_multiplier is not None:
            self.scale_multiplier = scale_multiplier

        if degf is not None:
            self.degf = degf

        assert self.i.shape == (self.M,)
        assert self.j.shape == (self.M,)
        assert self.y.shape == (self.M, 3)
        assert self.cov.shape == (self.M, 3, 3)

        info = jnp.linalg.inv(self.cov)
        base_scale = jnp.linalg.cholesky(info).swapaxes(1, 2)
        self.scale = self.scale_multiplier * base_scale
        """Link residual scaling matrix."""

    @property
    def link_odo(self):
        rotm2d = jax.jit(rotation_matrix_2d, backend='cpu')
        self_i = self.i.to_py()
        self_j = self.j.to_py()
        self_y = self.y.to_py()

        link_map = {(int(self_i[k]), int(self_j[k])): k for k in range(self.M)}
        poses = np.tile(self.x0, (self.N+1, 1))
        for i in range(self.N):
            j = i+1
            k = link_map[i, j]
            y = self_y[k]
            rot_i = rotm2d(poses[i, 2])
            poses[j, :2] = poses[i, :2] + rot_i @ y[:2]
            poses[j, 2] = poses[i, 2] + y[2]
        return poses

    @utils.jax_vectorize_method(signature='(N,3)->(K,3)')
    def prepend_anchor(self, x):
        return jnp.r_[self.x0[None], x]

    @utils.jax_vectorize_method(signature='(3),(3),(3)->(3)')
    def residuals(self, xi, xj, y):
        """Residuals associated to an observation"""
        inv_rot_i = rotation_matrix_2d(xi[2]).T
        pos_res = inv_rot_i @ (xj[:2] - xi[:2]) - y[:2]
        ang_res = normalize_angle(xj[2:] - xi[2:] - y[2:])
        return jnp.concatenate((pos_res, ang_res), -1)

    def path_residuals(self, x):
        """Residuals associated to an entire path"""
        x_anchored = self.prepend_anchor(x)
        xi = x_anchored[..., self.i, :]
        xj = x_anchored[..., self.j, :]
        return self.residuals(xi, xj, self.y)

    def _logpdf(self, x):
        r = self.path_residuals(x)
        scaled_r = (self.scale @ r[..., None])[..., 0]
        return multivariate_t(scaled_r, df=self.degf).sum(-1)
    
    @functools.cached_property
    def logpdf(self):
        if self.jit:
            return jax.jit(self._logpdf, backend=self.jit)
        else:
            return self._logpdf

    @property
    @functools.cache
    def logpdf_grad(self):
        grad = jax.grad(self.logpdf)
        return jax.jit(grad, backend=self.jit) if self.jit else grad

    @property
    @functools.cache
    def logpdf_hvp(self):
        hvp = lambda x, x_d: jax.jvp(self.logpdf_grad, (x,), (x_d,))[1]
        return jax.jit(hvp, backend=self.jit) if self.jit else hvp

    @property
    @functools.cache
    def logpdf_hess(self):
        hess = jax.jacobian(self.logpdf_grad)
        return jax.jit(hess, backend=self.jit) if self.jit else hess


class GaussianProblem(Problem):
    def _logpdf(self, x):
        r = self.path_residuals(x)
        scaled_r = (self.scale @ r[..., None])[..., 0]
        return -0.5 * jnp.square(scaled_r).sum((-2, -1))


class DenseProblem(Problem):
    def save(self, filename, mu, Sld):
        data = dict(
            mu=mu, Sld=Sld,
            i=self.i, j=self.j, y=self.y, cov=self.cov, x0=self.x0,
            scale_multiplier=self.scale_multiplier, degf=self.degf,
        )
        np.savez(filename, **data)

    @classmethod
    def load(cls, file):
        data = np.load(file)
        dec = jnp.array(data['mu']), jnp.array(data['Sld'])
        obj = cls(
            data['i'], data['j'], data['y'], data['cov'], data['x0'], 
            data['scale_multiplier'], data['degf']
        )
        return obj, dec

    @staticmethod
    def assemble_S(Sld):
        logdiag = Sld.diagonal()
        S = jnp.tril(Sld, -1) + jnp.diag(jnp.exp(logdiag))
        return logdiag, S

    @staticmethod
    def disassemble_S(S):
        return jnp.tril(S, k=-1) + jnp.diag(jnp.log(jnp.diagonal(S)))

    def S_position(self, Sld):
        """Cholesky factor of marginal position covariances."""
        logdiag, S = self.assemble_S(Sld)
        cov = S @ S.T
        cov_diag = cov.diagonal()

        cov_position = np.empty((self.N, 2, 2))
        cov_position[:, 0, 0] = cov_diag[::3]
        cov_position[:, 1, 1] = cov_diag[1::3]
        cov_position[:, 0, 1] = cov_position[:, 1, 0] = cov.diagonal(1)[::3]
        return np.linalg.cholesky(cov_position)
    
    @property
    @functools.cache
    def elbo_grad(self):
        return jax.grad(self.elbo, (0, 1))

    @utils.jax_jit_method
    def elbo_hvp(self, mu, Sld, mu_d, Sld_d, *args):
        primals = mu, Sld, *args
        tangents = mu_d, Sld_d, *[jnp.zeros_like(a) for a in args]
        return jax.jvp(self.elbo_grad, primals, tangents)[1]

    @property
    def avg_logpdf_grad(self):
        return jax.grad(self.avg_logpdf)

    @property
    def avg_logpdf_hess(self):
        hess = jax.jacfwd(self.avg_logpdf_grad)
        return jax.jit(hess, backend=self.jit) if self.jit else hess

    def elbo(self, mu, Sld, e):
        logpdf = self.avg_logpdf(mu, Sld, e)
        entropy = Sld.diagonal().sum()
        return logpdf + entropy


class GlobalDenseProblem(DenseProblem):
    def avg_logpdf(self, mu, Sld, e):
        logdiag, S = self.assemble_S(Sld)
        Se = jnp.inner(e, S).reshape(-1, self.N, 3)
        x = mu + Se
        return self.logpdf(x).mean(0)


class LinkwiseDenseProblem(DenseProblem):
    def avg_logpdf(self, mu, Sld, e):
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
        xi = mu_anchored[self.i] + Se[..., :3]
        xj = mu_anchored[self.j] + Se[..., 3:]
        r = self.residuals(xi, xj, self.y)
        scaled_r = (self.scale @ r[..., None])[..., 0]
        return multivariate_t(scaled_r, df=self.degf).sum(-1).mean(0)


def load_json(file):
    obj = json.load(file)
    link_specs = obj['links']
    node_data = obj['node_data']
    odo_pose = np.array(obj['odometric_pose'])
    tbx_pose = np.array(obj['slam_toolbox_pose'])
    return link_specs, node_data, odo_pose, tbx_pose


def problem_arrays(link_specs):
    M = len(link_specs)
    i = np.zeros(M, int)
    j = np.zeros(M, int)
    y = np.zeros((M, 3), float)
    cov = np.zeros((M, 3, 3), float)

    for k, spec in enumerate(link_specs):
        i[k] = spec['source']
        j[k] = spec['target']
        y[k] = spec['pose_difference']
        cov[k] = spec['covariance']
    
    return i, j, y, cov


def export_poses(filename, poses):
    n = len(poses)
    i = np.arange(n)
    data = np.c_[i, poses]
    np.savetxt(filename, data)


def search_then_converge(eta0, tau, c):
    def sched(i):
        i = int(i)
        num = 1 + c / eta0 * i / tau
        den = num  + i * i / tau
        return eta0 * num / den
    return sched


if __name__ == '__main__':
    # Define and parse command-line arguments
    parser = argparse.ArgumentParser(
        description=__doc__, fromfile_prefix_chars='@'
    )
    parser.add_argument('posegraph_file', type=open)
    parser.add_argument(
        '--map', default=False, action=argparse.BooleanOptionalAction,
        help='Perform MAP estimation.',
    )
    parser.add_argument(
        '--niter', default=10_000_000, type=int,
        help='Number of iterations of the stochastic optimization.',
    )
    parser.add_argument(
        '--mu0', choices=['tbx', 'odom', 's-tree-1'],
        help='Initial guess for GVI mean'
    )
    parser.add_argument(
        '--lrate0', default=1e-2, type=float,
        help='Stochastic optimization initial learning rate.',
    )
    parser.add_argument(
        '--lrate_tau', default=500, type=float,
        help='Stochastic optimization learning rate "tau" parameter.',
    )
    parser.add_argument(
        '--lrate_c', default=100, type=float,
        help='Stochastic optimization asymptotic learning rate numerator.',
    )
    parser.add_argument(
        '--progress', type=pathlib.Path, 
        default=pathlib.Path('data/gvi_progress.npz'),
        help='Save stochastic optimization progress to file.'
    )
    parser.add_argument(
        '--result', type=pathlib.Path,
        help='Save GVI result to file.'
    )
    args = parser.parse_args()

    # Parse JSON problem data
    link_specs, node_data, odo_pose, tbx_pose = load_json(args.posegraph_file)
    i, j, y, cov = problem_arrays(link_specs)

    # Create problem structure
    p = LinkwiseDenseProblem(i, j, y, cov, odo_pose[0])
    p.elbo = jax.jit(p.elbo)
    elbo_grad = jax.jit(p.elbo_grad)
    Nsamp = 2**13

    # Create initial guess for mean
    if args.mu0 == 'tbx':
        mu0 = tbx_pose[1:]
    elif args.mu0 == 'odom':
        mu0 = odo_pose[1:]
    else: 
        mu0 = p.link_odo[1:]
    
    # Perform MAP estimation, if requested
    if args.map:
        x0map = mu0.flatten()
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
        mu0 = mapur(mapsol.x)

    # Create initial guess of scale
    logdiag0 = jnp.tile(np.log([0.2, 0.2, 0.025]), p.N)
    Sld0 = jnp.diag(logdiag0)
    dec = [mu0, Sld0]

    # Initialize stochastic optimization
    mincost = np.inf
    seed = 1
    last_save_time = -np.inf
    sched = search_then_converge(args.lrate0, args.lrate_tau, args.lrate_c)
    optimizer = optax.adam(sched)
    opt_state = optimizer.init(dec)
    np.random.seed(seed)
    
    # Perform optimization
    for i in range(args.niter):
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
            p.save(args.progress, *dec)
            last_save_time = curr_time
            print("progress saved")

    # Save final result
    p.save(args.result or args.progress, *dec)
