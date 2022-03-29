#!/usr/bin/env python3
 
"""Gaussian Variational Inference SLAM."""

import argparse
import json
import signal

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np

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
    inv_rot_1 = rotation_matrix_2d(xi[2]).T
    poserr = inv_rot_1 @ (xj[:2] - xi[:2]) - y[:2]
    angerr = normalize_angle(xj[2:] - xi[2:] - y[2])

    errscaled = scale @ jnp.concatenate((poserr, angerr), -1)
    pos_logpdf = jsp.stats.t.logpdf(errscaled[:2], 4, scale=1.0).sum()
    ang_logpdf = jsp.stats.t.logpdf(errscaled[2], 4, scale=1.0)
    return pos_logpdf + ang_logpdf


class Problem:
    def __init__(self, i, j, y, cov):
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

        self.N = max(self.i.max(), self.j.max()) + 1
        """The number of nodes."""

        assert self.i.shape == (self.M,)
        assert self.j.shape == (self.M,)
        assert self.y.shape == (self.M, 3)
        assert self.cov.shape == (self.M, 3, 3)

        info = jnp.linalg.inv(self.cov)
        self.scale = jnp.linalg.cholesky(info).swapaxes(1, 2)
        """Link residual scaling matrix."""
    
    @classmethod
    def from_link_specs(cls, specs):
        M = len(specs)
        i = np.zeros(M, int)
        j = np.zeros(M, int)
        y = np.zeros((M,3), float)
        cov = np.zeros((M,3,3), float)

        for k, spec in enumerate(specs):
            i[k] = spec['source']
            j[k] = spec['target']
            y[k] = spec['pose_difference']
            cov[k] = spec['covariance']
        
        return cls(i, j, y, cov)

    def logpdf(self, x):
        xi = x[..., self.i, :]
        xj = x[..., self.j, :]
        return link_logpdf(xi, xj, self.y, self.scale).sum(-1)


class DenseProblem(Problem):
    def elbo(self, mu, Sld, e):
        logdiag = Sld.diagonal()
        S = jnp.tril(Sld) + jnp.diag(jnp.exp(logdiag))
        S = S.at[:3].set(0)
        Se = jnp.reshape(jnp.atleast_2d(e) @ S.T, (-1, self.N, 3))
        x = mu + Se
        logpdf = self.logpdf(x).sum(0)
        entropy = logdiag[3:].sum()
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
        '--stoch', default=True, action=argparse.BooleanOptionalAction,
        help='Perform GVI estimation with stochastic optimization.',
    )
    args = parser.parse_args()

    # Parse JSON problem data
    link_specs, node_data, odo_pose, tbx_pose = load_json(args.posegraph_file)

    p = DenseProblem.from_link_specs(link_specs)
    Sld = jnp.zeros((3*p.N, 3*p.N))
    