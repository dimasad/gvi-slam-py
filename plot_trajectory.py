#!/usr/bin/env python3
 
"""Plot trajectory results with uncertainty ellipses."""

import argparse
import functools
import json
import argparse
import functools
import json
import pathlib
import signal
import time

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import optax
import yaml
from scipy import optimize, stats

import gvi_slam
import utils


if __name__ == '__main__':
    # Define and parse command-line arguments
    parser = argparse.ArgumentParser(
        description=__doc__, fromfile_prefix_chars='@'
    )
    parser.add_argument(
        'solution', help='Saved solution to the GVI problem.'
    )
    parser.add_argument(
        '--posegraph_file', type=pathlib.Path, help='Posegraph JSON file.'
    )
    parser.add_argument(
        '--mapbase', type=pathlib.Path
    )
    parser.add_argument(
        '--interactive', default=True, action=argparse.BooleanOptionalAction
    )
    parser.add_argument(
        '--output', type=pathlib.Path, help='Image output file.'
    )
    parser.add_argument(
        '--laplace', default=True, action=argparse.BooleanOptionalAction,
        help="Plot the uncertainty ellipses given by Laplace's method."
    )
    parser.add_argument(
        '--ellipse-skip', default=3, type=int, dest='skip',
        help="Plot 1 out of every x uncertainty ellipses."
    )
    args = parser.parse_args()

    # Initialize figure
    fig = plt.figure(1)
    fig.clf()
    ax = fig.gca()

    # Load solution
    p, (mu, Sld) = gvi_slam.DenseProblem.load(args.solution)

    # Assemble GVI results
    mu_anchored = p.prepend_anchor(mu)
    S_position = p.S_position(Sld)
    th = np.linspace(0, 2*np.pi, 30)
    unit_circle = np.c_[np.cos(th), np.sin(th)]
    unc = 2*np.inner(unit_circle, S_position) + mu[:,:2]
    links = zip(mu_anchored[p.i, :2], mu_anchored[p.j, :2])
    lc = matplotlib.collections.LineCollection(
        links, colors='g', linewidths=0.5
    )

    # Parse JSON problem data
    if args.posegraph_file:
        link_specs, node_data, odo_pose, tbx_pose = gvi_slam.load_json(
            open(args.posegraph_file.expanduser())
        )
        ax.plot(tbx_pose[:, 0], tbx_pose[:, 1], 'b.')
        #ax.plot(odo_pose[:,0], odo_pose[:,1], 'x')
        if args.laplace:
            H = p.logpdf_hess(tbx_pose[1:]).reshape(p.N*3, p.N*3)
            S_lap = np.linalg.cholesky(np.linalg.inv(-H))            
            Sld_lap = p.disassemble_S(S_lap)
            S_position_lap = p.S_position(Sld_lap)
            unc_lap = 2*np.inner(unit_circle, S_position_lap) + tbx_pose[1:, :2]
            ax.plot(unc_lap[:, ::args.skip,0], unc_lap[:, ::args.skip,1], '-c')

    # Load and show map
    if args.mapbase:
        mapbase = args.mapbase.expanduser()
        img = plt.imread(mapbase.with_name(mapbase.name + '.pgm'))
        imgcfg = yaml.safe_load(open(mapbase.with_name(mapbase.name + '.yaml')))
        imgsz = np.array(img.shape)[::-1] * imgcfg['resolution']
        imgorigin = imgcfg['origin']
        imgextent = np.r_[np.r_[0, imgsz[0]] + imgorigin[0], 
                          np.r_[0, imgsz[1]] + imgorigin[1]]
        ax.imshow(img.max() - img, extent=imgextent, cmap='Greys')

    # Show GVI results
    ax.add_collection(lc)
    ax.plot(mu[:, 0], mu[:, 1], 'r.')
    ax.plot(unc[:, ::args.skip,0], unc[:, ::args.skip,1], '-k')
    ax.axis('equal')
    #ax.plot(p.link_odo[:,0], p.link_odo[:,1], 'x')

    if args.interactive:
        fig.show()

    if args.output:
        fig.savefig(args.output.expanduser())
