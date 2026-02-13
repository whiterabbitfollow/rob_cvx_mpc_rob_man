# Copyright (c) 2025, ABB Schweiz AG
# All rights reserved.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
# THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


import argparse
import pickle

import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib.patches import Circle

import matplotlib.pyplot as plt

from examples.planar_two_dof import P_EXAMPLE_2_DOF
from examples.planar_two_dof.world import DiskWorld

fig, (ax1, ax2) = plt.subplots(1, 2)


def render_nominal(world, ax1, ax2, inputs, outputs):
    (x, x_g_v, cs, rs) = inputs
    (X, U) = outputs
    q, _ = np.split(x, 2)
    world.render_world_space(ax1, q=q)
    world.render_configuration(ax1, q=q_g, color="g")
    world.render_configuration_space(ax2, q=q)
    ax = ax2
    ax.scatter(*x[:world.config_dim], c="tab:blue")
    ax.scatter(*q_g, c="tab:green")
    ax.plot(X[0], X[1], marker=".", color="r")
    ax.plot(path_centers[:, 0], path_centers[:, 1], color="k")
    ax.scatter(x_g_v[0], x_g_v[1])
    ax.add_collection(
        PatchCollection(
            [Circle(c, r) for c, r in zip(cs.T, rs)]
            , facecolors="none", edgecolors="k"
        )
    )



def render_rigid_tube(world, ax1, ax2, inputs, outputs):
    (x, z_g, cs, rs) = inputs
    (Z, V, r_p) = outputs
    q, _ = np.split(x, 2)
    world.render_world_space(ax1, q=q)
    world.render_configuration(ax1, q=q_g, color="g")
    world.render_configuration_space(ax2, q=q)
    ax = ax2
    ax.scatter(*x[:world.config_dim], c="tab:blue")
    ax.scatter(*q_g, c="tab:green")
    ax.plot(Z[0], Z[1], marker=".", color="r")
    ax.plot(path_centers[:, 0], path_centers[:, 1], color="k")
    ax.scatter(z_g[0], z_g[1])
    ax.add_collection(
        PatchCollection(
            [Circle(c, r_p) for c in Z.T], facecolors="none", edgecolors="r"
        )
    )
    ax.add_collection(
        PatchCollection(
            [Circle(c, r) for c, r in zip(cs.T, rs)]
            , facecolors="none", edgecolors="k"
        )
    )


def render_flexible_tube(world, ax1, ax2, inputs, outputs):
    (x, z_g, cs, rs, s_0) = inputs
    (Z, U, S, r_p) = outputs
    q, _ = np.split(x, 2)
    world.render_world_space(ax1, q=q)
    world.render_configuration(ax1, q=q_g, color="g")
    world.render_configuration_space(ax2, q=q)
    ax = ax2
    ax.scatter(*x[:world.config_dim], c="tab:blue")
    ax.scatter(*q_g, c="tab:green")
    ax.plot(Z[0], Z[1], marker=".", color="r")
    ax.plot(path_centers[:, 0], path_centers[:, 1], color="k")
    ax.scatter(z_g[0], z_g[1])
    ax.add_collection(
        PatchCollection(
            [Circle(c, r_p * s) for s, c in zip(S, Z.T)], facecolors="none", edgecolors="r"
        )
    )
    ax.add_collection(
        PatchCollection(
            [Circle(c, r) for c, r in zip(cs.T, rs)]
            , facecolors="none", edgecolors="k"
        )
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    names = (
        "nom_star",
        "rt",
        "ft"
    )
    parser.add_argument("--method", type=str, default="nom_star", choices=names)
    args = parser.parse_args()
    f_name = args.method
    world = DiskWorld.from_example()
    path_data = P_EXAMPLE_2_DOF / "data" / "dof_2_ef_0.1"
    paths = list(sorted((path_data / "motion").glob(f"{f_name}*.pckl")))
    for p in paths:
        _, nr = p.stem.rsplit("~", 1)
        nr = int(nr)
        data_corr = np.load(path_data / "corridors" / f"corr_{nr}.npz")
        path_centers = data_corr["path_centers"]
        with p.open("rb") as fp:
            all_results = pickle.load(fp)
        q_s, q_g = path_centers[[0, -1]]
        path_radii = world.sdf(path_centers)
        for results in all_results:
            if f_name == "nom_star" or f_name =="nom":
                render_nominal(world, ax1, ax2, results.inputs, results.outputs)
            elif f_name == "rt":
                render_rigid_tube(world, ax1, ax2, results.inputs, results.outputs)
            elif f_name == "ft":
                render_flexible_tube(world, ax1, ax2, results.inputs, results.outputs)
            plt.pause(.01)
            for ax in (ax1, ax2):
                ax.cla()
