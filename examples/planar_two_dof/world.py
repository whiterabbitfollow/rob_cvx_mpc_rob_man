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
import pathlib

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Polygon, Rectangle

from scipy.interpolate import interpn

from examples.planar_two_dof import P_EXAMPLE_2_DOF
from examples.planar_two_dof.manipulator.man_two_link import ManKin


class DiskWorld:

    def __init__(self, obs_p, obs_r, p_cach_dir=None):
        self.obs_p, self.obs_r = obs_p, obs_r
        self.man = ManKin()
        self.config_dim = 2
        self.world_lims = np.c_[np.ones(self.config_dim, ) * -1, np.ones(self.config_dim, )]
        self.lims = self.world_lims * np.pi
        self.p_cach_dir = p_cach_dir
        self.dist_margin = 0
        p_cdist_data = p_cach_dir / "cspace_data.npz"
        data = np.load(p_cdist_data)
        self.Q, self.D, self.G = [data[name] for name in "QDG"]
        self.boundaries = data["boundaries"]

    @classmethod
    def from_example(cls, **kwargs):
        obs_p = np.vstack([
            [0.75, 0.0],
            [-0.5, 0.25],
            [-0.75, -0.75],
            [0.4, 0.5]
        ]
        )
        obs_r = np.r_[0.15, 0.1, 0.6, 0.2]
        return cls(obs_p, obs_r, p_cach_dir=P_EXAMPLE_2_DOF)

    def sample(self, nr_samples=1):
        if nr_samples == 1:
            size = (self.config_dim,)
        else:
            size = (nr_samples, self.config_dim)
        return np.random.uniform(-np.pi, np.pi, size=size)

    def sample_coll_free(self, with_radius=False, size=1):
        xs_f = np.empty((0, self.config_dim))
        while xs_f.shape[0] < size:
            remaining = size - xs_f.shape[0]
            xs_batch = self.sample(nr_samples=remaining)
            if len(xs_batch.shape) == 1:
                xs_batch = xs_batch[None]
            m = self.coll_free(xs_batch)
            xs_batch_f = xs_batch[m]
            xs_f = np.append(xs_f, xs_batch_f, axis=0)
        xs_f = xs_f[:size]
        if size == 1:
            xs_f = xs_f.ravel()
        if with_radius:
            return np.c_[xs_f, self.sdf(xs_f)]
        else:
            return xs_f

    def coll_free(self, x, r_robot=0):
        sd = self.sdf(x)
        return sd > 0

    def is_collision_free_gt(self, q):
        return self.man.is_collision_free_sphere(q, self.obs_p, self.obs_r)

    def sdf(self, x):
        z = self.Q[:, 0, 0]
        return interpn((z, z), self.D, x, bounds_error=False, fill_value=None) - self.dist_margin

    def render_configuration(self, ax, q, color="k"):
        positions = self.man.get_wireframe(q)
        ax.plot(positions[:, 0], positions[:, 1], lw=3, color=color)
        ax.scatter(positions[:, 0], positions[:, 1], color=color)

    def render_world_space(self, ax, q, border=True, set_aspect=True, color="k"):
        self.render_configuration(ax, q)
        for p, r in zip(self.obs_p, self.obs_r):
            ax.add_patch(Circle(p, radius=r, color="blue"))
        if set_aspect:
            ax.set_aspect("equal")
            ax.set(xlim=(-1, 1), ylim=(-1, 1))

    def render_configuration_space(self, ax, q=None, border=True, set_aspect=True):
        if q is not None:
            q_ = q if q is not None else np.zeros(self.config_dim, )
            ax.scatter(*q_, c="k")
        ax.set_facecolor("yellow")
        for bnd in self.boundaries:
            ax.plot(bnd[:,0], bnd[:, 1])
            ax.add_patch(Polygon(bnd, color="purple"))
        ax.contour(self.Q[:, :, 0], self.Q[:, :, 1], self.D, levels=(0.0, ), colors=("purple", ))
        ax.add_patch(Rectangle((-np.pi, -np.pi), 2 * np.pi, 2 * np.pi, facecolor="none", edgecolor="k"))
        ax.set(xlim=(-np.pi, np.pi), ylim=(-np.pi, np.pi))
        ax.set_aspect("equal")


if __name__ == "__main__":
    man = ManKin()
    world = DiskWorld.from_example()
    fig, (ax1, ax2) = plt.subplots(1, 2)
    world.render_world_space(ax1, q=np.zeros(2, ))
    world.render_configuration_space(ax2)
    plt.show()