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

import numpy as np

from aux import index_compute_largets_margin_bubble, compute_goal_state

class BaseSimulator:

    def __init__(self, cont, integrator, world=None):
        self.cont = cont
        self.integrator = integrator
        self.world = world

    def get_shifted_trajectory(self, X, nr_shifts=1):
        X_shifted = np.vstack([
            X[:, nr_shifts:].T, X[:, -1][None].repeat(repeats=nr_shifts, axis=0)
        ]
        )
        return X_shifted

    def get_corridor_balls(self, path_traj, path_centers, path_radii, p_g):
        cs, rs = [], []
        for p in path_traj:
            i_max = index_compute_largets_margin_bubble(p, path_centers, path_radii)
            c, r = path_centers[i_max], path_radii[i_max]
            cs = np.r_[cs, c]
            rs = np.r_[rs, r]
        m_p = p_g.size
        cs = cs.reshape((-1, m_p))
        return cs, rs

    def is_path_in_collision(self, path):
        status = False
        if self.world is not None:
            for q in path:
                if not self.world.is_collision_free_gt(q):
                    status = True
                    break
        return status

    def step_write(self, ts, x, x_g, comp_time, status, write=False):
        if write:
            distance = x - x_g
            distance_position, _ = np.split(distance, 2)
            dist_2_goal = np.linalg.norm(distance_position)
            debug_text = (
                f"{self.cont}"
                f" | {ts: 0.2f}"
                f" | dist_2_goal: {dist_2_goal:0.4f}"
                # f" | progress_path: {i_max_goal / path_centers.shape[0] : 0.2f}"
                f" | comp_time: {comp_time * 1e3:0.4f}"
                # f" | radii: {rs[0]:0.2f}"
                # f" | max(v) {x[m_p:].max():0.4f}"
                # f" | ||traj|| {np.linalg.norm(path_traj[1:] - path_traj[:-1], axis=-1).sum(): 0.2f}"
                # f" | in goal {in_goal}"
                f" | status {status}"
            )
            print(debug_text)

    def end_write(self, ts, x, x_g, status, write=False):
        if write:
            distance = x - x_g
            distance_position, _ = np.split(distance, 2)
            dist_2_goal = np.linalg.norm(distance_position)
            debug_text = (
                f"{self.cont}"
                f" | {ts: 0.2f}"
                " | DONE "
                f" | dist_2_goal: {dist_2_goal:0.4f}"
                f" | status {status}"
            )
            print(debug_text)