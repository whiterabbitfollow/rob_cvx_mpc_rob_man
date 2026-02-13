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

import time
import numpy as np

from aux import  compute_goal_state, interpolate_equidistant
from corridor_simulators.base import BaseSimulator


class FlexibleTubeSimulator(BaseSimulator):

    def __init__(self, cont, integrator, world=None, aux_controller_steps=4):
        super().__init__(cont, integrator, world)
        self.aux_controller_steps = aux_controller_steps

    def simulate(
            self,
            path_centers,
            path_radii,
            nr_steps=1000,
            goal_tol=1e-2,
            verbose=False
    ):
        aux_controller_steps = self.aux_controller_steps
        cont = self.cont
        world = self.world
        integrator = self.integrator
        N = cont.N
        m_x, m_u = cont.A.shape
        m_p = int(m_x / 2)
        p_g = path_centers[-1]
        O_mp = np.zeros_like(p_g)
        x_g = np.r_[p_g, O_mp]
        x_0 = np.r_[path_centers[0], O_mp]
        path_traj = x_0[:m_p][None].repeat(N + 1, axis=0)
        x = x_0
        all_results = []
        ts = 0
        status = "not_in_goal"
        path_track = path_centers.copy()
        path_track = interpolate_equidistant(path_track, delta=0.001)
        x_believed = x.copy()
        # estimated tube size at the end of aux controllers horizon
        s_M = 0
        while ts < nr_steps:
            time_s = time.time()
            cs, rs = self.get_corridor_balls(path_traj, path_centers, path_radii, p_g)
            x_g_v, i_max_goal = compute_goal_state(cs, rs, path_track, p_g, return_index=True)
            time_e_corridor_planning = time.time() - time_s
            cont.set_parameter_values(x_believed, cs.T, rs, x_g_v, s_M)
            # assert (cont.delta_f * cont.r_p_0  < rs).all()
            time_mpc_s = time.time()
            try:
                success = cont.solve()
                status = cont.problem.status
            except:
                status = "failure"
                success = False
            time_mpc_e = time.time() - time_mpc_s
            if success:
                time_solver_e = cont.problem.solver_stats.solve_time
            else:
                time_solver_e = time_mpc_e
            time_e = time.time() - time_s
            x_t = x.copy()
            states_aux_controller = []
            noise = []
            if not success:
                all_results.append(
                    FlexibleTubeResults(
                        cont,
                        noise,
                        states_aux_controller,
                        times=(time_e, time_e_corridor_planning, time_mpc_e, time_solver_e),
                        x_t=x_t
                    )
                )
                break
            # predict future tube state
            x_believed = cont.get_predicted_state(k=aux_controller_steps)
            # predict future tube size
            s_M = cont.predict_tube_size(x, M=aux_controller_steps)
            for k in range(aux_controller_steps):
                v = cont.get_solved_control(x, k=k)
                noise.append(integrator.get_noise(x, v))    # for debug
                x = integrator.solve_time_step(x, v)
                ts += 1
                states_aux_controller.append(x)
            states_aux_controller = np.vstack(states_aux_controller)
            all_results.append(
                FlexibleTubeResults(
                    cont,
                    noise,
                    states_aux_controller,
                    times=(time_e, time_e_corridor_planning, time_mpc_e, time_solver_e),
                    x_t=x_t
                )
            )
            X = cont.get_predicted_traj()
            X_shifted = self.get_shifted_trajectory(X, nr_shifts=aux_controller_steps)
            path_traj = X_shifted[:, :m_p]
            if world is not None:
                path_aux_controller = states_aux_controller[:, :m_p]
                coll_check_qs = np.vstack([path_aux_controller, path_traj])
                if self.is_path_in_collision(coll_check_qs):
                    status = "collision"
                    break
            in_goal = np.linalg.norm(x - x_g) < goal_tol
            self.step_write(ts, x, x_g, time_e, status, write=verbose)
            if status == "collision":
                break
            if in_goal:
                status = "success"
                break
        self.end_write(ts, x, x_g, status, write=verbose)
        return (status, ts), all_results


class FlexibleTubeResults:

    def __init__(self, cont, noise, states_aux_controller, times, x_t):
        self.inputs = cont.get_solution_inputs()
        self.outputs = cont.get_solution_results()
        self.noise = noise
        self.states_aux_controller = states_aux_controller
        self.times = times
        self.x_t = x_t
