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

from aux import compute_model_error, index_compute_largets_margin_bubble, compute_goal_state


def run_controller_in_corridor(
        ps,
        cont,
        path_centers,
        path_radii,
        nr_steps = 1000,
        ignore_gravity=False,
        ignore_me=False,
        goal_tol = 1e-2,
        verbose=False,
        world=None,
        aux_controller_steps=1
):
    N = cont.N
    A = cont.A
    B = cont.B
    m_x, m_u = A.shape
    m_p = int(m_x/2)
    p_g = path_centers[-1]
    O_mp = np.zeros_like(p_g)
    x_g = np.r_[p_g, O_mp]
    x_0 = np.r_[path_centers[0], O_mp]
    path_traj = x_0[:m_p][None].repeat(N + 1, axis=0)
    x = x_0
    all_results = []
    ts = 0
    status = "not_in_goal"
    while ts < nr_steps:
        time_s = time.time()
        cs, rs = [], []
        for p in path_traj:
            i_max = index_compute_largets_margin_bubble(p, path_centers, path_radii)
            c, r = path_centers[i_max], path_radii[i_max]
            cs = np.r_[cs, c]
            rs = np.r_[rs, r]
        cs = cs.reshape((-1, m_p))
        x_g_v, i_max_goal = compute_goal_state(cs, rs, path_centers, p_g, return_index=True)
        time_e_corridor_planning = time.time() - time_s
        time_mpc_s = time.time()
        try:
            success = cont.solve(
                x, cs.T, rs, x_g_v
            )
        except:
            status = "failure"
            success = False
        time_mpc_e = time.time() - time_mpc_s
        time_e = time.time() - time_s
        if not success:
            status = "infeasible"
            break
        all_results.append(
            (
                cont.get_solution_inputs(),
                cont.get_solution_results(),
                (time_e, time_e_corridor_planning, time_mpc_e)
            )
        )
        path_aux_controller = []
        for k in range(aux_controller_steps):
            q, q_d = x[:m_p], x[m_p:]
            v = cont.get_solved_control(x, k=k)
            if ignore_me:
                delta_x_v = 0
            else:
                delta_x_v = compute_model_error(ps, q, q_d, v, ignore_gravity=ignore_gravity)
            x = A @ x + B @ (v + delta_x_v)
            ts += 1
            path_aux_controller.append(x[:m_p])
        X = cont.get_predicted_traj()
        path_traj = np.vstack([
            X[:, aux_controller_steps:].T, X[:, -1][None].repeat(repeats=aux_controller_steps, axis=0)
        ]
        )[:, :m_p]
        if world is not None:
            coll_check_qs = np.vstack([path_aux_controller, path_traj])
            for q in coll_check_qs:
                if not world.is_collision_free_gt(q):
                    status = "collision"
                    break
        in_goal = np.linalg.norm(x - x_g) < goal_tol
        if verbose:
            dist_2_goal = np.linalg.norm(x[:m_p] - x_g[:m_p])
            print(
                ts,
                "dist_2_goal: ", np.round(dist_2_goal, 2),
                "progress_path:", np.round(i_max_goal / path_centers.shape[0], 4),
                "comp_time", np.round(time_e * 1e3, 2),
                "radii", np.round(rs[0], 2),
                "||v||", np.round(np.linalg.norm(x[m_p:]), 4),
                "||traj||", np.round(np.linalg.norm(path_traj[1:] - path_traj[:-1], axis=-1).sum(), 2),
                "in goal", in_goal,
                "status", status
            )
        if status == "collision":
            break
        if in_goal:
            status = "success"
            break
    return (status, ts), all_results

