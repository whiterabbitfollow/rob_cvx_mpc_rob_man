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


import logging
import pickle
import sys

import numpy as np
import pandas as pd

from controllers.common import run_controller_in_corridor
from controllers.flexible_tube_controller import FlexibleTubeNaiveCorridorController
from controllers.nom_controller import NominalController
from controllers.tube_controller import RobustCorridorController
from aux import interpolate_equidistant
from examples.planar_two_dof import P_EXAMPLE_2_DOF
from examples.planar_two_dof.world import DiskWorld
from path_planner.rrt_bi_dir import is_collision_free_bisection, rrt_bi_dir_plan, simple_shortcut
from problem_scenario import ProblemScenarioMassAllPin



def sample_query(world, dist_margin=0):
    while True:
        q_s = world.sample_coll_free()
        r_s = world.sdf(q_s)
        if r_s < dist_margin:
            continue
        q_g = world.sample_coll_free()
        r_g = world.sdf(q_g)
        if r_g < dist_margin:
            continue
        status = is_collision_free_bisection(q_s, q_g, world.sdf)
        if not status:
            break
    return q_s, q_g


def run_experiments_from_dir(p_d, nr_runs=1, save_runs=False, seed=0):
    d_name = p_d.name
    p_cached_dir = p_d
    ps = ProblemScenarioMassAllPin.from_cached_dir(
        p_cached_dir
    )
    logger = logging.getLogger(d_name)
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    config_dim = ps.config_dim
    m_x = config_dim * 2
    m_u = config_dim
    m_p = config_dim

    Q = np.eye(m_x) * 10
    Q[config_dim:, config_dim:] *= .01
    Q_e = np.eye(m_x) * 1e4
    R = np.eye(m_u) * 1e-3

    N = 20
    goal_tol = 0.01

    fcont = FlexibleTubeNaiveCorridorController.from_cached_dir(
        p_cached_dir,
        Q,
        Q_e,
        R,
        N=N
    )
    rcont = RobustCorridorController.from_cached_dir(
        p_cached_dir,
        Q,
        Q_e,
        R,
        N
    )
    nom_cont = NominalController(
        fcont.A, fcont.B, Q, Q_e, R, N, u_lim=ps.u_amp_nom, v_lim=ps.v_amp_nom, p_amp=np.pi,
    )
    np.random.seed(seed)
    controllers = [
        fcont,
        rcont,
        nom_cont,
        nom_cont
    ]
    extra_kwargs = [
        None, #set {"aux_controller_steps": 4}, if aux steps needed
        None,
        None,
        {"ignore_me": True},
    ]
    names = [
        "ft",
        "rt",
        "nom",
        "nom_star"
    ]
    corridor_results = []
    iter_obj = list(range(nr_runs))
    world = DiskWorld.from_example()
    def is_collision_free_transition(q_1, q_2):
        return is_collision_free_bisection(q_1, q_2, sdf_func=world.sdf)
    for run_nr in iter_obj:
        ps.set_randomized_gt_model()
        d_margin = 0.1
        q_s, q_g = sample_query(world, dist_margin=d_margin)
        world.dist_margin = d_margin
        _, _, path_rrt = rrt_bi_dir_plan(
            q_s,
            q_g,
            trans_check_func=is_collision_free_transition,
            sample_func=world.sample_coll_free,
            max_iter=100
        )
        if path_rrt.size == 0:
            print(f"No path found {run_nr}")
            continue
        path_centers = interpolate_equidistant(path_rrt, delta=0.001)
        path_radii = world.sdf(path_centers)
        try:
            path_centers, _ = simple_shortcut(path_centers, path_radii)
        except:
            print(f"Short-cutting failed {run_nr}")
            continue
        path_centers = interpolate_equidistant(path_centers, delta=0.001)
        path_radii = world.sdf(path_centers)
        world.dist_margin = 0
        logger.debug(f"---run {run_nr}")
        for cont, extra_kw, name in zip(controllers, extra_kwargs, names):
            if cont is None:
                continue
            extra_kw = extra_kw or {}
            (status, ts_goal), all_results = run_controller_in_corridor(
                ps,
                cont,
                path_centers,
                path_radii,
                world=world,
                nr_steps=1000,
                ignore_gravity=True,
                goal_tol=goal_tol,
                verbose=False,
                **extra_kw
            )
            if len(all_results) > 1:
                comp_times = np.vstack(
                    [
                        (time_e, time_e_corridor_planning, time_mpc_e)
                        for *_, (time_e, time_e_corridor_planning, time_mpc_e) in all_results
                    ]
                )
                mean_time_all, mean_time_corr, mean_time_mpc = comp_times[1:].mean(axis=0)
            else:
                mean_time_all, mean_time_corr, mean_time_mpc = 3 * [np.nan]
            nr_steps = len(all_results)
            logger.debug(f"{name} time in goal: {status} {ts_goal} nr_steps: {nr_steps}")
            if save_runs:
                p_res = P_EXAMPLE_2_DOF / "debug" / p_cached_dir.name / name
                if not p_res.exists():
                    p_res.mkdir(exist_ok=True, parents=True)
                if p_res.exists():
                    cont_f_name = f'{name}~{run_nr}.pckl'
                    with (p_res / cont_f_name).open("wb") as fp:
                        pickle.dump((path_centers, all_results), fp)
            corridor_results.append(
                {
                    "run": run_nr,
                    "name": name,
                    "t2g": ts_goal,
                    "comp_time_ee": mean_time_all,
                    "comp_time_corr": mean_time_corr,
                    "comp_time_mpc": mean_time_mpc,
                    "status": status
                }
            )
    df = pd.DataFrame(corridor_results)
    df.to_csv(p_cached_dir / "results.csv", index=False)
    path_str = str(p_cached_dir / "results.csv")
    logger.debug(f"saved all results at {path_str}")
    logger.removeHandler(handler)
    handler.close()


if __name__ == "__main__":
    p_d = P_EXAMPLE_2_DOF / "data" / "dof_2_ef_0.1"
    run_experiments_from_dir(p_d, nr_runs=10, save_runs=True)