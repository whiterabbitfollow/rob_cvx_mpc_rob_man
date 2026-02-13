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

import cvxpy as cp
import numpy as np
import pandas as pd


from controllers.common import load_all_controllers
from aux import interpolate_equidistant, TimeStepIntegratorContinuous, TimeStepIntegratorDiscrete
from corridor_simulators.flexible_tube import FlexibleTubeSimulator
from corridor_simulators.nom import NomSimulator
from corridor_simulators.tube import TubeSimulator
from examples.planar_two_dof import P_EXAMPLE_2_DOF
from examples.planar_two_dof.world import DiskWorld
from path_planner.rrt_bi_dir import is_collision_free_bisection, rrt_bi_dir_plan, simple_shortcut
from problem_scenario import ProblemScenarioMassAllPin



def sample_query(world, dist_margin=0.):
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


def run_experiments_from_dir(p_d, nr_runs=10,  seed=0, goal_tol = 0.01, save_results=True):
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
    if not (ps.p_cached_dir / "ftube_controller.npz").exists():
        return
    fcont, rcont, nom_cont = load_all_controllers(ps, horizon_length=20)
    for c in (fcont, rcont, nom_cont):
        c.set_solver(cp.CLARABEL)
    np.random.seed(seed)
    world = DiskWorld.from_example()
    simulators = [
        FlexibleTubeSimulator(fcont, TimeStepIntegratorContinuous(dt=ps.dt), world, aux_controller_steps=1),
        TubeSimulator(rcont, TimeStepIntegratorContinuous(dt=ps.dt), world, aux_controller_steps=1),
        NomSimulator(nom_cont, TimeStepIntegratorContinuous(dt=ps.dt), world),
        NomSimulator(nom_cont, TimeStepIntegratorDiscrete(A=nom_cont.A, B=nom_cont.B, ignore_me=True), world)
    ]
    names = [
        "ft",
        "rt",
        "nom",
        "nom_star"
    ]
    corridor_statistics = []
    # computation_times = defaultdict(list)

    iter_obj = list(range(nr_runs))

    dyn_nom = ps.get_nominal_dynamics()

    def is_collision_free_transition(q_1, q_2):
        return is_collision_free_bisection(q_1, q_2, sdf_func=world.sdf)


    for run_nr in iter_obj:
        dyn_err = ps.get_err_dyn_random()
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
        if save_results:
            p_r = p_d / "corridors"
            p_r.mkdir(exist_ok=True)
            np.savez(p_r / f"corr_{run_nr}.npz", path_centers=path_centers, path_radii=path_radii)
        for simulator, name in zip(simulators, names):
            if simulator.cont is None:
                continue
            simulator.integrator.set_dyns(dyn_nom, dyn_err)
            (status, ts_goal), all_results = simulator.simulate(
                path_centers,
                path_radii,
                nr_steps=1_000,
                goal_tol=goal_tol,
                verbose=False
            )
            nr_steps = len(all_results)
            if save_results:
                p_r = p_d / "motion"
                p_r.mkdir(exist_ok=True)
                cont_f_name = f'{name}~{run_nr}.pckl'
                with (p_r / cont_f_name).open("wb") as fp:
                    pickle.dump(all_results, fp)
            logger.debug(f"{name} time in goal: {status} {ts_goal} nr_steps: {nr_steps}")
            corridor_statistics.append(
                {
                    "run_nr": run_nr,
                    "name": name,
                    "t2g": ts_goal,
                    "status": status
                }
            )
    df = pd.DataFrame(corridor_statistics)
    df.to_csv(p_cached_dir / "results.csv", index=False)
    path_str = str(p_cached_dir / "results.csv")
    logger.debug(f"saved all results at {path_str}")
    logger.removeHandler(handler)
    handler.close()


if __name__ == "__main__":
    p_d = P_EXAMPLE_2_DOF / "data" / "dof_2_ef_0.1"
    run_experiments_from_dir(p_d, nr_runs=10, save_results=True, seed=0)
