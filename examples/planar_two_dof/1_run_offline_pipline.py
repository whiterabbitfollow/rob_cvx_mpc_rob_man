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


import sys
import logging
from logging import StreamHandler
from logging.handlers import RotatingFileHandler

import numpy as np

from examples.planar_two_dof import P_EXAMPLE_2_DOF
from problem_scenario import ProblemScenarioMassAllPin

from offline_pipeline_a_disturbances import run_pipeline
from offline_pipeline_b_controller import run_controller_pipeline
from offline_pipeline_c_compute_constants import compute_converged_constants

P_CACHED_ROOT_DIR = P_EXAMPLE_2_DOF / "data"
P_CACHED_ROOT_DIR.mkdir(exist_ok=True, parents=True)


def get_problem_two_dof(dir_name, u_amp_nom, v_amp_nom, error_frac):
    p_urdf = P_EXAMPLE_2_DOF / "manipulator"  / "two_link.urdf"
    p_cached_dir = P_CACHED_ROOT_DIR / dir_name
    torque_limits = np.r_[20, 20]
    p_cached_dir.mkdir(exist_ok=True, parents=True)
    return ProblemScenarioMassAllPin(
        str(p_urdf),
        p_cached_dir,
        torque_limits,
        u_amp_nom=u_amp_nom,
        v_amp_nom=v_amp_nom,
        dt=1e-2,
        error_frac=error_frac,
        ignore_gravity=True
    )


def run_two_dof(error_frac, stream_to_console=True):
    dir_name = f"dof_2_ef_{error_frac}"
    ps = get_problem_two_dof(
        dir_name=dir_name,
        error_frac=error_frac,
        u_amp_nom=20,
        v_amp_nom=2
    )
    ps.dump_to_cached_dir()
    logger = logging.getLogger(dir_name)
    logger.setLevel(logging.DEBUG)
    if stream_to_console:
        handler = StreamHandler(sys.stdout)
    else:
        handler = RotatingFileHandler(ps.p_cached_dir / "offline.log")
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    path_results = ps.p_cached_dir
    run_pipeline(
        path_results=path_results,
        problem=ps,
        dt=ps.dt,
        logger=logger,
        nr_samples_m=1_000,
        nr_samples_q=1_000,
        nr_samples_disc_err=int(1e6)
    )
    run_controller_pipeline(path_results, ps, logger)
    compute_converged_constants(ps, logger, nr_samples=2_000)
    logger.removeHandler(handler)
    handler.close()


if __name__ == "__main__":
    run_two_dof(error_frac=0.10)