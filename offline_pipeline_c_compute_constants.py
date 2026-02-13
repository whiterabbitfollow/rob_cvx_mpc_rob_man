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
import pathlib

import numpy as np
from scipy.integrate import solve_ivp

from aux import (
    compute_model_error,
    get_hyper_box_corners,
    get_hyper_box_corners_from_limits
)


def compute_converged_constants(problem, logger, nr_samples=1000, err_tol=1e-5):
    path_results = problem.p_cached_dir
    data_system = np.load(path_results / "system.npz")
    A, B = [data_system[name] for name in ("A", "B")]
    data = np.load(path_results / "data_constraints.npz")
    u_amp, v_amp = data["u_amp"], data["v_amp"]
    data_ftube = np.load(path_results / "ftube_controller.npz")
    P_sqrt = data_ftube["P_sqrt"]
    a, b, c = get_model_and_disc_error_constants(
        problem,
        A,
        B,
        P_sqrt,
        u_amp,
        v_amp,
        logger,
        nr_samples=nr_samples,
        err_tol=err_tol,
        max_iter_cnt=np.inf
    )
    np.savez(path_results / "data_ME_and_DE_constants.npz", a=a, b=b, c=c)


def get_default_position_limits(n_conf):
    return np.c_[-np.ones(n_conf), np.ones(n_conf)].T * np.pi


def get_model_and_disc_error_constants(
        problem,
        A,
        B,
        P_sqrt,
        u_amp,
        v_amp,
        logger,
        nr_samples=1000,
        a = -np.inf,
        b = -np.inf,
        c = -np.inf,
        err_tol=1e-5,
        max_iter_cnt=np.inf
):
    P_sqrt_B = P_sqrt @ B
    dyn_nom = problem.get_nominal_dynamics()
    n_conf = problem.config_dim
    dt = problem.dt
    position_limits = problem.position_limits
    max_diff = np.inf

    cube = get_hyper_box_corners(n_conf, value=1)
    if position_limits is None:
        position_limits = get_default_position_limits(n_conf)
    conf_cube = get_hyper_box_corners_from_limits(position_limits.T)
    v_amp_cube = cube * v_amp
    u_verts = cube * u_amp
    logger.debug(f"Computing M.E and D.E constants | Initial values a: {a} b: {b} c: {c}")
    ratios_wc = 1 + get_hyper_box_corners(problem.parameter_dim, value=problem.frac)
    n_wc_ratios = ratios_wc.shape[0]
    n_total = nr_samples + n_wc_ratios
    time_s = time.time()
    cnt = 0
    while max_diff > err_tol and cnt < max_iter_cnt:
        a_old, b_old, c_old = a, b, c
        cnt += 1
        for i in range(n_total):
            if i < n_wc_ratios:
                dyn_err = problem.get_err_dyn_from_ratios(ratios_wc[i])
            else:
                dyn_err = problem.get_err_dyn_random()
            Qs = np.vstack(
                [
                    np.random.uniform(position_limits[0], position_limits[1], size=(nr_samples, n_conf)),
                    conf_cube,
                ]
            )
            Qds = np.vstack(
                [
                    v_amp_cube,
                    np.random.uniform(-v_amp, v_amp, size=(nr_samples, n_conf))
                 ]
            )
            for q, q_d in zip(Qs, Qds):
                M_nom = dyn_nom.mass_matrix(q)
                M_err = dyn_err.mass_matrix(q)

                M = M_nom + M_err
                M_inv = np.linalg.inv(M)
                M_tilde = M_inv @ M_err
                a = max(a, np.linalg.norm(P_sqrt_B @ M_tilde, ord=2))

                C_err = dyn_err.coroli_matrix(q, q_d)
                C_tilde = M_inv @ C_err
                b = max(b, np.linalg.norm(P_sqrt_B @ C_tilde, ord=2))

                grav_err = dyn_err.gravity(q)
                grav_tilde = M_inv @ grav_err

                def x_dot_fl(t, x_t, q_dd_t):
                    # simulate continuous FL with model-error, eq (3)
                    q_t, q_d_t = np.split(x_t, 2)
                    # compute model-error, i.e. delta_theta eq (4)
                    delta_theta_t = compute_model_error(dyn_nom, dyn_err, q_t, q_d_t, q_dd_t)
                    x_t_dot = np.r_[
                        q_d_t,
                        q_dd_t + delta_theta_t
                    ]
                    return x_t_dot

                acc = np.random.uniform(-u_amp, u_amp, size=(n_conf,))
                x = np.r_[q, q_d]
                # simulate continuous ODE (with model error)
                results = solve_ivp(x_dot_fl, [0, dt], x, args=(acc,))
                x_ode = results.y[:, -1]
                # simulate discrete time dynamics
                delta_theta = compute_model_error(dyn_nom, dyn_err, q, q_d, acc)
                x_discrete = A @ x + B @ (acc + delta_theta)  # eq (5) (without discretization error)
                # compute discretization error
                disc_err = x_ode - x_discrete
                c = max(c, np.linalg.norm(P_sqrt_B @ grav_tilde + P_sqrt @ disc_err, ord=2))
            if (i % 100) == 0:
                logger.debug(f"{cnt} [{i} / {n_total}] time_elapsed: {time.time() - time_s:0.4f}")
        max_diff = max(a - a_old, b - b_old, c - c_old)
        logger.debug(
            f"{cnt} | "
            f"coeffs max_diff: {max_diff : 0.6f} | "
            f"time_elapsed: {time.time() - time_s : 0.4f}"
        )
    logger.debug(
        f"DONE | a: {a:0.6f}, b: {b:0.6f}, c: {c:0.6f}, max_diff: {max_diff:0.4f}"
    )
    return a, b, c
