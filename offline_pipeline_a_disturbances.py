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
from scipy.spatial import ConvexHull
from scipy.integrate import solve_ivp

from aux import (
    get_linear_double_integrator_discrete_dynamics,
    compute_model_error,
    get_hyper_box_corners,
    get_hyper_box_corners_from_limits
)


def get_default_position_limits(n_conf):
    return np.c_[-np.ones(n_conf), np.ones(n_conf)].T * np.pi


def run_pipeline(
        path_results : pathlib.Path,
        problem,
        dt,
        logger,
        nr_samples_m=1000,
        nr_samples_q=1000,
        nr_samples_disc_err=int(1e6),
        err_tol=1e-3
):
    path_results.mkdir(exist_ok=True, parents=True)
    A, B = get_linear_double_integrator_discrete_dynamics(
        problem.config_dim, dt=dt, method="zoh"
    )
    A = np.array(A)
    B = np.array(B)
    np.savez(path_results / "system.npz", A=A, B=B)
    path_f = path_results / "data_constraints.npz"
    u_amp, v_amp = verify_torque_limits(
        problem, logger=logger, nr_samples_q=1_00_000
    )
    np.savez(path_f, u_amp=u_amp, v_amp=v_amp)
    data = np.load(path_f)
    u_amp, v_amp = data["u_amp"], data["v_amp"]
    path_f_verts = path_results / "data_model_error_verts.npz"
    if path_f_verts.exists():
        data = np.load(path_f_verts)
        verts_wc_old = data["verts_acc"]
        verts_wc_state_disc_old  = data["verts_state_disc"]
    else:
        verts_wc_old, verts_wc_state_disc_old = None, None
    verts_wc_acc = compute_bound_and_disturbance_set(
        problem,
        v_amp=v_amp,
        u_amp=u_amp,
        nr_samples_m=nr_samples_m,
        nr_samples_q=nr_samples_q,
        logger=logger,
        err_tol=err_tol,
        verts_wc=verts_wc_old
    )
    verts_wc_state_disc = compute_disturbance_set_discretization(
        ps=problem,
        A=A,
        B=B,
        u_amp=u_amp,
        v_amp=v_amp,
        logger=logger,
        nr_samples=nr_samples_disc_err,
        err_em_max_start=verts_wc_state_disc_old
    )
    np.savez(path_f_verts, verts_acc=verts_wc_acc, verts_state_disc=verts_wc_state_disc)


def verify_torque_limits(problem, logger, nr_samples_q = 1000):
    u_amp_nom = problem.u_amp_nom
    v_amp_nom = problem.v_amp_nom
    tau_lims = problem.torque_limits
    dyn_nom = problem.dyn_nom
    n_conf = problem.config_dim
    n_u = n_conf
    position_limits = problem.position_limits
    if position_limits is None:
        position_limits = get_default_position_limits(n_conf)
    Q = np.random.uniform(position_limits[0], position_limits[1], size=(nr_samples_q, n_conf)).T
    Q_d = np.random.uniform(-v_amp_nom, v_amp_nom, size=(n_conf, nr_samples_q))
    ns = np.c_[[dyn_nom.gravity(q) + dyn_nom.coroli(q, q_d) for q, q_d in zip(Q.T, Q_d.T)]]
    is_certified = False
    cube = get_hyper_box_corners(n_u, value=1)
    chull = ConvexHull(cube)
    cube = cube[chull.vertices]
    alpha = 1
    u_amp = u_amp_nom
    logger.debug("Certifying convex-set")
    while not is_certified:
        all_verts_tau = []
        u_amp *= alpha
        verts_u = np.vstack(cube) * u_amp
        for q, n in zip(Q.T, ns):
            M = dyn_nom.mass_matrix(q)
            verts_tau = verts_u @ M.T + n
            all_verts_tau.append(verts_tau)
        all_verts_tau = np.vstack(all_verts_tau)
        taus = np.abs(all_verts_tau).max(axis=0)
        is_certified = (taus <= tau_lims).all()
        if not is_certified:
            # shrink uniformly
            alpha *= 0.99
    logger.debug(f"alpha {alpha}")
    return u_amp, v_amp_nom


def compute_bound_and_disturbance_set(
        problem,
        v_amp,
        u_amp,
        nr_samples_m,
        nr_samples_q,
        logger,
        err_tol=1e-3,
        verts_wc=None
):
    """
    :param problem: problem scenario
    :param v_amp: amplitude of velocity
    :param u_amp: amplitude of acceleration
    :param nr_samples_m: batch size used to form estimate
    :param nr_samples_q: nr of samples to use in each batch iteration
    :param logger:
    :param err_tol: when the largest abs-difference of all estimates is smaller than err_tol, then it's defined as having converged
    :param a: acceleration constant
    :param b: velocity constant
    :param c: gravity constant
    :param verts_wc: worst case model-error set
    :return:
    """
    dyn_nom = problem.get_nominal_dynamics()
    n_conf = problem.config_dim
    position_limits = problem.position_limits
    max_diff = np.inf
    cnt = 0
    cube = get_hyper_box_corners(n_conf, value=1)
    if position_limits is None:
        position_limits = get_default_position_limits(n_conf)
    conf_cube = get_hyper_box_corners_from_limits(position_limits.T)
    v_amp_cube = cube * v_amp
    u_verts = cube * u_amp
    if verts_wc is None:
        verts_wc = np.zeros(n_conf)
    logger.debug("WC bound computation starting")
    logger.debug(f"verts_wc: {verts_wc}")
    ratios_wc = 1 + get_hyper_box_corners(problem.parameter_dim, value=problem.frac)
    n_wc_ratios = ratios_wc.shape[0]
    n_total = nr_samples_m + n_wc_ratios
    time_s = time.time()

    while max_diff > err_tol:
        verts_wc_old = verts_wc.copy()
        cnt += 1
        for i in range(n_total):
            if i < n_wc_ratios:
                dyn_err = problem.get_err_dyn_from_ratios(ratios_wc[i])
            else:
                dyn_err = problem.get_err_dyn_random()
            Qs = np.vstack(
                [
                    np.random.uniform(position_limits[0], position_limits[1], size=(nr_samples_q, n_conf)),
                    conf_cube,
                ]
            )
            Qds = np.vstack(
                [
                    v_amp_cube,
                    np.random.uniform(-v_amp, v_amp, size=(nr_samples_q, n_conf))
                 ]
            )
            deltas = []
            for q, q_d in zip(Qs, Qds):
                M_nom = dyn_nom.mass_matrix(q)
                M_err = dyn_err.mass_matrix(q)

                M = M_nom + M_err
                M_inv = np.linalg.inv(M)
                M_tilde = M_inv @ M_err
                delta_acc_ = - M_tilde @ u_verts.T

                C_err = dyn_err.coroli_matrix(q, q_d)
                C_tilde = M_inv @ C_err
                delta_cor = - C_tilde @ q_d

                grav_err = dyn_err.gravity(q)
                grav_tilde = M_inv @ grav_err
                delta_grav = - grav_tilde
                delta_ = delta_acc_.T + delta_cor + delta_grav
                deltas.extend(delta_)
            deltas = np.vstack(deltas)
            verts_ = np.abs(deltas).max(axis=0)
            verts_wc = np.vstack([verts_, verts_wc]).max(axis=0)
            if (i % 100) == 0:
                logger.debug(f"{cnt} [{i} / {n_total}] time_elapsed: {time.time() - time_s:0.4f}")
        max_diff_wc = np.abs(verts_wc - verts_wc_old).max()
        logger.debug(
            f"{cnt} verts_wc max_diff: {max_diff_wc:0.4f} | "
            f"time_elapsed: {time.time() - time_s:0.4f}"
        )
        max_diff = max_diff_wc
    return verts_wc


def compute_disturbance_set_discretization(
        ps,
        A,
        B,
        u_amp,
        v_amp,
        logger,
        nr_samples=int(1e6),
        err_em_max_start=None
):
    config_dim = ps.config_dim
    dt = ps.dt
    # iters = []
    # errs_max = []
    position_limits = ps.position_limits
    if position_limits is None:
        position_limits = get_default_position_limits(config_dim)
    err_em_max = err_em_max_start if err_em_max_start is not None else np.zeros(config_dim * 2)
    time_s = time.time()
    cnt_found = 0
    ping_freq = int(nr_samples / 10)
    dyn_nom = ps.get_nominal_dynamics()
    for cnt in range(nr_samples):
        # set gt model
        dyn_err = ps.get_err_dyn_random()
        # sample state
        q = np.random.uniform(position_limits[0], position_limits[1], size=config_dim)
        q_d = np.random.uniform(-v_amp, v_amp, size=config_dim)
        x = np.r_[q, q_d]
        # sample control (acceleration)
        acc = np.random.uniform(-u_amp, u_amp, size=(config_dim,))
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
        # simulate continuous ODE (with model error)
        results = solve_ivp(x_dot_fl, [0, dt], x, args=(acc,))
        x_ode = results.y[:, -1]
        # simulate discrete time dynamics
        delta_theta = compute_model_error(dyn_nom, dyn_err, q, q_d, acc)
        x_discrete = A @ x + B @ (acc + delta_theta)    # eq (5) (without discretization error)
        # compute discretization error (box set)
        err_em = np.abs(x_ode - x_discrete)
        if (err_em_max < err_em).any():
            difference = err_em - err_em_max
            i_max = difference.argmax()
            if err_em_max[i_max] > 0:
                rel_diff = difference[i_max] / err_em_max[i_max]
            else:
                rel_diff = np.nan
            diff_max = difference[i_max]
            # save worst case error (element wise)
            err_em_max = np.maximum(err_em, err_em_max)
            logger.debug(f"[{cnt} / {nr_samples}]: WC disc error changed | abs: {diff_max:0.4f} rel: {rel_diff*100:0.2f}, time_elapsed: {time.time() - time_s:0.4f}")
            cnt_found = cnt
            # errs_max.append(err_max)
            # iters.append(i)
        cnt_since_last_imp = cnt - cnt_found
        if cnt_since_last_imp > 0 and (cnt_since_last_imp % ping_freq) == 0:
            logger.debug(f"[{cnt} / {nr_samples}]: WC disc error the same, time_elapsed: {time.time() - time_s:0.4f}")
    return err_em_max
