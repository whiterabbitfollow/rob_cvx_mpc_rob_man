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


import multiprocessing
import time
import pathlib
import pickle
import traceback
from functools import partial
from itertools import product

import cvxpy as cp
import numpy as np
from scipy.linalg import sqrtm
from scipy.spatial import ConvexHull
import tqdm

from aux import (
    get_linear_box_constraints,
    get_linear_ordered_state_constraints,
    get_linear_double_integrator_discrete_dynamics
)


def verify_torque_limits(problem, logger, nr_samples_q = 1000):
    u_amp_nom = problem.u_amp_nom
    v_amp_nom = problem.v_amp_nom
    tau_lims = problem.torque_limits
    dyn_nom = problem.dyn_nom
    n_conf = problem.config_dim
    n_u = n_conf
    Q = np.random.uniform(-np.pi, np.pi, size=(n_conf, nr_samples_q))
    Q_d = np.random.uniform(-v_amp_nom, v_amp_nom, size=(n_conf, nr_samples_q))
    ns = np.c_[[dyn_nom.gravity(q) + dyn_nom.coroli(q, q_d) for q, q_d in zip(Q.T, Q_d.T)]]
    is_certified = False
    cube = np.vstack(list(product(*([(-1, 1)] * n_u))))
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
        ignore_gravity=True,
        a=-np.inf,
        b=-np.inf,
        c=-np.inf,
        verts_wc=None
):
    """
    :param problem: problem scenario
    :param v_amp: amplitude of velocity
    :param u_amp: amplitude of acceleration
    :param nr_samples_m: batch size used to form estimate
    :param nr_samples_q: nr of samples to use in each batch iteration
    :param logger:
    :param err_tol: when the largest abs-difference of all estimates is smaller than err_tol, its defined as having converged
    :param ignore_gravity:
    :param a: acceleration constant
    :param b: velocity constant
    :param c: gravity constant
    :param verts_wc: worst case model-error set
    :return:
    """
    dyn_nom = problem.dyn_nom
    n_conf = problem.config_dim

    max_diff = np.inf
    cnt = 0
    cube = np.vstack(list(product(*([(-1, 1)] * n_conf))))

    conf_cube = cube * np.pi
    v_amp_cube = cube * v_amp
    u_verts = cube * u_amp
    if verts_wc is None:
        verts_wc = np.zeros(n_conf)
    logger.debug("WC bound computation starting")
    logger.debug(f"a: {a} b: {b} c: {c}")
    logger.debug(f"verts_wc: {verts_wc}")
    ratios_wc = problem.get_ratios_parameters_vertices()
    n_wc_ratios = ratios_wc.shape[0]
    n_total = nr_samples_m + n_wc_ratios
    time_s = time.time()

    while max_diff > err_tol:
        a_old, b_old, c_old = a, b, c
        verts_wc_old = verts_wc.copy()
        cnt += 1
        for i in range(n_total):
            if i < n_wc_ratios:
                dyn_err_i = problem.get_error_dynamics_from_ratios(ratios_wc[i])
            else:
                dyn_err_i = problem.sample_error_dynamics()
            Qs = np.vstack(
                [
                    np.random.uniform(-np.pi, np.pi, size=(nr_samples_q, n_conf)),
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
                M_err = dyn_err_i.mass_matrix(q)

                M = M_nom + M_err
                M_inv = np.linalg.inv(M)
                M_tilde = M_inv @ M_err
                a = max(a, np.linalg.norm(M_tilde, ord=2))
                delta_acc_ = - M_tilde @ u_verts.T

                C_err = dyn_err_i.coroli_matrix(q, q_d)
                C_tilde = M_inv @ C_err
                b = max(b, np.linalg.norm(C_tilde, ord=2))
                delta_cor = - C_tilde @ q_d

                grav_err = dyn_err_i.gravity(q)
                grav_tilde = M_inv @ grav_err
                c = max(c, np.linalg.norm(grav_tilde, ord=2))
                delta_grav = - grav_tilde

                if not ignore_gravity:
                    delta_ = delta_acc_.T + delta_cor + delta_grav
                else:
                    delta_ = delta_acc_.T + delta_cor
                deltas.extend(delta_)
            deltas = np.vstack(deltas)
            verts_ = np.abs(deltas).max(axis=0)
            verts_wc = np.vstack([verts_, verts_wc]).max(axis=0)
            if (i % 10) == 0:
                logger.debug(f"{cnt} [{i} / {n_total}] time_elapsed: {time.time() - time_s:0.4f}")
        max_diff_wc = np.abs(verts_wc - verts_wc_old).max()
        max_diff_coeffs = max(abs(a - a_old), abs(b - b_old), abs(c - c_old))
        logger.debug(
            f"{cnt} verts_wc max_diff: {max_diff_wc:0.4f} "
            f"coeffs max_diff: {max_diff_coeffs:0.4f} "
            f"time_elapsed: {time.time() - time_s:0.4f}"
        )
        max_diff = max(max_diff_wc, max_diff_coeffs)
    return (a, b, c), verts_wc


def run_pipeline(
        path_results : pathlib.Path,
        problem,
        dt,
        logger,
        nr_samples_m=1000,
        nr_samples_q=1000,
        err_tol=1e-3
):
    path_results.mkdir(exist_ok=True, parents=True)
    A, B = get_linear_double_integrator_discrete_dynamics(
        problem.config_dim, dt=dt, method="euler"
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
    path_f_me = path_results / "data_model_error.npz"


    if path_f_me.exists():
        data = np.load(path_f_me)
        a_old, b_old, c_old = data["a"], data["b"], data["c"]
    else:
        a_old, b_old, c_old = [-np.inf] * 3

    path_f_verts = path_results / "data_model_error_verts.npz"
    if path_f_verts.exists():
        data = np.load(path_f_verts)
        verts_wc_old = data["verts_acc"]
    else:
        verts_wc_old = None

    (a, b, c), verts_wc = compute_bound_and_disturbance_set(
        problem,
        v_amp=v_amp,
        u_amp=u_amp,
        nr_samples_m=nr_samples_m,
        nr_samples_q=nr_samples_q,
        logger=logger,
        err_tol=err_tol,
        a=a_old,
        b=b_old,
        c=c_old,
        verts_wc=verts_wc_old
    )
    np.savez(path_f_me, a=a, b=b, c=c)
    np.savez(path_f_verts, verts_acc=verts_wc)


def run_controller_pipeline(
            path_results: pathlib.Path,
            problem,
            logger
    ):
    data_system = np.load(path_results / "system.npz")
    A = data_system["A"]
    B = data_system["B"]
    data = np.load(path_results / "data_constraints.npz")
    u_amp, v_amp = data["u_amp"], data["v_amp"]
    data = np.load(path_results / "data_model_error_verts.npz")
    verts_acc = data["verts_acc"]
    path_f = path_results / "data_controllers.pckl"
    if not path_f.exists():
        controller_results = compute_controllers(
            problem, A, B, verts_acc, u_amp, v_amp, nr_rhos=20, nr_workers=10
        )
        with path_f.open("wb") as fp:
            pickle.dump(controller_results, fp)
    pick_ftube_controller_from_results(
        problem.p_cached_dir, logger
    )
    pick_rtube_controller_from_results(
        problem.p_cached_dir, logger
    )


def compute_controllers(problem, A, B, verts_acc, u_amp, v_amp, nr_rhos=20, nr_workers=10):
    n_conf = problem.config_dim
    m_x = n_conf * 2
    wc_x = B @ verts_acc
    verts_w = np.vstack(list(product(*([(-1, 1)] * m_x)))) * wc_x
    m_x, m_u = B.shape
    m_p = int(m_x // 2)
    A_x, b_x = get_linear_ordered_state_constraints(m_p, p_amp=np.pi, v_amp=v_amp)
    A_u, b_u = get_linear_box_constraints(m_u, amp=u_amp)
    rhos = np.linspace(0.8, 0.99, nr_rhos)
    c_p_wht = 1 / 0.1
    c_v_wht = 1 / v_amp
    c_u_wht = 1 / u_amp # 1e2
    apa = partial(
        solve_rpi_with_constraints,
        A,
        B,
        A_x,
        A_u,
        verts_w,
        c_p_wht,
        c_v_wht,
        c_u_wht
    )
    with multiprocessing.Pool(nr_workers) as pool:
        results = []
        for res in tqdm.tqdm(pool.imap(apa, rhos), total=rhos.size):
            results.append(res)
    return results


def solve_rpi_with_constraints(
        A,
        B,
        A_x,
        A_u,
        verts_w,
        c_p_wht,
        c_v_wht,
        c_u_wht,
        rho
):
    m_x, m_u = B.shape

    n_x = A_x.shape[0]
    n_p = int(n_x / 2)

    # Assume position constraints make up the first half of rows in matrix, the other half is velocity constraints.
    # A_x = [[A_p], [A_v]], A_p, A_v \in R^{n_p, n}

    n_u = A_u.shape[0]
    n_w = verts_w.shape[0]

    E = cp.Variable((m_x, m_x), PSD=True)
    Y = cp.Variable((m_u, m_x))

    c_x_sq = cp.Variable((n_x, ))
    c_u_sq = cp.Variable((n_u,))

    w_bar_sq = cp.Variable()
    objective = 1 / (2 * (1 - rho)) * (
            (n_x + n_u) * w_bar_sq
            +
            c_x_sq[:n_p].sum() * c_p_wht
            +
            c_x_sq[n_p:].sum() * c_v_wht
            +
            c_u_sq.sum() * c_u_wht
    )
    const = [
        E >> np.eye(m_x),
        cp.bmat([
            [(rho**2) * E, (A @ E + B @ Y).T],
            [(A @ E + B @ Y), E]
        ]) >> 0,
        c_x_sq >= 0,
        c_u_sq >= 0,
        w_bar_sq >= 0
    ]
    for i in range(n_x):
        const += [
            cp.bmat([
                [c_x_sq[i][None, None], A_x[i, None] @ E],
                [(A_x[i, None] @ E).T,    E],
            ])
            >> 0
        ]
    for i in range(n_u):
        const += [
            cp.bmat([
                [c_u_sq[i][None, None], A_u[i, None] @ Y],
                [(A_u[i, None] @ Y).T, E],
            ]) >> 0
        ]
    for i in range(n_w):
        const += [
            cp.bmat([
                [w_bar_sq[None, None], verts_w[i][None]],
                [verts_w[i][:, None], E],
            ]) >> 0
        ]
    problem = cp.Problem(cp.Minimize(objective), constraints=const)
    try:
        loss = problem.solve(solver=cp.MOSEK, verbose=False)
        status = problem.status
        return status, E.value, Y.value, c_x_sq.value, c_u_sq.value, w_bar_sq.value, rho
    except Exception as e:
        print("Caught exception")
        traceback.print_exc()
        return "failure", None, None, None, None, None, rho


def pick_ftube_controller_from_results(p_dir, logger):
    data_system = np.load(p_dir / "system.npz")
    A, B = data_system["A"], data_system["B"]
    data_wc = np.load(p_dir / "data_model_error.npz")
    a, b, _ = [data_wc[name] for name in "abc"]
    data_const = np.load(p_dir / "data_constraints.npz")
    u_amp, v_amp = data_const["u_amp"], data_const["v_amp"]
    m_x, m_u = B.shape
    m_p = m_u

    A_x, b_x = get_linear_ordered_state_constraints(m_p, p_amp=np.pi, v_amp=v_amp)
    A_u, b_u = get_linear_box_constraints(m_u, amp=u_amp)

    with (p_dir / "data_controllers.pckl").open("rb") as fp:
        results = pickle.load(fp)

    beta_x_a = a * np.sqrt(m_p) * 1 + b * np.sqrt(m_p) * 1
    rhos, Ls = [], []
    fittness_values, outputs = [], []

    for (status, E, Y, c_x_sq, c_u_sq, w_bar_sq, rho) in results:
        if status == "failure" or status == "infeasible":
            continue
        P = np.linalg.inv(E)
        K = Y @ P
        P_sqrt = sqrtm(P)
        P_inv_sqrt = sqrtm(E)
        V = np.zeros((m_x, m_x))
        V[m_p:, m_p:] = np.eye(m_p)
        d = np.linalg.norm(P_sqrt @ B, ord=2)
        L = (
                d *
                (
                        a * np.linalg.norm(K @ P_inv_sqrt, ord=2) + b * np.linalg.norm(V @ P_inv_sqrt, ord=2)
                )
        )
        rho_L = rho + L
        if rho_L < 1:
            s_ = d * beta_x_a / (1 - (rho + L))
        else:
            s_ = np.inf
        c_x = np.linalg.norm(P_inv_sqrt.T @ A_x.T, axis=0)
        c_u = np.linalg.norm((K @ P_inv_sqrt).T @ A_u.T, axis=0)
        angle_ratio_tight = s_ * c_x[:m_x].max() / 0.1
        vel_ratio_tight = s_ *  c_x[m_x:].max() / v_amp
        control_ratio_tight = s_ * c_u.max() / u_amp
        fittness_value = max(control_ratio_tight, angle_ratio_tight, vel_ratio_tight)
        fittness_values.append(fittness_value)
        Ls.append(L)
        rhos.append(rho)
        outputs.append((rho, L, P, P_sqrt, P_inv_sqrt, K, E, Y, c_x_sq, c_u_sq, w_bar_sq))

    rhos = np.r_[rhos]
    Ls = np.r_[Ls]
    rho_L = rhos + Ls
    fittness_values = np.r_[fittness_values]
    mask = rho_L < 1
    indxs, = np.nonzero(mask)
    if indxs.size:
        i = indxs[fittness_values[indxs].argmin()]
        (rho, L, P, P_sqrt, P_inv_sqrt, K, E, Y, c_x_sq, c_u_sq, w_bar_sq) = outputs[i]
        logger.debug(
            f"ftube controller picked rho_tilde: {rho_L[i]: 0.2f}, fittness value: {fittness_values[i]: 0.2f}"
        )
        np.savez(
            p_dir / "ftube_controller.npz",
            rho=rho,
            L=L,
            P=P,
            K=K,
            E=E,
            Y=Y,
            P_sqrt=P_sqrt,
            P_inv_sqrt=P_inv_sqrt,
            c_x_sq=c_x_sq,
            c_u_sq=c_u_sq,
            w_bar_sq=w_bar_sq
        )
    else:
        logger.error("No flexible tube controller was computed")


def pick_rtube_controller_from_results(p_dir, logger):
    data_system = np.load(p_dir / "system.npz")
    A, B = data_system["A"], data_system["B"]
    data_wc = np.load(p_dir / "data_model_error.npz")
    data_const = np.load(p_dir / "data_constraints.npz")
    u_amp, v_amp = data_const["u_amp"], data_const["v_amp"]

    a, b, _ = [data_wc[name] for name in "abc"]
    m_x, m_u = B.shape
    m_p = m_u

    with (p_dir / "data_controllers.pckl").open("rb") as fp:
        results = pickle.load(fp)

    outputs = []
    fittness_values = []

    for (status, E, Y, c_x_sq, c_u_sq, w_bar_sq, rho) in results:
        if status == "failure" or status == "infeasible":
            continue
        P = np.linalg.inv(E)
        K = Y @ P
        P_sqrt = sqrtm(P)
        P_inv_sqrt = sqrtm(E)

        V = np.zeros((m_x, m_x))
        V[m_p:, m_p:] = np.eye(m_p)
        L = (
                np.linalg.norm(P_sqrt @ B, ord=2) *
                (
                        a * np.linalg.norm(K @ P_inv_sqrt, ord=2) + b * np.linalg.norm(V @ P_inv_sqrt, ord=2)
                )
        )
        w_bar = np.sqrt(w_bar_sq)
        delta = w_bar / (1 - rho)
        b_u_tilde = np.sqrt(c_u_sq) * delta
        b_x_tilde = np.sqrt(c_x_sq) * delta

        angle_ratio_tight = b_x_tilde[:m_x].max() / 0.1
        vel_ratio_tight = b_x_tilde[m_x:].max() / v_amp
        control_ratio_tight = b_u_tilde.max() / u_amp
        fittness_value = max(control_ratio_tight, angle_ratio_tight, vel_ratio_tight)
        outputs.append((rho, L, P, P_sqrt, P_inv_sqrt, K, E, Y, c_x_sq, c_u_sq, w_bar_sq))
        fittness_values.append(fittness_value)
    fittness_values = np.r_[fittness_values]
    if fittness_values.size:
        i = fittness_values.argmin()
        (rho, L, P, P_sqrt, P_inv_sqrt, K, E, Y, c_x_sq, c_u_sq, w_bar_sq) = outputs[i]
        logger.debug(
            f"rtube controller picked, fittness value: {fittness_values.min(): 0.2f} rho, {rho:0.2f}"
        )
        np.savez(
            p_dir / "rtube_controller.npz",
            rho=rho,
            L=L,
            P=P,
            K=K,
            E=E,
            Y=Y,
            P_sqrt=P_sqrt,
            P_inv_sqrt=P_inv_sqrt,
            c_x_sq=c_x_sq,
            c_u_sq=c_u_sq,
            w_bar_sq=w_bar_sq
        )
    else:
        logger.debug("No rigid tube controller was computed")


if __name__ == "__main__":
    pass