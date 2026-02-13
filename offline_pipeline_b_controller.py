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


from functools import partial
from itertools import product
import pathlib
import pickle
import multiprocessing
import traceback

import cvxpy as cp
import numpy as np
from scipy.linalg import sqrtm
import tqdm

from aux import (
    get_linear_box_constraints,
    get_linear_ordered_state_constraints
)

from offline_pipeline_c_compute_constants import get_model_and_disc_error_constants


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
    # box set that bounds disturbances due to model errors
    verts_acc = data["verts_acc"]
    # box set that bounds disturbances due to discretization errors
    verts_state_disc = data["verts_state_disc"]
    wc_x = (B @ verts_acc + verts_state_disc)
    path_f = path_results / "data_controllers.pckl"
    # if not path_f.exists():
    controller_results = compute_controllers(
        problem, A, B, wc_x, u_amp, v_amp, nr_rhos=20, nr_workers=10
    )
    with path_f.open("wb") as fp:
        pickle.dump(controller_results, fp)

    pick_ftube_controller_from_results(
        problem, logger
    )

    pick_rtube_controller_from_results(
        problem.p_cached_dir, logger
    )


def compute_controllers(
        problem,
        A,
        B,
        wc_x,
        u_amp,
        v_amp,
        nr_rhos=20,
        nr_workers=10
):
    n_conf = problem.config_dim
    m_x = n_conf * 2
    verts_w = np.vstack(list(product(*([(-1, 1)] * m_x)))) * wc_x
    m_x, m_u = B.shape
    m_p = int(m_x // 2)
    A_x, b_x = get_linear_ordered_state_constraints(m_p, p_amp=np.pi, v_amp=v_amp)
    A_u, b_u = get_linear_box_constraints(m_u, amp=u_amp)
    rhos = np.linspace(0.8, 0.99, nr_rhos)
    c_p_wht = 1 / 0.1
    c_v_wht = 1 / v_amp
    c_u_wht = 1 / u_amp
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
        for res in tqdm.tqdm(pool.imap(apa, rhos), total=rhos.size, desc="Solving optimization problems"):
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


def pick_ftube_controller_from_results(problem, logger):
    p_dir : pathlib.Path = problem.p_cached_dir
    p_f_tube_dir = p_dir / "f_controller_results"
    p_f_tube_dir.mkdir(exist_ok=True, parents=True)
    data_system = np.load(p_dir / "system.npz")
    A, B = data_system["A"], data_system["B"]

    data_const = np.load(p_dir / "data_constraints.npz")
    u_amp, v_amp = data_const["u_amp"], data_const["v_amp"]

    m_x, m_u = B.shape
    m_p = m_u

    A_x, b_x = get_linear_ordered_state_constraints(m_p, p_amp=np.pi, v_amp=v_amp)
    A_u, b_u = get_linear_box_constraints(m_u, amp=u_amp)

    with (p_dir / "data_controllers.pckl").open("rb") as fp:
        results = pickle.load(fp)

    L_and_rhos, fittness_values, outputs = [], [], []

    for i, (status, E, Y, c_x_sq, c_u_sq, w_bar_sq, rho) in enumerate(results):
        if status == "failure" or status == "infeasible":
            continue
        P = np.linalg.inv(E)
        K = Y @ P
        P_sqrt = sqrtm(P)
        P_inv_sqrt = sqrtm(E)
        V = np.zeros((m_x, m_x))
        V[m_p:, m_p:] = np.eye(m_p)
        a, b, c = get_model_and_disc_error_constants(
            problem,
            A,
            B,
            P_sqrt,
            u_amp,
            v_amp,
            logger,
            nr_samples=100,
            max_iter_cnt=1
        )
        np.savez(p_f_tube_dir / f"ME_DE_constants_est_{i}.npz", a=a, b=b, c=c)
        # OLD calculations:
        # d = np.linalg.norm(P_sqrt @ B, ord=2)
        # L = (
        #         d *
        #         (
        #                 a * np.linalg.norm(K @ P_inv_sqrt, ord=2) + b * np.linalg.norm(V @ P_inv_sqrt, ord=2)
        #         )
        # )
        L = a * np.linalg.norm(K @ P_inv_sqrt, ord=2) + b * np.linalg.norm(V @ P_inv_sqrt, ord=2)
        beta_x_a = a * np.sqrt(m_p) * 1 + b * np.sqrt(m_p) * 1 + c
        rho_L = rho + L
        s_ = np.inf
        delta_f = np.inf
        if rho_L < 1:
            s_ = beta_x_a / (1 - rho_L)
            delta_f = c / (1 - rho_L)

        c_x = np.linalg.norm(P_inv_sqrt.T @ A_x.T, axis=0)
        c_u = np.linalg.norm((K @ P_inv_sqrt).T @ A_u.T, axis=0)

        angle_ratio_tight = s_ * c_x[:m_x].max() / 0.1
        vel_ratio_tight = s_ *  c_x[m_x:].max() / v_amp
        control_ratio_tight = s_ * c_u.max() / u_amp

        fittness_value = max(control_ratio_tight, angle_ratio_tight, vel_ratio_tight)
        logger.debug(f"{i}: | rho: {rho} L: {L}, rho_L: {rho_L} delta_f: {delta_f:0.2f} fv: {fittness_value}")

        fittness_values.append(fittness_value)
        L_and_rhos.append((L, rho))
        outputs.append(
            dict(
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
        )
    L_and_rhos = np.vstack(L_and_rhos)
    fittness_values = np.r_[fittness_values]
    mask = L_and_rhos.sum(axis=1) < 1
    indxs, = np.nonzero(mask)
    if indxs.size:
        i = indxs[fittness_values[indxs].argmin()]
        L, rho = L_and_rhos[i]
        logger.debug(
            f"Ftube controller picked: rho: {rho} rho_tilde: {L + rho: 0.2f}, fittness value: {fittness_values[i]: 0.2f}"
        )
        np.savez(
            p_dir / "ftube_controller.npz",
            **outputs[i]
        )
    else:
        logger.error("No flexible tube controller was computed")


def pick_rtube_controller_from_results(p_dir, logger):
    data_system = np.load(p_dir / "system.npz")
    A, B = data_system["A"], data_system["B"]
    data_const = np.load(p_dir / "data_constraints.npz")
    u_amp, v_amp = data_const["u_amp"], data_const["v_amp"]
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
        L = 0
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