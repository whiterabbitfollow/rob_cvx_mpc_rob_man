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
import cvxpy as cp

from aux import compute_projected_matrix, get_linear_ordered_state_constraints, get_linear_box_constraints


class FlexibleTubeNaiveCorridorController:

    """
        Implementation of our suggested controller, referred to as flexible tube in paper
    """

    def __init__(
            self,
            A,
            B,
            Q,
            Q_e,
            R,
            N,
            A_x,
            b_x,
            A_u,
            b_u,
            rho,
            L,
            a,
            b,
            c,
            P,
            P_sqrt,
            P_inv_sqrt,
            K,
            w_s,
            ball_dim=None,
            eps=1e-6
    ):
        P_sqrt_B = np.linalg.norm(P_sqrt @ B, ord=2)
        m_x, m_u = B.shape
        m_p = int(m_x / 2)
        self.m_p = m_p
        self.ball_dim = ball_dim or m_p
        P_ = compute_projected_matrix(P, m_p)
        eig_vals, eig_vecs = np.linalg.eig(P_)
        r_p_0 = np.sqrt(1 / eig_vals).max()

        X = cp.Variable((m_x, N + 1), name="X")
        U = cp.Variable((m_u, N), name="U")
        S = cp.Variable((N + 1, ), name="S")
        x_0 = cp.Parameter(m_x, name="x_0")
        x_g_v = cp.Parameter(m_x, name="x_g_v")

        self.rho = rho
        rho_d = rho + L
        self.rho_d = rho_d
        assert rho_d < 1, "rho + L has to be smaller than 1"
        objective = 0
        for i in range(N):
            objective += cp.quad_form(X[:, i] - X[:, -1], Q) + cp.quad_form(U[:, i], R)
        objective += cp.quad_form(X[:, -1] - x_g_v, Q_e)
        objective += S[:-1].sum() * w_s
        objective += S[-1] * (w_s / (1 - rho_d))

        self.c_x = np.linalg.norm(P_inv_sqrt.T @ A_x.T, axis=0)
        self.c_u = np.linalg.norm((K @ P_inv_sqrt).T @ A_u.T, axis=0)
        self.b_x = b_x
        self.b_u = b_u

        cs  = cp.Parameter((self.ball_dim, N+1), name="cs")
        rs  = cp.Parameter((N+1, ), name="rs")

        const = ([
            cp.norm(P_sqrt @ (X[:, 0] - x_0)) <= S[0],
            X[:, 1:] == A @ X[:, :-1] + B @ U,
            X[m_p:, -1] == np.zeros(m_p, ),
            A_u @  U <= b_u[:, None] - cp.multiply(self.c_u[:, None], S[None, :-1]),
            A_x @ X[:, :-1] <= b_x[:, None] - cp.multiply(self.c_x[:, None], S[None, :-1]),
            A_x @ X[:, -1] <= b_x - self.c_x * (S[-1] + eps),
            cp.norm(X[:m_p] - cs, axis=0) <= rs - r_p_0 * S,
            (
                    S[1:]
                    >=
                    (
                            S[:-1] * rho_d
                            +
                            P_sqrt_B * (a * cp.norm(U, axis=0) + b * cp.norm(X[m_p:, :-1], axis=0) + c)
                    )
            ),
        ]
        )
        self.r_p_0 = r_p_0
        self.cs = cs
        self.rs = rs
        self.x_g_v = x_g_v
        self.P = P
        self.A = A
        self.B = B
        self.K = K
        self.N = N
        self.A_x = A_x
        self.b_x = b_x
        self.A_u = A_u
        self.b_u = b_u
        self.x_0 = x_0
        self.S = S
        self.X = X
        self.U = U
        self.P_sqrt = P_sqrt
        self.a = a
        self.b = b
        self.c = c
        self.problem = cp.Problem(
            cp.Minimize(objective),
            constraints=const
        )

    def compute_model_error_bound(self, v, x):
        return self.a * np.linalg.norm(v) + self.b * np.linalg.norm(x[:self.m_p])

    @classmethod
    def from_cached_dir(
            cls,
            p_cached_dir,
            Q,
            Q_e,
            R,
            N,
            weight_balls=1,
            ignore_gravity=True,
            **kwargs
    ):
        data_controller = np.load(p_cached_dir / "ftube_controller.npz")
        data_constraints = np.load(p_cached_dir / "data_constraints.npz")
        data_model_error = np.load(p_cached_dir / "data_model_error.npz")
        data_system = np.load(p_cached_dir / "system.npz")
        A, B = data_system["A"], data_system["B"]
        P, E, rho, L, K, P_sqrt, P_inv_sqrt = [data_controller[name] for name in ("P", "E", "rho", "L", "K", "P_sqrt", "P_inv_sqrt")]
        m_u, m_x = K.shape
        m_p = int(m_x / 2)
        v_amp, u_amp = data_constraints["v_amp"], data_constraints["u_amp"]
        a_me = data_model_error["a"]
        b_me = data_model_error["b"]
        if not ignore_gravity:
            c_me = data_model_error["c"]
        else:
            c_me = 0

        A_x, b_x = get_linear_ordered_state_constraints(m_p, p_amp=np.pi, v_amp=v_amp)
        A_u, b_u = get_linear_box_constraints(m_u, amp=u_amp)
        return cls(
            A,
            B,
            Q,
            Q_e,
            R,
            N,
            A_x,
            b_x,
            A_u,
            b_u,
            rho,
            L,
            a_me,
            b_me,
            c_me,
            P,
            P_sqrt,
            P_inv_sqrt,
            K,
            w_s=weight_balls,
            **kwargs
        )

    def solve(self, x_0, cs, rs, x_g_v):
        self.x_0.value = x_0
        self.x_g_v.value = x_g_v
        self.cs.value = cs[:self.ball_dim]
        self.rs.value = rs
        self.problem.solve()
        return self.X.value is not None

    def get_solved_control(self, x, k = 0):
        x_nom = self.X.value[:, k]
        u_nom = self.U.value[:, k]
        e = (x - x_nom)
        u = u_nom + self.K @ e
        return u

    def get_predicted_traj(self):
        return self.X.value

    def get_solution_results(self):
        return self.X.value, self.U.value, self.S.value, self.r_p_0

    def get_solution_inputs(self):
        return self.x_0.value, self.x_g_v.value, self.cs.value, self.rs.value

