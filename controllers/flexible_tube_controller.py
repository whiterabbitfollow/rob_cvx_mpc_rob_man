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
            ball_dim=None,
            eps=1e-6
    ):
        P_sqrt_B = np.linalg.norm(P_sqrt @ B, ord=2)
        self.P_sqrt_B = P_sqrt_B
        self.use_solver = cp.MOSEK
        m_x, m_u = B.shape
        m_p = int(m_x / 2)
        self.m_p = m_p
        self.ball_dim = ball_dim or m_p
        P_ = compute_projected_matrix(P, m_p)
        eig_vals, eig_vecs = np.linalg.eig(P_)
        r_p_0 = np.sqrt(1 / eig_vals).max()

        X = cp.Variable((m_x, N + 1), name="X")
        U = cp.Variable((m_u, N), name="U")
        S = cp.Variable((N + 1,), name="S")

        x_0 = cp.Parameter(m_x, name="x_0")
        x_g_v = cp.Parameter(m_x, name="x_g_v")
        s_0 = cp.Parameter(name="s_0")
        self.rho = rho
        rho_d = rho + L
        self.rho_d = rho_d

        delta_f = c / (1 - rho_d)

        assert rho_d < 1, "rho + L has to be smaller than 1"
        objective = 0
        for i in range(N):
            objective += cp.quad_form(X[:, i] - X[:, -1], Q) + cp.quad_form(U[:, i], R)
        objective += cp.quad_form(X[:, -1] - x_g_v, Q_e)

        self.c_x = np.linalg.norm(P_inv_sqrt.T @ A_x.T, axis=0)
        self.c_u = np.linalg.norm((K @ P_inv_sqrt).T @ A_u.T, axis=0)
        self.b_x = b_x
        self.b_u = b_u

        cs = cp.Parameter((self.ball_dim, N + 1), name="cs")
        rs = cp.Parameter((N + 1,), name="rs")

        const = ([
            cp.norm(P_sqrt @ (X[:, 0] - x_0)) <= S[0] - s_0,
            X[:, 1:] == A @ X[:, :-1] + B @ U,
            X[m_p:, -1] == np.zeros(m_p, ),
            A_u @ U <= b_u[:, None] - cp.multiply(self.c_u[:, None], S[None, :-1]),
            A_x @ X[:, :-1] <= b_x[:, None] - cp.multiply(self.c_x[:, None], S[None, :-1]),
            A_x @ X[:, -1] <= b_x - self.c_x * (S[-1] + eps),
            cp.norm(X[:m_p] - cs, axis=0) <= rs - r_p_0 * S,
            (
                    S[1:]
                    >=
                    (
                            S[:-1] * rho_d
                            +
                            (a * cp.norm(U, axis=0) + b * cp.norm(X[m_p:, :-1], axis=0) + c)
                    )
            ),
            S[-1] >= delta_f
        ]
        )
        self.s_0 = s_0
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
        self.delta_f = delta_f
        self.use_compiled = False
        self.problem = cp.Problem(
            cp.Minimize(objective),
            constraints=const
        )

    def set_solver(self, solver):
        self.use_solver = solver

    def compute_model_error_bound(self, u, x):
        return self.a * np.linalg.norm(u) + self.b * np.linalg.norm(x[self.m_p:]) + self.c

    @classmethod
    def from_cached_dir(
            cls,
            p_cached_dir,
            Q,
            Q_e,
            R,
            N,
            **kwargs
    ):
        data_controller = np.load(p_cached_dir / "ftube_controller.npz")
        data_constraints = np.load(p_cached_dir / "data_constraints.npz")
        data_model_error = np.load(p_cached_dir / "data_ME_and_DE_constants.npz")
        data_system = np.load(p_cached_dir / "system.npz")
        A, B = data_system["A"], data_system["B"]
        P, E, rho, L, K, P_sqrt, P_inv_sqrt = [data_controller[name] for name in
                                               ("P", "E", "rho", "L", "K", "P_sqrt", "P_inv_sqrt")]
        m_u, m_x = K.shape
        m_p = int(m_x / 2)
        v_amp, u_amp = data_constraints["v_amp"], data_constraints["u_amp"]
        a_me = data_model_error["a"]
        b_me = data_model_error["b"]
        c_me = data_model_error["c"]
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
            **kwargs
        )

    def set_parameter_values(self, x_0, cs, rs, x_g_v, s_0):
        self.x_0.value = x_0
        self.cs.value = cs[:self.ball_dim]
        self.rs.value = rs
        self.x_g_v.value = x_g_v
        self.s_0.value = s_0

    def solve(self):
        if self.use_compiled:
            self.problem.solve(method='CPG', verbose=False)  # updated_params = ['A', 'b']
        else:
            self.problem.solve(solver=self.use_solver)
        return self.X.value is not None

    def get_solved_control(self, x, k=0):
        x_nom = self.X.value[:, k]
        u_nom = self.U.value[:, k]
        e = (x - x_nom)
        u = u_nom + self.K @ e
        return u

    def predict_tube_size(self, x_0, M):
        S = self.S.value
        U_n = np.linalg.norm(self.U.value, axis=0)
        Xp_n = np.linalg.norm(self.X.value[self.m_p:, :-1], axis=0)
        betas = (self.a * U_n + self.b * Xp_n + self.c)
        s = np.linalg.norm(self.P_sqrt @ (self.X.value[:, 0] - x_0))
        for m in range(M):
            s = s * self.rho_d + betas[m]
        return s

    def get_predicted_traj(self):
        return self.X.value

    def get_predicted_state(self, k):
        return self.X.value[:, k]

    def get_solution_results(self):
        return self.X.value, self.U.value, self.S.value, self.r_p_0

    def get_solution_inputs(self):
        return self.x_0.value, self.x_g_v.value, self.cs.value, self.rs.value, self.s_0.value

    def __str__(self):
        return f"flexible({self.N})"
