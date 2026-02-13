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


import cvxpy as cp
import numpy as np

from aux import compute_projected_matrix, get_linear_box_constraints, get_linear_ordered_state_constraints


class RobustCorridorController:

    """
        Implementation of tube mpc, referred to as rigid tube in paper
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
            P,
            P_sqrt,
            K,
            c_x_sq,
            c_u_sq,
            w_bar_sq,
            ball_dim=None
    ):
        m_x, m_u = B.shape
        m_p = int(m_x / 2)
        self.ball_dim = ball_dim or m_p
        self.rho = rho
        w_bar = np.sqrt(w_bar_sq)
        delta = w_bar / (1 - rho)
        P_ = compute_projected_matrix((P / (delta ** 2)), m_p)
        eig_vals, eig_vecs = np.linalg.eig(P_)
        self.N = N
        self.r_p = np.sqrt(1 / eig_vals).max()
        self.delta = delta
        self.P_sqrt = P_sqrt
        self.P = P
        self.K = K
        self.A_k = (A + B @ K)
        b_x_tilde = np.sqrt(c_x_sq) * delta
        b_u_tilde = np.sqrt(c_u_sq) * delta
        Z = cp.Variable((m_x, N + 1))
        V = cp.Variable((m_u, N))
        x_0 = cp.Parameter(m_x, )
        objective = 0
        self.cs = cp.Parameter((self.ball_dim, N + 1))
        self.rs = cp.Parameter((N + 1, ))
        self.zg = cp.Parameter(m_x, )
        for i in range(N):
            objective += cp.quad_form(Z[:, i] - Z[:, -1], Q) + cp.quad_form(V[:, i], R)
        objective += cp.quad_form(Z[:, -1] - self.zg, Q_e)
        b_x_ = b_x - b_x_tilde
        b_u_ = b_u - b_u_tilde
        const = [
            Z[:, 1:] == A @ Z[:, :-1] + B @ V,
            A_x @ Z <= b_x_[:, None],
            A_u @ V <= b_u_[:, None],
            cp.norm(self.P_sqrt @ (Z[:, 0] - x_0)) <= delta,
            Z[m_p:, -1] == 0,
            cp.norm(Z[:self.ball_dim] - self.cs, axis=0) <= self.rs - self.r_p
        ]
        self.A = A
        self.B = B
        self.m_p = m_p
        self.m_x = m_x
        self.m_u = m_u
        self.A_x = A_x
        self.b_x = b_x_
        self.A_u = A_u
        self.b_u = b_u_
        self.x_0 = x_0
        self.Z = Z
        self.V = V
        self.problem = cp.Problem(
            cp.Minimize(objective),
            constraints=const
        )
        self.use_solver = cp.MOSEK

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
        data_constraints = np.load(p_cached_dir / "data_constraints.npz")
        data_controller = np.load(p_cached_dir / "rtube_controller.npz")
        data_system = np.load(p_cached_dir / "system.npz")
        A, B = data_system["A"], data_system["B"]
        v_amp = data_constraints["v_amp"]
        u_amp = data_constraints["u_amp"]
        rho = data_controller["rho"]
        P = data_controller["P"]
        P_sqrt = data_controller["P_sqrt"]
        K = data_controller["K"]
        c_x_sq = data_controller["c_x_sq"]
        c_u_sq = data_controller["c_u_sq"]
        w_bar_sq = data_controller["w_bar_sq"]
        m_x, m_u = B.shape
        m_p = m_u
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
            P,
            P_sqrt,
            K,
            c_x_sq,
            c_u_sq,
            w_bar_sq,
            **kwargs
        )

    def set_parameter_values(self, x_0, cs, rs, x_g_v):
        self.x_0.value = x_0
        self.zg.value = x_g_v
        self.cs.value = cs # [:self.ball_dim]
        self.rs.value = rs


    def set_solver(self, solver):
        self.use_solver = solver

    def solve(self):
        self.problem.solve(solver=self.use_solver)
        return self.Z.value is not None

    def get_solved_control(self, x, k = 0):
        # x = self.x_0.value
        x_nom = self.Z.value[:, k]
        u_nom = self.V.value[:, k]
        e = (x - x_nom)
        u = u_nom + self.K @ e
        return u

    def get_predicted_traj(self):
        return self.Z.value

    def get_predicted_state(self, k):
        return self.Z.value[:, k]

    def get_solution_results(self):
        return (self.Z.value, self.V.value, self.r_p)

    def get_solution_inputs(self):
        return self.x_0.value, self.zg.value, self.cs.value, self.rs.value

    def __str__(self):
        return f"tube({self.N})"
