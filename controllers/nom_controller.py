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


class NominalController:

    def __init__(
            self,
            A,
            B,
            Q,
            Q_e,
            R,
            N,
            u_lim,
            v_lim=0.1,
            p_amp=1,
            ball_dim=None
    ):
        self.A = A
        _, self.M_x = A.shape
        _, self.M_u = B.shape
        self.M_p = int(self.M_x / 2)
        self.B = B
        self.Q = Q
        self.R = R
        self.N = N
        self.u_lim = u_lim
        m = int(self.M_x / 2)
        self.ball_dim = ball_dim or m
        self.N = N
        self.X = cp.Variable((self.M_x, N + 1))
        self.U = cp.Variable((self.M_u, N))
        self.x_start = cp.Parameter(self.M_x)
        self.x_g = cp.Parameter(self.M_x)
        self.cs = cp.Parameter((self.ball_dim, N + 1))
        self.rs = cp.Parameter((N + 1, ))
        objective = 0
        for i in range(N):
            objective += cp.quad_form(self.X[:, i] - self.X[:, -1], Q) + cp.quad_form(self.U[:, i], R)
        objective += cp.quad_form(self.X[:, -1] - self.x_g, Q_e)
        const = [
            self.X[:, 1:] == A @ self.X[:, :-1] + B @ self.U,
            self.U <= u_lim,
            self.U >= -u_lim,
            self.X[:, 0] == self.x_start,
            self.X[:self.M_p, :] <= p_amp,
            self.X[:self.M_p, :] >= -p_amp,
            self.X[self.M_p:, -1] == np.zeros(self.M_p, ),
            self.U[:, -1] == np.zeros(self.M_p, ),
            cp.norm(self.X[:self.ball_dim] - self.cs, axis=0) <= self.rs,
            self.X[m:, :] <= v_lim,
            self.X[m:, :] >= -v_lim
        ]
        self.problem = cp.Problem(cp.Minimize(objective), const)

    def solve(self, x_0, cs, rs, x_g_v):
        self.x_start.value = x_0
        self.x_g.value = x_g_v
        self.cs.value = cs[:self.ball_dim]
        self.rs.value = rs
        try:
            self.loss = self.problem.solve(solver=cp.CLARABEL)
            self.status = self.problem.status
            status = self.X.value is not None
        except:
            self.loss = np.inf
            status = False
        return status

    def get_solution_results(self):
        return self.X.value, self.U.value

    def get_solution_inputs(self):
        return self.x_start.value, self.x_g.value, self.cs.value, self.rs.value

    def get_solved_control(self, x, k=0):
        u = self.U.value[:, k]
        return u

    def get_predicted_traj(self):
        return self.X.value
