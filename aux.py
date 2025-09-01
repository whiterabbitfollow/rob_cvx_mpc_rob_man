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


import functools

import control as ct
from control import matlab
import numpy as np
from scipy.spatial import ConvexHull
import pinocchio as pin


def get_linear_box_constraints(m, amp=1.):
    A = np.vstack([
        np.eye(m),
        -np.eye(m)
    ])
    b = np.ones(m * 2) * amp
    return A, b


def get_linear_ordered_state_constraints(m_p, p_amp, v_amp):
    """
    Outputs configuration and velocity constraint as linear inequality constraints on state
    """
    A_q = np.vstack([
        np.c_[np.eye(m_p), np.zeros((m_p, m_p))],
        -np.c_[np.eye(m_p), np.zeros((m_p, m_p))]
    ])
    b_q = np.ones(2 * m_p) * p_amp

    A_qd = np.vstack([
        np.c_[np.zeros((m_p, m_p)), np.eye(m_p)],
        -np.c_[np.zeros((m_p, m_p)), np.eye(m_p)]
    ])
    b_qd = np.ones(2 * m_p) * v_amp
    A_x = np.vstack([A_q, A_qd])
    b_x = np.r_[b_q, b_qd]
    return A_x, b_x


def compute_largets_margin_bubble(p, path_centers, path_radii):
    dists = np.linalg.norm(path_centers - p, axis=-1)
    diff = path_radii - dists
    i_max = diff.argmax()
    return path_centers[i_max], path_radii[i_max]


def index_compute_largets_margin_bubble(p, path_centers, path_radii):
    dists = np.linalg.norm(path_centers - p, axis=-1)
    diff = path_radii - dists
    return diff.argmax()


def compute_goal_state(cs, rs, path_centers, p_g, return_index=False):
    c, r = cs[-1], rs[-1]
    if np.linalg.norm(p_g - c) <= r:
        p_g_v = p_g
        i_max = cs.shape[0]
    else:
        dists = np.linalg.norm(path_centers - c, axis=-1)
        mask = dists <= r
        indxs, = np.nonzero(mask)
        i_max = indxs.max()
        p_g_v = path_centers[i_max]
    if return_index:
        return np.r_[p_g_v, np.zeros_like(p_g_v)], i_max
    else:
        return np.r_[p_g_v, np.zeros_like(p_g_v)]


def compute_projected_matrix(P, m=2):
    A = P[:m, :m]
    B = P[:m, m:]
    C = P[m:, m:]
    D = P[m:, :m]
    P_ = A - B @ np.linalg.inv(C) @ D
    return P_

def interpolate_equidistant(path, delta=0.1, return_s=False):
    distance = np.linalg.norm(path[1:] - path[:-1], axis=-1)
    distance_acc = np.r_[0, np.cumsum(distance)]
    s = distance_acc / distance_acc[-1]
    if not np.isfinite(s).all():
        print("")
    dist = distance_acc[-1]
    nr_pnts = int(np.ceil(dist / delta) + 1)
    ss = np.linspace(0, 1, nr_pnts)
    if return_s:
        return ss, np.vstack([np.interp(ss, s, th) for th in path.T]).T
    else:
        return np.vstack([np.interp(ss, s, th) for th in path.T]).T


def get_linear_double_integrator_discrete_dynamics(nr_dof, dt, method="zoh"):
    m = nr_dof
    M_x = 2 * m
    M_u = 2
    I_2 = np.eye(m)
    O_2 = np.zeros((m, m))
    A_c = np.block(
        [
            [O_2, I_2],
            [O_2, O_2]]
    )
    B_c = np.block([
        [O_2],
        [I_2]
    ])
    C = np.block([I_2, I_2])
    D = I_2
    sys = matlab.ss(A_c, B_c, C, D)
    sys_d = ct.sample_system(sys, dt, method=method)
    A, B = sys_d.A, sys_d.B
    return A, B


class DynPinWrapper:

    def __init__(self, model, damping=None):
        self.model = model
        self.data = self.model.createData()
        self.config_dim = self.model.nq
        self.damping = damping

    def gravity(self, q):
        return pin.computeGeneralizedGravity(self.model, self.data, q)

    def coroli(self, q, q_d):
        C = self.coroli_matrix(q, q_d)
        return C @ q_d

    def coroli_matrix(self, q, q_d):
        if self.damping is None:
            return pin.computeCoriolisMatrix(self.model, self.data, q, q_d)
        else:
            return (self.damping + pin.computeCoriolisMatrix(self.model, self.data, q, q_d))

    def mass_matrix(self, q):
        return pin.crba(self.model, self.data, q)


class DeltaDynPin:

    def __init__(self, nom : DynPinWrapper, sampled : DynPinWrapper):
        self.nom = nom
        self.sampled = sampled

    def __getattr__(self, item):
        nom_func = getattr(self.nom, item)
        sampled_func = getattr(self.sampled, item)
        def diff_method(*args, **kwargs):
            return sampled_func(*args, **kwargs) - nom_func(*args, **kwargs)
        return diff_method

    def __dir__(self):
        all_methods = [{name for name in dir(a) if callable(getattr(a, name))} for a in (self.nom, self.sampled)]
        shared_methods = set(list(functools.reduce(lambda x, y: x & y, all_methods)))
        return shared_methods | set(super().__dir__())


def compute_model_error(problem, q, q_d, u, ignore_gravity=False):
    dyn_nom = problem.dyn_nom
    dyn_err_gt = problem.dyn_err_gt
    M_nom = dyn_nom.mass_matrix(q)
    M_err = dyn_err_gt.mass_matrix(q)

    M = M_nom + M_err
    M_inv = np.linalg.inv(M)
    M_tilde = M_inv @ M_err

    c_err = dyn_err_gt.coroli(q, q_d)
    if ignore_gravity:
        grav_err = np.zeros_like(q)
    else:
        grav_err = dyn_err_gt.gravity(q)
    delta_norm = - M_tilde @ u - M_inv @ c_err - M_inv @ grav_err
    return delta_norm