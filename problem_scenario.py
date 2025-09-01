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


import pickle
from itertools import product

import numpy as np
import pinocchio as pin

from aux import DynPinWrapper, DeltaDynPin


class ProblemScenarioMassAllPin:

    def __init__(
            self,
            path_urdf,
            p_cached_dir,
            torque_limits,
            v_amp_nom,
            u_amp_nom,
            dt,
            error_frac=0.05
    ):
        self.p_cached_dir = p_cached_dir
        self.torque_limits = torque_limits
        self.v_amp_nom = v_amp_nom
        self.u_amp_nom = u_amp_nom
        self.dt = dt

        self.frac = error_frac

        model_nom = pin.buildModelFromUrdf(path_urdf)
        self.masses_nom = np.array([inertia.mass for inertia in model_nom.inertias])
        self.dyn_nom = DynPinWrapper(model_nom)
        self.model_nom = model_nom

        self.model_sampled = pin.buildModelFromUrdf(path_urdf)
        self.model_sampled, self.dyn_sampled = self.sample_model_within_bound(self.model_sampled)

        # Set GT model
        self.model_gt = pin.buildModelFromUrdf(path_urdf)
        self.set_randomized_gt_model()

        self.config_dim = self.model_gt.nq

        with (p_cached_dir / "ps.pckl").open("wb") as fp:
            pickle.dump(dict(
                path_urdf=path_urdf,
                p_cached_dir=p_cached_dir,
                torque_limits=torque_limits,
                v_amp_nom=v_amp_nom,
                u_amp_nom=u_amp_nom,
                dt=dt,
                error_frac=error_frac
            ),
                fp
            )

    @classmethod
    def from_cached_dir(cls, p_cached_dir):
        with (p_cached_dir / "ps.pckl").open("rb") as fp:
            data = pickle.load(fp)
        return cls(
            **data
        )

    def set_randomized_gt_model(self):
        self.model_gt, self.dyn_gt = self.sample_model_within_bound(self.model_gt)
        self.dyn_err_gt = DeltaDynPin(self.dyn_nom, self.dyn_gt)

    def sample_error_dynamics(self):
        self.model_sampled, self.dyn_sampled = self.sample_model_within_bound(self.model_sampled)
        return DeltaDynPin(self.dyn_nom, self.dyn_sampled)

    def sample_model_within_bound(self, model):
        for m_gt, inertia in zip(self.masses_nom, model.inertias):
            ratio_bias_err = 1 + np.random.uniform(-self.frac, self.frac)
            inertia.mass = m_gt * ratio_bias_err
        dyn = DynPinWrapper(model)
        return model, dyn

    def get_error_dynamics_from_ratios(self, ratios):
        model = self.model_sampled
        for i, (m_gt, inertia) in enumerate(zip(self.masses_nom, model.inertias)):
            inertia.mass = m_gt * ratios[i]
        dyn = DynPinWrapper(model)
        dyn_err = DeltaDynPin(self.dyn_nom, dyn)
        return dyn_err

    def get_ratios_parameters_vertices(self):
        n_masses = len(self.masses_nom)
        n_all = n_masses
        ratios_cube =  1 + np.vstack(list(product(*([(-1, 1)] * n_all)))) * self.frac
        return ratios_cube
