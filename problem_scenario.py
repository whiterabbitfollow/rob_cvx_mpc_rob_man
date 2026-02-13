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
            error_frac=0.05,
            ignore_gravity=False
    ):
        self.ignore_gravity = ignore_gravity
        self.p_cached_dir = p_cached_dir
        self.torque_limits = torque_limits
        self.v_amp_nom = v_amp_nom
        self.u_amp_nom = u_amp_nom
        self.dt = dt

        self.position_limits = None
        self.frac = error_frac
        self.path_urdf = path_urdf
        model_nom = pin.buildModelFromUrdf(path_urdf)

        self.masses_nom = np.array([inertia.mass for inertia in model_nom.inertias])
        self.parameter_dim = self.masses_nom.size

        self.model_nom = model_nom
        self.dyn_nom = DynPinWrapper(model_nom)

        # Set GT model
        # initialize model
        self.model_gt = pin.buildModelFromUrdf(path_urdf)
        self.dyn_gt = DynPinWrapper(model_nom)
        # sample within uncertainty range
        self.dyn_err_gt = self.get_err_dyn_random()

        self.config_dim = self.model_gt.nq

    def dump_to_cached_dir(self):
        with (self.p_cached_dir / "ps.pckl").open("wb") as fp:
            pickle.dump(dict(
                path_urdf=self.path_urdf,
                p_cached_dir=self.p_cached_dir,
                torque_limits=self.torque_limits,
                v_amp_nom=self.v_amp_nom,
                u_amp_nom=self.u_amp_nom,
                dt=self.dt,
                error_frac=self.frac,
                ignore_gravity=self.ignore_gravity
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

    def get_nominal_dynamics(self):
        return self.dyn_nom

    def get_err_dynamics(self):
        return self.dyn_err_gt

    def get_err_dyn_random(self):
        nr_masses = self.masses_nom.size
        ratios = 1 + np.random.uniform(-self.frac, self.frac, size=(nr_masses,))
        return self.get_err_dyn_from_ratios(ratios)

    def get_err_dyn_from_ratios(self, ratios):
        self.model_gt, self.dyn_gt = self.set_model_and_dynamics_from_ratios(self.model_gt, ratios)
        self.dyn_err_gt = DeltaDynPin(nom=self.dyn_nom, sampled=self.dyn_gt, ignore_gravity=self.ignore_gravity)
        return self.dyn_err_gt

    def set_model_and_dynamics_from_ratios(self, model, ratios):
        for m_gt, inertia, ratio in zip(self.masses_nom, model.inertias, ratios):
            inertia.mass = m_gt * ratio # ratio within [1-frac, 1+frac]
        dyn = DynPinWrapper(model)
        return model, dyn
