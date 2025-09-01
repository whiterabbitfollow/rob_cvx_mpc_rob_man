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


import pathlib
import pinocchio as pin
import numpy as np

PATH_ROOT = pathlib.Path(__file__).parent


def distance_line_segment_and_point(l_s, l_e, p_c):
    u = p_c - l_s
    v = l_e - l_s
    projection = np.inner(u, v) / np.inner(v, v) * v
    p_closest_l = projection + l_s
    line_seg = np.vstack([
        l_s,
        l_e
    ])
    p_closest_l = np.minimum(
        np.maximum(p_closest_l, line_seg.min(axis=0)),
        line_seg.max(axis=0)
    )
    u_n = p_c - p_closest_l
    dist = np.linalg.norm(u_n)
    return dist


class ManKin:

    def __init__(self):
        p_urdf = PATH_ROOT / "two_link.urdf"
        self.model = pin.buildModelFromUrdf(str(p_urdf))
        self.data = self.model.createData()
        self.link_names = ["link_1", "link_2", "link_ee"]
        self.link_ids = [self.model.getFrameId(ln) for ln in self.link_names]
        q = np.zeros(2)
        fk = self.get_link_fk(q)
        self.L1 = np.linalg.norm(fk["link_1"][:3, -1] - fk["link_2"][:3, -1])
        self.L2 = np.linalg.norm(fk["link_2"][:3, -1] - fk["link_ee"][:3, -1])
        self.config_dim = 2

    def get_wireframe(self, q=None):
        fk = self.get_link_fk(q)
        links = ["link_1", "link_2", "link_ee"]
        positions = np.vstack([fk[l][:2, -1] for l in links])
        return positions

    def get_link_fk(self, q):
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        return {f_name: self.data.oMf[f_id].homogeneous for f_name, f_id in zip(self.link_names, self.link_ids)}

    def is_collision_free_sphere(self, q, obs_p, obs_r):
        line_segs = self.get_wireframe(q)
        for c, r in zip(obs_p, obs_r):
            for l_s, l_e in zip(line_segs[:-1], line_segs[1:]):
                d = distance_line_segment_and_point(l_s, l_e, c) - r
                if d < 0:
                    return False
        return True



if __name__ == "__main__":
    pass





