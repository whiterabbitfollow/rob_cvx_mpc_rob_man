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

from path_planner import core


def is_collision_free_bisection(x_s, x_e, sdf_func, transition_points=None, dist_tol=1e-2):
    r_s, r_e = sdf_func(np.vstack([x_s, x_e]))
    status = check_connection_with_bisection(x_s, r_s, x_e, r_e, sdf_func, points=transition_points, dist_tol=dist_tol)
    return status


def check_connection_with_bisection(x_l, r_l, x_r, r_r, sdf_func, points=None, dist_tol=1e-2, radius_tol=0):
    direction = x_r - x_l
    dist = np.linalg.norm(direction)
    if dist < r_l + r_r:
        return True
    direction_n = direction / dist
    x_l_ = direction_n * r_l + x_l
    x_r_ = -direction_n * r_r + x_r
    x_m = (x_l_ + x_r_) / 2
    r_m = sdf_func(x_m)
    if points is not None:
        points.append(np.append(x_m, r_m))
    if r_m <= radius_tol:
        return False
    elif dist < dist_tol:
        return True
    dist_s_l = np.linalg.norm(x_l - x_m)
    connected_l = ((dist_s_l - r_l - r_m) < 0).all()
    if not connected_l:
        connected_l = check_connection_with_bisection(x_l, r_l, x_m, r_m, sdf_func, points, dist_tol=dist_tol, radius_tol=radius_tol)
    dist_s_r = np.linalg.norm(x_m - x_r)
    connected_r = ((dist_s_r - r_r - r_m) < 0).all()
    if connected_l and not connected_r:
        connected_r = check_connection_with_bisection(x_m, r_m, x_r, r_r, sdf_func, points, dist_tol=dist_tol, radius_tol=radius_tol)
    connected = connected_l and connected_r
    return connected


def rrt_bi_dir_plan(
        x_s,
        x_g,
        sample_func,
        trans_check_func,
        max_iter=100
):
    dim = x_s.shape[0]
    path = np.array([]).reshape((0, dim))
    graph_s = core.initialize_graph(dim=dim, max_size=max_iter)
    core.add_node_to_graph(x_s, graph_s)
    graph_g = core.initialize_graph(dim=dim, max_size=max_iter)
    core.add_node_to_graph(x_g, graph_g)
    graph_a, graph_b = graph_s, graph_g
    cnt = 0
    for i in range(max_iter):
        cnt += 1
        x = core.rrt_plan(graph_a, sample_func, trans_check_func)
        if x is not None:
            x_n = core.get_nearest(x, graph_b)
            if trans_check_func(x, x_n):
                core.add_node_to_graph(x, graph_b)
                core.add_edge_to_graph(x_parent=x_n, x_child=x, graph=graph_b)
                path = core.connect_path_from_trees(x, graph_s, graph_g)
                break
        graph_a, graph_b = graph_b, graph_a
    return (graph_s, graph_g), cnt, path



def simple_shortcut(cs, rs):
    D = np.linalg.norm(cs[:, None] - cs[None, :], axis=-1)
    R = rs[:, None] + rs[None, :]
    M = D < R
    i = 0
    indxs_path = [i]
    i_last = rs.size - 1
    while True:
        m = M[i]
        indxs, = np.nonzero(m)
        i_ = indxs.max()
        if i_ == i_last:
            indxs_path.append(i_)
            break
        elif i_ == i:
            raise RuntimeError("Bubbles are not intersecting")
        i = i_
        indxs_path.append(i)
    return cs[indxs_path], rs[indxs_path]
