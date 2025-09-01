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


def initialize_graph(dim = 2, max_size=100):
    V = np.zeros((max_size, dim))
    E = {}
    size = np.array(0)
    return (V, E, size)


def add_node_to_graph(x, graph):
    V, *_, size = graph
    V[size] = x
    size += 1


def add_edge_to_graph(x_parent, x_child, graph):
    E = graph[1]
    E[tuple(x_child)] = tuple(x_parent)


def rrt_plan(graph, sample_free, trans_check_func):
    x_r = sample_free()
    x_n = get_nearest(x_r, graph)
    if np.isclose(x_r, x_n).all():
        return None
    if trans_check_func(x_n, x_r):
        add_node_to_graph(x_r, graph)
        add_edge_to_graph(x_n, x_r, graph)
        return x_r
    else:
        return None


def get_nearest(x_r, graph):
    V, *_, size = graph
    dists = np.linalg.norm(V[:size] - x_r, axis=1)
    i_min = np.argmin(dists)
    return V[i_min]


def get_path_from_graph(x, graph):
    E = graph[1]
    path = [x]
    parent = tuple(x)
    while True:
        parent = E.get(parent)
        if parent is not None:
            path.append(parent)
        else:
            break
    return np.vstack(path)


def connect_path_from_trees(x, graph_s, graph_g):
    path_s = get_path_from_graph(x, graph_s)[::-1]
    path_g = get_path_from_graph(x, graph_g)
    return np.vstack([path_s, path_g[1:]])
