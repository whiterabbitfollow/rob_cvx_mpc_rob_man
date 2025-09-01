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


import pandas as pd

from examples.planar_two_dof import P_EXAMPLE_2_DOF

df = pd.read_csv(P_EXAMPLE_2_DOF / "data" / "dof_2_ef_0.1" / "results.csv")
df["reached_goal"] = df["status"] == "success"
df.rename(
    mapper={
        "t2g": "time_steps_to_goal",
        "comp_time_ee": "iteration_computation_time"
    },
    axis=1
    ,
    inplace=True
)
df_g = df.groupby("name")
print(df_g[["reached_goal", "iteration_computation_time"]].mean())

df_g = df.groupby(["run", "name"])["time_steps_to_goal"]
print(df_g.mean())
