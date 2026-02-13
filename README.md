# Robust Convex Model Predictive Control with collision avoidance guarantees for robot manipulators

<p align="center">
<img src="demo.gif"/>
</p>

This is the project repository for the paper:

https://arxiv.org/abs/2502.16205

The neural signed configuration distance function (nSCDF) is not included in this repo. The method is described in the paper: 

https://arxiv.org/abs/2502.16205

with corresponding repo:

https://github.com/whiterabbitfollow/nSCDF_PBRM

Overview of repository:
```bash
.
├── controllers
├── corridor_simulators
├── examples
│   └── planar_two_dof
│       ├── data
│       └── manipulator
└── path_planner
```

### Requirements:
- Mosek license (optional)
- Python 3.10

### Installation and setup:
1. Install python packages:

`pip install -r requirements.txt`

2. Add project folder to python path

`export PYTHONPATH="$(pwd):$PYTHONPATH`

### Examples:
#### Planar 2 DOF manipulator

The following demonstrates how to run the offline pipeline for a planar 2 DOF manipulator with 10 % uncertainty link masses.

##### Offline pipeline (optional)
1. A Mosek license is required to run the offline pipeline.
2. Install cvxpy to support Mosek:

`pip install cvxpy[CBC,CVXOPT,GLOP,GLPK,GUROBI,MOSEK,PDLP,SCIP,XPRESS]`

3. Run offline pipline:

`python examples/planar_two_dof/1_run_offline_pipline.py`

4. List offline results 

`ls examples/planar_two_dof/data/dof_2_ef_0.1`

##### Online Corridor Control
1. Run all methods:

`python examples/planar_two_dof/2_run_all.py`

2. Print results:

`python examples/planar_two_dof/3_print_results.py`

3. Run animation of selected method:

`python examples/planar_two_dof/4_visualize_mpcs.py --method {method_name}`

where method_name is one of "nom_star", "rt", "ft".
