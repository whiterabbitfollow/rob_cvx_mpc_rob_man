<p align="center">
<img src="demo.gif"/>
</p>

This is the project repository for the paper:

Robust Convex Model Predictive Control with collision avoidance guarantees for robot manipulators

A preview version can be found at: https://arxiv.org/abs/2502.16205

The signed configuration distance function (SCDF) is not included in this repo. The method is described in the paper: https://arxiv.org/abs/2502.16205

Overview of repository:
```bash
.
├── controllers
│   ├── common.py
│   ├── flexible_tube_controller.py
│   ├── nom_controller.py
│   └── tube_controller.py
├── examples
│   └── planar_two_dof
│       ├── manipulator
│       ├── 1_run_offline_pipline.py
│       ├── 2_run_all.py
│       ├── 3_print_results.py
│       ├── 4_visualize_mpcs.py
│       ├── __init__.py
│       └── world.py
├── path_planner
│   ├── core.py
│   └── rrt_bi_dir.py
├── aux.py
├── demo.gif
├── LICENSE
├── offline_pipeline.py
├── problem_scenario.py
├── README.md
└── requirements.txt
```

### Requirements:
- Mosek license
- Python 3.10

### Installation and setup:
1. Install python packages:

`pip install -r requirements.txt`

2. Add project folder to python path

`export PYTHONPATH="$(pwd):$PYTHONPATH"`

### Examples:
#### Planar 2 DOF manipulator
1. Run offline pipline 

`python examples/planar_two_dof/1_run_offline_pipline.py`

2. Run all methods

`python examples/planar_two_dof/2_run_all.py`

3. Print results

`python examples/planar_two_dof/3_print_results.py`

4. Run animation of selected method

`python examples/planar_two_dof/4_visualize_mpcs.py --method {method_name}`

where method_name is one of "nom_star", "rt", "ft".
