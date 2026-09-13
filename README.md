# POGEMA Toolbox

[![Downloads](https://static.pepy.tech/badge/pogema-toolbox)](https://pepy.tech/project/pogema-toolbox)
[<img src="https://img.shields.io/badge/license-Apache_2.0-blue">](https://github.com/tinkoff-ai/CORL/blob/main/LICENSE)
![PyPI](https://img.shields.io/pypi/v/pogema-toolbox?color=blue)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1XcLr-EmcgctKta3H1-zac_mnPqmG4Xxj?usp=sharing)

## Overview 

The POGEMA Toolbox is a comprehensive framework designed to facilitate the testing of learning-based approaches within the POGEMA environment. This toolbox offers a unified interface that enables the seamless execution of any learnable MAPF algorithm in POGEMA. 

- Firstly, the toolbox provides robust management tools for custom maps, allowing users to register and utilize these maps effectively within POGEMA. 
- Secondly, it enables the concurrent execution of multiple testing instances across various algorithms in a distributed manner, leveraging Dask for scalable processing. The results from these instances are then aggregated for analysis. 
- Lastly, the toolbox includes visualization capabilities, offering a convenient method to graphically represent aggregated results through detailed plots.

## Installation

Just install from PyPI:

```bash
pip install pogema-toolbox
```

## Features

### Register and use custom algorithms

To register an algorithm and then create an inference instance, one can use:

```python
from pogema import BatchAStarAgent

# Registring A* algorithm
ToolboxRegistry.register_algorithm('A*', BatchAStarAgent)

# Creating algorithm
algo = ToolboxRegistry.create_algorithm("A*")
```

The algorithm should provide a method act, which receives a batch of observations.

A more complex example is an algorithm with tunable hyperparameters. Here is an example with the Follower approach:
```python
ToolboxRegistry.register_algorithm('Follower', FollowerInference, FollowerInferenceConfig,
                                       follower_preprocessor)
```
For the full evaluation code example, please refer to [Follower repository](https://github.com/AIRI-Institute/learn-to-follow/blob/main/eval.py)

### Register custom maps

To create a custom map, the user first needs to define it using ASCII symbols or by uploading it from a file, and then register it using the toolbox:

```python
from pogema_toolbox.registry import ToolboxRegistry

# Creating cusom_map
custom_map = """
.......#.
...#...#.
.#.###.#.
"""

# Registring custom_map
ToolboxRegistry.register_maps({"custom_map": custom_map})
```

### Cities tiles

The fixed cities-tile maps are from
[Cognitive-AI-Systems/MAPF-GPT-DDG](https://github.com/Cognitive-AI-Systems/MAPF-GPT-DDG)
commit `842384ee18902b8e7c880c3f90c5fc8098520522`, asset path
`eval_configs/04-movingai/maps.yaml`. The packaged asset SHA-256 is
`3d357b87c64bab6a08a8ec43cfe81295ed37fd4a14db36d563e4ae599785664f`.
The upstream asset is licensed under the MIT License, with the notice
`Copyright (c) 2025 Alexey Skrynnik`; the exact license from that revision is
packaged in
[`pogema_toolbox/licenses/MAPF-GPT-DDG-LICENSE`](pogema_toolbox/licenses/MAPF-GPT-DDG-LICENSE).

```python
from pogema_toolbox.generators.cities_generator import CitiesTilesGenerator

CitiesTilesGenerator.official().generate()
CitiesTilesGenerator.single(
    map_name="Berlin_1_256_00",
    num_agents=(64, 128),
    num_samples=100,
    seed=42,
).generate()
CitiesTilesGenerator.random(
    num_agents=(64, 128, 192, 256),
    num_samples=1000,
    seed=42,
).generate()
```

`official` returns exactly 512 instances in map-major order: the 128 map names
in `CITY_NAMES`/tile-index order, each followed by agent counts 64, 128, 192,
and 256. Every official scenario uses seed 0. The fixed environment constants
are a 256-step limit, observation radius 5, `MAPF` observations, soft
collisions, and `on_target="nothing"`.

`single` fixes `map_name`; `random` selects each map uniformly from all 128
maps. Both modes independently select an entry uniformly from `num_agents` and
draw a non-negative 32-bit scenario seed for every sample. Sampling is with
replacement, so repeated map, agent-count, or complete scenario selections are
allowed, and the supplied generator `seed` makes their ordered sequence
reproducible. Every returned `CitiesTilesInstance` contains `grid_config`,
`city`, `tile_index`, `map_name`, `num_agents`, and `scenario_seed` metadata.

### Evaluation script

Example of the POGEMA Toolbox configuration for parallel testing of the RHCR approach and visualization of its results.

```yaml
environment: # Configuring Test Environments
  name: Environment
  on_target: 'restart'
  max_episode_steps: 128
  observation_type: 'POMAPF'
  collision_system: 'soft'
  seed: 
    grid_search: [ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 ]
  num_agents:
    grid_search: [ 8, 16, 24, 32, 48, 64 ]
  map_name:
    grid_search: [
        validation-mazes-seed-000, validation-mazes-seed-001, validation-mazes-seed-002, 
        validation-mazes-seed-003, validation-mazes-seed-004, validation-mazes-seed-005, 
    ]

algorithms: # Specifying algorithms and it's hyperparameters
  RHCR-5-10:
    name: RHCR
    parallel_backend: 'balanced_dask'
    num_process: 32
    simulation_window: 5
    planning_window: 10
    time_limit: 10
    low_level_planner: 'SIPP'
    solver: 'PBS'

results_views: # Defining results visualization 
  01-mazes:
    type: plot
    x: num_agents
    y: avg_throughput
    width: 4.0
    height: 3.1
    line_width: 2
    use_log_scale_x: True
    legend_font_size: 8
    font_size: 8
    name: Mazes
    ticks: [8, 16, 24, 32, 48, 64]

  TabularThroughput:
    type: tabular
    drop_keys: [ seed, map_name]
    print_results: True
```

The configuration is split into three main sections. The first one details the parameters of the POGEMA environment used for testing. It also includes iteration over the number of agents, seeds, and names of the maps (which were registered beforehand). The unified `grid_search` tag allows for the examination of any existing parameter of the environment. The second part of the configuration is a list of algorithms to be tested. Each algorithm has its alias (which will be shown in the results) and name, which specifies the family of methods. It also includes a list of hyperparameters common to different approaches, e.g., number of processes, parallel backend, etc., and the specific parameters of the algorithm.


## Citation
If you use this repository in your research or wish to cite it, please make a reference to our paper: 
```
@misc{skrynnik2024pogema,
      title={POGEMA: A Benchmark Platform for Cooperative Multi-Agent Navigation}, 
      author={Alexey Skrynnik and Anton Andreychuk and Anatolii Borzilov and Alexander Chernyavskiy and Konstantin Yakovlev and Aleksandr Panov},
      year={2024},
      eprint={2407.14931},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2407.14931}, 
}
```
