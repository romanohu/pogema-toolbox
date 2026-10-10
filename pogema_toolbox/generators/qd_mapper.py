"""Read local, unweighted QD-MAPPER JSON maps and replay stored scenarios."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
from pogema import GridConfig

from pogema_toolbox.generators.scenario_generator import _coordinate, _integer
from pogema_toolbox.moving_ai_ingestion import (
    MovingAIProblem, _cached_components, movingai_grid_config,
)


@dataclass(frozen=True)
class QDMap:
    obstacles: np.ndarray
    problems: List[List[MovingAIProblem]]
    name: str
    source_hash: str

    def grid_config(self, scenario_index, num_agents, **options) -> GridConfig:
        """Replay exactly the requested prefix of a stored scenario."""
        index = _integer(scenario_index, 'scenario_index')
        if index >= len(self.problems):
            raise ValueError('scenario_index exceeds available stored scenarios')
        return movingai_grid_config(self.obstacles, self.problems[index], num_agents, **options)


def load_qd_map(path) -> QDMap:
    """Validate a local JSON file, retaining dimensions and scenario order.

    Problem distance is None because QD JSON supplies no source distance;
    bucket records the zero-based stored scenario index.
    """
    source = Path(path).read_bytes()
    data = json.loads(source)
    required = {'name', 'weight', 'n_row', 'n_col', 'layout', 'start', 'goal'}
    if not isinstance(data, dict) or required.difference(data):
        raise ValueError('QD map must contain name, weight, n_row, n_col, layout, start and goal')
    if not isinstance(data['name'], str) or not data['name']:
        raise ValueError('name must be a nonempty string')
    if data['weight'] is not False:
        raise ValueError('only unweighted QD maps with weight=false are supported')
    height = _integer(data['n_row'], 'n_row', 1)
    width = _integer(data['n_col'], 'n_col', 1)
    layout = data['layout']
    if (not isinstance(layout, list) or len(layout) != height
            or any(not isinstance(row, str) or len(row) != width for row in layout)):
        raise ValueError('layout must match the declared n_row and n_col')
    if any(set(row) - {'.', '@'} for row in layout):
        raise ValueError('QD layout supports only . and @ symbols')
    obstacles = np.asarray([[int(symbol == '@') for symbol in row] for row in layout], dtype=np.int8)
    labels = _cached_components(obstacles.shape, obstacles.tobytes())
    starts, goals = data['start'], data['goal']
    if (not isinstance(starts, list) or not isinstance(goals, list)
            or not starts or len(starts) != len(goals)):
        raise ValueError('start and goal must be matching nonempty scenario arrays')
    problems = []
    for scenario_index, (scenario_starts, scenario_goals) in enumerate(zip(starts, goals)):
        if (not isinstance(scenario_starts, list) or not isinstance(scenario_goals, list)
                or not scenario_starts or len(scenario_starts) != len(scenario_goals)):
            raise ValueError(f'scenario {scenario_index}: start and goal must be matching nonempty arrays')
        scenario = []
        for start_index, goal_index in zip(scenario_starts, scenario_goals):
            start = _coordinate(divmod(_integer(start_index, 'start index'), width), obstacles)
            goal = _coordinate(divmod(_integer(goal_index, 'goal index'), width), obstacles)
            if labels[start] != labels[goal]:
                raise ValueError(f'scenario {scenario_index}: start and goal are disconnected')
            scenario.append(MovingAIProblem(start, goal, None, scenario_index, data['name']))
        if (len({problem.start for problem in scenario}) != len(scenario)
                or len({problem.goal for problem in scenario}) != len(scenario)):
            raise ValueError(f'scenario {scenario_index}: starts and goals must be unique within each role')
        problems.append(scenario)
    return QDMap(obstacles, problems, data['name'], hashlib.sha256(source).hexdigest())
