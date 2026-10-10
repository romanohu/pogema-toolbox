import requests
import zipfile
import io
import math
from dataclasses import dataclass
from functools import lru_cache
from typing import List

import numpy as np
from pogema import GridConfig

from pogema_toolbox.generators.scenario_generator import (
    _binary_grid, _coordinate, _grid_config, _integer, connected_components,
)

from pogema_toolbox.generators.generator_utils import maps_dict_to_yaml


@dataclass(frozen=True)
class MovingAIProblem:
    start: tuple
    goal: tuple
    distance: float
    bucket: int
    map_name: str


def parse_movingai_map(text: str) -> np.ndarray:
    """Parse a strict MovingAI octile map without trimming its border."""
    lines = text.splitlines()
    if len(lines) < 4 or lines[0].strip() != 'type octile' or lines[3].strip() != 'map':
        raise ValueError('expected MovingAI type octile and map headers')
    dimensions = []
    for line, name in zip(lines[1:3], ('height', 'width')):
        fields = line.split()
        if len(fields) != 2 or fields[0] != name:
            raise ValueError(f'expected {name} header')
        try:
            dimension = int(fields[1])
        except ValueError as error:
            raise ValueError(f'{name} must be a positive integer') from error
        dimensions.append(_integer(dimension, name, 1))
    height, width = dimensions
    rows = lines[4:]
    if len(rows) != height or any(len(row) != width for row in rows):
        raise ValueError('map dimensions do not match the declared height and width')
    if any(set(row) - {'.', '@', 'T'} for row in rows):
        raise ValueError('map supports only ., @ and T symbols')
    return np.asarray([[int(symbol != '.') for symbol in row] for row in rows], dtype=np.int8)


@lru_cache(maxsize=4)
def _cached_components(shape, data):
    labels = connected_components(np.frombuffer(data, dtype=np.int8).reshape(shape))
    labels.flags.writeable = False
    return labels


def parse_movingai_scenarios(text: str, map_name: str, obstacles) -> List[MovingAIProblem]:
    """Read source-order scenario rows, converting x/y to row/column."""
    grid = _binary_grid(obstacles)
    labels = _cached_components(grid.shape, grid.tobytes())
    lines = text.splitlines()
    if not lines or lines[0].strip() != 'version 1':
        raise ValueError('expected MovingAI scenario version 1')
    problems = []
    for line_number, line in enumerate(lines[1:], 2):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 9:
            raise ValueError(f'scenario line {line_number}: expected nine fields')
        try:
            bucket, width, height, start_x, start_y, goal_x, goal_y = (
                int(fields[index]) for index in (0, 2, 3, 4, 5, 6, 7)
            )
            distance = float(fields[8])
        except ValueError as error:
            raise ValueError(f'scenario line {line_number}: invalid numeric field') from error
        if fields[1] != map_name or (height, width) != grid.shape:
            raise ValueError(f'scenario line {line_number}: map name or dimensions mismatch')
        _integer(bucket, 'bucket')
        if not math.isfinite(distance) or distance < 0:
            raise ValueError(f'scenario line {line_number}: distance must be finite and nonnegative')
        start, goal = _coordinate((start_y, start_x), grid), _coordinate((goal_y, goal_x), grid)
        if labels[start] != labels[goal]:
            raise ValueError(f'scenario line {line_number}: start and goal are disconnected')
        problems.append(MovingAIProblem(start, goal, distance, bucket, map_name))
    return problems


def movingai_grid_config(obstacles, problems, num_agents: int, **grid_options) -> GridConfig:
    """Use exactly the first num_agents source rows as explicit positions."""
    grid = _binary_grid(obstacles)
    count = _integer(num_agents, 'num_agents', 1)
    if count > len(problems):
        raise ValueError('num_agents exceeds available scenario rows')
    if grid_options.get('on_target', 'nothing') not in ('nothing', 'finish'):
        raise ValueError('MovingAI prefixes require on_target=nothing or finish')
    labels = _cached_components(grid.shape, grid.tobytes())
    starts, goals = [], []
    for problem in problems[:count]:
        start, goal = _coordinate(problem.start, grid), _coordinate(problem.goal, grid)
        if labels[start] != labels[goal]:
            raise ValueError('scenario start and goal are disconnected')
        starts.append(start)
        goals.append(goal)
    if len(set(starts)) != count or len(set(goals)) != count:
        raise ValueError('scenario prefix must have unique starts and unique goals')
    grid_options.setdefault('on_target', 'nothing')
    return _grid_config(grid, starts, goals, **grid_options)


def download_moving_ai_maps(url):
    response = requests.get(url)

    zip_file = io.BytesIO(response.content)

    z = zipfile.ZipFile(zip_file, 'r')

    maps_dict = {}

    for file_name in z.namelist():
        if file_name.endswith('.map'):
            with z.open(file_name) as f:
                grid = map_to_grid(f)
                maps_dict[file_name.replace('.map', "")] = grid

    z.close()

    return maps_dict


def map_to_grid(file_in_zip, remove_border=True):
    lines = []
    with file_in_zip as f:
        type_ = f.readline().decode('utf-8').split(' ')[1]
        height = int(f.readline().decode('utf-8').split(' ')[1])
        width = int(f.readline().decode('utf-8').split(' ')[1])
        _ = f.readline()

        for _ in range(height):
            line = f.readline().decode('utf-8').rstrip()
            lines.append(line)

    m = []
    rmb = 1 if remove_border else 0
    for i in range(rmb, len(lines) - rmb):
        line = []
        for j in range(rmb, len(lines[i]) - rmb):
            symbol = lines[i][j]
            is_obstacle = symbol in ['@', 'O', 'T']
            line.append('#' if is_obstacle else '.')
        m.append("".join(line))
    return '\n'.join(m)


def main():
    url = 'https://movingai.com/benchmarks/street/street-map.zip'
    maps = download_moving_ai_maps(url)
    maps_dict_to_yaml('maps.yaml', maps)


if __name__ == '__main__':
    main()
