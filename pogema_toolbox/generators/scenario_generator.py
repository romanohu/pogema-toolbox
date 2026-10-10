"""Deterministic, explicit scenarios on binary grids."""

import sys
from dataclasses import dataclass
from typing import List, Optional, Tuple
from numbers import Integral

import numpy as np
from pogema import GridConfig


@dataclass
class TaskLayout:
    """A generated map with named (row, column) endpoints and its map seed."""

    obstacles: np.ndarray
    pickup_cells: List[Tuple[int, int]]
    delivery_cells: List[Tuple[int, int]]
    map_seed: Optional[int]


def _integer(value, name, minimum=0):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')
    return int(value)


def _binary_grid(obstacles):
    try:
        grid = np.asarray(obstacles)
    except (TypeError, ValueError) as error:
        raise ValueError('obstacles must be a nonempty rectangular binary grid') from error
    if grid.ndim != 2 or not all(grid.shape) or not np.isin(grid, [0, 1]).all():
        raise ValueError('obstacles must be a nonempty rectangular binary grid')
    return grid.astype(np.int8, copy=True)


def _coordinate(position, obstacles):
    try:
        if len(position) != 2:
            raise ValueError('positions must contain two integer coordinates')
        row, column = (_integer(value, 'coordinate') for value in position)
    except TypeError as error:
        raise ValueError('positions must contain two integer coordinates') from error
    if row >= obstacles.shape[0] or column >= obstacles.shape[1] or obstacles[row, column]:
        raise ValueError(f'position {(row, column)} must be an in-bounds free cell')
    return row, column


def connected_components(obstacles) -> np.ndarray:
    """Label four-neighbor free components; obstacles have label -1."""
    grid = _binary_grid(obstacles)
    labels = np.full(grid.shape, -1, dtype=np.int32)
    component = 0
    height, width = grid.shape
    for row, column in np.argwhere(grid == 0):
        if labels[row, column] != -1:
            continue
        labels[row, column] = component
        stack = [(int(row), int(column))]
        while stack:
            current_row, current_column = stack.pop()
            for next_row, next_column in (
                (current_row - 1, current_column), (current_row + 1, current_column),
                (current_row, current_column - 1), (current_row, current_column + 1),
            ):
                if (0 <= next_row < height and 0 <= next_column < width
                        and grid[next_row, next_column] == 0 and labels[next_row, next_column] == -1):
                    labels[next_row, next_column] = component
                    stack.append((next_row, next_column))
        component += 1
    return labels


def _grid_config(obstacles, starts, targets, **options):
    reserved = {'map', 'agents_xy', 'targets_xy', 'possible_agents_xy', 'possible_targets_xy',
                'num_agents', 'width', 'height', 'size', 'FREE', 'OBSTACLE'}
    conflict = reserved.intersection(options)
    if conflict:
        raise ValueError(f'grid_options cannot override {sorted(conflict)}')
    if 'seed' in options:
        seed = _integer(options['seed'], 'seed')
        if seed >= sys.maxsize:
            raise ValueError('seed must be smaller than sys.maxsize')
        options['seed'] = seed
    if 'max_episode_steps' in options:
        options['max_episode_steps'] = _integer(options['max_episode_steps'], 'max_episode_steps', 1)
    return GridConfig(map=obstacles.tolist(), width=int(obstacles.shape[1]),
                      height=int(obstacles.shape[0]), agents_xy=starts, targets_xy=targets,
                      num_agents=len(starts), **options)


def _choose(rng, cells, count):
    return [cells[int(index)] for index in rng.choice(len(cells), count, replace=False)]


def _choose_other(rng, cells, indices, previous):
    excluded = indices.get(previous)
    index = int(rng.integers(len(cells) - (excluded is not None)))
    if excluded is not None and index >= excluded:
        index += 1
    return cells[index]


def _pairs(rng, starts_pool, goals_pool, count, starts_indices, goals_indices):
    if count == 1:
        if len(goals_pool) == 1:
            start = _choose_other(rng, starts_pool, starts_indices, goals_pool[0])
        else:
            start = starts_pool[int(rng.integers(len(starts_pool)))]
        return [start], [_choose_other(rng, goals_pool, goals_indices, start)]
    starts, goals = _choose(rng, starts_pool, count), _choose(rng, goals_pool, count)
    equal = [index for index in range(count) if starts[index] == goals[index]]
    if len(equal) > 1:
        previous = goals[equal[-1]]
        for index in equal:
            goals[index], previous = previous, goals[index]
    elif equal:
        index = equal[0]
        other = (index + 1) % count
        goals[index], goals[other] = goals[other], goals[index]
    return starts, goals


class ScenarioSampler:
    """Reuse a map's components for seeded uniform or endpoint scenarios."""

    def __init__(self, obstacles, pickup_cells=None, delivery_cells=None):
        self.obstacles = _binary_grid(obstacles)
        self.components = connected_components(self.obstacles)
        # Group in one pass rather than scanning the entire map per component.
        self.cells = {}
        for row, column in np.argwhere(self.obstacles == 0):
            cell = (int(row), int(column))
            label = int(self.components[cell])
            self.cells.setdefault(label, []).append(cell)
        self.pickups, self.deliveries = {}, {}
        if (pickup_cells is None) != (delivery_cells is None):
            raise ValueError('pickup_cells and delivery_cells must be supplied together')
        if pickup_cells is not None:
            for values, groups in ((pickup_cells, self.pickups), (delivery_cells, self.deliveries)):
                positions = [_coordinate(cell, self.obstacles) for cell in values]
                if not positions or len(set(positions)) != len(positions):
                    raise ValueError('endpoints must be nonempty and unique within each role')
                for cell in positions:
                    groups.setdefault(int(self.components[cell]), []).append(cell)
        self.cell_indices = {label: {cell: index for index, cell in enumerate(cells)}
                             for label, cells in self.cells.items()}
        self.pickup_indices = {label: {cell: index for index, cell in enumerate(cells)}
                               for label, cells in self.pickups.items()}
        self.delivery_indices = {label: {cell: index for index, cell in enumerate(cells)}
                                 for label, cells in self.deliveries.items()}

    def generate(self, num_agents, seed, *, sampling='uniform', on_target='nothing',
                 max_episode_steps=256, **grid_options) -> GridConfig:
        num_agents = _integer(num_agents, 'num_agents', 1)
        seed = _integer(seed, 'seed')
        if seed >= sys.maxsize:
            raise ValueError('seed must be smaller than sys.maxsize')
        horizon = _integer(max_episode_steps, 'max_episode_steps', 1)
        if sampling not in ('uniform', 'endpoints'):
            raise ValueError(f'unsupported sampling: {sampling!r}')
        if on_target not in ('nothing', 'finish', 'restart'):
            raise ValueError(f'unsupported on_target: {on_target!r}')
        pools = {}
        restart_roles = {}
        capacities = {}
        for label, cells in self.cells.items():
            if sampling == 'uniform':
                if len(cells) < 2:
                    continue
                starts_pool, goals_pool = cells, cells
                capacity = len(cells)
            else:
                pickups, deliveries = self.pickups.get(label, []), self.deliveries.get(label, [])
                if not pickups or not deliveries or (len(pickups) == len(deliveries) == 1 and pickups == deliveries):
                    continue
                if on_target == 'restart':
                    # A role shared with a singleton opposite role cannot lead
                    # to a different next target, so retain only valid transitions.
                    restart_pickups = [cell for cell in pickups if len(deliveries) > 1 or cell != deliveries[0]]
                    restart_deliveries = [cell for cell in deliveries if len(pickups) > 1 or cell != pickups[0]]
                    restart_roles[label] = (
                        restart_pickups, restart_deliveries,
                        {cell: index for index, cell in enumerate(restart_pickups)},
                        {cell: index for index, cell in enumerate(restart_deliveries)},
                    )
                    starts_pool, goals_pool = cells, restart_pickups
                    capacity = len(cells)
                else:
                    starts_pool, goals_pool = pickups, deliveries
                    capacity = min(len(pickups), len(deliveries))
            pools[label] = starts_pool, goals_pool
            capacities[label] = capacity
        capacity = sum(capacities.values())
        if num_agents > capacity:
            raise ValueError(f'num_agents={num_agents} exceeds {sampling} placement capacity {capacity}')
        rng = np.random.default_rng(seed)
        slots = np.concatenate([np.full(count, label, dtype=np.int32) for label, count in capacities.items()])
        selected = rng.choice(slots, num_agents, replace=False)
        assignments = {}
        for label in pools:
            count = int(np.count_nonzero(selected == label))
            if not count:
                continue
            starts_pool, goals_pool = pools[label]
            if sampling == 'endpoints' and on_target == 'restart':
                starts = _choose(rng, starts_pool, count)
                goals = [goals_pool[int(rng.integers(len(goals_pool)))] for _ in starts]
            else:
                starts_indices = self.cell_indices[label] if sampling == 'uniform' else self.pickup_indices[label]
                goals_indices = self.cell_indices[label] if sampling == 'uniform' else self.delivery_indices[label]
                starts, goals = _pairs(rng, starts_pool, goals_pool, count, starts_indices, goals_indices)
            assignments[label] = iter(zip(starts, goals))
        starts, targets = [], []
        for label in selected:
            start, goal = next(assignments[int(label)])
            starts.append(start)
            if on_target != 'restart':
                targets.append(goal)
                continue
            sequence = [goal]
            for index in range(1, horizon + 1):
                if sampling == 'endpoints':
                    pickups, deliveries, pickup_indices, delivery_indices = restart_roles[int(label)]
                    cells = deliveries if index % 2 else pickups
                    indices = delivery_indices if index % 2 else pickup_indices
                else:
                    cells = self.cells[int(label)]
                    indices = self.cell_indices[int(label)]
                sequence.append(_choose_other(rng, cells, indices, sequence[-1]))
            targets.append(sequence)
        return _grid_config(self.obstacles, starts, targets, seed=seed, on_target=on_target,
                            max_episode_steps=horizon, **grid_options)
