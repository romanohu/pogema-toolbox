"""Immutable, content-addressed Pogema instances and lossless persistence.

Ported from Neural-MAPF; see licenses/Neural-MAPF-LICENSE for its MIT notice.
"""
from dataclasses import dataclass
from collections import deque
from functools import lru_cache
from contextlib import contextmanager
import hashlib
import json
from numbers import Integral
import os
from pathlib import Path
import tempfile
from typing import Optional

MOVES = ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1))


def _coordinates(values):
    points = []
    for point in values:
        if len(point) != 2 or any(isinstance(x, bool) or not isinstance(x, Integral) for x in point):
            raise ValueError('Positions must contain two integer coordinates')
        points.append(tuple(int(x) for x in point))
    return tuple(points)


@lru_cache(maxsize=4)
def _component_labels(rows):
    labels = [[-1] * len(rows[0]) for _ in rows]
    component = 0
    for x, row in enumerate(rows):
        for y, blocked in enumerate(row):
            if blocked or labels[x][y] != -1:
                continue
            labels[x][y] = component
            queue = deque([(x, y)])
            while queue:
                px, py = queue.popleft()
                for dx, dy in MOVES[1:]:
                    nx, ny = px + dx, py + dy
                    if 0 <= nx < len(rows) and 0 <= ny < len(rows[0]) and not rows[nx][ny] and labels[nx][ny] == -1:
                        labels[nx][ny] = component
                        queue.append((nx, ny))
            component += 1
    return tuple(tuple(row) for row in labels)


@dataclass(frozen=True, init=False)
class Instance:
    obstacles: tuple
    starts: tuple
    goals: tuple
    goal_sequences: Optional[tuple]
    _grid_json: str
    id: str

    def __init__(self, obstacles, starts, goals, grid_config, goal_sequences=None):
        rows = tuple(tuple(row) for row in obstacles)
        if not rows or not rows[0] or any(len(row) != len(rows[0]) for row in rows):
            raise ValueError('Obstacle map must be nonempty and rectangular')
        if any(not isinstance(x, Integral) or x not in (0, 1) for row in rows for x in row):
            raise ValueError('Obstacles must be binary integers')
        rows = tuple(tuple(int(x) for x in row) for row in rows)
        starts, goals = _coordinates(starts), _coordinates(goals)
        if not starts or len(starts) != len(goals) or len(set(starts)) != len(starts):
            raise ValueError('Starts must be unique and match the nonempty goals')
        sequences = None if goal_sequences is None else tuple(_coordinates(s) for s in goal_sequences)
        if sequences is not None and (len(sequences) != len(starts) or
                any(not s or s[0] != g for s, g in zip(sequences, goals))):
            raise ValueError('Goal sequences must begin with each agent initial goal')
        points = starts + goals + (() if sequences is None else tuple(p for s in sequences for p in s))
        for x, y in points:
            if not (0 <= x < len(rows) and 0 <= y < len(rows[0])) or rows[x][y]:
                raise ValueError('Position outside the map or on an obstacle')
        config = dict(grid_config)
        config.setdefault('on_target', 'nothing')
        if config.get('on_target', 'nothing') not in ('nothing', 'finish', 'restart'):
            raise ValueError('Unsupported on_target protocol')
        if config.get('on_target') == 'restart' and (sequences is None or any(len(s) < 2 for s in sequences)):
            raise ValueError('Restart instances require fixed goal sequences with at least two goals')
        if config.get('on_target') != 'restart' and len(set(goals)) != len(goals):
            raise ValueError('One-shot goals must be unique')
        labels = _component_labels(rows)
        for index, start in enumerate(starts):
            targets = (goals[index],) if sequences is None else sequences[index]
            if any(labels[target[0]][target[1]] != labels[start[0]][start[1]] for target in targets):
                raise ValueError('Starts and goals must be connected by free cells')
        for name, value in [('obstacles', rows), ('starts', starts), ('goals', goals),
                            ('goal_sequences', sequences), ('_grid_json', json.dumps(config, sort_keys=True, allow_nan=False))]:
            object.__setattr__(self, name, value)
        object.__setattr__(self, 'id', hashlib.sha256(self._canonical().encode()).hexdigest())

    @property
    def grid_config(self):
        return json.loads(self._grid_json)

    def _payload(self):
        return {'obstacles': [list(r) for r in self.obstacles], 'starts': [list(p) for p in self.starts],
                'goals': [list(p) for p in self.goals], 'grid_config': self.grid_config,
                'goal_sequences': None if self.goal_sequences is None else
                [[list(p) for p in s] for s in self.goal_sequences]}

    def _canonical(self):
        return json.dumps(self._payload(), sort_keys=True, separators=(',', ':'), allow_nan=False)

    def to_dict(self):
        return dict(self._payload(), id=self.id)

    @classmethod
    def from_dict(cls, data):
        instance = cls(data['obstacles'], data['starts'], data['goals'], data['grid_config'],
                       data.get('goal_sequences'))
        if data.get('id') != instance.id:
            raise ValueError('Instance content hash does not match saved id')
        return instance

    @classmethod
    def from_grid_config(cls, grid_config):
        """Capture explicit map/positions without materializing a random environment."""
        from pogema import GridConfig
        if not isinstance(grid_config, GridConfig):
            raise TypeError('grid_config must be a Pogema GridConfig')
        parameters = grid_config.model_dump()
        obstacles = parameters.pop('map')
        starts = parameters.pop('agents_xy')
        targets = parameters.pop('targets_xy')
        parameters.pop('map_name', None)
        if obstacles is None or starts is None or targets is None:
            raise ValueError('Instance conversion requires an explicit map, starts and targets')
        sequences = targets if parameters['on_target'] == 'restart' else None
        goals = [sequence[0] for sequence in sequences] if sequences else targets
        return cls(obstacles, starts, goals, parameters, sequences)

    def to_grid_config(self):
        """Reconstruct a Pogema GridConfig with this instance's explicit data."""
        from pogema import GridConfig
        parameters = {key: value for key, value in self.grid_config.items() if value is not None}
        reserved = {'map', 'map_name', 'agents_xy', 'targets_xy',
                    'possible_agents_xy', 'possible_targets_xy'}
        conflict = reserved.intersection(parameters)
        owned = {'num_agents': len(self.starts), 'width': len(self.obstacles[0]),
                 'height': len(self.obstacles), 'size': max(len(self.obstacles), len(self.obstacles[0]))}
        conflict.update(key for key, value in owned.items()
                        if key in parameters and parameters[key] != value)
        if conflict:
            raise ValueError(f'grid_config cannot override instance data: {sorted(conflict)}')
        parameters.update(map=[list(r) for r in self.obstacles],
                          agents_xy=[list(p) for p in self.starts],
                          targets_xy=[[list(p) for p in s] for s in self.goal_sequences]
                          if self.goal_sequences else [list(p) for p in self.goals],
                          num_agents=len(self.starts),
                          width=len(self.obstacles[0]), height=len(self.obstacles))
        if owned['size'] == 1:
            # Pogema validates size >= 2 before the map derives 1x1 dimensions.
            for key in ('width', 'height', 'size'):
                parameters.pop(key, None)
        return GridConfig(**parameters)


@contextmanager
def _atomic_text(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f'.{path.name}.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(descriptor, 'w') as stream:
            yield stream
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def save_instances(instances, path):
    with _atomic_text(path) as stream:
        for instance in instances:
            stream.write(json.dumps(instance.to_dict(), allow_nan=False) + '\n')


def load_instances(path):
    with Path(path).open() as stream:
        return [Instance.from_dict(json.loads(line)) for line in stream if line.strip()]
