"""Connected Sortation layouts with spaced bins and peripheral pickups."""

from dataclasses import dataclass

import numpy as np

from pogema_toolbox.generators.scenario_generator import TaskLayout, _choose, _integer


def _validate(width, height, bin_spacing, pickup_count, delivery_count, seed):
    width = _integer(width, 'width', 3)
    height = _integer(height, 'height', 4)
    bin_spacing = _integer(bin_spacing, 'bin_spacing', 1)
    pickup_count = _integer(pickup_count, 'pickup_count', 1)
    delivery_count = _integer(delivery_count, 'delivery_count', 1)
    bin_capacity = len(range(1, height - 2, bin_spacing + 1)) * len(range(1, width - 1, bin_spacing + 1))
    if delivery_count > bin_capacity:
        raise ValueError(f'delivery_count exceeds bin capacity {bin_capacity}')
    pickup_capacity = 2 * width + 2 * height - 4
    if pickup_count > pickup_capacity:
        raise ValueError(f'pickup_count exceeds peripheral capacity {pickup_capacity}')
    if seed is not None:
        seed = _integer(seed, 'seed')
    return width, height, bin_spacing, pickup_count, delivery_count, seed


@dataclass
class SortationRangeSettings:
    width_min: int = 17
    width_max: int = 33
    height_min: int = 17
    height_max: int = 33
    bin_spacing_min: int = 1
    bin_spacing_max: int = 3
    pickup_count_min: int = 2
    pickup_count_max: int = 8
    delivery_count_min: int = 2
    delivery_count_max: int = 8

    def sample(self, seed=None):
        """Draw inclusive ranges independently; reject an infeasible draw."""
        if seed is not None:
            seed = _integer(seed, 'seed')
        rng = np.random.default_rng(seed)
        settings = {'seed': seed}
        for name in ('width', 'height', 'bin_spacing', 'pickup_count', 'delivery_count'):
            lower = _integer(getattr(self, name + '_min'), name + '_min', 1)
            upper = _integer(getattr(self, name + '_max'), name + '_max', lower)
            settings[name] = int(rng.integers(lower, upper + 1))
        _validate(**settings)
        return settings


def generate_sortation(width=17, height=17, bin_spacing=2, pickup_count=4,
                       delivery_count=4, seed=None) -> TaskLayout:
    """Place one-cell bins, select cells below bins, and select edge pickups."""
    width, height, bin_spacing, pickup_count, delivery_count, seed = _validate(
        width, height, bin_spacing, pickup_count, delivery_count, seed,
    )
    rng = np.random.default_rng(seed)
    obstacles = np.zeros((height, width), dtype=np.int8)
    bins = [(row, column) for row in range(1, height - 2, bin_spacing + 1)
            for column in range(1, width - 1, bin_spacing + 1)]
    for cell in bins:
        obstacles[cell] = 1
    peripheral = [(row, column) for row in range(height) for column in range(width)
                  if row in (0, height - 1) or column in (0, width - 1)]
    pickups = _choose(rng, peripheral, pickup_count)
    deliveries = _choose(rng, [(row + 1, column) for row, column in bins], delivery_count)
    return TaskLayout(obstacles, pickups, deliveries, seed)
