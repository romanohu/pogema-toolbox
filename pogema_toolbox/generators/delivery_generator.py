"""Connected block-and-aisle Delivery layouts with peripheral stations."""

from dataclasses import dataclass

import numpy as np

from pogema_toolbox.generators.scenario_generator import TaskLayout, _integer


def _validate(width, height, block_width, block_height, aisle_width, seed):
    values = [_integer(value, name, 1) for name, value in (
        ('width', width), ('height', height), ('block_width', block_width),
        ('block_height', block_height), ('aisle_width', aisle_width),
    )]
    width, height, block_width, block_height, aisle_width = values
    if width < block_width + 2 * aisle_width or height < block_height + 2 * aisle_width:
        raise ValueError('dimensions must fit a block and an aisle on each side')
    if seed is not None:
        seed = _integer(seed, 'seed')
    return width, height, block_width, block_height, aisle_width, seed


@dataclass
class DeliveryRangeSettings:
    width_min: int = 17
    width_max: int = 33
    height_min: int = 17
    height_max: int = 33
    block_width_min: int = 3
    block_width_max: int = 5
    block_height_min: int = 3
    block_height_max: int = 5
    aisle_width_min: int = 2
    aisle_width_max: int = 3

    def sample(self, seed=None):
        """Draw inclusive ranges independently; reject an infeasible draw."""
        if seed is not None:
            seed = _integer(seed, 'seed')
        rng = np.random.default_rng(seed)
        settings = {'seed': seed}
        for name in ('width', 'height', 'block_width', 'block_height', 'aisle_width'):
            lower = _integer(getattr(self, name + '_min'), name + '_min', 1)
            upper = _integer(getattr(self, name + '_max'), name + '_max', lower)
            settings[name] = int(rng.integers(lower, upper + 1))
        _validate(**settings)
        return settings


def generate_delivery(width=17, height=17, block_width=3, block_height=3, aisle_width=2,
                      seed=None) -> TaskLayout:
    """Place complete blocks and choose opposite-edge endpoints per block row."""
    width, height, block_width, block_height, aisle_width, seed = _validate(
        width, height, block_width, block_height, aisle_width, seed,
    )
    rng = np.random.default_rng(seed)
    obstacles = np.zeros((height, width), dtype=np.int8)
    pickups, deliveries = [], []
    for row in range(aisle_width, height - aisle_width - block_height + 1,
                     block_height + aisle_width):
        for column in range(aisle_width, width - aisle_width - block_width + 1,
                            block_width + aisle_width):
            obstacles[row:row + block_height, column:column + block_width] = 1
        pickups.append((int(rng.integers(row, row + block_height)), 0))
        deliveries.append((int(rng.integers(row, row + block_height)), width - 1))
    return TaskLayout(obstacles, pickups, deliveries, seed)
