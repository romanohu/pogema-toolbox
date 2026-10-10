from collections import deque

import numpy as np
import pytest
from pogema import pogema_v0

from pogema_toolbox.generators.scenario_generator import ScenarioSampler, connected_components


def _delivery(**settings):
    from pogema_toolbox.generators.delivery_generator import generate_delivery
    return generate_delivery(**settings)


def _sortation(**settings):
    from pogema_toolbox.generators.sortation_generator import generate_sortation
    return generate_sortation(**settings)


def _ascii(layout):
    return [''.join('#' if cell else '.' for cell in row) for row in layout.obstacles]


def _assert_connected(layout):
    labels = connected_components(layout.obstacles)
    assert set(labels[layout.obstacles == 0]) == {0}
    assert layout.pickup_cells and layout.delivery_cells
    assert len(set(layout.pickup_cells)) == len(layout.pickup_cells)
    assert len(set(layout.delivery_cells)) == len(layout.delivery_cells)
    assert not set(layout.pickup_cells).intersection(layout.delivery_cells)
    assert all(layout.obstacles[cell] == 0 for cell in layout.pickup_cells + layout.delivery_cells)
    assert {int(labels[cell]) for cell in layout.pickup_cells + layout.delivery_cells} == {0}


def test_delivery_literal_layout_preserves_two_cell_aisles_and_edge_endpoints():
    layout = _delivery(width=11, height=9, block_width=2, block_height=1, aisle_width=2, seed=7)
    assert _ascii(layout) == [
        '...........', '...........', '..##..##...', '...........', '...........',
        '..##..##...', '...........', '...........', '...........',
    ]
    assert layout.pickup_cells == [(2, 0), (5, 0)]
    assert layout.delivery_cells == [(2, 10), (5, 10)]
    assert layout.map_seed == 7
    _assert_connected(layout)


def test_delivery_endpoints_are_reachable_and_seeded():
    args = dict(width=17, height=17, block_width=3, block_height=3, aisle_width=2)
    first = _delivery(**args, seed=7)
    same = _delivery(**args, seed=7)
    other = _delivery(**args, seed=8)
    assert first.pickup_cells == same.pickup_cells and first.delivery_cells == same.delivery_cells
    assert (first.pickup_cells, first.delivery_cells) != (other.pickup_cells, other.delivery_cells)
    assert np.array_equal(first.obstacles, other.obstacles)
    _assert_connected(first)


def test_sortation_literal_layout_preserves_bin_spacing_and_named_roles():
    layout = _sortation(width=3, height=4, bin_spacing=1, pickup_count=10, delivery_count=1, seed=7)
    assert _ascii(layout) == ['...', '.#.', '...', '...']
    assert set(layout.pickup_cells) == {
        (0, 0), (0, 1), (0, 2), (1, 0), (1, 2),
        (2, 0), (2, 2), (3, 0), (3, 1), (3, 2),
    }
    assert layout.delivery_cells == [(2, 1)]
    assert layout.map_seed == 7
    _assert_connected(layout)


def test_sortation_literal_nonsquare_bin_grid_has_connected_transit_aisles():
    layout = _sortation(width=8, height=7, bin_spacing=2, pickup_count=3, delivery_count=4, seed=9)
    assert _ascii(layout) == [
        '........', '.#..#...', '........', '........', '.#..#...', '........', '........',
    ]
    assert set(layout.delivery_cells) == {(2, 1), (2, 4), (5, 1), (5, 4)}
    assert len(layout.pickup_cells) == 3
    assert all(row in (0, 6) or column in (0, 7) for row, column in layout.pickup_cells)
    _assert_connected(layout)


def test_sortation_endpoint_choices_are_seeded():
    args = dict(width=11, height=9, bin_spacing=2, pickup_count=4, delivery_count=3)
    first, same, other = (_sortation(**args, seed=seed) for seed in (7, 7, 8))
    assert (first.pickup_cells, first.delivery_cells) == (same.pickup_cells, same.delivery_cells)
    assert (first.pickup_cells, first.delivery_cells) != (other.pickup_cells, other.delivery_cells)
    assert np.array_equal(first.obstacles, other.obstacles)


@pytest.mark.parametrize('factory, args', [
    (_delivery, dict(width=3, height=3, block_width=1, block_height=1, aisle_width=1)),
    (_sortation, dict(width=3, height=4, bin_spacing=9, pickup_count=1, delivery_count=1)),
])
def test_tight_valid_dimensions_keep_connected_endpoints(factory, args):
    layout = factory(**args)
    assert layout.obstacles.shape == (args['height'], args['width'])
    _assert_connected(layout)


@pytest.mark.parametrize('updates', [
    {'width': 4}, {'height': 4}, {'width': True}, {'height': 5.0},
    {'block_width': 0}, {'block_height': -1}, {'aisle_width': 0},
    {'block_width': True}, {'aisle_width': 1.5}, {'seed': -1}, {'seed': True},
])
def test_delivery_rejects_invalid_dimensions_spacing_and_seeds(updates):
    args = dict(width=7, height=7, block_width=3, block_height=3, aisle_width=1, seed=7)
    args.update(updates)
    with pytest.raises(ValueError):
        _delivery(**args)


@pytest.mark.parametrize('updates', [
    {'width': 2}, {'height': 3}, {'width': True}, {'height': 4.0},
    {'bin_spacing': 0}, {'bin_spacing': True}, {'bin_spacing': 1.5},
    {'pickup_count': 0}, {'pickup_count': 11}, {'pickup_count': True},
    {'delivery_count': 0}, {'delivery_count': 2}, {'delivery_count': 1.5},
    {'seed': -1}, {'seed': True},
])
def test_sortation_rejects_invalid_geometry_and_endpoint_capacities(updates):
    args = dict(width=3, height=4, bin_spacing=1, pickup_count=1, delivery_count=1, seed=7)
    args.update(updates)
    with pytest.raises(ValueError):
        _sortation(**args)


@pytest.mark.parametrize('family', ['delivery', 'sortation'])
def test_range_settings_are_reproducible_inclusive_and_generate_exact_dimensions(family):
    if family == 'delivery':
        from pogema_toolbox.generators.delivery_generator import DeliveryRangeSettings
        settings = DeliveryRangeSettings(width_min=9, width_max=12, height_min=7, height_max=10,
                                         block_width_min=1, block_width_max=3,
                                         block_height_min=1, block_height_max=2,
                                         aisle_width_min=1, aisle_width_max=2)
        factory = _delivery
    else:
        from pogema_toolbox.generators.sortation_generator import SortationRangeSettings
        settings = SortationRangeSettings(width_min=7, width_max=10, height_min=9, height_max=12,
                                          bin_spacing_min=1, bin_spacing_max=2,
                                          pickup_count_min=1, pickup_count_max=3,
                                          delivery_count_min=1, delivery_count_max=2)
        factory = _sortation
    seen = {name[:-4]: set() for name in vars(settings) if name.endswith('_min')}
    for seed in range(50):
        sampled = settings.sample(seed)
        assert sampled == settings.sample(seed)
        assert sampled['seed'] == seed
        layout = factory(**sampled)
        assert layout.obstacles.shape == (sampled['height'], sampled['width'])
        _assert_connected(layout)
        for name in seen:
            value = sampled[name]
            assert type(value) is int
            assert getattr(settings, name + '_min') <= value <= getattr(settings, name + '_max')
            seen[name].add(value)
    for name, values in seen.items():
        assert getattr(settings, name + '_min') in values
        assert getattr(settings, name + '_max') in values


@pytest.mark.parametrize('family', ['delivery', 'sortation'])
@pytest.mark.parametrize('updates', [{'width_min': 10, 'width_max': 9}, {'width_min': True},
                                     {'height_max': 1.5}, {'width_min': 0}])
def test_invalid_ranges_fail(family, updates):
    if family == 'delivery':
        from pogema_toolbox.generators.delivery_generator import DeliveryRangeSettings as Settings
    else:
        from pogema_toolbox.generators.sortation_generator import SortationRangeSettings as Settings
    with pytest.raises(ValueError):
        Settings(**updates).sample(7)


def test_ranges_reject_infeasible_sampled_settings_without_resizing():
    from pogema_toolbox.generators.delivery_generator import DeliveryRangeSettings
    from pogema_toolbox.generators.sortation_generator import SortationRangeSettings
    with pytest.raises(ValueError):
        DeliveryRangeSettings(width_min=3, width_max=3, block_width_min=3, block_width_max=3).sample(7)
    with pytest.raises(ValueError, match='capacity'):
        SortationRangeSettings(width_min=3, width_max=3, height_min=4, height_max=4,
                               delivery_count_min=2, delivery_count_max=2).sample(7)


def _path(obstacles, start, goal):
    queue = deque([(start, [])])
    visited = {start}
    while queue:
        cell, actions = queue.popleft()
        if cell == goal:
            return actions
        for action, (dr, dc) in enumerate(((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1))):
            neighbor = (cell[0] + dr, cell[1] + dc)
            if (0 <= neighbor[0] < obstacles.shape[0] and 0 <= neighbor[1] < obstacles.shape[1]
                    and not obstacles[neighbor] and neighbor not in visited):
                visited.add(neighbor)
                queue.append((neighbor, actions + [action]))
    raise AssertionError('goal is unreachable')


@pytest.mark.parametrize('factory, args', [
    (_delivery, dict(width=7, height=6, block_width=2, block_height=2, aisle_width=1)),
    (_sortation, dict(width=7, height=6, bin_spacing=2, pickup_count=2, delivery_count=2)),
])
def test_generated_layout_scenarios_replay_uniform_endpoint_and_lifelong_goals(factory, args):
    layout = factory(**args, seed=7)
    sampler = ScenarioSampler(layout.obstacles, layout.pickup_cells, layout.delivery_cells)
    for sampling in ('uniform', 'endpoints'):
        config = sampler.generate(1, 4, sampling=sampling, max_episode_steps=100)
        env = pogema_v0(grid_config=config)
        env.reset()
        assert [tuple(cell) for cell in env.unwrapped.grid.get_agents_xy(ignore_borders=True)] == config.agents_xy
        assert [tuple(cell) for cell in env.unwrapped.grid.get_targets_xy(ignore_borders=True)] == config.targets_xy
        if sampling == 'endpoints':
            assert config.agents_xy[0] in layout.pickup_cells
            assert config.targets_xy[0] in layout.delivery_cells
        for action in _path(layout.obstacles, config.agents_xy[0], config.targets_xy[0]):
            env.step([action])
        assert tuple(env.unwrapped.grid.get_agents_xy(ignore_borders=True)[0]) == config.targets_xy[0]
    config = sampler.generate(1, 4, sampling='endpoints', on_target='restart', max_episode_steps=100)
    sequence = config.targets_xy[0]
    assert len(sequence) == 101
    assert all(goal in (layout.pickup_cells if index % 2 == 0 else layout.delivery_cells)
               for index, goal in enumerate(sequence))
    env = pogema_v0(grid_config=config)
    env.reset()
    env.unwrapped.current_goal_indices = [1]
    position = config.agents_xy[0]
    for index in range(3):
        for action in _path(layout.obstacles, position, sequence[index]) or [0]:
            env.step([action])
        position = sequence[index]
        assert tuple(env.unwrapped.grid.get_agents_xy(ignore_borders=True)[0]) == position
        assert tuple(env.unwrapped.grid.get_targets_xy(ignore_borders=True)[0]) == sequence[index + 1]
    with pytest.raises(ValueError, match='capacity'):
        sampler.generate(min(len(layout.pickup_cells), len(layout.delivery_cells)) + 1, 4, sampling='endpoints')
    with pytest.raises(ValueError):
        ScenarioSampler(layout.obstacles, [], layout.delivery_cells)
