import hashlib
import json
from dataclasses import FrozenInstanceError

import pytest
from pogema import GridConfig, pogema_v0

from pogema_toolbox import Instance, load_instances, save_instances


def test_saved_identity_and_immutable_config(tmp_path):
    payload = dict(obstacles=[[0, 0, 0], [0, 1, 0]], starts=[[0, 0]],
                   goals=[[1, 2]], grid_config={'on_target': 'nothing', 'max_episode_steps': 7},
                   goal_sequences=None)
    expected = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':'),
                                        allow_nan=False).encode()).hexdigest()
    instance = Instance(**payload)
    assert instance.id == expected
    save_instances([instance], tmp_path / 'instances.jsonl')
    assert load_instances(tmp_path / 'instances.jsonl')[0].to_dict() == dict(payload, id=expected)
    payload['grid_config']['max_episode_steps'] = 99
    instance.grid_config['max_episode_steps'] = 100
    assert instance.grid_config['max_episode_steps'] == 7
    with pytest.raises(FrozenInstanceError):
        instance.id = 'changed'
    tampered = instance.to_dict()
    tampered['grid_config']['max_episode_steps'] = 8
    with pytest.raises(ValueError, match='hash'):
        Instance.from_dict(tampered)


@pytest.mark.parametrize('protocol', ['nothing', 'finish', 'restart'])
def test_conversion_preserves_rectangular_borders_protocol_and_parameters(protocol):
    obstacles = [[1, 1, 1, 1, 1], [1, 0, 0, 0, 1], [1, 0, 1, 0, 1], [1, 1, 1, 1, 1]]
    sequences = [[[2, 3], [1, 2], [1, 3]]] if protocol == 'restart' else None
    options = dict(on_target=protocol, obs_radius=1, max_episode_steps=7, seed=13,
                   collision_system='priority', observation_type='POMAPF')
    original = Instance(obstacles, [[1, 1]], [[2, 3]], options, sequences)
    grid = original.to_grid_config()
    assert isinstance(grid, GridConfig)
    assert grid.map == obstacles
    assert grid.agents_xy == [[1, 1]]
    assert grid.targets_xy == (sequences if sequences else [[2, 3]])
    for name, value in options.items():
        assert getattr(grid, name) == value
    converted = Instance.from_grid_config(grid)
    assert (converted.obstacles, converted.starts, converted.goals, converted.goal_sequences) == (
        original.obstacles, original.starts, original.goals, original.goal_sequences)
    assert Instance.from_grid_config(converted.to_grid_config()).to_dict() == converted.to_dict()
    assert Instance.from_dict(original.to_dict()).id == original.id
    env = pogema_v0(grid)
    try:
        env.reset()
        assert env.unwrapped.get_obstacles(ignore_borders=True).tolist() == obstacles
        assert tuple(map(tuple, env.unwrapped.get_agents_xy(ignore_borders=True))) == original.starts
        if sequences:
            assert env.unwrapped.get_lifelong_targets_xy(ignore_borders=True) == sequences
    finally:
        env.close()


@pytest.mark.parametrize('options', [{'map': [[0, 0]]}, {'map_name': 'other'},
    {'agents_xy': [[0, 1]]}, {'targets_xy': [[0, 0]]}, {'num_agents': 2},
    {'possible_agents_xy': [[0, 1]]}, {'possible_targets_xy': [[0, 0]]},
    {'width': 3, 'height': 1}, {'size': 3}])
def test_conversion_rejects_conflicting_environment_data(options):
    instance = Instance([[0, 0]], [[0, 0]], [[0, 1]], options)
    with pytest.raises(ValueError, match='conflict|override'):
        instance.to_grid_config()


def test_conversion_requires_explicit_map_and_positions():
    with pytest.raises(ValueError, match='explicit'):
        Instance.from_grid_config(GridConfig(size=8, num_agents=1))


@pytest.mark.parametrize('starts, goals, config, sequences, message', [
    ([[0, 0], [0, 0]], [[0, 1], [0, 3]], {}, None, 'unique'),
    ([[0, 0]], [[0, 3]], {}, None, 'connected'),
    ([[0, 0]], [[0, 1]], {'on_target': 'restart'}, None, 'fixed goal sequences'),
    ([[0, 0]], [[0, 1]], {'on_target': 'restart'}, [[[0, 1], [0, 3]]], 'connected'),
])
def test_invalid_positions_and_restart_sequences(starts, goals, config, sequences, message):
    with pytest.raises(ValueError, match=message):
        Instance([[0, 0, 1, 0]], starts, goals, config, sequences)


def test_conversion_owns_large_rectangular_dimensions():
    instance = Instance([[0] * 10 for _ in range(2)], [[1, 9]], [[0, 8]],
                        {'on_target': 'finish', 'empty_outside': False, 'persistent': True,
                         'max_episode_steps': 19, 'auto_reset': False})
    grid = instance.to_grid_config()
    assert (grid.height, grid.width) == (2, 10)
    assert grid.agents_xy == [[1, 9]] and grid.targets_xy == [[0, 8]]
    for key, value in instance.grid_config.items():
        assert getattr(grid, key) == value


def test_original_sparse_config_id_remains_valid():
    instance = Instance.from_dict(dict(obstacles=[[0, 0, 1, 0]], starts=[[0, 0]], goals=[[0, 1]],
        grid_config={'on_target': 'nothing'}, goal_sequences=None,
        id='3c6142bb9a90b36e063a590f8b7667f22a4d5f6e306cb478b3cb91f99b27d6d4'))
    instance.to_grid_config()
    assert instance.id == '3c6142bb9a90b36e063a590f8b7667f22a4d5f6e306cb478b3cb91f99b27d6d4'


@pytest.mark.parametrize('captured', [False, True])
def test_single_cell_conversion_preserves_explicit_grid_and_saved_payload(captured):
    grid = GridConfig(map=[[0]], agents_xy=[[0, 0]], targets_xy=[[0, 0]],
                      num_agents=1, on_target='nothing')
    instance = (Instance.from_grid_config(grid) if captured else
                Instance([[0]], [[0, 0]], [[0, 0]], {'on_target': 'nothing'}))
    saved = instance.to_dict()
    expected = hashlib.sha256(json.dumps({key: value for key, value in saved.items() if key != 'id'},
        sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    converted = instance.to_grid_config()
    assert converted.map == [[0]]
    assert converted.agents_xy == converted.targets_xy == [[0, 0]]
    assert (converted.width, converted.height, converted.size, converted.num_agents) == (1, 1, 1, 1)
    assert converted.on_target == 'nothing'
    assert instance.to_dict() == saved and instance.id == expected
    round_trip = Instance.from_grid_config(converted)
    assert Instance.from_grid_config(round_trip.to_grid_config()).to_dict() == round_trip.to_dict()
    if captured:
        assert round_trip.to_dict() == saved
