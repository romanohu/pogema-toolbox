import hashlib
import json

import pytest
from pogema import pogema_v0


def _api():
    from pogema_toolbox.generators import qd_mapper
    return qd_mapper


def _write(tmp_path, **updates):
    data = dict(name='fixture', weight=False, n_row=2, n_col=3,
                layout=['...', '@..'], start=[[2, 0], [5, 1]], goal=[[4, 1], [1, 4]])
    data.update(updates)
    path = tmp_path / 'map.json'
    path.write_text(json.dumps(data))
    return path


def test_qd_linear_indices_are_row_major(tmp_path):
    path = tmp_path / 'map.json'
    path.write_text('{"name":"fixture","weight":false,"n_row":2,"n_col":3,"layout":["...","@.."],"start":[[2]],"goal":[[4]]}')
    config = _api().load_qd_map(path).grid_config(0, 1)
    assert config.agents_xy == [(0, 2)]
    assert config.targets_xy == [(1, 1)]
    assert config.map == [[0, 0, 0], [1, 0, 0]]
    assert (config.height, config.width) == (2, 3)


def test_qd_preserves_multiple_scenarios_prefix_order_and_provenance(tmp_path):
    path = _write(tmp_path)
    qd_map = _api().load_qd_map(path)
    assert qd_map.name == 'fixture'
    assert qd_map.source_hash == hashlib.sha256(path.read_bytes()).hexdigest()
    assert qd_map.obstacles.tolist() == [[0, 0, 0], [1, 0, 0]]
    assert [[(p.start, p.goal) for p in scenario] for scenario in qd_map.problems] == [
        [((0, 2), (1, 1)), ((0, 0), (0, 1))],
        [((1, 2), (0, 1)), ((0, 1), (1, 1))],
    ]
    first = qd_map.grid_config(0, 1, seed=7, max_episode_steps=20)
    assert first.agents_xy == [(0, 2)] and first.targets_xy == [(1, 1)]
    assert first.seed == 7 and first.max_episode_steps == 20
    second = qd_map.grid_config(1, 2, on_target='finish')
    assert second.agents_xy == [(1, 2), (0, 1)]
    assert second.targets_xy == [(0, 1), (1, 1)]
    env = pogema_v0(grid_config=second)
    env.reset()
    assert [tuple(cell) for cell in env.unwrapped.grid.get_agents_xy(ignore_borders=True)] == second.agents_xy
    assert [tuple(cell) for cell in env.unwrapped.grid.get_targets_xy(ignore_borders=True)] == second.targets_xy


@pytest.mark.parametrize('updates', [
    {'weight': True}, {'weight': 0}, {'weight': None}, {'weight': 'false'},
    {'n_row': True}, {'n_row': 2.0}, {'n_row': 0}, {'n_row': 3},
    {'n_col': False}, {'n_col': 3.0}, {'n_col': -1}, {'n_col': 2},
    {'layout': ['...', '@.']}, {'layout': ['...', '@..', '...']},
    {'layout': '..\n@..'}, {'layout': [[0, 0, 0], [1, 0, 0]]},
    {'layout': ['.T.', '@..']}, {'layout': ['.#.', '@..']},
    {'layout': ['.1.', '@..']}, {'layout': ['. .', '@..']},
    {'name': ''}, {'name': 1},
    {'start': [[2, 0]]}, {'goal': [[4]]}, {'start': []}, {'goal': []},
    {'start': [[], [5, 1]], 'goal': [[], [1, 4]]},
    {'start': '2'}, {'goal': [4, [1, 4]]},
    {'start': [[True, 0], [5, 1]]}, {'goal': [[4.0, 1], [1, 4]]},
    {'start': [[-1, 0], [5, 1]]}, {'goal': [[6, 1], [1, 4]]},
    {'start': [[3, 0], [5, 1]]}, {'goal': [[3, 1], [1, 4]]},
    {'start': [[2, 2], [5, 1]]}, {'goal': [[4, 4], [1, 4]]},
    {'start': [[2, 0], [5, 5]]}, {'goal': [[4, 1], [1, 1]]},
    {'layout': ['.@.', '@@.'], 'start': [[0]], 'goal': [[2]]},
])
def test_qd_rejects_invalid_maps_and_every_stored_scenario(tmp_path, updates):
    with pytest.raises(ValueError):
        _api().load_qd_map(_write(tmp_path, **updates))


@pytest.mark.parametrize('text', [
    '{}', '[]', 'null', '{',
    '{"name":"fixture","n_row":2,"n_col":3,"layout":["...","@.."],"start":[[2]],"goal":[[4]]}',
])
def test_qd_rejects_missing_fields_and_invalid_json(tmp_path, text):
    path = tmp_path / 'map.json'
    path.write_text(text)
    with pytest.raises(ValueError):
        _api().load_qd_map(path)


@pytest.mark.parametrize('scenario_index', [-1, True, 0.0, 2])
def test_qd_rejects_invalid_scenario_indices(tmp_path, scenario_index):
    with pytest.raises(ValueError):
        _api().load_qd_map(_write(tmp_path)).grid_config(scenario_index, 1)


@pytest.mark.parametrize('kwargs', [
    {'num_agents': True}, {'num_agents': 0}, {'num_agents': 1.5}, {'num_agents': 3},
    {'on_target': 'restart'}, {'map': [[0]]}, {'agents_xy': [(0, 0)]},
    {'targets_xy': [(0, 0)]}, {'width': 3}, {'FREE': 0},
])
def test_qd_prefix_arguments_cannot_replace_stored_data(tmp_path, kwargs):
    arguments = dict(num_agents=1)
    arguments.update(kwargs)
    with pytest.raises(ValueError):
        _api().load_qd_map(_write(tmp_path)).grid_config(0, **arguments)


def test_qd_allows_cross_role_overlap_and_stationary_pairs(tmp_path):
    qd_map = _api().load_qd_map(_write(tmp_path, start=[[2, 0]], goal=[[0, 2]]))
    config = qd_map.grid_config(0, 2)
    assert config.agents_xy == [(0, 2), (0, 0)]
    assert config.targets_xy == [(0, 0), (0, 2)]
    config = _api().load_qd_map(_write(tmp_path, start=[[2]], goal=[[2]])).grid_config(0, 1)
    assert config.agents_xy == config.targets_xy == [(0, 2)]
