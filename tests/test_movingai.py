import io

import numpy as np
import pytest
from pogema import pogema_v0


def _api():
    from pogema_toolbox import moving_ai_ingestion as ingestion
    return ingestion


def test_original_border_and_coordinates():
    api = _api()
    obstacles = api.parse_movingai_map('type octile\nheight 2\nwidth 3\nmap\n...\n@..\n')
    problems = api.parse_movingai_scenarios('version 1\n0 fixture.map 3 2 2 0 1 1 2\n', 'fixture.map', obstacles)
    config = api.movingai_grid_config(obstacles, problems, 1)
    assert config.map == [[0, 0, 0], [1, 0, 0]]
    assert config.agents_xy == [(0, 2)]
    assert config.targets_xy == [(1, 1)]
    env = pogema_v0(grid_config=config)
    env.reset()
    assert [tuple(cell) for cell in env.unwrapped.grid.get_agents_xy(ignore_borders=True)] == [(0, 2)]
    assert [tuple(cell) for cell in env.unwrapped.grid.get_targets_xy(ignore_borders=True)] == [(1, 1)]


@pytest.mark.parametrize('text', [
    'type octile\nheight 2\nwidth 3\nmap\n...\n',
    'type octile\nheight 2\nwidth 3\nmap\n..\n..\n..\n',
    'type octile\nheight 2\nwidth 3\nmap\n...\n...\n...\n',
    'type octile\nheight 2\nwidth 3\nmap\n.O.\n...\n',
    'type octile\nheight 0\nwidth 3\nmap\n',
    'type octile\nheight 2\nwidth 3\ngrid\n...\n...\n',
])
def test_malformed_maps_are_rejected(text):
    with pytest.raises(ValueError):
        _api().parse_movingai_map(text)


def test_tree_cells_are_blocked_and_legacy_border_default_remains():
    text = 'type octile\nheight 3\nwidth 3\nmap\n@@@\n@T@\n@@@\n'
    assert _api().parse_movingai_map(text).tolist() == [[1, 1, 1]] * 3
    assert _api().map_to_grid(io.BytesIO(text.encode())) == '#'


def test_prefix_preserves_literal_source_order_and_cross_agent_overlap():
    api = _api()
    obstacles = np.zeros((2, 3), dtype=int)
    text = 'version 1\n4 fixture.map 3 2 2 1 0 0 3\n1 fixture.map 3 2 0 0 1 1 2.5\n'
    problems = api.parse_movingai_scenarios(text, 'fixture.map', obstacles)
    assert [(p.bucket, p.start, p.goal, p.distance, p.map_name) for p in problems] == [
        (4, (1, 2), (0, 0), 3.0, 'fixture.map'),
        (1, (0, 0), (1, 1), 2.5, 'fixture.map'),
    ]
    config = api.movingai_grid_config(obstacles, problems, 2)
    assert config.agents_xy == [(1, 2), (0, 0)]
    assert config.targets_xy == [(0, 0), (1, 1)]
    assert api.movingai_grid_config(obstacles, problems, 1).agents_xy == [(1, 2)]


@pytest.mark.parametrize('row', [
    '0 other.map 3 2 0 0 1 1 2',
    '0 fixture.map 2 3 0 0 1 1 2',
    '0 fixture.map 3 2 3 0 1 1 2',
    '0 fixture.map 3 2 0 1 1 1 2',
    '0 fixture.map 3 2 0.0 0 1 1 2',
    '-1 fixture.map 3 2 0 0 1 1 2',
    '0 fixture.map 3 2 0 0 1 1 nan',
    '0 fixture.map 3 2 0 0 1 1 -2',
    '0 fixture.map 3 2 0 0 1 1',
])
def test_invalid_scenario_rows_are_rejected(row):
    with pytest.raises(ValueError):
        _api().parse_movingai_scenarios('version 1\n' + row, 'fixture.map', [[0, 0, 0], [1, 0, 0]])


@pytest.mark.parametrize('count', [True, 1.5, 0, -1, 3])
def test_prefix_count_is_validated(count):
    api = _api()
    problems = api.parse_movingai_scenarios('version 1\n0 fixture.map 3 2 0 0 1 1 2\n', 'fixture.map', [[0] * 3] * 2)
    with pytest.raises(ValueError):
        api.movingai_grid_config([[0] * 3] * 2, problems, count)


def test_unreachable_and_duplicate_prefixes_are_rejected():
    api = _api()
    obstacles = [[0, 1, 0], [0, 1, 0]]
    text = 'version 1\n0 fixture.map 3 2 0 0 2 0 2\n'
    with pytest.raises(ValueError):
        api.parse_movingai_scenarios(text, 'fixture.map', obstacles)
    problems = api.parse_movingai_scenarios('version 1\n0 fixture.map 3 2 0 0 0 1 1\n', 'fixture.map', obstacles)
    with pytest.raises(ValueError):
        api.movingai_grid_config(obstacles, problems * 2, 2)
    with pytest.raises(ValueError):
        api.movingai_grid_config(obstacles, problems, 1, on_target='restart')


def test_parse_and_prefix_reuse_component_labels_for_the_same_map(monkeypatch):
    api = _api()
    obstacles = [[0, 0, 0, 0], [0, 1, 1, 0]]
    text = 'version 1\n0 cache.map 4 2 0 0 3 1 4\n'
    api.parse_movingai_scenarios(text, 'cache.map', obstacles)

    def cannot_recompute(obstacles):
        raise AssertionError('components were recomputed for another scenario on the same map')

    monkeypatch.setattr(api, 'connected_components', cannot_recompute)
    problems = api.parse_movingai_scenarios(text, 'cache.map', obstacles)
    assert api.movingai_grid_config(obstacles, problems, 1).targets_xy == [(1, 3)]
