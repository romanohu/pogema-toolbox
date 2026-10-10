import numpy as np
import pytest
from pogema import pogema_v0


def _api():
    from pogema_toolbox.generators import scenario_generator
    return scenario_generator


def _positions(config):
    return config.agents_xy, config.targets_xy


def test_components_use_four_neighbors_and_preserve_blocked_cells():
    labels = _api().connected_components([[0, 1, 0], [0, 1, 0], [1, 0, 1]])
    assert labels.tolist() == [[0, -1, 1], [0, -1, 1], [-1, 2, -1]]


@pytest.mark.parametrize('obstacles', [[], [[]], [[0], [0, 1]], [[2, 0]], [[float('nan'), 0]]])
def test_invalid_binary_grids_are_rejected(obstacles):
    with pytest.raises(ValueError):
        _api().ScenarioSampler(obstacles)


def test_sampler_is_seeded_explicit_and_retains_its_map():
    sampler = _api().ScenarioSampler(np.zeros((3, 5), dtype=int))
    first = sampler.generate(6, 42)
    assert _positions(first) == _positions(sampler.generate(6, 42))
    assert _positions(first) != _positions(sampler.generate(6, 43))
    assert first.map == [[0] * 5] * 3
    assert first.width == 5 and first.height == 3
    assert len(set(first.agents_xy)) == len(set(first.targets_xy)) == 6
    assert all(start != goal for start, goal in zip(*_positions(first)))
    env = pogema_v0(grid_config=first)
    env.reset()
    assert [tuple(cell) for cell in env.unwrapped.grid.get_agents_xy(ignore_borders=True)] == first.agents_xy
    assert [tuple(cell) for cell in env.unwrapped.grid.get_targets_xy(ignore_borders=True)] == first.targets_xy


def test_full_capacity_and_disconnected_placement_do_not_discard_valid_instances():
    sampler = _api().ScenarioSampler([[0, 0, 1, 0, 0, 0], [1, 1, 1, 1, 1, 1]])
    for seed in range(20):
        config = sampler.generate(5, seed)
        assert set(config.agents_xy) == set(config.targets_xy) == {(0, 0), (0, 1), (0, 3), (0, 4), (0, 5)}
        assert all(a != b and (a[1] < 2) == (b[1] < 2) for a, b in zip(*_positions(config)))
    with pytest.raises(ValueError, match='capacity'):
        sampler.generate(6, 0)
    with pytest.raises(ValueError, match='capacity'):
        _api().ScenarioSampler([[0, 1, 0], [1, 1, 1]]).generate(1, 0)


@pytest.mark.parametrize('kwargs', [
    {'num_agents': True}, {'num_agents': 1.5}, {'num_agents': 0},
    {'seed': True}, {'seed': 1.5}, {'seed': -1},
    {'sampling': 'unknown'}, {'on_target': 'unknown'},
    {'max_episode_steps': True}, {'max_episode_steps': 1.5}, {'max_episode_steps': 0},
    {'agents_xy': [(0, 0)]}, {'targets_xy': [(0, 1)]}, {'width': 9},
])
def test_invalid_generation_arguments_fail(kwargs):
    args = dict(num_agents=1, seed=0)
    args.update(kwargs)
    with pytest.raises(ValueError):
        _api().ScenarioSampler([[0, 0], [0, 0]]).generate(**args)


@pytest.mark.parametrize('pickups, deliveries', [
    ([(False, 0)], [(0, 1)]), ([(0.0, 0)], [(0, 1)]),
    ([(3, 0)], [(0, 1)]), ([(0, 0), (0, 0)], [(0, 1)]),
    ([], [(0, 1)]), ([(0, 0)], None),
    ([(1, 0)], [(0, 1)]),
])
def test_invalid_endpoints_are_rejected(pickups, deliveries):
    with pytest.raises(ValueError):
        _api().ScenarioSampler([[0, 0], [1, 0]], pickups, deliveries)


def test_endpoint_restart_sequences_alternate_roles_and_replay():
    pickups = [(0, 0), (1, 0)]
    deliveries = [(0, 3), (1, 3)]
    sampler = _api().ScenarioSampler([[0] * 4] * 2, pickups, deliveries)
    config = sampler.generate(2, 7, sampling='endpoints', on_target='restart', max_episode_steps=8)
    assert len(set(config.agents_xy)) == 2
    for start, sequence in zip(config.agents_xy, config.targets_xy):
        assert len(sequence) >= 9
        assert all(cell in (pickups if index % 2 == 0 else deliveries) for index, cell in enumerate(sequence))
    assert _positions(config) == _positions(sampler.generate(2, 7, sampling='endpoints', on_target='restart', max_episode_steps=8))
    left, right = pogema_v0(grid_config=config), pogema_v0(grid_config=config)
    left.reset()
    right.reset()
    left.unwrapped.current_goal_indices = [1, 1]
    right.unwrapped.current_goal_indices = [1, 1]
    for actions in [[4, 4], [4, 4], [4, 4], [0, 0]]:
        left.step(actions)
        right.step(actions)
        assert left.unwrapped.grid.get_agents_xy() == right.unwrapped.grid.get_agents_xy()
        assert left.unwrapped.grid.get_targets_xy() == right.unwrapped.grid.get_targets_xy()


def test_endpoint_capacity_accounts_for_components_and_overlap():
    sampler = _api().ScenarioSampler([[0, 0, 1, 0, 0]], [(0, 0), (0, 3)], [(0, 1)])
    config = sampler.generate(1, 2, sampling='endpoints')
    assert config.agents_xy == [(0, 0)] and config.targets_xy == [(0, 1)]
    with pytest.raises(ValueError, match='capacity'):
        sampler.generate(2, 2, sampling='endpoints')
    sampler = _api().ScenarioSampler([[0, 0]], [(0, 0), (0, 1)], [(0, 0), (0, 1)])
    config = sampler.generate(2, 0, sampling='endpoints')
    assert all(a != b for a, b in zip(*_positions(config)))


def test_uniform_restart_targets_stay_reachable_and_nontrivial():
    sampler = _api().ScenarioSampler([[0, 0, 1, 0, 0]])
    config = sampler.generate(4, 5, on_target='restart', max_episode_steps=4)
    assert len({sequence[0] for sequence in config.targets_xy}) == 4
    for start, sequence in zip(config.agents_xy, config.targets_xy):
        assert len(sequence) == 5
        previous = start
        for goal in sequence:
            assert goal != previous and (goal[1] < 2) == (start[1] < 2)
            previous = goal


def test_endpoint_restart_supports_fleets_larger_than_station_counts():
    sampler = _api().ScenarioSampler([[0, 0, 0, 0]], [(0, 0)], [(0, 3)])
    config = sampler.generate(4, 1, sampling='endpoints', on_target='restart', max_episode_steps=4)
    assert set(config.agents_xy) == {(0, 0), (0, 1), (0, 2), (0, 3)}
    assert config.targets_xy == [[(0, 0), (0, 3), (0, 0), (0, 3), (0, 0)]] * 4


def test_endpoint_restart_replay_consumes_explicit_sequences():
    sampler = _api().ScenarioSampler([[0, 0]], [(0, 0)], [(0, 1)])
    config = sampler.generate(1, 1, sampling='endpoints', on_target='restart', max_episode_steps=4)
    env = pogema_v0(grid_config=config)
    env.reset()
    # The evaluator starts the cursor after the already active first target.
    env.unwrapped.current_goal_indices = [1]
    if config.agents_xy[0] == (0, 1):
        env.step([3])
    else:
        env.step([0])
    assert [tuple(cell) for cell in env.unwrapped.grid.get_targets_xy(ignore_borders=True)] == [(0, 1)]
    env.step([4])
    assert [tuple(cell) for cell in env.unwrapped.grid.get_targets_xy(ignore_borders=True)] == [(0, 0)]


@pytest.mark.parametrize('pickups, deliveries, expected', [
    ([(0, 0), (0, 1)], [(0, 0)], [(0, 1), (0, 0), (0, 1)]),
    ([(0, 0)], [(0, 0), (0, 1)], [(0, 0), (0, 1), (0, 0)]),
])
def test_overlapping_endpoint_roles_never_strand_restart_sequences(pickups, deliveries, expected):
    sampler = _api().ScenarioSampler([[0, 0]], pickups, deliveries)
    for seed in range(10):
        config = sampler.generate(2, seed, sampling='endpoints', on_target='restart', max_episode_steps=2)
        assert config.targets_xy == [expected, expected]


def test_sampler_reuses_components_after_construction(monkeypatch):
    api = _api()
    sampler = api.ScenarioSampler([[0, 0], [0, 0]])

    def cannot_recompute(obstacles):
        raise AssertionError('components were recomputed during generation')

    monkeypatch.setattr(api, 'connected_components', cannot_recompute)
    assert len(sampler.generate(4, 0).agents_xy) == 4
    assert len(sampler.generate(4, 1, on_target='restart', max_episode_steps=2).targets_xy) == 4
