from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from pogema import pogema_v0
import pytest
import yaml

from pogema_toolbox import Instance, generate_suite, load_instances, save_instances
from pogema_toolbox.generators.maze_generator import MazeGenerator, MazeRangeSettings
from pogema_toolbox.generators.random_generator import MapRangeSettings, generate_map
from pogema_toolbox.generators.room_generator import RoomRangeSettings, generate_room
from pogema_toolbox.generators.scenario_generator import ScenarioSampler, connected_components
from pogema_toolbox.generators.warehouse_generator import WarehouseRangeSettings, generate_warehouse


ROOT = Path(__file__).resolve().parents[1]
GENERATORS = {
    'maze': (MazeRangeSettings, MazeGenerator.generate_maze),
    'warehouse': (WarehouseRangeSettings, generate_warehouse),
    'room': (RoomRangeSettings, generate_room),
    'random': (MapRangeSettings, generate_map),
}
LAYOUTS = {
    'maze': dict(width_min=8, width_max=10, height_min=7, height_max=9,
                 obstacle_density_min=0.15, obstacle_density_max=0.3,
                 wall_components_min=1, wall_components_max=3),
    'warehouse': dict(size_min=13, size_max=15, wall_width_min=2, wall_width_max=3,
                      wall_height_min=2, wall_height_max=3, vertical_gap=2),
    'room': dict(room_width_min=3, room_width_max=5, room_height_min=3, room_height_max=5,
                 num_rows_min=2, num_rows_max=3, num_cols_min=2, num_cols_max=3,
                 obstacle_density_min=0.1, obstacle_density_max=0.2,
                 uniform=False, only_centre_obstacles=True),
    'random': dict(width_min=8, width_max=10, height_min=7, height_max=9,
                   obstacle_density_min=0.2, obstacle_density_max=0.3),
}


def configured(dataset, protocol='nothing'):
    return dict(dataset=dataset, map_seeds=[3, 0], agent_counts=[2, 4], episodes=2,
                generation_seed=17, sampling='uniform', layout=deepcopy(LAYOUTS[dataset]),
                grid=dict(obs_radius=2, max_episode_steps=5, on_target=protocol,
                          collision_system='priority', observation_type='POMAPF'))


def original_map(dataset, layout, seed):
    settings_type, generator = GENERATORS[dataset]
    parameters = settings_type(**layout).sample(seed)
    result = generator(parameters) if dataset == 'random' else generator(**parameters)
    if isinstance(result, str):
        result = [[1 if cell == '#' else 0 for cell in row] for row in result.splitlines()]
    return parameters, np.asarray(result, dtype=np.int8)


def read_suite(result):
    index = Path(result['suite_path'])
    return json.loads(index.read_text()), index.parent


@pytest.mark.parametrize('dataset', GENERATORS)
@pytest.mark.parametrize('protocol', ['nothing', 'finish', 'restart'])
def test_original_layout_and_sampled_scenarios_are_saved_losslessly(tmp_path, dataset, protocol):
    benchmark = configured(dataset, protocol)
    before = deepcopy(benchmark)
    result = generate_suite(benchmark, output_dir=tmp_path)
    suite, directory = read_suite(result)
    assert benchmark == before and suite['generation'] == before
    assert result['groups'] == result['ready_groups'] == 4
    assert result['episodes'] == 8 and not result['invalid_groups']
    generator = GENERATORS[dataset][1]
    assert suite['provenance']['generator'] == generator.__module__ + '.' + generator.__qualname__
    assert [(group['metadata']['map_seed'], group['metadata']['num_agents'])
            for group in suite['groups']] == [(3, 2), (3, 4), (0, 2), (0, 4)]
    for position, group in enumerate(suite['groups']):
        metadata = group['metadata']
        parameters, obstacles = original_map(dataset, benchmark['layout'], metadata['map_seed'])
        assert metadata['layout_parameters'] == parameters
        assert metadata['dimensions'] == list(obstacles.shape)
        assert metadata['binary_grid_sha256'] == hashlib.sha256(obstacles.tobytes()).hexdigest()
        assert metadata['generation_seeds'] == [17 + (position // 2) * 2 + i for i in range(2)]
        path = directory / group['instances_path']
        assert group['sha256'] == hashlib.sha256(path.read_bytes()).hexdigest()
        instances = load_instances(path)
        sampler = ScenarioSampler(obstacles)
        labels = connected_components(obstacles)
        for instance, seed in zip(instances, metadata['generation_seeds']):
            expected = Instance.from_grid_config(sampler.generate(metadata['num_agents'], seed,
                                                sampling='uniform', **benchmark['grid']))
            assert instance.to_dict() == expected.to_dict()
            assert np.array_equal(instance.obstacles, obstacles)
            assert (instance.grid_config['height'], instance.grid_config['width']) == obstacles.shape
            assert len(set(instance.starts)) == metadata['num_agents']
            targets = instance.goal_sequences or [(goal,) for goal in instance.goals]
            assert all(labels[start] >= 0 and all(labels[target] == labels[start] for target in sequence)
                       for start, sequence in zip(instance.starts, targets))
            if protocol == 'restart':
                assert all(len(sequence) == 6 for sequence in instance.goal_sequences)
                env = pogema_v0(instance.to_grid_config())
                try:
                    env.reset()
                    sequences = env.unwrapped.get_lifelong_targets_xy(ignore_borders=True)
                    assert [[tuple(cell) for cell in sequence] for sequence in sequences] == [
                        list(sequence) for sequence in instance.goal_sequences]
                finally:
                    env.close()
            else:
                assert len(set(instance.goals)) == metadata['num_agents']
                assert instance.goal_sequences is None
        save_instances(instances, tmp_path / 'roundtrip.jsonl')
        assert (tmp_path / 'roundtrip.jsonl').read_bytes() == path.read_bytes()


def test_maze_records_original_odd_dimensions_without_cropping(tmp_path):
    benchmark = configured('maze')
    benchmark['layout'].update(width_min=8, width_max=8, height_min=6, height_max=6)
    suite, directory = read_suite(generate_suite(benchmark, output_dir=tmp_path))
    for group in suite['groups']:
        assert group['metadata']['layout_parameters']['width'] == 8
        assert group['metadata']['layout_parameters']['height'] == 6
        assert group['metadata']['dimensions'] == [7, 9]
        instance = load_instances(directory / group['instances_path'])[0]
        assert (len(instance.obstacles), len(instance.obstacles[0])) == (7, 9)


def test_maze_accepts_an_exclusive_probability_upper_bound_of_one(tmp_path):
    benchmark = configured('maze')
    benchmark['layout'].update(go_straight_min=0.75, go_straight_max=1.0)
    suite, _ = read_suite(generate_suite(benchmark, output_dir=tmp_path))
    assert all(group['status'] == 'ready' for group in suite['groups'])
    for group in suite['groups']:
        parameters, obstacles = original_map('maze', benchmark['layout'], group['metadata']['map_seed'])
        assert parameters['go_straight'] < 1
        assert group['metadata']['binary_grid_sha256'] == hashlib.sha256(obstacles.tobytes()).hexdigest()


@pytest.mark.parametrize('block_extra_space', [False, True])
def test_warehouse_wall_count_ranges_preserve_the_original_square_output(tmp_path, block_extra_space):
    benchmark = configured('warehouse')
    benchmark['layout'].update(num_wall_rows_min=2, num_wall_rows_max=3,
        num_wall_cols_min=3, num_wall_cols_max=4, side_pad_min=1, side_pad_max=2,
        horizontal_gap_min=1, horizontal_gap_max=2, vertical_gap_min=2, vertical_gap_max=3,
        block_extra_space=block_extra_space, wfi_instance=True)
    suite, directory = read_suite(generate_suite(benchmark, output_dir=tmp_path))
    assert all(group['status'] == 'ready' for group in suite['groups'])
    for group in suite['groups']:
        parameters, obstacles = original_map('warehouse', benchmark['layout'], group['metadata']['map_seed'])
        assert group['metadata']['layout_parameters'] == parameters
        assert group['metadata']['dimensions'] == list(obstacles.shape)
        assert group['metadata']['binary_grid_sha256'] == hashlib.sha256(obstacles.tobytes()).hexdigest()
        assert np.array_equal(load_instances(directory / group['instances_path'])[0].obstacles, obstacles)


@pytest.mark.parametrize('dataset', GENERATORS)
def test_map_and_scenario_seeds_are_independent_and_repeated_suites_match(tmp_path, dataset):
    benchmark = configured(dataset)
    results = [generate_suite(benchmark, output_dir=tmp_path / str(index)) for index in range(2)]
    first, directory = read_suite(results[0])
    repeated, repeated_directory = read_suite(results[1])
    assert first == repeated
    for group in first['groups']:
        assert (directory / group['instances_path']).read_bytes() == (
            repeated_directory / group['instances_path']).read_bytes()
    benchmark['generation_seed'] = 91
    changed, changed_directory = read_suite(generate_suite(benchmark, output_dir=tmp_path / 'changed'))
    old_positions, new_positions = [], []
    for old, new in zip(first['groups'], changed['groups']):
        for key in ('map_seed', 'layout_parameters', 'dimensions', 'binary_grid_sha256'):
            assert old['metadata'][key] == new['metadata'][key]
        old_instances = load_instances(directory / old['instances_path'])
        new_instances = load_instances(changed_directory / new['instances_path'])
        assert [instance.obstacles for instance in old_instances] == [instance.obstacles for instance in new_instances]
        old_positions.extend((instance.starts, instance.goals) for instance in old_instances)
        new_positions.extend((instance.starts, instance.goals) for instance in new_instances)
    assert old_positions != new_positions
    benchmark['map_seeds'] = [5, 1]
    changed_maps, _ = read_suite(generate_suite(benchmark, output_dir=tmp_path / 'maps'))
    assert [group['metadata']['generation_seeds'] for group in changed_maps['groups']] == [
        group['metadata']['generation_seeds'] for group in changed['groups']]
    for group in changed_maps['groups']:
        parameters, obstacles = original_map(dataset, benchmark['layout'], group['metadata']['map_seed'])
        assert group['metadata']['layout_parameters'] == parameters
        assert group['metadata']['binary_grid_sha256'] == hashlib.sha256(obstacles.tobytes()).hexdigest()


@pytest.mark.parametrize('dataset', GENERATORS)
def test_capacity_failure_is_saved_as_an_invalid_group(tmp_path, dataset):
    benchmark = configured(dataset)
    benchmark['agent_counts'] = [2, 99999]
    result = generate_suite(benchmark, output_dir=tmp_path)
    suite, directory = read_suite(result)
    assert result['ready_groups'] == 2 and result['episodes'] == 4
    assert len(result['invalid_groups']) == 2
    assert [group['status'] for group in suite['groups']] == ['ready', 'invalid', 'ready', 'invalid']
    for group in result['invalid_groups']:
        assert 'capacity' in group['error'] and group['episode_count'] == 0
        assert group['metadata']['layout_parameters']
        assert not (directory / group['name']).exists()


@pytest.mark.parametrize('dataset, updates, field', [
    ('random', {'obstacle_density_max': 1.1}, 'obstacle_density'),
    ('random', {'obstacle_density_min': -0.1}, 'obstacle_density'),
    ('random', {'obstacle_density_min': float('nan')}, 'obstacle_density'),
    ('random', {'obstacle_density_max': float('inf')}, 'obstacle_density'),
    ('random', {'obstacle_density_max': True}, 'obstacle_density'),
    ('random', {'width_min': 0}, 'width'),
    ('random', {'width_min': True}, 'width'),
    ('random', {'height_max': 3.5}, 'height'),
    ('random', {'width_min': 11, 'width_max': 10}, 'width'),
    ('maze', {'wall_components_min': -1}, 'wall_components'),
    ('maze', {'go_straight_min': 1.0, 'go_straight_max': 1.0}, 'go_straight'),
    ('maze', {'go_straight_min': -0.1}, 'go_straight'),
    ('maze', {'obstacle_density_max': 1.1}, 'obstacle_density'),
    ('room', {'num_rows_min': 0}, 'num_rows'),
    ('room', {'uniform': 'false'}, 'uniform'),
    ('room', {'only_centre_obstacles': 1}, 'only_centre_obstacles'),
    ('room', {'obstacle_density_min': 0.3, 'obstacle_density_max': 0.2}, 'obstacle_density'),
    ('warehouse', {'wall_width_min': 0}, 'wall_width'),
    ('warehouse', {'side_pad': -1}, 'side_pad'),
    ('warehouse', {'horizontal_gap': True}, 'horizontal_gap'),
    ('warehouse', {'vertical_gap_min': 2}, 'vertical_gap'),
    ('warehouse', {'num_wall_rows_min': 1, 'num_wall_rows_max': 2}, 'num_wall'),
    ('warehouse', {'block_extra_space': 'true'}, 'block_extra_space'),
    ('warehouse', {'unknown': 1}, 'Unknown layout'),
    ('maze', {'width_min': None, 'width_max': None}, 'width'),
    ('room', {'obstacle_density_min': None, 'obstacle_density_max': None}, 'obstacle_density'),
    ('random', {'height_min': None, 'height_max': None}, 'height'),
    ('warehouse', {'size_min': None, 'size_max': None}, 'size'),
])
def test_malformed_layout_fails_before_generator_or_output(tmp_path, monkeypatch, dataset, updates, field):
    benchmark = configured(dataset)
    benchmark['layout'].update(updates)
    generator = GENERATORS[dataset][1]

    def unexpected_generation(*args, **kwargs):
        pytest.fail('invalid layout reached the original generator')

    if dataset == 'maze':
        monkeypatch.setattr(MazeGenerator, 'generate_maze', unexpected_generation)
    else:
        monkeypatch.setattr(sys.modules[generator.__module__], generator.__name__, unexpected_generation)
    with pytest.raises(ValueError, match=field):
        generate_suite(benchmark, output_dir=tmp_path / 'unused')
    assert not (tmp_path / 'unused').exists()


@pytest.mark.parametrize('field, value', [
    ('map_seeds', []), ('map_seeds', [-1]), ('map_seeds', [True]), ('map_seeds', [0, 0]),
    ('agent_counts', [0]), ('agent_counts', [True]), ('agent_counts', [2, 2]),
    ('episodes', 0), ('episodes', True), ('generation_seed', -1), ('generation_seed', True),
    ('sampling', 'endpoints'), ('sampling', 'replay'),
    ('map_seeds', None), ('map_seeds', 3), ('agent_counts', 3), ('layout', None), ('grid', None),
])
def test_malformed_procedural_request_fails_before_output(tmp_path, field, value):
    benchmark = configured('random')
    benchmark[field] = value
    with pytest.raises(ValueError, match=field):
        generate_suite(benchmark, output_dir=tmp_path / 'unused')
    assert not (tmp_path / 'unused').exists()


def run_cli(tmp_path, benchmark):
    (tmp_path / 'benchmark.yaml').write_text(yaml.safe_dump(benchmark))
    return subprocess.run([sys.executable, '-m', 'pogema_toolbox', 'datasets', 'generate',
                           '--config', 'benchmark.yaml', '--output-dir', 'generated',
                           '--artifact-root', 'artifacts'], cwd=tmp_path,
                          env=dict(os.environ, PYTHONPATH=str(ROOT)), capture_output=True, text=True)


@pytest.mark.parametrize('dataset', GENERATORS)
def test_plain_yaml_cli_uses_the_public_procedural_route(tmp_path, dataset):
    benchmark = configured(dataset, 'finish')
    benchmark['suite_path'] = 'suites/' + dataset + '.json'
    completed = run_cli(tmp_path, benchmark)
    assert completed.returncode == 0, completed.stderr
    assert not completed.stderr and not (tmp_path / 'generated').exists()
    result = json.loads(completed.stdout)
    suite, directory = read_suite(result)
    assert Path(result['suite_path']) == tmp_path / 'artifacts/suites' / (dataset + '.json')
    assert result['ready_groups'] == 4 and result['episodes'] == 8
    expected, expected_directory = read_suite(generate_suite(configured(dataset, 'finish'), output_dir=tmp_path / 'api'))
    assert suite['groups'] == expected['groups']
    for group in suite['groups']:
        assert (directory / group['instances_path']).read_bytes() == (
            expected_directory / group['instances_path']).read_bytes()


@pytest.mark.parametrize('dataset', GENERATORS)
@pytest.mark.parametrize('field, value, message', [
    ('layout', {'unknown': 1}, 'Unknown layout'),
    ('map_seeds', '7', 'list'),
    ('grid', None, 'mapping'),
])
def test_cli_rejects_malformed_new_family_config(tmp_path, dataset, field, value, message):
    benchmark = configured(dataset)
    benchmark[field] = value
    completed = run_cli(tmp_path, benchmark)
    assert completed.returncode != 0 and message in completed.stderr
    assert 'Traceback' not in completed.stderr and not completed.stdout
    assert not (tmp_path / 'generated').exists() and not (tmp_path / 'artifacts').exists()
