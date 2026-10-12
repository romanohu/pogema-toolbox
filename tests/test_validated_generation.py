import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from pogema_toolbox import generate_suite, load_instances


def configured(dataset='random'):
    layouts = {
        'random': dict(width_min=8, width_max=10, height_min=7, height_max=9,
                       obstacle_density_min=.2, obstacle_density_max=.3),
        'maze': dict(width_min=8, width_max=10, height_min=7, height_max=9,
                     obstacle_density_min=.15, obstacle_density_max=.3,
                     wall_components_min=1, wall_components_max=3),
        'warehouse': dict(size_min=13, size_max=15, wall_width_min=2, wall_width_max=3,
                          wall_height_min=2, wall_height_max=3, vertical_gap=2),
        'room': dict(room_width_min=3, room_width_max=5, room_height_min=3, room_height_max=5,
                     num_rows_min=2, num_rows_max=3, num_cols_min=2, num_cols_max=3,
                     obstacle_density_min=.1, obstacle_density_max=.2),
        'delivery': dict(width_min=9, width_max=12, height_min=9, height_max=12,
                         block_width_min=2, block_width_max=3, block_height_min=2,
                         block_height_max=3, aisle_width_min=1, aisle_width_max=2),
        'sortation': dict(width_min=9, width_max=12, height_min=9, height_max=12,
                          bin_spacing_min=2, bin_spacing_max=3, pickup_count_min=3,
                          pickup_count_max=5, delivery_count_min=3, delivery_count_max=5),
    }
    return dict(dataset=dataset, map_seeds=[3], agent_counts=[2], episodes=2,
                generation_seed=17, sampling='uniform', layout=layouts[dataset],
                grid=dict(obs_radius=2, max_episode_steps=5, on_target='nothing',
                          collision_system='priority', observation_type='POMAPF'))


def saved(result):
    index = Path(result['suite_path'])
    return json.loads(index.read_text()), index.parent


@pytest.mark.parametrize('dataset', ['random', 'maze', 'warehouse', 'room', 'delivery', 'sortation'])
def test_rejected_episode_replaces_its_whole_map_and_preserves_accepted_episodes(tmp_path, dataset):
    benchmark = configured(dataset)
    original, directory = saved(generate_suite(benchmark, output_dir=tmp_path / 'original'))
    initial = load_instances(directory / original['groups'][0]['instances_path'])

    def validator(instance):
        accepted = instance.grid_config['seed'] != 17
        return {'accepted': accepted, 'status': 'accepted' if accepted else 'over_step_limit',
                'makespan': 2 if accepted else 8}

    result = generate_suite(benchmark, output_dir=tmp_path / 'validated', validator=validator, max_attempts=3)
    suite, directory = saved(result)
    assert result['episodes'] == 2 and result['ready_groups'] == 1
    group = suite['groups'][0]
    instances = load_instances(directory / group['instances_path'])
    assert instances[0].id != initial[0].id
    assert instances[0].obstacles != initial[0].obstacles
    assert instances[1].to_dict() == initial[1].to_dict()
    metadata = group['metadata']
    assert metadata['requested_map_seed'] == 3
    assert metadata['generation_seeds'] == [19, 18]
    assert 'binary_grid_sha256' not in metadata
    attempts = metadata['validation']['attempts']
    assert [(a['map_seed'], a['generation_seed'], a['accepted']) for a in attempts] == [
        (3, 17, False), (4, 19, True), (3, 18, True)]
    assert [a['instance_id'] for a in attempts if a['accepted']] == [i.id for i in instances]
    for instance, actual_map in zip(instances, metadata['instances']):
        grid = np.asarray(instance.obstacles, dtype=np.int8)
        assert actual_map['instance_id'] == instance.id
        assert actual_map['binary_grid_sha256'] == hashlib.sha256(grid.tobytes()).hexdigest()
        assert actual_map['dimensions'] == list(grid.shape)


def test_validated_instances_and_retry_records_reproduce_from_the_same_seeds(tmp_path):
    def validator(instance):
        return {'accepted': instance.grid_config['seed'] >= 19, 'status': 'threshold'}

    benchmark = configured()
    first, first_dir = saved(generate_suite(benchmark, output_dir=tmp_path / 'a', validator=validator))
    second, second_dir = saved(generate_suite(benchmark, output_dir=tmp_path / 'b', validator=validator))
    assert first == second
    relative = first['groups'][0]['instances_path']
    assert (first_dir / relative).read_bytes() == (second_dir / relative).read_bytes()


def test_replacement_seed_streams_do_not_overlap_other_episode_slots(tmp_path):
    benchmark = configured()
    benchmark['map_seeds'] = [3, 0]
    benchmark['agent_counts'] = [2, 3]
    suite, _ = saved(generate_suite(benchmark, output_dir=tmp_path,
        validator=lambda instance: {'accepted': instance.grid_config['seed'] >= 21, 'status': 'threshold'}))
    accepted = [attempt for group in suite['groups']
                for attempt in group['metadata']['validation']['attempts'] if attempt['accepted']]
    assert [attempt['map_seed'] for attempt in accepted] == list(range(4, 12))
    assert [attempt['generation_seed'] for attempt in accepted] == list(range(21, 29))


def test_exhausted_episode_does_not_save_a_shorter_ready_group(tmp_path):
    def validator(instance):
        return {'accepted': instance.grid_config['seed'] == 17, 'status': 'no_solution'}

    result = generate_suite(configured(), output_dir=tmp_path, validator=validator, max_attempts=2)
    suite, directory = saved(result)
    assert result['episodes'] == result['ready_groups'] == 0
    group = suite['groups'][0]
    assert group['status'] == 'invalid' and group['requested_episode_count'] == 2
    assert group['episode_count'] == 0 and 'attempt' in group['error']
    assert [a['generation_seed'] for a in group['metadata']['validation']['attempts']] == [17, 18, 20]
    assert not list(directory.rglob('instances.jsonl'))


def test_unsampleable_candidates_also_regenerate_with_a_bounded_attempt_count(tmp_path):
    benchmark = configured()
    benchmark['agent_counts'] = [10000]
    result = generate_suite(benchmark, output_dir=tmp_path,
                           validator=lambda instance: pytest.fail('An invalid candidate reached validation'),
                           max_attempts=2)
    suite, _ = saved(result)
    assert result['ready_groups'] == 0
    attempts = suite['groups'][0]['metadata']['validation']['attempts']
    assert [a['status'] for a in attempts] == ['generation_error', 'generation_error']
    assert [a['map_seed'] for a in attempts] == [3, 4]


def test_validator_errors_abort_instead_of_becoming_rejections(tmp_path):
    def validator(instance):
        raise ValueError('Invalid solver output')

    with pytest.raises(RuntimeError, match='Invalid solver output'):
        generate_suite(configured(), output_dir=tmp_path, validator=validator)
    assert not (tmp_path / 'suite.json').exists()
    assert not list(tmp_path.rglob('instances.jsonl'))


@pytest.mark.parametrize('report', [None, True, {}, {'accepted': 1, 'status': 'ok'},
    {'accepted': True, 'status': ''}])
def test_invalid_validator_reports_abort(tmp_path, report):
    with pytest.raises(TypeError, match='validator must return'):
        generate_suite(configured(), output_dir=tmp_path, validator=lambda instance: report)
    assert not (tmp_path / 'suite.json').exists()


def test_noncallable_validator_fails_before_writing(tmp_path):
    with pytest.raises(TypeError, match='validator must be callable'):
        generate_suite(configured(), output_dir=tmp_path / 'new', validator=True)
    assert not (tmp_path / 'new').exists()


@pytest.mark.parametrize('attempts', [0, -1, True, 1.5])
def test_invalid_attempt_limit_fails_before_writing(tmp_path, attempts):
    destination = tmp_path / 'new'
    with pytest.raises(ValueError, match='max_attempts'):
        generate_suite(configured(), output_dir=destination, validator=lambda instance: {}, max_attempts=attempts)
    assert not destination.exists()


@pytest.mark.parametrize('dataset', ['delivery', 'sortation'])
def test_invalid_task_layout_ranges_are_configuration_errors_not_rejections(tmp_path, dataset):
    benchmark = configured(dataset)
    benchmark['layout'].update(width_min=12, width_max=9)
    with pytest.raises(ValueError, match='width_min'):
        generate_suite(benchmark, output_dir=tmp_path / 'new', validator=lambda instance: {}, max_attempts=2)
    assert not (tmp_path / 'new').exists()


@pytest.mark.parametrize('key,value', [('collision_system', 'unknown'), ('observation_type', 'unknown'),
    ('obs_radius', -1), ('num_agents', 1)])
def test_static_grid_configuration_errors_fail_before_sampling_or_writing(tmp_path, key, value):
    benchmark = configured()
    benchmark['grid'][key] = value
    with pytest.raises(ValueError):
        generate_suite(benchmark, output_dir=tmp_path / 'new', validator=lambda instance: {}, max_attempts=2)
    assert not (tmp_path / 'new').exists()


@pytest.mark.parametrize('dataset', ['random', 'maze', 'warehouse', 'room', 'delivery', 'sortation'])
def test_generic_toolbox_validator_preserves_valid_restart_protocol(tmp_path, dataset):
    benchmark = configured(dataset)
    benchmark['grid']['on_target'] = 'restart'
    result = generate_suite(benchmark, output_dir=tmp_path,
                           validator=lambda instance: {'accepted': True, 'status': 'accepted'})
    suite, directory = saved(result)
    assert result['episodes'] == 2 and result['ready_groups'] == 1
    instances = load_instances(directory / suite['groups'][0]['instances_path'])
    assert all(i.grid_config['on_target'] == 'restart' for i in instances)
    assert all(len(sequence) == 6 for i in instances for sequence in i.goal_sequences)


@pytest.mark.parametrize('dataset', ['movingai', 'cities_tiles', 'qd_mapper'])
def test_fixed_datasets_reject_the_regenerated_map_mode_before_reading_data(tmp_path, dataset):
    benchmark = configured()
    benchmark['dataset'] = dataset
    with pytest.raises(ValueError, match='procedural'):
        generate_suite(benchmark, output_dir=tmp_path / 'new', validator=lambda instance: {})
    assert not (tmp_path / 'new').exists()
