from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

from pogema_toolbox import generate_instances, generate_suite, load_instances, ToolboxRegistry


ROOT = Path(__file__).resolve().parents[1]
BASELINE_NEURAL = Path('/tmp/neural-mapf-organized')
BASELINE_TOOLBOX = Path('/tmp/neural-mapf-toolbox-benchmarks')


def benchmark_defaults(dataset):
    grid = dict(obs_radius=5, max_episode_steps=256, on_target='nothing',
                collision_system='priority', observation_type='POMAPF')
    benchmark = dict(dataset=dataset, manifest_path=None, episodes=1, generation_seed=0,
                     instances_path=None)
    if dataset == 'cities_tiles':
        benchmark.update(mode='official', map_name=None, agent_counts=[64, 128, 192, 256])
    elif dataset in ('movingai', 'qd_mapper'):
        benchmark.update(grid=grid, map_names=None, agent_counts=[8, 16, 32],
                         sampling='replay', scenario_numbers=[1, 2])
        if dataset == 'movingai':
            benchmark['scenario_family'] = 'random'
        else:
            benchmark['collections'] = ['CBS', 'EECBS']
    else:
        grid.update(obs_radius=2, max_episode_steps=32, on_target='restart')
        layout = dict(width_min=9, width_max=9, height_min=9, height_max=9)
        if dataset == 'delivery':
            layout.update(block_width_min=2, block_width_max=2, block_height_min=2,
                          block_height_max=2, aisle_width_min=1, aisle_width_max=1)
        else:
            layout.update(bin_spacing_min=2, bin_spacing_max=2, pickup_count_min=4,
                          pickup_count_max=4, delivery_count_min=4, delivery_count_max=4)
        benchmark.update(grid=grid, layout=layout, agent_counts=[2], map_seeds=[0], sampling='endpoints')
    return benchmark


@pytest.fixture
def fixed_sources(tmp_path):
    moving = tmp_path / 'maps/movingai/data'
    (moving / 'scen-random').mkdir(parents=True)
    map_bytes = b'type octile\nheight 2\nwidth 3\nmap\n...\n...\n'
    scenario_bytes = b'version 1\n0 tiny.map 3 2 2 1 0 0 3\n0 tiny.map 3 2 0 0 2 1 3\n'
    (moving / 'tiny.map').write_bytes(map_bytes)
    members = {'tiny.map': hashlib.sha256(map_bytes).hexdigest()}
    for number in [1, 2]:
        relative = f'scen-random/tiny-random-{number}.scen'
        (moving / relative).write_bytes(scenario_bytes)
        members[relative] = hashlib.sha256(scenario_bytes).hexdigest()
    moving_manifest = dict(schema_version=1, dataset='movingai', maps=['tiny'],
                           attribution='Owned controlled fixture', rights='Owned test content',
                           archives=[dict(name='fixture', members=members)])
    qd = tmp_path / 'maps/qd_mapper/data'
    raw = json.dumps({'name': 'literal', 'weight': False, 'n_row': 2, 'n_col': 4,
                      'layout': ['....', '....'], 'start': [[7, 0], [1, 4]],
                      'goal': [[0, 7], [6, 3]]}).encode()
    archives = []
    for collection in ('CBS', 'EECBS'):
        (qd / collection).mkdir(parents=True)
        relative = collection + '/tiny.json'
        (qd / relative).write_bytes(raw)
        archives.append(dict(name=collection, members={relative: hashlib.sha256(raw).hexdigest()}))
    qd_manifest = dict(schema_version=1, dataset='qd_mapper', rights={'status': 'permission_unverified'},
                       archives=archives)
    for dataset, manifest in [('movingai', moving_manifest), ('qd_mapper', qd_manifest)]:
        (tmp_path / (dataset + '.yaml')).write_text(yaml.safe_dump(manifest))
        receipt = dict(schema_version=1, dataset=dataset, acquired_via='verified_local_directory',
                       rights=manifest['rights'], archives=manifest['archives'])
        (tmp_path / 'maps' / dataset / 'data/receipt.json').write_text(json.dumps(receipt))
    return {'movingai': moving_manifest, 'qd_mapper': qd_manifest}


def configured(dataset, mode=None):
    benchmark = benchmark_defaults(dataset)
    benchmark['generation_seed'] = 17
    benchmark['episodes'] = 2
    benchmark['agent_counts'] = [2, 99999]
    if dataset in ('movingai', 'qd_mapper'):
        benchmark['sampling'] = mode or 'replay'
        benchmark['scenario_numbers'] = [2, 1]
        benchmark['grid']['max_episode_steps'] = 3
        if dataset == 'movingai':
            benchmark['map_names'] = ['tiny']
        else:
            benchmark['collections'] = ['EECBS', 'CBS']
    elif dataset == 'cities_tiles':
        benchmark['mode'] = mode or 'single'
        benchmark['map_name'] = 'Berlin_1_256_00' if benchmark['mode'] == 'single' else None
    else:
        benchmark['map_seeds'] = [3, 0]
        benchmark['grid']['max_episode_steps'] = 3
        if mode == 'nothing':
            benchmark['grid']['on_target'] = 'nothing'
        elif mode == 'ranges':
            benchmark['layout'].update(width_max=12, height_max=12)
            if dataset == 'delivery':
                benchmark['layout'].update(block_width_max=3, block_height_max=3, aisle_width_max=2)
            else:
                benchmark['layout'].update(bin_spacing_max=3, pickup_count_min=3, pickup_count_max=5,
                                           delivery_count_min=3, delivery_count_max=5)
    return benchmark


@pytest.mark.skipif(not BASELINE_NEURAL.is_dir() or not BASELINE_TOOLBOX.is_dir(),
                    reason='Pinned read-only parity checkouts are unavailable')
@pytest.mark.parametrize('dataset, mode', [('movingai', 'replay'), ('movingai', 'uniform'),
    ('qd_mapper', 'replay'), ('qd_mapper', 'uniform'), ('cities_tiles', 'single'),
    ('cities_tiles', 'random'), ('delivery', None), ('sortation', None),
    ('delivery', 'nothing'), ('sortation', 'nothing'), ('delivery', 'ranges'), ('sortation', 'ranges')])
def test_baseline_suite_bytes_order_and_metadata(tmp_path, fixed_sources, dataset, mode):
    benchmark = configured(dataset, mode)
    if dataset in fixed_sources:
        benchmark['manifest_path'] = dataset + '.yaml'
    before = deepcopy(benchmark)
    result = generate_suite(benchmark, output_dir=tmp_path / 'new', artifact_root=tmp_path)
    assert benchmark == before
    assert Path(result['suite_path']) == tmp_path / 'new/suite.json'
    suite = json.loads(Path(result['suite_path']).read_text())
    assert result['groups'] == len(suite['groups'])
    assert result['invalid_groups']
    assert suite['provenance']['toolbox_version']
    assert suite['provenance']['toolbox_revision'] != 'e909376cb585ec8293cd67a4a38d7203c3e829ff'
    baseline_config = dict(benchmark, suite_path=str(tmp_path / 'old/suite.json'))
    config_file = tmp_path / 'baseline.json'
    config_file.write_text(json.dumps(baseline_config))
    script = '''
import json
from pathlib import Path
import sys
from omegaconf import OmegaConf
from neural_mapf.benchmarks.suites import generate_benchmark
import pogema_toolbox, neural_mapf
assert Path('/tmp/neural-mapf-toolbox-benchmarks').resolve() in Path(pogema_toolbox.__file__).resolve().parents
assert Path('/tmp/neural-mapf-organized/src').resolve() in Path(neural_mapf.__file__).resolve().parents
benchmark = json.loads(Path(sys.argv[1]).read_text())
config = OmegaConf.create({'benchmark': benchmark, 'runtime': {'artifact_root': sys.argv[2]}})
print(json.dumps(generate_benchmark(config)))
'''
    completed = subprocess.run([sys.executable, '-c', script, str(config_file), str(tmp_path)], cwd=tmp_path,
        env=dict(os.environ, PYTHONPATH=os.pathsep.join([str(BASELINE_NEURAL / 'src'), str(BASELINE_TOOLBOX)])),
        check=True, capture_output=True, text=True)
    old_result = json.loads(completed.stdout)
    old_suite = json.loads(Path(old_result['suite_path']).read_text())
    assert suite['dataset'] == old_suite['dataset']
    assert suite['groups'] == old_suite['groups']
    for group in suite['groups']:
        if group['status'] == 'ready':
            old_bytes = (Path(old_result['suite_path']).parent / group['instances_path']).read_bytes()
            new_bytes = (Path(result['suite_path']).parent / group['instances_path']).read_bytes()
            assert new_bytes == old_bytes
            instances = load_instances(Path(result['suite_path']).parent / group['instances_path'])
            if dataset in ('delivery', 'sortation') and mode == 'nothing':
                assert all(i.goal_sequences is None and i.grid_config['on_target'] == 'nothing' for i in instances)
            elif dataset in ('delivery', 'sortation'):
                assert all(i.grid_config['on_target'] == 'restart' and i.goal_sequences for i in instances)
                assert all(len(s) == 4 for i in instances for s in i.goal_sequences)
    for data in [suite, old_suite]:
        data['generation'].pop('suite_path', None)
        data['provenance'].pop('generator', None)
        data['provenance'].pop('toolbox_revision', None)
        data['provenance'].pop('toolbox_version', None)
    assert suite == old_suite


def test_cities_official_order_and_constants(tmp_path):
    benchmark = benchmark_defaults('cities_tiles')
    result = generate_suite(benchmark, output_dir=tmp_path)
    suite = json.loads(Path(result['suite_path']).read_text())
    cities = ['Berlin_1_256', 'Boston_0_256', 'London_2_256', 'Milan_0_256',
              'Moscow_0_256', 'NewYork_1_256', 'Paris_1_256', 'Paris_2_256']
    expected = [(f'{city}_{tile:02d}', count) for city in cities for tile in range(16)
                for count in (64, 128, 192, 256)]
    assert [(g['metadata']['map_name'], g['metadata']['num_agents']) for g in suite['groups']] == expected
    assert result['episodes'] == result['ready_groups'] == 512
    for position in (0, 3, 255, 511):
        instance = load_instances(tmp_path / suite['groups'][position]['instances_path'])[0]
        assert {k: instance.grid_config[k] for k in ('seed', 'obs_radius', 'max_episode_steps',
            'observation_type', 'collision_system', 'on_target')} == dict(seed=0, obs_radius=5,
            max_episode_steps=256, observation_type='MAPF', collision_system='soft', on_target='nothing')
    assert suite['provenance']['asset_sha256'] == '3d357b87c64bab6a08a8ec43cfe81295ed37fd4a14db36d563e4ae599785664f'


def test_resolved_custom_manifest_and_explicit_index(tmp_path, fixed_sources):
    benchmark = configured('movingai')
    benchmark['manifest'] = fixed_sources['movingai']
    benchmark['suite_path'] = 'explicit/index.json'
    before = deepcopy(benchmark)
    result = generate_suite(benchmark, output_dir=tmp_path / 'unused', artifact_root=tmp_path)
    assert Path(result['suite_path']) == tmp_path / 'explicit/index.json'
    assert result['ready_groups'] == 1
    assert benchmark == before
    assert not (tmp_path / 'unused').exists()


@pytest.mark.parametrize('dataset', ['unknown', 'random'])
def test_unsupported_dataset_fails_before_output(tmp_path, dataset):
    with pytest.raises(ValueError, match='Unsupported dataset'):
        generate_suite({'dataset': dataset}, output_dir=tmp_path / 'unused')
    assert not (tmp_path / 'unused').exists()


@pytest.mark.parametrize('field, value', [('map_seeds', [-1]), ('map_seeds', [0, 0]),
    ('agent_counts', [0]), ('agent_counts', [2, 2]), ('episodes', 0), ('generation_seed', -1)])
def test_procedural_invalid_requests_fail_before_output(tmp_path, field, value):
    benchmark = configured('delivery')
    benchmark[field] = value
    with pytest.raises(ValueError):
        generate_suite(benchmark, output_dir=tmp_path / 'unused')
    assert not (tmp_path / 'unused').exists()


def test_missing_fixed_sources_are_visible_and_actionable(tmp_path, fixed_sources):
    benchmark = configured('movingai')
    benchmark['manifest'] = fixed_sources['movingai']
    (tmp_path / 'maps/movingai/data/tiny.map').unlink()
    result = generate_suite(benchmark, output_dir=tmp_path / 'new', artifact_root=tmp_path)
    assert result['ready_groups'] == 0
    assert len(result['invalid_groups']) == 2
    assert all('pogema-toolbox datasets prepare' in g['error'] for g in result['invalid_groups'])


def test_simple_generation_materializes_maps_and_preserves_registry(tmp_path):
    benchmark = dict(grid=dict(size=8, density=0, num_agents=2, max_episode_steps=7,
                                on_target='nothing'), episodes=2, generation_seed=11)
    before = deepcopy(benchmark)
    first = generate_instances(benchmark)
    assert [i.grid_config['seed'] for i in first] == [11, 12]
    assert [i.id for i in generate_instances(benchmark)] == [i.id for i in first]
    assert benchmark == before
    previous = ToolboxRegistry._maps
    try:
        ToolboxRegistry._maps = {'miniature': '...\n...'}
        benchmark['grid'] = dict(map_name='miniature', num_agents=1, on_target='finish')
        instances = generate_instances(benchmark)
        assert instances[0].obstacles == ((0, 0, 0), (0, 0, 0))
        assert instances[0].grid_config['on_target'] == 'finish'
        benchmark['grid']['map_name'] = 'absent'
        with pytest.raises(ValueError, match='Unknown toolbox map'):
            generate_instances(benchmark)
    finally:
        ToolboxRegistry._maps = previous


def test_source_revision_belongs_to_this_checkout(tmp_path):
    if not (ROOT / '.git').exists():
        pytest.skip('Source revision check requires this checkout')
    expected = subprocess.run(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], check=True,
                              capture_output=True, text=True).stdout.strip()
    benchmark = configured('delivery')
    result = generate_suite(benchmark, output_dir=tmp_path)
    provenance = json.loads(Path(result['suite_path']).read_text())['provenance']
    assert provenance['toolbox_revision'] == expected


@pytest.mark.skipif(not BASELINE_NEURAL.is_dir() or not BASELINE_TOOLBOX.is_dir(),
                    reason='Pinned read-only parity checkouts are unavailable')
@pytest.mark.parametrize('protocol', ['nothing', 'restart'])
def test_simple_generation_baseline_payload_parity(tmp_path, protocol):
    benchmark = dict(grid=dict(size=8, density=0, num_agents=2, max_episode_steps=7,
                               on_target=protocol, observation_type='POMAPF', collision_system='priority'),
                     episodes=2, generation_seed=11, instances_path=None)
    if protocol == 'restart':
        benchmark['grid'].update(map=[[0, 0, 0], [0, 0, 0]], agents_xy=[[0, 0], [1, 0]],
            targets_xy=[[[0, 2], [1, 2], [0, 1]], [[1, 2], [0, 2], [1, 1]]])
    instances = generate_instances(benchmark)
    config_file = tmp_path / 'benchmark.json'
    config_file.write_text(json.dumps(benchmark))
    script = '''
import json
from pathlib import Path
import sys
from omegaconf import OmegaConf
from neural_mapf.benchmarks.instances import generate_instances, save_instances
import pogema_toolbox, neural_mapf
assert Path('/tmp/neural-mapf-toolbox-benchmarks').resolve() in Path(pogema_toolbox.__file__).resolve().parents
assert Path('/tmp/neural-mapf-organized/src').resolve() in Path(neural_mapf.__file__).resolve().parents
benchmark = json.loads(Path(sys.argv[1]).read_text())
config = OmegaConf.create({'benchmark': benchmark, 'runtime': {'artifact_root': sys.argv[2]}})
save_instances(generate_instances(config), Path(sys.argv[2]) / 'old.jsonl')
'''
    subprocess.run([sys.executable, '-c', script, str(config_file), str(tmp_path)], cwd=tmp_path,
        env=dict(os.environ, PYTHONPATH=os.pathsep.join([str(BASELINE_NEURAL / 'src'), str(BASELINE_TOOLBOX)])),
        check=True, capture_output=True, text=True)
    from pogema_toolbox import save_instances
    save_instances(instances, tmp_path / 'new.jsonl')
    assert (tmp_path / 'new.jsonl').read_bytes() == (tmp_path / 'old.jsonl').read_bytes()
