"""Generate persistent dataset groups from plain benchmark mappings.

Ported from Neural-MAPF; see licenses/Neural-MAPF-LICENSE for its MIT notice.
"""
import hashlib
import json
from dataclasses import fields
from math import isfinite
from numbers import Integral, Real
from pathlib import Path
import re
from copy import deepcopy
from importlib.metadata import PackageNotFoundError, distribution
import subprocess
from types import SimpleNamespace

from . import __version__
from .datasets import _artifact_path, data_directory, dataset_manifest, selected_path, sha256_file
from .instances import Instance, save_instances


def _toolbox_revision():
    """Report installed VCS provenance or the Git revision owning this source."""
    try:
        installed = distribution('pogema-toolbox')
        direct_url = (installed.read_text('direct_url.json')
                      if Path(installed.locate_file('pogema_toolbox/benchmarks.py')).resolve()
                      == Path(__file__).resolve() else None)
    except PackageNotFoundError:
        direct_url = None
    if direct_url:
        commit = json.loads(direct_url).get('vcs_info', {}).get('commit_id')
        if commit:
            return commit
    root = Path(__file__).resolve().parents[1]
    if not (root / '.git').exists():
        return None
    try:
        repository = subprocess.run(['git', '-C', str(root), 'rev-parse', '--show-toplevel'],
                                    check=True, capture_output=True, text=True)
        if Path(repository.stdout.strip()).resolve() != root:
            return None
        revision = subprocess.run(['git', '-C', str(root), 'rev-parse', 'HEAD'],
                                  check=True, capture_output=True, text=True)
        return revision.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def generate_instances(benchmark):
    """Materialize seeded random/registry maps as immutable explicit instances."""
    from pogema import GridConfig, pogema_v0
    from .registry import ToolboxRegistry
    benchmark = deepcopy(dict(benchmark))
    result = []
    for episode in range(benchmark['episodes']):
        parameters = deepcopy(benchmark['grid'])
        map_name = parameters.pop('map_name', None)
        if map_name:
            maps = ToolboxRegistry.get_maps()
            if map_name not in maps:
                raise ValueError(f'Unknown toolbox map: {map_name}')
            parameters['map'] = maps[map_name]
        parameters['seed'] = benchmark['generation_seed'] + episode
        env = pogema_v0(GridConfig(**parameters))
        try:
            env.reset()
            base = env.unwrapped
            sequences = (base.get_lifelong_targets_xy(ignore_borders=True)
                         if base.grid_config.on_target == 'restart' else None)
            grid = base.grid_config.model_dump()
            for key in ('map', 'map_name', 'agents_xy', 'targets_xy'):
                grid.pop(key, None)
            result.append(Instance(base.get_obstacles(ignore_borders=True).astype(int).tolist(),
                base.get_agents_xy(ignore_borders=True), base.get_targets_xy(ignore_borders=True), grid, sequences))
        finally:
            env.close()
    return result


def _positive_integers(values, name):
    if not values or any(isinstance(v, bool) or not isinstance(v, Integral) or v < 1 for v in values):
        raise ValueError(f'{name} must contain positive integers')
    if len(set(values)) != len(values):
        raise ValueError(f'{name} must not contain duplicates')
    return list(values)


def _name(value):
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9_-]+', value):
        raise ValueError(f'Unsafe group/map name: {value!r}')
    return value


def _from_grid(grid):
    return Instance.from_grid_config(grid)


def _verified_source(data, relative, members, dataset='movingai'):
    instruction = f'Run pogema-toolbox datasets prepare {dataset} before generating'
    if relative not in members:
        raise ValueError(f'Unlisted {dataset} source {relative}. {instruction}')
    try:
        content = selected_path(data, relative).read_bytes()
    except (OSError, ValueError) as exc:
        raise ValueError(f'Missing or unsafe {dataset} source {relative}. {instruction}') from exc
    digest = hashlib.sha256(content).hexdigest()
    if digest != members[relative]:
        raise ValueError(f'{dataset} source SHA256 mismatch for {relative}. {instruction}')
    return content.decode('utf-8'), digest


def _save_group(index, group, instances):
    relative = f'{group["name"]}/instances.jsonl'
    path = selected_path(index.parent, relative)
    save_instances(instances, path)
    group.update(status='ready', instances_path=relative, sha256=sha256_file(path), episode_count=len(instances))


def _save_suite(benchmark, manifest, index, groups, provenance):
    suite = {'schema_version': 1, 'dataset': manifest['dataset'],
             'generation': benchmark,
             'provenance': dict(attribution=manifest.get('attribution'), rights=manifest.get('rights'),
                                toolbox_version=__version__, toolbox_revision=_toolbox_revision(), **provenance),
             'groups': groups}
    index.write_text(json.dumps(suite, indent=2, allow_nan=False) + '\n')
    return {'suite_path': str(index), 'groups': len(groups),
            'ready_groups': sum(g['status'] == 'ready' for g in groups),
            'invalid_groups': [g for g in groups if g['status'] == 'invalid'],
            'episodes': sum(g['episode_count'] for g in groups)}


def _generate_cities(benchmark, manifest, index):
    from pogema import pogema_v0
    from pogema_toolbox.generators.cities_generator import CitiesTilesGenerator, ASSET_SHA256
    from pogema_toolbox.generators.scenario_generator import ScenarioSampler
    # Toolbox exposes selections only after eager capacity validation. Defer that
    # check to each record so saved suites retain invalid requests without
    # duplicating its selection/RNG logic or changing the public toolbox API.
    class SuiteCitiesGenerator(CitiesTilesGenerator):
        def _validate_capacities(self, maps, map_names):
            if self.mode == 'official':
                super()._validate_capacities(maps, map_names)

        def _instance(self, maps, map_name, num_agents, scenario_seed):
            try:
                return super()._instance(maps, map_name, num_agents, scenario_seed)
            except ValueError as exc:
                city, tile = map_name.rsplit('_', 1)
                return SimpleNamespace(map_name=map_name, city=city, tile_index=int(tile),
                    num_agents=num_agents, scenario_seed=scenario_seed, error=exc)

    if manifest['asset_sha256'] != ASSET_SHA256:
        raise ValueError('Cities asset SHA256 does not match catalog')
    generator = SuiteCitiesGenerator(mode=benchmark['mode'], map_name=benchmark['map_name'],
        num_agents=_positive_integers(benchmark['agent_counts'], 'agent_counts'),
        num_samples=None if benchmark['mode'] == 'official' else benchmark['episodes'], seed=benchmark['generation_seed'])
    selections = generator.generate()
    index.parent.mkdir(parents=True, exist_ok=True)
    groups = []
    for position, record in enumerate(selections):
        metadata = {'map_name': record.map_name, 'city': record.city, 'tile_index': record.tile_index,
                    'num_agents': record.num_agents, 'generation_seeds': [record.scenario_seed],
                    'sampling': 'official' if benchmark['mode'] == 'official' else 'uniform',
                    'scenario_origin': 'released_configuration' if benchmark['mode'] == 'official' else 'new_scenarios'}
        name = f'{record.map_name}-{record.num_agents}'
        if benchmark['mode'] != 'official':
            name += f'-{position}'
        group = {'name': _name(name), 'metadata': metadata, 'requested_episode_count': 1}
        try:
            if getattr(record, 'error', None) is not None:
                raise record.error
            env = pogema_v0(record.grid_config)
            try:
                env.reset()
                base = env.unwrapped
                obstacles = base.get_obstacles(ignore_borders=True).astype(int)
                parameters = base.grid_config.model_dump()
                for key in ('map', 'map_name', 'agents_xy', 'targets_xy'):
                    parameters.pop(key, None)
                if benchmark['mode'] == 'official':
                    instance = Instance(obstacles.tolist(), base.get_agents_xy(ignore_borders=True),
                                        base.get_targets_xy(ignore_borders=True), parameters)
                else:
                    parameters = {key: parameters[key] for key in ('obs_radius', 'max_episode_steps',
                        'observation_type', 'collision_system', 'on_target')}
                    instance = _from_grid(ScenarioSampler(obstacles).generate(record.num_agents, record.scenario_seed, **parameters))
            finally:
                env.close()
            metadata.update(binary_grid_sha256=hashlib.sha256(obstacles.astype('int8').tobytes()).hexdigest(),
                            dimensions=list(obstacles.shape), asset_sha256=ASSET_SHA256,
                            coordinate_transform='packaged tile (row, column); no cropping')
            _save_group(index, group, [instance])
        except (OSError, ValueError) as exc:
            group.update(status='invalid', episode_count=0, error=f'{type(exc).__name__}: {exc}')
        groups.append(group)
    return _save_suite(benchmark, manifest, index, groups, {'asset_sha256': ASSET_SHA256,
        'source_url': manifest['source_url'], 'source_revision': manifest['source_revision'],
        'generator': _generate_cities.__module__ + '.' + _generate_cities.__name__})


def _generate_qd(benchmark, manifest, index, artifact_root):
    from pogema_toolbox.generators.qd_mapper import load_qd_map
    from pogema_toolbox.generators.scenario_generator import ScenarioSampler
    from .datasets import selected_archives
    archives = selected_archives(manifest, benchmark.get('collections'))
    members = {relative: digest for archive in archives for relative, digest in archive['members'].items()}
    inventory = [relative for archive in archives for relative in sorted(archive['members']) if relative.endswith('.json')]
    named = {_name(Path(relative).parent.name + '-' + Path(relative).stem): relative for relative in inventory}
    names = list(benchmark['map_names']) if benchmark['map_names'] is not None else list(named)
    if not names or len(set(names)) != len(names) or any(name not in named for name in names):
        raise ValueError('map_names must be nonempty, unique known collection-map names')
    counts = _positive_integers(benchmark['agent_counts'], 'agent_counts')
    numbers = _positive_integers(benchmark['scenario_numbers'], 'scenario_numbers')
    if benchmark['sampling'] not in ('replay', 'uniform'):
        raise ValueError('sampling must be replay or uniform')
    seed = benchmark['generation_seed']
    if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
        raise ValueError('generation_seed must be a nonnegative integer')
    total = len(numbers) if benchmark['sampling'] == 'replay' else _positive_integers([benchmark['episodes']], 'episodes')[0]
    index.parent.mkdir(parents=True, exist_ok=True)
    options = benchmark['grid']
    data = data_directory(benchmark['dataset'], artifact_root=artifact_root)
    groups = []
    for position, name in enumerate(names):
        relative = named[name]
        try:
            _, source_hash = _verified_source(data, relative, members, 'qd_mapper')
            source = load_qd_map(selected_path(data, relative))
            sampler = ScenarioSampler(source.obstacles) if benchmark['sampling'] == 'uniform' else None
            error = None
        except (OSError, ValueError) as exc:
            error = f'{type(exc).__name__}: {exc}'
        for count in counts:
            seeds = [int(seed) + position * total + i for i in range(total)]
            metadata = {'map_name': name, 'collection': Path(relative).parent.name, 'source_path': relative,
                        'num_agents': count, 'sampling': benchmark['sampling'], 'generation_seeds': seeds,
                        'scenario_origin': 'stored_pairs' if benchmark['sampling'] == 'replay' else 'new_scenarios',
                        'scenario_numbers': numbers if benchmark['sampling'] == 'replay' else []}
            group = {'name': f'{name}-{count}', 'metadata': metadata, 'requested_episode_count': total}
            try:
                if error:
                    raise ValueError(error)
                metadata.update(source_name=source.name, source_sha256=source_hash,
                    binary_grid_sha256=hashlib.sha256(source.obstacles.tobytes()).hexdigest(),
                    dimensions=list(source.obstacles.shape),
                    coordinate_transform='source linear index -> (row=index // n_col, column=index % n_col); no cropping')
                instances = [_from_grid(source.grid_config(numbers[i] - 1, count, seed=episode_seed, **options)
                             if benchmark['sampling'] == 'replay' else sampler.generate(count, episode_seed, **options))
                             for i, episode_seed in enumerate(seeds)]
                _save_group(index, group, instances)
            except (OSError, ValueError) as exc:
                group.update(status='invalid', episode_count=0, error=f'{type(exc).__name__}: {exc}')
            groups.append(group)
    receipt = selected_path(data, 'receipt.json')
    return _save_suite(benchmark, manifest, index, groups,
                       {'receipt': json.loads(receipt.read_text()) if receipt.is_file() else None,
                        'generator': _generate_qd.__module__ + '.' + _generate_qd.__name__})


def _validate_procedural_ranges(settings):
    from pogema_toolbox.generators.scenario_generator import _integer
    for field in fields(settings):
        if not field.name.endswith('_min'):
            continue
        name = field.name[:-4]
        lower, upper = getattr(settings, name + '_min'), getattr(settings, name + '_max')
        if lower is None and upper is None and field.default is None:
            continue
        if lower is None or upper is None:
            raise ValueError(f'{name}_min and {name}_max must be supplied together')
        if name in ('obstacle_density', 'go_straight'):
            if any(isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value)
                   or not 0 <= value <= 1 for value in (lower, upper)):
                raise ValueError(f'{name} ranges must contain finite numbers in [0, 1]')
            if name == 'go_straight' and lower == 1:
                raise ValueError('go_straight_min must be smaller than 1')
        else:
            minimum = 0 if name in ('wall_components', 'side_pad', 'horizontal_gap', 'vertical_gap',
                                    'num_wall_rows', 'num_wall_cols') else 1
            lower = _integer(lower, name + '_min', minimum)
            upper = _integer(upper, name + '_max', minimum)
        if lower > upper:
            raise ValueError(f'{name}_min must not exceed {name}_max')
    for name in ('side_pad', 'horizontal_gap', 'vertical_gap'):
        if hasattr(settings, name):
            _integer(getattr(settings, name), name)
    for name in ('uniform', 'only_centre_obstacles', 'wfi_instance', 'block_extra_space'):
        if hasattr(settings, name) and not isinstance(getattr(settings, name), bool):
            raise ValueError(f'{name} must be a boolean')
    if hasattr(settings, 'num_wall_rows_min'):
        ranges = [settings.num_wall_rows_min, settings.num_wall_rows_max,
                  settings.num_wall_cols_min, settings.num_wall_cols_max]
        if any(value is None for value in ranges) and not all(value is None for value in ranges):
            raise ValueError('num_wall_rows and num_wall_cols ranges must be supplied together')


def _generate_procedural(benchmark, index, validator=None, max_attempts=100):
    import numpy as np
    from pogema_toolbox.generators.delivery_generator import DeliveryRangeSettings, generate_delivery
    from pogema_toolbox.generators.maze_generator import MazeRangeSettings, MazeGenerator
    from pogema_toolbox.generators.random_generator import MapRangeSettings, generate_map
    from pogema_toolbox.generators.room_generator import RoomRangeSettings, generate_room
    from pogema_toolbox.generators.sortation_generator import SortationRangeSettings, generate_sortation
    from pogema_toolbox.generators.warehouse_generator import WarehouseRangeSettings, generate_warehouse
    from pogema_toolbox.generators.scenario_generator import ScenarioSampler, _grid_config
    settings_type, generator = {'delivery': (DeliveryRangeSettings, generate_delivery),
                               'sortation': (SortationRangeSettings, generate_sortation),
                               'maze': (MazeRangeSettings, MazeGenerator.generate_maze),
                               'warehouse': (WarehouseRangeSettings, generate_warehouse),
                               'room': (RoomRangeSettings, generate_room),
                               'random': (MapRangeSettings, generate_map)}[benchmark['dataset']]
    task_layout = benchmark['dataset'] in ('delivery', 'sortation')
    if not task_layout:
        for name in ('grid', 'layout'):
            if not isinstance(benchmark[name], dict):
                raise ValueError(f'{name} must be a mapping')
        for name in ('map_seeds', 'agent_counts'):
            if not isinstance(benchmark[name], (list, tuple)):
                raise ValueError(f'{name} must be a list')
    map_seeds = list(benchmark['map_seeds'])
    if (not map_seeds or any(isinstance(s, bool) or not isinstance(s, Integral) or s < 0 for s in map_seeds)
            or len(set(map_seeds)) != len(map_seeds)):
        raise ValueError('map_seeds must contain unique nonnegative integers')
    counts = _positive_integers(benchmark['agent_counts'], 'agent_counts')
    total = _positive_integers([benchmark['episodes']], 'episodes')[0]
    seed = benchmark['generation_seed']
    if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
        raise ValueError('generation_seed must be a nonnegative integer')
    if benchmark['sampling'] not in (('uniform', 'endpoints') if task_layout else ('uniform',)):
        raise ValueError('sampling must be uniform or endpoints' if task_layout else 'sampling must be uniform')
    options = benchmark['grid']
    on_target = options['on_target'] if task_layout else options.get('on_target', 'nothing')
    if task_layout and on_target not in ('nothing', 'restart'):
        raise ValueError('procedural protocols must be nothing or restart')
    if not task_layout and on_target not in ('nothing', 'finish', 'restart'):
        raise ValueError('on_target must be nothing, finish or restart')
    unknown = set(benchmark['layout']) - {field.name for field in fields(settings_type)}
    if unknown:
        raise ValueError('Unknown layout fields: ' + ', '.join(str(key) for key in unknown))
    settings = settings_type(**benchmark['layout'])
    if not task_layout or validator is not None:
        _validate_procedural_ranges(settings)
    if validator is not None:
        targets = [[(1, 1), (0, 1)]] if on_target == 'restart' else [(1, 1)]
        _grid_config(np.zeros((2, 2), dtype=np.int8), [(0, 0)], targets, **options)

    def create_layout(map_seed):
        parameters = settings.sample(int(map_seed))
        layout = generator(parameters) if benchmark['dataset'] == 'random' else generator(**parameters)
        if task_layout:
            obstacles = layout.obstacles
            sampler = ScenarioSampler(obstacles, layout.pickup_cells, layout.delivery_cells)
        else:
            obstacles = (MazeGenerator.string_to_array(layout) if isinstance(layout, str) else layout)
            sampler = ScenarioSampler(obstacles)
            obstacles = sampler.obstacles
            parameters = {key: value.item() if isinstance(value, np.generic) else value
                          for key, value in parameters.items()}
        metadata = {'layout_parameters': parameters, 'map_seed': int(map_seed),
            'binary_grid_sha256': hashlib.sha256(obstacles.tobytes()).hexdigest(),
            'dimensions': list(obstacles.shape),
            'coordinate_transform': 'generated (row, column); no cropping'}
        if task_layout:
            metadata.update(pickup_cells=layout.pickup_cells, delivery_cells=layout.delivery_cells)
        return sampler, metadata

    index.parent.mkdir(parents=True, exist_ok=True)
    groups = []
    retry_stride = len(map_seeds) * len(counts) * total
    retry_map_seed = max(map_seeds) + 1
    retry_scenario_seed = int(seed) + len(map_seeds) * total
    for position, map_seed in enumerate(map_seeds):
        name = f"{benchmark['dataset']}-{map_seed}"
        try:
            sampler, map_metadata = create_layout(map_seed)
            error = None
        except ValueError as exc:
            error = f'{type(exc).__name__}: {exc}'
        for count_position, count in enumerate(counts):
            seeds = [int(seed) + position * total + i for i in range(total)]
            metadata = {'map_name': name, 'map_seed': int(map_seed), 'num_agents': count,
                        'sampling': benchmark['sampling'], 'on_target': on_target,
                        'generation_seeds': seeds, 'scenario_origin': 'new_scenarios'}
            group = {'name': f'{name}-{count}', 'metadata': metadata, 'requested_episode_count': total}
            try:
                if validator is None:
                    if error:
                        raise ValueError(error)
                    metadata.update(map_metadata)
                    instances = [_from_grid(sampler.generate(count, episode_seed,
                        sampling=benchmark['sampling'], **options)) for episode_seed in seeds]
                else:
                    metadata['requested_map_seed'] = metadata.pop('map_seed')
                    metadata['requested_generation_seeds'] = seeds
                    metadata['generation_seeds'] = []
                    metadata['scenario_origin'] = 'validated_new_maps'
                    metadata['instances'] = []
                    metadata['validation'] = {'regenerate_layout': True, 'max_attempts': max_attempts, 'attempts': []}
                    instances = []
                    for episode, initial_seed in enumerate(seeds):
                        slot = (position * len(counts) + count_position) * total + episode
                        for attempt in range(max_attempts):
                            offset = slot + (attempt - 1) * retry_stride
                            actual_map_seed = int(map_seed) if attempt == 0 else retry_map_seed + offset
                            actual_seed = initial_seed if attempt == 0 else retry_scenario_seed + offset
                            record = {'episode_index': episode, 'attempt': attempt + 1,
                                      'map_seed': actual_map_seed, 'generation_seed': actual_seed,
                                      'instance_id': None, 'accepted': False}
                            try:
                                if attempt == 0:
                                    if error:
                                        raise ValueError(error)
                                    current_sampler, current_map = sampler, map_metadata
                                else:
                                    current_sampler, current_map = create_layout(actual_map_seed)
                                instance = _from_grid(current_sampler.generate(count, actual_seed,
                                    sampling=benchmark['sampling'], **options))
                            except ValueError as exc:
                                record.update(status='generation_error', error=f'{type(exc).__name__}: {exc}')
                            else:
                                try:
                                    report = validator(instance)
                                except Exception as exc:
                                    raise RuntimeError(f'Instance validator failed: {exc}') from exc
                                if (not isinstance(report, dict) or not isinstance(report.get('accepted'), bool)
                                        or not isinstance(report.get('status'), str) or not report['status']):
                                    raise TypeError('validator must return a mapping with boolean accepted and nonempty status')
                                record = {**report, **{key: value for key, value in record.items() if key != 'accepted'},
                                          'instance_id': instance.id}
                            metadata['validation']['attempts'].append(record)
                            if record['accepted']:
                                instances.append(instance)
                                metadata['generation_seeds'].append(actual_seed)
                                metadata['instances'].append(dict(current_map, instance_id=instance.id,
                                                                  generation_seed=actual_seed))
                                break
                        else:
                            raise ValueError(f'Episode {episode} exhausted {max_attempts} validation attempts')
                _save_group(index, group, instances)
            except (OSError, ValueError) as exc:
                group.update(status='invalid', episode_count=0, error=f'{type(exc).__name__}: {exc}')
            groups.append(group)
    return _save_suite(benchmark, {'dataset': benchmark['dataset']}, index, groups,
        {'generator': generator.__module__ + '.' + generator.__qualname__})


def generate_suite(benchmark, *, output_dir, artifact_root='.', validator=None, max_attempts=100) -> dict:
    """Persist seeded dataset groups without downloading or constructing a policy."""
    from pogema_toolbox.moving_ai_ingestion import (
        parse_movingai_map, parse_movingai_scenarios, movingai_grid_config,
    )
    from pogema_toolbox.generators.scenario_generator import ScenarioSampler
    benchmark = deepcopy(dict(benchmark))
    dataset = benchmark.get('dataset')
    if dataset not in ('movingai', 'cities_tiles', 'qd_mapper', 'delivery', 'sortation',
                        'maze', 'warehouse', 'room', 'random'):
        raise ValueError('Unsupported dataset')
    if validator is not None:
        if not callable(validator):
            raise TypeError('validator must be callable')
        max_attempts = _positive_integers([max_attempts], 'max_attempts')[0]
        if dataset not in ('delivery', 'sortation', 'maze', 'warehouse', 'room', 'random'):
            raise ValueError('Validated map regeneration requires a procedural dataset')
    index = (_artifact_path(artifact_root, benchmark['suite_path']) if benchmark.get('suite_path')
             else Path(output_dir).expanduser().resolve() / 'suite.json')
    if dataset in ('delivery', 'sortation', 'maze', 'warehouse', 'room', 'random'):
        return _generate_procedural(benchmark, index, validator, max_attempts)
    manifest = dataset_manifest(dataset, artifact_root=artifact_root,
                                manifest_path=benchmark.get('manifest_path'),
                                manifest=benchmark.get('manifest'))
    if dataset == 'cities_tiles':
        return _generate_cities(benchmark, manifest, index)
    if dataset == 'qd_mapper':
        return _generate_qd(benchmark, manifest, index, artifact_root)
    members = {relative: digest for archive in manifest['archives']
               for relative, digest in archive['members'].items()}
    names = list(benchmark['map_names']) if benchmark['map_names'] is not None else manifest['maps']
    if not names or len(set(names)) != len(names):
        raise ValueError('map_names must be nonempty and unique')
    names = [_name(name) for name in names]
    counts = _positive_integers(benchmark['agent_counts'], 'agent_counts')
    numbers = _positive_integers(benchmark['scenario_numbers'], 'scenario_numbers')
    if any(n > 25 for n in numbers) or benchmark['scenario_family'] not in ('random', 'even'):
        raise ValueError('MovingAI scenarios require random/even and numbers 1..25')
    if benchmark['sampling'] not in ('replay', 'uniform'):
        raise ValueError('sampling must be replay or uniform')
    seed = benchmark['generation_seed']
    if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
        raise ValueError('generation_seed must be a nonnegative integer')
    total = len(numbers) if benchmark['sampling'] == 'replay' else _positive_integers([benchmark['episodes']], 'episodes')[0]
    options = benchmark['grid']
    index.parent.mkdir(parents=True, exist_ok=True)
    data = data_directory(benchmark['dataset'], artifact_root=artifact_root)
    groups = []
    for map_index, name in enumerate(names):
        try:
            text, map_hash = _verified_source(data, name + '.map', members)
            obstacles = parse_movingai_map(text)
            problems, scenario_hashes = [], []
            if benchmark['sampling'] == 'replay':
                for number in numbers:
                    relative = f"scen-{benchmark['scenario_family']}/{name}-{benchmark['scenario_family']}-{number}.scen"
                    text, scenario_hash = _verified_source(data, relative, members)
                    problems.append(parse_movingai_scenarios(text, name + '.map', obstacles))
                    scenario_hashes.append({'path': relative, 'sha256': scenario_hash})
            map_metadata = {'map_sha256': map_hash, 'scenarios': scenario_hashes,
                            'binary_grid_sha256': hashlib.sha256(obstacles.tobytes()).hexdigest(),
                            'dimensions': list(obstacles.shape),
                            'coordinate_transform': 'source (x, y) -> instance (row=y, column=x); no cropping'}
            sampler = ScenarioSampler(obstacles) if benchmark['sampling'] == 'uniform' else None
            map_error = None
        except (OSError, ValueError) as exc:
            map_error = f'{type(exc).__name__}: {exc}'
        for count in counts:
            group_name = f'{name}-{count}'
            seeds = [int(seed) + map_index * total + i for i in range(total)]
            metadata = {'map_name': name, 'num_agents': count, 'sampling': benchmark['sampling'],
                        'scenario_family': benchmark['scenario_family'],
                        'scenario_numbers': numbers if benchmark['sampling'] == 'replay' else [],
                        'generation_seeds': seeds}
            group = {'name': group_name, 'metadata': metadata, 'requested_episode_count': total}
            try:
                if map_error:
                    raise ValueError(map_error)
                metadata.update(map_metadata)
                instances = []
                for i, episode_seed in enumerate(seeds):
                    grid = (movingai_grid_config(obstacles, problems[i], count, seed=episode_seed, **options)
                            if benchmark['sampling'] == 'replay' else
                            sampler.generate(count, episode_seed, **options))
                    instances.append(_from_grid(grid))
                relative = f'{group_name}/instances.jsonl'
                path = selected_path(index.parent, relative)
                save_instances(instances, path)
                group.update(status='ready', instances_path=relative, sha256=sha256_file(path), episode_count=len(instances))
            except (OSError, ValueError) as exc:
                group.update(status='invalid', episode_count=0, error=f'{type(exc).__name__}: {exc}')
            groups.append(group)
    receipt_path = selected_path(data, 'receipt.json')
    return _save_suite(benchmark, manifest, index, groups,
                       {'receipt': json.loads(receipt_path.read_text()) if receipt_path.is_file() else None,
                        'generator': generate_suite.__module__ + '.' + generate_suite.__name__})
