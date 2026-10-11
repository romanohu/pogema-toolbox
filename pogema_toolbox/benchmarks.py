"""Generate persistent dataset groups from plain benchmark mappings.

Ported from Neural-MAPF; see licenses/Neural-MAPF-LICENSE for its MIT notice.
"""
import hashlib
import json
from numbers import Integral
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


def _generate_procedural(benchmark, index):
    from pogema_toolbox.generators.delivery_generator import DeliveryRangeSettings, generate_delivery
    from pogema_toolbox.generators.sortation_generator import SortationRangeSettings, generate_sortation
    from pogema_toolbox.generators.scenario_generator import ScenarioSampler
    settings_type, generator = {'delivery': (DeliveryRangeSettings, generate_delivery),
                               'sortation': (SortationRangeSettings, generate_sortation)}[benchmark['dataset']]
    map_seeds = list(benchmark['map_seeds'])
    if (not map_seeds or any(isinstance(s, bool) or not isinstance(s, Integral) or s < 0 for s in map_seeds)
            or len(set(map_seeds)) != len(map_seeds)):
        raise ValueError('map_seeds must contain unique nonnegative integers')
    counts = _positive_integers(benchmark['agent_counts'], 'agent_counts')
    total = _positive_integers([benchmark['episodes']], 'episodes')[0]
    seed = benchmark['generation_seed']
    if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
        raise ValueError('generation_seed must be a nonnegative integer')
    if benchmark['sampling'] not in ('uniform', 'endpoints'):
        raise ValueError('sampling must be uniform or endpoints')
    options = benchmark['grid']
    if options['on_target'] not in ('nothing', 'restart'):
        raise ValueError('procedural protocols must be nothing or restart')
    settings = settings_type(**benchmark['layout'])
    index.parent.mkdir(parents=True, exist_ok=True)
    groups = []
    for position, map_seed in enumerate(map_seeds):
        name = f"{benchmark['dataset']}-{map_seed}"
        try:
            parameters = settings.sample(int(map_seed))
            layout = generator(**parameters)
            sampler = ScenarioSampler(layout.obstacles, layout.pickup_cells, layout.delivery_cells)
            map_metadata = {'layout_parameters': parameters, 'map_seed': layout.map_seed,
                'pickup_cells': layout.pickup_cells, 'delivery_cells': layout.delivery_cells,
                'binary_grid_sha256': hashlib.sha256(layout.obstacles.tobytes()).hexdigest(),
                'dimensions': list(layout.obstacles.shape),
                'coordinate_transform': 'generated (row, column); no cropping'}
            error = None
        except ValueError as exc:
            error = f'{type(exc).__name__}: {exc}'
        for count in counts:
            seeds = [int(seed) + position * total + i for i in range(total)]
            metadata = {'map_name': name, 'map_seed': int(map_seed), 'num_agents': count,
                        'sampling': benchmark['sampling'], 'on_target': options['on_target'],
                        'generation_seeds': seeds, 'scenario_origin': 'new_scenarios'}
            group = {'name': f'{name}-{count}', 'metadata': metadata, 'requested_episode_count': total}
            try:
                if error:
                    raise ValueError(error)
                metadata.update(map_metadata)
                instances = [_from_grid(sampler.generate(count, episode_seed,
                    sampling=benchmark['sampling'], **options)) for episode_seed in seeds]
                _save_group(index, group, instances)
            except (OSError, ValueError) as exc:
                group.update(status='invalid', episode_count=0, error=f'{type(exc).__name__}: {exc}')
            groups.append(group)
    return _save_suite(benchmark, {'dataset': benchmark['dataset']}, index, groups,
        {'generator': generator.__module__ + '.' + generator.__name__})


def generate_suite(benchmark, *, output_dir, artifact_root='.') -> dict:
    """Persist seeded dataset groups without downloading or constructing a policy."""
    from pogema_toolbox.moving_ai_ingestion import (
        parse_movingai_map, parse_movingai_scenarios, movingai_grid_config,
    )
    from pogema_toolbox.generators.scenario_generator import ScenarioSampler
    benchmark = deepcopy(dict(benchmark))
    dataset = benchmark.get('dataset')
    if dataset not in ('movingai', 'cities_tiles', 'qd_mapper', 'delivery', 'sortation'):
        raise ValueError('Unsupported dataset')
    index = (_artifact_path(artifact_root, benchmark['suite_path']) if benchmark.get('suite_path')
             else Path(output_dir).expanduser().resolve() / 'suite.json')
    if dataset in ('delivery', 'sortation'):
        return _generate_procedural(benchmark, index)
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
