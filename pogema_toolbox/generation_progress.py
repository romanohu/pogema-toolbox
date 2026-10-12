"""Atomic checkpoints for deterministic suite generation."""
from copy import deepcopy
import hashlib
from importlib.metadata import version
import json
from pathlib import Path

from . import __version__
from .datasets import data_directory, selected_archives, selected_path, sha256_file
from .instances import Instance, _atomic_text, load_instances


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def generation_identity(benchmark, manifest, artifact_root, validator, max_attempts, resume, caller_provenance=None):
    if caller_provenance is None:
        caller_provenance = {}
    if not isinstance(caller_provenance, dict):
        raise TypeError('provenance must be a mapping')
    provenance = getattr(validator, 'provenance', None) if validator is not None else None
    if validator is not None and resume and (not isinstance(provenance, dict) or not provenance):
        raise ValueError('Resuming validation requires a nonempty stable validator.provenance mapping')
    try:
        provenance = json.loads(json.dumps(provenance, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ValueError('validator.provenance must be a stable JSON mapping') from exc
    root = Path(__file__).parent
    modules = [root / name for name in ('benchmarks.py', 'instances.py', 'datasets.py',
                                       'moving_ai_ingestion.py', 'generation_progress.py')]
    modules.extend(sorted((root / 'generators').glob('*.py')))
    generator = {'toolbox_version': __version__, 'pogema_version': version('pogema'),
                 'numpy_version': version('numpy'),
                 'source': {str(path.relative_to(root)): sha256_file(path) for path in modules}}
    sources = {}
    dataset = benchmark['dataset']
    if dataset == 'cities_tiles':
        sources['cities-tiles.yaml'] = sha256_file(root / 'maps/cities-tiles.yaml')
    elif dataset in ('movingai', 'qd_mapper'):
        data = data_directory(dataset, artifact_root=artifact_root)
        if dataset == 'movingai':
            names = benchmark['map_names'] if benchmark['map_names'] is not None else manifest['maps']
            relatives = [name + '.map' for name in names]
            if benchmark['sampling'] == 'replay':
                family = benchmark['scenario_family']
                relatives.extend(f'scen-{family}/{name}-{family}-{number}.scen'
                                 for name in names for number in benchmark['scenario_numbers'])
        else:
            archives = selected_archives(manifest, benchmark.get('collections'))
            relatives = [relative for archive in archives for relative in archive['members']
                         if relative.endswith('.json') and (benchmark['map_names'] is None or
                         Path(relative).parent.name + '-' + Path(relative).stem in benchmark['map_names'])]
        for relative in relatives + ['receipt.json']:
            path = selected_path(data, relative)
            sources[relative] = sha256_file(path) if path.is_file() else None
    return json.loads(json.dumps({'benchmark': benchmark, 'manifest': manifest, 'generator': generator,
        'sources': sources, 'validator': provenance, 'validation_enabled': validator is not None,
        'max_attempts': max_attempts if validator is not None else None,
        'caller_provenance': caller_provenance}, allow_nan=False))


class GenerationProgress:
    def __init__(self, index, identity, resume):
        self.provenance = identity['caller_provenance']
        self.path = index.with_name(index.name + '.progress.json')
        self._fresh = not resume or not self.path.exists()
        self.data = {'schema_version': 1, 'identity': identity, 'groups': [], 'current': None, 'suite': None}
        if not resume:
            return
        if not self.path.exists():
            if index.exists():
                raise ValueError('Cannot resume an existing suite without generation progress identity')
            return
        try:
            data = json.loads(self.path.read_text())
            checksum = data.pop('sha256')
            if data['schema_version'] != 1 or checksum != _digest(data):
                raise ValueError('Progress content checksum does not match')
            if not isinstance(data['groups'], list):
                raise ValueError('Invalid completed groups')
        except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
            raise ValueError(f'Invalid generation progress: {exc}') from exc
        if _digest(data['identity']) != _digest(identity):
            raise ValueError('Generation resume identity mismatch; benchmark, generator, sources, validator or caller provenance changed')
        try:
            names = set()
            for group in data['groups']:
                if group['name'] in names:
                    raise ValueError('Duplicate completed group')
                names.add(group['name'])
                if group['status'] == 'ready':
                    path = selected_path(index.parent, group['instances_path'])
                    if sha256_file(path) != group['sha256']:
                        raise ValueError('Saved instances checksum does not match')
                    if len(load_instances(path)) != group['episode_count'] or group['episode_count'] != group['requested_episode_count']:
                        raise ValueError('Saved instance count does not match')
                elif group['status'] != 'invalid' or group['episode_count'] != 0:
                    raise ValueError('Invalid completed group status')
                if 'validation_plans_path' in group:
                    path = selected_path(index.parent, group['validation_plans_path'])
                    if sha256_file(path) != group['validation_plans_sha256']:
                        raise ValueError('Saved validation plans checksum does not match')
            if data['current'] is not None:
                current = data['current']
                instances = [Instance.from_dict(item) for item in current['accepted_instances']]
                group = current['group']
                accepted = [record['instance_id'] for record in group['metadata']['validation']['attempts']
                            if record['accepted']]
                if group['name'] in names or accepted != [instance.id for instance in instances]:
                    raise ValueError('Accepted progress instances do not match attempts')
            if data['suite'] is not None:
                if data['current'] is not None or data['suite']['groups'] != data['groups']:
                    raise ValueError('Completed suite does not match progress groups')
                if _digest(json.loads(index.read_text())) != _digest(data['suite']):
                    raise ValueError('Saved suite does not match progress')
        except (OSError, ValueError, TypeError, KeyError) as exc:
            raise ValueError(f'Invalid generation progress or instances: {exc}') from exc
        self.data = data

    @property
    def suite(self):
        return self.data['suite']

    def group(self, name):
        return next((deepcopy(group) for group in self.data['groups'] if group['name'] == name), None)


    def start(self):
        if self._fresh:
            self._save()
            self._fresh = False

    def partial(self, group):
        current = self.data['current']
        if current is None:
            return group, []
        if current['group']['name'] != group['name']:
            raise ValueError('Invalid generation progress group order')
        return deepcopy(current['group']), [Instance.from_dict(item) for item in current['accepted_instances']]

    def attempt(self, group, instances):
        self.data['current'] = {'group': deepcopy(group),
                                'accepted_instances': [instance.to_dict() for instance in instances]}
        self._save()

    def complete(self, group):
        self.data['groups'].append(deepcopy(group))
        self.data['current'] = None
        self._save()

    def finish(self, suite):
        self.data['suite'] = suite
        self._save()

    def _save(self):
        data = dict(self.data, sha256=_digest(self.data))
        with _atomic_text(self.path) as stream:
            stream.write(json.dumps(data, allow_nan=False) + '\n')
