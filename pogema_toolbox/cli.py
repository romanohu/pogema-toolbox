"""Dataset commands using the public toolbox API and plain YAML mappings."""
import argparse
from dataclasses import fields
import json
from pathlib import Path
import sys
import zipfile

import yaml

import pogema_toolbox as toolbox


def _catalogs():
    summaries = []
    for name in ('movingai', 'cities_tiles', 'qd_mapper'):
        manifest = toolbox.dataset_manifest(name)
        summary = dict(name=name, description=manifest['attribution'],
                       source=manifest.get('source_url', manifest['attribution']),
                       rights=manifest['rights'])
        if name == 'qd_mapper':
            summary['collections'] = [archive['name'] for archive in manifest['archives']]
        summaries.append(summary)
    return summaries


def _check_interpolation(value):
    if isinstance(value, str) and '${' in value:
        raise ValueError('Unresolved interpolation: replace ${...} expressions with literal values in the benchmark YAML')
    if isinstance(value, dict):
        for key, item in value.items():
            _check_interpolation(key)
            _check_interpolation(item)
    elif isinstance(value, list):
        for item in value:
            _check_interpolation(item)


def _benchmark(path):
    try:
        benchmark = yaml.safe_load(Path(path).expanduser().read_text())
    except yaml.YAMLError as exc:
        raise ValueError(f'Invalid YAML in {path}: {getattr(exc, "problem", str(exc))}') from exc
    if not isinstance(benchmark, dict):
        raise ValueError('Config must be a plain benchmark mapping; see examples/delivery.yaml')
    _check_interpolation(benchmark)
    dataset = benchmark.get('dataset')
    if dataset is None:
        raise ValueError('Config must include dataset at the root, without a benchmark wrapper')
    required = {
        'delivery': ('layout', 'map_seeds', 'sampling', 'grid'),
        'sortation': ('layout', 'map_seeds', 'sampling', 'grid'),
        'cities_tiles': ('mode', 'map_name'),
        'movingai': ('map_names', 'scenario_numbers', 'scenario_family', 'sampling', 'grid'),
        'qd_mapper': ('map_names', 'scenario_numbers', 'sampling', 'grid'),
    }
    if not isinstance(dataset, str) or dataset not in required:
        raise ValueError('Unsupported dataset')
    missing = [key for key in ('agent_counts', 'episodes', 'generation_seed') + required[dataset]
               if key not in benchmark]
    if missing:
        raise ValueError('Benchmark is missing required fields: ' + ', '.join(missing))
    for key in ('suite_path', 'manifest_path', 'instances_path'):
        if benchmark.get(key) is not None and not isinstance(benchmark[key], str):
            raise ValueError(f'{key} must be a string or null')
    for key in ('grid', 'layout'):
        if key in benchmark and not isinstance(benchmark[key], dict):
            raise ValueError(f'{key} must be a mapping')
    if benchmark.get('manifest') is not None and not isinstance(benchmark['manifest'], dict):
        raise ValueError('manifest must be a mapping')
    for key in ('agent_counts', 'map_seeds', 'scenario_numbers'):
        if key in benchmark and not isinstance(benchmark[key], list):
            raise ValueError(f'{key} must be a list')
    for key in ('map_names', 'collections'):
        if benchmark.get(key) is not None:
            if not isinstance(benchmark[key], list):
                raise ValueError(f'{key} must be a list')
            if any(not isinstance(name, str) for name in benchmark[key]):
                raise ValueError(f'{key} must contain strings')
    if dataset in ('delivery', 'sortation') and 'on_target' not in benchmark['grid']:
        raise ValueError('grid is missing required field: on_target')
    if dataset in ('delivery', 'sortation'):
        settings = toolbox.DeliveryRangeSettings if dataset == 'delivery' else toolbox.SortationRangeSettings
        unknown = set(benchmark['layout']) - {field.name for field in fields(settings)}
        if unknown:
            raise ValueError('Unknown layout fields: ' + ', '.join(str(key) for key in unknown))
    return benchmark


def main(argv=None):
    parser = argparse.ArgumentParser(prog='pogema-toolbox', description='Prepare datasets and generate explicit instance suites.')
    groups = parser.add_subparsers(dest='command')
    datasets = groups.add_parser('datasets', help='List, prepare and generate datasets')
    operations = datasets.add_subparsers(dest='operation')
    operations.add_parser('list', help='Show packaged catalog summaries and rights')
    prepare = operations.add_parser('prepare', help='Explicitly verify or acquire a dataset')
    prepare.add_argument('dataset', metavar='DATASET')
    prepare.add_argument('--artifact-root', default='.', metavar='DIR')
    prepare.add_argument('--collections', nargs='+', metavar='NAME')
    prepare.add_argument('--staged-dir', metavar='DIR')
    prepare.add_argument('--local-dir', metavar='DIR')
    prepare.add_argument('--manifest-path', metavar='FILE')
    prepare.add_argument('--offline', action='store_true')
    prepare.add_argument('--timeout-seconds', type=float, default=30, metavar='NUMBER')
    generate = operations.add_parser('generate', help='Generate a suite from a plain benchmark YAML mapping')
    generate.add_argument('--config', required=True, metavar='BENCHMARK.yaml')
    generate.add_argument('--output-dir', required=True, metavar='DIR')
    generate.add_argument('--artifact-root', default='.', metavar='DIR')
    arguments = parser.parse_args(argv)
    if arguments.command is None:
        parser.print_help()
        return 0
    if arguments.operation is None:
        datasets.print_help()
        return 0
    custom_manifest = False
    try:
        if arguments.operation == 'list':
            result = _catalogs()
        elif arguments.operation == 'prepare':
            custom_manifest = arguments.manifest_path is not None
            result = toolbox.prepare_dataset(arguments.dataset, artifact_root=arguments.artifact_root,
                collections=arguments.collections, staged_dir=arguments.staged_dir,
                local_dir=arguments.local_dir, manifest_path=arguments.manifest_path,
                offline=arguments.offline, timeout_seconds=arguments.timeout_seconds)
        else:
            benchmark = _benchmark(arguments.config)
            custom_manifest = benchmark.get('manifest_path') is not None or benchmark.get('manifest') is not None
            result = toolbox.generate_suite(benchmark, output_dir=arguments.output_dir,
                                            artifact_root=arguments.artifact_root)
        output = json.dumps(result, allow_nan=False)
    except KeyError as exc:
        if not custom_manifest:
            raise
        print(f'pogema-toolbox: error: Invalid custom dataset manifest: missing required field {exc.args[0]!r}', file=sys.stderr)
        return 1
    except yaml.YAMLError as exc:
        print(f'pogema-toolbox: error: Invalid manifest YAML: {getattr(exc, "problem", str(exc))}', file=sys.stderr)
        return 1
    except (ValueError, OSError, zipfile.BadZipFile) as exc:
        print(f'pogema-toolbox: error: {exc}', file=sys.stderr)
        return 1
    print(output)
    return 0
