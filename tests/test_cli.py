import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile

import pytest
import yaml

from pogema_toolbox import load_instances, save_instances


ROOT = Path(__file__).resolve().parents[1]


def run_cli(*arguments, cwd):
    return subprocess.run([sys.executable, '-m', 'pogema_toolbox', *arguments],
                          cwd=cwd, env=dict(os.environ, PYTHONPATH=str(ROOT)),
                          capture_output=True, text=True)


@pytest.mark.parametrize('arguments, expected', [
    ([], 'datasets'), (['--help'], 'datasets'),
    (['datasets'], 'prepare'), (['datasets', '--help'], 'generate'),
    (['datasets', 'list', '--help'], 'list'),
    (['datasets', 'prepare', '--help'], '--collections'),
    (['datasets', 'generate', '--help'], '--config'),
])
def test_help_and_empty_command_succeed(tmp_path, arguments, expected):
    result = run_cli(*arguments, cwd=tmp_path)
    assert result.returncode == 0, result.stderr
    assert 'usage:' in result.stdout and expected in result.stdout
    assert not result.stderr


@pytest.mark.parametrize('arguments', [
    ['unknown'], ['datasets', 'unknown'], ['datasets', 'list', '--unknown'],
    ['datasets', 'prepare'], ['datasets', 'generate'],
])
def test_invalid_commands_fail_on_stderr(tmp_path, arguments):
    result = run_cli(*arguments, cwd=tmp_path)
    assert result.returncode != 0
    assert 'error:' in result.stderr and 'Traceback' not in result.stderr
    assert not result.stdout


def test_list_reports_catalog_rights_without_creating_artifacts(tmp_path):
    result = run_cli('datasets', 'list', cwd=tmp_path)
    assert result.returncode == 0, result.stderr
    catalogs = {entry['name']: entry for entry in json.loads(result.stdout)}
    assert set(catalogs) == {'movingai', 'cities_tiles', 'qd_mapper'}
    assert all(entry['description'] and entry['source'] and entry['rights'] for entry in catalogs.values())
    assert catalogs['qd_mapper']['rights']['status'] == 'permission_unverified'
    assert catalogs['qd_mapper']['collections'] == ['CBS', 'EECBS', 'PBS', 'PBS_EECBS', 'LTF', 'PIBT', 'PIBT_LTF']
    assert not list(tmp_path.iterdir())


def test_prepare_passes_relative_manifest_staging_and_offline_options(tmp_path):
    staged = tmp_path / 'staged'
    staged.mkdir()
    content = b'type octile\nheight 1\nwidth 3\nmap\n...\n'
    archive = staged / 'maps.zip'
    with zipfile.ZipFile(archive, 'w') as stream:
        stream.writestr('tiny.map', content)
    manifest = dict(schema_version=1, dataset='movingai', maps=['tiny'], archives=[dict(
        name='maps', filename='maps.zip', url='https://example.org/maps.zip',
        sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
        members={'tiny.map': hashlib.sha256(content).hexdigest()})])
    (tmp_path / 'manifest.yaml').write_text(yaml.safe_dump(manifest))
    result = run_cli('datasets', 'prepare', 'movingai', '--artifact-root', 'artifacts',
                     '--manifest-path', '../manifest.yaml', '--staged-dir', '../staged',
                     '--offline', '--timeout-seconds', '2.5', cwd=tmp_path)
    assert result.returncode == 0, result.stderr
    prepared = json.loads(result.stdout)
    assert prepared['map_count'] == 1
    assert (Path(prepared['data_dir']) / 'tiny.map').read_bytes() == content
    assert prepared['receipt']['acquired_via'] == 'verified_staging'
    assert not result.stderr


def test_prepare_local_collections_preserves_supplied_order(tmp_path):
    archives = []
    for name in ('CBS', 'LTF'):
        (tmp_path / 'local' / name).mkdir(parents=True)
        (tmp_path / 'local' / name / 'tiny.json').write_bytes(b'{}')
        archives.append(dict(name=name, url='https://example.org/' + name,
                             members={name + '/tiny.json': hashlib.sha256(b'{}').hexdigest()}))
    (tmp_path / 'manifest.yaml').write_text(yaml.safe_dump(dict(
        schema_version=1, dataset='qd_mapper', archives=archives)))
    result = run_cli('datasets', 'prepare', 'qd_mapper', '--artifact-root', 'artifacts',
                     '--manifest-path', '../manifest.yaml', '--local-dir', '../local',
                     '--collections', 'LTF', 'CBS', '--offline', cwd=tmp_path)
    assert result.returncode == 0, result.stderr
    assert [a['name'] for a in json.loads(result.stdout)['receipt']['archives']] == ['LTF', 'CBS']


def test_procedural_generation_uses_cwd_config_and_artifact_paths(tmp_path):
    benchmark = yaml.safe_load((ROOT / 'examples/delivery.yaml').read_text())
    benchmark['suite_path'] = 'suites/delivery.json'
    (tmp_path / 'delivery.yaml').write_text(yaml.safe_dump(benchmark))
    result = run_cli('datasets', 'generate', '--config', 'delivery.yaml',
                     '--output-dir', 'unused', '--artifact-root', 'artifacts', cwd=tmp_path)
    assert result.returncode == 0, result.stderr
    generated = json.loads(result.stdout)
    index = tmp_path / 'artifacts/suites/delivery.json'
    assert Path(generated['suite_path']) == index
    assert generated['ready_groups'] == 4 and generated['episodes'] == 8
    assert not generated['invalid_groups'] and not (tmp_path / 'unused').exists()
    suite = json.loads(index.read_text())
    for group in suite['groups']:
        path = index.parent / group['instances_path']
        instances = load_instances(path)
        assert len(instances) == 2
        assert all(i.grid_config['on_target'] == 'restart' for i in instances)
        assert all(len(s) == 33 for i in instances for s in i.goal_sequences)
        copy = tmp_path / 'roundtrip.jsonl'
        save_instances(instances, copy)
        assert copy.read_bytes() == path.read_bytes()


@pytest.mark.parametrize('text, message', [
    ('42', 'mapping'), ('[]', 'mapping'), ('benchmark: {}', 'dataset'),
    ('dataset: delivery\n', 'missing'),
    ('dataset: ${benchmark.dataset}', 'interpolation'),
    ('dataset: delivery\nlayout: {width_min: "${width}"}', 'interpolation'),
    ('"${key}": delivery', 'interpolation'), ('[invalid', 'YAML'),
    ('dataset: unsupported', 'Unsupported dataset'),
])
def test_invalid_config_fails_with_actionable_error(tmp_path, text, message):
    (tmp_path / 'invalid.yaml').write_text(text)
    result = run_cli('datasets', 'generate', '--config', 'invalid.yaml',
                     '--output-dir', 'generated', cwd=tmp_path)
    assert result.returncode != 0
    assert message in result.stderr and 'Traceback' not in result.stderr
    assert not result.stdout and not (tmp_path / 'generated').exists()


def test_missing_config_and_qd_acquisition_errors_are_visible(tmp_path):
    for arguments, message in [
        (['generate', '--config', 'missing.yaml', '--output-dir', 'generated'], 'missing.yaml'),
        (['prepare', 'qd_mapper', '--artifact-root', 'artifacts'], 'permission_unverified'),
    ]:
        result = run_cli('datasets', *arguments, cwd=tmp_path)
        assert result.returncode != 0 and message in result.stderr
        assert not result.stdout and 'Traceback' not in result.stderr
    assert not (tmp_path / 'artifacts').exists()


@pytest.mark.parametrize('field, value', [
    ('layout', None), ('layout', {'unknown': 2}), ('map_seeds', None),
    ('grid', []), ('map_seeds', '7'), ('episodes', '2'),
])
def test_malformed_benchmark_fields_fail_concisely(tmp_path, field, value):
    benchmark = yaml.safe_load((ROOT / 'examples/delivery.yaml').read_text())
    benchmark[field] = value
    (tmp_path / 'invalid.yaml').write_text(yaml.safe_dump(benchmark))
    result = run_cli('datasets', 'generate', '--config', 'invalid.yaml',
                     '--output-dir', 'generated', cwd=tmp_path)
    assert result.returncode != 0 and 'error:' in result.stderr
    assert not result.stdout and 'Traceback' not in result.stderr
    assert not (tmp_path / 'generated').exists()


@pytest.mark.parametrize('error', [RuntimeError, KeyError, TypeError])
def test_unexpected_programming_error_is_not_success_json(tmp_path, monkeypatch, capsys, error):
    from pogema_toolbox.cli import main

    def broken_generation(*args, **kwargs):
        raise error('unexpected programming error')

    monkeypatch.setattr('pogema_toolbox.generate_suite', broken_generation)
    with pytest.raises(error, match='unexpected programming error'):
        main(['datasets', 'generate', '--config', str(ROOT / 'examples/delivery.yaml'),
              '--output-dir', str(tmp_path)])
    assert not capsys.readouterr().out


def test_malformed_preparation_manifest_fails_concisely(tmp_path):
    (tmp_path / 'bad.yaml').write_text('[invalid')
    result = run_cli('datasets', 'prepare', 'movingai', '--artifact-root', str(tmp_path),
                     '--manifest-path', 'bad.yaml', '--offline', cwd=tmp_path)
    assert result.returncode != 0 and 'YAML' in result.stderr
    assert not result.stdout and 'Traceback' not in result.stderr


def test_nonfinite_result_is_not_emitted_as_json(monkeypatch, capsys):
    from pogema_toolbox.cli import main

    monkeypatch.setattr('pogema_toolbox.prepare_dataset', lambda *args, **kwargs: {'value': float('nan')})
    assert main(['datasets', 'prepare', 'cities_tiles']) != 0
    captured = capsys.readouterr()
    assert not captured.out and 'error:' in captured.err


@pytest.mark.parametrize('operation', ['prepare', 'generate'])
def test_malformed_custom_catalog_reports_missing_field(tmp_path, operation):
    (tmp_path / 'manifest.yaml').write_text('schema_version: 1\ndataset: movingai\n')
    if operation == 'prepare':
        arguments = ['prepare', 'movingai', '--manifest-path', 'manifest.yaml', '--offline']
    else:
        benchmark = dict(dataset='movingai', manifest_path='manifest.yaml', map_names=None,
                         scenario_numbers=[1], scenario_family='random', sampling='replay',
                         agent_counts=[1], episodes=1, generation_seed=0, grid={})
        (tmp_path / 'benchmark.yaml').write_text(yaml.safe_dump(benchmark))
        arguments = ['generate', '--config', 'benchmark.yaml', '--output-dir', 'generated']
    result = run_cli('datasets', *arguments, '--artifact-root', str(tmp_path), cwd=tmp_path)
    assert result.returncode != 0
    assert 'custom dataset manifest' in result.stderr and 'archives' in result.stderr
    assert not result.stdout and 'Traceback' not in result.stderr
    assert not (tmp_path / 'maps').exists() and not (tmp_path / 'generated').exists()


@pytest.mark.parametrize('field', ['suite_path', 'manifest_path', 'instances_path'])
@pytest.mark.parametrize('value', [['bad'], {'bad': 'path'}, 7, False])
def test_malformed_benchmark_path_fails_concisely(tmp_path, field, value):
    benchmark = yaml.safe_load((ROOT / 'examples/delivery.yaml').read_text())
    benchmark[field] = value
    (tmp_path / 'invalid.yaml').write_text(yaml.safe_dump(benchmark))
    result = run_cli('datasets', 'generate', '--config', 'invalid.yaml',
                     '--output-dir', 'generated', cwd=tmp_path)
    assert result.returncode != 0 and field in result.stderr
    assert 'string' in result.stderr and 'Traceback' not in result.stderr
    assert not result.stdout and not (tmp_path / 'generated').exists()


def test_optional_null_benchmark_paths_keep_generation_defaults(tmp_path):
    benchmark = yaml.safe_load((ROOT / 'examples/delivery.yaml').read_text())
    benchmark.update(suite_path=None, manifest_path=None, instances_path=None)
    (tmp_path / 'delivery.yaml').write_text(yaml.safe_dump(benchmark))
    result = run_cli('datasets', 'generate', '--config', 'delivery.yaml',
                     '--output-dir', 'generated', cwd=tmp_path)
    assert result.returncode == 0, result.stderr
    assert not result.stderr
    generated = json.loads(result.stdout)
    assert Path(generated['suite_path']) == tmp_path / 'generated/suite.json'
    assert generated['ready_groups'] == 4 and generated['episodes'] == 8
