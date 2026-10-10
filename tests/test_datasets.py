import hashlib
import json
import stat
import zipfile

import pytest
from pathlib import Path

import yaml

from pogema_toolbox.datasets import dataset_manifest, data_directory, prepare_maps, selected_archives


@pytest.fixture
def staged_dataset(tmp_path):
    staged = tmp_path / 'staged'
    staged.mkdir()
    archive = staged / 'maps.zip'
    contents = b'type octile\nheight 1\nwidth 3\nmap\n...\n'
    with zipfile.ZipFile(archive, 'w') as stream:
        stream.writestr('tiny.map', contents)
    manifest = {'schema_version': 1, 'dataset': 'movingai', 'maps': ['tiny'], 'archives': [
        {'name': 'maps', 'filename': 'maps.zip', 'url': 'https://example.org/maps.zip',
         'sha256': hashlib.sha256(archive.read_bytes()).hexdigest(),
         'members': {'tiny.map': hashlib.sha256(contents).hexdigest()}}]}
    source = tmp_path / 'manifest.yaml'
    source.write_text(yaml.safe_dump(manifest))
    cfg = {'artifact_root': tmp_path / 'artifacts', 'manifest_path': source,
           'staged_dir': staged}
    return cfg, archive, source, manifest


def test_offline_prepare_checks_hash_and_preserves_existing_data(staged_dataset):
    cfg, archive, _, _ = staged_dataset
    cfg['offline'] = True
    cfg['staged_dir'] = '../staged'
    result = prepare_maps('movingai', **cfg)
    data = Path(result['data_dir'])
    assert (data / 'tiny.map').read_text().endswith('...\n')
    assert json.loads((data / 'receipt.json').read_text())['archives'][0]['sha256'] == hashlib.sha256(archive.read_bytes()).hexdigest()
    assert result['receipt']['archives'][0]['original_local_path'] == str(archive.resolve())
    assert json.loads((data / 'receipt.json').read_text()) == result['receipt']
    archive.write_bytes(b'bad replacement')
    with pytest.raises(ValueError, match='SHA256'):
        prepare_maps('movingai', **cfg)
    assert (data / 'tiny.map').read_text().endswith('...\n')


@pytest.mark.parametrize('kind', ['traversal', 'duplicate', 'symlink', 'unlisted'])
def test_prepare_rejects_unsafe_archive_members(staged_dataset, kind):
    cfg, archive, source, manifest = staged_dataset
    with zipfile.ZipFile(archive, 'w') as stream:
        if kind == 'symlink':
            member = zipfile.ZipInfo('tiny.map')
            member.external_attr = (stat.S_IFLNK | 0o777) << 16
            stream.writestr(member, 'target')
        else:
            stream.writestr('tiny.map', b'content')
            if kind == 'duplicate':
                with pytest.warns(UserWarning, match='Duplicate name'):
                    stream.writestr('tiny.map', b'content')
            else:
                stream.writestr({'traversal': '../escaped', 'unlisted': 'extra.map'}[kind], b'content')
    manifest['archives'][0]['sha256'] = hashlib.sha256(archive.read_bytes()).hexdigest()
    manifest['archives'][0]['members']['tiny.map'] = hashlib.sha256(b'content').hexdigest()
    source.write_text(yaml.safe_dump(manifest))
    with pytest.raises(ValueError):
        prepare_maps('movingai', **cfg)
    assert not (archive.parent.parent / 'escaped').exists()


@pytest.mark.parametrize('nested', [False, True])
def test_prepare_rejects_staged_directory_symlink(staged_dataset, tmp_path, nested):
    cfg, archive, _, _ = staged_dataset
    alias = tmp_path / 'alias'
    if nested:
        alias.symlink_to(archive.parent.parent, target_is_directory=True)
        cfg['staged_dir'] = str(alias / archive.parent.name)
    else:
        alias.symlink_to(archive.parent, target_is_directory=True)
        cfg['staged_dir'] = str(alias)
    with pytest.raises(ValueError, match='symlink'):
        prepare_maps('movingai', **cfg)


def test_prepare_rejects_destination_alias_and_preserves_existing_data(staged_dataset):
    cfg, archive, source, manifest = staged_dataset
    prepared = prepare_maps('movingai', **cfg)
    data = Path(prepared['data_dir'])
    original = (data / 'tiny.map').read_bytes()
    original_receipt = (data / 'receipt.json').read_bytes()
    with zipfile.ZipFile(archive, 'w') as stream:
        stream.writestr('nested/tiny.map', b'first')
        stream.writestr('nested//tiny.map', b'second')
    manifest['archives'][0]['sha256'] = hashlib.sha256(archive.read_bytes()).hexdigest()
    manifest['archives'][0]['members'] = {
        'nested/tiny.map': hashlib.sha256(b'first').hexdigest(),
        'nested//tiny.map': hashlib.sha256(b'second').hexdigest(),
    }
    source.write_text(yaml.safe_dump(manifest))
    with pytest.raises(ValueError, match='Duplicate'):
        prepare_maps('movingai', **cfg)
    assert (data / 'tiny.map').read_bytes() == original
    assert (data / 'receipt.json').read_bytes() == original_receipt
    assert not (data / 'nested/tiny.map').exists()


@pytest.fixture
def local_qd_dataset(tmp_path):
    local = tmp_path / 'local'
    contents = b'{"name":"tiny","weight":false,"n_row":1,"n_col":3,"layout":["..."],"start":[[0]],"goal":[[2]]}\n'
    archives = []
    for name in ['CBS', 'LTF']:
        (local / name).mkdir(parents=True)
        (local / name / 'tiny.json').write_bytes(contents)
        archives.append({'name': name, 'url': 'https://example.org/' + name + '.zip',
                         'members': {name + '/tiny.json': '8694dd6df193ee716ba9f1fb6cf8b3af3f50bd9eeefff85e4ce5b3cc1fe15ac2'}})
    manifest = {'schema_version': 1, 'dataset': 'qd_mapper', 'archives': archives,
                'rights': {'status': 'permission_unverified'}, 'attribution': 'fixture'}
    source = tmp_path / 'qd.yaml'
    source.write_text(yaml.safe_dump(manifest))
    return {'artifact_root': tmp_path / 'artifacts', 'manifest_path': '../qd.yaml',
            'local_dir': '../local'}, local, contents


@pytest.mark.parametrize('collections, names', [(None, ['CBS', 'LTF']), (['LTF', 'CBS'], ['LTF', 'CBS']), (['LTF'], ['LTF'])])
def test_qd_local_import_preserves_selection_and_receipt(local_qd_dataset, collections, names):
    options, local, contents = local_qd_dataset
    result = prepare_maps('qd_mapper', collections=collections, **options)
    data = Path(result['data_dir'])
    assert result['map_count'] == len(names)
    assert [entry['name'] for entry in result['receipt']['archives']] == names
    assert result['receipt']['rights']['status'] == 'permission_unverified'
    assert result['receipt']['original_local_path'] == str(local.resolve())
    assert result['receipt']['acquired_via'] == 'verified_local_directory'
    assert json.loads((data / 'receipt.json').read_text()) == result['receipt']
    assert sorted(str(path.relative_to(data)) for path in data.rglob('*.json') if path.name != 'receipt.json') == sorted(name + '/tiny.json' for name in names)
    for name in names:
        assert (data / name / 'tiny.json').read_bytes() == contents


@pytest.mark.parametrize('collections', [[], ['CBS', 'CBS'], ['unknown']])
def test_qd_rejects_invalid_collection_selection(local_qd_dataset, collections):
    options, _, _ = local_qd_dataset
    manifest = dataset_manifest('qd_mapper', artifact_root=options['artifact_root'], manifest_path=options['manifest_path'])
    with pytest.raises(ValueError, match='collections'):
        selected_archives(manifest, collections)
    with pytest.raises(ValueError, match='collections'):
        prepare_maps('qd_mapper', collections=collections, **options)
    assert not options['artifact_root'].exists()


def test_qd_rejects_automatic_acquisition_before_side_effects(tmp_path, monkeypatch):
    def forbid_network(*args, **kwargs):
        raise AssertionError('QD preparation must not make network calls')
    monkeypatch.setattr('urllib.request.urlopen', forbid_network)
    root = tmp_path / 'artifacts'
    with pytest.raises(ValueError, match='permission_unverified'):
        prepare_maps('qd_mapper', artifact_root=root)
    assert not root.exists()


def test_qd_local_hash_failure_preserves_existing_data(local_qd_dataset):
    options, local, contents = local_qd_dataset
    result = prepare_maps('qd_mapper', **options)
    data = Path(result['data_dir'])
    receipt = (data / 'receipt.json').read_bytes()
    (local / 'LTF/tiny.json').write_bytes(b'bad replacement')
    with pytest.raises(ValueError, match='SHA256'):
        prepare_maps('qd_mapper', **options)
    assert (data / 'LTF/tiny.json').read_bytes() == contents
    assert (data / 'receipt.json').read_bytes() == receipt


@pytest.mark.parametrize('kind', ['root', 'parent', 'member'])
def test_qd_rejects_local_symlinks(local_qd_dataset, tmp_path, kind):
    options, local, _ = local_qd_dataset
    if kind == 'member':
        member = local / 'CBS/tiny.json'
        member.unlink()
        member.symlink_to(local / 'LTF/tiny.json')
    else:
        alias = tmp_path / 'alias'
        alias.symlink_to(local if kind == 'root' else tmp_path, target_is_directory=True)
        options['local_dir'] = alias if kind == 'root' else alias / 'local'
    with pytest.raises(ValueError, match='symlink'):
        prepare_maps('qd_mapper', **options)


def test_data_directory_uses_artifact_root(tmp_path):
    assert data_directory('movingai', artifact_root=tmp_path) == tmp_path / 'maps/movingai/data'


@pytest.mark.parametrize('dataset', ['movingai', 'cities_tiles', 'qd_mapper'])
def test_packaged_catalogs_match_dataset(dataset):
    assert dataset_manifest(dataset)['dataset'] == dataset


def test_manifest_rejects_wrong_dataset(tmp_path):
    source = tmp_path / 'bad.yaml'
    source.write_text('schema_version: 1\ndataset: qd_mapper\n')
    with pytest.raises(ValueError, match='Invalid movingai manifest'):
        dataset_manifest('movingai', artifact_root=tmp_path, manifest_path='bad.yaml')


def test_prepare_verifies_packaged_cities():
    result = prepare_maps('cities_tiles')
    assert result == {'dataset': 'cities_tiles', 'map_count': 128,
                      'asset_sha256': '3d357b87c64bab6a08a8ec43cfe81295ed37fd4a14db36d563e4ae599785664f'}


def test_selected_archives_defaults_to_catalog_order(local_qd_dataset):
    options, _, _ = local_qd_dataset
    manifest = dataset_manifest('qd_mapper', artifact_root=options['artifact_root'], manifest_path=options['manifest_path'])
    assert [entry['name'] for entry in selected_archives(manifest)] == ['CBS', 'LTF']


def test_prepare_restores_existing_data_when_replacement_fails(staged_dataset, monkeypatch):
    import os

    options, _, _, _ = staged_dataset
    result = prepare_maps('movingai', **options)
    data = Path(result['data_dir'])
    original = (data / 'tiny.map').read_bytes()
    receipt = (data / 'receipt.json').read_bytes()
    replace = os.replace

    def fail_new_data(source, destination):
        if Path(source).name == 'data' and Path(source).parent.name.startswith('.prepare-'):
            raise OSError('replacement failed')
        return replace(source, destination)

    monkeypatch.setattr('pogema_toolbox.datasets.os.replace', fail_new_data)
    with pytest.raises(OSError, match='replacement failed'):
        prepare_maps('movingai', **options)
    assert (data / 'tiny.map').read_bytes() == original
    assert (data / 'receipt.json').read_bytes() == receipt
    assert not list(data.parent.glob('.prepare-*'))
