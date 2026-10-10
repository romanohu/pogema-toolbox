"""Pinned dataset acquisition into the caller's artifact root.

Ported from Neural-MAPF; see licenses/Neural-MAPF-LICENSE for its MIT notice.
"""
import hashlib
from importlib.resources import read_text
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import tempfile
import urllib.request
import zipfile

import yaml


def _artifact_path(artifact_root, value):
    root = Path(artifact_root).expanduser().resolve()
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def dataset_manifest(dataset, *, artifact_root=".", manifest_path=None):
    """Load a pinned packaged catalog or an explicit caller-supplied manifest."""
    if dataset not in ('movingai', 'cities_tiles', 'qd_mapper'):
        raise ValueError('Unsupported dataset')
    text = (_artifact_path(artifact_root, manifest_path).read_text() if manifest_path else
            read_text('pogema_toolbox.maps.' + dataset, 'manifest.yaml'))
    manifest = yaml.safe_load(text)
    if not isinstance(manifest, dict) or manifest.get('schema_version') != 1 or manifest.get('dataset') != dataset:
        raise ValueError(f'Invalid {dataset} manifest')
    return manifest


def data_directory(dataset, *, artifact_root="."):
    """Select the dataset's local artifact directory without acquiring data."""
    return selected_path(_artifact_path(artifact_root, '.'), f'maps/{dataset}/data')


def prepare_maps(dataset, *, artifact_root=".", collections=None, staged_dir=None,
                 local_dir=None, manifest_path=None, offline=False, timeout_seconds=30) -> dict:
    """Verify all inputs before atomically replacing a prepared dataset.

    Relative input paths use artifact_root. QD requires explicit local inputs.
    """
    manifest = dataset_manifest(dataset, artifact_root=artifact_root, manifest_path=manifest_path)
    dataset = manifest['dataset']
    if dataset == 'cities_tiles':
        from pogema_toolbox.generators.cities_generator import load_cities_tiles, ASSET_SHA256
        maps = load_cities_tiles()
        if ASSET_SHA256 != manifest['asset_sha256']:
            raise ValueError('Cities asset SHA256 does not match catalog')
        return {'dataset': dataset, 'map_count': len(maps), 'asset_sha256': ASSET_SHA256}
    if dataset == 'qd_mapper' and not staged_dir and not local_dir:
        raise ValueError('QD-MAPPER permission_unverified: supply explicit local ZIPs via staged_dir or a local directory via local_dir')
    if staged_dir and local_dir:
        raise ValueError('Select only one local import directory')
    archives = selected_archives(manifest, collections)
    destination = data_directory(dataset, artifact_root=artifact_root)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if staged_dir or local_dir:
        directory = staged_dir or local_dir
        raw = Path(directory).expanduser()
        raw = raw if raw.is_absolute() else _artifact_path(artifact_root, '.') / raw
        if any(part.is_symlink() and part != Path('/tmp') for part in (raw, *raw.parents)):
            raise ValueError('Staged directory is a symlink')
    if offline and not (staged_dir or local_dir):
        raise ValueError('Offline map preparation requires staged_dir')
    receipts = []
    with tempfile.TemporaryDirectory(dir=destination.parent, prefix='.prepare-') as temporary:
        staging = Path(temporary)
        data = staging / 'data'
        data.mkdir()
        destinations = set()
        for spec in archives:
            if local_dir:
                for relative, digest in sorted(spec['members'].items()):
                    source = selected_path(_artifact_path(artifact_root, local_dir), relative)
                    target = selected_path(data, relative)
                    if target in destinations:
                        raise ValueError(f'Duplicate archive member: {relative}')
                    destinations.add(target)
                    content = source.read_bytes()
                    if hashlib.sha256(content).hexdigest() != digest:
                        raise ValueError(f'Member SHA256 mismatch for {relative}')
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(content)
                receipts.append({'name': spec['name'], 'url': spec['url'], 'members': spec['members']})
                continue
            archive = staging / (spec['name'] + '.zip')
            source = (selected_path(_artifact_path(artifact_root, staged_dir), spec['filename']).open('rb')
                      if staged_dir else urllib.request.urlopen(spec['url'], timeout=timeout_seconds))
            with source, archive.open('wb') as stream:
                shutil.copyfileobj(source, stream)
            if sha256_file(archive) != spec['sha256']:
                raise ValueError(f'SHA256 mismatch for {spec["name"]}')
            if spec.get('bytes') is not None and archive.stat().st_size != spec['bytes']:
                raise ValueError(f'Byte count mismatch for {spec["name"]}')
            with zipfile.ZipFile(archive) as zipped:
                expected = dict(spec['members'], **spec.get('ignored_members', {}))
                members = set()
                for member in zipped.infolist():
                    relative = member.filename.rstrip('/') if member.is_dir() else member.filename
                    target = selected_path(data, relative)
                    mode = member.external_attr >> 16
                    if stat.S_ISLNK(mode) or (stat.S_IFMT(mode) and not (stat.S_ISREG(mode) or stat.S_ISDIR(mode))):
                        raise ValueError(f'Unsafe archive member: {member.filename}')
                    if member.is_dir():
                        continue
                    if target in destinations or relative not in expected:
                        raise ValueError(f'Duplicate or unlisted archive member: {relative}')
                    destinations.add(target)
                    members.add(relative)
                    content = zipped.read(member)
                    if hashlib.sha256(content).hexdigest() != expected[relative]:
                        raise ValueError(f'Member SHA256 mismatch for {relative}')
                    if relative in spec['members']:
                        target.parent.mkdir(parents=True, exist_ok=True)
                        target.write_bytes(content)
                if members != set(expected):
                    raise ValueError(f'Missing archive members in {spec["name"]}')
            receipts.append({key: spec[key] for key in ('name', 'url', 'sha256')})
            if staged_dir:
                receipts[-1]['original_local_path'] = str(selected_path(_artifact_path(artifact_root, staged_dir), spec['filename']))
        receipt = {'schema_version': 1, 'dataset': dataset, 'archives': receipts,
                   'manifest_sha256': hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest(),
                   'acquired_via': ('verified_local_directory' if local_dir else
                                    'verified_staging' if staged_dir else 'https'),
                   'attribution': manifest.get('attribution'), 'rights': manifest.get('rights')}
        if local_dir:
            receipt['original_local_path'] = str(_artifact_path(artifact_root, local_dir))
        (data / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
        # All archives and members must pass before the existing dataset moves.
        backup = staging / 'previous'
        if destination.exists():
            os.replace(destination, backup)
        try:
            os.replace(data, destination)
        except BaseException:
            if backup.exists():
                os.replace(backup, destination)
            raise
    return {'dataset': dataset, 'data_dir': str(destination), 'map_count': len(manifest['maps']) if dataset == 'movingai' else sum(len(a['members']) for a in archives),
            'receipt': receipt}


def selected_archives(manifest, collections=None):
    """Select QD collections in supplied order, defaulting to catalog order."""
    if manifest['dataset'] != 'qd_mapper':
        return manifest['archives']
    names = [a['name'] for a in manifest['archives']] if collections is None else list(collections)
    available = {a['name']: a for a in manifest['archives']}
    if not names or len(set(names)) != len(names) or any(n not in available for n in names):
        raise ValueError('collections must be nonempty, unique known QD collection names')
    return [available[name] for name in names]


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def selected_path(root, relative):
    """Select beneath the explicit artifact/staging boundary, checking every child."""
    value = PurePosixPath(relative)
    if (not relative or '\\' in relative or value.is_absolute() or
            any(part in ('..', '.') for part in relative.split('/'))):
        raise ValueError(f'Unsafe selected file path: {relative!r}')
    # /tmp is the intentional /private/tmp platform alias on macOS.
    if Path(root).is_symlink() and Path(root) != Path('/tmp'):
        raise ValueError(f'Selected root is a symlink: {root}')
    target = Path(root).resolve()
    for part in value.parts:
        target = target / part
        if target.is_symlink():
            raise ValueError(f'Selected path contains a symlink: {target}')
    return target
