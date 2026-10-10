import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import zipfile


ROOT = Path(__file__).resolve().parents[1]


def test_distributions_package_catalogs_and_support_standalone_preparation(tmp_path):
    source = tmp_path / 'source'
    shutil.copytree(ROOT, source, ignore=shutil.ignore_patterns('.git', 'build', 'dist', '*.egg-info', '__pycache__', '.pytest_cache'))
    raw = source / 'pogema_toolbox/maps/qd_mapper/data'
    raw.mkdir(parents=True, exist_ok=True)
    (raw / 'private.json').write_text('{}')
    dist = tmp_path / 'dist'
    subprocess.run([sys.executable, 'setup.py', 'sdist', 'bdist_wheel', '--dist-dir', str(dist)], cwd=source, check=True, capture_output=True, text=True)
    # sdist uses its own default dist directory when both commands are requested.
    sdist = next((source / 'dist').glob('*.tar.gz'))
    wheel = next(dist.glob('*.whl'))
    with zipfile.ZipFile(wheel) as archive:
        wheel_names = archive.namelist()
    with tarfile.open(sdist) as archive:
        sdist_names = [name.split('/', 1)[1] for name in archive.getnames() if '/' in name]
    required = ['pogema_toolbox/maps/' + dataset + '/manifest.yaml' for dataset in ['movingai', 'cities_tiles', 'qd_mapper']]
    required += ['pogema_toolbox/maps/movingai/notices/ATTRIBUTION.md',
                 'pogema_toolbox/maps/cities_tiles/notices/ATTRIBUTION.md',
                 'pogema_toolbox/maps/cities_tiles/notices/DDG-LICENSE.txt',
                 'pogema_toolbox/maps/qd_mapper/notices/RIGHTS.md',
                 'pogema_toolbox/licenses/Neural-MAPF-LICENSE']
    for names in [wheel_names, sdist_names]:
        assert set(required).issubset(names)
        assert not any('/data/' in name or name.endswith(('.map', '.scen', '.zip')) for name in names)
    installed = tmp_path / 'installed'
    subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-deps', '--no-index', '--target', str(installed), str(wheel)], check=True, capture_output=True, text=True)
    script = '''
import importlib.abc
from pathlib import Path
import sys
sys.path.insert(0, sys.argv[1])
class ForbidNeural(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('neural_mapf', 'hydra', 'omegaconf'):
            raise AssertionError('standalone preparation imported ' + fullname)
sys.meta_path.insert(0, ForbidNeural())
import pogema_toolbox.datasets as datasets
assert Path(datasets.__file__).is_relative_to(Path(sys.argv[1]))
for dataset in ('movingai', 'cities_tiles', 'qd_mapper'):
    assert datasets.dataset_manifest(dataset)['dataset'] == dataset
assert datasets.prepare_maps('cities_tiles')['map_count'] == 128
root = Path(sys.argv[2])
try:
    datasets.prepare_maps('qd_mapper', artifact_root=root)
except ValueError as exc:
    assert 'permission_unverified' in str(exc)
else:
    raise AssertionError('QD acquisition must be rejected')
assert not root.exists()
print('standalone preparation verified')
'''
    env = dict(os.environ)
    env.pop('PYTHONPATH', None)
    smoke = subprocess.run([sys.executable, '-I', '-c', script, str(installed), str(tmp_path / 'artifacts')], cwd=tmp_path, env=env, check=True, capture_output=True, text=True)
    assert smoke.stdout.strip() == 'standalone preparation verified'
