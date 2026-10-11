import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
EXPORTS = ('Instance', 'save_instances', 'load_instances', 'generate_instances', 'generate_suite',
           'prepare_dataset', 'evaluate', 'dataset_manifest', 'data_directory', 'ScenarioSampler',
           'TaskLayout', 'CitiesTilesGenerator', 'CitiesTilesInstance', 'DeliveryRangeSettings',
           'SortationRangeSettings', 'generate_delivery', 'generate_sortation', 'parse_movingai_map',
           'parse_movingai_scenarios', 'movingai_grid_config', 'MovingAIProblem', 'load_qd_map',
           'QDMap', 'ToolboxRegistry')


def test_root_exports_are_discoverable_and_lazy():
    script = '''
import sys
import pogema_toolbox as toolbox
assert set(sys.argv[1].split(',')).issubset(dir(toolbox))
assert 'pogema_toolbox.evaluator' not in sys.modules
assert 'pogema_toolbox.instances' not in sys.modules
for name in sys.argv[1].split(','):
    assert getattr(toolbox, name) is not None, name
'''
    subprocess.run([sys.executable, '-c', script, ','.join(EXPORTS)], cwd=ROOT,
                   env=dict(os.environ, PYTHONPATH=str(ROOT)), check=True, capture_output=True, text=True)


def test_generation_and_preparation_run_without_optional_backends(tmp_path):
    script = '''
import importlib.abc
import sys
class Forbidden(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('neural_mapf', 'hydra', 'omegaconf', 'torch', 'dask', 'matplotlib'):
            raise AssertionError('standalone API imported ' + fullname)
sys.meta_path.insert(0, Forbidden())
from pogema_toolbox import Instance, generate_instances, generate_suite, prepare_dataset
assert Instance([[0, 0]], [[0, 0]], [[0, 1]], {}).to_grid_config().num_agents == 1
assert len(generate_instances({'grid': {'size': 8, 'num_agents': 1, 'density': 0},
                               'episodes': 2, 'generation_seed': 5})) == 2
assert prepare_dataset('cities_tiles')['map_count'] == 128
result = generate_suite({'dataset': 'delivery', 'map_seeds': [0], 'agent_counts': [2],
    'episodes': 1, 'generation_seed': 0, 'sampling': 'endpoints',
    'layout': {'width_min': 9, 'width_max': 9, 'height_min': 9, 'height_max': 9,
               'block_width_min': 2, 'block_width_max': 2, 'block_height_min': 2,
               'block_height_max': 2, 'aisle_width_min': 1, 'aisle_width_max': 1},
    'grid': {'on_target': 'restart', 'max_episode_steps': 7}}, output_dir=sys.argv[1])
assert result['episodes'] == 1
print('standalone APIs verified')
'''
    completed = subprocess.run([sys.executable, '-c', script, str(tmp_path)], cwd=ROOT,
                               env=dict(os.environ, PYTHONPATH=str(ROOT)), check=True,
                               capture_output=True, text=True)
    assert completed.stdout.strip() == 'standalone APIs verified'
