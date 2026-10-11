__version__ = '0.2.0'

from importlib import import_module


_EXPORTS = {
    'Instance': 'instances',
    'save_instances': 'instances',
    'load_instances': 'instances',
    'generate_instances': 'benchmarks',
    'generate_suite': 'benchmarks',
    'prepare_dataset': 'datasets',
    'dataset_manifest': 'datasets',
    'data_directory': 'datasets',
    'evaluate': 'evaluator',
    'ScenarioSampler': 'generators.scenario_generator',
    'TaskLayout': 'generators.scenario_generator',
    'CitiesTilesGenerator': 'generators.cities_generator',
    'CitiesTilesInstance': 'generators.cities_generator',
    'DeliveryRangeSettings': 'generators.delivery_generator',
    'generate_delivery': 'generators.delivery_generator',
    'SortationRangeSettings': 'generators.sortation_generator',
    'generate_sortation': 'generators.sortation_generator',
    'parse_movingai_map': 'moving_ai_ingestion',
    'parse_movingai_scenarios': 'moving_ai_ingestion',
    'movingai_grid_config': 'moving_ai_ingestion',
    'MovingAIProblem': 'moving_ai_ingestion',
    'load_qd_map': 'generators.qd_mapper',
    'QDMap': 'generators.qd_mapper',
    'ToolboxRegistry': 'registry',
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
    value = getattr(import_module('.' + _EXPORTS[name], __name__), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()).union(_EXPORTS))
