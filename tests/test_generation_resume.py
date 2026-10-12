import hashlib
import json

import pytest

from pogema_toolbox import generate_suite, load_instances, save_instances
from test_benchmarks import configured as fixed_configured, fixed_sources
from test_validated_generation import configured, saved


class StableValidator:
    provenance = {'solver': 'controlled-seed-validator', 'revision': '1'}

    def __init__(self, interrupt_seed=None):
        self.interrupt_seed = interrupt_seed
        self.seeds = []

    def __call__(self, instance):
        seed = instance.grid_config['seed']
        self.seeds.append(seed)
        if seed == self.interrupt_seed:
            raise KeyboardInterrupt
        accepted = seed != 17
        return {'accepted': accepted, 'status': 'accepted' if accepted else 'over_step_limit',
                'paths': [[list(point)] for point in instance.starts] if accepted else None,
                'soc': 0 if accepted else None, 'arrival_times': [0] * len(instance.starts)}


@pytest.mark.parametrize('dataset', ['random', 'maze', 'warehouse', 'room', 'delivery', 'sortation'])
def test_resume_preserves_accepted_prefix_attempts_and_instance_bytes(tmp_path, dataset):
    benchmark = configured(dataset)
    baseline = generate_suite(benchmark, output_dir=tmp_path / 'baseline', validator=StableValidator())
    interrupted = StableValidator(interrupt_seed=18)
    destination = tmp_path / 'resumed'
    with pytest.raises(KeyboardInterrupt):
        generate_suite(benchmark, output_dir=destination, validator=interrupted)
    assert interrupted.seeds == [17, 19, 18]
    assert not (destination / 'suite.json').exists()
    assert not list(destination.rglob('instances.jsonl'))

    continued = StableValidator()
    result = generate_suite(benchmark, output_dir=destination, validator=continued, resume=True)
    assert continued.seeds == [18]
    expected, baseline_dir = saved(baseline)
    actual, resumed_dir = saved(result)
    assert actual == expected
    relative = actual['groups'][0]['instances_path']
    assert (resumed_dir / relative).read_bytes() == (baseline_dir / relative).read_bytes()
    assert actual['groups'][0]['metadata']['validation']['attempts'][1]['soc'] == 0


def test_resume_starts_at_unfinished_attempt_after_a_saved_rejection(tmp_path):
    with pytest.raises(KeyboardInterrupt):
        generate_suite(configured(), output_dir=tmp_path, validator=StableValidator(interrupt_seed=19))
    continued = StableValidator()
    result = generate_suite(configured(), output_dir=tmp_path, validator=continued, resume=True)
    assert continued.seeds == [19, 18]
    assert result['validation_summary']['total_attempts'] == 3


@pytest.mark.parametrize('dataset', ['movingai', 'qd_mapper', 'cities_tiles', 'random', 'maze',
                                    'warehouse', 'room', 'delivery', 'sortation'])
def test_completed_suites_resume_without_regenerating_instances(tmp_path, fixed_sources, monkeypatch, dataset):
    benchmark = fixed_configured(dataset) if dataset in ('movingai', 'qd_mapper', 'cities_tiles') else configured(dataset)
    benchmark['agent_counts'] = [2]
    if dataset in fixed_sources:
        benchmark['manifest'] = fixed_sources[dataset]
    destination = tmp_path / 'suite'
    original = generate_suite(benchmark, output_dir=destination, artifact_root=tmp_path)
    before = {path: path.read_bytes() for path in destination.rglob('*') if path.is_file()}
    monkeypatch.setattr('pogema_toolbox.benchmarks._from_grid',
                        lambda *args: pytest.fail('A completed instance was regenerated'))
    resumed = generate_suite(benchmark, output_dir=destination, artifact_root=tmp_path, resume=True)
    assert resumed == original
    assert {path: path.read_bytes() for path in destination.rglob('*') if path.is_file()} == before


def test_resume_reuses_completed_fixed_groups_before_interruption(tmp_path, fixed_sources, monkeypatch):
    import pogema_toolbox.benchmarks as benchmarks
    benchmark = fixed_configured('movingai')
    benchmark.update(agent_counts=[1, 2], manifest=fixed_sources['movingai'])
    baseline = generate_suite(benchmark, output_dir=tmp_path / 'baseline', artifact_root=tmp_path)
    convert = benchmarks._from_grid

    def interrupt_second_group(grid):
        if grid.num_agents == 2:
            raise KeyboardInterrupt
        return convert(grid)

    with monkeypatch.context() as patch:
        patch.setattr(benchmarks, '_from_grid', interrupt_second_group)
        with pytest.raises(KeyboardInterrupt):
            generate_suite(benchmark, output_dir=tmp_path / 'resumed', artifact_root=tmp_path)

    def refuse_first_group(grid):
        assert grid.num_agents == 2, 'The completed group was regenerated'
        return convert(grid)

    monkeypatch.setattr(benchmarks, '_from_grid', refuse_first_group)
    result = generate_suite(benchmark, output_dir=tmp_path / 'resumed', artifact_root=tmp_path, resume=True)
    assert saved(result)[0] == saved(baseline)[0]


@pytest.mark.parametrize('mismatch', ['benchmark', 'validator', 'attempt_limit'])
def test_resume_rejects_identity_mismatches_before_changing_files(tmp_path, mismatch):
    benchmark = configured()
    generate_suite(benchmark, output_dir=tmp_path, validator=StableValidator())
    before = {path: path.read_bytes() for path in tmp_path.rglob('*') if path.is_file()}
    validator = StableValidator()
    attempts = 100
    if mismatch == 'benchmark':
        benchmark['grid']['max_episode_steps'] = 6
    elif mismatch == 'validator':
        validator.provenance = dict(validator.provenance, revision='2')
    else:
        attempts = 2
    with pytest.raises(ValueError, match='identity'):
        generate_suite(benchmark, output_dir=tmp_path, validator=validator, max_attempts=attempts, resume=True)
    assert not validator.seeds
    assert {path: path.read_bytes() for path in tmp_path.rglob('*') if path.is_file()} == before


def test_resume_rejects_changed_fixed_sources(tmp_path, fixed_sources):
    benchmark = fixed_configured('movingai')
    benchmark['manifest'] = fixed_sources['movingai']
    destination = tmp_path / 'suite'
    generate_suite(benchmark, output_dir=destination, artifact_root=tmp_path)
    before = (destination / 'suite.json').read_bytes()
    source = tmp_path / 'maps/movingai/data/tiny.map'
    source.write_bytes(source.read_bytes().replace(b'...', b'.@.', 1))
    with pytest.raises(ValueError, match='identity'):
        generate_suite(benchmark, output_dir=destination, artifact_root=tmp_path, resume=True)
    assert (destination / 'suite.json').read_bytes() == before


def test_resume_compares_validator_identity_types_exactly(tmp_path):
    original = StableValidator()
    original.provenance = dict(original.provenance, revision=1)
    generate_suite(configured(), output_dir=tmp_path, validator=original)
    changed = StableValidator()
    changed.provenance = dict(original.provenance, revision=True)
    with pytest.raises(ValueError, match='identity'):
        generate_suite(configured(), output_dir=tmp_path, validator=changed, resume=True)
    assert not changed.seeds


def test_resume_requires_declared_validator_identity(tmp_path):
    validator = lambda instance: {'accepted': True, 'status': 'accepted'}
    generate_suite(configured(), output_dir=tmp_path, validator=validator)
    before = (tmp_path / 'suite.json').read_bytes()
    with pytest.raises(ValueError, match='validator.provenance'):
        generate_suite(configured(), output_dir=tmp_path, validator=validator, resume=True)
    assert (tmp_path / 'suite.json').read_bytes() == before


def test_caller_provenance_is_saved_atomically_and_participates_in_resume_identity(tmp_path):
    provenance = {'tested_toolbox_revision': 'controlled-revision',
                  'instance_validation': StableValidator.provenance}
    original = generate_suite(configured(), output_dir=tmp_path, validator=StableValidator(), provenance=provenance)
    suite, directory = saved(original)
    assert suite['provenance']['tested_toolbox_revision'] == 'controlled-revision'
    assert suite['provenance']['instance_validation'] == StableValidator.provenance
    before = (directory / 'suite.json').read_bytes()
    resumed = generate_suite(configured(), output_dir=tmp_path, validator=StableValidator(),
                             provenance=provenance, resume=True)
    assert resumed == original and (directory / 'suite.json').read_bytes() == before
    changed = dict(provenance, tested_toolbox_revision='changed-revision')
    with pytest.raises(ValueError, match='identity'):
        generate_suite(configured(), output_dir=tmp_path, validator=StableValidator(),
                       provenance=changed, resume=True)
    assert (directory / 'suite.json').read_bytes() == before


@pytest.mark.parametrize('provenance', [[], 'unknown'])
def test_caller_provenance_must_be_a_mapping_before_writing(tmp_path, provenance):
    destination = tmp_path / 'new'
    with pytest.raises(TypeError, match='provenance.*mapping'):
        generate_suite(configured(), output_dir=destination, provenance=provenance)
    assert not destination.exists()


def test_validation_reports_keep_complete_paths_in_an_atomic_sidecar_and_resume(tmp_path):
    result = generate_suite(configured(), output_dir=tmp_path, validator=StableValidator())
    suite, directory = saved(result)
    group = suite['groups'][0]
    path = directory / group['validation_plans_path']
    assert hashlib.sha256(path.read_bytes()).hexdigest() == group['validation_plans_sha256']
    plans = json.loads(path.read_text())
    assert plans['schema_version'] == 1
    instances = load_instances(directory / group['instances_path'])
    assert list(plans['plans']) == [instance.id for instance in instances]
    for instance in instances:
        report = plans['plans'][instance.id]
        assert report['paths'] == [[list(point)] for point in instance.starts]
        assert report['soc'] == 0 and report['arrival_times'] == [0, 0]
        assert report['generation_seed'] == instance.grid_config['seed']
    assert all('paths' not in record for record in group['metadata']['validation']['attempts'] if record['accepted'])
    before = path.read_bytes()
    validator = StableValidator()
    resumed = generate_suite(configured(), output_dir=tmp_path, validator=validator, resume=True)
    assert resumed == result and not validator.seeds and path.read_bytes() == before


def test_discarded_accepted_paths_are_spilled_without_becoming_eligible_episodes(tmp_path):
    validator = StableValidator()
    validator.provenance = dict(validator.provenance, revision='discard-prefix')

    def accept_first_only(instance):
        report = validator(instance)
        report['accepted'] = instance.grid_config['seed'] == 17
        report['status'] = 'accepted' if report['accepted'] else 'no_solution'
        if report['accepted']:
            report['paths'] = [[list(point)] for point in instance.starts]
        return report

    result = generate_suite(configured(), output_dir=tmp_path, validator=accept_first_only, max_attempts=2)
    suite, directory = saved(result)
    group = suite['groups'][0]
    assert group['status'] == 'invalid' and group['episode_count'] == 0
    assert not list(tmp_path.rglob('instances.jsonl'))
    plans = json.loads((directory / group['validation_plans_path']).read_text())['plans']
    assert len(plans) == 1 and next(iter(plans.values()))['generation_seed'] == 17
    assert all('paths' not in attempt for attempt in group['metadata']['validation']['attempts'] if attempt['accepted'])


def test_resume_rejects_tampered_validation_plan_sidecar_before_solver_execution(tmp_path):
    result = generate_suite(configured(), output_dir=tmp_path, validator=StableValidator())
    suite, directory = saved(result)
    path = directory / suite['groups'][0]['validation_plans_path']
    path.write_text('{}')
    before = (directory / 'suite.json').read_bytes()
    validator = StableValidator()
    with pytest.raises(ValueError, match='progress|plans'):
        generate_suite(configured(), output_dir=tmp_path, validator=validator, resume=True)
    assert not validator.seeds and (directory / 'suite.json').read_bytes() == before


@pytest.mark.parametrize('damage', ['truncated', 'changed_attempt', 'instance_file'])
def test_resume_rejects_damaged_progress_or_instances(tmp_path, damage):
    generate_suite(configured(), output_dir=tmp_path, validator=StableValidator())
    progress = tmp_path / 'suite.json.progress.json'
    assert progress.is_file(), 'Generation did not save recoverable progress'
    if damage == 'truncated':
        progress.write_text('{')
    elif damage == 'changed_attempt':
        data = json.loads(progress.read_text())
        data['groups'][0]['metadata']['validation']['attempts'][0]['accepted'] = True
        progress.write_text(json.dumps(data))
    else:
        next(tmp_path.rglob('instances.jsonl')).write_text('{}\n')
    before = {path: path.read_bytes() for path in tmp_path.rglob('*') if path.is_file()}
    validator = StableValidator()
    with pytest.raises(ValueError, match='progress|instance'):
        generate_suite(configured(), output_dir=tmp_path, validator=validator, resume=True)
    assert not validator.seeds
    assert {path: path.read_bytes() for path in tmp_path.rglob('*') if path.is_file()} == before


def test_validation_summary_distinguishes_discarded_acceptance_from_retained_episodes(tmp_path):
    benchmark = configured()
    benchmark['agent_counts'] = [2, 3]

    def validator(instance):
        accepted = len(instance.starts) == 2 or instance.grid_config['seed'] == 17
        return {'accepted': accepted, 'status': 'accepted' if accepted else 'over_step_limit'}

    result = generate_suite(benchmark, output_dir=tmp_path, validator=validator, max_attempts=2)
    expected = dict(enabled=True, total_attempts=5, solver_checked_candidates=5,
                    accepted_attempts=3, retained_episodes=2, discarded_accepted_attempts=1,
                    invalid_groups=1, rejection_reasons={'over_step_limit': 2},
                    acceptance_rate=.6, solver_acceptance_rate=.6)
    assert result['validation_summary'] == expected
    assert saved(result)[0]['validation_summary'] == expected
    assert result['invalid_groups'][0]['episode_count'] == 0


def test_generation_errors_are_not_counted_as_solver_checked_candidates(tmp_path):
    benchmark = configured()
    benchmark['agent_counts'] = [10000]
    result = generate_suite(benchmark, output_dir=tmp_path, validator=StableValidator(), max_attempts=2)
    assert result['validation_summary'] == dict(enabled=True, total_attempts=2, solver_checked_candidates=0,
        accepted_attempts=0, retained_episodes=0, discarded_accepted_attempts=0, invalid_groups=1,
        rejection_reasons={'generation_error': 2}, acceptance_rate=0., solver_acceptance_rate=None)


def test_unfiltered_summary_marks_validation_statistics_unavailable(tmp_path):
    result = generate_suite(configured(), output_dir=tmp_path)
    assert result['validation_summary'] == dict(enabled=False, total_attempts=0, solver_checked_candidates=0,
        accepted_attempts=0, retained_episodes=2, discarded_accepted_attempts=0, invalid_groups=0,
        rejection_reasons={}, acceptance_rate=None, solver_acceptance_rate=None)


def test_failed_instance_save_keeps_previous_artifact_and_removes_temporary_files(tmp_path):
    result = generate_suite(configured(), output_dir=tmp_path)
    suite, directory = saved(result)
    path = directory / suite['groups'][0]['instances_path']
    instances = load_instances(path)
    previous = path.read_bytes()

    def broken_stream():
        yield instances[0]
        raise RuntimeError('Interrupted serialization')

    with pytest.raises(RuntimeError, match='Interrupted serialization'):
        save_instances(broken_stream(), path)
    assert path.read_bytes() == previous
    assert sorted(item.name for item in path.parent.iterdir()) == ['instances.jsonl']


@pytest.mark.parametrize('artifact', ['suite.json.progress.json', 'instances.jsonl', 'validation-plans.json', 'suite.json'])
def test_resume_after_interrupted_atomic_artifact_replacement_matches_uninterrupted_generation(tmp_path, monkeypatch, artifact):
    import os
    from pathlib import Path
    baseline = generate_suite(configured(), output_dir=tmp_path / 'baseline', validator=StableValidator())
    destination = tmp_path / 'interrupted'
    replace = os.replace
    def interrupt_replace(source, target):
        if Path(target).name == artifact:
            current = json.loads(Path(source).read_text())['current'] if artifact == 'suite.json.progress.json' else None
            if artifact != 'suite.json.progress.json' or (current is not None and
                    current['group']['metadata']['validation']['attempts'][-1]['generation_seed'] == 18):
                raise KeyboardInterrupt
        return replace(source, target)

    with monkeypatch.context() as patch:
        patch.setattr(os, 'replace', interrupt_replace)
        with pytest.raises(KeyboardInterrupt):
            generate_suite(configured(), output_dir=destination, validator=StableValidator())
    assert not list(destination.rglob('*.tmp'))
    continued = StableValidator()
    result = generate_suite(configured(), output_dir=destination, validator=continued, resume=True)
    assert continued.seeds == ([18] if artifact == 'suite.json.progress.json' else [])
    actual, directory = saved(result)
    expected, baseline_dir = saved(baseline)
    assert actual == expected
    group = actual['groups'][0]
    for relative in (group['instances_path'], group['validation_plans_path']):
        assert (directory / relative).read_bytes() == (baseline_dir / relative).read_bytes()


@pytest.mark.parametrize('dataset', ['random', 'movingai', 'qd_mapper', 'cities_tiles'])
def test_restart_after_real_instance_replacement_resumes_the_new_identity(tmp_path, fixed_sources, monkeypatch, dataset):
    from copy import deepcopy
    import os
    from pathlib import Path
    benchmark = configured() if dataset == 'random' else fixed_configured(dataset)
    benchmark['agent_counts'] = [2]
    if dataset in fixed_sources:
        benchmark['manifest'] = fixed_sources[dataset]
    destination = tmp_path / 'restarted'
    generate_suite(benchmark, output_dir=destination, artifact_root=tmp_path)
    changed = deepcopy(benchmark)
    if 'grid' in changed:
        changed['grid']['max_episode_steps'] += 1
    else:
        changed['generation_seed'] += 1
    baseline = generate_suite(changed, output_dir=tmp_path / 'baseline', artifact_root=tmp_path)
    replace = os.replace

    def interrupt_after_replacement(source, target):
        replace(source, target)
        if Path(target).name == 'instances.jsonl':
            raise KeyboardInterrupt

    with monkeypatch.context() as patch:
        patch.setattr(os, 'replace', interrupt_after_replacement)
        with pytest.raises(KeyboardInterrupt):
            generate_suite(changed, output_dir=destination, artifact_root=tmp_path)
    resumed = generate_suite(changed, output_dir=destination, artifact_root=tmp_path, resume=True)
    actual, directory = saved(resumed)
    expected, baseline_dir = saved(baseline)
    assert actual == expected
    for group in actual['groups']:
        relative = group['instances_path']
        assert (directory / relative).read_bytes() == (baseline_dir / relative).read_bytes()
    with pytest.raises(ValueError, match='identity'):
        generate_suite(benchmark, output_dir=destination, artifact_root=tmp_path, resume=True)


@pytest.mark.parametrize('failed_seed, resumed_seeds', [(19, [19, 18]), (18, [18])])
def test_checkpoint_io_error_preserves_committed_attempts_and_resumes_accepted_prefix(tmp_path, monkeypatch, failed_seed, resumed_seeds):
    import os
    from pathlib import Path
    destination = tmp_path / 'interrupted'
    baseline = generate_suite(configured(), output_dir=tmp_path / 'baseline', validator=StableValidator())
    progress = destination / 'suite.json.progress.json'
    replace = os.replace
    failed = False
    committed = None

    def fail_attempt_checkpoint(source, target):
        nonlocal failed, committed
        if Path(target).name == 'suite.json.progress.json':
            current = json.loads(Path(source).read_text())['current']
            if not failed and current is not None and current['group']['metadata']['validation']['attempts'][-1]['generation_seed'] == failed_seed:
                failed = True
                committed = progress.read_bytes()
                raise OSError('Temporary checkpoint write failure')
        return replace(source, target)

    with monkeypatch.context() as patch:
        patch.setattr(os, 'replace', fail_attempt_checkpoint)
        with pytest.raises(OSError, match='Temporary checkpoint write failure'):
            generate_suite(configured(), output_dir=destination, validator=StableValidator())
    assert progress.read_bytes() == committed and not (destination / 'suite.json').exists()
    assert not list(destination.rglob('*.tmp'))
    continued = StableValidator()
    result = generate_suite(configured(), output_dir=destination, validator=continued, resume=True)
    assert continued.seeds == resumed_seeds
    actual, directory = saved(result)
    expected, baseline_dir = saved(baseline)
    assert actual == expected
    relative = actual['groups'][0]['instances_path']
    assert (directory / relative).read_bytes() == (baseline_dir / relative).read_bytes()


@pytest.mark.parametrize('dataset', ['random', 'movingai', 'qd_mapper', 'cities_tiles'])
def test_instance_io_error_does_not_complete_a_feasible_group_as_invalid(tmp_path, fixed_sources, monkeypatch, dataset):
    import os
    from pathlib import Path
    benchmark = configured() if dataset == 'random' else fixed_configured(dataset)
    benchmark['agent_counts'] = [2]
    if dataset in fixed_sources:
        benchmark['manifest'] = fixed_sources[dataset]
    baseline = generate_suite(benchmark, output_dir=tmp_path / 'baseline', artifact_root=tmp_path)
    destination = tmp_path / 'interrupted'
    replace = os.replace
    failed = False

    def fail_instance_write(source, target):
        nonlocal failed
        if not failed and Path(target).name == 'instances.jsonl':
            failed = True
            raise OSError('Temporary instance write failure')
        return replace(source, target)

    with monkeypatch.context() as patch:
        patch.setattr(os, 'replace', fail_instance_write)
        with pytest.raises(OSError, match='Temporary instance write failure'):
            generate_suite(benchmark, output_dir=destination, artifact_root=tmp_path)
    assert not (destination / 'suite.json').exists()
    assert json.loads((destination / 'suite.json.progress.json').read_text())['groups'] == []
    result = generate_suite(benchmark, output_dir=destination, artifact_root=tmp_path, resume=True)
    assert saved(result)[0] == saved(baseline)[0]


@pytest.mark.parametrize('validated', [False, True])
def test_completed_suite_rejects_changed_saved_provenance_json_types(tmp_path, validated):
    provenance = {'solver_setting': 1}
    generate_suite(configured(), output_dir=tmp_path,
                   validator=StableValidator() if validated else None, provenance=provenance)
    index = tmp_path / 'suite.json'
    suite = json.loads(index.read_text())
    suite['provenance']['solver_setting'] = True
    index.write_text(json.dumps(suite))
    before = index.read_bytes()
    validator = StableValidator() if validated else None
    with pytest.raises(ValueError, match='suite|progress'):
        generate_suite(configured(), output_dir=tmp_path, validator=validator,
                       provenance=provenance, resume=True)
    assert index.read_bytes() == before
    assert validator is None or not validator.seeds
