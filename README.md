# POGEMA Toolbox

This fork of [Tviskaron/pogema-toolbox](https://github.com/Tviskaron/pogema-toolbox)
provides explicit MAPF instances, deterministic benchmark generation, verified
dataset preparation, and the upstream algorithm registry and evaluation tools.
It supports Python 3.8+. Preparation and generation work with toolbox alone,
without Neural-MAPF, Hydra, OmegaConf, Torch, Dask or plotting imports.

Install this fork from Git, or install a local checkout:

```bash
pip install "git+https://github.com/romanohu/pogema-toolbox.git@main"
# From this repository:
pip install .
```

For reproducible runs, replace `main` with your tested commit. The upstream
`pip install pogema-toolbox` PyPI package does not include this fork's additions.

## Python quick start

```python
from pogema_toolbox import generate_instances, save_instances, load_instances

instances = generate_instances({
    "grid": {"size": 8, "density": 0, "num_agents": 2,
             "on_target": "nothing", "max_episode_steps": 32},
    "episodes": 2,
    "generation_seed": 42,
})
save_instances(instances, "artifacts/instances.jsonl")
restored = load_instances("artifacts/instances.jsonl")
assert [instance.id for instance in restored] == [instance.id for instance in instances]
grid_config = restored[0].to_grid_config()
```

`Instance` stores explicit obstacles, starts, targets, grid options and optional
lifelong target sequences. JSONL round-trips preserve its canonical ID.
Use `generate_suite(benchmark, output_dir="artifacts/delivery")` for grouped
benchmark generation, using the same plain mapping as the CLI example below.
To recover saved work, call it again with the same mapping and `resume=True`.
The result and saved suite contain `validation_summary` alongside group counts
and the suite path. Optional `provenance={...}` caller metadata is saved before
the final checkpoint and checked when resuming.

## Dataset CLI

```bash
pogema-toolbox datasets list
pogema-toolbox datasets prepare cities_tiles
pogema-toolbox datasets generate --config examples/delivery.yaml --output-dir artifacts/delivery
pogema-toolbox datasets generate --config examples/delivery.yaml --output-dir artifacts/delivery --resume
pogema-toolbox datasets list --json
python -m pogema_toolbox datasets list
```

The Delivery example is self-contained and requires no external dataset. From
an installed package outside the checkout, save the following as `delivery.yaml`
and use `--config delivery.yaml`:

```yaml
dataset: delivery
layout:
  width_min: 17
  width_max: 21
  height_min: 17
  height_max: 21
  block_width_min: 3
  block_width_max: 4
  block_height_min: 3
  block_height_max: 4
  aisle_width_min: 2
  aisle_width_max: 3
map_seeds: [7, 8]
generation_seed: 42
agent_counts: [4, 8]
episodes: 2
sampling: endpoints
grid:
  obs_radius: 2
  max_episode_steps: 32
  on_target: restart
  collision_system: priority
  observation_type: POMAPF
```

This produces four groups and eight episodes with explicit lifelong goals.
Commands print concise English summaries by default. Add `--json` to list,
prepare or generate for the full machine-readable result. Generation with no
ready episodes exits nonzero; partially invalid suites show ready and invalid
group counts and save each failed request. Root and dataset-group commands
without arguments show help; invalid input exits nonzero with an error on
stderr. Use `--help` on any command for its options.

`--config` is relative to the caller's working directory. It contains a plain
benchmark mapping, with literal values and no Hydra wrapper or `${...}`
interpolation. Benchmark paths such as `manifest_path` and `suite_path` remain
relative to `--artifact-root` (default `.`). Without `suite_path`, generation
writes `suite.json` and group JSONL files beneath `--output-dir`.

`--resume` reuses completed groups for every supported dataset. Generation saves
an atomic progress file beside the suite index and checks the benchmark,
generator, source files and validation identity before resuming. Changed inputs
or damaged progress produce an error before saved work is overwritten. Existing
suites remain readable; resuming requires a saved progress identity.

Python callers can pass `validator=callback` and `max_attempts=100` to
`generate_suite` for procedural Random, Maze, Warehouse, Room, Delivery and
Sortation groups. The callback receives an `Instance` and returns a JSON mapping
with boolean `accepted` and a nonempty `status`. Toolbox bundles no solver.
For recovery, declare the callback's stable identity in a nonempty
`callback.provenance` mapping, including its solver revision and selection
settings, then use `resume=True`. Each finished attempt and accepted episode is
checkpointed with its actual seeds. Accepted episodes become eligible only when
their requested group is complete; exhausted retries leave an invalid group
with zero eligible episodes.

`validation_summary.enabled` distinguishes filtered and unfiltered suites.
`total_attempts` includes generation failures, while `solver_checked_candidates`
counts candidates passed to the callback. `accepted_attempts` includes accepted
episodes later discarded from invalid groups; `retained_episodes` counts ready
episodes and `discarded_accepted_attempts` records the difference. The summary
also reports `invalid_groups`, `rejection_reasons`, `acceptance_rate`
(accepted/total) and `solver_acceptance_rate` (accepted/checked). Unfiltered suites
have zero validation counters and unavailable rates (`null`). Full accepted
reports containing `paths` are kept in an atomic per-group
`validation-plans.json` sidecar, with its path and fingerprint in the group
record; costs, seeds and statuses remain in the suite index.

Cities Tiles uses the packaged asset; Delivery and Sortation are procedural.
MovingAI suites require explicitly prepared maps/scenarios. QD-MAPPER requires
verified caller-supplied local inputs because acquisition permission remains
unverified. Preparation is explicit: evaluation, parsing and generation never
download missing artifacts. See [dataset preparation and suite generation](docs/datasets.md),
[scenario parsing and replay](docs/scenarios.md), and
[Delivery and Sortation layouts](docs/task_generators.md).

## Algorithms and evaluation

Use `from pogema_toolbox import ToolboxRegistry, evaluate` to register algorithms
and call `evaluate(evaluation_config, eval_dir=None)`. Evaluation uses a separate
mapping containing `environment`, `algorithms` and optional `results_views`;
it is not the dataset CLI's benchmark YAML. Algorithms provide `act(observations)`.
The upstream sequential, multiprocessing and Dask backends and result views
remain available. See the [Follower evaluation example](https://github.com/AIRI-Institute/learn-to-follow/blob/main/eval.py)
for algorithm integration; use this fork's canonical `evaluate` entry point.

## Licenses and assets

Toolbox code retains the upstream [Apache License 2.0](LICENSE). Copied original
MIT code retains the exact [Neural-MAPF MIT notice](pogema_toolbox/licenses/Neural-MAPF-LICENSE).
Code licenses do not replace third-party dataset terms; preserve the exact
[asset notices and acquisition gates](docs/datasets.md#notices-and-rights).

The fixed cities-tile maps are from
[Cognitive-AI-Systems/MAPF-GPT-DDG](https://github.com/Cognitive-AI-Systems/MAPF-GPT-DDG)
commit `842384ee18902b8e7c880c3f90c5fc8098520522`, asset path
`eval_configs/04-movingai/maps.yaml`. The packaged asset is verified against its
pinned catalog fingerprint.
The DDG repository carries the MIT License notice
`Copyright (c) 2025 Alexey Skrynnik`; the exact notice from that revision is
retained in
[`pogema_toolbox/licenses/MAPF-GPT-DDG-LICENSE`](pogema_toolbox/licenses/MAPF-GPT-DDG-LICENSE).
That repository notice does not establish the rights to the underlying
MovingAI database or its source content. Those rights require separate
verification before reuse or redistribution. QD-MAPPER dataset permission
also remains unverified; its reader accepts local files without bundling a dataset.

## Citation

If you use this repository in research, cite the upstream POGEMA paper:

```bibtex
@misc{skrynnik2024pogema,
      title={POGEMA: A Benchmark Platform for Cooperative Multi-Agent Navigation},
      author={Alexey Skrynnik and Anton Andreychuk and Anatolii Borzilov and Alexander Chernyavskiy and Konstantin Yakovlev and Aleksandr Panov},
      year={2024},
      eprint={2407.14931},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2407.14931},
}
```
