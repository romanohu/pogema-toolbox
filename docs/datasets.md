# Pinned dataset preparation

`prepare_dataset` from `pogema_toolbox` prepares the pinned MovingAI and
QD-MAPPER catalogs without Neural-MAPF, Hydra or OmegaConf. Raw maps and scenarios
remain external caller-owned artifacts. The wheel and source distribution contain catalogs,
notices and the existing Cities Tiles asset; they do not contain the acquired
MovingAI or QD-MAPPER files.

Preparation is explicit. Parsers, scenario samplers and generators never acquire
missing data. A successful import verifies every selected archive and member
before atomically replacing the data directory. Failed verification preserves
existing data and its receipt. The returned `receipt` is also written to
`receipt.json`, including source identities, attribution and rights.

```python
from pathlib import Path
from pogema_toolbox import (
    prepare_dataset, parse_movingai_map, parse_movingai_scenarios,
    movingai_grid_config, ScenarioSampler,
)

prepared = prepare_dataset("movingai", artifact_root="./artifacts")
data = Path(prepared["data_dir"])
map_name = "random-32-32-10"
obstacles = parse_movingai_map((data / (map_name + ".map")).read_text())
problems = parse_movingai_scenarios(
    (data / "scen-random" / (map_name + "-random-1.scen")).read_text(),
    map_name + ".map", obstacles,
)
replay = movingai_grid_config(obstacles, problems, 32, max_episode_steps=256)
sampled = ScenarioSampler(obstacles).generate(32, 42, sampling="uniform")
```

MovingAI may download the catalog's pinned URLs, or import caller-supplied ZIPs
with `staged_dir="/path/to/archives"`. Use `offline=True` to require local inputs.
ZIP filenames, byte counts, archive SHA-256 values and member hashes must match
the catalog. Unsafe paths, symlinks, duplicate destinations, extra members and
missing members are rejected. Original map cells, borders and source bytes are
preserved.

QD-MAPPER dataset permission remains `permission_unverified`. Automatic QD
acquisition is rejected before filesystem or network side effects. Supply
explicit local collection ZIPs or an extracted directory:

```python
from pogema_toolbox import load_qd_map

qd = prepare_dataset(
    "qd_mapper", artifact_root="./artifacts",
    local_dir="/absolute/path/to/extracted-collections",
    collections=["CBS"], offline=True,
)
# Alternatively use staged_dir="/absolute/path/to/collection-zips".
qd_map = load_qd_map(Path(qd["data_dir"]) / "CBS/0_30.json")
replay = qd_map.grid_config(0, 32, max_episode_steps=256)
sampled = ScenarioSampler(qd_map.obstacles).generate(32, 42, sampling="uniform")
```

Extracted inputs must retain the catalog's member paths, such as
`CBS/0_30.json`; local ZIPs use catalog filenames, such as `CBS.zip`. Omitting
`collections` imports all QD collections in catalog order. Explicit selections
retain their supplied order and must be nonempty, unique known names. Local
receipts retain the original directory or ZIP path and `permission_unverified`;
verification does not establish redistribution permission.

The root public API also includes `dataset_manifest(dataset, *, artifact_root=".",
manifest_path=None, manifest=None)` and `data_directory(dataset, *, artifact_root=".")`.
`prepare_dataset` accepts these
keyword arguments: `artifact_root="."`, `collections=None`, `staged_dir=None`,
`local_dir=None`, `manifest_path=None`, `manifest=None`, `offline=False`, `timeout_seconds=30`.
Relative manifest and import paths are relative to `artifact_root`. Prepared
files use `<artifact_root>/maps/<dataset>/data`; explicit custom manifests must
use schema version 1 and the requested dataset name. Instead of `manifest_path`,
both manifest loading and preparation accept a plain resolved dictionary via
`manifest=...`; it must have the same schema version and dataset name. Supplying
both inputs is rejected. Toolbox does not resolve interpolation expressions;
configuration integrations resolve them before passing the dictionary.
`staged_dir` and `local_dir` cannot be combined.

The installed console and module entry points expose the same preparation options:

```bash
pogema-toolbox datasets list
pogema-toolbox datasets prepare movingai --artifact-root ./artifacts
python -m pogema_toolbox datasets prepare movingai --artifact-root ./artifacts --staged-dir ../archives --offline
pogema-toolbox datasets prepare qd_mapper --artifact-root ./artifacts --local-dir ../collections --collections CBS EECBS --offline
```

`--collections` takes one or more names in the supplied order. Preparation also
accepts `--manifest-path FILE` and `--timeout-seconds NUMBER` (default `30`).
The CLI uses the public API's defaults: artifact root `.`, no collection filter,
no local input or manifest override, and online preparation unless `--offline`
is supplied. Relative manifest, staged and local paths use the artifact root.
Listing reads packaged catalogs and reports attribution/source, rights and
available QD collections; it does not acquire data or construct a policy.

`prepare_dataset("cities_tiles")` verifies the existing packaged asset and returns
its map count and SHA-256, without creating a raw data directory. Use the
[Cities Tiles generator](../README.md#dataset-cli) to generate scenarios.
See [scenario parsing and sampling](scenarios.md) for parser and sampler details.

## Suite generation

```python
import json
from pathlib import Path
import yaml
from pogema_toolbox import generate_suite, load_instances

benchmark = yaml.safe_load(Path("examples/delivery.yaml").read_text())
result = generate_suite(benchmark, output_dir="artifacts/delivery", artifact_root="artifacts")
index = Path(result["suite_path"])
suite = json.loads(index.read_text())
first_group = suite["groups"][0]
instances = load_instances(index.parent / first_group["instances_path"])
```

```bash
pogema-toolbox datasets generate --config examples/delivery.yaml --output-dir artifacts/delivery --artifact-root artifacts
```

Use the complete [Delivery YAML example](../examples/delivery.yaml), also printed
in the [quick start](../README.md#dataset-cli) for use outside a checkout. It
includes layout ranges, separate map/scenario seeds, agent counts, episode counts,
and restart/horizon settings. Delivery and Sortation require no external assets;
Cities Tiles uses the packaged asset. MovingAI and QD-MAPPER generation require
already verified external files beneath `<artifact_root>/maps/<dataset>/data`.
Missing or mismatched inputs are recorded as invalid groups with preparation
instructions; generation never downloads them.

The config path is relative to the caller's working directory. Its YAML root
must be the benchmark mapping itself, without a `benchmark:` wrapper, and all
values must be literal: unresolved `${...}` interpolation is rejected. Paths
inside that mapping retain public API artifact-root semantics. An explicit
`suite_path` selects the index beneath the artifact root; otherwise `suite.json`
is written beneath `--output-dir`. Group JSONL files are relative to the index.
The returned JSON reports the index path, group/episode counts and invalid groups.
The index records generation parameters, source hashes, rights and toolbox
version/revision. A wheel without VCS provenance reports a null revision.

## Notices and rights

Preserve the packaged [MovingAI attribution](../pogema_toolbox/maps/movingai/notices/ATTRIBUTION.md),
[Cities Tiles notices](../pogema_toolbox/maps/cities_tiles/notices/ATTRIBUTION.md)
and [QD rights notice](../pogema_toolbox/maps/qd_mapper/notices/RIGHTS.md).
Copied preparation code retains the exact
[Neural-MAPF MIT notice](../pogema_toolbox/licenses/Neural-MAPF-LICENSE).
These repository code licenses do not replace third-party dataset terms.
